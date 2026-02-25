import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL


class LoadMedicalMaskd(transforms.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            path = str(d[key]).lower()
            if path.endswith(".nrrd") or path.endswith(".nhdr"):
                arr, _ = nrrd.read(d[key])
            elif path.endswith(".nii") or path.endswith(".nii.gz"):
                arr = nib.load(d[key]).get_fdata()
            else:
                raise ValueError(f"Unsupported file format: {d[key]}")
            arr = np.asarray(arr, dtype=np.float32)
            d[key] = arr
        return d


def parse_args():
    parser = argparse.ArgumentParser("Train 3D AutoencoderKL for vessel masks")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing vessel masks (.nrrd/.nhdr/.nii/.nii.gz)",
    )
    parser.add_argument("--output_dir", type=str, default="./phase1-codex/outputs_vae", help="Output directory")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[64, 64, 64], help="Patch size, e.g. 64 64 64")
    parser.add_argument(
        "--spatial_mode",
        type=str,
        default="crop",
        choices=["crop", "resize"],
        help="crop: pad+crop patches, resize: direct resize to spatial_size",
    )
    parser.add_argument(
        "--roi_mode",
        type=str,
        default="foreground",
        choices=["center", "foreground"],
        help="Only used when spatial_mode=crop. foreground crops around vessel region before patch crop.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_rate", type=float, default=0.0, help="0 means no cache dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--resume_ckpt", type=str, default="", help="Resume from checkpoint path")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="Load pretrained weights for fine-tuning (state_dict or checkpoint with model/state_dict key)",
    )
    parser.add_argument(
        "--pretrained_strict",
        action="store_true",
        help="Use strict=True when loading pretrained weights (default: strict=False)",
    )
    parser.add_argument(
        "--target_label",
        type=int,
        default=1,
        help="Keep only this label as foreground (default: 1 for vessel). Set <0 to disable label selection.",
    )
    parser.add_argument("--binarize", action="store_true", help="Binarize masks by >0.5")

    # AutoencoderKL (adapted from brain model config with lighter channels for mask training)
    parser.add_argument("--latent_channels", type=int, default=3)
    parser.add_argument("--num_channels", type=int, nargs="+", default=[64, 128, 128, 128])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)
    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[0, 0, 0, 0])
    return parser.parse_args()


def find_medical_mask_files(data_dir: Path):
    exts = ("*.nrrd", "*.nhdr", "*.nii", "*.nii.gz")
    files = []
    for ext in exts:
        files.extend(data_dir.rglob(ext))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No supported files found in {data_dir}. Expect .nrrd/.nhdr/.nii/.nii.gz")
    return files


def split_data(paths, val_ratio, seed):
    indices = list(range(len(paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:n_val])
    train = [{"mask": str(paths[i])} for i in indices if i not in val_indices]
    val = [{"mask": str(paths[i])} for i in indices if i in val_indices]
    return train, val


def _spatial_ops_train(spatial_size, spatial_mode, roi_mode):
    if spatial_mode == "resize":
        return [
            transforms.Resized(keys=["mask"], spatial_size=spatial_size, mode="nearest-exact"),
        ]
    ops = []
    if roi_mode == "foreground":
        ops.append(
            transforms.CropForegroundd(
                keys=["mask"],
                source_key="mask",
                select_fn=lambda x: x > 0,
                margin=8,
                allow_smaller=True,
            )
        )
    ops.extend(
        [
        transforms.SpatialPadd(keys=["mask"], spatial_size=spatial_size),
        transforms.RandSpatialCropd(keys=["mask"], roi_size=spatial_size, random_size=False),
        ]
    )
    return ops


def _spatial_ops_val(spatial_size, spatial_mode, roi_mode):
    if spatial_mode == "resize":
        return [
            transforms.Resized(keys=["mask"], spatial_size=spatial_size, mode="nearest-exact"),
        ]
    ops = []
    if roi_mode == "foreground":
        ops.append(
            transforms.CropForegroundd(
                keys=["mask"],
                source_key="mask",
                select_fn=lambda x: x > 0,
                margin=8,
                allow_smaller=True,
            )
        )
    ops.extend(
        [
        transforms.SpatialPadd(keys=["mask"], spatial_size=spatial_size),
        transforms.CenterSpatialCropd(keys=["mask"], roi_size=spatial_size),
        ]
    )
    return ops


def build_transforms(spatial_size, spatial_mode, roi_mode, is_train, binarize, target_label):
    ops = [
        LoadMedicalMaskd(keys=["mask"]),
        transforms.EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["mask"], dtype=torch.float32, track_meta=False),
    ]
    if target_label >= 0:
        # For label maps like {0: background, 1: vessel, 2: tumor}, keep only vessel by default.
        ops.append(transforms.Lambdad(keys=["mask"], func=lambda x: (x == float(target_label)).float()))
    else:
        # Fallback path for non-label-map masks.
        ops.append(transforms.ScaleIntensityRanged(keys=["mask"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True))

    if binarize:
        ops.append(transforms.Lambdad(keys=["mask"], func=lambda x: (x > 0.5).float()))

    if is_train:
        ops.extend(_spatial_ops_train(spatial_size, spatial_mode, roi_mode))
        ops.extend(
            [
                transforms.RandFlipd(keys=["mask"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["mask"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["mask"], prob=0.5, spatial_axis=2),
                transforms.RandRotate90d(keys=["mask"], prob=0.3, spatial_axes=(0, 1)),
            ]
        )
    else:
        ops.extend(_spatial_ops_val(spatial_size, spatial_mode, roi_mode))
    return transforms.Compose(ops)


def kl_loss(z_mu, z_sigma):
    k = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-8) - 1, dim=[1, 2, 3, 4])
    return torch.mean(k)


def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "model" in loaded_obj and isinstance(loaded_obj["model"], dict):
            return loaded_obj["model"]
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
    return loaded_obj


def _load_pretrained_with_shape_filter(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_sd:
            skipped.append((k, "missing_in_model"))
            continue
        if model_sd[k].shape != v.shape:
            skipped.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_sd[k].shape)}"))
            continue
        filtered[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    return filtered, skipped, missing_keys, unexpected_keys


def main():
    args = parse_args()
    set_determinism(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = out / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    files = find_medical_mask_files(Path(args.data_dir))
    train_data, val_data = split_data(files, args.val_ratio, args.seed)
    print(f"Found {len(files)} masks. Train={len(train_data)} Val={len(val_data)}")

    train_tf = build_transforms(
        args.spatial_size,
        args.spatial_mode,
        args.roi_mode,
        is_train=True,
        binarize=args.binarize,
        target_label=args.target_label,
    )
    val_tf = build_transforms(
        args.spatial_size,
        args.spatial_mode,
        args.roi_mode,
        is_train=False,
        binarize=args.binarize,
        target_label=args.target_label,
    )

    if args.cache_rate > 0:
        train_ds = CacheDataset(train_data, transform=train_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
        val_ds = CacheDataset(val_data, transform=val_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    else:
        train_ds = Dataset(train_data, transform=train_tf)
        val_ds = Dataset(val_data, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    attention_levels = [bool(x) for x in args.attention_levels]
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=tuple(args.num_channels),
        num_res_blocks=args.num_res_blocks,
        norm_num_groups=args.norm_num_groups,
        norm_eps=args.norm_eps,
        attention_levels=tuple(attention_levels),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0
    best_val = float("inf")
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from {args.resume_ckpt}, start epoch {start_epoch}")
    elif args.pretrained_model:
        loaded = torch.load(args.pretrained_model, map_location="cpu")
        state_dict = _extract_state_dict(loaded)
        if args.pretrained_strict:
            load_info = model.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained model with strict=True from {args.pretrained_model}")
            print("Fine-tuning starts from epoch 0 with current optimizer settings.")
            return_missing = getattr(load_info, "missing_keys", [])
            return_unexpected = getattr(load_info, "unexpected_keys", [])
            if return_missing:
                print(f"Missing keys: {len(return_missing)}")
            if return_unexpected:
                print(f"Unexpected keys: {len(return_unexpected)}")
        else:
            filtered, skipped, missing_keys, unexpected_keys = _load_pretrained_with_shape_filter(model, state_dict)
            print(f"Loaded pretrained model (shape-filtered) from {args.pretrained_model}")
            print(f"Loaded keys: {len(filtered)} / {len(state_dict)}")
            print(f"Skipped keys: {len(skipped)}")
            if skipped:
                print("Example skipped keys:")
                for k, reason in skipped[:10]:
                    print(f"  - {k}: {reason}")
            if missing_keys:
                print(f"Missing keys after load: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys after load: {len(unexpected_keys)}")
            print("Fine-tuning starts from epoch 0 with current optimizer settings.")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", ncols=110)

        for batch in pbar:
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                recon, z_mu, z_sigma = model(masks)
                recon_loss = F.l1_loss(recon.float(), masks.float())
                k_loss = kl_loss(z_mu, z_sigma)
                loss = recon_loss + args.kl_weight * k_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_recon += recon_loss.item()
            epoch_kl += k_loss.item()
            step = pbar.n if pbar.n > 0 else 1
            pbar.set_postfix(
                recon=f"{epoch_recon / step:.6f}",
                kl=f"{epoch_kl / step:.6f}",
                total=f"{(epoch_recon / step) + args.kl_weight * (epoch_kl / step):.6f}",
            )

        train_recon = epoch_recon / max(1, len(train_loader))
        train_kl = epoch_kl / max(1, len(train_loader))

        if (epoch + 1) % args.val_interval == 0 or epoch == 0 or (epoch + 1) == args.num_epochs:
            model.eval()
            val_recon = 0.0
            val_kl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    masks = batch["mask"].to(device)
                    with autocast(enabled=args.amp):
                        recon, z_mu, z_sigma = model(masks)
                        r = F.l1_loss(recon.float(), masks.float())
                        k = kl_loss(z_mu, z_sigma)
                    val_recon += r.item()
                    val_kl += k.item()

            val_recon /= max(1, len(val_loader))
            val_kl /= max(1, len(val_loader))
            val_total = val_recon + args.kl_weight * val_kl
            print(
                f"[Epoch {epoch + 1}] train_recon={train_recon:.6f} train_kl={train_kl:.6f} "
                f"val_recon={val_recon:.6f} val_kl={val_kl:.6f} val_total={val_total:.6f}"
            )

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / "last.pt")

            if val_total < best_val:
                best_val = val_total
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
                torch.save(model.state_dict(), ckpt_dir / "autoencoderkl_best_weights.pt")
                print(f"Saved new best model: {best_val:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "args": vars(args),
                },
                ckpt_dir / f"epoch_{epoch + 1}.pt",
            )

    torch.save(model.state_dict(), ckpt_dir / "autoencoderkl_final_weights.pt")
    print(f"Training completed. Final weights: {ckpt_dir / 'autoencoderkl_final_weights.pt'}")
    print(f"Best weights: {ckpt_dir / 'autoencoderkl_best_weights.pt'}")


if __name__ == "__main__":
    main()

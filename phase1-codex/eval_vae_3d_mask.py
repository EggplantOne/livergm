import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader, Dataset
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
            d[key] = np.asarray(arr, dtype=np.float32)
        return d


def parse_args():
    parser = argparse.ArgumentParser("Evaluate 3D AutoencoderKL for vessel masks")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to weights or checkpoint")
    parser.add_argument("--output_dir", type=str, default="./phase1-codex/eval_vae")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--spatial_mode", type=str, default="crop", choices=["crop", "resize"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_rate", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary metrics")
    parser.add_argument("--target_label", type=int, default=1, help="Keep only this label as foreground")
    parser.add_argument("--binarize_input", action="store_true")
    parser.add_argument("--max_visualizations", type=int, default=20)

    # Model args (must match training architecture)
    parser.add_argument("--latent_channels", type=int, default=3)
    parser.add_argument("--num_channels", type=int, nargs="+", default=[64, 128, 128, 128])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)
    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[0, 0, 0, 0])
    return parser.parse_args()


def find_medical_mask_files(data_dir: Path):
    files = []
    for ext in ("*.nrrd", "*.nhdr", "*.nii", "*.nii.gz"):
        files.extend(data_dir.rglob(ext))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No supported files found in {data_dir}")
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


def build_transforms(spatial_size, spatial_mode, target_label, binarize_input):
    ops = [
        LoadMedicalMaskd(keys=["mask"]),
        transforms.EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["mask"], dtype=torch.float32, track_meta=False),
    ]
    if target_label >= 0:
        ops.append(transforms.Lambdad(keys=["mask"], func=lambda x: (x == float(target_label)).float()))
    else:
        ops.append(transforms.ScaleIntensityRanged(keys=["mask"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True))
    if binarize_input:
        ops.append(transforms.Lambdad(keys=["mask"], func=lambda x: (x > 0.5).float()))

    if spatial_mode == "resize":
        ops.append(transforms.Resized(keys=["mask"], spatial_size=spatial_size, mode="nearest-exact"))
    else:
        ops.extend(
            [
                transforms.SpatialPadd(keys=["mask"], spatial_size=spatial_size),
                transforms.CenterSpatialCropd(keys=["mask"], roi_size=spatial_size),
            ]
        )
    return transforms.Compose(ops)


def extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "model" in loaded_obj and isinstance(loaded_obj["model"], dict):
            return loaded_obj["model"]
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
    return loaded_obj


def dice_score(pred_bin, gt_bin, eps=1e-8):
    inter = torch.sum(pred_bin * gt_bin, dim=[1, 2, 3, 4])
    denom = torch.sum(pred_bin, dim=[1, 2, 3, 4]) + torch.sum(gt_bin, dim=[1, 2, 3, 4])
    return ((2.0 * inter + eps) / (denom + eps)).cpu().numpy()


def iou_score(pred_bin, gt_bin, eps=1e-8):
    inter = torch.sum(pred_bin * gt_bin, dim=[1, 2, 3, 4])
    union = torch.sum((pred_bin + gt_bin) > 0, dim=[1, 2, 3, 4])
    return ((inter + eps) / (union + eps)).cpu().numpy()


def volume_relative_error(pred_bin, gt_bin, eps=1e-8):
    vp = torch.sum(pred_bin, dim=[1, 2, 3, 4])
    vg = torch.sum(gt_bin, dim=[1, 2, 3, 4])
    return (torch.abs(vp - vg) / (vg + eps)).cpu().numpy()


def save_orthogonal_comparison(gt, recon, out_path, threshold=0.5):
    gt_np = gt[0, 0].cpu().numpy()
    rc_np = recon[0, 0].cpu().numpy()
    rb_np = (rc_np > float(threshold)).astype(np.float32)

    # Pick slices with most foreground instead of fixed center slices.
    z = int(np.argmax(np.sum(gt_np, axis=(0, 1))))
    y = int(np.argmax(np.sum(gt_np, axis=(0, 2))))
    x = int(np.argmax(np.sum(gt_np, axis=(1, 2))))
    if np.sum(gt_np) == 0:
        z = gt_np.shape[2] // 2
        y = gt_np.shape[1] // 2
        x = gt_np.shape[0] // 2

    def norm01(a):
        a = a.astype(np.float32)
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn)

    rc_vis = norm01(rc_np)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))

    # Row 1: GT slices
    axes[0, 0].imshow(gt_np[:, :, z], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].imshow(gt_np[:, y, :], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].imshow(gt_np[x, :, :], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("GT Axial")
    axes[0, 1].set_title("GT Coronal")
    axes[0, 2].set_title("GT Sagittal")

    # Row 2: Raw recon slices (normalized for visibility)
    axes[1, 0].imshow(rc_vis[:, :, z], cmap="gray", vmin=0, vmax=1)
    axes[1, 1].imshow(rc_vis[:, y, :], cmap="gray", vmin=0, vmax=1)
    axes[1, 2].imshow(rc_vis[x, :, :], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Recon(raw) Axial")
    axes[1, 1].set_title("Recon(raw) Coronal")
    axes[1, 2].set_title("Recon(raw) Sagittal")

    # Row 3: Binary recon slices
    axes[2, 0].imshow(rb_np[:, :, z], cmap="gray", vmin=0, vmax=1)
    axes[2, 1].imshow(rb_np[:, y, :], cmap="gray", vmin=0, vmax=1)
    axes[2, 2].imshow(rb_np[x, :, :], cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("Recon(bin) Axial")
    axes[2, 1].set_title("Recon(bin) Coronal")
    axes[2, 2].set_title("Recon(bin) Sagittal")

    # Row 4: MIP projections
    axes[3, 0].imshow(np.max(gt_np, axis=2), cmap="gray", vmin=0, vmax=1)
    axes[3, 1].imshow(np.max(rc_vis, axis=2), cmap="gray", vmin=0, vmax=1)
    axes[3, 2].imshow(np.max(rb_np, axis=2), cmap="gray", vmin=0, vmax=1)
    axes[3, 0].set_title("GT MIP(z)")
    axes[3, 1].set_title("Recon(raw) MIP(z)")
    axes[3, 2].set_title("Recon(bin) MIP(z)")

    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    files = find_medical_mask_files(Path(args.data_dir))
    train_data, val_data = split_data(files, args.val_ratio, args.seed)
    if args.split == "train":
        eval_data = train_data
    elif args.split == "val":
        eval_data = val_data
    else:
        eval_data = [{"mask": str(p)} for p in files]
    print(f"Total={len(files)} eval_split={args.split} eval_count={len(eval_data)}")

    tf = build_transforms(
        spatial_size=args.spatial_size,
        spatial_mode=args.spatial_mode,
        target_label=args.target_label,
        binarize_input=args.binarize_input,
    )
    if args.cache_rate > 0:
        ds = CacheDataset(eval_data, transform=tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    else:
        ds = Dataset(eval_data, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=tuple(args.num_channels),
        num_res_blocks=args.num_res_blocks,
        norm_num_groups=args.norm_num_groups,
        norm_eps=args.norm_eps,
        attention_levels=tuple(bool(x) for x in args.attention_levels),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)
    loaded = torch.load(args.model_path, map_location="cpu")
    state_dict = extract_state_dict(loaded)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    per_case = []
    vis_count = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl, desc="Evaluating", ncols=100)):
            gt = batch["mask"].to(device)
            recon, _, _ = model(gt)
            recon = recon.float()
            gt = gt.float()

            pred_bin = (recon > args.threshold).float()
            gt_bin = (gt > 0.5).float()

            l1 = F.l1_loss(recon, gt, reduction="none").mean(dim=[1, 2, 3, 4]).cpu().numpy()
            mae = torch.abs(recon - gt).mean(dim=[1, 2, 3, 4]).cpu().numpy()
            dice = dice_score(pred_bin, gt_bin)
            iou = iou_score(pred_bin, gt_bin)
            vre = volume_relative_error(pred_bin, gt_bin)

            bs = gt.shape[0]
            for b in range(bs):
                sample_idx = i * args.batch_size + b
                case_path = eval_data[sample_idx]["mask"]
                per_case.append(
                    {
                        "case": case_path,
                        "dice": float(dice[b]),
                        "iou": float(iou[b]),
                        "l1": float(l1[b]),
                        "mae": float(mae[b]),
                        "volume_rel_error": float(vre[b]),
                    }
                )
                if vis_count < args.max_visualizations:
                    save_orthogonal_comparison(
                        gt[b : b + 1].cpu(),
                        recon[b : b + 1].cpu(),
                        vis_dir / f"case_{vis_count:04d}.png",
                        threshold=args.threshold,
                    )
                    vis_count += 1

    def mean_of(key):
        return float(np.mean([x[key] for x in per_case])) if per_case else 0.0

    summary = {
        "num_cases": len(per_case),
        "split": args.split,
        "threshold": args.threshold,
        "dice_mean": mean_of("dice"),
        "iou_mean": mean_of("iou"),
        "l1_mean": mean_of("l1"),
        "mae_mean": mean_of("mae"),
        "volume_rel_error_mean": mean_of("volume_rel_error"),
        "model_path": args.model_path,
    }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_case_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice", "iou", "l1", "mae", "volume_rel_error"])
        writer.writeheader()
        writer.writerows(per_case)

    print("Evaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {out_dir / 'metrics_summary.json'}")
    print(f"Saved per-case: {out_dir / 'per_case_metrics.csv'}")
    print(f"Saved visualizations: {vis_dir}")


if __name__ == "__main__":
    main()

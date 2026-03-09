"""Train 3D AutoencoderKL for CT image reconstruction.

Unlike the mask VAE (BCE+Dice loss on binary data), this uses:
  - L1 reconstruction loss (continuous CT values in [-1, 1])
  - KL divergence regularization
  - Optional PatchDiscriminator adversarial loss
  - Optional perceptual loss
  - No sigmoid on output (direct reconstruction)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from vessel_ldm_utils import (
    VesselImagePtDataset,
    extract_state_dict,
    kl_loss,
    load_pretrained_with_shape_filter,
    save_ct_recon_comparison,
    save_json,
    set_seed,
    split_pt_files_train_val_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train 3D AutoencoderKL for CT image reconstruction")
    # Data
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./phase2-image-vae/outputs_vae_image_128")
    p.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128])
    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--val_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=5e-4)
    # Loss weights
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=1e-6)
    p.add_argument("--adv_weight", type=float, default=0.01)
    p.add_argument("--perceptual_weight", type=float, default=0.001)
    p.add_argument("--warmup_epochs", type=int, default=5,
                    help="Epochs before enabling adversarial loss")
    p.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (0=disable)")
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")
    # Model
    p.add_argument("--latent_channels", type=int, default=3)
    p.add_argument("--num_channels", type=int, nargs="+", default=[64, 128, 128, 128])
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--norm_num_groups", type=int, default=32)
    p.add_argument("--norm_eps", type=float, default=1e-6)
    p.add_argument("--attention_levels", type=int, nargs="+", default=[0, 0, 0, 0],
                    help="No attention by default (matching pretrained brain VAE)")
    p.add_argument("--use_checkpointing", action="store_true")
    # Pretrained / resume
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--pretrained_model", type=str, default="",
                    help="Path to pretrained VAE weights (e.g. pretrained_models/autoencoder.pth)")
    p.add_argument("--pretrained_strict", action="store_true")
    # Augmentation
    p.add_argument("--augment", dest="augment", action="store_true")
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.set_defaults(augment=True)
    # Visualization
    p.add_argument("--val_max_visualizations", type=int, default=8)
    # Perceptual loss
    p.add_argument("--perceptual_cache_dir", type=str, default="")
    p.add_argument("--disable_perceptual_fallback", action="store_true")
    return p.parse_args()


def build_autoencoder(args: argparse.Namespace) -> AutoencoderKL:
    return AutoencoderKL(
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
        use_checkpointing=args.use_checkpointing,
    )


def build_discriminator() -> PatchDiscriminator:
    return PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
    )


def load_model_weights(model: AutoencoderKL, pretrained_model: str, strict: bool) -> None:
    loaded = torch.load(pretrained_model, map_location="cpu")
    state_dict = extract_state_dict(loaded)
    if strict:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded pretrained VAE with strict=True from {pretrained_model}")
        return
    filtered, skipped, missing_keys, unexpected_keys = load_pretrained_with_shape_filter(model, state_dict)
    print(f"Loaded pretrained VAE from {pretrained_model}")
    print(f"  Loaded keys: {len(filtered)} / {len(state_dict)}")
    if skipped:
        print("  Skipped keys (first 10):")
        for key, reason in skipped[:10]:
            print(f"    - {key}: {reason}")
    if missing_keys:
        print(f"  Missing keys after load: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys after load: {len(unexpected_keys)}")


def evaluate(
    model: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_device: str,
    vis_dir: Path,
    epoch: int,
    max_visualizations: int,
) -> dict[str, float]:
    model.eval()
    metrics = {"val_l1": 0.0, "val_kl": 0.0, "val_mse": 0.0}
    vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                recon, z_mu, z_sigma = model(images)
                l1_val = F.l1_loss(recon.float(), images.float())
                mse_val = F.mse_loss(recon.float(), images.float())
                kl_val = kl_loss(z_mu, z_sigma)

            metrics["val_l1"] += float(l1_val.item())
            metrics["val_mse"] += float(mse_val.item())
            metrics["val_kl"] += float(kl_val.item())

            if vis_count < max_visualizations:
                count = min(max_visualizations - vis_count, images.shape[0])
                for s in range(count):
                    save_ct_recon_comparison(
                        gt=images[s:s+1].cpu(),
                        recon=recon[s:s+1].float().cpu(),
                        out_path=vis_dir / f"epoch_{epoch+1:04d}_case_{batch_idx:04d}_{s:02d}.png",
                    )
                    vis_count += 1

    denom = max(1, len(loader))
    for key in metrics:
        metrics[key] /= denom
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "train_config.json")

    # Data split
    train_files, val_files, test_files = split_pt_files_train_val_test(
        args.cache_dir, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    save_json(
        {
            "seed": args.seed,
            "num_train": len(train_files),
            "num_val": len(val_files),
            "num_test": len(test_files),
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
            "test_files": [str(p) for p in test_files],
        },
        output_dir / "data_split.json",
    )

    train_ds = VesselImagePtDataset(train_files, augment=args.augment, spatial_size=tuple(args.spatial_size))
    val_ds = VesselImagePtDataset(val_files, augment=False, spatial_size=tuple(args.spatial_size))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    model = build_autoencoder(args).to(device)
    discriminator = build_discriminator().to(device) if args.adv_weight > 0 else None
    adv_loss = PatchAdversarialLoss(criterion="least_squares") if discriminator is not None else None

    # Perceptual loss
    perceptual_loss = None
    effective_perceptual_weight = float(args.perceptual_weight)
    if effective_perceptual_weight > 0:
        try:
            perceptual_loss = PerceptualLoss(
                spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2,
                cache_dir=args.perceptual_cache_dir or None,
            ).to(device)
        except Exception as error:
            if args.disable_perceptual_fallback:
                raise
            perceptual_loss = None
            effective_perceptual_weight = 0.0
            print(f"Warning: perceptual loss init failed ({error}), continuing without it.")

    # Optimizers
    optimizer_g = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d) if discriminator is not None else None
    scaler_g = torch.amp.GradScaler(amp_device, enabled=use_amp)
    scaler_d = torch.amp.GradScaler(amp_device, enabled=use_amp) if discriminator is not None else None

    start_epoch = 0
    best_l1 = float("inf")
    no_improve_count = 0
    history: list[dict[str, float | int]] = []

    # Resume or pretrained
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if discriminator is not None and ckpt.get("discriminator") is not None:
            discriminator.load_state_dict(ckpt["discriminator"], strict=True)
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if optimizer_d is not None and ckpt.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        if "scaler_g" in ckpt:
            scaler_g.load_state_dict(ckpt["scaler_g"])
        if scaler_d is not None and "scaler_d" in ckpt and ckpt["scaler_d"] is not None:
            scaler_d.load_state_dict(ckpt["scaler_d"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_l1 = float(ckpt.get("best_l1", best_l1))
        history = list(ckpt.get("history", []))
        print(f"Resumed from {args.resume_ckpt} at epoch {start_epoch}")
    elif args.pretrained_model:
        load_model_weights(model, args.pretrained_model, strict=args.pretrained_strict)

    print(
        f"Training Image VAE: {len(train_ds)} train, {len(val_ds)} val, {len(test_files)} test\n"
        f"  device={device}, amp={use_amp}, arch={list(args.num_channels)}, latent={args.latent_channels}\n"
        f"  l1_weight={args.l1_weight}, kl_weight={args.kl_weight}, "
        f"adv_weight={args.adv_weight}, perceptual_weight={effective_perceptual_weight}"
    )

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()

        running = {"train_l1": 0.0, "train_kl": 0.0, "train_p": 0.0, "train_g_adv": 0.0, "train_d": 0.0}
        use_adversarial = discriminator is not None and epoch >= args.warmup_epochs
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", ncols=120)

        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device)

            # ---- Generator step ----
            optimizer_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                recon, z_mu, z_sigma = model(images)
                l1_value = F.l1_loss(recon.float(), images.float())
                current_kl = kl_loss(z_mu, z_sigma)
                loss_g = args.l1_weight * l1_value + args.kl_weight * current_kl

                perceptual_value = torch.tensor(0.0, device=device)
                generator_adv = torch.tensor(0.0, device=device)

                if perceptual_loss is not None and effective_perceptual_weight > 0:
                    perceptual_value = perceptual_loss(recon.float(), images.float())
                    loss_g = loss_g + effective_perceptual_weight * perceptual_value

                if use_adversarial:
                    logits_fake = discriminator(recon.contiguous().float())[-1]
                    generator_adv = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g = loss_g + args.adv_weight * generator_adv

            scaler_g.scale(loss_g).backward()
            if args.gradient_clip > 0:
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # ---- Discriminator step ----
            discriminator_value = torch.tensor(0.0, device=device)
            if use_adversarial:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    logits_fake = discriminator(recon.detach().contiguous().float())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().float())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_value = args.adv_weight * 0.5 * (loss_d_fake + loss_d_real)

                scaler_d.scale(discriminator_value).backward()
                if args.gradient_clip > 0:
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.gradient_clip)
                scaler_d.step(optimizer_d)
                scaler_d.update()

            running["train_l1"] += float(l1_value.item())
            running["train_kl"] += float(current_kl.item())
            running["train_p"] += float(perceptual_value.item())
            running["train_g_adv"] += float(generator_adv.item())
            running["train_d"] += float(discriminator_value.item())
            progress.set_postfix(
                l1=f"{running['train_l1']/step:.5f}",
                kl=f"{running['train_kl']/step:.2f}",
            )

        epoch_summary: dict[str, float | int] = {"epoch": epoch + 1}
        for key, value in running.items():
            epoch_summary[key] = value / max(1, len(train_loader))

        # ---- Validation ----
        if (epoch + 1) % args.val_interval == 0 or epoch == 0 or (epoch + 1) == args.num_epochs:
            val_metrics = evaluate(
                model=model, loader=val_loader, device=device,
                use_amp=use_amp, amp_device=amp_device,
                vis_dir=vis_dir, epoch=epoch, max_visualizations=args.val_max_visualizations,
            )
            epoch_summary.update(val_metrics)
            print(json.dumps(epoch_summary, indent=2))

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict() if discriminator is not None else None,
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
                "best_l1": best_l1,
                "history": history + [epoch_summary],
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / "last.pt")

            current_l1 = float(val_metrics["val_l1"])
            if current_l1 < best_l1:
                best_l1 = current_l1
                no_improve_count = 0
                ckpt["best_l1"] = best_l1
                torch.save(ckpt, ckpt_dir / "best.pt")
                torch.save(model.state_dict(), ckpt_dir / "autoencoderkl_best_weights.pt")
                print(f"  New best: val_l1={best_l1:.6f}")
            else:
                no_improve_count += 1
                print(f"  No improvement for {no_improve_count} val interval(s) (patience={args.patience})")

        history.append(epoch_summary)
        save_json(history, output_dir / "metrics_history.json")

        if args.patience > 0 and no_improve_count >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "discriminator": discriminator.state_dict() if discriminator is not None else None,
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
                    "scaler_g": scaler_g.state_dict(),
                    "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
                    "best_l1": best_l1,
                    "history": history,
                    "args": vars(args),
                },
                ckpt_dir / f"epoch_{epoch+1}.pt",
            )

    torch.save(model.state_dict(), ckpt_dir / "autoencoderkl_final_weights.pt")
    print(f"Training completed. Best val_l1: {best_l1:.6f}")
    print(f"Final weights: {ckpt_dir / 'autoencoderkl_final_weights.pt'}")


if __name__ == "__main__":
    main()

"""Train unconditional 3D latent diffusion model for CT image generation.

Step 2 of the ControlNet pipeline:
  1. Image VAE (done) -> encodes CT to latent space
  2. Unconditional LDM (this script) -> learns CT latent distribution
  3. ControlNet -> conditions on vessel mask

Based on phase1-ldm/train_ldm_3d_mask.py, adapted for CT images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from vessel_ldm_utils import (
    VesselImagePtDataset,
    extract_state_dict,
    save_ct_overview,
    save_json,
    set_seed,
    split_pt_files_train_val_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train unconditional 3D LDM for CT images")
    # Data
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--vae_checkpoint", type=str, required=True,
                    help="Path to trained image VAE weights")
    p.add_argument("--output_dir", type=str, default="./phase2-image-ldm/outputs_ldm_image")
    p.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128])
    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--val_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (0 to disable)")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--augment", dest="augment", action="store_true")
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.set_defaults(augment=True)
    # VAE architecture (must match trained image VAE)
    p.add_argument("--latent_channels", type=int, default=3)
    p.add_argument("--vae_num_channels", type=int, nargs="+", default=[64, 128, 128, 128])
    p.add_argument("--vae_num_res_blocks", type=int, default=2)
    p.add_argument("--vae_norm_num_groups", type=int, default=32)
    p.add_argument("--vae_norm_eps", type=float, default=1e-6)
    p.add_argument("--vae_attention_levels", type=int, nargs="+", default=[0, 0, 0, 0])
    # UNet architecture
    p.add_argument("--unet_num_channels", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--unet_num_res_blocks", type=int, default=2)
    p.add_argument("--unet_attention_levels", type=int, nargs="+", default=[0, 1, 1])
    p.add_argument("--unet_num_head_channels", type=int, nargs="+", default=[0, 128, 256])
    p.add_argument("--unet_norm_num_groups", type=int, default=32)
    p.add_argument("--unet_norm_eps", type=float, default=1e-6)
    # Scheduler
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="scaled_linear_beta")
    p.add_argument("--beta_start", type=float, default=0.0015)
    p.add_argument("--beta_end", type=float, default=0.0195)
    p.add_argument("--scale_factor", type=float, default=0.0,
                    help="If <=0, estimate from training latents")
    p.add_argument("--scale_factor_batches", type=int, default=50)
    # Sampling
    p.add_argument("--sample_inference_steps", type=int, default=100)
    p.add_argument("--val_num_samples", type=int, default=3)
    return p.parse_args()


def build_autoencoder(args: argparse.Namespace) -> AutoencoderKL:
    return AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=tuple(args.vae_num_channels),
        num_res_blocks=args.vae_num_res_blocks,
        norm_num_groups=args.vae_norm_num_groups,
        norm_eps=args.vae_norm_eps,
        attention_levels=tuple(bool(x) for x in args.vae_attention_levels),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )


def build_unet(args: argparse.Namespace) -> DiffusionModelUNet:
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        num_channels=tuple(args.unet_num_channels),
        num_res_blocks=args.unet_num_res_blocks,
        attention_levels=tuple(bool(x) for x in args.unet_attention_levels),
        norm_num_groups=args.unet_norm_num_groups,
        norm_eps=args.unet_norm_eps,
        num_head_channels=tuple(args.unet_num_head_channels),
        with_conditioning=False,
    )


def build_ddpm_scheduler(args: argparse.Namespace) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )


def build_ddim_scheduler(args: argparse.Namespace) -> DDIMScheduler:
    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(args.sample_inference_steps)
    return scheduler


def infer_latent_shape(autoencoder: AutoencoderKL, sample: torch.Tensor) -> tuple[int, ...]:
    with torch.no_grad():
        latent = autoencoder.encode_stage_2_inputs(sample)
    return tuple(latent.shape[1:])


def estimate_scale_factor(
    autoencoder: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    latents = []
    autoencoder.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            latents.append(autoencoder.encode_stage_2_inputs(images).float().cpu())
            if batch_idx + 1 >= max_batches:
                break
    latent_tensor = torch.cat(latents, dim=0)
    std = float(torch.std(latent_tensor).item())
    if std <= 0:
        raise ValueError("Estimated latent std is not positive")
    return 1.0 / std


def validate_epoch(
    autoencoder: AutoencoderKL,
    inferer: LatentDiffusionInferer,
    unet: DiffusionModelUNet,
    loader: DataLoader,
    latent_shape: tuple[int, ...],
    scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool,
) -> float:
    unet.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            noise = torch.randn((images.shape[0],) + latent_shape, device=device)
            timesteps = torch.randint(
                low=0, high=scheduler.num_train_timesteps,
                size=(images.shape[0],), device=device,
            ).long()
            with autocast("cuda", enabled=use_amp):
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())
            losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def run_sampling(
    args: argparse.Namespace,
    autoencoder: AutoencoderKL,
    unet: DiffusionModelUNet,
    scale_factor: float,
    latent_shape: tuple[int, ...],
    device: torch.device,
    out_dir: Path,
    epoch: int,
) -> None:
    ddim_scheduler = build_ddim_scheduler(args)
    inferer = LatentDiffusionInferer(ddim_scheduler, scale_factor=scale_factor)
    noise = torch.randn((args.val_num_samples,) + latent_shape, device=device)
    with autocast("cuda", enabled=bool(args.amp)):
        samples = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=ddim_scheduler,
            verbose=False,
        ).float()
    for idx in range(samples.shape[0]):
        save_ct_overview(
            volume=samples[idx:idx+1].cpu(),
            out_path=out_dir / f"epoch_{epoch+1:04d}_sample_{idx:02d}.png",
            title=f"Sample {idx}",
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "val_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "train_config.json")

    # Data
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
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    # Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    autoencoder = build_autoencoder(args).to(device)
    vae_loaded = torch.load(args.vae_checkpoint, map_location="cpu")
    autoencoder.load_state_dict(extract_state_dict(vae_loaded), strict=True)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    sample_batch = next(iter(train_loader))["image"][:1].to(device)
    latent_shape = infer_latent_shape(autoencoder, sample_batch)
    scale_factor = args.scale_factor if args.scale_factor > 0 else estimate_scale_factor(
        autoencoder=autoencoder, loader=train_loader, device=device,
        max_batches=args.scale_factor_batches,
    )

    scheduler = build_ddpm_scheduler(args)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    unet = build_unet(args).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    no_improve_count = 0
    history: list[dict[str, float | int]] = []

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        unet.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        history = list(ckpt.get("history", []))
        scale_factor = float(ckpt.get("scale_factor", scale_factor))
        latent_shape = tuple(ckpt.get("latent_shape", latent_shape))
        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
        print(f"Resumed from {args.resume_ckpt} at epoch {start_epoch}")

    print(
        f"Training Image LDM: {len(train_ds)} train, {len(val_ds)} val, {len(test_files)} test\n"
        f"  device={device}, amp={use_amp}, latent_shape={latent_shape}, scale_factor={scale_factor:.6f}\n"
        f"  unet_channels={list(args.unet_num_channels)}"
    )

    for epoch in range(start_epoch, args.num_epochs):
        unet.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", ncols=120)

        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            noise = torch.randn((images.shape[0],) + latent_shape, device=device)
            timesteps = torch.randint(
                low=0, high=scheduler.num_train_timesteps,
                size=(images.shape[0],), device=device,
            ).long()

            with autocast("cuda", enabled=use_amp):
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            progress.set_postfix(mse=f"{running_loss / step:.6f}")

        epoch_summary: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_mse": running_loss / max(1, len(train_loader)),
        }

        if (epoch + 1) % args.val_interval == 0 or epoch == 0 or (epoch + 1) == args.num_epochs:
            val_loss = validate_epoch(
                autoencoder=autoencoder, inferer=inferer, unet=unet,
                loader=val_loader, latent_shape=latent_shape,
                scheduler=scheduler, device=device, use_amp=use_amp,
            )
            epoch_summary["val_mse"] = val_loss
            print(json.dumps(epoch_summary, indent=2))

            run_sampling(
                args=args, autoencoder=autoencoder, unet=unet,
                scale_factor=scale_factor, latent_shape=latent_shape,
                device=device, out_dir=sample_dir, epoch=epoch,
            )

            ckpt = {
                "epoch": epoch,
                "model": unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history + [epoch_summary],
                "scale_factor": scale_factor,
                "latent_shape": list(latent_shape),
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                ckpt["best_val_loss"] = best_val_loss
                torch.save(ckpt, ckpt_dir / "best.pt")
                torch.save(unet.state_dict(), ckpt_dir / "diffusion_unet_best_weights.pt")
                print(f"  New best: val_mse={best_val_loss:.6f}")
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
                    "model": unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "scale_factor": scale_factor,
                    "latent_shape": list(latent_shape),
                    "args": vars(args),
                },
                ckpt_dir / f"epoch_{epoch+1}.pt",
            )

    torch.save(unet.state_dict(), ckpt_dir / "diffusion_unet_final_weights.pt")
    print(f"Training completed. Best val_mse: {best_val_loss:.6f}")
    print(f"Final weights: {ckpt_dir / 'diffusion_unet_final_weights.pt'}")


if __name__ == "__main__":
    main()

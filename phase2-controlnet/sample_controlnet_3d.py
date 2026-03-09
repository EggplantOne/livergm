"""Sample CT images conditioned on vessel masks using trained ControlNet.

Loads trained VAE + DM + ControlNet and generates CT volumes from mask conditions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.inferers import ControlNetLatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.nets.controlnet import ControlNet
from generative.networks.schedulers import DDIMScheduler
from vessel_ldm_utils import (
    VesselImageMaskPtDataset,
    extract_state_dict,
    save_controlnet_comparison,
    save_nifti_volume,
    set_seed,
    split_pt_files_train_val_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sample with 3D ControlNet")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--vae_checkpoint", type=str, required=True)
    p.add_argument("--dm_checkpoint", type=str, required=True)
    p.add_argument("--cn_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./phase2-controlnet/samples")
    p.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_nifti", action="store_true")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    # Architecture (must match training)
    p.add_argument("--latent_channels", type=int, default=3)
    p.add_argument("--vae_num_channels", type=int, nargs="+", default=[64, 128, 128, 128])
    p.add_argument("--vae_num_res_blocks", type=int, default=2)
    p.add_argument("--vae_attention_levels", type=int, nargs="+", default=[0, 0, 0, 0])
    p.add_argument("--unet_num_channels", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--unet_num_res_blocks", type=int, default=2)
    p.add_argument("--unet_attention_levels", type=int, nargs="+", default=[0, 1, 1])
    p.add_argument("--unet_num_head_channels", type=int, nargs="+", default=[0, 128, 256])
    p.add_argument("--cn_embedding_channels", type=int, nargs="+", default=[16])
    # Scheduler
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="scaled_linear_beta")
    p.add_argument("--beta_start", type=float, default=0.0015)
    p.add_argument("--beta_end", type=float, default=0.0195)
    p.add_argument("--scale_factor", type=float, required=True,
                    help="Scale factor (from DM training)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_files, val_files, test_files = split_pt_files_train_val_test(
        args.cache_dir, val_ratio=0.1, test_ratio=0.1, seed=args.seed,
    )
    split_map = {"train": train_files, "val": val_files, "test": test_files}
    files = split_map[args.split]
    ds = VesselImageMaskPtDataset(files, augment=False, spatial_size=tuple(args.spatial_size))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Models
    autoencoder = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=tuple(args.vae_num_channels),
        num_res_blocks=args.vae_num_res_blocks,
        attention_levels=tuple(bool(x) for x in args.vae_attention_levels),
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    ).to(device)
    autoencoder.load_state_dict(extract_state_dict(torch.load(args.vae_checkpoint, map_location="cpu")), strict=True)
    autoencoder.eval()

    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=args.latent_channels, out_channels=args.latent_channels,
        num_channels=tuple(args.unet_num_channels),
        num_res_blocks=args.unet_num_res_blocks,
        attention_levels=tuple(bool(x) for x in args.unet_attention_levels),
        num_head_channels=tuple(args.unet_num_head_channels),
        with_conditioning=False,
    ).to(device)
    unet.load_state_dict(extract_state_dict(torch.load(args.dm_checkpoint, map_location="cpu")), strict=True)
    unet.eval()

    controlnet = ControlNet(
        spatial_dims=3, in_channels=args.latent_channels,
        num_channels=tuple(args.unet_num_channels),
        num_res_blocks=args.unet_num_res_blocks,
        attention_levels=tuple(bool(x) for x in args.unet_attention_levels),
        num_head_channels=tuple(args.unet_num_head_channels),
        with_conditioning=False,
        conditioning_embedding_in_channels=1,
        conditioning_embedding_num_channels=tuple(args.cn_embedding_channels),
    ).to(device)
    controlnet.load_state_dict(extract_state_dict(torch.load(args.cn_checkpoint, map_location="cpu")), strict=True)
    controlnet.eval()

    # Scheduler and inferer
    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(args.inference_steps)
    inferer = ControlNetLatentDiffusionInferer(scheduler, scale_factor=args.scale_factor)

    # Infer latent shape
    sample_batch = next(iter(loader))["image"][:1].to(device)
    with torch.no_grad():
        latent_shape = tuple(autoencoder.encode_stage_2_inputs(sample_batch).shape[1:])
    print(f"Latent shape: {latent_shape}, scale_factor: {args.scale_factor}")

    # Sample
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= args.num_samples:
                break
            masks = batch["mask"].to(device)
            gt_images = batch["image"].to(device)
            noise = torch.randn((1,) + latent_shape, device=device)

            with autocast("cuda", enabled=use_amp):
                sample = inferer.sample(
                    input_noise=noise,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    controlnet=controlnet,
                    cn_cond=masks,
                    scheduler=scheduler,
                    verbose=True,
                ).float()

            save_controlnet_comparison(
                mask=masks[:1].cpu(),
                generated=sample[:1].cpu(),
                out_path=output_dir / f"sample_{count:04d}.png",
                gt_image=gt_images[:1].cpu(),
            )

            if args.save_nifti:
                save_nifti_volume(sample[:1].cpu(), output_dir / f"sample_{count:04d}_generated.nii.gz")
                save_nifti_volume(gt_images[:1].cpu(), output_dir / f"sample_{count:04d}_gt.nii.gz")
                save_nifti_volume(masks[:1].cpu(), output_dir / f"sample_{count:04d}_mask.nii.gz")

            count += 1
            print(f"Generated {count}/{args.num_samples}")

    print(f"Done. Samples saved to {output_dir}")


if __name__ == "__main__":
    main()

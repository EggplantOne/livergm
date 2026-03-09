from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from vessel_ldm_utils import (
    connected_component_stats,
    extract_state_dict,
    save_generated_overview,
    save_json,
    save_nifti_volume,
    set_seed,
    vessel_ratio,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Sample synthetic 3D hepatic vessel masks from a latent diffusion model")
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--ldm_checkpoint", type=str, required=True, help="Prefer the training checkpoint best.pt")
    parser.add_argument("--output_dir", type=str, default="./phase1-ldm/samples")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_nifti", action="store_true")

    parser.add_argument("--latent_channels", type=int, default=3)
    parser.add_argument("--vae_num_channels", type=int, nargs="+", default=[32, 64, 64])
    parser.add_argument("--vae_num_res_blocks", type=int, default=1)
    parser.add_argument("--vae_norm_num_groups", type=int, default=32)
    parser.add_argument("--vae_norm_eps", type=float, default=1e-6)
    parser.add_argument("--vae_attention_levels", type=int, nargs="+", default=[0, 0, 1])
    parser.add_argument("--unet_num_channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--unet_num_res_blocks", type=int, default=2)
    parser.add_argument("--unet_attention_levels", type=int, nargs="+", default=[0, 1, 1])
    parser.add_argument("--unet_num_head_channels", type=int, nargs="+", default=[0, 128, 256])
    parser.add_argument("--unet_norm_num_groups", type=int, default=32)
    parser.add_argument("--unet_norm_eps", type=float, default=1e-6)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="scaled_linear_beta")
    parser.add_argument("--beta_start", type=float, default=0.0015)
    parser.add_argument("--beta_end", type=float, default=0.0195)
    parser.add_argument("--scale_factor", type=float, default=0.0)
    parser.add_argument("--latent_shape", type=int, nargs="+", default=[3, 16, 16, 16])
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_autoencoder(config: dict) -> AutoencoderKL:
    return AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=int(config["latent_channels"]),
        num_channels=tuple(config["vae_num_channels"]),
        num_res_blocks=int(config["vae_num_res_blocks"]),
        norm_num_groups=int(config["vae_norm_num_groups"]),
        norm_eps=float(config["vae_norm_eps"]),
        attention_levels=tuple(bool(x) for x in config["vae_attention_levels"]),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )


def build_unet(config: dict) -> DiffusionModelUNet:
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=int(config["latent_channels"]),
        out_channels=int(config["latent_channels"]),
        num_channels=tuple(config["unet_num_channels"]),
        num_res_blocks=int(config["unet_num_res_blocks"]),
        attention_levels=tuple(bool(x) for x in config["unet_attention_levels"]),
        norm_num_groups=int(config["unet_norm_num_groups"]),
        norm_eps=float(config["unet_norm_eps"]),
        num_head_channels=tuple(config["unet_num_head_channels"]),
        with_conditioning=False,
    )


def build_scheduler(config: dict, scheduler_type: str, num_inference_steps: int):
    common_kwargs = {
        "num_train_timesteps": int(config["num_train_timesteps"]),
        "schedule": config["schedule"],
        "beta_start": float(config["beta_start"]),
        "beta_end": float(config["beta_end"]),
    }
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(**common_kwargs)
    else:
        scheduler = DDIMScheduler(clip_sample=False, **common_kwargs)
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def merged_config(args: argparse.Namespace, ldm_ckpt: dict) -> dict:
    config = vars(args).copy()
    parser = build_parser()
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }
    ckpt_args = ldm_ckpt.get("args", {}) if isinstance(ldm_ckpt, dict) else {}
    for key, value in ckpt_args.items():
        if key in config and config[key] == defaults.get(key):
            config[key] = value
    if isinstance(ldm_ckpt, dict) and ldm_ckpt.get("scale_factor") is not None and args.scale_factor <= 0:
        config["scale_factor"] = float(ldm_ckpt["scale_factor"])
    if isinstance(ldm_ckpt, dict) and ldm_ckpt.get("latent_shape") is not None:
        config["latent_shape"] = list(ldm_ckpt["latent_shape"])
    return config


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ldm_loaded = torch.load(args.ldm_checkpoint, map_location="cpu")
    config = merged_config(args, ldm_loaded if isinstance(ldm_loaded, dict) else {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = build_autoencoder(config).to(device)
    autoencoder.load_state_dict(extract_state_dict(torch.load(args.vae_checkpoint, map_location="cpu")), strict=True)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    unet = build_unet(config).to(device)
    unet.load_state_dict(extract_state_dict(ldm_loaded), strict=True)
    unet.eval()

    scale_factor = float(config["scale_factor"])
    if scale_factor <= 0:
        raise ValueError("scale_factor is missing. Use the training checkpoint best.pt or pass --scale_factor explicitly.")
    latent_shape = tuple(int(v) for v in config["latent_shape"])
    scheduler = build_scheduler(config, args.scheduler_type, args.num_inference_steps)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_pt"
    bin_dir = output_dir / "binary_pt"
    vis_dir = output_dir / "visualizations"
    nifti_dir = output_dir / "nifti"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_nifti:
        nifti_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "sample_config.json")

    all_metrics: list[dict[str, float | int | str | None]] = []
    generated = 0
    while generated < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - generated)
        noise = torch.randn((current_batch,) + latent_shape, device=device)
        with torch.no_grad():
            raw_samples = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                scheduler=scheduler,
                verbose=False,
            ).float()
            samples = torch.sigmoid(raw_samples)
        binary = (samples > args.threshold).float()
        ratios = vessel_ratio(binary).cpu().numpy()

        for idx in range(current_batch):
            sample_id = generated + idx
            raw_path = raw_dir / f"sample_{sample_id:04d}_raw.pt"
            bin_path = bin_dir / f"sample_{sample_id:04d}_bin.pt"
            torch.save({"sample": samples[idx : idx + 1].cpu()}, raw_path)
            torch.save({"sample": binary[idx : idx + 1].cpu()}, bin_path)
            save_generated_overview(
                mask=samples[idx : idx + 1].cpu(),
                out_path=vis_dir / f"sample_{sample_id:04d}.png",
                threshold=args.threshold,
                title="Generated",
            )
            if args.save_nifti:
                save_nifti_volume(samples[idx : idx + 1].cpu(), nifti_dir / f"sample_{sample_id:04d}_raw.nii.gz")
                save_nifti_volume(binary[idx : idx + 1].cpu(), nifti_dir / f"sample_{sample_id:04d}_bin.nii.gz")

            component_stats = connected_component_stats(binary[idx : idx + 1].cpu())
            all_metrics.append(
                {
                    "sample_id": sample_id,
                    "vessel_ratio": float(ratios[idx]),
                    "connected_components": component_stats["connected_components"],
                    "largest_component_voxels": component_stats["largest_component_voxels"],
                    "largest_component_ratio": component_stats["largest_component_ratio"],
                }
            )
        generated += current_batch

    summary = {
        "num_samples": len(all_metrics),
        "scheduler_type": args.scheduler_type,
        "num_inference_steps": args.num_inference_steps,
        "threshold": args.threshold,
        "scale_factor": scale_factor,
        "latent_shape": list(latent_shape),
        "vessel_ratio_mean": float(np.mean([item["vessel_ratio"] for item in all_metrics])) if all_metrics else 0.0,
        "vessel_ratio_std": float(np.std([item["vessel_ratio"] for item in all_metrics])) if all_metrics else 0.0,
        "connected_components_mean": float(
            np.mean([item["connected_components"] for item in all_metrics if item["connected_components"] is not None])
        )
        if any(item["connected_components"] is not None for item in all_metrics)
        else None,
        "largest_component_ratio_mean": float(
            np.mean(
                [item["largest_component_ratio"] for item in all_metrics if item["largest_component_ratio"] is not None]
            )
        )
        if any(item["largest_component_ratio"] is not None for item in all_metrics)
        else None,
    }
    save_json(summary, output_dir / "samples_summary.json")
    save_json(all_metrics, output_dir / "per_sample_metrics.json")

    print(json.dumps(summary, indent=2))
    print(f"Saved samples to {output_dir}")


if __name__ == "__main__":
    main()

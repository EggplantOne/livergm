"""
Download pretrained 3D VAE models from MONAI Model Zoo

Available models:
1. Brain MRI VAE - Trained on UK Biobank (31,740 3D brain MRI scans)
2. Chest X-ray VAE - Trained on MIMIC-CXR (90,000 2D chest X-rays)
"""

import os
import argparse
from pathlib import Path
import gdown
import torch


PRETRAINED_MODELS = {
    "brain_mri": {
        "autoencoder_url": "https://drive.google.com/uc?export=download&id=1CZHwxHJWybOsDavipD0EorDPOo_mzNeX",
        "diffusion_url": "https://drive.google.com/uc?export=download&id=1XO-ak93ZuOcGTCpgRtqgIeZq3dG5ExN6",
        "description": "3D Brain MRI VAE (160x224x160, trained on UK Biobank)",
        "spatial_dims": 3,
        "config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 3,
            "num_channels": [64, 128, 128, 128],
            "num_res_blocks": 2,
            "norm_num_groups": 32,
            "attention_levels": [False, False, False, False],
        }
    },
    "chest_xray": {
        "autoencoder_url": "https://drive.google.com/uc?export=download&id=1paDN1m-Q_Oy8d_BanPkRTi3RlNB_Sv_h",
        "diffusion_url": "https://drive.google.com/uc?export=download&id=1CjcmiPu5_QWr-f7wDJsXrCCcVeczneGT",
        "description": "2D Chest X-ray VAE (512x512, trained on MIMIC-CXR)",
        "spatial_dims": 2,
        "config": {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 3,
            "num_channels": [128, 256, 512, 512],
            "num_res_blocks": 2,
            "norm_num_groups": 32,
            "attention_levels": [False, True, True, True],
        }
    }
}


def download_model(url, output_path):
    """Download model from Google Drive."""
    print(f"Downloading to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(output_path), quiet=False)
    print(f"Downloaded successfully!")


def inspect_model(model_path):
    """Inspect the structure of a pretrained model."""
    print(f"\nInspecting model: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {checkpoint.keys()}")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print(f"\nModel architecture (first 10 layers):")
    for i, (key, value) in enumerate(list(state_dict.items())[:10]):
        print(f"  {key}: {value.shape}")

    print(f"\nTotal parameters: {len(state_dict)}")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameter count: {total_params:,}")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained VAE models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["brain_mri", "chest_xray", "both"],
        default="brain_mri",
        help="Which pretrained model to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pretrained_models",
        help="Directory to save downloaded models"
    )
    parser.add_argument(
        "--download_diffusion",
        action="store_true",
        help="Also download the diffusion model (not just VAE)"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the downloaded model structure"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_download = ["brain_mri", "chest_xray"] if args.model == "both" else [args.model]

    for model_name in models_to_download:
        model_info = PRETRAINED_MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Configuration:")
        for key, value in model_info['config'].items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        # Download autoencoder
        autoencoder_path = output_dir / f"{model_name}_autoencoder.pth"
        if not autoencoder_path.exists():
            download_model(model_info["autoencoder_url"], autoencoder_path)
        else:
            print(f"Autoencoder already exists at {autoencoder_path}")

        if args.inspect:
            inspect_model(autoencoder_path)

        # Download diffusion model if requested
        if args.download_diffusion:
            diffusion_path = output_dir / f"{model_name}_diffusion.pth"
            if not diffusion_path.exists():
                download_model(model_info["diffusion_url"], diffusion_path)
            else:
                print(f"Diffusion model already exists at {diffusion_path}")

            if args.inspect:
                inspect_model(diffusion_path)

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Models saved to: {output_dir}")
    print(f"\nTo use for fine-tuning, add this argument to train_vae_vessel.py:")
    print(f"  --pretrained_model {output_dir}/brain_mri_autoencoder.pth")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

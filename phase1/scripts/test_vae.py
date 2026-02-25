"""
Test and visualize trained 3D VAE model

This script loads a trained VAE and performs:
1. Reconstruction test on real data
2. Latent space interpolation
3. Random sampling from latent space
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import nrrd
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from monai import transforms
from monai.data import Dataset, DataLoader

from generative.networks.nets import AutoencoderKL


def tensor_only_collate(batch):
    """Collate dict samples into plain torch.Tensor batches."""
    images = []
    for item in batch:
        image = item["image"]
        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        else:
            image = torch.as_tensor(image)
        images.append(image.clone().detach().float())
    return {"image": torch.stack(images, dim=0)}


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained VAE model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained VAE model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test NRRD files")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128], help="Spatial size (must match training)")

    # Model architecture (must match training config)
    parser.add_argument("--latent_channels", type=int, default=3, help="Number of latent channels")
    parser.add_argument("--num_channels", type=int, nargs="+", default=[64, 128, 128, 128], help="Channel multipliers")
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[0, 0, 0, 0], help="Attention levels")

    return parser.parse_args()


def load_medical_image(filepath):
    """Load NRRD or NIfTI file."""
    filepath = str(filepath)

    if filepath.endswith(('.nrrd', '.nrrd.gz')):
        data, header = nrrd.read(filepath)
        return data
    elif filepath.endswith(('.nii', '.nii.gz')):
        img = nib.load(filepath)
        data = img.get_fdata()
        return data
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


class LoadMedicalImaged(transforms.MapTransform):
    """Custom transform to load NRRD or NIfTI files."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = load_medical_image(d[key])
            if d[key].ndim == 3:
                d[key] = d[key][np.newaxis, ...]
        return d


def visualize_3d_slice(volume, title, save_path):
    """Visualize 3 orthogonal slices of a 3D volume."""
    volume = volume.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice (Z)
    axes[0].imshow(volume[:, :, volume.shape[2] // 2], cmap='gray')
    axes[0].set_title(f'{title} - Axial')
    axes[0].axis('off')

    # Coronal slice (Y)
    axes[1].imshow(volume[:, volume.shape[1] // 2, :], cmap='gray')
    axes[1].set_title(f'{title} - Coronal')
    axes[1].axis('off')

    # Sagittal slice (X)
    axes[2].imshow(volume[volume.shape[0] // 2, :, :], cmap='gray')
    axes[2].set_title(f'{title} - Sagittal')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_reconstruction(original, reconstructed, save_path):
    """Visualize original and reconstructed volumes side by side."""
    original = original.squeeze().cpu().numpy()
    reconstructed = reconstructed.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original - 3 views
    axes[0, 0].imshow(original[:, :, original.shape[2] // 2], cmap='gray')
    axes[0, 0].set_title('Original - Axial')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(original[:, original.shape[1] // 2, :], cmap='gray')
    axes[0, 1].set_title('Original - Coronal')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(original[original.shape[0] // 2, :, :], cmap='gray')
    axes[0, 2].set_title('Original - Sagittal')
    axes[0, 2].axis('off')

    # Reconstructed - 3 views
    axes[1, 0].imshow(reconstructed[:, :, reconstructed.shape[2] // 2], cmap='gray')
    axes[1, 0].set_title('Reconstructed - Axial')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(reconstructed[:, reconstructed.shape[1] // 2, :], cmap='gray')
    axes[1, 1].set_title('Reconstructed - Coronal')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(reconstructed[reconstructed.shape[0] // 2, :, :], cmap='gray')
    axes[1, 2].set_title('Reconstructed - Sagittal')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(original, reconstructed):
    """Compute reconstruction metrics."""
    if original.shape != reconstructed.shape:
        # Align to common spatial shape to avoid crashes on edge cases.
        min_d = min(original.shape[2], reconstructed.shape[2])
        min_h = min(original.shape[3], reconstructed.shape[3])
        min_w = min(original.shape[4], reconstructed.shape[4])
        original = original[:, :, :min_d, :min_h, :min_w]
        reconstructed = reconstructed[:, :, :min_d, :min_h, :min_w]

    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()

    # PSNR
    max_val = 1.0
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')

    return {"MSE": mse, "MAE": mae, "PSNR": psnr}


def test_reconstruction(model, data_loader, device, output_dir, num_samples):
    """Test reconstruction quality."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Testing Reconstruction Quality")
    print("="*60)

    all_metrics = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx >= num_samples:
                break

            images = batch["image"].to(device)
            reconstruction, z_mu, z_sigma = model(images)

            # Compute metrics
            metrics = compute_metrics(images, reconstruction)
            all_metrics.append(metrics)

            print(f"\nSample {idx + 1}:")
            print(f"  MSE: {metrics['MSE']:.6f}")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  PSNR: {metrics['PSNR']:.2f} dB")
            print(f"  Latent mean: {z_mu.mean().item():.4f} ± {z_mu.std().item():.4f}")
            print(f"  Latent std: {z_sigma.mean().item():.4f} ± {z_sigma.std().item():.4f}")

            # Visualize
            save_path = output_dir / f"reconstruction_{idx + 1}.png"
            visualize_reconstruction(images[0], reconstruction[0], save_path)
            print(f"  Saved visualization to {save_path}")

    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    print(f"\n{'='*60}")
    print("Average Metrics:")
    print(f"  MSE: {avg_metrics['MSE']:.6f}")
    print(f"  MAE: {avg_metrics['MAE']:.6f}")
    print(f"  PSNR: {avg_metrics['PSNR']:.2f} dB")
    print(f"{'='*60}\n")


def test_latent_interpolation(model, data_loader, device, output_dir, num_steps=10):
    """Test latent space interpolation between two samples."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Testing Latent Space Interpolation")
    print("="*60)

    with torch.no_grad():
        # Get two samples
        batch_iter = iter(data_loader)
        try:
            batch1 = next(batch_iter)
            batch2 = next(batch_iter)
        except StopIteration as exc:
            raise RuntimeError("Need at least 2 samples in data_loader for interpolation.") from exc

        img1 = batch1["image"].to(device)
        img2 = batch2["image"].to(device)

        # Encode
        z1_mu, _ = model.encode(img1)
        z2_mu, _ = model.encode(img2)

        print(f"Interpolating between two latent codes...")
        print(f"Latent shape: {z1_mu.shape}")

        # Interpolate
        alphas = torch.linspace(0, 1, num_steps).to(device)

        fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 2, 4))

        for i, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z1_mu + alpha * z2_mu
            reconstruction = model.decode(z_interp)

            # Visualize axial and coronal slices
            recon_np = reconstruction[0, 0].cpu().numpy()
            axes[0, i].imshow(recon_np[:, :, recon_np.shape[2] // 2], cmap='gray')
            axes[0, i].set_title(f'α={alpha:.2f}')
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_np[:, recon_np.shape[1] // 2, :], cmap='gray')
            axes[1, i].axis('off')

        plt.tight_layout()
        save_path = output_dir / "latent_interpolation.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved interpolation to {save_path}\n")


def test_random_sampling(model, device, output_dir, spatial_size, num_samples=5):
    """Test random sampling from latent space."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Testing Random Sampling from Latent Space")
    print("="*60)

    with torch.no_grad():
        # Infer latent shape from the current model by a single encode pass.
        dummy = torch.zeros(1, 1, *spatial_size, device=device)
        z_mu, _ = model.encode(dummy)
        latent_spatial = list(z_mu.shape[2:])
        latent_shape = (num_samples, model.latent_channels, *latent_spatial)

        print(f"Sampling {num_samples} random latent codes...")
        print(f"Latent shape: {latent_shape}")

        # Sample from standard normal
        z_random = torch.randn(latent_shape).to(device)

        # Decode
        samples = model.decode(z_random)

        # Visualize
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

        for i in range(num_samples):
            sample_np = samples[i, 0].cpu().numpy()

            # Axial
            axes[0, i].imshow(sample_np[:, :, sample_np.shape[2] // 2], cmap='gray')
            axes[0, i].set_title(f'Sample {i+1}')
            axes[0, i].axis('off')

            # Coronal
            axes[1, i].imshow(sample_np[:, sample_np.shape[1] // 2, :], cmap='gray')
            axes[1, i].axis('off')

            # Sagittal
            axes[2, i].imshow(sample_np[sample_np.shape[0] // 2, :, :], cmap='gray')
            axes[2, i].axis('off')

        plt.tight_layout()
        save_path = output_dir / "random_samples.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved random samples to {save_path}\n")


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=args.num_channels,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=[bool(x) for x in args.attention_levels],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Prepare data
    data_dir = Path(args.data_dir)
    nrrd_files = list(data_dir.glob("**/*.nrrd")) + list(data_dir.glob("**/*.nrrd.gz"))
    nifti_files = list(data_dir.glob("**/*.nii")) + list(data_dir.glob("**/*.nii.gz"))

    all_files = nrrd_files + nifti_files

    if len(all_files) == 0:
        raise ValueError(f"No NRRD or NIfTI files found in {data_dir}")

    print(f"Found {len(nrrd_files)} NRRD files and {len(nifti_files)} NIfTI files")

    data_dicts = [{"image": str(f)} for f in all_files[:args.num_samples * 2]]

    # Create transforms (use Resize to match training)
    test_transforms = transforms.Compose([
        LoadMedicalImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim=0),
        transforms.EnsureTyped(keys=["image"], track_meta=False),
        transforms.Resized(keys=["image"], spatial_size=args.spatial_size, mode="nearest"),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
    ])

    test_ds = Dataset(data=data_dicts, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=tensor_only_collate)

    # Run tests
    output_dir = Path(args.output_dir)

    # Test 1: Reconstruction
    test_reconstruction(model, test_loader, device, output_dir / "reconstruction", args.num_samples)

    # Test 2: Latent interpolation
    test_latent_interpolation(model, test_loader, device, output_dir / "interpolation")

    # Test 3: Random sampling
    test_random_sampling(model, device, output_dir / "random_samples", args.spatial_size)

    print("\n" + "="*60)
    print("All tests completed!")
    print(f"Results saved to {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

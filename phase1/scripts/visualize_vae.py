"""
Visualize and Sample from Trained VAE

This script allows you to:
1. Reconstruct real vessel masks
2. Generate synthetic vessel masks by sampling from latent space
3. Interpolate between two vessel masks
4. Save results as images and NRRD/NIfTI files
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import nrrd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from monai import transforms
from monai.data import Dataset, DataLoader

from generative.networks.nets import AutoencoderKL


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and sample from trained VAE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained VAE model")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of synthetic samples to generate")
    parser.add_argument("--save_format", type=str, default="nrrd", choices=["nrrd", "nifti"], help="Format to save 3D volumes")

    # Optional: provide real data for reconstruction
    parser.add_argument("--data_dir", type=str, default=None, help="Directory with real data for reconstruction")
    parser.add_argument("--num_reconstructions", type=int, default=5, help="Number of reconstructions to show")

    # Model architecture (must match training)
    parser.add_argument("--latent_channels", type=int, default=3, help="Number of latent channels")
    parser.add_argument("--num_channels", type=int, nargs="+", default=[32, 64, 128], help="Channel multipliers")
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[0, 0, 1], help="Attention levels")

    # Sampling parameters
    parser.add_argument("--latent_size", type=int, nargs=3, default=[8, 8, 8], help="Latent spatial size")
    parser.add_argument("--output_size", type=int, nargs=3, default=[64, 64, 64], help="Output volume size")

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


def save_volume(volume, filepath, format="nrrd"):
    """Save 3D volume as NRRD or NIfTI."""
    volume = volume.squeeze().cpu().numpy()

    if format == "nrrd":
        nrrd.write(str(filepath), volume)
    elif format == "nifti":
        img = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(img, str(filepath))


def visualize_3d_volume(volume, title, save_path, num_slices=5):
    """Visualize multiple slices of a 3D volume."""
    volume = volume.squeeze()

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, num_slices, figure=fig)

    # Axial slices
    for i in range(num_slices):
        ax = fig.add_subplot(gs[0, i])
        slice_idx = int(volume.shape[2] * (i + 1) / (num_slices + 1))
        ax.imshow(volume[:, :, slice_idx], cmap='gray')
        ax.set_title(f'Axial {slice_idx}')
        ax.axis('off')

    # Coronal slices
    for i in range(num_slices):
        ax = fig.add_subplot(gs[1, i])
        slice_idx = int(volume.shape[1] * (i + 1) / (num_slices + 1))
        ax.imshow(volume[:, slice_idx, :], cmap='gray')
        ax.set_title(f'Coronal {slice_idx}')
        ax.axis('off')

    # Sagittal slices
    for i in range(num_slices):
        ax = fig.add_subplot(gs[2, i])
        slice_idx = int(volume.shape[0] * (i + 1) / (num_slices + 1))
        ax.imshow(volume[slice_idx, :, :], cmap='gray')
        ax.set_title(f'Sagittal {slice_idx}')
        ax.axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_3d_rendering(volume, save_path):
    """Create a simple 3D rendering using maximum intensity projection."""
    volume = volume.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MIP along each axis
    axes[0].imshow(np.max(volume, axis=2), cmap='hot')
    axes[0].set_title('MIP - Axial View')
    axes[0].axis('off')

    axes[1].imshow(np.max(volume, axis=1), cmap='hot')
    axes[1].set_title('MIP - Coronal View')
    axes[1].axis('off')

    axes[2].imshow(np.max(volume, axis=0), cmap='hot')
    axes[2].set_title('MIP - Sagittal View')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_synthetic_samples(model, device, num_samples, latent_shape, output_dir, save_format):
    """Generate synthetic vessel masks by sampling from latent space."""
    print("\n" + "="*60)
    print("Generating Synthetic Vessel Masks")
    print("="*60)

    model.eval()
    output_dir = Path(output_dir) / "synthetic_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(num_samples):
            # Sample from standard normal distribution
            z = torch.randn(1, *latent_shape).to(device)

            # Decode to generate vessel mask
            synthetic_vessel = model.decode(z)

            # Clamp to [0, 1]
            synthetic_vessel = torch.clamp(synthetic_vessel, 0, 1)

            print(f"Sample {i+1}/{num_samples}:")
            print(f"  Shape: {synthetic_vessel.shape}")
            print(f"  Value range: [{synthetic_vessel.min():.3f}, {synthetic_vessel.max():.3f}]")
            print(f"  Mean: {synthetic_vessel.mean():.3f}")

            # Save as image
            img_path = output_dir / f"synthetic_vessel_{i+1:03d}_slices.png"
            visualize_3d_volume(synthetic_vessel[0, 0].cpu().numpy(),
                              f"Synthetic Vessel {i+1}", img_path)

            # Save MIP
            mip_path = output_dir / f"synthetic_vessel_{i+1:03d}_mip.png"
            create_3d_rendering(synthetic_vessel[0, 0].cpu().numpy(), mip_path)

            # Save 3D volume
            if save_format == "nrrd":
                volume_path = output_dir / f"synthetic_vessel_{i+1:03d}.nrrd"
            else:
                volume_path = output_dir / f"synthetic_vessel_{i+1:03d}.nii.gz"

            save_volume(synthetic_vessel[0, 0], volume_path, save_format)

            print(f"  Saved to: {volume_path}")

    print(f"\n✅ Generated {num_samples} synthetic vessel masks")
    print(f"📁 Saved to: {output_dir}")


def reconstruct_real_data(model, data_dir, device, num_reconstructions, output_dir, save_format):
    """Reconstruct real vessel masks."""
    print("\n" + "="*60)
    print("Reconstructing Real Vessel Masks")
    print("="*60)

    # Load data
    data_dir = Path(data_dir)
    nrrd_files = list(data_dir.glob("**/*.nrrd")) + list(data_dir.glob("**/*.nrrd.gz"))
    nifti_files = list(data_dir.glob("**/*.nii")) + list(data_dir.glob("**/*.nii.gz"))
    all_files = nrrd_files + nifti_files

    if len(all_files) == 0:
        print("⚠️  No data files found, skipping reconstruction")
        return

    print(f"Found {len(all_files)} files")

    # Take first few files
    files_to_reconstruct = all_files[:num_reconstructions]

    output_dir = Path(output_dir) / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for i, filepath in enumerate(files_to_reconstruct):
            # Load image
            data = load_medical_image(filepath)
            if data.ndim == 3:
                data = data[np.newaxis, ...]

            # Convert to tensor
            data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)

            # Normalize
            data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-8)

            # Reconstruct
            reconstruction, z_mu, z_sigma = model(data_tensor)

            print(f"\nReconstruction {i+1}/{num_reconstructions}:")
            print(f"  Original file: {filepath.name}")
            print(f"  Shape: {reconstruction.shape}")
            print(f"  Latent mean: {z_mu.mean():.4f} ± {z_mu.std():.4f}")

            # Save comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            original = data_tensor[0, 0].cpu().numpy()
            recon = reconstruction[0, 0].cpu().numpy()

            # Original
            axes[0, 0].imshow(original[:, :, original.shape[2]//2], cmap='gray')
            axes[0, 0].set_title('Original - Axial')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(original[:, original.shape[1]//2, :], cmap='gray')
            axes[0, 1].set_title('Original - Coronal')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(original[original.shape[0]//2, :, :], cmap='gray')
            axes[0, 2].set_title('Original - Sagittal')
            axes[0, 2].axis('off')

            # Reconstruction
            axes[1, 0].imshow(recon[:, :, recon.shape[2]//2], cmap='gray')
            axes[1, 0].set_title('Reconstructed - Axial')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(recon[:, recon.shape[1]//2, :], cmap='gray')
            axes[1, 1].set_title('Reconstructed - Coronal')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(recon[recon.shape[0]//2, :, :], cmap='gray')
            axes[1, 2].set_title('Reconstructed - Sagittal')
            axes[1, 2].axis('off')

            plt.tight_layout()
            save_path = output_dir / f"reconstruction_{i+1:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Save reconstructed volume
            if save_format == "nrrd":
                volume_path = output_dir / f"reconstructed_{i+1:03d}.nrrd"
            else:
                volume_path = output_dir / f"reconstructed_{i+1:03d}.nii.gz"

            save_volume(reconstruction[0, 0], volume_path, save_format)

            print(f"  Saved to: {save_path}")

    print(f"\n✅ Reconstructed {len(files_to_reconstruct)} vessel masks")
    print(f"📁 Saved to: {output_dir}")


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    # Generate synthetic samples
    latent_shape = (args.latent_channels, *args.latent_size)
    generate_synthetic_samples(model, device, args.num_samples, latent_shape,
                              output_dir, args.save_format)

    # Reconstruct real data if provided
    if args.data_dir:
        reconstruct_real_data(model, args.data_dir, device, args.num_reconstructions,
                            output_dir, args.save_format)

    print("\n" + "="*60)
    print("✅ All visualizations completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*60)
    print("\nGenerated files:")
    print("  - synthetic_samples/synthetic_vessel_*.png (slice views)")
    print("  - synthetic_samples/synthetic_vessel_*_mip.png (3D rendering)")
    print(f"  - synthetic_samples/synthetic_vessel_*.{args.save_format} (3D volumes)")
    if args.data_dir:
        print("  - reconstructions/reconstruction_*.png (comparisons)")
        print(f"  - reconstructions/reconstructed_*.{args.save_format} (3D volumes)")
    print("\n💡 You can open the 3D volumes in:")
    print("  - 3D Slicer (https://www.slicer.org/)")
    print("  - ITK-SNAP (http://www.itksnap.org/)")
    print("  - ParaView (https://www.paraview.org/)")
    print()


if __name__ == "__main__":
    main()

"""
Phase 1 - Step 1: Train 3D AutoencoderKL for Vessel Mask Compression

This script trains a 3D VAE to compress vessel masks into latent space.
The trained VAE will be used in both Phase 1 (unconditional generation) and Phase 2 (ControlNet).
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import nrrd
import nibabel as nib
from tqdm import tqdm

from monai import transforms
from monai.data import Dataset, DataLoader, CacheDataset
from monai.utils import set_determinism

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D VAE for vessel masks")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing NRRD vessel masks")
    parser.add_argument("--output_dir", type=str, default="./outputs/vae", help="Output directory for checkpoints")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="Cache rate for data loading (0-1)")

    # Model architecture
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[64, 64, 64], help="Spatial size for training patches")
    parser.add_argument("--latent_channels", type=int, default=3, help="Number of latent channels")
    parser.add_argument("--num_channels", type=int, nargs="+", default=[32, 64, 128], help="Channel multipliers")
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[False, False, True], help="Attention at each level")

    # Training
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="Weight for KL divergence loss")
    parser.add_argument("--adv_weight", type=float, default=0.01, help="Weight for adversarial loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.0, help="Weight for perceptual loss")

    # System
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10, help="Checkpoint save interval (epochs)")
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable tqdm progress bars")

    # Pretrained model
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained VAE model for fine-tuning")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights during fine-tuning")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")

    return parser.parse_args()


def get_data_dicts(data_dir):
    """Scan directory for NRRD and NIfTI files and create data dictionaries."""
    data_dir = Path(data_dir)

    # Support both NRRD and NIfTI formats
    nrrd_files = list(data_dir.glob("**/*.nrrd")) + list(data_dir.glob("**/*.nrrd.gz"))
    nifti_files = list(data_dir.glob("**/*.nii")) + list(data_dir.glob("**/*.nii.gz"))

    all_files = nrrd_files + nifti_files

    if len(all_files) == 0:
        raise ValueError(f"No NRRD or NIfTI files found in {data_dir}")

    print(f"Found {len(nrrd_files)} NRRD files and {len(nifti_files)} NIfTI files")
    print(f"Total: {len(all_files)} files")

    data_dicts = [{"image": str(f)} for f in all_files]

    # Split into train/val (90/10)
    split_idx = int(len(data_dicts) * 0.9)
    train_dicts = data_dicts[:split_idx]
    val_dicts = data_dicts[split_idx:]

    print(f"Train: {len(train_dicts)}, Val: {len(val_dicts)}")

    return train_dicts, val_dicts


def load_medical_image(filepath):
    """Load NRRD or NIfTI file and return as numpy array."""
    filepath = str(filepath)

    if filepath.endswith(('.nrrd', '.nrrd.gz')):
        # Load NRRD
        data, header = nrrd.read(filepath)
        return data
    elif filepath.endswith(('.nii', '.nii.gz')):
        # Load NIfTI
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
            # Ensure channel first: (C, H, W, D)
            if d[key].ndim == 3:
                d[key] = d[key][np.newaxis, ...]
        return d


def get_transforms(spatial_size, is_train=True):
    """Create data transforms for vessel masks."""
    base_transforms = [
        LoadMedicalImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim=0),
        transforms.EnsureTyped(keys=["image"], track_meta=False),
    ]

    if is_train:
        # Training: random crop + augmentation
        train_transforms = base_transforms + [
            transforms.RandSpatialCropd(keys=["image"], roi_size=spatial_size, random_size=False),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            transforms.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        ]
        # Normalize to [0, 1] - vessel masks are typically binary or probability maps
        train_transforms.append(transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0, b_max=1, clip=True))
        return transforms.Compose(train_transforms)
    else:
        # Validation: center crop only
        val_transforms = base_transforms + [
            transforms.CenterSpatialCropd(keys=["image"], roi_size=spatial_size),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
        ]
        return transforms.Compose(val_transforms)


def main():
    args = parse_args()

    # Set seed
    set_determinism(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data
    train_dicts, val_dicts = get_data_dicts(args.data_dir)

    # Create datasets
    train_transforms = get_transforms(args.spatial_size, is_train=True)
    val_transforms = get_transforms(args.spatial_size, is_train=False)

    if args.cache_rate > 0:
        train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
        val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    else:
        train_ds = Dataset(data=train_dicts, transform=train_transforms)
        val_ds = Dataset(data=val_dicts, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, persistent_workers=True if args.num_workers > 0 else False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create models
    autoencoderkl = AutoencoderKL(
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

    # Load pretrained model if specified
    start_epoch = 0
    if args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
        try:
            pretrained_dict = torch.load(args.pretrained_model, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]

            # Load with strict=False to allow partial loading
            missing_keys, unexpected_keys = autoencoderkl.load_state_dict(pretrained_dict, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys in pretrained model: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in pretrained model: {unexpected_keys[:5]}...")

            print("Pretrained model loaded successfully!")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Starting from scratch...")

    # Freeze encoder if specified
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        for name, param in autoencoderkl.named_parameters():
            if "encoder" in name or "quant_conv" in name:
                param.requires_grad = False
        print(f"Trainable parameters: {sum(p.numel() for p in autoencoderkl.parameters() if p.requires_grad):,}")

    autoencoderkl = autoencoderkl.to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=3,
        in_channels=1,
        num_layers_d=3,
        num_channels=64,
        kernel_size=4,
        activation="LEAKYRELU",
        norm="BATCH",
    )
    discriminator = discriminator.to(device)

    # Perceptual loss network (optional, disabled by default to avoid extra external downloads)
    perceptual_loss = None
    if args.perceptual_weight > 0:
        perceptual_loss = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.5,
        ).to(device)
        print(f"Perceptual loss enabled (weight={args.perceptual_weight})")
    else:
        print("Perceptual loss disabled (set --perceptual_weight > 0 to enable)")

    # Loss functions
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    l1_loss = torch.nn.L1Loss()

    # Optimizers (only optimize trainable parameters)
    optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, autoencoderkl.parameters()), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Gradient scaler for mixed precision
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # Resume from checkpoint if specified
    global_step = 0
    best_val_loss = float('inf')

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        autoencoderkl.load_state_dict(checkpoint["autoencoderkl_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.num_epochs):
        autoencoderkl.train()
        discriminator.train()

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=args.no_progress_bar,
        )

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device)

            # Generator update
            optimizer_g.zero_grad()

            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoderkl(images)

                # Reconstruction loss
                recon_loss = l1_loss(reconstruction, images)

                # KL divergence loss
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                kl_loss = torch.mean(kl_loss)

                # Perceptual loss (optional)
                if perceptual_loss is not None:
                    p_loss = perceptual_loss(reconstruction, images)
                else:
                    p_loss = torch.tensor(0.0, device=device)

                # Adversarial loss (generator)
                logits_fake = discriminator(reconstruction)
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                # Total generator loss
                loss_g = recon_loss + args.kl_weight * kl_loss + args.perceptual_weight * p_loss + args.adv_weight * generator_loss

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # Discriminator update
            optimizer_d.zero_grad()

            with autocast(enabled=True):
                logits_real = discriminator(images)
                logits_fake = discriminator(reconstruction.detach())

                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d = (loss_d_real + loss_d_fake) * 0.5

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # Logging
            epoch_loss += loss_g.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_gen_loss += generator_loss.item()
            epoch_disc_loss += loss_d.item()

            if not args.no_progress_bar:
                progress_bar.set_postfix({
                    "loss": f"{loss_g.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_loss.item():.6f}",
                })

            # TensorBoard logging
            if global_step % 10 == 0:
                writer.add_scalar("train/total_loss", loss_g.item(), global_step)
                writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)
                writer.add_scalar("train/gen_loss", generator_loss.item(), global_step)
                writer.add_scalar("train/disc_loss", loss_d.item(), global_step)

            global_step += 1

        # Epoch statistics
        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_kl_loss /= len(train_loader)
        epoch_gen_loss /= len(train_loader)
        epoch_disc_loss /= len(train_loader)

        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.6f}, Gen: {epoch_gen_loss:.4f}, Disc: {epoch_disc_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            autoencoderkl.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)

                    recon_loss = l1_loss(reconstruction, images)
                    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                    kl_loss = torch.mean(kl_loss)

                    loss = recon_loss + args.kl_weight * kl_loss
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            writer.add_scalar("val/loss", val_loss, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(autoencoderkl.state_dict(), output_dir / "best_vae.pth")
                print(f"Saved best model with val_loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "autoencoderkl_state_dict": autoencoderkl.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(autoencoderkl.state_dict(), output_dir / "final_vae.pth")
    print("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()

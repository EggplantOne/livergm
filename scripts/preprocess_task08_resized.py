"""Preprocess Task08_HepaticVessel NIfTI → .pt cache (1mm isotropic + center crop).

Pipeline per volume:
  1. Load NIfTI CT image + vessel label (raw HU)
  2. Crop to vessel bounding box + margin_mm (raw HU)
  3. Resample to target_spacing isotropic (raw HU)
     - image: order=1 (linear), matching DiffTumor STEP1/2 for anisotropic z-axis
     - label: order=0 (nearest), preserving binary values
  4. Center-crop to target_size
     - At 1mm iso + 16mm margin, all Task08 volumes > 128³, no padding needed
  5. CT windowing → [-1, 1]
  6. Binarize mask
  7. Save {image, mask} as .pt

Design decisions (see commit message for full reasoning):
  - 1mm isotropic: DiffTumor STEP1/2 validated for liver CT generation;
    3D conv requires isotropic; at 1mm, ROI > 128³ → no black padding
  - order=1 for image resample: DiffTumor uses bilinear; compromise between
    cubic (hallucinated detail) and nearest (staircase artifacts) for 5→1mm z-axis
  - CT window [-3, 243] (center=120, width=246): nnUNet Task08 foreground
    P0.5/P99.5 data-driven range. Clips only 1.8% vessel voxels with good
    contrast. CVI's [-30, 170] clips 43.5%; DiffTumor's [-175, 250] has
    too low contrast (width=425).
  - Center crop: all spatial operations in raw HU space before windowing,
    which is more physically meaningful for interpolation
  - References: DiffTumor (CVPR 2024), nnUNet, Medical Diffusion (Khader 2023)

Usage:
    conda activate gm
    python scripts/preprocess_task08_resized.py \\
        --data_root /mnt/no1/yinhaojie/Task08_HepaticVessel \\
        --output_dir /home/yinhaojie/GenerativeModels/data/cache_1mm \\
        --target_size 128 128 128 \\
        --target_spacing 1.0 \\
        --margin_mm 16.0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm


# ── CT windowing ──────────────────────────────────────────────────────────────
# nnUNet Task08 HepaticVessel: clip to [-3, 243] (P0.5/P99.5 of foreground voxels).
# This is data-driven from DiffTumor STEP3 / nnUNet dataset statistics.
# Comparison:
#   [-30, 170] (old CVI):   clips 43.5% vessel voxels, too narrow
#   [-175, 250] (DiffTumor STEP1/2): clips 1.4%, but contrast too low (width=425)
#   [-3, 243] (nnUNet P0.5/P99.5):  clips 1.8%, good contrast (width=246)
CT_WINDOW_CENTER = 120.0  # (-3 + 243) / 2
CT_WINDOW_WIDTH = 246.0   # 243 - (-3)


def apply_ct_window(volume: np.ndarray,
                    center: float = CT_WINDOW_CENTER,
                    width: float = CT_WINDOW_WIDTH) -> np.ndarray:
    """Apply CT windowing and normalize to [-1, 1]."""
    lo = center - width / 2
    hi = center + width / 2
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)   # [0, 1]
    volume = volume * 2.0 - 1.0          # [-1, 1]
    return volume.astype(np.float32)


# ── NIfTI loading ─────────────────────────────────────────────────────────────
def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI, return (data, voxel_spacing).

    Spacing is extracted via header.get_zooms() which correctly handles
    oblique acquisitions (affine with off-diagonal elements).
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    return data, spacing


# ── Crop to vessel ROI ────────────────────────────────────────────────────────
def crop_to_vessel_roi(image: np.ndarray,
                       label: np.ndarray,
                       vessel_label: int,
                       margin_mm: float,
                       spacing: np.ndarray,
                       target_spacing: float = 1.0,
                       min_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and label to vessel bounding box + margin (mm).

    Margin is converted from mm to voxels using the original spacing.
    If any dimension after resample would be < min_size, margin is
    automatically increased to ensure sufficient size.
    Crop is clipped to image bounds.
    """
    vessel = (label == vessel_label)
    if vessel.sum() == 0:
        return image, label

    coords = np.argwhere(vessel)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size = bbox_max - bbox_min + 1
    img_shape = np.array(image.shape)

    # Minimum crop size in original voxels so that after resample >= min_size
    min_vox = np.ceil(min_size * target_spacing / spacing).astype(int)
    margin_vox = np.ceil(margin_mm / spacing).astype(int)
    needed = np.maximum((min_vox - bbox_size + 1) // 2 + 1, margin_vox)

    crop_min = np.maximum(bbox_min - needed, 0)
    crop_max = np.minimum(bbox_max + needed + 1, img_shape)

    # If one side hit the image boundary, expand the other side to compensate
    crop_size = crop_max - crop_min
    for i in range(3):
        if crop_size[i] < min_vox[i]:
            deficit = min_vox[i] - crop_size[i]
            if crop_min[i] > 0:
                expand = min(deficit, crop_min[i])
                crop_min[i] -= expand
                deficit -= expand
            if deficit > 0 and crop_max[i] < img_shape[i]:
                crop_max[i] = min(crop_max[i] + deficit, img_shape[i])

    slices = tuple(slice(lo, hi) for lo, hi in zip(crop_min, crop_max))
    return image[slices], label[slices]


# ── Resample to isotropic spacing ────────────────────────────────────────────
def resample_volume(volume: np.ndarray,
                    src_spacing: np.ndarray,
                    target_spacing: float,
                    order: int = 1) -> np.ndarray:
    """Resample volume from src_spacing to isotropic target_spacing.

    Args:
        order: 1 for image (linear, DiffTumor STEP1/2 uses bilinear),
               0 for label (nearest, preserves binary values).
               Linear is chosen over cubic (order=3) because Task08 z-axis
               is highly anisotropic (5mm→1mm = 5x upsampling); cubic would
               hallucinate fine detail that doesn't exist in the original data.
    """
    zoom_factors = src_spacing / target_spacing
    return zoom(volume, zoom_factors, order=order).astype(np.float32)


# ── Center crop ──────────────────────────────────────────────────────────────
def center_crop(volume: np.ndarray,
                target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center-crop volume to target_shape.

    At 1mm isotropic with 16mm margin, all Task08 vessel ROI > 128³.
    Raises ValueError if any dimension is smaller than target.
    """
    for i, (s, t) in enumerate(zip(volume.shape, target_shape)):
        if s < t:
            raise ValueError(
                f"Volume dim {i} ({s}) < target ({t}). "
                f"Volume shape {volume.shape} cannot be center-cropped to {target_shape}."
            )

    slices = []
    for s, t in zip(volume.shape, target_shape):
        start = (s - t) // 2
        slices.append(slice(start, start + t))

    return volume[tuple(slices)].astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        "Preprocess Task08 HepaticVessel: 1mm isotropic + center crop"
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to Task08_HepaticVessel root (contains dataset.json)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for .pt cache files")
    p.add_argument("--target_size", type=int, nargs=3, default=[128, 128, 128],
                   help="Target volume size (D H W)")
    p.add_argument("--target_spacing", type=float, default=1.0,
                   help="Target isotropic spacing in mm (default: 1.0)")
    p.add_argument("--margin_mm", type=float, default=16.0,
                   help="Margin around vessel bbox in mm")
    p.add_argument("--vessel_label", type=int, default=1,
                   help="Label value for vessel in NIfTI (Task08: 1=vessel, 2=tumor)")
    p.add_argument("--ct_window_center", type=float, default=CT_WINDOW_CENTER)
    p.add_argument("--ct_window_width", type=float, default=CT_WINDOW_WIDTH)
    p.add_argument("--force", action="store_true",
                   help="Re-process even if .pt already exists")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Process at most N samples (0=all, for preview)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.target_size)

    # Load dataset.json to get image/label pairs
    with open(data_root / "dataset.json", "r", encoding="utf-8-sig") as f:
        info = json.load(f)

    pairs = []
    for entry in info["training"]:
        img_path = data_root / entry["image"].lstrip("./")
        lbl_path = data_root / entry["label"].lstrip("./")
        if img_path.exists() and lbl_path.exists():
            pairs.append((str(img_path), str(lbl_path)))

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    print(f"Found {len(pairs)} image-label pairs")
    print(f"Target: {target_size} at {args.target_spacing}mm isotropic")
    print(f"Margin: {args.margin_mm}mm, CT window: [{args.ct_window_center - args.ct_window_width/2}, "
          f"{args.ct_window_center + args.ct_window_width/2}] HU")
    print(f"Output: {output_dir}")

    resampled_sizes = []
    skipped_empty = []
    processed = 0
    skipped = 0

    for img_path, lbl_path in tqdm(pairs, desc="Preprocessing"):
        name = os.path.basename(img_path).replace(".nii.gz", "")
        pt_path = output_dir / f"{name}.pt"

        if pt_path.exists() and not args.force:
            skipped += 1
            continue

        # 1. Load NIfTI (raw HU)
        image, spacing = load_nifti(img_path)
        label, _ = load_nifti(lbl_path)

        # Skip cases with no vessel annotation
        if (label == args.vessel_label).sum() == 0:
            skipped_empty.append(name)
            print(f"  WARNING: {name} has no vessel label, skipping")
            continue

        # 2. Crop to vessel ROI + margin (raw HU)
        image_crop, label_crop = crop_to_vessel_roi(
            image, label, args.vessel_label, args.margin_mm, spacing,
            target_spacing=args.target_spacing, min_size=max(target_size)
        )

        # 3. Resample to isotropic spacing (raw HU)
        #    image: order=1 (linear) — DiffTumor STEP1/2 convention
        #    label: order=0 (nearest) — preserve binary values
        image_resampled = resample_volume(
            image_crop, spacing, args.target_spacing, order=1
        )
        label_resampled = resample_volume(
            label_crop, spacing, args.target_spacing, order=0
        )
        resampled_sizes.append(image_resampled.shape)

        # 4. Center crop to target_size (raw HU)
        image_final = center_crop(image_resampled, target_size)
        label_final = center_crop(label_resampled, target_size)

        # 5. CT windowing → [-1, 1]
        image_final = apply_ct_window(
            image_final, args.ct_window_center, args.ct_window_width
        )

        # 6. Binarize mask
        mask_final = (label_final == args.vessel_label).astype(np.float32)

        # 7. Save
        torch.save({
            "image": torch.from_numpy(image_final).unsqueeze(0),   # (1,D,H,W)
            "mask": torch.from_numpy(mask_final).unsqueeze(0),
        }, pt_path)

        processed += 1

    # Save metadata for reproducibility
    metadata = {
        "target_size": list(target_size),
        "target_spacing": args.target_spacing,
        "margin_mm": args.margin_mm,
        "vessel_label": args.vessel_label,
        "ct_window_center": args.ct_window_center,
        "ct_window_width": args.ct_window_width,
        "ct_window_range": [
            args.ct_window_center - args.ct_window_width / 2,
            args.ct_window_center + args.ct_window_width / 2,
        ],
        "total_pairs": len(pairs),
        "processed": processed,
        "skipped_cached": skipped,
        "skipped_empty_vessel": skipped_empty,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    print(f"Done: {processed} processed, {skipped} skipped (cached), "
          f"{len(skipped_empty)} skipped (no vessel)")

    if resampled_sizes:
        resampled_sizes = np.array(resampled_sizes)
        print(f"\nResampled sizes at {args.target_spacing}mm before center crop:")
        print(f"  mean:   {resampled_sizes.mean(0).astype(int)}")
        print(f"  min:    {resampled_sizes.min(0)}")
        print(f"  max:    {resampled_sizes.max(0)}")
        print(f"  median: {np.median(resampled_sizes, 0).astype(int)}")


if __name__ == "__main__":
    main()

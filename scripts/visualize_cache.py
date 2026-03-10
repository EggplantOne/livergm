"""Visualize preprocessed .pt cache: axial/coronal/sagittal slices for image + mask."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_sample(pt_path: Path, out_path: Path) -> None:
    data = torch.load(pt_path, map_location="cpu")
    image = data["image"][0].numpy()  # (D, H, W)
    mask = data["mask"][0].numpy()

    mid_d = image.shape[0] // 2
    mid_h = image.shape[1] // 2
    mid_w = image.shape[2] // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Row 0: Image
    axes[0, 0].imshow(image[mid_d], cmap="gray")
    axes[0, 0].set_title(f"Image Axial (d={mid_d})")
    axes[0, 1].imshow(image[:, mid_h, :], cmap="gray")
    axes[0, 1].set_title(f"Image Coronal (h={mid_h})")
    axes[0, 2].imshow(image[:, :, mid_w], cmap="gray")
    axes[0, 2].set_title(f"Image Sagittal (w={mid_w})")

    # Row 1: Mask
    axes[1, 0].imshow(mask[mid_d], cmap="gray")
    axes[1, 0].set_title("Mask Axial")
    axes[1, 1].imshow(mask[:, mid_h, :], cmap="gray")
    axes[1, 1].set_title("Mask Coronal")
    axes[1, 2].imshow(mask[:, :, mid_w], cmap="gray")
    axes[1, 2].set_title("Mask Sagittal")

    # Row 2: Image + Mask overlay
    for col, (sl_img, sl_mask, title) in enumerate([
        (image[mid_d], mask[mid_d], "Overlay Axial"),
        (image[:, mid_h, :], mask[:, mid_h, :], "Overlay Coronal"),
        (image[:, :, mid_w], mask[:, :, mid_w], "Overlay Sagittal"),
    ]):
        axes[2, col].imshow(sl_img, cmap="gray")
        axes[2, col].imshow(sl_mask, cmap="Reds", alpha=0.3 * (sl_mask > 0.5))
        axes[2, col].set_title(title)

    for ax in axes.flatten():
        ax.axis("off")

    name = pt_path.stem
    fig.suptitle(f"{name}  shape={list(image.shape)}  "
                 f"vessel_ratio={mask.mean():.4f}  "
                 f"image_range=[{image.min():.2f}, {image.max():.2f}]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    p = argparse.ArgumentParser("Visualize .pt cache files")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(cache_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {cache_dir}")

    for pt_path in pt_files:
        out_path = output_dir / f"{pt_path.stem}.png"
        visualize_sample(pt_path, out_path)


if __name__ == "__main__":
    main()

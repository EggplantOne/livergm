from __future__ import annotations

"""
Preprocess Task08 hepatic vessel NIfTI data into .pt cache files.

Output format (one file per case):
{
    "image": FloatTensor(1, D, H, W),  # CT windowed to [-1, 1]
    "mask":  FloatTensor(1, D, H, W),  # binary vessel mask {0, 1}
    "sdf":   FloatTensor(1, D, H, W),  # optional normalized SDF in [-1, 1]
}

Pipeline:
1. Load CT and label NIfTI.
2. Crop to vessel ROI (label == vessel_label) with physical margin.
3. Resample to isotropic spacing (image: linear / label: nearest).
4. CT windowing and vessel binarization.
5. Center pad/crop to fixed volume shape.
6. Optionally compute SDF and save .pt cache.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, zoom


CT_WINDOW_CENTER = 70.0
CT_WINDOW_WIDTH = 200.0
DEFAULT_VOLUME_SHAPE = (128, 128, 128)  # D, H, W
DEFAULT_TARGET_SPACING = 2.0  # mm isotropic
DEFAULT_SDF_TRUNCATE = 10.0  # voxels


def apply_ct_window(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)  # [0, 1]
    volume = volume * 2.0 - 1.0  # [-1, 1]
    return volume.astype(np.float32)


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    spacing = np.abs(np.diag(img.affine)[:3]).astype(np.float32)
    return data, spacing


def crop_to_vessel_roi(
    image: np.ndarray,
    label: np.ndarray,
    vessel_label: int,
    margin_mm: float,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vessel = label == vessel_label
    if vessel.sum() == 0:
        return image, label

    coords = np.argwhere(vessel)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    margin_vox = np.ceil(margin_mm / spacing).astype(int)

    crop_min = np.maximum(bbox_min - margin_vox, 0)
    crop_max = np.minimum(bbox_max + margin_vox + 1, np.array(image.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(crop_min, crop_max))
    return image[slices], label[slices]


def resample_volume(volume: np.ndarray, src_spacing: np.ndarray, target_spacing: float, order: int) -> np.ndarray:
    zoom_factors = src_spacing / float(target_spacing)
    return zoom(volume, zoom_factors, order=order).astype(np.float32)


def pad_or_crop_to_shape(volume: np.ndarray, target_shape: Tuple[int, int, int], pad_value: float) -> np.ndarray:
    result = np.full(target_shape, pad_value, dtype=np.float32)
    src_shape = volume.shape

    src_slices = []
    dst_slices = []
    for src_dim, tgt_dim in zip(src_shape, target_shape):
        if src_dim <= tgt_dim:
            offset = (tgt_dim - src_dim) // 2
            src_slices.append(slice(0, src_dim))
            dst_slices.append(slice(offset, offset + src_dim))
        else:
            offset = (src_dim - tgt_dim) // 2
            src_slices.append(slice(offset, offset + tgt_dim))
            dst_slices.append(slice(0, tgt_dim))

    result[tuple(dst_slices)] = volume[tuple(src_slices)]
    return result


def mask_to_sdf(mask: np.ndarray, truncate: float) -> np.ndarray:
    mask_bool = mask.astype(bool)
    dist_inside = distance_transform_edt(mask_bool)
    dist_outside = distance_transform_edt(~mask_bool)
    sdf = dist_inside - dist_outside  # positive inside vessel
    sdf = np.clip(sdf, -truncate, truncate) / truncate
    return sdf.astype(np.float32)


def _case_name(image_path: Path) -> str:
    name = image_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return image_path.stem


def preprocess_and_cache(
    data_root: Path,
    split_json: Path,
    cache_dir: Path,
    volume_shape: Tuple[int, int, int],
    target_spacing: float,
    margin_mm: float,
    vessel_label: int,
    ct_window_center: float,
    ct_window_width: float,
    sdf_truncate: float,
    save_sdf: bool,
    force_preprocess: bool,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    with split_json.open("r", encoding="utf-8-sig") as f:
        info = json.load(f)

    pairs: list[tuple[Path, Path]] = []
    for entry in info["training"]:
        image_rel = entry["image"].lstrip("./")
        label_rel = entry["label"].lstrip("./")
        image_path = data_root / image_rel
        label_path = data_root / label_rel
        if image_path.exists() and label_path.exists():
            pairs.append((image_path, label_path))

    print(f"Preprocessing {len(pairs)} volumes -> {cache_dir}")
    print(
        "  shape="
        f"{volume_shape}, target_spacing={target_spacing}mm, margin={margin_mm}mm, "
        f"vessel_label={vessel_label}, sdf_truncate={sdf_truncate}, save_sdf={save_sdf}"
    )

    processed = 0
    cached = 0
    missing_vessel = 0

    for idx, (image_path, label_path) in enumerate(pairs, start=1):
        pt_path = cache_dir / f"{_case_name(image_path)}.pt"
        if pt_path.exists() and not force_preprocess:
            cached += 1
            continue

        image, spacing = load_nifti(image_path)
        label, _ = load_nifti(label_path)

        image, label = crop_to_vessel_roi(
            image=image,
            label=label,
            vessel_label=vessel_label,
            margin_mm=margin_mm,
            spacing=spacing,
        )
        image = resample_volume(image, spacing, target_spacing, order=1)
        label = resample_volume(label, spacing, target_spacing, order=0)

        image = apply_ct_window(image, center=ct_window_center, width=ct_window_width)
        mask = (label == vessel_label).astype(np.float32)
        if mask.sum() == 0:
            missing_vessel += 1

        image = pad_or_crop_to_shape(image, target_shape=volume_shape, pad_value=-1.0)
        mask = pad_or_crop_to_shape(mask, target_shape=volume_shape, pad_value=0.0)
        case_dict: dict[str, torch.Tensor] = {
            "image": torch.from_numpy(image).unsqueeze(0),  # (1, D, H, W)
            "mask": torch.from_numpy(mask).unsqueeze(0),
        }
        if save_sdf:
            sdf = mask_to_sdf(mask, truncate=sdf_truncate)
            case_dict["sdf"] = torch.from_numpy(sdf).unsqueeze(0)

        torch.save(case_dict, pt_path)

        processed += 1
        if idx % 50 == 0 or idx == len(pairs):
            print(f"  [{idx}/{len(pairs)}] processed={processed}, cached={cached}")

    print(
        f"Done: processed={processed}, cached={cached}, "
        f"missing_vessel_after_processing={missing_vessel}"
    )
    return {
        "num_pairs": len(pairs),
        "processed": processed,
        "cached": cached,
        "missing_vessel_after_processing": missing_vessel,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preprocess hepatic vessel NIfTI volumes into .pt cache")
    parser.add_argument("--data_root", type=str, required=True, help="Task08 data root containing imagesTr/labelsTr")
    parser.add_argument(
        "--split_json",
        type=str,
        default="",
        help="Path to dataset.json (default: <data_root>/dataset.json)",
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="Output cache directory for .pt files")
    parser.add_argument("--volume_shape", type=int, nargs=3, default=list(DEFAULT_VOLUME_SHAPE))
    parser.add_argument("--target_spacing", type=float, default=DEFAULT_TARGET_SPACING)
    parser.add_argument("--margin_mm", type=float, default=16.0)
    parser.add_argument("--vessel_label", type=int, default=1)
    parser.add_argument("--ct_window_center", type=float, default=CT_WINDOW_CENTER)
    parser.add_argument("--ct_window_width", type=float, default=CT_WINDOW_WIDTH)
    parser.add_argument("--sdf_truncate", type=float, default=DEFAULT_SDF_TRUNCATE)
    parser.add_argument("--save_sdf", action="store_true", help="Save SDF field in cache (default: off)")
    parser.add_argument("--force_preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_json = Path(args.split_json) if args.split_json else data_root / "dataset.json"
    cache_dir = Path(args.cache_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not split_json.exists():
        raise FileNotFoundError(f"split_json does not exist: {split_json}")

    preprocess_and_cache(
        data_root=data_root,
        split_json=split_json,
        cache_dir=cache_dir,
        volume_shape=tuple(int(v) for v in args.volume_shape),
        target_spacing=float(args.target_spacing),
        margin_mm=float(args.margin_mm),
        vessel_label=int(args.vessel_label),
        ct_window_center=float(args.ct_window_center),
        ct_window_width=float(args.ct_window_width),
        sdf_truncate=float(args.sdf_truncate),
        save_sdf=bool(args.save_sdf),
        force_preprocess=bool(args.force_preprocess),
    )


if __name__ == "__main__":
    main()

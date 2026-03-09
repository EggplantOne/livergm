from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional runtime dependency
    nib = None

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - optional runtime dependency
    ndimage = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def split_pt_files_train_val_test(
    cache_dir: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    cache_dir = Path(cache_dir)
    pt_files = sorted(cache_dir.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt cache files found in {cache_dir}")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(f"val_ratio/test_ratio must be >= 0, got val_ratio={val_ratio}, test_ratio={test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0, got val_ratio={val_ratio}, test_ratio={test_ratio}"
        )

    rng = random.Random(seed)
    indices = list(range(len(pt_files)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    if test_ratio > 0 and n_test == 0 and n_total >= 3:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and n_total - n_test >= 2:
        n_val = 1
    while n_test + n_val >= n_total:
        if n_val > 0:
            n_val -= 1
            continue
        if n_test > 0:
            n_test -= 1
            continue
        break

    test_indices = set(indices[:n_test])
    val_indices = set(indices[n_test : n_test + n_val])
    train_indices = set(indices[n_test + n_val :])

    train_files = [pt_files[i] for i in indices if i in train_indices]
    val_files = [pt_files[i] for i in indices if i in val_indices]
    test_files = [pt_files[i] for i in indices if i in test_indices]
    return train_files, val_files, test_files


def split_pt_files(cache_dir: str | Path, val_ratio: float = 0.1, seed: int = 42) -> tuple[list[Path], list[Path]]:
    train_files, val_files, _ = split_pt_files_train_val_test(
        cache_dir=cache_dir,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    return train_files, val_files


def ensure_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape (C, H, W, D) or (H, W, D), got {tuple(mask.shape)}")
    return mask


def ensure_binary_mask(mask: torch.Tensor, source: str | Path | None = None) -> torch.Tensor:
    invalid = (mask != 0.0) & (mask != 1.0)
    if torch.any(invalid):
        bad_values = torch.unique(mask[invalid]).detach().cpu()
        preview = bad_values[:8].tolist()
        suffix = f" in {source}" if source is not None else ""
        raise ValueError(
            f"Expected binary mask values {{0, 1}}{suffix}, "
            f"but found non-binary values (examples={preview}, total_invalid_unique={bad_values.numel()})"
        )
    return mask


def center_crop_or_pad(mask: torch.Tensor, spatial_size: tuple[int, int, int]) -> torch.Tensor:
    spatial_size = tuple(int(v) for v in spatial_size)
    mask = ensure_mask_tensor(mask)
    current = mask.shape[1:]

    slices = [slice(None)]
    for cur_dim, target_dim in zip(current, spatial_size):
        if cur_dim <= target_dim:
            slices.append(slice(0, cur_dim))
            continue
        start = (cur_dim - target_dim) // 2
        slices.append(slice(start, start + target_dim))
    mask = mask[tuple(slices)]

    current = mask.shape[1:]
    pad = []
    for cur_dim, target_dim in reversed(list(zip(current, spatial_size))):
        deficit = max(0, target_dim - cur_dim)
        pad.extend([deficit // 2, deficit - deficit // 2])
    if any(pad):
        mask = F.pad(mask, pad, mode="constant", value=0.0)
    return mask


class VesselMaskPtDataset(Dataset):
    def __init__(self, pt_files: list[Path], augment: bool = False, spatial_size: tuple[int, int, int] | None = None):
        self.pt_files = [Path(p) for p in pt_files]
        self.augment = augment
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.pt_files[index]
        data = torch.load(path, map_location="cpu")
        if "mask" not in data:
            raise KeyError(f"Cache file {path} does not contain 'mask'")

        mask = ensure_mask_tensor(data["mask"])
        if self.spatial_size is not None and tuple(mask.shape[1:]) != self.spatial_size:
            mask = center_crop_or_pad(mask, self.spatial_size)
        mask = ensure_binary_mask(mask, source=path)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                mask = torch.flip(mask, dims=(1,))
            if torch.rand(1).item() > 0.5:
                mask = torch.flip(mask, dims=(2,))
            if torch.rand(1).item() > 0.5:
                mask = torch.flip(mask, dims=(3,))

        return {"mask": mask.contiguous(), "path": str(path)}


def kl_loss(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    dims = list(range(1, z_mu.ndim))
    loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-8) - 1,
        dim=dims,
    )
    return torch.mean(loss)


def _spatial_dims(x: torch.Tensor) -> list[int]:
    return list(range(1, x.ndim))


def dice_score(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = _spatial_dims(pred_bin)
    inter = torch.sum(pred_bin * gt_bin, dim=dims)
    denom = torch.sum(pred_bin, dim=dims) + torch.sum(gt_bin, dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def iou_score(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = _spatial_dims(pred_bin)
    inter = torch.sum(pred_bin * gt_bin, dim=dims)
    union = torch.sum((pred_bin + gt_bin) > 0, dim=dims)
    return (inter + eps) / (union + eps)


def volume_relative_error(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = _spatial_dims(pred_bin)
    pred_vol = torch.sum(pred_bin, dim=dims)
    gt_vol = torch.sum(gt_bin, dim=dims)
    return torch.abs(pred_vol - gt_vol) / (gt_vol + eps)


def vessel_ratio(mask_bin: torch.Tensor) -> torch.Tensor:
    dims = _spatial_dims(mask_bin)
    return torch.mean(mask_bin.float(), dim=dims)


def extract_state_dict(loaded_obj: dict | torch.Tensor) -> dict:
    if isinstance(loaded_obj, dict):
        if "model" in loaded_obj and isinstance(loaded_obj["model"], dict):
            return loaded_obj["model"]
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
    return loaded_obj


def load_pretrained_with_shape_filter(model: torch.nn.Module, state_dict: dict) -> tuple[dict, list[tuple[str, str]], list, list]:
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape_mismatch ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}"))
            continue
        filtered[key] = value
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    return filtered, skipped, missing_keys, unexpected_keys


def infer_case_name(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    return name[:-3] if name.endswith(".pt") else path.stem


def _normalize01(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - low) / (high - low)


def _pick_slices(mask_np: np.ndarray) -> tuple[int, int, int]:
    if float(mask_np.sum()) <= 0:
        return mask_np.shape[2] // 2, mask_np.shape[1] // 2, mask_np.shape[0] // 2
    z = int(np.argmax(mask_np.sum(axis=(0, 1))))
    y = int(np.argmax(mask_np.sum(axis=(0, 2))))
    x = int(np.argmax(mask_np.sum(axis=(1, 2))))
    return z, y, x


def save_recon_comparison(gt: torch.Tensor, recon: torch.Tensor, out_path: str | Path, threshold: float = 0.5) -> None:
    gt_np = gt[0, 0].detach().cpu().numpy()
    recon_np = recon[0, 0].detach().cpu().numpy()
    recon_bin = (recon_np > float(threshold)).astype(np.float32)
    recon_vis = _normalize01(recon_np)
    z, y, x = _pick_slices(gt_np)

    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    rows = [
        (gt_np, "GT"),
        (recon_vis, "Recon"),
        (recon_bin, "ReconBin"),
    ]
    titles = ("Axial", "Coronal", "Sagittal")
    for row_idx, (array, prefix) in enumerate(rows):
        axes[row_idx, 0].imshow(array[:, :, z], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].imshow(array[:, y, :], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].imshow(array[x, :, :], cmap="gray", vmin=0, vmax=1)
        for col_idx, title in enumerate(titles):
            axes[row_idx, col_idx].set_title(f"{prefix} {title}")

    axes[3, 0].imshow(np.max(gt_np, axis=2), cmap="gray", vmin=0, vmax=1)
    axes[3, 1].imshow(np.max(recon_vis, axis=2), cmap="gray", vmin=0, vmax=1)
    axes[3, 2].imshow(np.max(recon_bin, axis=2), cmap="gray", vmin=0, vmax=1)
    axes[3, 0].set_title("GT MIP(z)")
    axes[3, 1].set_title("Recon MIP(z)")
    axes[3, 2].set_title("ReconBin MIP(z)")

    for axis in axes.flatten():
        axis.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_generated_overview(mask: torch.Tensor, out_path: str | Path, threshold: float = 0.5, title: str = "Sample") -> None:
    mask_np = mask[0, 0].detach().cpu().numpy()
    mask_bin = (mask_np > float(threshold)).astype(np.float32)
    mask_vis = _normalize01(mask_np)
    z, y, x = _pick_slices(mask_bin)

    fig, axes = plt.subplots(4, 3, figsize=(12, 13))
    views = [
        (mask_vis[:, :, z], mask_bin[:, :, z], "Axial"),
        (mask_vis[:, y, :], mask_bin[:, y, :], "Coronal"),
        (mask_vis[x, :, :], mask_bin[x, :, :], "Sagittal"),
    ]
    for col_idx, (raw_view, bin_view, view_name) in enumerate(views):
        axes[0, col_idx].imshow(raw_view, cmap="gray", vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"{title} {view_name}")
        axes[1, col_idx].imshow(bin_view, cmap="gray", vmin=0, vmax=1)
        axes[1, col_idx].set_title(f"{title}Bin {view_name}")

    mip_raw_views = [
        (np.max(mask_vis, axis=2), "MIP(z)"),
        (np.max(mask_vis, axis=1), "MIP(y)"),
        (np.max(mask_vis, axis=0), "MIP(x)"),
    ]
    mip_bin_views = [
        (np.max(mask_bin, axis=2), "Bin MIP(z)"),
        (np.max(mask_bin, axis=1), "Bin MIP(y)"),
        (np.max(mask_bin, axis=0), "Bin MIP(x)"),
    ]
    for col_idx, (mip_view, name) in enumerate(mip_raw_views):
        axes[2, col_idx].imshow(mip_view, cmap="gray", vmin=0, vmax=1)
        axes[2, col_idx].set_title(f"{title} {name}")
    for col_idx, (mip_view, name) in enumerate(mip_bin_views):
        axes[3, col_idx].imshow(mip_view, cmap="gray", vmin=0, vmax=1)
        axes[3, col_idx].set_title(f"{title} {name}")

    for axis in axes.flatten():
        if axis.has_data():
            axis.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_nifti_volume(volume: torch.Tensor | np.ndarray, out_path: str | Path) -> None:
    if nib is None:
        raise ImportError("nibabel is required to save NIfTI files")
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    if volume.ndim == 5:
        volume = volume[0, 0]
    elif volume.ndim == 4:
        volume = volume[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4, dtype=np.float32)), str(out_path))


class VesselImagePtDataset(Dataset):
    """Dataset that loads CT image volumes from .pt cache files."""

    def __init__(self, pt_files: list[Path], augment: bool = False, spatial_size: tuple[int, int, int] | None = None):
        self.pt_files = [Path(p) for p in pt_files]
        self.augment = augment
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.pt_files[index]
        data = torch.load(path, map_location="cpu")
        if "image" not in data:
            raise KeyError(f"Cache file {path} does not contain 'image'")

        image = data["image"].float()
        if image.ndim == 3:
            image = image.unsqueeze(0)

        if self.spatial_size is not None and tuple(image.shape[1:]) != self.spatial_size:
            image = center_crop_or_pad(image, self.spatial_size)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=(1,))
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=(2,))
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=(3,))

        return {"image": image.contiguous(), "path": str(path)}


def save_ct_recon_comparison(
    gt: torch.Tensor, recon: torch.Tensor, out_path: str | Path,
) -> None:
    """Save GT vs Recon comparison for CT images.

    Uses matplotlib auto-scaling (no manual vmin/vmax), following MONAI tutorial style.
    """
    gt_np = gt[0, 0].detach().cpu().numpy()
    recon_np = recon[0, 0].detach().cpu().numpy()
    mid_d = gt_np.shape[0] // 2
    mid_h = gt_np.shape[1] // 2
    mid_w = gt_np.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    # Row 0: GT
    axes[0, 0].imshow(gt_np[mid_d, ...], cmap="gray")
    axes[0, 1].imshow(gt_np[:, mid_h, :], cmap="gray")
    axes[0, 2].imshow(gt_np[..., mid_w], cmap="gray")
    # Row 1: Recon
    axes[1, 0].imshow(recon_np[mid_d, ...], cmap="gray")
    axes[1, 1].imshow(recon_np[:, mid_h, :], cmap="gray")
    axes[1, 2].imshow(recon_np[..., mid_w], cmap="gray")

    axes[0, 0].set_ylabel("GT")
    axes[1, 0].set_ylabel("Recon")
    for col, title in enumerate(("Axial", "Coronal", "Sagittal")):
        axes[0, col].set_title(title)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class VesselImageMaskPtDataset(Dataset):
    """Dataset that loads both CT image and vessel mask from .pt cache files.

    Used for ControlNet training where we need (image, mask) pairs.
    """

    def __init__(self, pt_files: list[Path], augment: bool = False, spatial_size: tuple[int, int, int] | None = None):
        self.pt_files = [Path(p) for p in pt_files]
        self.augment = augment
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.pt_files[index]
        data = torch.load(path, map_location="cpu")
        if "image" not in data:
            raise KeyError(f"Cache file {path} does not contain 'image'")
        if "mask" not in data:
            raise KeyError(f"Cache file {path} does not contain 'mask'")

        image = data["image"].float()
        mask = data["mask"].float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        if self.spatial_size is not None:
            if tuple(image.shape[1:]) != self.spatial_size:
                image = center_crop_or_pad(image, self.spatial_size)
            if tuple(mask.shape[1:]) != self.spatial_size:
                mask = center_crop_or_pad(mask, self.spatial_size)

        if self.augment:
            for dim in (1, 2, 3):
                if torch.rand(1).item() > 0.5:
                    image = torch.flip(image, dims=(dim,))
                    mask = torch.flip(mask, dims=(dim,))

        return {"image": image.contiguous(), "mask": mask.contiguous(), "path": str(path)}


def save_ct_overview(
    volume: torch.Tensor, out_path: str | Path, title: str = "Generated CT",
) -> None:
    """Save axial/coronal/sagittal mid-slices of a CT volume."""
    vol_np = volume[0, 0].detach().cpu().numpy()
    mid_d = vol_np.shape[0] // 2
    mid_h = vol_np.shape[1] // 2
    mid_w = vol_np.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vol_np[mid_d, ...], cmap="gray")
    axes[0].set_title(f"{title} Axial")
    axes[1].imshow(vol_np[:, mid_h, :], cmap="gray")
    axes[1].set_title(f"{title} Coronal")
    axes[2].imshow(vol_np[..., mid_w], cmap="gray")
    axes[2].set_title(f"{title} Sagittal")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_controlnet_comparison(
    mask: torch.Tensor, generated: torch.Tensor, out_path: str | Path,
    gt_image: torch.Tensor | None = None,
) -> None:
    """Save mask condition + generated CT (+ optional GT) comparison."""
    mask_np = mask[0, 0].detach().cpu().numpy()
    gen_np = generated[0, 0].detach().cpu().numpy()
    mid_d = gen_np.shape[0] // 2
    mid_h = gen_np.shape[1] // 2
    mid_w = gen_np.shape[2] // 2

    n_rows = 3 if gt_image is not None else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    # Row 0: Mask condition
    axes[0, 0].imshow(mask_np[mid_d, ...], cmap="gray")
    axes[0, 1].imshow(mask_np[:, mid_h, :], cmap="gray")
    axes[0, 2].imshow(mask_np[..., mid_w], cmap="gray")
    axes[0, 0].set_ylabel("Mask")
    # Row 1: Generated
    axes[1, 0].imshow(gen_np[mid_d, ...], cmap="gray")
    axes[1, 1].imshow(gen_np[:, mid_h, :], cmap="gray")
    axes[1, 2].imshow(gen_np[..., mid_w], cmap="gray")
    axes[1, 0].set_ylabel("Generated")
    # Row 2: GT (optional)
    if gt_image is not None:
        gt_np = gt_image[0, 0].detach().cpu().numpy()
        axes[2, 0].imshow(gt_np[mid_d, ...], cmap="gray")
        axes[2, 1].imshow(gt_np[:, mid_h, :], cmap="gray")
        axes[2, 2].imshow(gt_np[..., mid_w], cmap="gray")
        axes[2, 0].set_ylabel("GT")

    for col, title in enumerate(("Axial", "Coronal", "Sagittal")):
        axes[0, col].set_title(title)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def connected_component_stats(mask_bin: torch.Tensor | np.ndarray) -> dict[str, float | int | None]:
    if isinstance(mask_bin, torch.Tensor):
        mask_bin = mask_bin.detach().cpu().numpy()
    if mask_bin.ndim == 5:
        mask_bin = mask_bin[0, 0]
    elif mask_bin.ndim == 4:
        mask_bin = mask_bin[0]

    foreground = mask_bin.astype(np.uint8)
    foreground_voxels = int(foreground.sum())
    if foreground_voxels == 0:
        return {
            "connected_components": 0,
            "largest_component_voxels": 0,
            "largest_component_ratio": 0.0,
            "component_stats_available": ndimage is not None,
        }

    if ndimage is None:
        return {
            "connected_components": None,
            "largest_component_voxels": None,
            "largest_component_ratio": None,
            "component_stats_available": False,
        }

    labeled, num_components = ndimage.label(foreground)
    counts = np.bincount(labeled.reshape(-1))[1:]
    largest = int(counts.max()) if counts.size > 0 else 0
    return {
        "connected_components": int(num_components),
        "largest_component_voxels": largest,
        "largest_component_ratio": float(largest / max(1, foreground_voxels)),
        "component_stats_available": True,
    }

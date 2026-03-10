"""Analyze HU distribution in vessel ROI of Task08."""
import nibabel as nib
import numpy as np
import json, os

data_root = '/mnt/no1/yinhaojie/Task08_HepaticVessel'
with open(os.path.join(data_root, 'dataset.json'), 'r') as f:
    info = json.load(f)

pairs = []
for entry in info['training']:
    img_p = os.path.join(data_root, entry['image'].lstrip('./'))
    lbl_p = os.path.join(data_root, entry['label'].lstrip('./'))
    if os.path.exists(img_p) and os.path.exists(lbl_p):
        pairs.append((img_p, lbl_p))

np.random.seed(42)
sample_idx = np.random.choice(len(pairs), 20, replace=False)

vessel_hus = []
liver_hus = []
all_roi_hus = []

for idx in sample_idx:
    img_p, lbl_p = pairs[idx]
    img = nib.load(img_p).get_fdata(dtype=np.float32)
    lbl = nib.load(lbl_p).get_fdata(dtype=np.float32)
    spacing = np.abs(np.diag(nib.load(img_p).affine)[:3])

    vessel = (lbl == 1)
    if vessel.sum() == 0:
        continue
    coords = np.argwhere(vessel)
    margin_vox = np.ceil(16.0 / spacing).astype(int)
    crop_min = np.maximum(coords.min(0) - margin_vox, 0)
    crop_max = np.minimum(coords.max(0) + margin_vox + 1, np.array(img.shape))
    sl = tuple(slice(lo, hi) for lo, hi in zip(crop_min, crop_max))

    img_roi = img[sl]
    lbl_roi = lbl[sl]

    vessel_mask = (lbl_roi == 1)
    vessel_hus.extend(img_roi[vessel_mask].tolist())

    fg_mask = (img_roi > -200) & (~vessel_mask)
    liver_hus.extend(img_roi[fg_mask].tolist())

    all_roi_hus.extend(img_roi[img_roi > -200].tolist())

vessel_hus = np.array(vessel_hus)
liver_hus = np.array(liver_hus)
all_roi_hus = np.array(all_roi_hus)

print("=== Vessel voxels HU ===")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  P{p}: {np.percentile(vessel_hus, p):.0f}")
print(f"  mean: {vessel_hus.mean():.0f}, std: {vessel_hus.std():.0f}")
print(f"  min: {vessel_hus.min():.0f}, max: {vessel_hus.max():.0f}")

print("\n=== Non-vessel tissue HU ===")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  P{p}: {np.percentile(liver_hus, p):.0f}")
print(f"  mean: {liver_hus.mean():.0f}, std: {liver_hus.std():.0f}")

print("\n=== All foreground in ROI ===")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  P{p}: {np.percentile(all_roi_hus, p):.0f}")

for lo, hi, name in [(-30, 170, "current [-30,170]"),
                      (-175, 250, "DiffTumor [-175,250]"),
                      (-3, 243, "DiffTumor Task08 [-3,243]"),
                      (-100, 300, "wider [-100,300]")]:
    clipped_below = (all_roi_hus < lo).sum() / len(all_roi_hus) * 100
    clipped_above = (all_roi_hus > hi).sum() / len(all_roi_hus) * 100
    vessel_above = (vessel_hus > hi).sum() / len(vessel_hus) * 100
    print(f"\nWindow {name}:")
    print(f"  ROI clipped below: {clipped_below:.1f}%, above: {clipped_above:.1f}%")
    print(f"  Vessel clipped above: {vessel_above:.1f}%")

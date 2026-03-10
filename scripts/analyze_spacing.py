"""Analyze vessel ROI sizes at different target spacings."""
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
sample_idx = np.random.choice(len(pairs), min(30, len(pairs)), replace=False)

margin_mm = 16.0
results_1mm = []
results_2mm = []
orig_spacings = []

for idx in sample_idx:
    img_p, lbl_p = pairs[idx]
    lbl = nib.load(lbl_p)
    lbl_data = lbl.get_fdata(dtype=np.float32)
    spacing = np.abs(np.diag(lbl.affine)[:3])
    orig_spacings.append(spacing)

    vessel = (lbl_data == 1)
    if vessel.sum() == 0:
        continue
    coords = np.argwhere(vessel)
    bbox_min = coords.min(0)
    bbox_max = coords.max(0)

    bbox_size_mm = (bbox_max - bbox_min + 1) * spacing + 2 * margin_mm

    size_1mm = np.ceil(bbox_size_mm / 1.0).astype(int)
    size_2mm = np.ceil(bbox_size_mm / 2.0).astype(int)
    results_1mm.append(size_1mm)
    results_2mm.append(size_2mm)

results_1mm = np.array(results_1mm)
results_2mm = np.array(results_2mm)
orig_spacings = np.array(orig_spacings)

print("=== Original spacings (axis0, axis1, axis2) mm ===")
print("  mean:", np.round(orig_spacings.mean(0), 2))
print("  min: ", np.round(orig_spacings.min(0), 2))
print("  max: ", np.round(orig_spacings.max(0), 2))
print()

print("=== Vessel ROI + 16mm margin at 1mm isotropic ===")
print("  mean:", results_1mm.mean(0).astype(int))
print("  min: ", results_1mm.min(0))
print("  max: ", results_1mm.max(0))
print("  median:", np.median(results_1mm, 0).astype(int))
print("  >128 in any dim:", (results_1mm.max(1) > 128).sum(), "/", len(results_1mm))
print("  all dims <=128:", (results_1mm.max(1) <= 128).sum(), "/", len(results_1mm))
print()

print("=== At 2mm isotropic ===")
print("  mean:", results_2mm.mean(0).astype(int))
print("  min: ", results_2mm.min(0))
print("  max: ", results_2mm.max(0))
print("  median:", np.median(results_2mm, 0).astype(int))
print()

# Fill ratio estimates
fill_2mm = np.prod(np.minimum(results_2mm, 128), axis=1) / 128.0**3
fill_1mm = np.prod(np.minimum(results_1mm, 128), axis=1) / 128.0**3
print("=== Fill ratio in 128^3 ===")
print(f"  At 2mm: mean={fill_2mm.mean():.3f}, min={fill_2mm.min():.3f}")
print(f"  At 1mm: mean={fill_1mm.mean():.3f}, min={fill_1mm.min():.3f}")
print()

# What if we increase margin?
for margin in [16, 24, 32, 48]:
    sizes = []
    for idx in sample_idx:
        _, lbl_p = pairs[idx]
        lbl = nib.load(lbl_p)
        lbl_data = lbl.get_fdata(dtype=np.float32)
        spacing = np.abs(np.diag(lbl.affine)[:3])
        vessel = (lbl_data == 1)
        if vessel.sum() == 0:
            continue
        coords = np.argwhere(vessel)
        bbox_mm = (coords.max(0) - coords.min(0) + 1) * spacing + 2 * margin
        sizes.append(np.ceil(bbox_mm / 1.0).astype(int))
    sizes = np.array(sizes)
    fill = np.prod(np.minimum(sizes, 128), axis=1) / 128.0**3
    over = (sizes.max(1) > 128).sum()
    print(f"  margin={margin}mm, 1mm iso: mean_fill={fill.mean():.3f}, >128_any={over}/{len(sizes)}, mean_size={sizes.mean(0).astype(int)}")

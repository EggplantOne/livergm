# Phase 1 - 3D AutoencoderKL (Vessel Mask)

Train a 3D `AutoencoderKL` for vessel mask compression (`.nrrd/.nhdr/.nii/.nii.gz` input).

## Script

- `train_vae_3d_mask.py`

## Quick Start

```powershell
cd F:\code\fuwuqi\GenerativeModels
python .\phase1-codex\train_vae_3d_mask.py `
  --data_dir "F:\your_data\vessel_masks_nrrd" `
  --output_dir ".\phase1-codex\outputs_vae_64" `
  --spatial_size 64 64 64 `
  --spatial_mode crop `
  --batch_size 2 `
  --num_epochs 200 `
  --val_interval 5 `
  --amp `
  --binarize
```

Use downloaded pretrained weights for fine-tuning:

```powershell
python .\phase1-codex\train_vae_3d_mask.py `
  --data_dir "F:\your_data\vessel_masks" `
  --output_dir ".\phase1-codex\outputs_vae_ft" `
  --pretrained_model "F:\your_ckpt\autoencoder_pretrained.pt" `
  --spatial_size 64 64 64 `
  --batch_size 2 `
  --amp `
  --binarize
```

Server example (your paths, NIfTI label maps, vessel label=1):

```bash
cd /home/yinhaojie/GenerativeModels
python ./phase1-codex/train_vae_3d_mask.py \
  --data_dir /home/yinhaojie/GenerativeModels/data/vessel_masks \
  --output_dir ./phase1-codex/outputs_vae_vessel_ft \
  --pretrained_model /home/yinhaojie/GenerativeModels/pretrained_models/autoencoder.pth \
  --num_channels 64 128 128 128 \
  --latent_channels 3 \
  --num_res_blocks 2 \
  --attention_levels 0 0 0 0 \
  --target_label 1 \
  --spatial_size 64 64 64 \
  --spatial_mode crop \
  --batch_size 2 \
  --num_epochs 200 \
  --val_interval 5 \
  --amp
```

If pretrained architecture differs from current settings, the script will load only shape-matching parameters by default (`--pretrained_strict` is off).

Resume interrupted training:

```powershell
python .\phase1-codex\train_vae_3d_mask.py `
  --data_dir "F:\your_data\vessel_masks" `
  --output_dir ".\phase1-codex\outputs_vae_ft" `
  --resume_ckpt ".\phase1-codex\outputs_vae_ft\checkpoints\last.pt"
```

For larger GPUs:

```powershell
python .\phase1-codex\train_vae_3d_mask.py `
  --data_dir "F:\your_data\vessel_masks_nrrd" `
  --output_dir ".\phase1-codex\outputs_vae_128" `
  --spatial_size 128 128 128 `
  --spatial_mode crop `
  --batch_size 1 `
  --amp `
  --binarize
```

## Outputs

- `checkpoints/last.pt`: latest checkpoint
- `checkpoints/best.pt`: best validation checkpoint
- `checkpoints/autoencoderkl_best_weights.pt`: best model weights only
- `checkpoints/autoencoderkl_final_weights.pt`: final epoch model weights only
- `train_config.json`: training config snapshot

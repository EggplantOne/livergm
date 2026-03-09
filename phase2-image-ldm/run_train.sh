#!/bin/bash
# Step 2: Train unconditional 3D LDM in image VAE latent space
# Prerequisites: trained image VAE from phase2-image-vae

CACHE_DIR="/home/yinhaojie/CVI/logs/hepaticvessel_3d/cache"
VAE_CKPT="./phase2-image-vae/outputs_vae_image_128/checkpoints/autoencoderkl_best_weights.pt"

python phase2-image-ldm/train_ldm_3d_image.py \
    --cache_dir "$CACHE_DIR" \
    --vae_checkpoint "$VAE_CKPT" \
    --output_dir ./phase2-image-ldm/outputs_ldm_image \
    --batch_size 1 \
    --num_epochs 300 \
    --val_interval 20 \
    --save_interval 50 \
    --lr 1e-4 \
    --patience 5 \
    --amp \
    --vae_num_channels 64 128 128 128 \
    --vae_num_res_blocks 2 \
    --vae_attention_levels 0 0 0 0 \
    --unet_num_channels 64 128 256 \
    --unet_num_res_blocks 2 \
    --unet_attention_levels 0 1 1 \
    --unet_num_head_channels 0 128 256

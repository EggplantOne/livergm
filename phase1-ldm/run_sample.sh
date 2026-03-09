#!/bin/bash
cd /home/yinhaojie/GenerativeModels
python -u phase1-ldm/sample_ldm_3d_mask.py \
  --vae_checkpoint ./phase1-codex/outputs_vae_vessel_128_scratch/checkpoints/autoencoderkl_best_weights.pt \
  --ldm_checkpoint ./phase1-ldm/outputs_ldm_vessel/checkpoints/best.pt \
  --output_dir ./phase1-ldm/samples_best_epoch20 \
  --num_samples 10 \
  --batch_size 2 \
  --num_inference_steps 100 \
  --scheduler_type ddim \
  --seed 42 \
  --save_nifti

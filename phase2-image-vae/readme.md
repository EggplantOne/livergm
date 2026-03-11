CUDA_VISIBLE_DEVICES=0 bash phase2-image-vae/run_train.sh
基于预训练微调
CUDA_VISIBLE_DEVICES=1 bash phase2-image-vae/run_train_scratch.sh
从头开始

断点续跑：
CUDA_VISIBLE_DEVICES=0 python phase2-image-vae/train_vae_3d_image.py \
    --cache_dir /mnt/no1/yinhaojie/Task08_HepaticVessel/cache_1mm \
    --output_dir ./phase2-image-vae/outputs_vae_image_128_v2 \
    --resume_ckpt ./phase2-image-vae/outputs_vae_image_128_v2/checkpoints/last.pt \
    --spatial_size 128 128 128 \
    --batch_size 1 \
    --num_epochs 200 \
    --val_interval 10 \
    --save_interval 20 \
    --lr 1e-4 \
    --lr_d 5e-4 \
    --l1_weight 1.0 \
    --kl_weight 1e-6 \
    --adv_weight 0.01 \
    --perceptual_weight 0.001 \
    --warmup_epochs 5 \
    --patience 5 \
    --gradient_clip 1.0 \
    --num_workers 4 \
    --seed 42 \
    --amp \
    --augment \
    --num_channels 64 128 128 128 \
    --attention_levels 0 0 0 0 \
    --latent_channels 3 \
    --num_res_blocks 2 \
    --use_checkpointing
CUDA_VISIBLE_DEVICES=0 bash phase2-image-vae/run_train.sh
基于预训练微调
CUDA_VISIBLE_DEVICES=1 bash phase2-image-vae/run_train_scratch.sh
从头开始
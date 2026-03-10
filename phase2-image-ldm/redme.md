对接微调vae
CUDA_VISIBLE_DEVICES=0─bash─phase2-image-ldm/run_train.sh
从头开始训练vae
CUDA_VISIBLE_DEVICES=1 bash phase2-image-ldm/run_train_scratch.sh
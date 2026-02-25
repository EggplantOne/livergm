
```
  python phase1/scripts/visualize_vae.py \
    --model_path /home/yinhaojie/GenerativeModels/pretrained_models/autoencoder.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations \
    --num_samples 10 \
    --latent_channels 3 \
    --num_channels 64 128 128 128 \
    --attention_levels 0 0 0 0 \
    --output_size 64 64 64 \
    --reconstruct_on_cpu
---

### 3. 训练 VAE

```bash
  python phase1/scripts/train_vae_vessel.py \
    --data_dir /home/yinhaojie/GenerativeModels/data/vessel_masks \
    --output_dir /home/yinhaojie/GenerativeModels/outputs/vae_vessel \
    --pretrained_model /home/yinhaojie/GenerativeModels/pretrained_models/autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5
```

### 4. 测试模型

```bash
python phase1/scripts/test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./test_results
```

### 5. 生成合成血管 ⭐

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations \
    --num_samples 10
```

### 6. 监控训练

```bash
tensorboard --logdir ./outputs
```

---


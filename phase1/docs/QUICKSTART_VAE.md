# 快速开始：训练 3D VAE for 血管 Mask

## 🚀 最快路径（使用预训练模型）

### 1. 安装依赖

```bash
pip install gdown nrrd tensorboard
```

### 2. 下载预训练模型

```bash
python download_pretrained_vae.py --model brain_mri --output_dir ./pretrained_models
```

### 3. 准备数据

将你的 NRRD 血管 mask 文件放到一个目录：

```
data/vessel_masks/
├── vessel_001.nrrd
├── vessel_002.nrrd
└── ...
```

### 4. 开始微调

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5
```

### 5. 监控训练

```bash
tensorboard --logdir ./outputs/vae_vessel/logs
```

打开浏览器访问 `http://localhost:6006`

### 6. 使用训练好的模型

训练完成后，最佳模型保存在：
```
./outputs/vae_vessel/best_vae.pth
```

---

## 📊 不同场景的推荐配置

### 场景 1: 数据量少（< 100 样本）

**策略：** 使用预训练模型 + 冻结编码器

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_small_data \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --freeze_encoder \
    --spatial_size 64 64 64 \
    --batch_size 1 \
    --num_epochs 30 \
    --lr 1e-4 \
    --cache_rate 1.0
```

**预期：** 30 epochs 内收敛，训练时间约 1-2 小时（取决于 GPU）

---

### 场景 2: 数据量中等（100-500 样本）

**策略：** 使用预训练模型 + 完整微调

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_medium_data \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5 \
    --kl_weight 1e-6
```

**预期：** 50 epochs 内收敛，训练时间约 3-5 小时

---

### 场景 3: 数据量大（> 500 样本）

**策略：** 从头训练或使用预训练

```bash
# 选项 A: 使用预训练（更快收敛）
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_large_data \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 128 128 128 \
    --batch_size 2 \
    --num_epochs 100 \
    --lr 1e-4

# 选项 B: 从头训练
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_from_scratch \
    --spatial_size 128 128 128 \
    --batch_size 2 \
    --num_epochs 150 \
    --lr 1e-4 \
    --num_channels 64 128 256 \
    --latent_channels 4
```

**预期：** 100-150 epochs 收敛，训练时间约 10-20 小时

---

### 场景 4: 低显存（8GB GPU）

**策略：** 减小 batch size 和 patch size

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_low_memory \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 48 48 48 \
    --batch_size 1 \
    --num_epochs 50 \
    --lr 5e-5 \
    --num_channels 16 32 64 \
    --cache_rate 0.0
```

---

## 🔍 训练监控指标

在 TensorBoard 中关注以下指标：

### 关键指标
- **train/recon_loss**: 重建损失，应该稳定下降到 0.01-0.05
- **train/kl_loss**: KL 散度，通常在 0.001-0.01 范围
- **val/loss**: 验证损失，应该跟随训练损失下降

### 健康的训练曲线
```
Epoch 1:  recon=0.15, kl=0.008, val=0.16
Epoch 10: recon=0.08, kl=0.005, val=0.09
Epoch 30: recon=0.03, kl=0.003, val=0.04
Epoch 50: recon=0.02, kl=0.002, val=0.03
```

### 问题诊断
- **重建损失不下降**: 降低 kl_weight 或增加 perceptual_weight
- **验证损失上升**: 过拟合，减少 epochs 或增加数据增强
- **生成器/判别器损失失衡**: 调整 adv_weight

---

## 🛠️ 常见问题

### Q: 显存不足 (CUDA out of memory)
```bash
# 解决方案 1: 减小 batch size
--batch_size 1

# 解决方案 2: 减小 patch size
--spatial_size 48 48 48

# 解决方案 3: 减少通道数
--num_channels 16 32 64

# 解决方案 4: 关闭数据缓存
--cache_rate 0.0
```

### Q: 训练中断，如何恢复？
```bash
python train_vae_vessel.py \
    --resume_from ./outputs/vae_vessel/checkpoint_epoch_20.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    # ... 其他参数保持一致
```

### Q: 如何验证模型质量？
训练完成后，检查：
1. 验证损失是否收敛
2. TensorBoard 中查看重建样本
3. 使用测试脚本生成样本并可视化

---

## 📝 下一步

训练完成后，你将得到：
- `best_vae.pth` - 最佳 VAE 模型
- `final_vae.pth` - 最终模型
- TensorBoard 日志

这个 VAE 将用于：
1. **Phase 1 Step 2**: 训练无条件扩散模型
2. **Phase 2**: 作为 ControlNet 的编码器

继续阅读 `README_PHASE1.md` 了解如何训练扩散模型。

---

## 💡 最佳实践总结

1. ✅ **优先使用预训练模型** - 节省时间，提高质量
2. ✅ **从小尺寸开始** - 先用 64³ 验证流程
3. ✅ **监控 TensorBoard** - 及时发现问题
4. ✅ **定期保存检查点** - 防止训练中断
5. ✅ **验证数据质量** - 确保 NRRD 文件正确加载
6. ✅ **调整学习率** - 微调时使用更小的学习率
7. ✅ **平衡损失权重** - 根据重建质量调整 kl_weight

---

## 📚 参考资料

- [MONAI Generative Models 文档](https://docs.monai.io/en/latest/apps.html#generative-models)
- [AutoencoderKL 论文](https://arxiv.org/abs/2112.10752)
- [3D LDM 教程](tutorials/generative/3d_ldm/3d_ldm_tutorial.py)

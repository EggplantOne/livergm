# Phase 1: 无条件 3D 血管生成

这个阶段训练一个无条件的 3D Latent Diffusion Model (LDM) 来生成血管 mask。

## 流程概览

```
Phase 1 流程:
1. 训练 3D VAE (AutoencoderKL) - 将血管 mask 压缩到潜空间
2. 训练 3D 扩散模型 (DiffusionModelUNet) - 在潜空间学习生成
3. 推理生成新的血管 mask
```

## Step 1: 训练 3D VAE

### 选项 A: 使用预训练模型微调（推荐）

MONAI 提供了在大规模医学数据上预训练的 3D VAE 模型，可以大幅加速训练并提高质量。

#### 1. 下载预训练模型

```bash
# 安装依赖
pip install gdown

# 下载 3D Brain MRI VAE（推荐用于 3D 医学图像）
python download_pretrained_vae.py --model brain_mri --output_dir ./pretrained_models

# 或下载 2D Chest X-ray VAE
python download_pretrained_vae.py --model chest_xray --output_dir ./pretrained_models

# 查看模型结构
python download_pretrained_vae.py --model brain_mri --inspect
```

**可用的预训练模型：**

| 模型 | 数据集 | 样本数 | 分辨率 | 适用场景 |
|------|--------|--------|--------|----------|
| brain_mri | UK Biobank | 31,740 | 160×224×160 | 3D 医学图像（推荐） |
| chest_xray | MIMIC-CXR | 90,000 | 512×512 | 2D 医学图像 |

#### 2. 微调预训练模型

```bash
# 完整微调（推荐）
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel_finetuned \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5

# 冻结编码器，只训练解码器（更快，适合数据少的情况）
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel_decoder_only \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --freeze_encoder \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 30 \
    --lr 1e-4
```

**微调优势：**
- ✅ 训练速度快 2-3 倍
- ✅ 需要更少的数据（50-100 样本即可）
- ✅ 更好的重建质量
- ✅ 更稳定的训练过程

### 选项 B: 从头训练

如果你的数据与预训练模型差异很大，或者想完全自定义架构。

#### 数据准备

将你的 3D 血管 mask 数据（NRRD 格式）放在一个目录下：

```
data/vessel_masks/
├── vessel_001.nrrd
├── vessel_002.nrrd
├── vessel_003.nrrd
└── ...
```

**数据要求：**
- 格式：NRRD (.nrrd 或 .nrrd.gz)
- 维度：3D 体积数据
- 值范围：建议归一化到 [0, 1]（脚本会自动处理）
- 数量：建议至少 50-100 个样本用于训练

#### 训练命令

**基础训练（64x64x64 patch）：**
```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 100 \
    --num_workers 4
```

**大尺寸训练（128x128x128 patch，需要更多显存）：**
```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel_128 \
    --spatial_size 128 128 128 \
    --batch_size 1 \
    --num_epochs 100 \
    --latent_channels 4 \
    --num_channels 64 128 256 \
    --attention_levels 0 1 1
```

**低显存配置（适用于 8GB GPU）：**
```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel_small \
    --spatial_size 64 64 64 \
    --batch_size 1 \
    --num_channels 16 32 64 \
    --cache_rate 0.0
```

### 通用参数说明

**数据参数：**
- `--data_dir`: 包含 NRRD 文件的目录
- `--cache_rate`: 数据缓存比例 (0-1)，1.0 表示全部缓存到内存

**模型参数：**
- `--spatial_size`: 训练 patch 大小，格式：H W D
- `--latent_channels`: 潜空间通道数（默认 3）
- `--num_channels`: 编码器/解码器的通道数列表
- `--attention_levels`: 每层是否使用注意力机制（0/1）

**训练参数：**
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--lr`: 学习率（默认 1e-4）
- `--kl_weight`: KL 散度损失权重（默认 1e-6）
- `--adv_weight`: 对抗损失权重（默认 0.01）
- `--perceptual_weight`: 感知损失权重（默认 0.001）

**系统参数：**
- `--num_workers`: 数据加载线程数
- `--val_interval`: 验证间隔（epoch）
- `--save_interval`: 保存检查点间隔（epoch）

**预训练和恢复：**
- `--pretrained_model`: 预训练模型路径（用于微调）
- `--freeze_encoder`: 冻结编码器权重（只训练解码器）
- `--resume_from`: 从检查点恢复训练

### 监控训练

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir ./outputs/vae_vessel/logs
```

**关键指标：**
- `train/recon_loss`: 重建损失（越低越好）
- `train/kl_loss`: KL 散度损失
- `train/gen_loss`: 生成器对抗损失
- `train/disc_loss`: 判别器损失
- `val/loss`: 验证集总损失

### 输出文件

训练完成后，输出目录包含：

```
outputs/vae_vessel/
├── best_vae.pth              # 验证集上最佳模型
├── final_vae.pth             # 最终模型
├── checkpoint_epoch_10.pth   # 定期检查点
├── checkpoint_epoch_20.pth
└── logs/                     # TensorBoard 日志
```

### 训练建议

#### 使用预训练模型时：
1. **优先使用 brain_mri 模型**：对 3D 医学图像效果最好
2. **降低学习率**：使用 5e-5 或 1e-5，避免破坏预训练权重
3. **减少训练轮数**：通常 30-50 epochs 即可收敛
4. **考虑冻结编码器**：如果数据量少（<100 样本），冻结编码器可以防止过拟合

#### 从头训练时：
1. **从小尺寸开始**：先用 64x64x64 训练，确保流程正常
2. **调整 KL 权重**：如果重建质量差，降低 `kl_weight`；如果潜空间不规则，增加 `kl_weight`
3. **监控损失平衡**：
   - 重建损失应该稳定下降
   - 生成器和判别器损失应该保持平衡（不要一方压倒另一方）
4. **显存优化**：
   - 减小 `batch_size`
   - 减小 `spatial_size`
   - 减少 `num_channels`
   - 设置 `cache_rate=0`

### 常见问题

**Q: 训练时显存不足 (OOM)**
A: 尝试以下方法：
- 减小 batch_size 到 1
- 减小 spatial_size（如 48x48x48）
- 减少 num_channels（如 [16, 32, 64]）
- 关闭数据缓存（cache_rate=0）

**Q: 重建质量不好**
A:
- 增加训练轮数
- 降低 kl_weight（如 1e-7）
- 增加 perceptual_weight
- 增加模型容量（更多 channels）

**Q: 训练不稳定**
A:
- 降低学习率（如 5e-5）
- 调整 adv_weight（如 0.001）
- 检查数据是否正确归一化

**Q: 如何选择是否使用预训练模型？**
A:
- 数据量 < 100：强烈推荐使用预训练 + 冻结编码器
- 数据量 100-500：推荐使用预训练 + 完整微调
- 数据量 > 500：可以从头训练，但预训练仍能加速收敛
- 数据与医学图像差异大：考虑从头训练

**Q: 预训练模型加载失败**
A:
- 检查模型架构是否匹配（latent_channels, num_channels 等）
- 使用 `--inspect` 查看预训练模型结构
- 脚本会自动处理部分不匹配的层（strict=False）

## Step 2: 训练扩散模型

（待完成 - 下一步）

## Step 3: 推理生成

（待完成 - 下一步）

# Phase 1 - Step 1: 3D VAE 训练完整指南

## 📦 已创建的文件

### 核心脚本
1. **`train_vae_vessel.py`** - VAE 训练主脚本
   - 支持从头训练和预训练模型微调
   - 支持 NRRD 格式数据
   - 包含完整的训练循环和验证
   - 自动保存最佳模型和检查点

2. **`download_pretrained_vae.py`** - 预训练模型下载工具
   - 下载 MONAI 官方预训练的 3D VAE
   - 支持 Brain MRI 和 Chest X-ray 模型
   - 可查看模型结构

3. **`test_vae.py`** - VAE 测试和可视化脚本
   - 重建质量测试（MSE, MAE, PSNR）
   - 潜空间插值可视化
   - 随机采样测试

### 文档
4. **`README_PHASE1.md`** - Phase 1 完整文档
   - 详细的训练流程说明
   - 参数配置指南
   - 常见问题解答

5. **`QUICKSTART_VAE.md`** - 快速开始指南
   - 不同场景的推荐配置
   - 训练监控指标说明
   - 最佳实践总结

6. **`configs/vae_vessel_config.yaml`** - 配置文件模板

---

## 🚀 完整工作流程

### Step 1: 环境准备

```bash
# 安装依赖
pip install torch torchvision
pip install monai
pip install nrrd tensorboard gdown matplotlib

# 克隆或进入项目目录
cd GenerativeModels
```

### Step 2: 下载预训练模型（推荐）

```bash
# 下载 3D Brain MRI VAE
python download_pretrained_vae.py \
    --model brain_mri \
    --output_dir ./pretrained_models \
    --inspect
```

**输出：**
```
pretrained_models/
└── brain_mri_autoencoder.pth  (约 200MB)
```

### Step 3: 准备数据

将你的 3D 血管 mask（NRRD 格式）放到数据目录：

```bash
mkdir -p data/vessel_masks
# 复制你的 .nrrd 文件到这个目录
```

**数据要求：**
- 格式：`.nrrd` 或 `.nrrd.gz`
- 维度：3D 体积数据
- 建议数量：至少 50-100 个样本

### Step 4: 训练 VAE

根据你的数据量选择合适的策略：

#### 策略 A: 数据少（< 100 样本）- 冻结编码器

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --freeze_encoder \
    --spatial_size 64 64 64 \
    --batch_size 1 \
    --num_epochs 30 \
    --lr 1e-4 \
    --val_interval 5 \
    --save_interval 10
```

#### 策略 B: 数据中等（100-500 样本）- 完整微调

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5 \
    --val_interval 5 \
    --save_interval 10
```

#### 策略 C: 数据多（> 500 样本）- 从头训练

```bash
python train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --spatial_size 128 128 128 \
    --batch_size 2 \
    --num_epochs 100 \
    --lr 1e-4 \
    --num_channels 64 128 256 \
    --latent_channels 4 \
    --val_interval 5 \
    --save_interval 10
```

### Step 5: 监控训练

在另一个终端启动 TensorBoard：

```bash
tensorboard --logdir ./outputs/vae_vessel/logs
```

打开浏览器访问 `http://localhost:6006`

**关注指标：**
- `train/recon_loss`: 应该稳定下降到 0.01-0.05
- `train/kl_loss`: 通常在 0.001-0.01
- `val/loss`: 应该跟随训练损失下降

### Step 6: 测试模型

训练完成后，测试模型质量：

```bash
python test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./test_results \
    --num_samples 5 \
    --spatial_size 64 64 64 \
    --latent_channels 3 \
    --num_channels 32 64 128 \
    --attention_levels 0 0 1
```

**输出：**
```
test_results/
├── reconstruction/
│   ├── reconstruction_1.png
│   ├── reconstruction_2.png
│   └── ...
├── interpolation/
│   └── latent_interpolation.png
└── random_samples/
    └── random_samples.png
```

### Step 7: 评估结果

检查测试输出：

1. **重建质量**：查看 `reconstruction/` 中的对比图
   - 原始 vs 重建应该非常相似
   - PSNR > 25 dB 表示质量良好

2. **潜空间平滑性**：查看 `interpolation/` 中的插值图
   - 插值应该平滑过渡
   - 没有突变或伪影

3. **随机采样**：查看 `random_samples/` 中的样本
   - 样本应该看起来像真实的血管结构
   - 多样性良好

---

## 📊 预期结果

### 训练时间估计

| 配置 | 数据量 | Epochs | GPU | 预计时间 |
|------|--------|--------|-----|----------|
| 冻结编码器 | 50 | 30 | RTX 3090 | 1-2 小时 |
| 完整微调 | 200 | 50 | RTX 3090 | 3-5 小时 |
| 从头训练 | 500 | 100 | RTX 3090 | 10-15 小时 |

### 质量指标

**良好的训练结果：**
- 重建 MSE < 0.01
- 重建 MAE < 0.05
- PSNR > 25 dB
- 验证损失稳定收敛

---

## 🔧 故障排除

### 问题 1: 显存不足

**错误信息：** `CUDA out of memory`

**解决方案：**
```bash
# 减小 batch size
--batch_size 1

# 减小 patch size
--spatial_size 48 48 48

# 减少通道数
--num_channels 16 32 64

# 关闭数据缓存
--cache_rate 0.0
```

### 问题 2: 重建质量差

**症状：** 重建图像模糊或失真

**解决方案：**
```bash
# 降低 KL 权重
--kl_weight 1e-7

# 增加感知损失权重
--perceptual_weight 0.01

# 增加训练轮数
--num_epochs 100
```

### 问题 3: 训练不稳定

**症状：** 损失震荡或发散

**解决方案：**
```bash
# 降低学习率
--lr 5e-5

# 调整对抗损失权重
--adv_weight 0.001

# 使用预训练模型
--pretrained_model ./pretrained_models/brain_mri_autoencoder.pth
```

### 问题 4: 数据加载错误

**错误信息：** `No NRRD files found`

**检查：**
1. 确认数据目录路径正确
2. 确认文件扩展名是 `.nrrd` 或 `.nrrd.gz`
3. 确认文件可以被 `nrrd.read()` 读取

### 问题 5: 训练中断

**恢复训练：**
```bash
python train_vae_vessel.py \
    --resume_from ./outputs/vae_vessel/checkpoint_epoch_20.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    # ... 其他参数保持一致
```

---

## 📁 输出文件说明

训练完成后，输出目录结构：

```
outputs/vae_vessel/
├── best_vae.pth              # 验证集上最佳模型（用于后续步骤）
├── final_vae.pth             # 最终模型
├── checkpoint_epoch_10.pth   # 定期检查点
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_30.pth
└── logs/                     # TensorBoard 日志
    └── events.out.tfevents...
```

**重要文件：**
- `best_vae.pth` - 用于 Phase 1 Step 2（训练扩散模型）
- `best_vae.pth` - 用于 Phase 2（ControlNet）

---

## ✅ 验收标准

训练成功的标志：

1. ✅ 训练损失稳定下降并收敛
2. ✅ 验证损失不上升（无过拟合）
3. ✅ 重建 PSNR > 25 dB
4. ✅ 重建图像视觉上与原图相似
5. ✅ 潜空间插值平滑
6. ✅ 随机采样生成合理的血管结构

---

## 🎯 下一步

训练好 VAE 后，继续 Phase 1 的下一步：

**Phase 1 - Step 2: 训练 3D 扩散模型**
- 使用训练好的 VAE 编码器
- 在潜空间训练扩散模型
- 实现无条件血管生成

查看 `README_PHASE1.md` 中的 Step 2 部分（即将创建）。

---

## 📚 参考资料

- [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
- [AutoencoderKL 论文](https://arxiv.org/abs/2112.10752)
- [3D LDM 教程](tutorials/generative/3d_ldm/)
- [Brain MRI 模型文档](model-zoo/models/brain_image_synthesis_latent_diffusion_model/)

---

## 💡 最佳实践

1. **数据质量优先**：确保 NRRD 文件正确加载和归一化
2. **从小规模开始**：先用 64³ 和少量数据验证流程
3. **使用预训练模型**：可以节省 50% 以上的训练时间
4. **监控 TensorBoard**：及时发现训练问题
5. **定期保存检查点**：防止训练中断导致损失
6. **测试模型质量**：不要只看损失数值，要可视化结果
7. **记录超参数**：方便后续复现和调优

---

## 🤝 获取帮助

如果遇到问题：

1. 查看 `QUICKSTART_VAE.md` 中的常见问题
2. 检查 TensorBoard 日志
3. 运行 `test_vae.py` 诊断模型
4. 查看 MONAI 官方文档和示例

---

**祝训练顺利！🎉**

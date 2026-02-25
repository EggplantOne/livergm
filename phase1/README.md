# Phase 1: 无条件 3D 血管生成

这是 Phase 1 的完整实现，用于训练无条件的 3D 血管 mask 生成模型。

## 📁 目录结构

```
phase1/
├── scripts/                          # Python 脚本
│   ├── train_vae_vessel.py          # VAE 训练脚本
│   ├── download_pretrained_vae.py   # 下载预训练模型
│   ├── test_vae.py                  # 测试和评估
│   └── visualize_vae.py             # 可视化和生成合成血管
├── configs/                          # 配置文件
│   └── vae_vessel_config.yaml       # VAE 配置模板
├── docs/                             # 文档
│   ├── QUICKSTART_VAE.md            # 快速开始指南 ⭐
│   ├── PHASE1_STEP1_SUMMARY.md      # 完整工作流程
│   ├── README_PHASE1.md             # 技术文档
│   ├── DATA_FORMAT_SUPPORT.md       # 数据格式说明
│   └── ...
├── quickstart_vae.sh                 # Linux/Mac 快速启动
├── quickstart_vae.bat                # Windows 快速启动
└── README.md                         # 本文件
```

---

## 🚀 快速开始

### 1. 准备数据

```bash
# 在项目根目录
mkdir -p data/vessel_masks

# 复制你的 NRRD 或 NIfTI 文件
cp /path/to/your/*.nrrd data/vessel_masks/
# 或
cp /path/to/your/*.nii.gz data/vessel_masks/
```

### 2. 运行快速启动脚本

**Linux/Mac:**
```bash
bash phase1/quickstart_vae.sh
```

**Windows:**
```bash
phase1\quickstart_vae.bat
```

选择适合你的场景（1-7）：
1. 小数据集（< 100 样本）
2. 中等数据集（100-500 样本）
3. 大数据集（> 500 样本）
4. 低显存（8GB GPU）
5. 下载预训练模型
6. 测试模型
7. 生成合成血管 ⭐

### 3. 监控训练

```bash
tensorboard --logdir ./outputs
```

---

## 📊 支持的数据格式

- ✅ NRRD (`.nrrd`, `.nrrd.gz`)
- ✅ NIfTI (`.nii`, `.nii.gz`)
- ✅ 可以混合使用

详见：[docs/DATA_FORMAT_SUPPORT.md](docs/DATA_FORMAT_SUPPORT.md)

---

## 🎯 Phase 1 流程

### Step 1: 训练 3D VAE ✅ (当前)

**目标：** 学习将 3D 血管 mask 压缩到潜空间

**脚本：** `scripts/train_vae_vessel.py`

**输出：** `outputs/vae_vessel/best_vae.pth`

**文档：** [docs/QUICKSTART_VAE.md](docs/QUICKSTART_VAE.md)

### Step 2: 训练扩散模型 ⏳ (计划中)

**目标：** 在潜空间学习血管分布

**使用：** 训练好的 VAE + DiffusionModelUNet

### Step 3: 生成合成血管 ⏳ (计划中)

**目标：** 生成高质量的合成血管 mask

**使用：** VAE + 扩散模型

---

## 💡 使用场景

### 场景 1: 查看 VAE 重建效果

训练完成后，测试 VAE 的重建质量：

```bash
python phase1/scripts/test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./test_results
```

### 场景 2: 生成合成血管（VAE 采样）

⚠️ **注意：** VAE 单独使用生成的质量有限，建议完成 Step 2 后使用扩散模型生成。

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --output_dir ./visualizations \
    --num_samples 10
```

**输出：**
- `visualizations/synthetic_samples/*.nrrd` - 合成的血管 mask
- `visualizations/synthetic_samples/*.png` - 可视化图像
- `visualizations/reconstructions/` - 重建对比

### 场景 3: 生成高质量合成血管（推荐）

完成 Phase 1 Step 2（训练扩散模型）后，使用 VAE + 扩散模型生成。

---

## 📚 文档导航

### 快速上手
- **[docs/QUICKSTART_VAE.md](docs/QUICKSTART_VAE.md)** - 5分钟快速开始 ⭐
- **[docs/README_VAE_TRAINING.md](docs/README_VAE_TRAINING.md)** - 项目概览

### 详细指南
- **[docs/PHASE1_STEP1_SUMMARY.md](docs/PHASE1_STEP1_SUMMARY.md)** - 完整工作流程
- **[docs/README_PHASE1.md](docs/README_PHASE1.md)** - 技术文档
- **[docs/DATA_FORMAT_SUPPORT.md](docs/DATA_FORMAT_SUPPORT.md)** - 数据格式说明

### 参考资料
- **[docs/FILE_INDEX.md](docs/FILE_INDEX.md)** - 文件导航
- **[docs/DELIVERY_SUMMARY.md](docs/DELIVERY_SUMMARY.md)** - 交付总结
- **[docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - 项目总结

---

## 🔧 手动运行命令

### 下载预训练模型

```bash
python phase1/scripts/download_pretrained_vae.py \
    --model brain_mri \
    --output_dir ./pretrained_models
```

### 训练 VAE

```bash
python phase1/scripts/train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5
```

### 测试模型

```bash
python phase1/scripts/test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./test_results
```

### 生成合成血管

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations \
    --num_samples 10 \
    --num_reconstructions 5
```

---

## 📦 依赖安装

```bash
pip install torch torchvision
pip install monai
pip install pynrrd nibabel
pip install matplotlib tensorboard gdown
```

---

## 🎯 预期效果

### 训练时间

| 配置 | 数据量 | GPU | 时间 |
|------|--------|-----|------|
| 小数据 + 预训练 | 50 | RTX 3090 | 1-2h |
| 中数据 + 微调 | 200 | RTX 3090 | 3-5h |
| 大数据 + 从头 | 500 | RTX 3090 | 10-15h |

### 质量指标

- 重建 PSNR > 25 dB
- 重建 MSE < 0.01
- 验证损失稳定收敛

---

## ⚠️ 重要说明

### 关于合成血管生成

**VAE 单独使用：**
- ✅ 可以重建真实血管（质量很好）
- ⚠️ 从随机噪声生成（质量有限）
- 💡 适合验证 VAE 训练效果

**VAE + 扩散模型（推荐）：**
- ✅ 生成高质量合成血管
- ✅ 真实感强，多样性好
- 💡 需要完成 Phase 1 Step 2

---

## 🔄 下一步

完成 Phase 1 Step 1 后：

1. **验证 VAE** - 运行 `test_vae.py` 检查重建质量
2. **可视化** - 运行 `visualize_vae.py` 查看效果
3. **Phase 1 Step 2** - 训练扩散模型（即将创建）
4. **Phase 2** - 训练 ControlNet（条件生成）

---

## 💬 获取帮助

- 快速问题 → [docs/QUICKSTART_VAE.md](docs/QUICKSTART_VAE.md)
- 故障排除 → [docs/PHASE1_STEP1_SUMMARY.md](docs/PHASE1_STEP1_SUMMARY.md)
- 技术细节 → [docs/README_PHASE1.md](docs/README_PHASE1.md)

---

**开始训练：** `bash phase1/quickstart_vae.sh` 🚀

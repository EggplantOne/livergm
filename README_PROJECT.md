# 3D 血管生成项目

基于 MONAI Generative Models 的 3D 血管 mask 生成系统。

## 📁 项目结构

```
GenerativeModels/
├── phase1/                          # Phase 1: 无条件生成
│   ├── scripts/                     # Python 脚本
│   ├── configs/                     # 配置文件
│   ├── docs/                        # 文档
│   ├── quickstart_vae.sh            # 快速启动（Linux/Mac）
│   ├── quickstart_vae.bat           # 快速启动（Windows）
│   └── README.md                    # Phase 1 说明
├── data/                            # 数据目录
│   └── vessel_masks/                # 血管 mask 数据
├── pretrained_models/               # 预训练模型
├── outputs/                         # 训练输出
├── visualizations/                  # 可视化结果
└── README.md                        # 本文件
```

---

## 🚀 快速开始

### Phase 1: 训练 3D VAE

```bash
# 1. 准备数据
mkdir -p data/vessel_masks
# 复制你的 .nrrd 或 .nii.gz 文件到 data/vessel_masks/

# 2. 运行快速启动脚本
bash phase1/quickstart_vae.sh  # Linux/Mac
# 或
phase1\quickstart_vae.bat      # Windows

# 3. 选择场景（1-7）
# 推荐：场景 2（中等数据集 + 预训练模型）
```

详细说明：[phase1/README.md](phase1/README.md)

---

## 📚 文档

### 快速上手
- **[phase1/docs/QUICKSTART_VAE.md](phase1/docs/QUICKSTART_VAE.md)** - 5分钟快速开始 ⭐
- **[phase1/README.md](phase1/README.md)** - Phase 1 完整说明

### 详细指南
- **[phase1/docs/PHASE1_STEP1_SUMMARY.md](phase1/docs/PHASE1_STEP1_SUMMARY.md)** - 完整工作流程
- **[phase1/docs/README_PHASE1.md](phase1/docs/README_PHASE1.md)** - 技术文档
- **[phase1/docs/DATA_FORMAT_SUPPORT.md](phase1/docs/DATA_FORMAT_SUPPORT.md)** - 数据格式说明

---

## 🎯 项目阶段

### ✅ Phase 1 - Step 1: 训练 3D VAE（已完成）

**目标：** 学习将 3D 血管 mask 压缩到潜空间

**功能：**
- ✅ 支持 NRRD 和 NIfTI 格式
- ✅ 预训练模型微调
- ✅ 混合精度训练
- ✅ 完整的测试和可视化工具

**快速启动：**
```bash
bash phase1/quickstart_vae.sh
```

**文档：** [phase1/README.md](phase1/README.md)

### ⏳ Phase 1 - Step 2: 训练扩散模型（计划中）

**目标：** 在潜空间学习血管分布，生成高质量合成血管

### ⏳ Phase 2: 训练 ControlNet（计划中）

**目标：** 添加条件控制，实现条件生成

---

## 💡 主要功能

### 1. 数据格式支持
- ✅ NRRD (`.nrrd`, `.nrrd.gz`)
- ✅ NIfTI (`.nii`, `.nii.gz`)
- ✅ 可以混合使用

### 2. 训练策略
- ✅ 使用预训练模型微调（推荐）
- ✅ 从头训练
- ✅ 冻结编码器训练
- ✅ 低显存优化

### 3. 可视化和生成
- ✅ 重建质量测试
- ✅ 潜空间插值
- ✅ 生成合成血管
- ✅ 3D 体积保存（NRRD/NIfTI）

---

## 📦 依赖安装

```bash
# 基础依赖
pip install torch torchvision
pip install monai

# 数据格式支持
pip install pynrrd nibabel

# 可视化和工具
pip install matplotlib tensorboard gdown
```

---

## 🔧 使用示例

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

### 生成合成血管

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations \
    --num_samples 10
```

### 监控训练

```bash
tensorboard --logdir ./outputs
```

---

## 📊 预期效果

| 配置 | 数据量 | 训练时间 | 重建 PSNR |
|------|--------|----------|-----------|
| 小数据 + 预训练 | 50 | 1-2h | > 25 dB |
| 中数据 + 微调 | 200 | 3-5h | > 28 dB |
| 大数据 + 从头 | 500 | 10-15h | > 30 dB |

---

## 🎓 参考资料

- [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
- [AutoencoderKL 论文](https://arxiv.org/abs/2112.10752)
- [Brain MRI LDM 论文](https://arxiv.org/abs/2209.07162)

---

## 💬 获取帮助

### 快速问题
- [phase1/docs/QUICKSTART_VAE.md](phase1/docs/QUICKSTART_VAE.md) - 常见问题

### 技术问题
- [phase1/docs/PHASE1_STEP1_SUMMARY.md](phase1/docs/PHASE1_STEP1_SUMMARY.md) - 故障排除

### 文件导航
- [phase1/docs/FILE_INDEX.md](phase1/docs/FILE_INDEX.md) - 所有文件说明

---

## 🔄 更新日志

### v1.0 - Phase 1 Step 1（当前）
- ✅ 完整的 3D VAE 训练系统
- ✅ 支持 NRRD 和 NIfTI 格式
- ✅ 预训练模型支持
- ✅ 完整的测试和可视化工具
- ✅ 详细的文档和指南

### 计划中
- ⏳ Phase 1 Step 2: 扩散模型训练
- ⏳ Phase 2: ControlNet 训练
- ⏳ 推理和部署工具

---

**开始使用：** `bash phase1/quickstart_vae.sh` 🚀

# 3D 血管生成项目 - Phase 1: 无条件生成

这是一个基于 MONAI Generative Models 的 3D 血管 mask 生成项目。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision
pip install monai
pip install nrrd tensorboard gdown matplotlib
```

### 2. 准备数据

```bash
mkdir -p data/vessel_masks
# 将你的 .nrrd 血管 mask 文件复制到这个目录
```

### 3. 开始训练

**Windows:**
```bash
quickstart_vae.bat
```

**Linux/Mac:**
```bash
bash quickstart_vae.sh
```

选择适合你的场景（1-6），脚本会自动处理一切！

## 📚 文档

- **[QUICKSTART_VAE.md](QUICKSTART_VAE.md)** - 5分钟快速上手 ⭐ 从这里开始
- **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** - 完整交付总结
- **[PHASE1_STEP1_SUMMARY.md](PHASE1_STEP1_SUMMARY.md)** - 详细工作流程
- **[FILE_INDEX.md](FILE_INDEX.md)** - 文件导航
- **[README_PHASE1.md](README_PHASE1.md)** - 技术文档

## 📦 项目结构

```
GenerativeModels/
├── train_vae_vessel.py          # VAE 训练脚本
├── download_pretrained_vae.py   # 下载预训练模型
├── test_vae.py                  # 测试和可视化
├── quickstart_vae.bat           # Windows 快速启动
├── quickstart_vae.sh            # Linux/Mac 快速启动
├── configs/
│   └── vae_vessel_config.yaml   # 配置模板
├── data/
│   └── vessel_masks/            # 你的 NRRD 数据
├── pretrained_models/           # 预训练模型（自动下载）
├── outputs/                     # 训练输出
└── test_results/                # 测试结果
```

## 🎯 当前进度

### ✅ Phase 1 - Step 1: 训练 3D VAE（已完成）

- ✅ 完整的训练脚本
- ✅ 预训练模型支持
- ✅ 测试和可视化工具
- ✅ 详细文档

### ⏳ Phase 1 - Step 2: 训练扩散模型（计划中）

使用训练好的 VAE，在潜空间训练扩散模型。

### ⏳ Phase 2: 训练 ControlNet（计划中）

添加条件控制，实现条件生成。

## 💡 使用场景

### 场景 1: 数据少（< 100 样本）
使用预训练模型 + 冻结编码器，30 epochs，约 1-2 小时

### 场景 2: 数据中等（100-500 样本）
使用预训练模型 + 完整微调，50 epochs，约 3-5 小时

### 场景 3: 数据多（> 500 样本）
从头训练或使用预训练，100 epochs，约 10-15 小时

### 场景 4: 低显存（8GB GPU）
优化配置，支持小显存 GPU

## 📊 预期结果

训练完成后，你将得到：

- `outputs/vae_vessel/best_vae.pth` - 最佳 VAE 模型
- 重建 PSNR > 25 dB
- 平滑的潜空间
- 可用于后续扩散模型训练

## 🔧 核心功能

- ✅ 支持 NRRD 格式 3D 数据
- ✅ 预训练模型微调
- ✅ 混合精度训练
- ✅ 自动保存最佳模型
- ✅ TensorBoard 监控
- ✅ 完整的测试工具
- ✅ 跨平台支持

## 📖 详细使用

查看 [QUICKSTART_VAE.md](QUICKSTART_VAE.md) 了解：

- 不同场景的详细配置
- 训练监控指标
- 常见问题解答
- 最佳实践

## 🤝 获取帮助

1. 查看 [QUICKSTART_VAE.md](QUICKSTART_VAE.md) 的常见问题
2. 查看 [PHASE1_STEP1_SUMMARY.md](PHASE1_STEP1_SUMMARY.md) 的故障排除
3. 查看 [FILE_INDEX.md](FILE_INDEX.md) 快速查找

## 📚 参考资料

- [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
- [AutoencoderKL 论文](https://arxiv.org/abs/2112.10752)
- [Brain MRI LDM 论文](https://arxiv.org/abs/2209.07162)

## 🎉 开始使用

```bash
# 1. 准备数据
mkdir -p data/vessel_masks
# 复制你的 .nrrd 文件

# 2. 运行快速启动脚本
quickstart_vae.bat  # Windows
# 或
bash quickstart_vae.sh  # Linux/Mac

# 3. 监控训练
tensorboard --logdir ./outputs

# 4. 测试模型
python test_vae.py --model_path ./outputs/vae_*/best_vae.pth --data_dir ./data/vessel_masks
```

**祝训练顺利！** 🚀

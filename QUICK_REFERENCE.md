# 🚀 Phase 1 快速参考

## 📁 目录结构

```
phase1/
├── scripts/          # 4个Python脚本
├── configs/          # 1个配置文件
├── docs/             # 9个文档
├── quickstart_vae.sh # Linux/Mac启动
├── quickstart_vae.bat# Windows启动
└── README.md         # Phase 1说明
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

## ⚡ 快速命令

### 1. 快速启动（推荐）

```bash
# Linux/Mac
bash phase1/quickstart_vae.sh

# Windows
phase1\quickstart_vae.bat
```

### 2. 下载预训练模型

```bash
python phase1/scripts/download_pretrained_vae.py \
    --model brain_mri \
    --output_dir ./pretrained_models
```

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

## 📚 文档快速链接

| 文档 | 用途 |
|------|------|
| [phase1/README.md](phase1/README.md) | Phase 1 完整说明 |
| [phase1/docs/QUICKSTART_VAE.md](phase1/docs/QUICKSTART_VAE.md) | 5分钟快速上手 ⭐ |
| [phase1/docs/PHASE1_STEP1_SUMMARY.md](phase1/docs/PHASE1_STEP1_SUMMARY.md) | 完整工作流程 |
| [phase1/docs/DATA_FORMAT_SUPPORT.md](phase1/docs/DATA_FORMAT_SUPPORT.md) | 数据格式说明 |

---

## 🎯 服务器使用

### 路径配置

```bash
# 服务器路径
PROJECT_ROOT=/home/yinhaojie/GenerativeModels
DATA_DIR=$PROJECT_ROOT/data/vessel_masks
OUTPUT_DIR=$PROJECT_ROOT/outputs/vae_vessel
PRETRAINED=$PROJECT_ROOT/pretrained_models/brain_mri_autoencoder.pth
```

### 后台训练

```bash
cd /home/yinhaojie/GenerativeModels

nohup python phase1/scripts/train_vae_vessel.py \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel \
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
    --spatial_size 64 64 64 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 5e-5 \
    > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### TensorBoard（端口转发）

```bash
# 服务器上
tensorboard --logdir ./outputs --host 0.0.0.0 --port 6006

# 本地电脑上
ssh -L 6006:localhost:6006 yinhaojie@your-server

# 浏览器打开
http://localhost:6006
```

---

## 📊 支持的数据格式

- ✅ NRRD (`.nrrd`, `.nrrd.gz`)
- ✅ NIfTI (`.nii`, `.nii.gz`)
- ✅ 可混合使用

---

## 🎯 训练场景

| 场景 | 数据量 | 命令 |
|------|--------|------|
| 小数据集 | < 100 | `quickstart_vae.sh` 选项 1 |
| 中数据集 | 100-500 | `quickstart_vae.sh` 选项 2 |
| 大数据集 | > 500 | `quickstart_vae.sh` 选项 3 |
| 低显存 | 任意 | `quickstart_vae.sh` 选项 4 |

---

## 💡 常用操作

### 查看帮助

```bash
python phase1/scripts/train_vae_vessel.py --help
python phase1/scripts/visualize_vae.py --help
```

### 恢复训练

```bash
python phase1/scripts/train_vae_vessel.py \
    --resume_from ./outputs/vae_vessel/checkpoint_epoch_20.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./outputs/vae_vessel
```

### 查看输出

```bash
# 训练输出
ls -lh outputs/vae_vessel/

# 可视化结果
ls -lh visualizations/synthetic_samples/
ls -lh visualizations/reconstructions/
```

---

## 🔧 依赖安装

```bash
pip install torch torchvision monai pynrrd nibabel matplotlib tensorboard gdown
```

---

## ⚠️ 重要提示

### 关于合成血管生成

**VAE 单独使用：**
- ✅ 重建真实血管（质量好）
- ⚠️ 随机生成（质量有限）

**VAE + 扩散模型（推荐）：**
- ✅ 高质量合成血管
- 需要完成 Phase 1 Step 2

---

## 📞 获取帮助

- 快速问题 → `phase1/docs/QUICKSTART_VAE.md`
- 故障排除 → `phase1/docs/PHASE1_STEP1_SUMMARY.md`
- 技术细节 → `phase1/docs/README_PHASE1.md`

---

**开始训练：** `bash phase1/quickstart_vae.sh` 🚀

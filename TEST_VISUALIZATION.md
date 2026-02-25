# 🎨 快速测试可视化功能

## 📋 前提条件

在测试可视化之前，你需要：

1. ✅ 已训练好的 VAE 模型（`best_vae.pth`）
2. ✅ 一些测试数据（NRRD 或 NIfTI 格式）

**如果还没有训练好的模型，可以：**
- 选项 A: 先训练一个模型（推荐）
- 选项 B: 使用预训练模型测试（仅用于验证流程）

---

## 🚀 方法 1: 使用快速启动脚本（最简单）

### 在服务器上

```bash
cd /home/yinhaojie/GenerativeModels

# 运行快速启动脚本
bash phase1/quickstart_vae.sh

# 选择选项 7: Visualize and generate synthetic vessels
```

脚本会自动：
1. 查找训练好的模型
2. 运行可视化脚本
3. 生成结果到 `visualizations/` 目录

---

## 🔧 方法 2: 手动运行（更灵活）

### Step 1: 检查是否有训练好的模型

```bash
cd /home/yinhaojie/GenerativeModels

# 查找模型文件
find outputs -name "best_vae.pth" -o -name "final_vae.pth"

# 如果找到了，记下路径，例如：
# outputs/vae_vessel/best_vae.pth
```

### Step 2: 准备测试数据

```bash
# 检查数据目录
ls -lh data/vessel_masks/

# 应该有一些 .nrrd 或 .nii.gz 文件
```

### Step 3: 运行可视化脚本

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations \
    --num_samples 10 \
    --num_reconstructions 5 \
    --save_format nrrd
```

**参数说明：**
- `--model_path`: 训练好的 VAE 模型路径
- `--data_dir`: 测试数据目录（用于重建测试）
- `--output_dir`: 输出目录
- `--num_samples`: 生成多少个合成血管
- `--num_reconstructions`: 重建多少个真实血管
- `--save_format`: 保存格式（`nrrd` 或 `nifti`）

### Step 4: 查看结果

```bash
# 查看生成的文件
ls -lh visualizations/

# 应该看到：
# visualizations/
# ├── synthetic_samples/      # 合成的血管
# └── reconstructions/         # 重建对比
```

---

## 📊 输出文件说明

### 1. 合成血管（synthetic_samples/）

```
synthetic_samples/
├── synthetic_vessel_001.nrrd          # 3D 血管 mask（可用 3D Slicer 打开）
├── synthetic_vessel_001_slices.png    # 多层切片可视化
├── synthetic_vessel_001_mip.png       # 最大强度投影（3D 渲染）
├── synthetic_vessel_002.nrrd
├── synthetic_vessel_002_slices.png
├── synthetic_vessel_002_mip.png
└── ...
```

### 2. 重建对比（reconstructions/）

```
reconstructions/
├── reconstruction_001.png             # 原始 vs 重建对比图
├── reconstructed_001.nrrd             # 重建的 3D 体积
├── reconstruction_002.png
├── reconstructed_002.nrrd
└── ...
```

---

## 🖼️ 查看可视化结果

### 方法 1: 查看 PNG 图像（最简单）

```bash
# 下载 PNG 图像到本地
scp yinhaojie@your-server:/home/yinhaojie/GenerativeModels/visualizations/synthetic_samples/*.png ./

# 在本地用图片查看器打开
```

### 方法 2: 查看 3D 体积（推荐）

下载 NRRD 文件并用专业软件打开：

```bash
# 下载 NRRD 文件
scp yinhaojie@your-server:/home/yinhaojie/GenerativeModels/visualizations/synthetic_samples/*.nrrd ./
```

**推荐软件：**
- **3D Slicer** (https://www.slicer.org/) - 免费，功能强大 ⭐
- **ITK-SNAP** (http://www.itksnap.org/) - 免费，易用
- **ParaView** (https://www.paraview.org/) - 免费，科学可视化

### 方法 3: 在服务器上快速预览

如果服务器有图形界面：

```bash
# 使用 matplotlib 快速查看
python -c "
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('visualizations/synthetic_samples/synthetic_vessel_001_slices.png')
plt.imshow(img)
plt.axis('off')
plt.savefig('preview.png')
print('Saved to preview.png')
"
```

---

## 🧪 测试场景

### 场景 1: 只想看看效果（没有训练模型）

如果你还没有训练模型，可以先用预训练模型测试流程：

```bash
# 1. 下载预训练模型
python phase1/scripts/download_pretrained_vae.py \
    --model brain_mri \
    --output_dir ./pretrained_models

# 2. 用预训练模型测试（注意：效果可能不好，因为不是针对血管训练的）
python phase1/scripts/visualize_vae.py \
    --model_path ./pretrained_models/brain_mri_autoencoder.pth \
    --output_dir ./visualizations_test \
    --num_samples 5 \
    --latent_channels 3 \
    --num_channels 64 128 128 128 \
    --attention_levels 0 0 0 0
```

⚠️ **注意：** 预训练模型是在脑部 MRI 上训练的，用于血管效果不会很好。这只是为了测试流程。

### 场景 2: 已有训练好的模型

```bash
# 直接运行可视化
bash phase1/quickstart_vae.sh
# 选择选项 7
```

### 场景 3: 只想测试重建（不生成合成血管）

```bash
python phase1/scripts/test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./test_results \
    --num_samples 5
```

这会生成重建对比图，但不会生成合成血管。

---

## 📝 完整示例

### 示例 1: 生成 10 个合成血管

```bash
cd /home/yinhaojie/GenerativeModels

python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations \
    --num_samples 10 \
    --num_reconstructions 5 \
    --save_format nrrd \
    --latent_channels 3 \
    --num_channels 32 64 128 \
    --attention_levels 0 0 1
```

### 示例 2: 只生成合成血管（不测试重建）

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --output_dir ./visualizations \
    --num_samples 20 \
    --save_format nrrd
```

### 示例 3: 保存为 NIfTI 格式

```bash
python phase1/scripts/visualize_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./visualizations_nifti \
    --num_samples 10 \
    --save_format nifti
```

---

## 🔍 检查输出

### 查看生成的文件

```bash
# 查看目录结构
tree visualizations/

# 或
find visualizations -type f

# 查看文件大小
du -sh visualizations/*
```

### 查看脚本输出

脚本会打印详细信息：

```
==========================================
Generating Synthetic Vessel Masks
==========================================
Sample 1/10:
  Shape: torch.Size([1, 1, 64, 64, 64])
  Value range: [0.000, 0.998]
  Mean: 0.123
  Saved to: visualizations/synthetic_samples/synthetic_vessel_001.nrrd

Sample 2/10:
  ...

✅ Generated 10 synthetic vessel masks
📁 Saved to: visualizations/synthetic_samples
```

---

## ⚠️ 常见问题

### Q1: 找不到模型文件

```bash
# 检查是否有训练好的模型
find outputs -name "*.pth"

# 如果没有，需要先训练
bash phase1/quickstart_vae.sh  # 选择选项 1-4
```

### Q2: 模型架构不匹配

确保 `--latent_channels`, `--num_channels`, `--attention_levels` 与训练时一致。

查看训练日志或使用默认值：
```bash
--latent_channels 3 \
--num_channels 32 64 128 \
--attention_levels 0 0 1
```

### Q3: 生成的血管质量不好

这是正常的！VAE 单独使用生成质量有限。要生成高质量血管，需要：
1. 完成 Phase 1 Step 2（训练扩散模型）
2. 使用 VAE + 扩散模型生成

当前的可视化主要用于：
- ✅ 验证 VAE 训练效果（通过重建）
- ✅ 测试流程是否正常
- ⚠️ 生成的合成血管质量有限

### Q4: 如何下载结果到本地

```bash
# 下载整个 visualizations 目录
scp -r yinhaojie@your-server:/home/yinhaojie/GenerativeModels/visualizations ./

# 或只下载 PNG 图像
scp yinhaojie@your-server:/home/yinhaojie/GenerativeModels/visualizations/*/*.png ./
```

---

## 🎯 下一步

### 如果可视化成功
1. ✅ 查看重建质量（应该很好）
2. ✅ 查看合成血管（质量有限是正常的）
3. ⏳ 继续 Phase 1 Step 2（训练扩散模型）

### 如果遇到问题
1. 检查模型路径是否正确
2. 检查模型架构参数是否匹配
3. 查看错误信息
4. 参考 `phase1/docs/PHASE1_STEP1_SUMMARY.md` 故障排除

---

**快速开始：** `bash phase1/quickstart_vae.sh` 选择选项 7 🚀

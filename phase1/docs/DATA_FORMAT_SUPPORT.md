# 数据格式支持更新

## ✅ 现在支持的格式

代码已更新，现在同时支持：

### 1. NRRD 格式
- `.nrrd`
- `.nrrd.gz`

### 2. NIfTI 格式 ✨ 新增
- `.nii`
- `.nii.gz`

---

## 📦 依赖要求

### 必需安装

```bash
# NRRD 支持
pip install pynrrd

# NIfTI 支持
pip install nibabel

# 其他依赖
pip install torch monai matplotlib tensorboard
```

### 完整安装命令

```bash
pip install torch torchvision
pip install monai
pip install pynrrd nibabel
pip install matplotlib tensorboard gdown
```

---

## 🔧 使用方法

### 数据目录结构

你可以混合使用 NRRD 和 NIfTI 格式：

```
data/vessel_masks/
├── vessel_001.nrrd
├── vessel_002.nii.gz
├── vessel_003.nrrd.gz
├── vessel_004.nii
└── ...
```

### 训练命令（不变）

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

脚本会自动检测并加载两种格式的文件。

### 输出示例

```
Found 50 NRRD files and 30 NIfTI files
Total: 80 files
Train: 72, Val: 8
```

---

## 📝 技术细节

### 自动格式检测

代码会根据文件扩展名自动选择加载方法：

```python
def load_medical_image(filepath):
    """Load NRRD or NIfTI file."""
    if filepath.endswith(('.nrrd', '.nrrd.gz')):
        # Load NRRD
        data, header = nrrd.read(filepath)
        return data
    elif filepath.endswith(('.nii', '.nii.gz')):
        # Load NIfTI
        img = nib.load(filepath)
        data = img.get_fdata()
        return data
```

### 数据处理流程

1. **扫描目录** - 查找所有 `.nrrd`, `.nrrd.gz`, `.nii`, `.nii.gz` 文件
2. **自动加载** - 根据扩展名选择合适的加载器
3. **统一处理** - 转换为相同的 numpy 数组格式
4. **标准化** - 应用相同的预处理和增强

---

## 🎯 格式选择建议

### NRRD 格式
- ✅ 医学图像常用格式
- ✅ 保留完整的元数据
- ✅ 支持压缩（.nrrd.gz）

### NIfTI 格式
- ✅ 神经影像学标准格式
- ✅ 广泛的工具支持
- ✅ 更好的压缩率（.nii.gz）

**建议：** 两种格式都很好，选择你最方便的即可。代码会自动处理。

---

## ⚠️ 注意事项

### 1. 数据方向

NIfTI 和 NRRD 可能有不同的坐标系统。代码中使用了 `Orientationd` 转换来统一方向：

```python
transforms.Orientationd(keys=["image"], axcodes="RAS")
```

这会将所有图像转换为 RAS（Right-Anterior-Superior）坐标系。

### 2. 数据类型

两种格式都会被转换为 float32 类型，并归一化到 [0, 1] 范围。

### 3. 混合使用

可以在同一个数据集中混合使用两种格式，脚本会自动处理。

---

## 🧪 测试

### 测试 NRRD 文件

```bash
python test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks \
    --output_dir ./test_results
```

### 测试 NIfTI 文件

```bash
python test_vae.py \
    --model_path ./outputs/vae_vessel/best_vae.pth \
    --data_dir ./data/vessel_masks_nifti \
    --output_dir ./test_results_nifti
```

---

## 🔄 格式转换（可选）

如果需要在两种格式之间转换：

### NRRD → NIfTI

```python
import nrrd
import nibabel as nib

# 读取 NRRD
data, header = nrrd.read('vessel.nrrd')

# 保存为 NIfTI
img = nib.Nifti1Image(data, affine=np.eye(4))
nib.save(img, 'vessel.nii.gz')
```

### NIfTI → NRRD

```python
import nrrd
import nibabel as nib

# 读取 NIfTI
img = nib.load('vessel.nii.gz')
data = img.get_fdata()

# 保存为 NRRD
nrrd.write('vessel.nrrd', data)
```

---

## ✅ 更新的文件

1. **train_vae_vessel.py** - 训练脚本
   - 添加 NIfTI 支持
   - 自动格式检测
   - 统一数据加载

2. **test_vae.py** - 测试脚本
   - 添加 NIfTI 支持
   - 自动格式检测

---

## 📚 相关文档

- [NRRD 格式规范](http://teem.sourceforge.net/nrrd/format.html)
- [NIfTI 格式规范](https://nifti.nimh.nih.gov/)
- [nibabel 文档](https://nipy.org/nibabel/)
- [pynrrd 文档](https://pynrrd.readthedocs.io/)

---

**现在你可以使用 NRRD 或 NIfTI 格式的数据进行训练了！** 🎉

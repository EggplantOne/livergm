#!/bin/bash

# 创建 phase1 目录结构
mkdir -p phase1/scripts
mkdir -p phase1/configs
mkdir -p phase1/docs

# 移动 Python 脚本
mv train_vae_vessel.py phase1/scripts/
mv download_pretrained_vae.py phase1/scripts/
mv test_vae.py phase1/scripts/
mv visualize_vae.py phase1/scripts/

# 移动启动脚本
mv quickstart_vae.sh phase1/
mv quickstart_vae.bat phase1/

# 移动配置文件
mv configs/vae_vessel_config.yaml phase1/configs/

# 移动文档
mv QUICKSTART_VAE.md phase1/docs/
mv PHASE1_STEP1_SUMMARY.md phase1/docs/
mv README_PHASE1.md phase1/docs/
mv README_VAE_TRAINING.md phase1/docs/
mv FILE_INDEX.md phase1/docs/
mv DELIVERY_SUMMARY.md phase1/docs/
mv PROJECT_SUMMARY.md phase1/docs/
mv DATA_FORMAT_SUPPORT.md phase1/docs/
mv UPDATE_NIFTI_SUPPORT.md phase1/docs/

echo "✅ Files organized into phase1/ directory"

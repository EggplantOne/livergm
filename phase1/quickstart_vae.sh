#!/bin/bash

# Phase 1 - Step 1: Quick Start Script for 3D VAE Training
# This script provides ready-to-use commands for different scenarios

echo "=========================================="
echo "3D VAE Training - Quick Start"
echo "=========================================="
echo ""

# Check if data directory exists
if [ ! -d "data/vessel_masks" ]; then
    echo "⚠️  Warning: data/vessel_masks directory not found!"
    echo "Please create it and add your NRRD/NIfTI files:"
    echo "  mkdir -p data/vessel_masks"
    echo "  # Copy your .nrrd or .nii.gz files to data/vessel_masks/"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Menu
echo "Choose your training scenario:"
echo ""
echo "1. Small dataset (< 100 samples) - Use pretrained + freeze encoder"
echo "2. Medium dataset (100-500 samples) - Use pretrained + full fine-tuning"
echo "3. Large dataset (> 500 samples) - Train from scratch"
echo "4. Low memory (8GB GPU) - Optimized for limited VRAM"
echo "5. Download pretrained model only"
echo "6. Test existing model"
echo "7. Visualize and generate synthetic vessels"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "Scenario 1: Small Dataset + Pretrained"
        echo "=========================================="
        echo ""

        # Download pretrained model if not exists
        if [ ! -f "pretrained_models/brain_mri_autoencoder.pth" ]; then
            echo "Downloading pretrained model..."
            python phase1/scripts/download_pretrained_vae.py \
                --model brain_mri \
                --output_dir ./pretrained_models
        fi

        echo ""
        echo "Starting training..."
        python phase1/scripts/train_vae_vessel.py \
            --data_dir ./data/vessel_masks \
            --output_dir ./outputs/vae_small_dataset \
            --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
            --freeze_encoder \
            --spatial_size 64 64 64 \
            --batch_size 1 \
            --num_epochs 30 \
            --lr 1e-4 \
            --val_interval 5 \
            --save_interval 10 \
            --num_workers 4
        ;;

    2)
        echo ""
        echo "=========================================="
        echo "Scenario 2: Medium Dataset + Fine-tuning"
        echo "=========================================="
        echo ""

        # Download pretrained model if not exists
        if [ ! -f "pretrained_models/brain_mri_autoencoder.pth" ]; then
            echo "Downloading pretrained model..."
            python phase1/scripts/download_pretrained_vae.py \
                --model brain_mri \
                --output_dir ./pretrained_models
        fi

        echo ""
        echo "Starting training..."
        python phase1/scripts/train_vae_vessel.py \
            --data_dir ./data/vessel_masks \
            --output_dir ./outputs/vae_medium_dataset \
            --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
            --spatial_size 64 64 64 \
            --batch_size 2 \
            --num_epochs 50 \
            --lr 5e-5 \
            --val_interval 5 \
            --save_interval 10 \
            --num_workers 4
        ;;

    3)
        echo ""
        echo "=========================================="
        echo "Scenario 3: Large Dataset - From Scratch"
        echo "=========================================="
        echo ""
        echo "Starting training..."
        python phase1/scripts/train_vae_vessel.py \
            --data_dir ./data/vessel_masks \
            --output_dir ./outputs/vae_large_dataset \
            --spatial_size 128 128 128 \
            --batch_size 2 \
            --num_epochs 100 \
            --lr 1e-4 \
            --num_channels 64 128 256 \
            --latent_channels 4 \
            --attention_levels 0 1 1 \
            --val_interval 5 \
            --save_interval 10 \
            --num_workers 4
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "Scenario 4: Low Memory (8GB GPU)"
        echo "=========================================="
        echo ""

        # Download pretrained model if not exists
        if [ ! -f "pretrained_models/brain_mri_autoencoder.pth" ]; then
            echo "Downloading pretrained model..."
            python phase1/scripts/download_pretrained_vae.py \
                --model brain_mri \
                --output_dir ./pretrained_models
        fi

        echo ""
        echo "Starting training with memory optimizations..."
        python phase1/scripts/train_vae_vessel.py \
            --data_dir ./data/vessel_masks \
            --output_dir ./outputs/vae_low_memory \
            --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth \
            --spatial_size 48 48 48 \
            --batch_size 1 \
            --num_epochs 50 \
            --lr 5e-5 \
            --num_channels 16 32 64 \
            --cache_rate 0.0 \
            --val_interval 5 \
            --save_interval 10 \
            --num_workers 2
        ;;

    5)
        echo ""
        echo "=========================================="
        echo "Downloading Pretrained Models"
        echo "=========================================="
        echo ""
        python phase1/scripts/download_pretrained_vae.py \
            --model brain_mri \
            --output_dir ./pretrained_models \
            --inspect
        echo ""
        echo "✅ Download complete!"
        echo "Model saved to: pretrained_models/brain_mri_autoencoder.pth"
        ;;

    6)
        echo ""
        echo "=========================================="
        echo "Testing Existing Model"
        echo "=========================================="
        echo ""

        # Find the most recent model
        if [ -f "outputs/vae_small_dataset/best_vae.pth" ]; then
            MODEL_PATH="outputs/vae_small_dataset/best_vae.pth"
        elif [ -f "outputs/vae_medium_dataset/best_vae.pth" ]; then
            MODEL_PATH="outputs/vae_medium_dataset/best_vae.pth"
        elif [ -f "outputs/vae_large_dataset/best_vae.pth" ]; then
            MODEL_PATH="outputs/vae_large_dataset/best_vae.pth"
        else
            echo "⚠️  No trained model found!"
            echo "Please train a model first (options 1-4)"
            exit 1
        fi

        echo "Testing model: $MODEL_PATH"
        echo ""
        python phase1/scripts/test_vae.py \
            --model_path $MODEL_PATH \
            --data_dir ./data/vessel_masks \
            --output_dir ./test_results \
            --num_samples 5 \
            --spatial_size 64 64 64 \
            --latent_channels 3 \
            --num_channels 32 64 128 \
            --attention_levels 0 0 1

        echo ""
        echo "✅ Testing complete!"
        echo "Results saved to: test_results/"
        ;;

    7)
        echo ""
        echo "=========================================="
        echo "Visualize and Generate Synthetic Vessels"
        echo "=========================================="
        echo ""

        # Find the most recent model
        if [ -f "outputs/vae_small_dataset/best_vae.pth" ]; then
            MODEL_PATH="outputs/vae_small_dataset/best_vae.pth"
        elif [ -f "outputs/vae_medium_dataset/best_vae.pth" ]; then
            MODEL_PATH="outputs/vae_medium_dataset/best_vae.pth"
        elif [ -f "outputs/vae_large_dataset/best_vae.pth" ]; then
            MODEL_PATH="outputs/vae_large_dataset/best_vae.pth"
        else
            echo "⚠️  No trained model found!"
            echo "Please train a model first (options 1-4)"
            exit 1
        fi

        echo "Using model: $MODEL_PATH"
        echo ""
        python phase1/scripts/visualize_vae.py \
            --model_path $MODEL_PATH \
            --data_dir ./data/vessel_masks \
            --output_dir ./visualizations \
            --num_samples 10 \
            --num_reconstructions 5 \
            --save_format nrrd

        echo ""
        echo "✅ Visualization complete!"
        echo "Results saved to: visualizations/"
        echo ""
        echo "Generated files:"
        echo "  - synthetic_samples/ (synthetic vessel masks)"
        echo "  - reconstructions/ (real vs reconstructed)"
        ;;

    *)
        echo "Invalid choice. Please run the script again and choose 1-7."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""

if [ "$choice" != "5" ] && [ "$choice" != "6" ] && [ "$choice" != "7" ]; then
    echo "1. Monitor training with TensorBoard:"
    echo "   tensorboard --logdir ./outputs/*/logs"
    echo ""
    echo "2. After training, test the model:"
    echo "   bash phase1/quickstart_vae.sh"
    echo "   (Choose option 6)"
    echo ""
    echo "3. Generate synthetic vessels:"
    echo "   bash phase1/quickstart_vae.sh"
    echo "   (Choose option 7)"
fi

echo ""
echo "For more details, see:"
echo "  - phase1/docs/QUICKSTART_VAE.md"
echo "  - phase1/docs/PHASE1_STEP1_SUMMARY.md"
echo "  - phase1/docs/README_PHASE1.md"
echo ""

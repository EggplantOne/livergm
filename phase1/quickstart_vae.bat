@echo off
REM Phase 1 - Step 1: Quick Start Script for 3D VAE Training (Windows)

echo ==========================================
echo 3D VAE Training - Quick Start
echo ==========================================
echo.

REM Check if data directory exists
if not exist "data\vessel_masks" (
    echo WARNING: data\vessel_masks directory not found!
    echo Please create it and add your NRRD/NIfTI files:
    echo   mkdir data\vessel_masks
    echo   REM Copy your .nrrd or .nii.gz files to data\vessel_masks\
    echo.
    pause
)

REM Menu
echo Choose your training scenario:
echo.
echo 1. Small dataset (^< 100 samples^) - Use pretrained + freeze encoder
echo 2. Medium dataset (100-500 samples^) - Use pretrained + full fine-tuning
echo 3. Large dataset (^> 500 samples^) - Train from scratch
echo 4. Low memory (8GB GPU^) - Optimized for limited VRAM
echo 5. Download pretrained model only
echo 6. Test existing model
echo 7. Visualize and generate synthetic vessels
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto scenario1
if "%choice%"=="2" goto scenario2
if "%choice%"=="3" goto scenario3
if "%choice%"=="4" goto scenario4
if "%choice%"=="5" goto scenario5
if "%choice%"=="6" goto scenario6
if "%choice%"=="7" goto scenario7
goto invalid

:scenario1
echo.
echo ==========================================
echo Scenario 1: Small Dataset + Pretrained
echo ==========================================
echo.

REM Download pretrained model if not exists
if not exist "pretrained_models\brain_mri_autoencoder.pth" (
    echo Downloading pretrained model...
    python phase1\scripts\download_pretrained_vae.py --model brain_mri --output_dir ./pretrained_models
)

echo.
echo Starting training...
python phase1\scripts\train_vae_vessel.py ^
    --data_dir ./data/vessel_masks ^
    --output_dir ./outputs/vae_small_dataset ^
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth ^
    --freeze_encoder ^
    --spatial_size 64 64 64 ^
    --batch_size 1 ^
    --num_epochs 30 ^
    --lr 1e-4 ^
    --val_interval 5 ^
    --save_interval 10 ^
    --num_workers 4
goto end

:scenario2
echo.
echo ==========================================
echo Scenario 2: Medium Dataset + Fine-tuning
echo ==========================================
echo.

REM Download pretrained model if not exists
if not exist "pretrained_models\brain_mri_autoencoder.pth" (
    echo Downloading pretrained model...
    python phase1\scripts\download_pretrained_vae.py --model brain_mri --output_dir ./pretrained_models
)

echo.
echo Starting training...
python phase1\scripts\train_vae_vessel.py ^
    --data_dir ./data/vessel_masks ^
    --output_dir ./outputs/vae_medium_dataset ^
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth ^
    --spatial_size 64 64 64 ^
    --batch_size 2 ^
    --num_epochs 50 ^
    --lr 5e-5 ^
    --val_interval 5 ^
    --save_interval 10 ^
    --num_workers 4
goto end

:scenario3
echo.
echo ==========================================
echo Scenario 3: Large Dataset - From Scratch
echo ==========================================
echo.
echo Starting training...
python phase1\scripts\train_vae_vessel.py ^
    --data_dir ./data/vessel_masks ^
    --output_dir ./outputs/vae_large_dataset ^
    --spatial_size 128 128 128 ^
    --batch_size 2 ^
    --num_epochs 100 ^
    --lr 1e-4 ^
    --num_channels 64 128 256 ^
    --latent_channels 4 ^
    --attention_levels 0 1 1 ^
    --val_interval 5 ^
    --save_interval 10 ^
    --num_workers 4
goto end

:scenario4
echo.
echo ==========================================
echo Scenario 4: Low Memory (8GB GPU)
echo ==========================================
echo.

REM Download pretrained model if not exists
if not exist "pretrained_models\brain_mri_autoencoder.pth" (
    echo Downloading pretrained model...
    python phase1\scripts\download_pretrained_vae.py --model brain_mri --output_dir ./pretrained_models
)

echo.
echo Starting training with memory optimizations...
python phase1\scripts\train_vae_vessel.py ^
    --data_dir ./data/vessel_masks ^
    --output_dir ./outputs/vae_low_memory ^
    --pretrained_model ./pretrained_models/brain_mri_autoencoder.pth ^
    --spatial_size 48 48 48 ^
    --batch_size 1 ^
    --num_epochs 50 ^
    --lr 5e-5 ^
    --num_channels 16 32 64 ^
    --cache_rate 0.0 ^
    --val_interval 5 ^
    --save_interval 10 ^
    --num_workers 2
goto end

:scenario5
echo.
echo ==========================================
echo Downloading Pretrained Models
echo ==========================================
echo.
python phase1\scripts\download_pretrained_vae.py ^
    --model brain_mri ^
    --output_dir ./pretrained_models ^
    --inspect
echo.
echo Download complete!
echo Model saved to: pretrained_models\brain_mri_autoencoder.pth
goto end

:scenario6
echo.
echo ==========================================
echo Testing Existing Model
echo ==========================================
echo.

REM Find the most recent model
set MODEL_PATH=
if exist "outputs\vae_small_dataset\best_vae.pth" (
    set MODEL_PATH=outputs\vae_small_dataset\best_vae.pth
) else if exist "outputs\vae_medium_dataset\best_vae.pth" (
    set MODEL_PATH=outputs\vae_medium_dataset\best_vae.pth
) else if exist "outputs\vae_large_dataset\best_vae.pth" (
    set MODEL_PATH=outputs\vae_large_dataset\best_vae.pth
) else (
    echo WARNING: No trained model found!
    echo Please train a model first (options 1-4^)
    pause
    exit /b 1
)

echo Testing model: %MODEL_PATH%
echo.
python phase1\scripts\test_vae.py ^
    --model_path %MODEL_PATH% ^
    --data_dir ./data/vessel_masks ^
    --output_dir ./test_results ^
    --num_samples 5 ^
    --spatial_size 64 64 64 ^
    --latent_channels 3 ^
    --num_channels 32 64 128 ^
    --attention_levels 0 0 1

echo.
echo Testing complete!
echo Results saved to: test_results\
goto end

:scenario7
echo.
echo ==========================================
echo Visualize and Generate Synthetic Vessels
echo ==========================================
echo.

REM Find the most recent model
set MODEL_PATH=
if exist "outputs\vae_small_dataset\best_vae.pth" (
    set MODEL_PATH=outputs\vae_small_dataset\best_vae.pth
) else if exist "outputs\vae_medium_dataset\best_vae.pth" (
    set MODEL_PATH=outputs\vae_medium_dataset\best_vae.pth
) else if exist "outputs\vae_large_dataset\best_vae.pth" (
    set MODEL_PATH=outputs\vae_large_dataset\best_vae.pth
) else (
    echo WARNING: No trained model found!
    echo Please train a model first (options 1-4^)
    pause
    exit /b 1
)

echo Using model: %MODEL_PATH%
echo.
python phase1\scripts\visualize_vae.py ^
    --model_path %MODEL_PATH% ^
    --data_dir ./data/vessel_masks ^
    --output_dir ./visualizations ^
    --num_samples 10 ^
    --num_reconstructions 5 ^
    --save_format nrrd

echo.
echo Visualization complete!
echo Results saved to: visualizations\
echo.
echo Generated files:
echo   - synthetic_samples\ (synthetic vessel masks^)
echo   - reconstructions\ (real vs reconstructed^)
goto end

:invalid
echo Invalid choice. Please run the script again and choose 1-7.
pause
exit /b 1

:end
echo.
echo ==========================================
echo Next Steps:
echo ==========================================
echo.

if not "%choice%"=="5" if not "%choice%"=="6" if not "%choice%"=="7" (
    echo 1. Monitor training with TensorBoard:
    echo    tensorboard --logdir ./outputs
    echo.
    echo 2. After training, test the model:
    echo    phase1\quickstart_vae.bat
    echo    (Choose option 6^)
    echo.
    echo 3. Generate synthetic vessels:
    echo    phase1\quickstart_vae.bat
    echo    (Choose option 7^)
)

echo.
echo For more details, see:
echo   - phase1\docs\QUICKSTART_VAE.md
echo   - phase1\docs\PHASE1_STEP1_SUMMARY.md
echo   - phase1\docs\README_PHASE1.md
echo.
pause

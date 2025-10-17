@echo off
REM alpha-beta-CROWN Setup Script (Windows)
REM
REM This script automatically sets up alpha-beta-CROWN for neural network verification.
REM It handles submodule initialization, dependency installation, and environment configuration.
REM
REM Usage:
REM   setup_abcrown.bat                  Full setup
REM   setup_abcrown.bat --test           Full setup + test verification
REM   setup_abcrown.bat --skip-deps      Skip dependency installation
REM
REM Requirements:
REM   - Python 3.9+ (3.11 recommended)
REM   - PyTorch 2.0+ (should be pre-installed)
REM   - CUDA 12.1+ (for GPU acceleration, optional)

setlocal enabledelayedexpansion

REM Parse arguments
set SKIP_DEPS=false
set RUN_TEST=false
:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--skip-deps" (
    set SKIP_DEPS=true
    shift
    goto :parse_args
)
if /i "%~1"=="--test" (
    set RUN_TEST=true
    shift
    goto :parse_args
)
echo Unknown option: %~1
echo Usage: %0 [--skip-deps] [--test]
exit /b 1
:end_parse_args

REM Get project root (4 levels up from this script)
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%..\..\.."
set PROJECT_ROOT=%CD%

echo ========================================
echo alpha-beta-CROWN Setup Script
echo ========================================
echo.
echo Project root: %PROJECT_ROOT%
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+ first.
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected

REM Check PyTorch installation
echo [2/6] Checking PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo Error: PyTorch not found. Please install PyTorch 2.0+ first:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    exit /b 1
)

for /f "delims=" %%i in ('python -c "import torch; print(torch.__version__)"') do set TORCH_VERSION=%%i
echo PyTorch %TORCH_VERSION% detected

REM Check CUDA availability
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo CUDA not available, will use CPU ^(verification will be slower^)
) else (
    for /f "delims=" %%i in ('python -c "import torch; print(torch.version.cuda)"') do set CUDA_VERSION=%%i
    echo CUDA !CUDA_VERSION! available
)

REM Initialize/update git submodules
echo [3/6] Initializing alpha-beta-CROWN submodule...
cd /d "%PROJECT_ROOT%"

if not exist "modules\alpha-beta-CROWN\.git" (
    echo Initializing submodule for the first time...
    git submodule update --init --recursive modules/alpha-beta-CROWN
    if errorlevel 1 (
        echo Error: Failed to initialize submodule
        exit /b 1
    )
    echo Submodule initialized
) else (
    echo Submodule already initialized, updating...
    git submodule update --remote --merge modules/alpha-beta-CROWN
    echo Submodule updated
)

REM Check that auto_LiRPA submodule is also initialized
if not exist "modules\alpha-beta-CROWN\auto_LiRPA\.git" (
    echo Initializing auto_LiRPA submodule...
    cd /d "%PROJECT_ROOT%\modules\alpha-beta-CROWN"
    git submodule update --init --recursive
    cd /d "%PROJECT_ROOT%"
    echo auto_LiRPA submodule initialized
)

REM Install dependencies
if "%SKIP_DEPS%"=="false" (
    echo [4/6] Installing dependencies...

    REM Install auto_LiRPA in development mode
    echo Installing auto_LiRPA...
    cd /d "%PROJECT_ROOT%\modules\alpha-beta-CROWN\auto_LiRPA"
    pip install -e . --quiet
    cd /d "%PROJECT_ROOT%"
    echo auto_LiRPA installed

    REM Install alpha-beta-CROWN requirements
    echo Installing alpha-beta-CROWN requirements...
    pip install -r "%PROJECT_ROOT%\modules\alpha-beta-CROWN\complete_verifier\requirements.txt" --quiet
    echo alpha-beta-CROWN requirements installed

    REM Install additional utilities
    echo Installing additional utilities...
    pip install appdirs pyyaml --quiet
    echo Additional utilities installed
) else (
    echo [4/6] Skipping dependency installation ^(--skip-deps^)
)

REM Verify installation
echo [5/6] Verifying installation...
cd /d "%PROJECT_ROOT%\modules\alpha-beta-CROWN\complete_verifier"
set PYTHONPATH=..\auto_LiRPA;%PYTHONPATH%

python -c "import sys; sys.path.insert(0, '../auto_LiRPA'); import auto_LiRPA; print(f'auto_LiRPA version {auto_LiRPA.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo Error: Cannot import auto_LiRPA
    exit /b 1
)
echo auto_LiRPA can be imported

python -c "import onnx, onnxruntime; print('ONNX runtime available')" >nul 2>&1
if errorlevel 1 (
    echo Error: ONNX runtime not available
    exit /b 1
)
echo ONNX runtime available

REM Create necessary directories
echo [6/6] Creating artifact directories...
cd /d "%PROJECT_ROOT%"
if not exist "artifacts\abcrown_models" mkdir artifacts\abcrown_models
if not exist "artifacts\abcrown_configs" mkdir artifacts\abcrown_configs
if not exist "artifacts\vnnlib_specs" mkdir artifacts\vnnlib_specs
echo Directories created

REM Run test verification if requested
if "%RUN_TEST%"=="true" (
    echo.
    echo ========================================
    echo Running Test Verification
    echo ========================================
    echo.

    REM Check if we have a test model
    cd /d "%PROJECT_ROOT%"
    set TEST_MODEL=
    for /f "delims=" %%i in ('dir /b /s artifacts\*small_cnn_best*.pth 2^>nul ^| findstr /v /c:"checkpoint"') do (
        set TEST_MODEL=%%i
        goto :found_model
    )
    for /f "delims=" %%i in ('dir /b /s artifacts\*tiny_cnn_best*.pth 2^>nul ^| findstr /v /c:"checkpoint"') do (
        set TEST_MODEL=%%i
        goto :found_model
    )

    :found_model
    if "!TEST_MODEL!"=="" (
        echo No trained model found for testing.
        echo To test verification, first train a model:
        echo   python train.py --dataset GTSRB --model_arch small_cnn --epochs 10
    ) else (
        echo Found test model: !TEST_MODEL!

        REM Determine dataset from filename
        set DATASET=GTSRB
        echo !TEST_MODEL! | findstr /i "gtsrb" >nul && set DATASET=GTSRB
        echo !TEST_MODEL! | findstr /i "cifar" >nul && set DATASET=CIFAR10
        echo !TEST_MODEL! | findstr /i "mnist" >nul && set DATASET=MNIST

        echo Running verification on 5 test images...
        python src\Formal_Neural_Network_Verification\alpha-beta-crown\verify_expert_abcrown.py ^
            --model_path "!TEST_MODEL!" ^
            --dataset !DATASET! ^
            --epsilon 0.00784 ^
            --num_images 5 ^
            --timeout 60 ^
            --auto_run
    )
)

REM Print success message and instructions
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo alpha-beta-CROWN is now ready to use.
echo.
echo Quick start:
echo   1. Export and verify a trained model:
echo      python src\Formal_Neural_Network_Verification\alpha-beta-crown\verify_expert_abcrown.py ^
echo        --model_path artifacts\gtsrb_small_cnn_best.pth ^
echo        --dataset GTSRB ^
echo        --epsilon 0.00784 ^
echo        --num_images 10 ^
echo        --auto_run
echo.
echo   2. Or run verification manually:
echo      cd modules\alpha-beta-CROWN\complete_verifier
echo      set PYTHONPATH=..\auto_LiRPA;%%PYTHONPATH%%
echo      python abcrown.py --config ^<path-to-config.yaml^>
echo.
echo Documentation:
echo   - Quick start: src\Formal_Neural_Network_Verification\alpha-beta-crown\ABCROWN_QUICKSTART.md
echo   - Project docs: CLAUDE.md
echo   - Official docs: modules\alpha-beta-CROWN\README.md
echo.

endlocal

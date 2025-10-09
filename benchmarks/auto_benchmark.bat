@echo off
REM Automated benchmark script for Windows
REM Runs training and testing across multiple datasets

setlocal EnableDelayedExpansion

REM ==================== CONFIGURATION ====================
set EPOCHS=100
set RUNS=3
set MODEL_ARCH=small_cnn
set DEVICE=cuda
set DATASETS=GTSRB CIFAR10 MNIST
set ATTACK_MODE=PGD

REM Parse command-line arguments
:parse_args
if "%1"=="" goto start_benchmark
if /i "%1"=="--epochs" (
    set EPOCHS=%2
    shift
    shift
    goto parse_args
)
if /i "%1"=="--runs" (
    set RUNS=%2
    shift
    shift
    goto parse_args
)
if /i "%1"=="--model_arch" (
    set MODEL_ARCH=%2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:start_benchmark
echo ================================================================================
echo AUTOMATED BENCHMARK TESTING
echo ================================================================================
echo Datasets: %DATASETS%
echo Model Architecture: %MODEL_ARCH%
echo Epochs per run: %EPOCHS%
echo Runs per configuration: %RUNS%
echo Attack mode: %ATTACK_MODE%
echo Device: %DEVICE%
echo ================================================================================
echo.

REM Create results directory
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set RESULTS_DIR=artifacts\benchmark_results\run_%TIMESTAMP%
mkdir "%RESULTS_DIR%" 2>nul

echo Results will be saved to: %RESULTS_DIR%
echo.

REM Create results CSV header
echo Dataset,Training_Type,Run,Clean_Acc,Adv_Acc > "%RESULTS_DIR%\results.csv"

REM Loop through datasets
for %%D in (%DATASETS%) do (
    echo.
    echo ################################################################################
    echo DATASET: %%D
    echo ################################################################################
    echo.

    REM Clean Training (NAT)
    echo ----------------------------------------
    echo [1/2] Natural (Clean) Training
    echo ----------------------------------------
    for /L %%R in (1,1,%RUNS%) do (
        echo.
        echo === Run %%R/%RUNS% ===
        echo Training %%D with clean (natural) training...

        python src\Vision_Transformer_Pytorch\train_moe.py ^
            --dataset %%D ^
            --model_arch %MODEL_ARCH% ^
            --epochs %EPOCHS% ^
            --device %DEVICE% ^
            > "%RESULTS_DIR%\%%D_NAT_run%%R.log" 2>&1

        if errorlevel 1 (
            echo ERROR: Training failed! Check %RESULTS_DIR%\%%D_NAT_run%%R.log
        ) else (
            echo Training completed successfully

            REM Run adversarial test
            echo Testing with %ATTACK_MODE% attack...
            python src\Vision_Transformer_Pytorch\train_moe.py ^
                --dataset %%D ^
                --model_arch %MODEL_ARCH% ^
                --art_attack ^
                --art_attack_mode %ATTACK_MODE% ^
                --device %DEVICE% ^
                >> "%RESULTS_DIR%\%%D_NAT_run%%R.log" 2>&1

            echo Test completed
        )
    )

    REM Adversarial Training (AT)
    echo.
    echo ----------------------------------------
    echo [2/2] Adversarial Training
    echo ----------------------------------------
    for /L %%R in (1,1,%RUNS%) do (
        echo.
        echo === Run %%R/%RUNS% ===
        echo Training %%D with adversarial training...

        python src\Vision_Transformer_Pytorch\train_moe.py ^
            --dataset %%D ^
            --model_arch %MODEL_ARCH% ^
            --epochs %EPOCHS% ^
            --device %DEVICE% ^
            --adv_training ^
            > "%RESULTS_DIR%\%%D_AT_run%%R.log" 2>&1

        if errorlevel 1 (
            echo ERROR: Training failed! Check %RESULTS_DIR%\%%D_AT_run%%R.log
        ) else (
            echo Training completed successfully

            REM Run adversarial test
            echo Testing with %ATTACK_MODE% attack...
            python src\Vision_Transformer_Pytorch\train_moe.py ^
                --dataset %%D ^
                --model_arch %MODEL_ARCH% ^
                --art_attack ^
                --art_attack_mode %ATTACK_MODE% ^
                --device %DEVICE% ^
                >> "%RESULTS_DIR%\%%D_AT_run%%R.log" 2>&1

            echo Test completed
        )
    )
)

echo.
echo ================================================================================
echo BENCHMARK COMPLETED
echo ================================================================================
echo Results saved to: %RESULTS_DIR%
echo.
echo To analyze results, run:
echo     python analyze_benchmark.py "%RESULTS_DIR%"
echo.

endlocal

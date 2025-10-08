@echo off
REM Installation script for MoE Test Suite dependencies (Windows)

echo ========================================
echo MoE Test Suite - Dependency Installer
echo ========================================
echo.

REM Check if Go is installed
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Go is not installed!
    echo.
    echo Please install Go from: https://golang.org/dl/
    echo Recommended version: Go 1.21 or higher
    echo.
    pause
    exit /b 1
)

echo [1/5] Checking Go installation...
go version
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Python is not found in PATH
    echo You may need Python for running model inference
    echo.
) else (
    echo [2/5] Checking Python installation...
    python --version
    echo.
)

echo [3/5] Initializing Go module...
go mod tidy
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to initialize Go module
    pause
    exit /b 1
)
echo.

echo [4/5] Installing Go dependencies...
echo   - Installing gofpdf (PDF generation)...
go get github.com/jung-kurt/gofpdf
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install gofpdf
    pause
    exit /b 1
)

echo   - Installing go-chart (visualization)...
go get github.com/wcharczuk/go-chart/v2
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install go-chart
    pause
    exit /b 1
)

echo   - Installing testify (testing utilities)...
go get github.com/stretchr/testify/assert
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install testify
    pause
    exit /b 1
)
echo.

echo [5/5] Verifying installation...
go mod tidy
go build ./cmd/report_generator
go build ./cmd/test_runner
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build verification failed
    pause
    exit /b 1
)
echo.

REM Clean up build artifacts
del report_generator.exe 2>nul
del test_runner.exe 2>nul

echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Installed packages:
echo   - github.com/jung-kurt/gofpdf
echo   - github.com/wcharczuk/go-chart/v2
echo   - github.com/stretchr/testify
echo.
echo Next steps:
echo   1. Create test images: python scripts\create_test_image.py
echo   2. Run tests: run_tests.bat
echo   3. Or use: go run cmd\test_runner\main.go
echo.
echo Documentation:
echo   - Quick start: QUICKSTART.md
echo   - Full guide: README.md
echo.

pause

@echo off
REM Setup script for alpha-beta-CROWN environment
REM This script sets the PYTHONPATH to include auto_LiRPA for alpha-beta-CROWN to work

SET ABCROWN_ROOT=%~dp0modules\alpha-beta-CROWN
SET PYTHONPATH=%ABCROWN_ROOT%\auto_LiRPA;%PYTHONPATH%

echo ========================================
echo alpha-beta-CROWN Environment Setup
echo ========================================
echo ABCROWN_ROOT: %ABCROWN_ROOT%
echo PYTHONPATH: %PYTHONPATH%
echo ========================================
echo.
echo Environment configured successfully!
echo You can now run: python run_router_verification.py
echo.

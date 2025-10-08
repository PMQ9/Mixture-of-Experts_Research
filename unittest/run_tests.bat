@echo off
REM Windows batch script to run MoE tests and generate PDF report

echo ========================================
echo MoE Research Test Suite
echo ========================================
echo.

REM Check if Go is installed
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Go is not installed or not in PATH
    echo Please install Go from https://golang.org/dl/
    pause
    exit /b 1
)

echo Running tests...
echo.

REM Run tests and save output
go test -v ./... > test_output.txt 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo Warning: Some tests failed
) else (
    echo All tests passed!
)

echo.
echo Generating PDF report...
echo.

REM Generate report
go run cmd/report_generator/main.go

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to generate report
    pause
    exit /b 1
)

echo.
echo ========================================
echo Tests completed!
echo.
echo Check the reports/ folder for the PDF
echo ========================================
echo.

REM Ask if user wants to open the report
set /p OPEN="Open the report now? (y/n): "
if /i "%OPEN%"=="y" (
    REM Find the most recent report
    for /f "delims=" %%i in ('dir /b /od reports\test_report_*.pdf 2^>nul') do set LATEST=%%i
    if defined LATEST (
        start "" "reports\%LATEST%"
    ) else (
        echo No report found in reports/ folder
    )
)

pause

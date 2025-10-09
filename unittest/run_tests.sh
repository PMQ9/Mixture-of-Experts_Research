#!/bin/bash
# Shell script to run MoE tests and generate PDF report

set -e

echo "========================================"
echo "MoE Research Test Suite"
echo "========================================"
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed or not in PATH"
    echo "Please install Go from https://golang.org/dl/"
    exit 1
fi

echo "Running tests..."
echo ""

# Run tests and save output
set +e
go test -v ./... | tee test_output.txt
TEST_EXIT_CODE=$?
set -e

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Warning: Some tests failed"
else
    echo "All tests passed!"
fi

echo ""
echo "Generating PDF report..."
echo ""

# Generate report
go run cmd/report_generator/main.go

echo ""
echo "========================================"
echo "Tests completed!"
echo ""
echo "Check the reports/ folder for the PDF"
echo "========================================"
echo ""

# Ask if user wants to open the report (on macOS/Linux with display)
if [[ "$OSTYPE" == "darwin"* ]]; then
    LATEST=$(ls -t reports/test_report_*.pdf 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        read -p "Open the report now? (y/n): " OPEN
        if [[ "$OPEN" =~ ^[Yy]$ ]]; then
            open "$LATEST"
        fi
    fi
elif [[ -n "$DISPLAY" ]]; then
    LATEST=$(ls -t reports/test_report_*.pdf 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        read -p "Open the report now? (y/n): " OPEN
        if [[ "$OPEN" =~ ^[Yy]$ ]]; then
            xdg-open "$LATEST" 2>/dev/null || echo "Please open $LATEST manually"
        fi
    fi
fi

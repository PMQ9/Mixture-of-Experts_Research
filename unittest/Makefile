# Makefile for MoE Research Test Suite

.PHONY: all test test-verbose test-unit test-regression report clean install help

# Default target
all: test report

# Install Go dependencies
install:
	@echo "Installing Go dependencies..."
	@echo "This will install:"
	@echo "  - github.com/jung-kurt/gofpdf (PDF generation)"
	@echo "  - github.com/wcharczuk/go-chart/v2 (visualization)"
	@echo "  - github.com/stretchr/testify (testing utilities)"
	@echo ""
	go get github.com/jung-kurt/gofpdf
	go get github.com/wcharczuk/go-chart/v2
	go get github.com/stretchr/testify/assert
	go mod tidy
	@echo ""
	@echo "✓ Dependencies installed successfully!"
	@echo "Run 'make test' to run tests or 'make full' for complete suite"

# Run all tests
test:
	@echo "Running all tests..."
	go test -v ./... | tee test_output.txt

# Run tests with verbose output
test-verbose:
	@echo "Running tests (verbose)..."
	go test -v -json ./... | tee test_output.txt

# Run only unit tests
test-unit:
	@echo "Running unit tests..."
	go test -v ./internal/models | tee test_output.txt

# Run only regression tests
test-regression:
	@echo "Running regression tests..."
	go test -v ./internal/regression | tee test_output.txt

# Run specific test by name
test-run:
	@echo "Usage: make test-run TEST=TestName"
	go test -v -run $(TEST) ./...

# Generate PDF report
report:
	@echo "Generating PDF report..."
	go run cmd/report_generator/main.go

# Run full test suite with report
full:
	@echo "Running full test suite..."
	go run cmd/test_runner/main.go

# Run tests in CI mode (for GitLab CI)
ci:
	@echo "Running tests in CI mode..."
	go test -v ./... | tee test_output.txt
	go run cmd/report_generator/main.go -output reports

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f test_output.txt
	rm -rf reports/*.pdf
	rm -rf testdata/baselines/*.json
	@echo "Clean complete!"

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Check code coverage
coverage:
	@echo "Generating coverage report..."
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Format code
fmt:
	@echo "Formatting Go code..."
	go fmt ./...

# Run linter
lint:
	@echo "Running linter..."
	golangci-lint run ./...

# Show help
help:
	@echo "MoE Research Test Suite - Available Commands:"
	@echo ""
	@echo "  make install         - Install Go dependencies"
	@echo "  make test            - Run all tests"
	@echo "  make test-verbose    - Run tests with verbose output"
	@echo "  make test-unit       - Run only unit tests"
	@echo "  make test-regression - Run only regression tests"
	@echo "  make test-run TEST=X - Run specific test"
	@echo "  make report          - Generate PDF report"
	@echo "  make full            - Run full suite with report"
	@echo "  make ci              - Run in CI mode"
	@echo "  make clean           - Clean generated files"
	@echo "  make bench           - Run benchmarks"
	@echo "  make coverage        - Generate coverage report"
	@echo "  make fmt             - Format code"
	@echo "  make lint            - Run linter"
	@echo "  make help            - Show this help"
	@echo ""

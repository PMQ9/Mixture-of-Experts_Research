package report

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/jung-kurt/gofpdf"
)

// TestResult represents a single test result
type TestResult struct {
	Name     string
	Status   string // "PASS", "FAIL", "SKIP"
	Duration time.Duration
	Output   string
	Error    string
}

// TestSuite represents a collection of test results
type TestSuite struct {
	Name      string
	Tests     []TestResult
	StartTime time.Time
	EndTime   time.Time
}

// ReportGenerator generates PDF reports from test results
type ReportGenerator struct {
	OutputDir   string
	ProjectName string
}

// NewReportGenerator creates a new report generator
func NewReportGenerator(outputDir, projectName string) *ReportGenerator {
	if outputDir == "" {
		outputDir = "reports"
	}
	if projectName == "" {
		projectName = "MoE Research Test Suite"
	}
	return &ReportGenerator{
		OutputDir:   outputDir,
		ProjectName: projectName,
	}
}

// GenerateReport creates a comprehensive PDF report
func (rg *ReportGenerator) GenerateReport(suites []TestSuite) (string, error) {
	// Ensure output directory exists
	if err := os.MkdirAll(rg.OutputDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create output directory: %w", err)
	}

	// Initialize PDF
	pdf := gofpdf.New("P", "mm", "A4", "")
	pdf.SetMargins(20, 20, 20)

	// Add cover page
	rg.addCoverPage(pdf, suites)

	// Add executive summary
	rg.addExecutiveSummary(pdf, suites)

	// Add detailed results for each suite
	for _, suite := range suites {
		rg.addSuiteResults(pdf, suite)
	}

	// Add performance metrics
	rg.addPerformanceMetrics(pdf, suites)

	// Generate filename with timestamp
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := fmt.Sprintf("test_report_%s.pdf", timestamp)
	filepath := filepath.Join(rg.OutputDir, filename)

	// Save PDF
	if err := pdf.OutputFileAndClose(filepath); err != nil {
		return "", fmt.Errorf("failed to save PDF: %w", err)
	}

	return filepath, nil
}

// addCoverPage adds the cover page to the PDF
func (rg *ReportGenerator) addCoverPage(pdf *gofpdf.Fpdf, suites []TestSuite) {
	pdf.AddPage()

	// Title
	pdf.SetFont("Arial", "B", 28)
	pdf.SetTextColor(0, 51, 102)
	pdf.CellFormat(0, 20, "", "", 1, "", false, 0, "")
	pdf.CellFormat(0, 15, rg.ProjectName, "", 1, "C", false, 0, "")

	// Subtitle
	pdf.SetFont("Arial", "", 16)
	pdf.SetTextColor(80, 80, 80)
	pdf.CellFormat(0, 8, "Automated Test Report", "", 1, "C", false, 0, "")

	// Date
	pdf.SetFont("Arial", "", 12)
	pdf.CellFormat(0, 15, time.Now().Format("January 2, 2006 15:04:05"), "", 1, "C", false, 0, "")

	// Summary box
	totalTests := 0
	passedTests := 0
	failedTests := 0
	skippedTests := 0

	for _, suite := range suites {
		for _, test := range suite.Tests {
			totalTests++
			switch test.Status {
			case "PASS":
				passedTests++
			case "FAIL":
				failedTests++
			case "SKIP":
				skippedTests++
			}
		}
	}

	pdf.SetY(pdf.GetY() + 10)
	pdf.SetFont("Arial", "B", 14)
	pdf.SetTextColor(0, 0, 0)
	pdf.CellFormat(0, 8, "Test Summary", "", 1, "L", false, 0, "")

	pdf.SetFont("Arial", "", 12)
	pdf.CellFormat(90, 10, fmt.Sprintf("Total Tests: %d", totalTests), "1", 0, "L", false, 0, "")
	pdf.CellFormat(90, 10, fmt.Sprintf("Duration: %.2fs", rg.getTotalDuration(suites).Seconds()), "1", 1, "L", false, 0, "")

	pdf.SetFillColor(200, 255, 200)
	pdf.CellFormat(90, 10, fmt.Sprintf("Passed: %d", passedTests), "1", 0, "L", true, 0, "")

	pdf.SetFillColor(255, 200, 200)
	pdf.CellFormat(90, 10, fmt.Sprintf("Failed: %d", failedTests), "1", 1, "L", true, 0, "")

	pdf.SetFillColor(255, 255, 200)
	pdf.CellFormat(90, 10, fmt.Sprintf("Skipped: %d", skippedTests), "1", 0, "L", true, 0, "")

	successRate := float64(passedTests) / float64(totalTests) * 100
	pdf.SetFillColor(255, 255, 255)
	pdf.CellFormat(90, 10, fmt.Sprintf("Success Rate: %.1f%%", successRate), "1", 1, "L", false, 0, "")
}

// addExecutiveSummary adds executive summary page
func (rg *ReportGenerator) addExecutiveSummary(pdf *gofpdf.Fpdf, suites []TestSuite) {
	pdf.AddPage()

	pdf.SetFont("Arial", "B", 18)
	pdf.SetTextColor(0, 51, 102)
	pdf.CellFormat(0, 10, "Executive Summary", "", 1, "L", false, 0, "")
	pdf.Ln(3)

	pdf.SetFont("Arial", "", 11)
	pdf.SetTextColor(0, 0, 0)

	// Overall status
	allPassed := true
	for _, suite := range suites {
		for _, test := range suite.Tests {
			if test.Status == "FAIL" {
				allPassed = false
				break
			}
		}
	}

	if allPassed {
		pdf.SetTextColor(0, 128, 0)
		pdf.MultiCell(0, 6, "Overall Status: ALL TESTS PASSED", "", "L", false)
	} else {
		pdf.SetTextColor(255, 0, 0)
		pdf.MultiCell(0, 6, "Overall Status: SOME TESTS FAILED", "", "L", false)
	}

	pdf.SetTextColor(0, 0, 0)
	pdf.Ln(3)

	// Test suite breakdown
	pdf.SetFont("Arial", "B", 12)
	pdf.CellFormat(0, 7, "Test Suite Breakdown:", "", 1, "L", false, 0, "")

	pdf.SetFont("Arial", "", 10)
	for _, suite := range suites {
		passed := 0
		failed := 0
		skipped := 0

		for _, test := range suite.Tests {
			switch test.Status {
			case "PASS":
				passed++
			case "FAIL":
				failed++
			case "SKIP":
				skipped++
			}
		}

		status := "PASS"
		if failed > 0 {
			status = "FAIL"
		}

		duration := suite.EndTime.Sub(suite.StartTime).Seconds()

		pdf.CellFormat(80, 7, fmt.Sprintf("  %s", suite.Name), "1", 0, "L", false, 0, "")
		pdf.CellFormat(30, 7, status, "1", 0, "C", false, 0, "")
		pdf.CellFormat(30, 7, fmt.Sprintf("%d/%d", passed, passed+failed), "1", 0, "C", false, 0, "")
		pdf.CellFormat(30, 7, fmt.Sprintf("%.2fs", duration), "1", 1, "C", false, 0, "")
	}
}

// addSuiteResults adds detailed results for a test suite
func (rg *ReportGenerator) addSuiteResults(pdf *gofpdf.Fpdf, suite TestSuite) {
	pdf.AddPage()

	pdf.SetFont("Arial", "B", 16)
	pdf.SetTextColor(0, 51, 102)
	pdf.CellFormat(0, 10, fmt.Sprintf("Test Suite: %s", suite.Name), "", 1, "L", false, 0, "")
	pdf.Ln(3)

	pdf.SetFont("Arial", "", 10)
	pdf.SetTextColor(0, 0, 0)

	for _, test := range suite.Tests {
		// Test name and status
		pdf.SetFont("Arial", "B", 11)

		switch test.Status {
		case "PASS":
			pdf.SetTextColor(0, 128, 0)
		case "FAIL":
			pdf.SetTextColor(255, 0, 0)
		case "SKIP":
			pdf.SetTextColor(128, 128, 0)
		}

		pdf.CellFormat(140, 7, test.Name, "", 0, "L", false, 0, "")
		pdf.CellFormat(20, 7, test.Status, "", 0, "R", false, 0, "")
		pdf.SetTextColor(0, 0, 0)
		pdf.CellFormat(20, 7, fmt.Sprintf("%.2fs", test.Duration.Seconds()), "", 1, "R", false, 0, "")

		// Test output/details
		if test.Output != "" || test.Error != "" {
			pdf.SetFont("Courier", "", 8)
			pdf.SetTextColor(80, 80, 80)

			if test.Output != "" {
				pdf.MultiCell(0, 4, test.Output, "", "L", false)
			}

			if test.Error != "" {
				pdf.SetTextColor(200, 0, 0)
				pdf.MultiCell(0, 4, fmt.Sprintf("Error: %s", test.Error), "", "L", false)
			}

			pdf.SetTextColor(0, 0, 0)
		}

		pdf.Ln(2)
	}
}

// addPerformanceMetrics adds performance metrics page
func (rg *ReportGenerator) addPerformanceMetrics(pdf *gofpdf.Fpdf, suites []TestSuite) {
	pdf.AddPage()

	pdf.SetFont("Arial", "B", 16)
	pdf.SetTextColor(0, 51, 102)
	pdf.CellFormat(0, 10, "Performance Metrics", "", 1, "L", false, 0, "")
	pdf.Ln(3)

	pdf.SetFont("Arial", "", 10)
	pdf.SetTextColor(0, 0, 0)

	// Calculate metrics
	totalDuration := rg.getTotalDuration(suites)
	totalTests := 0
	for _, suite := range suites {
		totalTests += len(suite.Tests)
	}

	avgTestDuration := totalDuration.Seconds() / float64(totalTests)

	pdf.CellFormat(90, 8, "Total Execution Time:", "", 0, "L", false, 0, "")
	pdf.CellFormat(90, 8, fmt.Sprintf("%.2f seconds", totalDuration.Seconds()), "", 1, "R", false, 0, "")

	pdf.CellFormat(90, 8, "Total Tests:", "", 0, "L", false, 0, "")
	pdf.CellFormat(90, 8, fmt.Sprintf("%d", totalTests), "", 1, "R", false, 0, "")

	pdf.CellFormat(90, 8, "Average Test Duration:", "", 0, "L", false, 0, "")
	pdf.CellFormat(90, 8, fmt.Sprintf("%.3f seconds", avgTestDuration), "", 1, "R", false, 0, "")

	pdf.Ln(3)

	// Slowest tests
	pdf.SetFont("Arial", "B", 12)
	pdf.CellFormat(0, 7, "Top 5 Slowest Tests:", "", 1, "L", false, 0, "")

	// Collect all tests
	var allTests []TestResult
	for _, suite := range suites {
		allTests = append(allTests, suite.Tests...)
	}

	// Sort by duration (simple bubble sort for top 5)
	for i := 0; i < len(allTests) && i < 5; i++ {
		for j := i + 1; j < len(allTests); j++ {
			if allTests[j].Duration > allTests[i].Duration {
				allTests[i], allTests[j] = allTests[j], allTests[i]
			}
		}
	}

	pdf.SetFont("Arial", "", 9)
	for i := 0; i < len(allTests) && i < 5; i++ {
		pdf.CellFormat(140, 6, fmt.Sprintf("%d. %s", i+1, allTests[i].Name), "", 0, "L", false, 0, "")
		pdf.CellFormat(40, 6, fmt.Sprintf("%.3fs", allTests[i].Duration.Seconds()), "", 1, "R", false, 0, "")
	}
}

// getTotalDuration calculates total duration across all suites
func (rg *ReportGenerator) getTotalDuration(suites []TestSuite) time.Duration {
	var total time.Duration
	for _, suite := range suites {
		total += suite.EndTime.Sub(suite.StartTime)
	}
	return total
}

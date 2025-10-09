package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"moe-research/unittest/internal/utils"
)

func main() {
	// Command line flags
	configPath := flag.String("config", "config.json", "Path to configuration file")
	verbose := flag.Bool("v", false, "Verbose output")
	generateReport := flag.Bool("report", true, "Generate PDF report after tests")
	skipTests := flag.Bool("skip-tests", false, "Skip running tests, only generate report")

	flag.Parse()

	fmt.Println("╔════════════════════════════════════════════╗")
	fmt.Println("║  MoE Research Test Runner                 ║")
	fmt.Println("║  Mixture-of-Experts Testing Suite         ║")
	fmt.Println("╚════════════════════════════════════════════╝")
	fmt.Println()

	// Load configuration
	config, err := utils.LoadConfig(*configPath)
	if err != nil {
		log.Printf("Warning: Failed to load config: %v", err)
		log.Println("Using default configuration")
		config = utils.DefaultConfig()
	}

	// Ensure we're in the right directory
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		log.Fatalf("Failed to find project root: %v", err)
	}

	unittestDir := filepath.Join(projectRoot, "unittest")
	if err := os.Chdir(unittestDir); err != nil {
		log.Fatalf("Failed to change to unittest directory: %v", err)
	}

	fmt.Printf("Project Root: %s\n", projectRoot)
	fmt.Printf("Working Directory: %s\n", unittestDir)
	fmt.Println()

	// Run tests
	var testOutputFile string
	if !*skipTests {
		fmt.Println("Running tests...")
		fmt.Println("──────────────────────────────────────────")

		testOutputFile = runTests(*verbose)

		fmt.Println()
		fmt.Println("Tests completed!")
		fmt.Println()
	} else {
		testOutputFile = "test_output.txt"
		fmt.Println("Skipping test execution (using existing test_output.txt)")
	}

	// Generate report
	if *generateReport {
		fmt.Println("Generating PDF report...")
		fmt.Println("──────────────────────────────────────────")

		reportPath, err := generatePDFReport(testOutputFile, config.ReportOutputDir)
		if err != nil {
			log.Fatalf("Failed to generate report: %v", err)
		}

		fmt.Println()
		fmt.Printf("✓ Report generated successfully!\n")
		fmt.Printf("  Location: %s\n", reportPath)
		fmt.Println()

		// Open report (optional)
		fmt.Println("To view the report:")
		fmt.Printf("  start %s\n", reportPath)
	}

	fmt.Println("╔════════════════════════════════════════════╗")
	fmt.Println("║  Test run completed successfully!         ║")
	fmt.Println("╚════════════════════════════════════════════╝")
}

// runTests executes the test suite
func runTests(verbose bool) string {
	outputFile := "test_output.txt"

	// Prepare test command
	args := []string{"test", "-v"}

	// Add all packages
	args = append(args, "./...")

	cmd := exec.Command("go", args...)

	// Set up output capture
	f, err := os.Create(outputFile)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer f.Close()

	if verbose {
		// Output to both file and console
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	} else {
		// Output to file only, show progress dots
		cmd.Stdout = f
		cmd.Stderr = f
		fmt.Print("Running: ")
		go func() {
			for {
				time.Sleep(500 * time.Millisecond)
				fmt.Print(".")
			}
		}()
	}

	// Run tests
	startTime := time.Now()
	err = cmd.Run()
	duration := time.Since(startTime)

	if !verbose {
		fmt.Println() // New line after progress dots
	}

	// Note: go test returns non-zero exit code if tests fail
	// but we still want to generate the report
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			log.Fatalf("Failed to run tests: %v", err)
		}
		fmt.Printf("⚠ Some tests failed (duration: %.2fs)\n", duration.Seconds())
	} else {
		fmt.Printf("✓ All tests passed (duration: %.2fs)\n", duration.Seconds())
	}

	return outputFile
}

// generatePDFReport generates a PDF report from test output
func generatePDFReport(inputFile, outputDir string) (string, error) {
	cmd := exec.Command("go", "run", "./cmd/report_generator",
		"-input", inputFile,
		"-output", outputDir)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("report generation failed: %w\nOutput: %s", err, string(output))
	}

	// Parse output to get report path
	// The report generator prints "Report generated: <path>"
	outputStr := string(output)
	if verbose := false; verbose {
		fmt.Println(outputStr)
	}

	// Find the report file (most recent in output directory)
	files, err := filepath.Glob(filepath.Join(outputDir, "test_report_*.pdf"))
	if err != nil || len(files) == 0 {
		return "", fmt.Errorf("report file not found in %s", outputDir)
	}

	// Get most recent file
	mostRecent := files[0]
	var mostRecentTime time.Time

	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil {
			continue
		}
		if info.ModTime().After(mostRecentTime) {
			mostRecent = file
			mostRecentTime = info.ModTime()
		}
	}

	return mostRecent, nil
}

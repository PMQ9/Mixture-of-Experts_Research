package regression_test

import (
	"os"
	"path/filepath"
	"testing"

	"moe-research/unittest/internal/regression"
	"moe-research/unittest/internal/utils"
)

func TestRegressionBaseline(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	bridge := utils.NewPythonBridge("python", projectRoot)
	baselineDir := filepath.Join(projectRoot, "unittest", "testdata", "baselines")
	manager := regression.NewBaselineManager(baselineDir)

	testImagePath := filepath.Join(projectRoot, "unittest", "testdata", "images", "test_sample.png")
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("Test image not found, skipping regression test")
	}

	modelPath := filepath.Join(projectRoot, "artifacts", "results", "gtsrb_small_cnn_best.pth")
	if err := utils.CheckModelExists(modelPath); err != nil {
		t.Skipf("Model not found: %v", err)
	}

	t.Run("CreateBaseline", func(t *testing.T) {
		// Run inference
		output, err := bridge.RunInference(modelPath, testImagePath, "single")
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}

		// Save as baseline
		err = manager.SaveBaseline(modelPath, testImagePath, output, "single")
		if err != nil {
			t.Fatalf("Failed to save baseline: %v", err)
		}

		t.Log("Baseline created successfully")
	})

	t.Run("CompareWithBaseline", func(t *testing.T) {
		// Load baseline
		baseline, err := manager.LoadBaseline(modelPath, testImagePath)
		if err != nil {
			t.Skipf("Baseline not found (run CreateBaseline first): %v", err)
		}

		// Run current inference
		current, err := bridge.RunInference(modelPath, testImagePath, "single")
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}

		// Compare
		result := manager.CompareWithBaseline(baseline, current, 1e-5)

		if !result.Passed {
			t.Errorf("Regression test failed:\n%s", result.GetSummary())
		} else {
			t.Log("Regression test passed: output matches baseline")
		}
	})
}

func TestRegressionMetaMoE(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	bridge := utils.NewPythonBridge("python", projectRoot)
	baselineDir := filepath.Join(projectRoot, "unittest", "testdata", "baselines")
	manager := regression.NewBaselineManager(baselineDir)

	testImagePath := filepath.Join(projectRoot, "unittest", "testdata", "images", "test_sample.png")
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("Test image not found, skipping MetaMoE regression test")
	}

	modelPath := filepath.Join(projectRoot, "artifacts", "meta_moe_small_cnn_best.pth")
	if err := utils.CheckModelExists(modelPath); err != nil {
		t.Skipf("MetaMoE model not found: %v", err)
	}

	t.Run("MetaMoERegression", func(t *testing.T) {
		// Run inference
		output, err := bridge.RunInference(modelPath, testImagePath, "meta")
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}

		// Try to load existing baseline, or create new one
		baseline, err := manager.LoadBaseline(modelPath, testImagePath)
		if err != nil {
			// Create new baseline
			t.Log("Creating new baseline for MetaMoE")
			err = manager.SaveBaseline(modelPath, testImagePath, output, "meta")
			if err != nil {
				t.Fatalf("Failed to save baseline: %v", err)
			}
			t.Skip("Baseline created, run test again to compare")
		}

		// Compare with baseline
		result := manager.CompareWithBaseline(baseline, output, 1e-5)

		if !result.Passed {
			t.Errorf("MetaMoE regression test failed:\n%s", result.GetSummary())
		} else {
			t.Log("MetaMoE regression test passed")
		}
	})
}

func TestMultipleImages(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	bridge := utils.NewPythonBridge("python", projectRoot)
	baselineDir := filepath.Join(projectRoot, "unittest", "testdata", "baselines")
	manager := regression.NewBaselineManager(baselineDir)

	modelPath := filepath.Join(projectRoot, "artifacts", "results", "gtsrb_small_cnn_best.pth")
	if err := utils.CheckModelExists(modelPath); err != nil {
		t.Skipf("Model not found: %v", err)
	}

	testImagesDir := filepath.Join(projectRoot, "unittest", "testdata", "images")
	images, err := os.ReadDir(testImagesDir)
	if err != nil || len(images) == 0 {
		t.Skip("No test images found")
	}

	passCount := 0
	failCount := 0

	for _, imgFile := range images {
		if imgFile.IsDir() {
			continue
		}

		imgPath := filepath.Join(testImagesDir, imgFile.Name())

		t.Run(imgFile.Name(), func(t *testing.T) {
			// Run inference
			output, err := bridge.RunInference(modelPath, imgPath, "single")
			if err != nil {
				t.Logf("Inference failed: %v", err)
				failCount++
				return
			}

			// Load or create baseline
			baseline, err := manager.LoadBaseline(modelPath, imgPath)
			if err != nil {
				// Create new baseline
				err = manager.SaveBaseline(modelPath, imgPath, output, "single")
				if err != nil {
					t.Logf("Failed to save baseline: %v", err)
					failCount++
					return
				}
				t.Log("Baseline created")
				passCount++
				return
			}

			// Compare
			result := manager.CompareWithBaseline(baseline, output, 1e-5)
			if !result.Passed {
				t.Errorf("Regression failed:\n%s", result.GetSummary())
				failCount++
			} else {
				passCount++
			}
		})
	}

	t.Logf("Regression summary: %d passed, %d failed", passCount, failCount)
}

func TestBaselineManager(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	baselineDir := filepath.Join(projectRoot, "unittest", "testdata", "baselines")
	manager := regression.NewBaselineManager(baselineDir)

	t.Run("ListBaselines", func(t *testing.T) {
		baselines, err := manager.ListBaselines()
		if err != nil {
			t.Fatalf("Failed to list baselines: %v", err)
		}

		t.Logf("Found %d baselines:", len(baselines))
		for _, baseline := range baselines {
			t.Logf("  - %s", baseline)
		}
	})
}

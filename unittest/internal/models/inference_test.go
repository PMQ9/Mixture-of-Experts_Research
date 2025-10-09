package models_test

import (
	"os"
	"path/filepath"
	"testing"

	"moe-research/unittest/internal/utils"
)

func TestModelInferenceBasic(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	bridge := utils.NewPythonBridge("python", projectRoot)

	// Test with a sample image (you'll need to create test images)
	testImagePath := filepath.Join(projectRoot, "unittest", "testdata", "images", "test_sample.png")

	// Skip if test image doesn't exist
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("Test image not found, skipping inference test")
	}

	// Find a model to test
	modelPath := filepath.Join(projectRoot, "artifacts", "results", "gtsrb_small_cnn_best.pth")
	if err := utils.CheckModelExists(modelPath); err != nil {
		t.Skipf("Model not found: %v", err)
	}

	t.Run("SingleExpertInference", func(t *testing.T) {
		result, err := bridge.RunInference(modelPath, testImagePath, "single")
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}

		// Validate output
		if result.Class < 0 {
			t.Errorf("Invalid class prediction: %d", result.Class)
		}

		if result.Confidence < 0 || result.Confidence > 1 {
			t.Errorf("Invalid confidence value: %f", result.Confidence)
		}

		if len(result.Predictions) == 0 {
			t.Error("No predictions returned")
		}

		if result.InferenceMs <= 0 {
			t.Error("Invalid inference time")
		}

		t.Logf("Predicted class: %d, Confidence: %.4f, Time: %.2fms",
			result.Class, result.Confidence, result.InferenceMs)
	})
}

func TestMetaMoEInference(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	bridge := utils.NewPythonBridge("python", projectRoot)

	testImagePath := filepath.Join(projectRoot, "unittest", "testdata", "images", "test_sample.png")
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("Test image not found, skipping MetaMoE test")
	}

	modelPath := filepath.Join(projectRoot, "artifacts", "meta_moe_small_cnn_best.pth")
	if err := utils.CheckModelExists(modelPath); err != nil {
		t.Skipf("MetaMoE model not found: %v", err)
	}

	t.Run("MetaMoERouting", func(t *testing.T) {
		result, err := bridge.RunInference(modelPath, testImagePath, "meta")
		if err != nil {
			t.Fatalf("MetaMoE inference failed: %v", err)
		}

		// Validate router info
		if len(result.RouterInfo) == 0 {
			t.Error("No router information returned from MetaMoE")
		}

		// Router weights should sum to approximately 1.0
		sum := 0.0
		for _, weight := range result.RouterInfo {
			sum += weight
		}

		if sum < 0.99 || sum > 1.01 {
			t.Errorf("Router weights don't sum to 1.0: got %f", sum)
		}

		t.Logf("Router weights: %v", result.RouterInfo)
		t.Logf("Selected class: %d, Confidence: %.4f", result.Class, result.Confidence)
	})
}

func TestModelValidation(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	t.Run("ModelFileExists", func(t *testing.T) {
		modelPath := filepath.Join(projectRoot, "artifacts", "results", "gtsrb_small_cnn_best.pth")
		if err := utils.CheckModelExists(modelPath); err != nil {
			t.Skipf("Skipping: %v", err)
		}
	})

	t.Run("InvalidModelPath", func(t *testing.T) {
		invalidPath := filepath.Join(projectRoot, "nonexistent_model.pth")
		if err := utils.CheckModelExists(invalidPath); err == nil {
			t.Error("Expected error for nonexistent model, got nil")
		}
	})
}

func TestInferencePerformance(t *testing.T) {
	projectRoot, err := utils.GetProjectRoot()
	if err != nil {
		t.Fatalf("Failed to get project root: %v", err)
	}

	bridge := utils.NewPythonBridge("python", projectRoot)

	testImagePath := filepath.Join(projectRoot, "unittest", "testdata", "images", "test_sample.png")
	if _, err := os.Stat(testImagePath); os.IsNotExist(err) {
		t.Skip("Test image not found, skipping performance test")
	}

	modelPath := filepath.Join(projectRoot, "artifacts", "results", "gtsrb_small_cnn_best.pth")
	if err := utils.CheckModelExists(modelPath); err != nil {
		t.Skipf("Model not found: %v", err)
	}

	const numRuns = 5
	var totalTime float64

	for i := 0; i < numRuns; i++ {
		result, err := bridge.RunInference(modelPath, testImagePath, "single")
		if err != nil {
			t.Fatalf("Inference failed on run %d: %v", i+1, err)
		}
		totalTime += result.InferenceMs
	}

	avgTime := totalTime / numRuns
	t.Logf("Average inference time over %d runs: %.2f ms", numRuns, avgTime)

	// Performance threshold (adjust based on your hardware)
	const maxAvgTime = 1000.0 // 1 second
	if avgTime > maxAvgTime {
		t.Errorf("Average inference time (%.2f ms) exceeds threshold (%.2f ms)",
			avgTime, maxAvgTime)
	}
}

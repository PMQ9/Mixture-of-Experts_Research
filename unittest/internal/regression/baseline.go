package regression

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"moe-research/unittest/internal/utils"
)

// Baseline represents a saved baseline output for regression testing
type Baseline struct {
	ModelPath   string             `json:"model_path"`
	ImagePath   string             `json:"image_path"`
	ModelType   string             `json:"model_type"`
	Predictions []float64          `json:"predictions"`
	Class       int                `json:"class"`
	Confidence  float64            `json:"confidence"`
	RouterInfo  map[string]float64 `json:"router_info,omitempty"`
	Timestamp   time.Time          `json:"timestamp"`
	GitCommit   string             `json:"git_commit,omitempty"`
}

// BaselineManager manages baseline outputs for regression testing
type BaselineManager struct {
	BaselineDir string
}

// NewBaselineManager creates a new baseline manager
func NewBaselineManager(baselineDir string) *BaselineManager {
	if baselineDir == "" {
		baselineDir = "testdata/baselines"
	}
	return &BaselineManager{BaselineDir: baselineDir}
}

// SaveBaseline saves a model output as a baseline
func (bm *BaselineManager) SaveBaseline(modelPath, imagePath string, output *utils.ModelOutput, modelType string) error {
	// Ensure baseline directory exists
	if err := os.MkdirAll(bm.BaselineDir, 0755); err != nil {
		return fmt.Errorf("failed to create baseline directory: %w", err)
	}

	baseline := Baseline{
		ModelPath:   modelPath,
		ImagePath:   imagePath,
		ModelType:   modelType,
		Predictions: output.Predictions,
		Class:       output.Class,
		Confidence:  output.Confidence,
		RouterInfo:  output.RouterInfo,
		Timestamp:   time.Now(),
	}

	// Generate baseline filename
	baselineName := bm.getBaselineName(modelPath, imagePath)
	baselinePath := filepath.Join(bm.BaselineDir, baselineName)

	// Save to JSON
	data, err := json.MarshalIndent(baseline, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal baseline: %w", err)
	}

	if err := os.WriteFile(baselinePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write baseline: %w", err)
	}

	return nil
}

// LoadBaseline loads a baseline from file
func (bm *BaselineManager) LoadBaseline(modelPath, imagePath string) (*Baseline, error) {
	baselineName := bm.getBaselineName(modelPath, imagePath)
	baselinePath := filepath.Join(bm.BaselineDir, baselineName)

	data, err := os.ReadFile(baselinePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read baseline: %w", err)
	}

	var baseline Baseline
	if err := json.Unmarshal(data, &baseline); err != nil {
		return nil, fmt.Errorf("failed to unmarshal baseline: %w", err)
	}

	return &baseline, nil
}

// CompareWithBaseline compares current output with baseline
func (bm *BaselineManager) CompareWithBaseline(baseline *Baseline, current *utils.ModelOutput, tolerance float64) *RegressionResult {
	result := &RegressionResult{
		ModelPath:    baseline.ModelPath,
		ImagePath:    baseline.ImagePath,
		Passed:       true,
		Tolerance:    tolerance,
		Differences:  make(map[string]float64),
		BaselineTime: baseline.Timestamp,
		CurrentTime:  time.Now(),
	}

	// Compare class predictions
	if baseline.Class != current.Class {
		result.Passed = false
		result.Differences["class_mismatch"] = float64(current.Class - baseline.Class)
	}

	// Compare confidence
	confidenceDiff := math.Abs(baseline.Confidence - current.Confidence)
	if confidenceDiff > tolerance {
		result.Passed = false
		result.Differences["confidence_diff"] = confidenceDiff
	}

	// Compare predictions (logits)
	maxDiff := 0.0
	for i, baselineVal := range baseline.Predictions {
		if i >= len(current.Predictions) {
			result.Passed = false
			result.Differences["predictions_length_mismatch"] = float64(len(current.Predictions) - len(baseline.Predictions))
			break
		}
		diff := math.Abs(baselineVal - current.Predictions[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > tolerance {
		result.Passed = false
		result.Differences["max_prediction_diff"] = maxDiff
	}

	// Compare router info (if available)
	if len(baseline.RouterInfo) > 0 {
		for expert, baselineWeight := range baseline.RouterInfo {
			if currentWeight, exists := current.RouterInfo[expert]; exists {
				diff := math.Abs(baselineWeight - currentWeight)
				if diff > tolerance {
					result.Passed = false
					result.Differences[fmt.Sprintf("router_%s_diff", expert)] = diff
				}
			} else {
				result.Passed = false
				result.Differences[fmt.Sprintf("router_%s_missing", expert)] = 1.0
			}
		}
	}

	return result
}

// getBaselineName generates a unique baseline filename
func (bm *BaselineManager) getBaselineName(modelPath, imagePath string) string {
	modelBase := filepath.Base(modelPath)
	imageBase := filepath.Base(imagePath)

	// Remove extensions
	modelName := modelBase[:len(modelBase)-len(filepath.Ext(modelBase))]
	imageName := imageBase[:len(imageBase)-len(filepath.Ext(imageBase))]

	return fmt.Sprintf("%s_%s_baseline.json", modelName, imageName)
}

// ListBaselines returns all available baselines
func (bm *BaselineManager) ListBaselines() ([]string, error) {
	files, err := os.ReadDir(bm.BaselineDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var baselines []string
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			baselines = append(baselines, file.Name())
		}
	}

	return baselines, nil
}

// RegressionResult holds the result of a regression test
type RegressionResult struct {
	ModelPath    string
	ImagePath    string
	Passed       bool
	Tolerance    float64
	Differences  map[string]float64
	BaselineTime time.Time
	CurrentTime  time.Time
}

// GetSummary returns a human-readable summary
func (rr *RegressionResult) GetSummary() string {
	if rr.Passed {
		return "PASSED: Output matches baseline within tolerance"
	}

	summary := "FAILED: Output differs from baseline:\n"
	for key, diff := range rr.Differences {
		summary += fmt.Sprintf("  - %s: %.6f\n", key, diff)
	}
	return summary
}

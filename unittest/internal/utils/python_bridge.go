package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// PythonBridge handles communication with Python scripts
type PythonBridge struct {
	PythonExec  string
	ProjectRoot string
}

// ModelOutput represents the output from a model inference
type ModelOutput struct {
	Predictions []float64          `json:"predictions"`
	Class       int                `json:"class"`
	Confidence  float64            `json:"confidence"`
	RouterInfo  map[string]float64 `json:"router_info,omitempty"`
	InferenceMs float64            `json:"inference_ms"`
}

// NewPythonBridge creates a new Python bridge
func NewPythonBridge(pythonExec, projectRoot string) *PythonBridge {
	if pythonExec == "" {
		pythonExec = "python"
	}
	if projectRoot == "" {
		projectRoot = ".."
	}
	return &PythonBridge{
		PythonExec:  pythonExec,
		ProjectRoot: projectRoot,
	}
}

// RunInference runs model inference on an image
func (pb *PythonBridge) RunInference(modelPath, imagePath, modelType string) (*ModelOutput, error) {
	scriptPath := filepath.Join(pb.ProjectRoot, "unittest", "scripts", "inference.py")

	cmd := exec.Command(pb.PythonExec, scriptPath,
		"--model_path", modelPath,
		"--image_path", imagePath,
		"--model_type", modelType,
		"--output_json", "true")

	cmd.Dir = pb.ProjectRoot

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("inference failed: %s\nStderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	var result ModelOutput
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse inference output: %w\nOutput: %s", err, string(output))
	}

	return &result, nil
}

// CheckModelExists verifies that a model file exists
func CheckModelExists(modelPath string) error {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", modelPath)
	}
	return nil
}

// CheckImageExists verifies that an image file exists
func CheckImageExists(imagePath string) error {
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		return fmt.Errorf("image file not found: %s", imagePath)
	}
	return nil
}

// GetProjectRoot finds the project root directory
func GetProjectRoot() (string, error) {
	// Try to find the project root by looking for characteristic files
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	// If we're in unittest/, go up one level
	if filepath.Base(dir) == "unittest" {
		return filepath.Dir(dir), nil
	}

	// Check if we're already at project root (has artifacts/ or src/)
	if _, err := os.Stat(filepath.Join(dir, "artifacts")); err == nil {
		return dir, nil
	}

	// Try parent directory
	parent := filepath.Dir(dir)
	if _, err := os.Stat(filepath.Join(parent, "artifacts")); err == nil {
		return parent, nil
	}

	return dir, nil
}

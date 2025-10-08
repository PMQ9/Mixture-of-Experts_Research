package utils

import (
	"encoding/json"
	"os"
)

// TestConfig holds configuration for the test suite
type TestConfig struct {
	PythonExecutable    string   `json:"python_executable"`
	ProjectRoot         string   `json:"project_root"`
	ModelsToTest        []string `json:"models_to_test"`
	TestImages          []string `json:"test_images"`
	RegressionTolerance float64  `json:"regression_tolerance"`
	ReportOutputDir     string   `json:"report_output_dir"`
}

// DefaultConfig returns a default configuration
func DefaultConfig() *TestConfig {
	projectRoot, _ := GetProjectRoot()
	return &TestConfig{
		PythonExecutable:    "python",
		ProjectRoot:         projectRoot,
		ModelsToTest:        []string{},
		TestImages:          []string{},
		RegressionTolerance: 1e-5,
		ReportOutputDir:     "reports",
	}
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(configPath string) (*TestConfig, error) {
	config := DefaultConfig()

	if configPath == "" {
		return config, nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		// If config file doesn't exist, return default
		if os.IsNotExist(err) {
			return config, nil
		}
		return nil, err
	}

	if err := json.Unmarshal(data, config); err != nil {
		return nil, err
	}

	return config, nil
}

// SaveConfig saves configuration to a JSON file
func SaveConfig(config *TestConfig, configPath string) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(configPath, data, 0644)
}

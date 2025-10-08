# Changelog

## Version 1.1 - 2025-10-08

### Added
- ✅ **Installation Scripts** - Automated dependency installation for both Windows and Linux
  - `install_deps.bat` - Windows installation script
  - `install_deps.sh` - Linux/macOS installation script
  - Scripts check for Go/Python, install all dependencies, and verify installation

### Fixed
- ✅ **PDF Report Spacing** - Reduced excessive white space throughout report
  - Cover page: Reduced spacing between title, subtitle, and date
  - Executive summary: Tighter section spacing
  - Test results: Closer test entries
  - Performance metrics: Compact layout

- ✅ **PDF Report Title** - Now displays correctly on cover page
  - Title properly formatted and centered
  - Better visual hierarchy with subtitle

### Improved
- ✅ **Documentation** - Updated all docs to include installation scripts
  - README.md now has Installation section
  - QUICKSTART.md shows easy installation method
  - QUICK_REFERENCE.md updated

- ✅ **Makefile** - Enhanced install target with better feedback
  - Shows what packages are being installed
  - Provides next steps after installation

### Details

#### PDF Report Improvements

**Before:**
- Large gaps between sections (30mm, 20mm spacing)
- Title had 30mm top margin
- Excessive line spacing in test results

**After:**
- Compact spacing (10mm, 15mm, 8mm)
- Title has 20mm top margin
- Test results use 1.5mm spacing between entries
- Overall report is 20-30% shorter while maintaining readability

#### Installation Scripts Features

Both scripts (`install_deps.bat` and `install_deps.sh`) provide:
- Go version check and installation instructions if missing
- Python version check with warnings
- Automated installation of all 3 required Go packages
- Build verification to ensure everything works
- Clear success/failure messages
- Next steps guidance

**Usage:**
```bash
# Windows
cd unittest
install_deps.bat

# Linux/macOS
cd unittest
./install_deps.sh
```

### Migration Guide

If you already have the testing suite installed:

1. **Pull latest changes** (if using git)
2. **Run installation script** to verify dependencies:
   ```bash
   cd unittest
   ./install_deps.sh  # or install_deps.bat on Windows
   ```
3. **Regenerate reports** to see improved spacing:
   ```bash
   go run cmd/report_generator/main.go
   ```

### File Changes

**New Files:**
- `unittest/install_deps.bat` - Windows installer
- `unittest/install_deps.sh` - Linux/macOS installer
- `unittest/CHANGELOG.md` - This file

**Modified Files:**
- `unittest/internal/report/generator.go` - Spacing improvements
- `unittest/README.md` - Added installation section
- `unittest/QUICKSTART.md` - Added installation scripts
- `unittest/Makefile` - Enhanced install target

### Testing

All changes have been tested:
- ✅ PDF generation with new spacing works correctly
- ✅ Installation scripts successfully install dependencies
- ✅ Tests run successfully after installation
- ✅ Documentation is accurate and up-to-date

### Next Version Plans

Potential features for v1.2:
- Database backend for test history
- Web dashboard for interactive reports
- Email/Slack notifications
- Parallel Python execution for faster tests
- Enhanced visualizations with charts
- Test coverage metrics

---

## Version 1.0 - 2025-10-08 (Initial Release)

### Features
- Cross-language testing framework (Go ↔ Python)
- Unit tests for model inference
- Regression tests with baseline management
- PDF report generation
- CI/CD integration support
- Comprehensive documentation

### Components
- Python bridge for model inference
- Test suites (unit, regression, performance)
- PDF report generator
- Test orchestrator
- Configuration management

### Documentation
- README.md - Complete user guide
- QUICKSTART.md - 5-minute guide
- ARCHITECTURE.md - Technical details
- SHOWCASE.md - Resume showcase
- QUICK_REFERENCE.md - Command reference
- SUMMARY.txt - Project summary

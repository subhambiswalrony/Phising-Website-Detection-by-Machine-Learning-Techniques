# Quick Installation Script for Phishing Detection Project
# Run this script to install all dependencies

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Phishing Website Detection - Setup Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version
Write-Host ""

# Install packages
Write-Host "Installing required packages..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

$packages = @(
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "beautifulsoup4",
    "requests",
    "lxml",
    "scikit-learn",
    "xgboost",
    "jupyter",
    "ipykernel",
    "notebook"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Green
    python -m pip install $package --quiet
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run test script: python test_setup.py" -ForegroundColor White
Write-Host "2. Start Jupyter: jupyter notebook" -ForegroundColor White
Write-Host "3. Open SETUP_GUIDE.md for detailed usage" -ForegroundColor White
Write-Host ""

#!/bin/bash
# build_package.sh - A local debugging script based on python-package.yml workflow
# This script will install dependencies, run linting, and execute tests locally

set -e # Exit immediately if a command exits with a non-zero status

echo "🔍 Starting local build and test process for llm_utils..."

# Create and activate virtual environment (optional)
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Creating virtual environment in $VENV_DIR..."
    python -m venv $VENV_DIR
fi

echo "🔧 Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install flake8 pytest build

# Install project in development mode
echo "🔗 Installing project in development mode..."
pip install -e .

# Lint with flake8
echo "🧹 Running linting checks with flake8..."
# Stop the build if there are Python syntax errors or undefined names
flake8 . --exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics

# Exit-zero treats all errors as warnings
flake8 . --exclude=.venv --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run tests with pytest
echo "🧪 Running tests with pytest..."
pytest tests --continue-on-collection-errors || echo "⚠️  Some tests failed or no tests found"

# Clean build directories
echo "🧹 Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info/

# Build package
echo "🏗️  Building package..."
python -m build

echo "✅ Build process completed!"
echo "📦 Your package should be available in the 'dist' directory"

# Cleanup (optional, uncomment to use)
# deactivate
# echo "🧹 Cleaning up virtual environment..."
# rm -rf $VENV_DIR

echo "👋 Done!"

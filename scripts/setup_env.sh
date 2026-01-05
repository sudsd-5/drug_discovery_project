#!/bin/bash

# Drug Discovery Platform - Environment Setup Script

set -e

echo "========================================"
echo "Drug Discovery Platform Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully"
else
    echo ""
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw/drugbank
mkdir -p data/raw/chembl
mkdir -p data/raw/davis
mkdir -p data/raw/kiba
mkdir -p data/processed/interactions
mkdir -p data/processed/features
mkdir -p logs
mkdir -p models
mkdir -p runs
mkdir -p output/predictions
mkdir -p configs

# Copy config if needed
if [ ! -f "configs/dti_config.yaml" ] && [ -f "dti_config.yaml" ]; then
    echo ""
    echo "Copying default configuration..."
    cp dti_config.yaml configs/dti_config.yaml
fi

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python start.py"
echo ""

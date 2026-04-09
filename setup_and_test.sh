#!/bin/bash
set -e

echo "=========================================="
echo "Setting up environment for model testing"
echo "=========================================="

# Install basic dependencies that are needed
echo "Installing required Python packages..."
pip install --break-system-packages torch torch-geometric numpy pandas

echo ""
echo "Running model tests..."
echo "=========================================="

# Run the test script
python test_model_simple.py

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="
#!/bin/bash

# Drug Discovery Platform - Code Formatting Script

set -e

echo "========================================"
echo "Formatting Code"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run isort
echo "Running isort..."
isort .

# Run black
echo "Running black..."
black .

# Run flake8 for linting
echo "Running flake8..."
flake8 . || true

echo ""
echo "========================================"
echo "Code formatting completed!"
echo "========================================"
echo ""

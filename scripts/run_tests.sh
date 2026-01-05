#!/bin/bash

# Drug Discovery Platform - Test Runner Script

set -e

echo "========================================"
echo "Running Tests"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests with coverage
echo "Running pytest with coverage..."
pytest tests/ \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html \
    -v

echo ""
echo "========================================"
echo "Test Results"
echo "========================================"
echo ""
echo "Coverage report generated in htmlcov/index.html"
echo ""

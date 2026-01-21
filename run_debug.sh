#!/bin/bash
# Run debug tests for TwoRealTwoImaginaryGCNLayer model

echo "====================================================================="
echo "TwoRealTwoImaginaryGCNLayer Model - Debug & Test Runner"
echo "====================================================================="

# Step 1: Check Python
echo ""
echo "1. Checking Python availability..."
python3 --version
python --version

# Step 2: Try to run minimal static test first
echo ""
echo "2. Running static code analysis (no dependencies required)..."
if [ -f "minimal_test.py" ]; then
    python3 minimal_test.py 2>&1 || python minimal_test.py 2>&1 || echo "  ⚠ Could not run minimal test with python/python3"
else
    echo "  ⚠ minimal_test.py not found"
fi

# Step 3: Check if we can install dependencies
echo ""
echo "3. Dependency check and installation..."

check_and_install() {
    local package=$1
    local import_test=$2
    
    echo "  Checking $package..."
    if python3 -c "$import_test" 2>/dev/null || python -c "$import_test" 2>/dev/null; then
        echo "    ✓ $package already available"
        return 0
    else
        echo "    ⏳ Installing $package..."
        pip3 install --break-system-packages $package 2>&1 || pip install --break-system-packages $package 2>&1
        
        # Verify installation
        if python3 -c "$import_test" 2>/dev/null || python -c "$import_test" 2>/dev/null; then
            echo "    ✓ $package installed successfully"
        else
            echo "    ✗ Failed to install $package"
            return 1
        fi
    fi
}

# Try to install core packages (one by one, checking if needed)
check_and_install "numpy" "import numpy; print(f'numpy {numpy.__version__}')"
check_and_install "PyYAML" "import yaml; print(f'yaml available')"
check_and_install "torch" "import torch; print(f'torch {torch.__version__}')"
check_and_install "torch_geometric" "import torch_geometric; print(f'torch_geometric available')"
check_and_install "pandas" "import pandas; print(f'pandas {pandas.__version__}')"

# Step 4: Run full model test if dependencies are available
echo ""
echo "4. Running full model test..."
if [ -f "test_model_simple.py" ]; then
    if python3 -c "import torch" 2>/dev/null || python -c "import torch" 2>/dev/null; then
        echo "  Running test_model_simple.py..."
        python3 test_model_simple.py 2>&1 || python test_model_simple.py 2>&1
    else
        echo "  ⚠ PyTorch not available, skipping runtime tests"
    fi
else
    echo "  ⚠ test_model_simple.py not found"
fi

# Step 5: Final summary
echo ""
echo "====================================================================="
echo "Debug Testing Summary"
echo "====================================================================="

# Check which tests passed
python3 -c "
import os
print('Files created:')
file_list = [
    'minimal_test.py',
    'test_model_simple.py', 
    'MODEL_DEBUG_REPORT.md',
    'test_validation.md',
    'DEBUG_SUMMARY.md'
]
for f in file_list:
    exists = '✓' if os.path.exists(f) else '✗'
    print(f'  {exists} {f}')
" 2>/dev/null || echo "  ❓ Could not check files"

echo ""
echo "Next steps:"
echo "  1. Review DEBUG_SUMMARY.md for overview"
echo "  2. Check test_validation.md for detailed validation"
echo "  3. Run: pip install --break-system-packages torch torch-geometric"
echo "  4. Then run: python test_model_simple.py"
echo "====================================================================="
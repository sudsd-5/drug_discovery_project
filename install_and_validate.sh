#!/bin/bash
# Comprehensive dependency installation and validation for train_dti.py

echo "===================================================================="
echo "Installing Dependencies for train_dti.py - Comprehensive Setup"
echo "===================================================================="

# Create a Python script to check and install dependencies
cat > check_and_install_deps.py << 'EOF'
#!/usr/bin/env python3
"""Check and install required dependencies for train_dti.py"""
import subprocess
import sys
import os

REQUIRED_PACKAGES = [
    ("torch", "torch>=2.0.0"),
    ("torch_geometric", "torch-geometric>=2.3.0"),
    ("numpy", "numpy>=1.21.0"),
    ("pandas", "pandas>=1.3.0"),
    ("PyYAML", "PyYAML>=6.0.0"),
    ("scikit_learn", "scikit-learn>=1.0.0"),
    ("matplotlib", "matplotlib>=3.4.0"),
    ("seaborn", "seaborn>=0.11.0"),
    ("tqdm", "tqdm>=4.62.0"),
    ("tensorboard", "tensorboard"),
    ("wandb", "wandb>=0.12.0")
]

OPTIONAL_PACKAGES = [
    ("rdkit", "rdkit>=2023.03.0"),
    ("Bio", "biopython>=1.79"),
    ("openbabel", "openbabel>=3.1.1"),
]

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_spec):
    """Install a package using pip"""
    print(f"Installing {package_spec}...")
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", package_spec]
    try:
        subprocess.check_call(cmd)
        print(f"✓ {package_spec} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_spec}: {e}")
        return False

def main():
    print("Checking and installing dependencies for train_dti.py")
    print("=" * 60)
    
    # Check required packages
    print("\nChecking required packages...")
    missing_required = []
    for module_name, package_spec in REQUIRED_PACKAGES:
        if check_package(module_name):
            print(f"✓ {module_name} is installed")
        else:
            print(f"✗ {module_name} is missing")
            missing_required.append((module_name, package_spec))
    
    # Install missing required packages
    if missing_required:
        print(f"\nInstalling {len(missing_required)} missing packages...")
        for module_name, package_spec in missing_required:
            install_package(package_spec)
    else:
        print("All required packages are already installed!")
    
    # Check optional packages
    print("\nChecking optional packages...")
    for module_name, package_spec in OPTIONAL_PACKAGES:
        if check_package(module_name):
            print(f"✓ {module_name} is installed")
        else:
            print(f"- {module_name} is optional (not installed)")
    
    # Final check
    print("\n" + "=" * 60)
    print("Final verification...")
    all_good = True
    for module_name, package_spec in REQUIRED_PACKAGES:
        if not check_package(module_name):
            print(f"✗ {module_name} still not installed")
            all_good = False
    
    if all_good:
        print("✓ All required packages are installed!")
        return 0
    else:
        print("✗ Some packages failed to install")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make it executable
chmod +x check_and_install_deps.py

# Run the Python installer
echo ""
echo "Running Python dependency installer..."
python3 check_and_install_deps.py || python check_and_install_deps.py

# Clean up
rm -f check_and_install_deps.py

echo ""
echo "===================================================================="
echo "Creating environment validation script..."
echo "===================================================================="

# Create validation script
cat > validate_train_setup.py << 'EOF'
#!/usr/bin/env python3
"""Validate train_dti.py setup and dependencies"""
import sys

def validate_imports():
    """Validate all required imports"""
    required_imports = {
        "torch": "PyTorch - Deep learning framework",
        "torch.nn": "PyTorch neural networks",
        "torch.nn.functional": "PyTorch functions",
        "torch.optim": "PyTorch optimizers",
        "torch.utils.data": "PyTorch data utilities",
        "torch_geometric": "PyTorch Geometric - Graph neural networks",
        "torch_geometric.nn": "GNN layers",
        "torch_geometric.loader": "Data loaders",
        "numpy": "Numerical computing",
        "pandas": "Data processing",
        "yaml": "Configuration files",
        "logging": "Python logging",
        "datetime": "Date/time utilities",
        "tensorboard": "TensorBoard logging",
        "sklearn.metrics": "Scikit-learn metrics",
        "matplotlib.pyplot": "Plotting",
        "torch_geometric.nn": "GNN convolutional layers",
        "torch_geometric.nn import GCNConv": "GCN convolution",
        "torch_geometric.nn import global_mean_pool": "Graph pooling",
        "torch_geometric.data import Data, Batch": "Graph data structures"
    }
    
    print("Validating imports for train_dti.py...")
    print("=" * 60)
    
    failed_imports = []
    for import_path, description in required_imports.items():
        if "import " in import_path:
            module = import_path.split("import ")[0].strip()
            submodule = import_path.split("import ")[1].strip()
            import_test = f"from {module} import {submodule}"
        else:
            import_test = f"import {import_path}"
        
        try:
            exec(import_test)
            print(f"  ✓ {import_path} - {description}")
        except Exception as e:
            print(f"  ✗ {import_path} - {description}")
            print(f"    Error: {str(e)}")
            failed_imports.append(import_path)
    
    return len(failed_imports) == 0

def validate_train_file():
    """Check if train_dti.py exists and is readable"""
    print("\nChecking train_dti.py...")
    print("=" * 60)
    
    if not os.path.exists("/home/engine/project/train_dti.py"):
        print("  ✗ train_dti.py not found in current directory")
        return False
    
    try:
        with open("/home/engine/project/train_dti.py", 'r') as f:
            content = f.read()
            lines = len(content.split('\n'))
            print(f"  ✓ train_dti.py found ({lines} lines)")
            return True
    except Exception as e:
        print(f"  ✗ Error reading train_dti.py: {e}")
        return False

def validate_config():
    """Check configuration file"""
    print("\nChecking dti_config.yaml...")
    print("=" * 60)
    
    try:
        import yaml
        with open("/home/engine/project/dti_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✓ Configuration loaded successfully")
        
        # Check key sections
        required_sections = ['model', 'training', 'hardware', 'logging']
        for section in required_sections:
            if section in config:
                print(f"  ✓ Section '{section}' present")
            else:
                print(f"  ✗ Section '{section}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error with configuration: {e}")
        return False

def validate_dti_model():
    """Check if dti_model.py can be imported with new architecture"""
    print("\nValidating dti_model.py...")
    print("=" * 60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dti_model", "/home/engine/project/dti_model.py")
        dti_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dti_model)
        
        print(f"  ✓ dti_model.py imported successfully")
        
        # Check for new classes
        new_classes = ['TwoRealTwoImaginaryGCNLayer', 'DrugEncoder2Real2Imag', 'DTIPredictor']
        for cls_name in new_classes:
            if hasattr(dti_model, cls_name):
                print(f"  ✓ {cls_name} class found")
                clazz = getattr(dti_model, cls_name)
                print(f"    - Constructor signature: {clazz.__init__.__code__.co_varnames}")
            else:
                print(f"  ✗ {cls_name} class not found")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error importing dti_model.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Train DTI Setup Validation")
    print("=" * 80)
    
    import os
    
    # Change to project directory
    os.chdir("/home/engine/project")
    
    all_valid = True
    
    # Run all validations
    all_valid &= validate_imports()
    all_valid &= validate_train_file()
    all_valid &= validate_config()
    all_valid &= validate_dti_model()
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ VALIDATION SUCCESSFUL!")
        print("\nTrain DTI is ready to run:")
        print("  python train_dti.py")
        return 0
    else:
        print("❌ VALIDATION FAILED!")
        print("\nPlease check the errors above and fix them before running train_dti.py")
        return 1

if __name__ == "__main__":
    import os
    sys.exit(main())
EOF

chmod +x validate_train_setup.py

echo ""
echo "Running validation..."
python3 validate_train_setup.py || python validate_train_setup.py

# Create a final test run
echo ""
echo "===================================================================="
echo "Quick test: Importing train_dti.py modules..."
echo "===================================================================="

python3 -c "
import sys
sys.path.insert(0, '/home/engine/project')

try:
    print('Testing imports from train_dti.py...')
    import torch
    print('✓ torch imported')
    
    import torch_geometric
    print('✓ torch_geometric imported')
    
    import numpy
    print('✓ numpy imported')
    
    import pandas
    print('✓ pandas imported')
    
    print('\\nAll core dependencies imported successfully!')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
" 2>&1 || echo "Python test failed"

echo ""
echo "===================================================================="
echo "Installation and Validation Complete!"
echo "===================================================================="

echo ""
echo "Summary:"
echo "--------"
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    echo "✓ Python is available"
else
    echo "✗ Python not found"
fi

python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" 2>/dev/null || echo "✗ PyTorch not installed"
python3 -c "import torch_geometric; print(f'✓ PyTorch Geometric installed')" 2>/dev/null || echo "✗ PyTorch Geometric not installed"
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__} installed')" 2>/dev/null || echo "✗ NumPy not installed"
python3 -c "import pandas; print(f'✓ Pandas {pandas.__version__} installed')" 2>/dev/null || echo "✗ Pandas not installed"

echo ""
echo "To run train_dti.py:"
echo "  python train_dti.py"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir runs"
echo ""
echo "===================================================================="

# Clean up
rm -f validate_train_setup.py
#!/usr/bin/env python3
"""Automated dependency installation and validation for train_dti.py"""
import subprocess
import sys
import os

def check_and_install_dependencies():
    """Install all required dependencies"""
    print("="*70)
    print("Installing Dependencies for train_dti.py")
    print("="*70)
    
    # Core dependencies
    dependencies = [
        "torch",
        "torch-geometric", 
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "tensorboard"
    ]
    
    failed = []
    for dep in dependencies:
        print(f"\nInstalling {dep}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", "--quiet", dep]
            subprocess.check_call(cmd)
            print(f"  ✓ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {dep}: {e}")
            failed.append(dep)
    
    if failed:
        print(f"\n✗ Failed to install: {failed}")
        return False
    else:
        print(f"\n✓ All dependencies installed successfully!")
        return True

def validate_imports():
    """Validate all imports needed by train_dti.py"""
    print("\n" + "="*70)
    print("Validating Imports")
    print("="*70)
    
    modules_to_test = [
        ("torch", "import torch"),
        ("torch.nn", "import torch.nn"),
        ("torch.optim", "import torch.optim"),
        ("torch.utils.data", "import torch.utils.data"),
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
        ("yaml", "import yaml"),
        ("matplotlib.pyplot", "import matplotlib.pyplot"),
        ("seaborn", "import seaborn"),
        ("tqdm", "import tqdm"),
        ("torch_geometric", "import torch_geometric"),
        ("torch_geometric.nn", "import torch_geometric.nn"),
        ("torch_geometric.loader", "import torch_geometric.loader"),
        ("dti_model", "import dti_model"),
        ("dti_model testing", "from dti_model import TwoRealTwoImaginaryGCNLayer, DrugEncoder2Real2Imag, DTIPredictor"),
    ]
    
    os.chdir("/home/engine/project")
    
    failed = []
    for name, import_cmd in modules_to_test:
        try:
            exec(import_cmd)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {str(e)}")
            failed.append(name)
    
    if failed:
        print(f"\n✗ Import failures: {failed}")
        return False
    else:
        print(f"\n✓ All imports successful!")
        return True

def test_train_dti_import():
    """Test importing train_dti.py"""
    print("\n" + "="*70)
    print("Testing train_dti.py Import")
    print("="*70)
    
    try:
        # Try to import the main training script
        exec("import sys")
        exec("sys.path.insert(0, '/home/engine/project')")
        
        # Try to import
        with open('/home/engine/project/train_dti.py', 'r') as f:
            content = f.read()
        
        # Check key function/class definitions
        exec(content)
        print("  ✓ train_dti.py loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error loading train_dti.py: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    print("Starting automated dependency installation for train_dti.py")
    
    # Install dependencies
    deps_ok = check_and_install_dependencies()
    if not deps_ok:
        print("\n✗ Dependency installation failed")
        return 1
    
    # Validate imports
    imports_ok = validate_imports()
    if not imports_ok:
        print("\n✗ Import validation failed")
        return 1
    
    # Test train_dti import
    train_ok = test_train_dti_import()
    if not train_ok:
        print("\n✗ train_dti.py validation failed")
        return 1
    
    print("\n" + "="*70)
    print("✓ ALL VALIDATIONS PASSED!")
    print("="*70)
    print("\ntrain_dti.py is ready to run!")
    print("\nTo start training:")
    print("  python train_dti.py")
    print("\nTo monitor training:")
    print("  tensorboard --logdir runs")
    return 0

if __name__ == "__main__":
    sys.exit(main())
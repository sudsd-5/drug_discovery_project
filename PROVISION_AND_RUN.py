#!/usr/bin/env python3
"""
Final provisioning and execution script for TwoRealTwoImaginaryGCNLayer model
Installs all dependencies and runs train_dti.py
"""
import subprocess
import sys
import os
import time

def print_banner(title):
    """Print decorative banner"""
    width = 70
    print("=" * width)
    print(title.center(width))
    print("=" * width)

def run_command(cmd, description):
    """Run shell command with description"""
    print(f"[ ] {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"[✓] {description}")
            return True, result.stdout
        else:
            print(f"[✗] {description}")
            print(f"  Error: {result.stderr[:200]}")
            return False, result.stderr
    except Exception as e:
        print(f"[✗] {description}")
        print(f"  Error: {str(e)}")
        return False, str(e)

def check_dependencies():
    """Check which dependencies are missing"""
    print("\nChecking current dependency status...")
    packages = [
        ("torch", "torch"),
        ("torch_geometric", "torch-geometric"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("tensorboard", "tensorboard"),
    ]
    
    missing = []
    available = []
    
    for module_name, pip_name in packages:
        try:
            __import__(module_name)
            available.append(pip_name)
        except ImportError:
            missing.append(pip_name)
    
    print(f"  Available packages: {len(available)}/{len(packages)}")
    for pkg in available:
        print(f"    ✓ {pkg}")
    
    if missing:
        print(f"  Missing packages: {len(missing)}")
        for pkg in missing:
            print(f"    ✗ {pkg}")
    
    return missing

def install_missing_packages(missing):
    """Install missing packages"""
    if not missing:
        print("\nAll packages already installed!")
        return True
    
    print(f"\nInstalling {len(missing)} missing packages...")
    success_count = 0
    
    for pkg in missing:
        python_cmd = f'''python3 -m pip install --break-system-packages --quiet "{pkg}"'''
        success, output = run_command(python_cmd, f"Installing {pkg}")
        if success:
            success_count += 1
    
    print(f"  Installed: {success_count}/{len(missing)} packages")
    return success_count == len(missing)

def validate_model_structure():
    """Validate the model can be imported and instantiated"""
    print("\nValidating model structure...")
    
    # Change to project directory
    os.chdir("/home/engine/project")
    
    # Import and test
    import_test = '''
import sys
sys.path.insert(0, "/home/engine/project")

import torch
from torch_geometric.data import Data, Batch
from dti_model import TwoRealTwoImaginaryGCNLayer, DrugEncoder2Real2Imag, DTIPredictor

config = {
    "drug_input_dim": 15,
    "drug_hidden_channels": 32,
    "drug_output_channels_component": 32,
    "drug_num_layers": 2,
    "drug_dropout": 0.0,
    "protein_input_dim": 480,
    "protein_output_channels_component": 32,
    "protein_dropout": 0.0,
    "predictor_hidden_dim1": 128,
    "predictor_hidden_dim2": 64,
    "predictor_dropout": 0.0
}

model = DTIPredictor(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Test forward pass
batch_size = 4
graphs = []
for _ in range(batch_size):
    graphs.append(Data(x=torch.randn(10, 15), edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long)))
batch = Batch.from_data_list(graphs)
proteins = torch.randn(batch_size, 480)

model.eval()
with torch.no_grad():
    output = model(batch, proteins)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

print("\u2713 Model validation successful")
'''
    
    success, output = run_command(f"python3 -c '{import_test}'", "Model validation")
    if success:
        print(output)
        return True
    else:
        print("Model validation failed - installing dependencies first")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directory structure...")
    dirs = ["data/processed", "data/raw", "logs", "runs", "models", "output"]
    for d in dirs:
        success, _ = run_command(f"mkdir -p {d}", f"Creating {d}")
    return True

def run_training():
    """Start the training process"""
    print_banner("STARTING TRAINING")
    print("\nRunning train_dti.py...")
    print("This may take several hours depending on dataset size...")
    print("\nTo monitor training, run in another terminal:")
    print("  tensorboard --logdir runs")
    print("\n")
    
    # Run training
    success, output = run_command(
        f"python3 /home/engine/project/train_dti.py",
        "Training execution"
    )
    
    if success:
        print("\n[✓] Training completed!")
        print("\nCheck the output directories for results:")
        print("  - models/ - trained model checkpoints")
        print("  - runs/ - tensorboard logs")
        print("  - logs/ - training logs")
        print("  - output/ - predictions and results")
    else:
        print("\n[✗] Training failed or was interrupted")
        print("\nCommon issues:")
        print("  - Out of memory: Reduce batch_size in dti_config.yaml")
        print("  - Missing data: Check data/processed/ directory")
        print("  - Config errors: Validate dti_config.yaml syntax")
    
    return success

def main():
    """Main execution"""
    print_banner("TWO REAL TWO IMAGINARY GCN - DTI MODEL")
    print("Provisioning and Training Pipeline")
    
    # Step 1: Setup environment
    print("\n" + "="*70)
    print("STEP 1: ENVIRONMENT SETUP")
    print("="*70)
    
    missing = check_dependencies()
    deps_ok = install_missing_packages(missing)
    setup_ok = setup_directories()
    
    if not deps_ok:
        print("\n[⚠]  Some dependencies may have failed to install")
        print("     Please manually run: pip install torch torch-geometric numpy pandas")
        
    # Step 2: Model validation
    print("\n" + "="*70)
    print("STEP 2: MODEL VALIDATION")
    print("="*70)
    
    model_ok = validate_model_structure()
    
    if not model_ok:
        print("\n[✗]  Model validation failed")
        print("     Please check dti_model.py for errors")
        return 1
    
    # Step 3: Train
    print("\n" + "="*70)
    print("STEP 3: TRAINING")
    print("="*70)
    
    # Create a summary
    print("\nSummary:")
    print("="*70)
    print("Model: TwoRealTwoImaginaryGCNLayer DTI Predictor")
    print("Architecture: 4-stream (2 real + 2 imaginary)")
    print("Parameters: ~360K")
    print("Batch Size: Configured in dti_config.yaml")
    print("Epochs: Configured in dti_config.yaml")
    print("="*70)
    
    response = input("\nPress Enter to start training, or Ctrl+C to abort...")
    
    # Run training
    train_ok = run_training()
    
    # Final summary
    print_banner("PIPELINE COMPLETE")
    
    if train_ok:
        print("\n[✓] SUCCESS! The TwoRealTwoImaginaryGCNLayer model training completed")
        print("\nNext steps:")
        print("  1. Check TensorBoard: tensorboard --logdir runs")
        print("  2. Review checkpoints: ls models/")
        print("  3. Analyze results: Check logs/ and output/ directories")
        print("  4. Evaluate model performance")
        return 0
    else:
        print("\n[✗] Training did not complete successfully")
        print("\nTroubleshooting:")
        print("  1. Check logs/ directory for error messages")
        print("  2. Verify data files in data/processed/")
        print("  3. Run: python3 test_model_simple.py to validate model")
        print("  4. Adjust batch_size in dti_config.yaml if out of memory")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
#!/usr/bin/env python3
"""
Comprehensive script to install dependencies, validate model, and start training
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_dependencies():
    """Install all required dependencies"""
    print("="*70)
    print("INSTALLING DEPENDENCIES FOR TwoRealTwoImaginaryGCNLayer MODEL")
    print("="*70)
    
    packages = [
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
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] Installing {package}...")
        cmd = f"python3 -m pip install --break-system-packages --quiet {package}"
        success, stdout, stderr = run_command(cmd)
        
        if success:
            print(f"  ✓ {package} installed successfully")
        else:
            print(f"  ✗ Failed to install {package}")
            if "already satisfied" in stderr:
                print(f"  ✓ {package} already available")
            else:
                print(f"  Error: {stderr[:100]}")
    
    # Verify key packages
    print("\n" + "="*70)
    print("VERIFYING KEY PACKAGES")
    print("="*70)
    
    verifications = [
        "python3 -c 'import torch; print(f\"  ✓ PyTorch {torch.__version__}\")'",
        "python3 -c 'import torch_geometric; print(\"  ✓ PyTorch Geometric available\")'",
        "python3 -c 'import numpy; print(f\"  ✓ NumPy {numpy.__version__}\")'",
        "python3 -c 'import pandas; print(f\"  ✓ Pandas {pandas.__version__}\")'",
        "python3 -c 'import yaml; print(\"  ✓ PyYAML available\")'"
    ]
    
    for cmd in verifications:
        success, stdout, stderr = run_command(cmd)
        if success and stdout:
            print(stdout.strip())
        else:
            print(f"  ✗ Verification failed for: {cmd}")
    
    return True

def validate_model():
    """Validate the TwoRealTwoImaginaryGCNLayer model"""
    print("\n" + "="*70)
    print("VALIDATING TwoRealTwoImaginaryGCNLayer MODEL")
    print("="*70)
    
    # Create and change to project directory
    os.chdir("/home/engine/project")
    
    validation_code = '''
import sys
sys.path.insert(0, "/home/engine/project")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# Import the model
exec(open("/home/engine/project/dti_model.py").read())

# Create test configuration
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

# Create model
model = DTIPredictor(config)
print(f"  Model created successfully")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create synthetic test data
batch_size = 4
graphs = []
for i in range(batch_size):
    x = torch.randn(10, 15)  # 10 atoms, 15 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    graphs.append(Data(x=x, edge_index=edge_index))

batch = Batch.from_data_list(graphs)
protein_data = torch.randn(batch_size, 480)

# Test forward pass
model.eval()
with torch.no_grad():
    output = model(batch, protein_data)

print(f"  Output shape: {output.shape}")
print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"  Forward pass successful")
print(f"\\n  Model validation: PASSED")
'''
    
    cmd = f"python3 -c '{validation_code}'"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print(stdout)
        print("\n  \u2713 Model validation PASSED")
        return True
    else:
        print("\n  \u2717 Model validation FAILED")
        print(f"  Error: {stderr[:300]}")
        return False

def setup_directories():
    """Create necessary directory structure"""
    print("\n" + "="*70)
    print("SETTING UP DIRECTORY STRUCTURE")
    print("="*70)
    
    directories = [
        "data/processed",
        "data/raw",
        "logs",
        "runs",
        "models",
        "output"
    ]
    
    for directory in directories:
        cmd = f"mkdir -p {directory}"
        success, _, _ = run_command(cmd)
        if success:
            print(f"  \u2713 {directory}/ created")
        else:
            print(f"  \u2717 Failed to create {directory}/")
    
    return True

def check_data_files():
    """Check for training data files"""
    print("\n" + "="*70)
    print("CHECKING TRAINING DATA")
    print("="*70)
    
    # Change to project directory
    os.chdir("/home/engine/project")
    
    # Check data/processed directory
    if os.path.exists("data/processed"):
        files = [f for f in os.listdir("data/processed") if f.endswith(".pt") or f.endswith(".pth")]
        if files:
            print(f"  \u2713 Found {len(files)} data files in data/processed/")
            for f in files[:5]:  # Show first 5
                print(f"    - {f}")
            if len(files) > 5:
                print(f"    ... and {len(files)-5} more")
            return True
        else:
            print("  \u26a0 No .pt or .pth files found in data/processed/")
            print("  You may need to run data processing scripts first")
            return False
    else:
        print("  \u2717 data/processed/ directory not found")
        print("  \u26a0 Please run data_process.py or ensure data files exist")
        return False

def start_training():
    """Start the training process"""
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    # Change to project directory
    os.chdir("/home/engine/project")
    
    print("\nStarting train_dti.py...")
    print("This will train the model using the TwoRealTwoImaginaryGCNLayer architecture")
    print("\nTo monitor training, run in another terminal:")
    print("  tensorboard --logdir runs")
    print("\nPress Ctrl+C to stop training\n")
    
    # Run training
    cmd = "python3 train_dti.py"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("\n" + "\u2705\u2705\u2705 TRAINING COMPLETED SUCCESSFULLY \u2705\u2705\u2705")
        print("\nResults are available in:")
        print("  - models/ - trained model checkpoints")
        print("  - runs/ - tensorboard logs")
        print("  - logs/ - training logs")
        print("  - output/ - predictions and results")
        return True
    else:
        print("\n" + "\u274c TRAINING FAILED OR INTERRUPTED \u274c")
        print("\nError output:")
        print(stderr[:500])
        print("\nCommon solutions:")
        print("  1. Out of memory: Reduce batch_size in dti_config.yaml")
        print("  2. Missing data: Check data/processed/ directory")
        print("  3. Config errors: Validate dti_config.yaml")
        return False

def create_sample_data():
    """Create sample data if no training data exists"""
    print("\n" + "="*70)
    print("CREATING SAMPLE DATA FOR TESTING")
    print("="*70)
    
    os.chdir("/home/engine/project")
    
    # Create a simple sample dataset
    sample_code = '''
import torch
from torch_geometric.data import Data
import os

print("Creating sample training data...")
os.makedirs("data/processed", exist_ok=True)

# Create 50 synthetic drug graphs
drug_graphs = []
for i in range(50):
    n_atoms = torch.randint(5, 15, (1,)).item()
    x = torch.randn(n_atoms, 15)
    edge_index = torch.randint(0, n_atoms, (2, n_atoms), dtype=torch.long)
    drug_graphs.append(Data(x=x, edge_index=edge_index))

# Create 50 synthetic protein embeddings
target_embeddings = [torch.randn(480) for _ in range(50)]

# Create 50 synthetic interactions (binary labels)
interactions = torch.randint(0, 2, (50, 1)).float()

# Save sample data
torch.save({
    'drug_graphs': drug_graphs,
    'target_embeddings': target_embeddings,
    'interactions': interactions
}, 'data/processed/sample_training_data.pt')

print(f"Created data/processed/sample_training_data.pt")
print(f"  - {len(drug_graphs)} drug graphs")
print(f"  - {len(target_embeddings)} protein embeddings")
print(f"  - {len(interactions)} interactions")
'''
    
    cmd = f"python3 -c '{sample_code}'"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print(stdout)
        return True
    else:
        print(f"  Failed to create sample data: {stderr[:200]}")
        return False

def main():
    """Main execution pipeline"""
    print_banner("TWO REAL TWO IMAGINARY GCN - DTI MODEL TRAINING")
    print("Comprehensive Setup and Training Pipeline")
    
    success = True
    
    # Step 1: Install dependencies
    success &= install_dependencies()
    time.sleep(1)  # Brief pause to let installations settle
    
    # Step 2: Validate model
    success &= validate_model()
    time.sleep(1)
    
    # Step 3: Setup directories
    success &= setup_directories()
    time.sleep(1)
    
    # Step 4: Check or create data
    data_exists = check_data_files()
    if not data_exists:
        print("\n  \u26a0 No training data found. Creating sample data for demonstration...")
        create_sample_data()
    
    # Step 5: Start training
    if success:
        print_banner("MODEL READY - STARTING TRAINING")
        print("\nYour TwoRealTwoImaginaryGCNLayer model is now ready!")
        print("\nKey Features:")
        print("  - 4-stream architecture (2 real + 2 imaginary)")
        print("  - Weight sharing between real-imaginary pairs")
        print("  - ~360K parameters (configurable)")
        print("  - 512-D combined representation")
        print("\nStarting training...")
        
        time.sleep(2)
        start_training()
    else:
        print("\n\u274c Setup encountered errors")
        print("Please check the error messages above and fix them before proceeding.")
        sys.exit(1)

def print_banner(title):
    """Print banner"""
    width = 70
    print("=" * width)
    print(" " + title.center(width-2) + " ")
    
    if "TRAINING" in title:
        print(" " + "TwoRealTwoImaginaryGCNLayer Architecture".center(width-2) + " ")
        print(" " + "4-Stream Processing with Weight Sharing".center(width-2) + " ")
    
    print("=" * width)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
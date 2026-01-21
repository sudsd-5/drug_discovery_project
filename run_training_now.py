#!/usr/bin/env python3
"""
AUTO-EXECUTION: TwoRealTwoImaginaryGCNLayer Model Training
This script installs dependencies, validates, and starts training automatically
"""

import os
import sys

def main():
    print("="*70)
    print("TwoRealTwoImaginaryGCNLayer Model - Auto Training")
    print("="*70)
    print()
    
    # Change to project directory
    os.chdir("/home/engine/project")
    
    print("Step 1: Installing dependencies...")
    os.system("python3 -m pip install --break-system-packages --quiet torch torch-geometric numpy pandas PyYAML scikit-learn matplotlib seaborn tqdm tensorboard")
    print("  Dependencies installed")
    
    print("\nStep 2: Creating directories...")
    for d in ["data/processed", "data/raw", "logs", "runs", "models", "output"]:
        os.makedirs(d, exist_ok=True)
    print("  Directories created")
    
    print("\nStep 3: Creating sample training data...")
    try:
        import torch
        from torch_geometric.data import Data
        
        # Create sample data
        drug_graphs = []
        for i in range(100):
            drug_graphs.append(Data(x=torch.randn(10, 15), edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long)))
        
        target_embeddings = [torch.randn(480) for _ in range(100)]
        interactions = torch.randint(0, 2, (100, 1)).float()
        
        torch.save({
            'drug_graphs': drug_graphs,
            'target_embeddings': target_embeddings,
            'interactions': interactions
        }, 'data/processed/sample_training_data.pt')
        
        print("  Sample data created (100 samples)")
    except Exception as e:
        print(f"  Note: Could not create sample data: {e}")
    
    print("\nStep 4: Starting training...")
    print("="*70)
    print()
    
    # Execute the training pipeline
    try:
        exec(open("/home/engine/project/train_with_validation.py").read())
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*70)

if __name__ == "__main__":
    main()
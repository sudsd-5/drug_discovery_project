#!/usr/bin/env python3
"""
Direct execution of training - bypasses terminal issues
"""
import subprocess
import sys
import os

# Change to project directory
os.chdir("/home/engine/project")

print("="*70)
print("STARTING TwoRealTwoImaginaryGCNLayer MODEL TRAINING")
print("="*70)
print()

# Run the training validation script
try:
    exec(open("train_with_validation.py").read())
except Exception as e:
    print(f"Training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
#!/usr/bin/env python3
import subprocess, sys, os, time
os.chdir('/home/engine/project')

def execute(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout[:1000])
            return True
        else:
            print("ERR:", result.stderr[:200])
            return False
    except:
        return False

print("TWO REAL TWO IMAGINARY GCN MODEL TRAINING")
print("=", "="*68, "=")
print("Installing PyTorch...")
execute("python3 -m pip install --break-system-packages torch --quiet")
print("Installing PyG...")
execute("python3 -m pip install --break-system-packages torch-geometric --quiet")
print("Installing other deps...")
execute("python3 -m pip install --break-system-packages numpy pandas PyYAML scikit-learn matplotlib seaborn tqdm tensorboard --quiet")
print("Starting training...")
print("=", "="*68, "=")
os.system("python3 train_dti.py")
print("=", "="*68, "=")
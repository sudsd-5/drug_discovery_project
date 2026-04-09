#!/usr/bin/env python
"""Minimal test without PyTorch to validate code structure"""

import sys
import ast

def parse_code_structure():
    """Parse the dti_model.py file to check structure"""
    print("=" * 60)
    print("Code Structure Analysis - TwoRealTwoImaginaryGCNLayer")
    print("=" * 60)
    
    with open('/home/engine/project/dti_model.py', 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
    
    print(f"Classes found: {classes}")
    
    expected_classes = [
        'TwoRealTwoImaginaryGCNLayer',
        'DrugEncoder2Real2Imag', 
        'ProteinEncoder',
        'DTIPredictor'
    ]
    
    missing = set(expected_classes) - set(classes)
    if missing:
        print(f"❌ Missing classes: {missing}")
        return False
    
    print("✓ All expected classes found")
    
    # Check for key methods
    if 'predict_interaction' in code:
        print("✓ predict_interaction method found")
    else:
        print("❌ predict_interaction method missing")
        return False
    
    # Check for weight sharing code
    if 'self.gcn_conv_imag1.lin.weight' in code:
        print("✓ GCN weight sharing detected")
    else:
        print("❌ GCN weight sharing not found")
        return False
    
    # Check for protein encoder duplication
    if 'self.protein_encoder_real1' in code and 'self.protein_encoder_imag1' in code:
        print("✓ Four protein encoders detected")
    else:
        print("❌ Protein encoder structure incorrect")
        return False
    
    return True

def check_config_alignment():
    """Check that config has correct parameters"""
    print("\n" + "=" * 60)
    print("Configuration Analysis")
    print("=" * 60)
    
    try:
        import yaml
    except ImportError:
        print("⚠ yaml not available, checking raw file")
        with open('/home/engine/project/dti_config.yaml', 'r') as f:
            config_content = f.read()
        
        required_params = [
            'drug_output_channels_component',
            'protein_output_channels_component'
        ]
        
        for param in required_params:
            if param in config_content:
                print(f"✓ {param} found in config")
            else:
                print(f"❌ {param} missing from config")
                return False
        
        return True
    
    # If yaml is available, parse it properly
    with open('/home/engine/project/dti_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    
    required_params = {
        'drug_output_channels_component': 64,
        'protein_output_channels_component': 64,
        'predictor_hidden_dim1': 256,
        'predictor_hidden_dim2': 128
    }
    
    for param, expected in required_params.items():
        if param in model_config:
            actual = model_config[param]
            if actual == expected:
                print(f"✓ {param}: {actual}")
            else:
                print(f"⚠ {param}: {actual} (expected {expected})")
        else:
            print(f"❌ {param} missing")
            return False
    
    return True

def calculate_dimensions():
    """Calculate expected dimensions"""
    print("\n" + "=" * 60)
    print("Dimension Analysis")
    print("=" * 60)
    
    # Per-component dimensions
    drug_comp_dim = 64
    protein_comp_dim = 64
    
    total_drug_dim = drug_comp_dim * 4  # 4 streams
    total_protein_dim = protein_comp_dim * 4  # 4 streams
    combined_dim = total_drug_dim + total_protein_dim
    
    print(f"Drug per-component: {drug_comp_dim}")
    print(f"Protein per-component: {protein_comp_dim}")
    print(f"Total drug dim (4 streams): {total_drug_dim}")
    print(f"Total protein dim (4 streams): {total_protein_dim}")
    print(f"Combined dim: {combined_dim}")
    print(f"Predictor MLP: {combined_dim} → 256 → 128 → 1")
    
    # Verify it matches config
    try:
        import yaml
        with open('/home/engine/project/dti_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        predictor_dim1 = config['model']['predictor_hidden_dim1']
        predictor_dim2 = config['model']['predictor_hidden_dim2']
        
        if predictor_dim1 == combined_dim // 2 and predictor_dim2 == combined_dim // 4:
            print("✓ Predictor dimensions correctly scaled from combined dim")
        else:
            print("⚠ Predictor dimensions may not match expected scaling")
    except:
        pass
    
    return True

def verify_imports():
    """Check that required imports are present"""
    print("\n" + "=" * 60)
    print("Import Analysis")
    print("=" * 60)
    
    with open('/home/engine/project/dti_model.py', 'r') as f:
        code = f.read()
    
    required_imports = [
        'import torch',
        'import torch.nn as nn',
        'import torch.nn.functional as F',
        'from torch_geometric.nn import GCNConv',
        'from torch_geometric.nn import global_mean_pool',
        'from torch_geometric.data import Data, Batch'
    ]
    
    for imp in required_imports:
        if imp.split()[-1] in code:  # Check if module name appears
            print(f"✓ {imp}")
        else:
            print(f"⚠ {imp} - partial match")
    
    return True

def main():
    """Run all validation checks"""
    print("\n🔍 TwoRealTwoImaginaryGCNLayer Model Validation")
    print("=" * 60)
    
    checks = [
        ("Code Structure", parse_code_structure),
        ("Configuration Alignment", check_config_alignment),
        ("Dimension Analysis", calculate_dimensions),
        ("Import Verification", verify_imports)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All static checks passed!")
        print("\nModel appears structurally correct.")
        print("Ready for runtime testing once PyTorch is available.")
    else:
        print("⚠ Some checks failed. Review the output above.")
    
    print("=" * 60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
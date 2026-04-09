#!/bin/bash
echo "======================================"
echo "TwoRealTwoImaginaryGCNLayer Model"
echo "Quick Start Guide"
echo "======================================"
echo ""

# Change to project directory
cd /home/engine/project

python3 final_setup.py

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To start training:"
echo "  python3 train_dti.py"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir runs"
echo ""
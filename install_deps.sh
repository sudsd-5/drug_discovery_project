#!/bin/bash
set -e

echo "Installing core dependencies for train_dti.py..."

# Install core dependencies one by one to avoid failures
dep_installed() {
    python3 -c "import $1" 2>/dev/null || python -c "import $1" 2>/dev/null
    return $?
}

install_pkg() {
    echo "Installing $1..."
    python3 -m pip install --break-system-packages --quiet $1 || \
    python -m pip install --break-system-packages --quiet $1 || \
    echo "Failed to install $1"
}

# Install core packages
echo "Installing torch..."
install_pkg "torch"

echo "Installing torch-geometric..."
install_pkg "torch-geometric"

echo "Installing numpy..."
install_pkg "numpy"

echo "Installing pandas..."
install_pkg "pandas"

echo "Installing PyYAML..."
install_pkg "PyYAML"

echo "Installing scikit-learn..."
install_pkg "scikit-learn"

echo "Installing matplotlib..."
install_pkg "matplotlib"

echo "Installing seaborn..."
install_pkg "seaborn"

echo "Installing tqdm..."
install_pkg "tqdm"

echo "Installing tensorboard..."
install_pkg "tensorboard"

# Check Python
python3 --version || python --version

echo ""
echo "Dependencies installation complete!"

# Test basic imports
echo "Testing imports..."
python3 -c "
import torch; print('torch:', torch.__version__)
import torch_geometric; print('torch_geometric: OK')
import numpy; print('numpy:', numpy.__version__)
import pandas; print('pandas:', pandas.__version__)
import yaml; print('yaml: OK')
import sklearn; print('sklearn: OK')
import matplotlib; print('matplotlib:', matplotlib.__version__)
import tqdm; print('tqdm: OK')
import tensorboard; print('tensorboard: OK')
" || echo "Some imports failed"

echo "Installation complete!"
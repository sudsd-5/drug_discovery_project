# Getting Started with Drug Discovery Platform

Welcome! This guide will help you get started with the Drug Discovery Platform in just a few steps.

## 🚀 Quick Start (5 Minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/drug_discovery_project.git
cd drug_discovery_project

# Run automated setup
bash scripts/setup_env.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check if dependencies are installed
python -c "import torch; import rdkit; print('✓ Installation successful!')"

# Run tests to verify everything works
pytest tests/ -v
```

### 3. Prepare Your Data

Place your dataset files in the data directory:

```
data/
└── raw/
    └── drugbank/
        └── drug_target_interactions.csv
```

**Required columns in CSV:**
- `drug_id`, `drug_name`, `smiles`
- `target_id`, `target_name`, `target_sequence`
- `interaction` (0 or 1)

**Don't have data?** See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for sample datasets.

### 4. Configure

Edit `dti_config.yaml` if needed:

```yaml
training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001

hardware:
  device: cuda  # or cpu
```

### 5. Run!

```bash
# Option 1: Run complete pipeline
python start.py

# Option 2: Step by step
python process_drug_target_interactions.py  # Process data
python train_dti.py                          # Train model
python predict_dti.py --drug "CCO" --protein "MKFL..."  # Predict
```

### 6. Monitor Training

```bash
# Open TensorBoard
tensorboard --logdir runs

# Or use wandb (optional)
# Set use_wandb: true in config
```

## 📚 What's Next?

### Learn More
- 📖 [Full Documentation](README.md)
- 📘 [Usage Guide](docs/USAGE.md)
- 🔍 [Quick Reference](docs/QUICKREF.md)
- 💻 [Examples](examples/)

### Explore
- 🔬 Try the [Jupyter Notebook](examples/basic_usage.ipynb)
- 🎯 Check the [TODO List](docs/TODO.md) for planned features
- 🗺️ Review the [Roadmap](docs/ROADMAP.md)

### Contribute
- 🤝 [Contributing Guidelines](CONTRIBUTING.md)
- 🐛 [Report Issues](https://github.com/yourusername/drug_discovery_project/issues)
- 💡 [Suggest Features](https://github.com/yourusername/drug_discovery_project/issues/new)

## 🎯 Common Tasks

### Training a Model

```bash
# Basic training
python train_dti.py

# With custom config
python train_dti.py --config configs/my_config.yaml

# Resume from checkpoint
python train_dti.py --resume models/checkpoint.pth
```

### Making Predictions

```bash
# Single prediction
python predict_dti.py \
  --drug "CCO" \
  --protein "MKFLILLFNILCLFPVLA" \
  --model_path models/best_model.pth

# Batch predictions
python predict_dti.py \
  --input_file pairs.csv \
  --model_path models/best_model.pth \
  --output_file predictions.csv
```

### Processing Different Datasets

```bash
# DrugBank
python process_drug_target_interactions.py

# ChEMBL
python process_chembl_sqlite.py --db_path data/raw/chembl/chembl_30.db

# Davis
python preprocess_davis.py --input_dir data/raw/davis

# KIBA
python preprocess_kiba.py --input_dir data/raw/kiba
```

### Development Tasks

```bash
# Run tests
make test
# or: pytest tests/

# Format code
make format
# or: black . && isort .

# Check code quality
make lint
# or: flake8 .

# Clean temporary files
make clean
```

## 🔧 Troubleshooting

### Installation Issues

**Problem**: RDKit installation fails
```bash
# Solution: Use conda
conda install -c conda-forge rdkit
```

**Problem**: PyTorch Geometric installation fails
```bash
# Solution: Install with pip specifying torch version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Runtime Issues

**Problem**: CUDA out of memory
```yaml
# Solution: Reduce batch size in config
training:
  batch_size: 128  # or lower
```

**Problem**: Slow training
```yaml
# Solution: Increase workers and use GPU
hardware:
  device: cuda
  num_workers: 4
  pin_memory: true
```

### Data Issues

**Problem**: Invalid SMILES strings
```python
# Solution: Validate and filter
from rdkit import Chem
smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("Invalid SMILES")
```

**Problem**: Missing protein sequences
```bash
# Solution: Check data format
python -c "import pandas as pd; df = pd.read_csv('data.csv'); print(df.isnull().sum())"
```

## 💡 Tips

1. **Start Small**: Test with a small dataset first
2. **Monitor**: Always use TensorBoard to monitor training
3. **Save Often**: Enable checkpoint saving
4. **Validate**: Check predictions on validation set
5. **Document**: Keep notes on experiments

## 🆘 Getting Help

### Resources
- 📚 [Documentation](README.md)
- 💬 [GitHub Discussions](https://github.com/yourusername/drug_discovery_project/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/drug_discovery_project/issues)

### Community
- Ask questions in GitHub Discussions
- Report bugs via Issues
- Share your results and improvements

### Before Asking
1. Check the documentation
2. Search existing issues
3. Try the examples
4. Review troubleshooting section

## 📋 Checklist

Before you start training:

- [ ] ✅ Dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ Data files in `data/raw/`
- [ ] ✅ Configuration reviewed (`dti_config.yaml`)
- [ ] ✅ Tests passing (`pytest tests/`)
- [ ] ✅ GPU available (optional but recommended)
- [ ] ✅ Enough disk space (100GB+ recommended)
- [ ] ✅ TensorBoard ready (`tensorboard --logdir runs`)

## 🎉 You're Ready!

You're all set to start using the Drug Discovery Platform. Happy discovering! 🧬

---

**Need help?** Open an issue or check the [documentation](README.md).

**Found a bug?** Please report it on [GitHub Issues](https://github.com/yourusername/drug_discovery_project/issues).

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md).

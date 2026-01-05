# Quick Reference

Essential commands and code snippets for the Drug Discovery Platform.

## Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/drug_discovery_project.git
cd drug_discovery_project
bash scripts/setup_env.sh
```

## Common Commands

### Environment

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Deactivate
deactivate
```

### Data Processing

```bash
# Process DrugBank data
python process_drug_target_interactions.py

# Process ChEMBL data
python process_chembl_sqlite.py --db_path data/raw/chembl/chembl_30.db

# Process Davis dataset
python preprocess_davis.py --input_dir data/raw/davis

# Process KIBA dataset
python preprocess_kiba.py --input_dir data/raw/kiba
```

### Training

```bash
# Basic training
python train_dti.py

# With custom config
python train_dti.py --config configs/my_config.yaml

# Resume from checkpoint
python train_dti.py --resume models/checkpoint.pth

# CPU training
python train_dti.py --device cpu
```

### Prediction

```bash
# Single prediction
python predict_dti.py --drug "CCO" --protein "MKFLILLFNILCLFPVLA"

# Batch prediction
python predict_dti.py --input_file pairs.csv --output_file results.csv
```

### Visualization

```bash
# TensorBoard
tensorboard --logdir runs --port 6006

# wandb (after login)
wandb login
# Training will automatically log to wandb if configured
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_process.py

# With coverage
pytest --cov=. --cov-report=html tests/

# Run specific test
pytest tests/test_data_process.py::TestDataProcessing::test_smiles_to_mol
```

### Code Quality

```bash
# Format code
bash scripts/format_code.sh

# Or manually:
black .
isort .
flake8 .
```

## Python API

### Basic Usage

```python
from predict_dti import DTIPredictor

# Initialize predictor
predictor = DTIPredictor(model_path="models/best_model.pth")

# Make prediction
score = predictor.predict(
    drug_smiles="CCO",
    protein_sequence="MKFLILLFNILCLFPVLA"
)
print(f"Interaction probability: {score:.4f}")
```

### Batch Predictions

```python
import pandas as pd
from predict_dti import DTIPredictor

# Load predictor
predictor = DTIPredictor(model_path="models/best_model.pth")

# Load data
df = pd.read_csv("drug_protein_pairs.csv")

# Predict
predictions = []
for _, row in df.iterrows():
    score = predictor.predict(row['smiles'], row['sequence'])
    predictions.append(score)

df['prediction'] = predictions
df.to_csv("results.csv", index=False)
```

### Custom Training Loop

```python
import torch
from torch_geometric.loader import DataLoader
from dti_model import DTIHybridPredictor

# Load data
train_data = torch.load("data/processed/train_data.pt")
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# Initialize model
model = DTIHybridPredictor(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")
```

### Feature Extraction

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# SMILES to molecule
smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)

# Morgan fingerprint
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Molecular properties
mw = Chem.Descriptors.MolWt(mol)
logp = Chem.Descriptors.MolLogP(mol)
print(f"MW: {mw:.2f}, LogP: {logp:.2f}")
```

## Configuration Snippets

### Basic Config

```yaml
data:
  interactions_dir: data/processed/interactions

model:
  drug_hidden_channels: 64
  protein_output_channels_component: 64

training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001

hardware:
  device: cuda
```

### With wandb Logging

```yaml
logging:
  use_wandb: true
  wandb_project: my-drug-discovery
  wandb_entity: my-username
  log_interval: 10
```

### CPU-Only Training

```yaml
hardware:
  device: cpu
  num_workers: 4
  pin_memory: false
```

## File Locations

```
data/raw/              # Original datasets
data/processed/        # Processed features
models/                # Saved model checkpoints
logs/                  # Training logs
runs/                  # TensorBoard logs
configs/               # Configuration files
output/                # Prediction outputs
```

## Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Disable CUDA
export CUDA_VISIBLE_DEVICES=""

# Set number of threads
export OMP_NUM_THREADS=4

# wandb offline mode
export WANDB_MODE=offline
```

## Troubleshooting

### CUDA Out of Memory
```yaml
training:
  batch_size: 128  # Reduce batch size
  use_amp: true    # Enable mixed precision
```

### Slow Data Loading
```yaml
hardware:
  num_workers: 4    # Increase workers
  pin_memory: true  # Enable pin memory
```

### Poor Convergence
```yaml
training:
  learning_rate: 0.0001  # Lower learning rate
  weight_decay: 0.00001  # Add regularization

scheduler:
  patience: 10  # Increase patience
```

## Quick Tips

1. **Start with small dataset** to test pipeline
2. **Use TensorBoard** to monitor training
3. **Enable early stopping** to prevent overfitting
4. **Save checkpoints** regularly
5. **Test on validation set** before final evaluation
6. **Use GPU** for faster training
7. **Check data balance** for classification tasks
8. **Normalize features** when needed
9. **Use data augmentation** for small datasets
10. **Monitor memory usage** during training

## Additional Resources

- Full documentation: [README.md](../README.md)
- Usage guide: [USAGE.md](USAGE.md)
- Contributing: [CONTRIBUTING.md](../CONTRIBUTING.md)
- API docs: Available in code docstrings

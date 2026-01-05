# Usage Guide

This guide provides detailed instructions on how to use the Drug Discovery Platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Training Models](#training-models)
5. [Making Predictions](#making-predictions)
6. [Evaluation](#evaluation)
7. [Advanced Features](#advanced-features)

## Quick Start

### 1. Environment Setup

```bash
# Run the setup script
bash scripts/setup_env.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your dataset files in the appropriate directories:

```
data/
├── raw/
│   ├── drugbank/
│   │   └── drug_target_interactions.csv
│   ├── chembl/
│   │   └── chembl_data.db
│   ├── davis/
│   │   └── davis_data.txt
│   └── kiba/
│       └── kiba_data.txt
```

### 3. Run the Pipeline

```bash
python start.py
```

## Data Preparation

### DrugBank Dataset

The DrugBank dataset should contain drug-target interaction data:

**Required columns:**
- `drug_id`: Unique identifier for the drug
- `drug_name`: Name of the drug
- `smiles`: SMILES representation of the drug molecule
- `target_id`: Unique identifier for the target protein
- `target_name`: Name of the target protein
- `target_sequence`: Amino acid sequence of the target protein
- `interaction`: Binary label (1 for interaction, 0 for no interaction)

### ChEMBL Dataset

Process ChEMBL SQLite database:

```bash
python process_chembl_sqlite.py \
    --db_path data/raw/chembl/chembl_30.db \
    --output_dir data/processed \
    --sample_size 100000
```

### Davis and KIBA Datasets

Process benchmark datasets:

```bash
# Davis dataset
python preprocess_davis.py \
    --input_dir data/raw/davis \
    --output_dir data/processed/davis

# KIBA dataset
python preprocess_kiba.py \
    --input_dir data/raw/kiba \
    --output_dir data/processed/kiba
```

## Configuration

### Basic Configuration

Edit `dti_config.yaml` to customize the model and training:

```yaml
model:
  drug_hidden_channels: 64
  protein_output_channels_component: 64
  
training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  
hardware:
  device: cuda  # or cpu
```

### Advanced Configuration Options

**Model Architecture:**
```yaml
model:
  drug_input_dim: 15
  drug_hidden_channels: 64
  drug_output_channels_component: 64
  drug_num_layers: 3
  drug_dropout: 0.2
  
  protein_input_dim: 480
  protein_output_channels_component: 64
  protein_dropout: 0.2
  
  predictor_hidden_dim1: 256
  predictor_hidden_dim2: 128
  predictor_dropout: 0.2
```

**Training Options:**
```yaml
training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  early_stopping_patience: 15
  pos_weight: 1.0
  use_amp: false  # Mixed precision training
```

**Logging:**
```yaml
logging:
  use_wandb: true
  wandb_project: drug-discovery
  wandb_entity: your-username
  log_interval: 10
  eval_interval: 1
```

## Training Models

### Basic Training

```bash
python train_dti.py
```

### Training with Custom Config

```bash
python train_dti.py --config configs/my_config.yaml
```

### Training Options

```bash
python train_dti.py \
    --config configs/dti_config.yaml \
    --device cuda \
    --batch_size 256 \
    --epochs 100 \
    --learning_rate 0.001
```

### Monitoring Training

**Using TensorBoard:**
```bash
tensorboard --logdir runs
```

**Using wandb:**
1. Set `use_wandb: true` in config
2. Login: `wandb login`
3. Training will automatically log to wandb

## Making Predictions

### Single Prediction

```bash
python predict_dti.py \
    --drug "CCO" \
    --protein "MKFLILLFNILCLFPVLA" \
    --model_path models/best_model.pth
```

### Batch Predictions

```bash
python predict_dti.py \
    --input_file data/test_pairs.csv \
    --model_path models/best_model.pth \
    --output_file predictions.csv
```

### Python API

```python
from predict_dti import DTIPredictor

# Load model
predictor = DTIPredictor(model_path="models/best_model.pth")

# Single prediction
drug_smiles = "CCO"
protein_seq = "MKFLILLFNILCLFPVLA"
score = predictor.predict(drug_smiles, protein_seq)
print(f"Interaction score: {score:.4f}")

# Batch prediction
pairs = [
    ("CCO", "MKFLILLFNILCLFPVLA"),
    ("CC(=O)O", "MSHHWGYGKHNGPEHWHK"),
]
scores = predictor.predict_batch(pairs)
```

## Evaluation

### Evaluate on Test Set

```bash
python train_dti.py --evaluate \
    --model_path models/best_model.pth \
    --test_data data/processed/test_data.pt
```

### Metrics

The model reports the following metrics:
- **Accuracy**: Overall prediction accuracy
- **AUROC**: Area under the ROC curve
- **AUPRC**: Area under the precision-recall curve
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall

### Custom Evaluation Script

```python
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

# Load model and data
model = torch.load("models/best_model.pth")
test_data = torch.load("data/processed/test_data.pt")

# Run predictions
predictions = []
labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        predictions.extend(output.cpu().numpy())
        labels.extend(batch.y.cpu().numpy())

# Calculate metrics
auroc = roc_auc_score(labels, predictions)
accuracy = accuracy_score(labels, (predictions > 0.5).astype(int))

print(f"AUROC: {auroc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
```

## Advanced Features

### Molecular Dynamics Integration

Run molecular dynamics simulations to extract additional features:

```bash
python md_sim.py \
    --input_pdb protein.pdb \
    --output_dir md_results \
    --simulation_time 100  # ns
```

### Feature Extraction

Extract custom molecular features:

```python
from data_process import extract_molecular_features

# Extract features from SMILES
smiles = "CCO"
features = extract_molecular_features(smiles)

# Features include:
# - Molecular graph (nodes, edges)
# - Morgan fingerprints
# - Physicochemical properties
```

### Multi-GPU Training

Enable data parallelism in config:

```yaml
hardware:
  use_data_parallel: true
  device: cuda
```

### Custom Model Architecture

Modify `dti_model.py` to implement custom architectures:

```python
class CustomDTIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define your custom architecture
        
    def forward(self, data):
        # Implement forward pass
        return prediction
```

### Hyperparameter Tuning

Use wandb sweeps for hyperparameter optimization:

```yaml
# sweep_config.yaml
program: train_dti.py
method: bayes
metric:
  name: val_auroc
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [128, 256, 512]
  dropout:
    min: 0.1
    max: 0.5
```

```bash
wandb sweep sweep_config.yaml
wandb agent your-sweep-id
```

## Troubleshooting

### Out of Memory Errors

- Reduce batch size in config
- Enable gradient checkpointing
- Use mixed precision training (`use_amp: true`)

### Slow Training

- Increase `num_workers` for data loading
- Use GPU if available
- Enable mixed precision training
- Reduce model size

### Poor Performance

- Increase training epochs
- Adjust learning rate
- Try different model architectures
- Ensure data quality and balance
- Use data augmentation

## Additional Resources

- [API Documentation](API.md)
- [Model Architecture Details](ARCHITECTURE.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [FAQ](FAQ.md)

## Support

For issues or questions:
1. Check the documentation
2. Search existing GitHub issues
3. Open a new issue with detailed information

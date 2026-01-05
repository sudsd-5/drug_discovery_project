# Configuration Files

This directory contains configuration files for the Drug Discovery Platform.

## Available Configurations

### dti_config.yaml
Main configuration file for drug-target interaction prediction.

**Usage:**
```bash
python train_dti.py --config configs/dti_config.yaml
```

## Configuration Structure

### Data Section
```yaml
data:
  interactions_dir: data/processed/interactions
  raw_dir: data/raw
  processed_dir: data/processed
```

### Model Section
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
```

### Training Section
```yaml
training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  early_stopping_patience: 15
```

### Hardware Section
```yaml
hardware:
  device: cuda  # or cpu
  num_workers: 0
  pin_memory: false
```

### Logging Section
```yaml
logging:
  log_dir: logs
  save_model_dir: models
  tensorboard_dir: runs
  use_wandb: false
```

## Creating Custom Configurations

1. Copy the default config:
   ```bash
   cp configs/dti_config.yaml configs/my_config.yaml
   ```

2. Edit your configuration:
   ```bash
   nano configs/my_config.yaml
   ```

3. Use your custom config:
   ```bash
   python train_dti.py --config configs/my_config.yaml
   ```

## Configuration Examples

### High-Performance Training
```yaml
training:
  batch_size: 512
  learning_rate: 0.0005
  use_amp: true

hardware:
  device: cuda
  use_data_parallel: true
  num_workers: 4
  pin_memory: true
```

### CPU-Only Training
```yaml
training:
  batch_size: 64
  epochs: 50

hardware:
  device: cpu
  num_workers: 4
```

### Debugging Configuration
```yaml
training:
  batch_size: 32
  epochs: 10
  early_stopping_patience: 3

logging:
  log_interval: 1
  eval_interval: 1
```

## Environment Variable Overrides

You can override configuration values using environment variables:

```bash
export DTI_DEVICE=cuda
export DTI_BATCH_SIZE=256
export DTI_LEARNING_RATE=0.001

python train_dti.py
```

## Validation

Configurations are automatically validated on load. Check for:
- Valid paths
- Reasonable parameter ranges
- Compatible settings
- Required fields present

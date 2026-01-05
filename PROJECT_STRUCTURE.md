# Project Structure

```
drug_discovery_project/
├── .editorconfig                       # Editor configuration
├── .env.example                        # Environment variables template
├── .flake8                            # Flake8 linting configuration
├── .gitignore                         # Git ignore patterns
├── CHANGELOG.md                       # Project changelog
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT License
├── MANIFEST.in                        # Package manifest for distribution
├── Makefile                           # Common commands automation
├── README.md                          # Main documentation (English)
├── readme.md                          # Documentation (Chinese)
├── pytest.ini                         # Pytest configuration
├── pyproject.toml                     # Python project configuration
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup configuration
│
├── configs/                           # Configuration files
│   ├── README.md                      # Config documentation
│   └── dti_config.yaml               # Main DTI configuration
│
├── data/                              # Data directory
│   ├── raw/                          # Raw datasets
│   │   ├── .gitkeep
│   │   ├── drugbank/                 # DrugBank data
│   │   ├── chembl/                   # ChEMBL data
│   │   ├── davis/                    # Davis dataset
│   │   └── kiba/                     # KIBA dataset
│   └── processed/                    # Processed features
│       └── .gitkeep
│
├── docs/                              # Documentation
│   ├── DATA_FORMAT.md                # Data format specification
│   ├── QUICKREF.md                   # Quick reference guide
│   ├── ROADMAP.md                    # Project roadmap
│   ├── TODO.md                       # Todo list
│   └── USAGE.md                      # Detailed usage guide
│
├── examples/                          # Example scripts and notebooks
│   ├── README.md                     # Examples documentation
│   └── basic_usage.ipynb             # Basic usage notebook
│
├── logs/                              # Training logs
│   └── .gitkeep
│
├── models/                            # Saved model checkpoints
│   └── .gitkeep
│
├── output/                            # Prediction outputs
│   └── .gitkeep
│
├── runs/                              # TensorBoard logs
│   └── .gitkeep
│
├── scripts/                           # Utility scripts
│   ├── format_code.sh                # Code formatting script
│   ├── run_tests.sh                  # Test runner script
│   └── setup_env.sh                  # Environment setup script
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_config.py                # Configuration tests
│   └── test_data_process.py          # Data processing tests
│
└── Core Python Files:
    ├── data_process.py               # Data processing utilities
    ├── dti_config.yaml               # DTI configuration (root)
    ├── dti_model.py                  # Hybrid DTI model architecture
    ├── md_sim.py                     # Molecular dynamics simulation
    ├── predict_dti.py                # Inference script
    ├── preprocess_davis.py           # Davis dataset preprocessing
    ├── preprocess_kiba.py            # KIBA dataset preprocessing
    ├── process_chembl_sqlite.py      # ChEMBL database processing
    ├── process_drug_target_interactions.py  # DrugBank processing
    ├── start.py                      # Pipeline orchestration
    └── train_dti.py                  # Training script
```

## Directory Purposes

### Configuration (`configs/`)
Contains YAML configuration files for model architecture, training parameters, and system settings.

### Data (`data/`)
- `raw/`: Original datasets from various sources
- `processed/`: Preprocessed features ready for model training

### Documentation (`docs/`)
Comprehensive documentation including:
- Usage guides
- Data format specifications
- API references
- Development roadmap

### Examples (`examples/`)
Jupyter notebooks and Python scripts demonstrating platform usage.

### Logs (`logs/`)
Training logs and experiment records.

### Models (`models/`)
Saved model checkpoints and weights.

### Output (`output/`)
Generated predictions and analysis results.

### Runs (`runs/`)
TensorBoard event files for visualization.

### Scripts (`scripts/`)
Utility shell scripts for common development tasks.

### Tests (`tests/`)
Unit and integration tests using pytest.

## Main Components

### Data Processing
- `data_process.py`: Core data processing utilities
- `process_drug_target_interactions.py`: DrugBank data processing
- `process_chembl_sqlite.py`: ChEMBL database handling
- `preprocess_davis.py`: Davis dataset preprocessing
- `preprocess_kiba.py`: KIBA dataset preprocessing

### Model
- `dti_model.py`: Hybrid neural network architecture
  - Drug encoder (GAT)
  - Protein encoder (ESM-2)
  - Fusion module
  - Predictor

### Training & Inference
- `train_dti.py`: Model training with early stopping
- `predict_dti.py`: Inference on new drug-target pairs
- `start.py`: Complete pipeline orchestration

### Simulation
- `md_sim.py`: Molecular dynamics simulation integration

## Configuration Files

### Python
- `setup.py`: Package installation configuration
- `pyproject.toml`: Modern Python project configuration
- `pytest.ini`: Test configuration
- `.flake8`: Linting rules
- `MANIFEST.in`: Package distribution files

### Development
- `.gitignore`: Version control exclusions
- `.editorconfig`: Editor settings
- `.env.example`: Environment variable template
- `Makefile`: Task automation

## Key Features by File

### dti_model.py
- `DTIHybridPredictor`: Main model class
- `DrugEncoder2Real2Imag`: Drug graph encoder
- `ProteinEncoder`: Protein sequence encoder
- Attention-based fusion

### train_dti.py
- Training loop with validation
- Early stopping
- Learning rate scheduling
- TensorBoard/wandb logging
- Checkpoint management

### predict_dti.py
- Single prediction
- Batch processing
- Model loading
- Result export

## Development Workflow

1. **Setup**: `bash scripts/setup_env.sh`
2. **Configure**: Edit `configs/dti_config.yaml`
3. **Prepare Data**: Place files in `data/raw/`
4. **Train**: `python train_dti.py`
5. **Evaluate**: `tensorboard --logdir runs`
6. **Predict**: `python predict_dti.py`
7. **Test**: `bash scripts/run_tests.sh`

## Adding New Components

### New Dataset Processor
1. Create `preprocess_<dataset>.py`
2. Follow existing processor structure
3. Add tests in `tests/test_<dataset>.py`
4. Update documentation

### New Model Architecture
1. Extend classes in `dti_model.py`
2. Update configuration schema
3. Add unit tests
4. Document changes

### New Feature
1. Implement in appropriate module
2. Add configuration options
3. Write tests
4. Update documentation
5. Add examples

## Best Practices

### Code Organization
- Keep related functionality together
- Use clear, descriptive names
- Follow PEP 8 style guide
- Add docstrings to all functions

### Documentation
- Update relevant docs with changes
- Keep examples up to date
- Document configuration options
- Add inline comments for complex logic

### Testing
- Write tests for new features
- Maintain high test coverage
- Test edge cases
- Run tests before commits

### Version Control
- Use meaningful commit messages
- Keep commits focused and atomic
- Update CHANGELOG.md
- Tag releases appropriately

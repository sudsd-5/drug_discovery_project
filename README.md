# Drug Discovery Platform

[中文文档](readme.md) | English

A deep learning-based drug discovery platform focused on drug-target interaction prediction, molecular property prediction, and virtual screening.

## Overview

This project implements an end-to-end computational drug discovery platform leveraging state-of-the-art deep learning techniques, including Graph Neural Networks (GNN) and Convolutional Neural Networks (CNN), to predict interactions between drug candidate molecules and protein targets. Key features include:

1. **Drug-Target Interaction Prediction**: Predict the interaction probability between drug molecules and protein targets
2. **Molecular Property Prediction**: Predict physicochemical properties and biological activities of drug molecules
3. **Virtual Screening**: Screen large compound libraries for candidate molecules that may interact with specific targets

## Architecture

The platform uses a hybrid dual-stream neural network architecture:

- **Drug Encoder**: Multi-layer Graph Attention Network (GAT) processing molecular graph structures with complex-valued representations
- **Protein Encoder**: ESM-2 protein language model embeddings with projection layers
- **Fusion Module**: Attention-based fusion of drug-protein representations
- **Predictor**: Multi-layer perceptron for interaction prediction

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- transformers (HuggingFace)
- scikit-learn, NumPy, pandas
- matplotlib, seaborn
- TensorBoard / wandb

Recommended hardware:
- CUDA-compatible GPU (e.g., RTX 4060 or better)
- 16GB+ RAM
- 100GB+ storage

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drug_discovery_project.git
   cd drug_discovery_project
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n drug_disc python=3.8
   conda activate drug_disc
   ```
   
   Or using venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Data Preparation

The platform supports multiple datasets:
- DrugBank
- ChEMBL
- Davis
- KIBA

Place your data files in the appropriate directories:
```
data/
├── raw/
│   ├── drugbank/
│   │   └── drug_target_interactions.csv
│   ├── chembl/
│   ├── davis/
│   └── kiba/
└── processed/
```

### Run the Pipeline

Use the startup script to run the complete pipeline:

```bash
python start.py
```

Or run individual components:

```bash
# Data preprocessing
python process_drug_target_interactions.py

# Model training
python train_dti.py

# Prediction
python predict_dti.py --drug SMILES_STRING --protein PROTEIN_SEQUENCE
```

### Configuration

Edit `dti_config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data paths
- Hardware settings (GPU/CPU)

### Visualization

View training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

Or use wandb for experiment tracking (requires account setup).

## Project Structure

```
drug_discovery_project/
├── configs/                    # Configuration files
│   └── dti_config.yaml        # DTI prediction config
├── data/                       # Data directory
│   ├── raw/                   # Raw datasets
│   │   ├── drugbank/
│   │   ├── chembl/
│   │   ├── davis/
│   │   └── kiba/
│   └── processed/             # Processed features
├── logs/                       # Training logs
├── models/                     # Saved model checkpoints
├── runs/                       # TensorBoard logs
├── tests/                      # Unit tests
├── dti_model.py               # Hybrid DTI model architecture
├── train_dti.py               # Training script
├── predict_dti.py             # Inference script
├── data_process.py            # Data processing utilities
├── process_*.py               # Dataset-specific processors
├── md_sim.py                  # Molecular dynamics simulation
├── start.py                   # Pipeline orchestration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Model Performance

The model is evaluated using:
- **Accuracy**: Overall prediction accuracy
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the precision-recall curve

Typical performance on benchmark datasets:
- Davis: AUC-ROC ~0.89
- KIBA: AUC-ROC ~0.87

## Features

### Dataset Processing
- DrugBank XML/CSV parsing
- ChEMBL SQLite database sampling
- Davis and KIBA benchmark preprocessing
- SMILES to molecular graph conversion
- Morgan fingerprint generation
- ESM-2 protein embeddings

### Model Training
- Mixed precision training support
- Early stopping with patience
- Learning rate scheduling (ReduceLROnPlateau)
- Class imbalance handling with weighted loss
- TensorBoard and wandb logging
- Automatic checkpoint saving

### Molecular Dynamics
- GROMACS integration for MD simulations
- MDAnalysis-based feature extraction
- Trajectory analysis

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines. Format code with:

```bash
black .
flake8 .
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{DrugDiscoveryProject,
  author = {Drug Discovery Team},
  title = {Deep Learning-Based Drug Discovery Platform},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/drug_discovery_project}
}
```

## Acknowledgments

- PyTorch and PyTorch Geometric teams
- RDKit developers
- HuggingFace for transformers library
- The drug discovery research community

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## References

- Öztürk, H., et al. (2018). DeepDTA: deep drug-target binding affinity prediction.
- Nguyen, T., et al. (2021). GraphDTA: predicting drug–target binding affinity with graph neural networks.
- ESM-2: Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model.

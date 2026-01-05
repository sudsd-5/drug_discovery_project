# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README with English documentation
- .gitignore file for Python projects
- LICENSE file (MIT)
- Testing framework with pytest
- Configuration file with relative paths
- Contributing guidelines
- Development tools support (black, flake8, isort)
- TensorBoard logging support
- wandb integration option

### Changed
- Updated dti_config.yaml to use relative paths instead of absolute paths
- Enhanced requirements.txt with missing dependencies
- Improved project structure documentation

### Fixed
- Configuration file compatibility across different operating systems
- Path handling for cross-platform support

## [0.1.0] - 2024-01-05

### Added
- Initial project structure
- Drug-target interaction prediction model
- Data preprocessing for multiple datasets (DrugBank, ChEMBL, Davis, KIBA)
- Training pipeline with early stopping
- Prediction script for inference
- Molecular dynamics simulation support
- Chinese documentation (readme.md)

### Features
- Graph Attention Network (GAT) for drug encoding
- ESM-2 protein embeddings
- Hybrid model architecture with complex-valued representations
- Multi-dataset support
- TensorBoard visualization
- Checkpoint saving and loading

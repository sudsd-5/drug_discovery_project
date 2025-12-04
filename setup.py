from setuptools import setup, find_packages

setup(
    name="drug_discovery",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "rdkit>=2023.03.0",
        "biopython>=1.79",
        "openbabel>=3.1.1",
        "mdanalysis>=2.4.0",
        "gromacs>=2023.2",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "wandb>=0.12.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.62.0",
        "pytest>=6.2.5",
        "jupyter>=1.0.0"
    ],
) 
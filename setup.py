from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="drug_discovery",
    version="0.1.0",
    author="Drug Discovery Team",
    author_email="contact@example.com",
    description="Deep learning-based drug discovery platform for drug-target interaction prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drug_discovery_project",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "rdkit>=2023.03.0",
        "biopython>=1.79",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.10.0",
        "wandb>=0.12.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "md": [
            "mdanalysis>=2.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dti-train=train_dti:main",
            "dti-predict=predict_dti:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="drug-discovery machine-learning deep-learning drug-target-interaction bioinformatics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/drug_discovery_project/issues",
        "Source": "https://github.com/yourusername/drug_discovery_project",
        "Documentation": "https://github.com/yourusername/drug_discovery_project#readme",
    },
)

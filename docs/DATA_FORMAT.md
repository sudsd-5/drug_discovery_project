# Data Format Specification

This document describes the expected data formats for the Drug Discovery Platform.

## Table of Contents

1. [Drug-Target Interaction Data](#drug-target-interaction-data)
2. [Molecular Properties Data](#molecular-properties-data)
3. [Processed Data Format](#processed-data-format)
4. [Dataset-Specific Formats](#dataset-specific-formats)

## Drug-Target Interaction Data

### CSV Format

**File:** `data/raw/drugbank/drug_target_interactions.csv`

**Required Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| drug_id | string | Unique drug identifier | DB00001 |
| drug_name | string | Drug name | Lepirudin |
| smiles | string | SMILES representation | CC(C)C[C@H]... |
| target_id | string | Unique target identifier | P00734 |
| target_name | string | Target protein name | Prothrombin |
| target_sequence | string | Amino acid sequence | MWVPVVFLT... |
| interaction | int | Binary label (0 or 1) | 1 |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| affinity | float | Binding affinity (Kd, Ki, IC50) |
| affinity_type | string | Type of affinity measurement |
| source | string | Data source |
| pubmed_id | string | PubMed reference |

**Example:**

```csv
drug_id,drug_name,smiles,target_id,target_name,target_sequence,interaction
DB00001,Lepirudin,CC(C)NCC(=O)N,P00734,Prothrombin,MWVPVVFLTLSVTWICGESLADT,1
DB00002,Cetuximab,CCCN(CC)CC,P00533,EGFR,MRPSGTAGAALLALLAALCPASRA,1
DB00003,Dornase alfa,CCO,P24855,DNase,MRGMKLLGALLALAALLQGAVS,0
```

## Molecular Properties Data

### CSV Format

**File:** `data/raw/molecular_properties.csv`

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| drug_id | string | Unique drug identifier |
| smiles | string | SMILES representation |
| molecular_weight | float | Molecular weight (g/mol) |
| logp | float | Partition coefficient |
| hbd | int | Hydrogen bond donors |
| hba | int | Hydrogen bond acceptors |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| tpsa | float | Topological polar surface area |
| num_rotatable_bonds | int | Number of rotatable bonds |
| num_aromatic_rings | int | Number of aromatic rings |
| lipinski_violations | int | Number of Lipinski rule violations |
| bioavailability | float | Oral bioavailability |

**Example:**

```csv
drug_id,smiles,molecular_weight,logp,hbd,hba,tpsa
DB00001,CC(C)NCC(=O)N,117.15,0.54,2,2,55.12
DB00002,CCCN(CC)CC,129.25,1.85,0,1,12.03
DB00003,CCO,46.07,-0.18,1,1,20.23
```

## Processed Data Format

### PyTorch Geometric Data Objects

After preprocessing, data is stored as PyTorch Geometric `Data` objects:

```python
Data(
    x=tensor,              # Node features [num_nodes, num_node_features]
    edge_index=tensor,     # Edge connectivity [2, num_edges]
    edge_attr=tensor,      # Edge features [num_edges, num_edge_features]
    y=tensor,              # Labels [num_graphs] or [num_nodes]
    protein_feat=tensor,   # Protein features [protein_feat_dim]
    smiles=str,            # Original SMILES string
    protein_seq=str,       # Original protein sequence
    drug_id=str,           # Drug identifier
    target_id=str          # Target identifier
)
```

### Node Features (x)

For each atom in the molecule (15 features):

1. **Atom type** (one-hot): C, N, O, S, F, P, Cl, Br, I, Other (10 features)
2. **Degree** (one-hot): 0, 1, 2, 3, 4+ (5 features)

### Edge Features (edge_attr)

For each bond (4 features):

1. **Bond type** (one-hot): Single, Double, Triple, Aromatic (4 features)

### Protein Features (protein_feat)

- **ESM-2 embeddings**: 480-dimensional protein language model embeddings
- Extracted using `facebook/esm2_t6_8M_UR50D` or similar model

## Dataset-Specific Formats

### DrugBank

**Source:** https://go.drugbank.com/

**Format:** XML or CSV export

**Key Features:**
- Comprehensive drug information
- Approved drugs and experimental compounds
- Detailed target information
- Clinical data

### ChEMBL

**Source:** https://www.ebi.ac.uk/chembl/

**Format:** SQLite database or CSV

**Key Features:**
- Bioactivity data
- SAR information
- Multiple assay types
- IC50, Ki, Kd values

**Table Structure:**

```sql
SELECT 
    molecule_dictionary.chembl_id as drug_id,
    compound_structures.canonical_smiles as smiles,
    target_dictionary.chembl_id as target_id,
    component_sequences.sequence as target_sequence,
    activities.standard_value as affinity
FROM activities
JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
JOIN compound_structures ON molecule_dictionary.molregno = compound_structures.molregno
JOIN assays ON activities.assay_id = assays.assay_id
JOIN target_dictionary ON assays.tid = target_dictionary.tid
JOIN target_components ON target_dictionary.tid = target_components.tid
JOIN component_sequences ON target_components.component_id = component_sequences.component_id
```

### Davis Dataset

**Source:** Kinase inhibitor bioactivity data

**Format:** Text files with matrices

**Structure:**
```
davis/
├── proteins.txt          # Protein sequences
├── ligands.txt           # Drug SMILES
├── Y.txt                 # Binding affinity matrix
└── folds/                # Pre-defined train/test splits
    ├── train_fold_0.txt
    ├── test_fold_0.txt
    └── ...
```

**Proteins.txt:**
```
MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVD...
MSSNPPKKNSTAPAGPGPEGPAGGSAAPVPAAAGSGSNGVPLKMNHFSG...
```

**Ligands.txt:**
```
CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N
CN(C)CCC=C1C2=CC=CC=C2CCC3=C1C=C(C=C3)O
```

**Y.txt (affinity matrix):**
```
5.0 7.3 4.2 ...
6.1 5.5 8.0 ...
...
```

### KIBA Dataset

**Source:** Kinase-inhibitor bioactivity aggregated from multiple sources

**Format:** Similar to Davis, with KIBA scores

**KIBA Score:** Aggregated score combining Ki, Kd, and IC50 values

**Structure:**
```
kiba/
├── proteins.txt          # Protein sequences
├── ligands.txt           # Drug SMILES
├── Y.txt                 # KIBA score matrix
└── folds/                # Pre-defined splits
```

## Data Validation

### SMILES Validation

```python
from rdkit import Chem

def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Example
smiles = "CCO"
is_valid = validate_smiles(smiles)
```

### Protein Sequence Validation

```python
def validate_protein_sequence(sequence):
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(aa in valid_amino_acids for aa in sequence)

# Example
sequence = "MKFLILLFNILCLFPVLA"
is_valid = validate_protein_sequence(sequence)
```

### Data Quality Checks

1. **No missing values** in required columns
2. **Valid SMILES** strings (parseable by RDKit)
3. **Valid protein sequences** (only standard amino acids)
4. **Label balance** (check distribution of positive/negative samples)
5. **Duplicate removal** (unique drug-target pairs)

## Creating Custom Datasets

### Template CSV

```python
import pandas as pd

# Create template
data = {
    'drug_id': ['DRUG001', 'DRUG002', 'DRUG003'],
    'drug_name': ['DrugA', 'DrugB', 'DrugC'],
    'smiles': ['CCO', 'CC(=O)O', 'CC(C)O'],
    'target_id': ['TARGET001', 'TARGET002', 'TARGET003'],
    'target_name': ['ProteinA', 'ProteinB', 'ProteinC'],
    'target_sequence': ['MKFLILLFN...', 'MSHHWGYGK...', 'MWVPVVFLT...'],
    'interaction': [1, 0, 1]
}

df = pd.DataFrame(data)
df.to_csv('data/raw/custom_dataset.csv', index=False)
```

### Processing Custom Data

```python
from data_process import process_dti_data

# Process custom dataset
process_dti_data(
    input_file='data/raw/custom_dataset.csv',
    output_dir='data/processed/custom',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

## Data Statistics

Recommended dataset sizes for training:

| Split | Minimum | Recommended | Large-scale |
|-------|---------|-------------|-------------|
| Train | 1,000 | 10,000 | 100,000+ |
| Validation | 200 | 1,000 | 10,000+ |
| Test | 200 | 1,000 | 10,000+ |

## References

1. DrugBank: https://go.drugbank.com/
2. ChEMBL: https://www.ebi.ac.uk/chembl/
3. Davis et al., Nature Biotechnology, 2011
4. Tang et al., Bioinformatics, 2014 (KIBA)
5. RDKit: https://www.rdkit.org/
6. ESM-2: https://github.com/facebookresearch/esm

# 保存为 E:\AI\drug_discovery_project\src\preprocess.py
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from transformers import EsmModel, EsmTokenizer
import os
from multiprocessing import Pool, cpu_count


def smiles_to_graph(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() in ['', 'N/A']:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        atom_features = []
        for atom in mol.GetAtoms():
            feature = [
                atom.GetAtomicNum(), atom.GetDegree(), atom.GetTotalNumHs(), atom.GetIsAromatic(),
                atom.GetFormalCharge(), atom.GetNumImplicitHs(), atom.IsInRing(),
                atom.GetHybridization().real, atom.GetTotalDegree(), atom.GetTotalValence(),
                atom.GetMass(), atom.GetExplicitValence(), atom.GetImplicitValence(),
                1 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0,
                atom.GetNumRadicalElectrons()
            ]
            atom_features.append(feature[:15])
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


def generate_dynamics_data(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() in ['', 'N/A']:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        dynamics_traj = np.random.randn(10, num_atoms, 15)
        return torch.tensor(dynamics_traj, dtype=torch.float)
    except Exception:
        return None


def process_row(args):
    index, row, label_map = args
    try:
        smiles = row['smiles']
        drug_graph = smiles_to_graph(smiles)
        if drug_graph is None:
            return (index, None, None, None, 'invalid_smiles')

        dynamics = generate_dynamics_data(smiles)
        if dynamics is None:
            return (index, None, None, None, 'parse_error')

        label = row['label']
        if pd.isna(label):
            label = 0.0
        elif isinstance(label, str):
            label_str = label.strip().lower()
            label = label_map.get(label_str)
            if label is None:
                return (index, None, None, None, 'unknown_label')
        else:
            label = float(label)

        return (index, drug_graph, dynamics, label, None)
    except Exception as e:
        return (index, None, None, None, f"parse_error: {str(e)}")


def batch_generate_protein_embeddings(sequences, batch_size=32):
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    model = EsmModel.from_pretrained('facebook/esm2_t12_35M_UR50D').cuda()
    sequences = [seq if pd.notna(seq) and isinstance(seq, str) and seq.strip() not in ['', 'N/A'] else 'A' for seq in
                 sequences]

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True, max_length=1000)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        embeddings.append(batch_embeddings)
        print(f"Processed embedding batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size}")
        torch.cuda.empty_cache()  # 清理显存
    return torch.cat(embeddings, dim=0)


def load_data(file_path_drugbank, file_path_chembl):
    print(f"Loading data from DrugBank: {file_path_drugbank}")
    df_drugbank = pd.read_csv(file_path_drugbank)
    print(f"Loading data from ChEMBL: {file_path_chembl}")
    df_chembl = pd.read_csv(file_path_chembl)

    df_drugbank = df_drugbank[['smiles', 'target_sequence', 'label']]
    df_chembl = df_chembl[['smiles', 'target_sequence', 'label']]
    df = pd.concat([df_drugbank, df_chembl], ignore_index=True)
    print(f"Combined data loaded. Number of rows: {len(df)}")

    label_map = {
        'inhibitor': 0.0, 'antagonist': 0.0, 'agonist': 1.0, 'activator': 1.0, 'binder': 1.0,
        'degradation': 0.0, 'modulator': 1.0, 'car t-cell therapy': 1.0, 'potentiator': 1.0,
        'chelator': 0.0, 'ligand': 1.0, 'blocker': 0.0
    }

    print("Generating protein embeddings...")
    target_embeddings = batch_generate_protein_embeddings(df['target_sequence'].tolist(), batch_size=32)

    print("Processing SMILES and dynamics data with multiprocessing...")
    drug_graphs = [None] * len(df)
    dynamics_data = [None] * len(df)
    interactions = [None] * len(df)
    skip_counts = {'invalid_smiles': 0, 'unknown_label': 0, 'parse_error': 0}
    unknown_labels = set()

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_row, [(i, row, label_map) for i, row in df.iterrows()])

    for index, drug_graph, dynamics, label, error in results:
        if error:
            if error.startswith('parse_error'):
                skip_counts['parse_error'] += 1
                print(f"Error processing row {index}: {error}")
            elif error == 'invalid_smiles':
                skip_counts['invalid_smiles'] += 1
                print(f"Skipping row {index}: Invalid SMILES")
            elif error == 'unknown_label':
                skip_counts['unknown_label'] += 1
                unknown_labels.add(df.iloc[index]['label'])
                print(f"Skipping row {index}: Unknown label")
            continue
        drug_graphs[index] = drug_graph
        dynamics_data[index] = dynamics
        interactions[index] = label
        print(f"Row {index} processed")

    valid_indices = [i for i, x in enumerate(drug_graphs) if x is not None]
    drug_graphs = [drug_graphs[i] for i in valid_indices]
    dynamics_data = [dynamics_data[i] for i in valid_indices]
    target_embeddings = [target_embeddings[i] for i in valid_indices]
    interactions = [interactions[i] for i in valid_indices]

    print(f"Processed {len(drug_graphs)} samples successfully")
    print(f"Skipped rows: {skip_counts}")
    print(f"Unknown labels encountered: {unknown_labels}")

    output_dir = 'E:/AI/drug_discovery_project/data/processed/interactions'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(drug_graphs, os.path.join(output_dir, 'drug_graphs.pt'))
    torch.save(dynamics_data, os.path.join(output_dir, 'dynamics_data.pt'))
    torch.save(target_embeddings, os.path.join(output_dir, 'target_embeddings.pt'))
    torch.save(torch.tensor(interactions, dtype=torch.float), os.path.join(output_dir, 'interactions.pt'))

    from sklearn.model_selection import train_test_split
    indices = np.arange(len(drug_graphs))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    torch.save(train_idx, os.path.join(output_dir, 'train_idx.pt'))
    torch.save(val_idx, os.path.join(output_dir, 'val_idx.pt'))
    torch.save(test_idx, os.path.join(output_dir, 'test_idx.pt'))

    print(f"Data saved to {output_dir}")
    return drug_graphs, dynamics_data, target_embeddings, interactions


def main():
    file_path_drugbank = 'E:/AI/drug_discovery_project/data/raw/drugbank/drug_target_interactions.csv'
    file_path_chembl = 'E:/AI/drug_discovery_project/data/raw/chembl/molecular_properties.csv'
    drug_graphs, dynamics_data, target_embeddings, interactions = load_data(file_path_drugbank, file_path_chembl)


if __name__ == "__main__":
    main()
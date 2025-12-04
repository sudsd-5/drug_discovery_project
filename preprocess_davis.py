# 保存为 E:\AI\drug_discovery_project\src\preprocess_davis.py
import json
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from sklearn.model_selection import train_test_split
import os
from transformers import EsmModel, EsmTokenizer
import torch.multiprocessing as mp
from torch.cuda.amp import autocast  # 混合精度计算

# 设置全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 药物处理（在GPU上运行）
def smiles_to_graph(smiles):
    if not isinstance(smiles, str) or smiles.strip() in ['', 'N/A']:
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
        x = torch.tensor(atom_features, dtype=torch.float).to(device)  # 直接在GPU上创建张量

        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)  # 直接在GPU上创建张量

        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None

# 2. 动态数据（在GPU上运行）
def generate_dynamics_data(smiles):
    if not isinstance(smiles, str) or smiles.strip() in ['', 'N/A']:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        dynamics_traj = np.random.randn(10, num_atoms, 15)
        return torch.tensor(dynamics_traj, dtype=torch.float).to(device)  # 直接在GPU上创建张量
    except Exception:
        return None

# 3. 处理单条数据
def process_row(args):
    index, drug_id, smiles, protein_id, sequence, label = args
    try:
        drug_graph = smiles_to_graph(smiles)
        if drug_graph is None:
            return (index, None, None, None, 'invalid_smiles')

        dynamics = generate_dynamics_data(smiles)
        if dynamics is None:
            return (index, None, None, None, 'parse_error')

        return (index, drug_graph, dynamics, label, None)
    except Exception as e:
        return (index, None, None, None, f"parse_error: {str(e)}")

# 4. 批量生成蛋白嵌入（在GPU上运行）
def batch_generate_protein_embeddings(sequences, batch_size=16):
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    model = EsmModel.from_pretrained('facebook/esm2_t12_35M_UR50D').to(device)  # 将模型移到GPU
    model.eval()  # 确保模型处于评估模式
    sequences = [seq if isinstance(seq, str) and seq.strip() not in ['', 'N/A'] else 'A' for seq in sequences]

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True, max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入数据移到GPU
        with torch.no_grad():
            with autocast():  # 使用混合精度计算，减少显存占用
                outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()  # 结果移回CPU以节省显存
        embeddings.append(batch_embeddings)
        print(f"Processed embedding batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size}")
        torch.cuda.empty_cache()  # 清理显存
    return torch.cat(embeddings, dim=0)

# 5. 主函数：处理 Davis 数据
def process_davis_data():
    data_dir = "E:\\AI\\drug_discovery_project\\data\\davis"
    output_dir = "E:\\AI\\drug_discovery_project\\data\\processed\\interactions"
    os.makedirs(output_dir, exist_ok=True)

    # 读取 SMILES
    with open(os.path.join(data_dir, 'ligands.can.txt'), 'r') as f:
        ligands = json.load(f)
    drug_ids = list(ligands.keys())
    smiles_list = list(ligands.values())

    # 读取蛋白序列
    with open(os.path.join(data_dir, 'proteins.txt'), 'r') as f:
        proteins = json.load(f)
    protein_ids = list(proteins.keys())
    sequences = list(proteins.values())

    # 读取 Kd 矩阵
    Kd = np.loadtxt(os.path.join(data_dir, 'drug-target_interaction_affinities_Kd_Davis_et_al_2011v1.txt'))
    interactions = (Kd < 30).astype(float).flatten()

    # 生成蛋白嵌入
    print("Generating protein embeddings on GPU...")
    target_embeddings = batch_generate_protein_embeddings(sequences, batch_size=16)

    # 配对药物和靶点
    data_pairs = []
    idx = 0
    for i, drug_id in enumerate(drug_ids):
        for j, protein_id in enumerate(protein_ids):
            data_pairs.append((idx, drug_id, ligands[drug_id], protein_id, proteins[protein_id], interactions[idx]))
            idx += 1

    # 顺序处理 SMILES 和动态数据（在GPU上运行）
    print("Processing SMILES and dynamics data sequentially on GPU...")
    drug_graphs = [None] * len(data_pairs)
    dynamics_data = [None] * len(data_pairs)
    final_interactions = [None] * len(data_pairs)
    skip_counts = {'invalid_smiles': 0, 'parse_error': 0}

    for idx, pair in enumerate(data_pairs):
        result = process_row(pair)
        index, drug_graph, dynamics, label, error = result
        if error:
            if error.startswith('parse_error'):
                skip_counts['parse_error'] += 1
                print(f"Error processing pair {index}: {error}")
            elif error == 'invalid_smiles':
                skip_counts['invalid_smiles'] += 1
                print(f"Skipping pair {index}: Invalid SMILES")
            continue
        drug_graphs[index] = drug_graph
        dynamics_data[index] = dynamics
        final_interactions[index] = label
        print(f"Pair {index} processed")
        torch.cuda.empty_cache()  # 清理显存

    # 过滤无效数据
    valid_indices = [i for i, x in enumerate(drug_graphs) if x is not None]
    drug_graphs = [drug_graphs[i] for i in valid_indices]
    dynamics_data = [dynamics_data[i] for i in valid_indices]
    target_embeddings_expanded = []
    for i in range(len(drug_ids)):
        for j in range(len(protein_ids)):
            if i * len(protein_ids) + j in valid_indices:
                target_embeddings_expanded.append(target_embeddings[j])
    interactions = [final_interactions[i] for i in valid_indices]

    print(f"Processed {len(drug_graphs)} samples successfully")
    print(f"Skipped pairs: {skip_counts}")

    # 将数据移回CPU保存
    drug_graphs = [Data(x=g.x.cpu(), edge_index=g.edge_index.cpu()) for g in drug_graphs]
    dynamics_data = [d.cpu() for d in dynamics_data]
    target_embeddings_expanded = [t.cpu() for t in target_embeddings_expanded]
    interactions = torch.tensor(interactions, dtype=torch.float).cpu()

    # 保存
    torch.save(drug_graphs, os.path.join(output_dir, 'drug_graphs.pt'))
    torch.save(dynamics_data, os.path.join(output_dir, 'dynamics_data.pt'))
    torch.save(target_embeddings_expanded, os.path.join(output_dir, 'target_embeddings.pt'))
    torch.save(interactions, os.path.join(output_dir, 'interactions.pt'))

    # 划分索引
    indices = np.arange(len(drug_graphs))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    torch.save(torch.tensor(train_idx), os.path.join(output_dir, 'train_idx.pt'))
    torch.save(torch.tensor(val_idx), os.path.join(output_dir, 'val_idx.pt'))
    torch.save(torch.tensor(test_idx), os.path.join(output_dir, 'test_idx.pt'))

    print(f"Data saved to {output_dir}")
    return drug_graphs, dynamics_data, target_embeddings_expanded, interactions

def main():
    drug_graphs, dynamics_data, target_embeddings, interactions = process_davis_data()

if __name__ == "__main__":
    main()
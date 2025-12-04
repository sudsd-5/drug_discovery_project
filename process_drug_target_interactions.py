import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path='configs/dti_config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def smiles_to_graph(smiles):
    """将SMILES字符串转换为图结构数据"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 生成原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetIsAromatic(),
            ]
            atom_features.append(feat)
        x = torch.tensor(atom_features, dtype=torch.float)

        # 生成边索引
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # 添加双向边
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        if not edge_index:  # 如果没有边，返回 None
            return None
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        print(f"Error in smiles_to_graph for SMILES {smiles}: {e}")
        return None


def process_molecular_properties(file_path, output_dir='data/processed/features'):
    """处理分子性质数据"""
    print(f"处理分子性质数据: {file_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"读取了 {len(df)} 条记录")
        
        # 显示数据列
        print("数据列:", df.columns.tolist())
        print("前5行:", df.head())
        
        # 确保有SMILES和性质列
        if 'smiles' not in df.columns:
            print("错误: 未找到 'smiles' 列")
            return
        
        # 筛选有效的SMILES
        valid_mols = []
        features = []
        smiles_list = []
        property_values = []
        
        # 获取所有性质列（除了SMILES和ID列）
        property_cols = [col for col in df.columns if col not in ['smiles', 'id', 'molecule_id', 'name']]
        print(f"性质列: {property_cols}")
        
        for i, row in df.iterrows():
            try:
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                
                if mol is not None:
                    # 提取摩根指纹
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    features.append(np.array(list(fp.ToBitString()), dtype=np.float32))
                    
                    # 收集性质值
                    props = [row[col] if col in row else 0.0 for col in property_cols]
                    property_values.append(props)
                    smiles_list.append(smiles)
                    valid_mols.append(mol)
            except Exception as e:
                print(f"处理行 {i} 时出错: {e}")
        
        print(f"成功处理 {len(valid_mols)} 个有效分子")
        
        # 保存处理后的数据
        if valid_mols:
            features_array = np.stack(features)
            properties_array = np.stack(property_values)
            
            # 保存为NumPy数组
            features_path = os.path.join(output_dir, 'molecular_features.npy')
            properties_path = os.path.join(output_dir, 'molecular_properties.npy')
            np.save(features_path, features_array)
            np.save(properties_path, properties_array)
            
            # 保存SMILES列表
            smiles_path = os.path.join(output_dir, 'molecular_smiles.txt')
            with open(smiles_path, 'w') as f:
                f.write('\n'.join(smiles_list))
            
            # 保存性质列名
            property_names_path = os.path.join(output_dir, 'property_names.txt')
            with open(property_names_path, 'w') as f:
                f.write('\n'.join(property_cols))
            
            print(f"分子特征保存到 {features_path}")
            print(f"分子性质保存到 {properties_path}")
            print(f"分子SMILES保存到 {smiles_path}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")


def process_drug_target_interactions(file_path, output_dir='data/processed/interactions'):
    """处理药物-靶点相互作用数据"""
    print(f"处理药物-靶点相互作用数据: {file_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"读取了 {len(df)} 条记录")
        
        # 显示数据列
        print("数据列:", df.columns.tolist())
        print("前5行:", df.head())
        
        # 检查必要的列
        required_cols = ['smiles', 'target_sequence', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"错误: 缺少这些列: {missing_cols}")
            return
        
        # 将label列映射为数值类型的interaction列
        label_mapping = {
            'inhibitor': 1.0, 
            'binder': 1.0, 
            'ligand': 1.0, 
            'other': 0.5,
            'unknown': 0.5,
            'substrate': 0.5,
            'none': 0.0,
            'n/a': 0.0
        }
        
        # 如果label列有值，将其转换为相互作用强度
        df['interaction'] = df['label'].map(lambda x: label_mapping.get(str(x).lower(), 0.5))
        print("标签映射情况:", df['label'].value_counts())
        print("已将label列映射为interaction列")
        
        # 处理SMILES为None或N/A的情况
        df = df[df['smiles'].notna() & (df['smiles'] != 'N/A') & (df['smiles'] != 'NaN')]
        print(f"过滤后还有 {len(df)} 条记录")
        
        # 如果原始列是'smiles'，将其重命名为'drug_smiles'
        if 'drug_smiles' not in df.columns and 'smiles' in df.columns:
            df['drug_smiles'] = df['smiles']
        
        # 预处理数据
        drug_graphs = []
        target_sequences = []
        interactions = []
        drug_smiles = []
        
        for i, row in df.iterrows():
            try:
                smiles = row['drug_smiles']
                sequence = row['target_sequence']
                interaction = float(row['interaction'])
                
                # 确保序列是字符串
                if not isinstance(sequence, str):
                    print(f"行 {i}: 序列不是字符串，尝试转换: {sequence}")
                    try:
                        sequence = str(sequence)
                    except:
                        print(f"无法将序列转换为字符串，跳过此行")
                        continue
                
                # 清理序列标题行（以>开头的行）
                if '>' in sequence:
                    sequence = sequence.split('\n', 1)[1] if '\n' in sequence else sequence
                    sequence = ''.join(c for c in sequence if not c.isdigit() and c.isalpha())
                
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    graph.y = torch.tensor([interaction], dtype=torch.float)
                    drug_graphs.append(graph)
                    target_sequences.append(sequence)
                    interactions.append(interaction)
                    drug_smiles.append(smiles)
            except Exception as e:
                print(f"处理行 {i} 时出错: {e}")
        
        print(f"成功处理 {len(drug_graphs)} 个药物-靶点相互作用")
        
        # 保存处理后的数据
        if drug_graphs:
            # 保存图数据
            drug_graphs_path = os.path.join(output_dir, 'drug_graphs.pt')
            torch.save(drug_graphs, drug_graphs_path)
            
            # 保存靶点序列 - 确保所有序列都是字符串
            target_sequences_path = os.path.join(output_dir, 'target_sequences.txt')
            with open(target_sequences_path, 'w') as f:
                for seq in target_sequences:
                    if not isinstance(seq, str):
                        seq = str(seq)
                    f.write(seq + '\n')
            
            # 保存相互作用标签
            interactions_path = os.path.join(output_dir, 'interactions.pt')
            torch.save(torch.tensor(interactions, dtype=torch.float), interactions_path)
            
            # 保存SMILES列表
            drug_smiles_path = os.path.join(output_dir, 'drug_smiles.txt')
            with open(drug_smiles_path, 'w') as f:
                f.write('\n'.join(drug_smiles))
            
            print(f"药物图结构保存到 {drug_graphs_path}")
            print(f"靶点序列保存到 {target_sequences_path}")
            print(f"相互作用保存到 {interactions_path}")
            print(f"药物SMILES保存到 {drug_smiles_path}")
            
            # 划分训练、验证和测试集
            indices = list(range(len(drug_graphs)))
            train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)
            
            # 保存索引
            train_idx_path = os.path.join(output_dir, 'train_idx.pt')
            val_idx_path = os.path.join(output_dir, 'val_idx.pt')
            test_idx_path = os.path.join(output_dir, 'test_idx.pt')
            
            torch.save(torch.tensor(train_idx), train_idx_path)
            torch.save(torch.tensor(val_idx), val_idx_path)
            torch.save(torch.tensor(test_idx), test_idx_path)
            
            print(f"训练集大小: {len(train_idx)}")
            print(f"验证集大小: {len(val_idx)}")
            print(f"测试集大小: {len(test_idx)}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 获取配置
    config = load_config()
    
    # 从配置文件中获取路径
    try:
        molecular_properties_path = config['data']['raw_data']['molecular_properties']
        drug_target_path = config['data']['raw_data']['drug_target_interactions']
    except KeyError:
        # 如果配置文件中没有指定路径，使用默认路径
        molecular_properties_path = "data/raw/molecular_properties.csv"
        drug_target_path = "data/raw/drugbank/drug_target_interactions.csv"
    
    print(f"使用分子性质数据文件: {molecular_properties_path}")
    print(f"使用药物-靶点相互作用数据文件: {drug_target_path}")
    
    # 创建输出目录
    output_features_dir = config['data'].get('molecular_features_dir', 'data/processed/features')
    output_interactions_dir = config['data'].get('interactions_dir', 'data/processed/interactions')
    
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_interactions_dir, exist_ok=True)
    
    # 处理数据
    if os.path.exists(molecular_properties_path):
        process_molecular_properties(molecular_properties_path, output_features_dir)
    else:
        print(f"错误: 文件 {molecular_properties_path} 不存在")
    
    if os.path.exists(drug_target_path):
        process_drug_target_interactions(drug_target_path, output_interactions_dir)
    else:
        print(f"错误: 文件 {drug_target_path} 不存在")


if __name__ == "__main__":
    main() 
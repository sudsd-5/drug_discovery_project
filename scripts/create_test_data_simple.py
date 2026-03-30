#!/usr/bin/env python3
"""
create_test_data_simple.py - 创建最小测试数据集用于DTI模型调试

该脚本生成合成数据用于快速测试DTI模型，不依赖RDKit。
简化版本直接生成PyG Data对象。
"""

import torch
import os
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split


def create_simple_drug_graph(num_atoms=10, seed=0):
    """创建一个简单的药物分子图，不依赖RDKit"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 随机生成原子特征 (15维)
    x = torch.randn(num_atoms, 15)
    
    # 随机生成边 (确保是连通图)
    edge_list = []
    # 先生成一条链
    for i in range(num_atoms - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    # 再添加一些随机边
    num_extra_edges = max(1, num_atoms // 3)
    for _ in range(num_extra_edges):
        i = np.random.randint(0, num_atoms)
        j = np.random.randint(0, num_atoms)
        if i != j:
            edge_list.append([i, j])
            edge_list.append([j, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def create_minimal_test_dataset():
    """创建最小测试数据集"""
    print("=" * 60)
    print("创建DTI模型最小测试数据集")
    print("=" * 60)

    # 1. 生成10个药物分子图
    print("\n1. 生成药物分子图...")
    num_drugs = 10
    drug_graphs = []
    for i in range(num_drugs):
        # 每个药物有不同数量的原子 (5-20个)
        num_atoms = np.random.randint(5, 21)
        graph = create_simple_drug_graph(num_atoms, seed=i*100)
        drug_graphs.append(graph)
        print(f"  ✓ Drug {i+1}: {graph.num_nodes} 个原子, {graph.num_edges // 2} 个键")

    print(f"  成功生成 {num_drugs} 个药物分子图")

    # 2. 生成蛋白质embeddings (480维，与ESM-2一致)
    # 每个药物配对3个蛋白质，共30个样本
    num_proteins_per_drug = 3
    num_samples = num_drugs * num_proteins_per_drug

    print(f"\n2. 生成蛋白质embeddings (480维)...")
    print(f"  每个药物配对 {num_proteins_per_drug} 个蛋白质")
    print(f"  总样本数: {num_samples}")

    # 为每个样本生成蛋白质embeddings
    target_embeddings = []
    for i in range(num_drugs):
        for j in range(num_proteins_per_drug):
            # 使用固定随机种子确保可重复性
            torch.manual_seed(i * 100 + j)
            # 生成正态分布的embedding
            embedding = torch.randn(480) * 0.5
            target_embeddings.append(embedding)
    torch.manual_seed(42)  # 重置随机种子

    print(f"  成功生成 {len(target_embeddings)} 个蛋白质embeddings")

    # 3. 生成交互作用标签 (0或1)
    print(f"\n3. 生成交互作用标签...")
    np.random.seed(42)
    # 确保有一定的正负样本平衡 (约50%正样本)
    interactions = torch.tensor(
        np.random.choice([0.0, 1.0], size=num_samples, p=[0.5, 0.5]),
        dtype=torch.float
    )
    num_positive = int(interactions.sum())
    num_negative = num_samples - num_positive
    print(f"  正样本: {num_positive}, 负样本: {num_negative}")

    # 4. 划分数据集 (60%训练, 20%验证, 20%测试)
    print(f"\n4. 划分数据集...")
    indices = np.arange(num_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, random_state=42, stratify=interactions.numpy()
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )

    print(f"  训练集: {len(train_idx)} 样本")
    print(f"  验证集: {len(val_idx)} 样本")
    print(f"  测试集: {len(test_idx)} 样本")

    # 5. 保存数据
    print(f"\n5. 保存数据文件...")
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(script_dir, "data", "processed", "interactions")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(drug_graphs, os.path.join(output_dir, "drug_graphs.pt"))
    print(f"  ✓ drug_graphs.pt ({num_drugs} 个图)")

    torch.save(target_embeddings, os.path.join(output_dir, "target_embeddings.pt"))
    print(f"  ✓ target_embeddings.pt ({len(target_embeddings)} 个embedding)")

    torch.save(interactions, os.path.join(output_dir, "interactions.pt"))
    print(f"  ✓ interactions.pt ({num_samples} 个标签)")

    torch.save(torch.tensor(train_idx), os.path.join(output_dir, "train_idx.pt"))
    torch.save(torch.tensor(val_idx), os.path.join(output_dir, "val_idx.pt"))
    torch.save(torch.tensor(test_idx), os.path.join(output_dir, "test_idx.pt"))
    print(f"  ✓ 划分索引已保存")

    print(f"\n" + "=" * 60)
    print(f"测试数据集已保存到: {output_dir}")
    print("=" * 60)

    # 打印数据维度信息
    print(f"\n数据维度摘要:")
    print(f"  药物图特征维度: {drug_graphs[0].x.shape[1]} (应为15)")
    print(f"  蛋白质embedding维度: {target_embeddings[0].shape[0]} (应为480)")
    print(f"  交互标签形状: {interactions.shape}")
    
    # 列出所有生成的文件
    print(f"\n生成的文件:")
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        print(f"  - {filename}: {size:,} bytes")

    return drug_graphs, target_embeddings, interactions, train_idx, val_idx, test_idx


if __name__ == "__main__":
    create_minimal_test_dataset()

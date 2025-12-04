import os
import torch
import yaml
import argparse
import numpy as np
from dti_model import DTIPredictor, amino_acid_to_idx


def load_config(config_path=None):
    """加载配置文件"""
    default_path = 'E:/AI/drug_discovery_project/configs/dti_config.yaml'
    config_path = config_path or default_path  # 如果 config_path 是 None，则用默认路径
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path='models/best_model.pt', config=None):
    """加载训练好的模型"""
    if config is None:
        config = load_config()
    
    # 创建模型实例
    model = DTIPredictor(config['model'])
    
    # 加载训练好的参数
    model.load_state_dict(torch.load(model_path))
    
    # 设置为评估模式
    model.eval()
    
    return model


def predict_interaction(model, drug_data, protein_sequence, device='cpu'):
    """预测药物与蛋白质之间的相互作用概率"""
    # 确保模型在正确的设备上
    model = model.to(device)
    drug_data = drug_data.to(device)
    
    # 将蛋白质序列转换为张量
    max_seq_len = 1000  # 确保与训练时一致
    protein_tensor = amino_acid_to_idx(protein_sequence, max_seq_len).to(device)
    
    # 进行预测
    with torch.no_grad():
        prediction = model.predict_interaction(drug_data, protein_tensor)
    
    # 返回预测结果
    return prediction.item()


def load_test_data(data_dir='data/processed/interactions'):
    """加载一些测试数据用于演示"""
    # 加载药物图数据
    drug_graphs_path = os.path.join(data_dir, 'drug_graphs.pt')
    drug_graphs = torch.load(drug_graphs_path)
    
    # 加载靶点序列
    target_sequences_path = os.path.join(data_dir, 'target_sequences.txt')
    with open(target_sequences_path, 'r', encoding='utf-8') as f:
        target_sequences = f.read().strip().split('\n')
    
    # 加载测试集索引
    test_idx_path = os.path.join(data_dir, 'test_idx.pt')
    test_idx = torch.load(test_idx_path)
    
    return [drug_graphs[i] for i in test_idx[:10]], [target_sequences[i] for i in test_idx[:10]]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预测药物-靶点相互作用')
    parser.add_argument('--config', type=str, default='configs/dti_config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='模型路径')
    parser.add_argument('--device', type=str, default='cpu', help='使用的设备 (cpu 或 cuda)')
    args = parser.parse_args()
    
    # 加载配置和模型
    config = load_config(args.config)
    model = load_model(args.model, config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 加载一些测试数据
    test_drugs, test_proteins = load_test_data(config['data']['interactions_dir'])
    
    print("\n预测示例:")
    print(f"{'索引':<5}{'预测概率':<10}")
    print("-" * 15)
    
    # 对测试样本进行预测
    for i, (drug, protein) in enumerate(zip(test_drugs, test_proteins)):
        probability = predict_interaction(model, drug, protein, device)
        print(f"{i:<5}{probability:.6f}")
    
    print("\n预测完成!")


if __name__ == "__main__":
    main() 
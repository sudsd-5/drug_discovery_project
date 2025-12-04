# 保存为 E:\AI\drug_discovery_project\src\train_dti.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import yaml
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from dti_model import DTIPredictor
from torch_geometric.loader import DataLoader as GeometricDataLoader
import warnings

# 忽略 torch_geometric 的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric.data.collate')

# 启用 cuDNN 优化
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 自动选择最优算法

class DTIDataset(Dataset):
    """药物-靶点相互作用数据集"""
    def __init__(self, drug_graphs, target_embeddings, interactions, indices=None, device=None):
        if indices is not None:
            self.drug_graphs = [drug_graphs[i] for i in indices]
            self.target_embeddings = [target_embeddings[i].to(device) for i in indices]  # 预加载到 GPU
            self.interactions = interactions[indices].to(device)  # 预加载到 GPU
        else:
            self.drug_graphs = drug_graphs
            self.target_embeddings = [t.to(device) for t in target_embeddings]  # 预加载到 GPU
            self.interactions = interactions.to(device)  # 预加载到 GPU

    def __len__(self):
        return len(self.drug_graphs)

    def __getitem__(self, idx):
        drug = self.drug_graphs[idx]
        protein_embedding = self.target_embeddings[idx]
        interaction = self.interactions[idx]
        return drug, protein_embedding, interaction

def load_config(config_path='E:\\AI\\drug_discovery_project\\configs\\dti_config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        print("Loaded config:", config)  # 在赋值后再打印
        return config

def setup_logging(config):
    """设置日志记录"""
    log_dir = config['logging']['log_dir']
    models_dir = 'E:\\AI\\drug_discovery_project\\output\\models'  # 修改为新的模型保存路径
    tensorboard_dir = config['logging']['tensorboard_dir']
    output_log_dir = 'E:\\AI\\drug_discovery_project\\output'  # 输出日志目录

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)  # 确保新模型目录存在
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(output_log_dir, exist_ok=True)  # 确保输出目录存在

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'dti_training_{timestamp}.log')  # 保留原有日志文件
    output_log_file = os.path.join(output_log_dir, 'training.log')  # 固定输出日志文件

    # 配置日志记录，同时输出到控制台、原有日志文件和新的 training.log
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),           # 保留原有带时间戳的日志文件
            logging.FileHandler(output_log_file),    # 输出到固定 training.log
            logging.StreamHandler()                  # 输出到控制台
        ]
    )

    return SummaryWriter(os.path.join(tensorboard_dir, timestamp))

def load_data(config, device):
    """加载数据并预加载到 GPU"""
    interactions_dir = config['data']['interactions_dir']

    drug_graphs_path = os.path.join(interactions_dir, 'drug_graphs.pt')
    drug_graphs = torch.load(drug_graphs_path, map_location=device)  # 直接加载到 GPU
    print(f"Number of graphs: {len(drug_graphs)}")

    target_embeddings_path = os.path.join(interactions_dir, 'target_embeddings.pt')
    target_embeddings = torch.load(target_embeddings_path, map_location=device)  # 直接加载到 GPU
    print(f"Number of target embeddings: {len(target_embeddings)}")

    interactions_path = os.path.join(interactions_dir, 'interactions.pt')
    interactions = torch.load(interactions_path, map_location=device)  # 直接加载到 GPU
    print(f"Number of interactions: {len(interactions)}")

    train_idx_path = os.path.join(interactions_dir, 'train_idx.pt')
    val_idx_path = os.path.join(interactions_dir, 'val_idx.pt')
    test_idx_path = os.path.join(interactions_dir, 'test_idx.pt')

    train_idx = torch.load(train_idx_path, map_location=device)
    val_idx = torch.load(val_idx_path, map_location=device)
    test_idx = torch.load(test_idx_path, map_location=device)

    return drug_graphs, target_embeddings, interactions, train_idx, val_idx, test_idx

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions_proba_list = []
    labels_list = []
    for batch in loader:
        drug_graphs, target_embeddings, batch_labels = batch
        drug_graphs = drug_graphs.to(device)
        target_embeddings = target_embeddings.to(device)
        batch_labels = batch_labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(drug_graphs, target_embeddings)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions_proba_list.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
        labels_list.extend(batch_labels.detach().cpu().numpy().flatten())

    train_loss = total_loss / len(loader)
    labels_np = np.array(labels_list).astype(int)
    predictions_proba_np = np.array(predictions_proba_list)
    predictions_binary_np = np.round(predictions_proba_np).astype(int)

    train_acc = accuracy_score(labels_np, predictions_binary_np)
    train_auroc = roc_auc_score(labels_np, predictions_proba_np)
    train_auprc = average_precision_score(labels_np, predictions_proba_np)
    train_f1 = f1_score(labels_np, predictions_binary_np) # <-- Calculate F1 score
    return train_loss, train_acc, train_auroc, train_auprc, train_f1 # <-- Return F1 score

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds_proba = []

    with torch.no_grad():
        for batch in val_loader:
            drug_data, protein_data, labels = batch
            drug_data = drug_data.to(device)
            protein_data = protein_data.to(device)
            labels = labels.to(device)

            outputs = model(drug_data, protein_data)
            labels_unsqueeze = labels.unsqueeze(1)
            loss = criterion(outputs, labels_unsqueeze)

            total_loss += loss.item()
            all_labels.append(labels.cpu())
            all_preds_proba.append(torch.sigmoid(outputs).cpu()) # <-- Store probabilities

    val_loss = total_loss / len(val_loader)
    all_labels_tensor = torch.cat(all_labels)
    all_preds_proba_tensor = torch.cat(all_preds_proba)

    all_labels_np = all_labels_tensor.squeeze().numpy().astype(int)
    all_preds_proba_np = all_preds_proba_tensor.squeeze().numpy()
    all_preds_binary_np = (all_preds_proba_np > 0.5).astype(int)

    val_acc = accuracy_score(all_labels_np, all_preds_binary_np)
    val_auroc = roc_auc_score(all_labels_np, all_preds_proba_np)
    val_auprc = average_precision_score(all_labels_np, all_preds_proba_np)
    val_f1 = f1_score(all_labels_np, all_preds_binary_np) # <-- Calculate F1 score

    return val_loss, val_acc, val_auroc, val_auprc, val_f1 # <-- Return F1 score

def plot_training_results(train_metrics, val_metrics, save_dir, results_dir):
    """绘制训练结果图并保存到指定目录"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)  # 确保结果目录存在

    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['accuracy'], label='Training Accuracy')
    plt.plot(val_metrics['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['auroc'], label='Training AUROC')
    plt.plot(val_metrics['auroc'], label='Validation AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'auroc_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['auprc'], label='Training AUPRC')
    plt.plot(val_metrics['auprc'], label='Validation AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('Training and Validation AUPRC')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'auprc_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6)) # <-- Add F1 plot
    plt.plot(train_metrics['f1'], label='Training F1 Score')
    plt.plot(val_metrics['f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'f1_score_curve.png'))
    plt.close()

def main():
    """主函数"""
    config = load_config()
    writer = setup_logging(config)

    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    drug_graphs, target_embeddings, interactions, train_idx, val_idx, test_idx = load_data(config, device)

    train_dataset = DTIDataset(drug_graphs, target_embeddings, interactions, train_idx, device=device)
    val_dataset = DTIDataset(drug_graphs, target_embeddings, interactions, val_idx, device=device)
    test_dataset = DTIDataset(drug_graphs, target_embeddings, interactions, test_idx, device=device)

    batch_size = config['training']['batch_size']
    train_loader = GeometricDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = GeometricDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = DTIPredictor(config['model']).to(device)

    logging.info("torch.compile is not supported on Windows, proceeding without model compilation.")

    criterion = torch.nn.BCEWithLogitsLoss()
    # 使用标准 AdamW，不传入不被所有 torch 版本支持的额外参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr']
    )

    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    early_stopping_patience = config['training']['early_stopping_patience']
    early_stopping_counter = 0

    train_metrics = {'loss': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'f1': []} # <-- Add F1 to metrics dict
    val_metrics = {'loss': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'f1': []}   # <-- Add F1 to metrics dict

    for epoch in range(epochs):
        train_loss, train_acc, train_auroc, train_auprc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device) # <-- Get F1
        val_loss, val_acc, val_auroc, val_auprc, val_f1 = validate(model, val_loader, criterion, device) # <-- Get F1

        scheduler.step(val_loss)

        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_acc)
        train_metrics['auroc'].append(train_auroc)
        train_metrics['auprc'].append(train_auprc)
        train_metrics['f1'].append(train_f1) # <-- Store F1

        val_metrics['loss'].append(val_loss)
        val_metrics['accuracy'].append(val_acc)
        val_metrics['auroc'].append(val_auroc)
        val_metrics['auprc'].append(val_auprc)
        val_metrics['f1'].append(val_f1) # <-- Store F1

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUROC/train', train_auroc, epoch)
        writer.add_scalar('AUROC/val', val_auroc, epoch)
        writer.add_scalar('AUPRC/train', train_auprc, epoch)
        writer.add_scalar('AUPRC/val', val_auprc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch) # <-- Log F1 to TensorBoard
        writer.add_scalar('F1/val', val_f1, epoch)   # <-- Log F1 to TensorBoard

        logging.info(f'Epoch {epoch + 1}/{epochs}:')
        logging.info(
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}, Train F1: {train_f1:.4f}') # <-- Log F1
        logging.info(
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}, Val F1: {val_f1:.4f}') # <-- Log F1
        logging.info('------')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('E:\\AI\\drug_discovery_project\\output\\models', 'best_model.pt'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            logging.info(f'Early stopping at epoch {epoch + 1}')
            break

    results_dir = 'E:\\AI\\drug_discovery_project\\output\\results'
    plot_training_results(train_metrics, val_metrics, 'E:\\AI\\drug_discovery_project\\output\\models', results_dir)

    model.load_state_dict(torch.load(os.path.join('E:\\AI\\drug_discovery_project\\output\\models', 'best_model.pt')))
    test_loss, test_acc, test_auroc, test_auprc, test_f1 = validate(model, test_loader, criterion, device) # <-- Get F1 for test set

    logging.info(f'Test results:')
    logging.info(f'Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}') # <-- Log F1 for test set

    writer.close()

if __name__ == "__main__":
    main()
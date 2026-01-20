# E:\AI\drug_discovery_project\src\dti_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool  # GATConv不再直接被这个版本使用，但保留导入以备不时之需
from torch_geometric.data import Data, Batch


# ==============================================================================
# 1. 新的 TwoRealTwoImaginaryGCNLayer (双实双虚GCN层)
# ==============================================================================
class TwoRealTwoImaginaryGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoRealTwoImaginaryGCNLayer, self).__init__()
        # 为两条实部路径分别创建GCNConv层
        self.gcn_conv_real1 = GCNConv(in_channels, out_channels)
        self.gcn_conv_real2 = GCNConv(in_channels, out_channels)

        # 为两条虚部路径分别创建GCNConv层
        self.gcn_conv_imag1 = GCNConv(in_channels, out_channels)
        self.gcn_conv_imag2 = GCNConv(in_channels, out_channels)

        # 实现权重共享（对称性）:
        # imag1 与 real1 对称
        self.gcn_conv_imag1.lin.weight = self.gcn_conv_real1.lin.weight
        if self.gcn_conv_real1.lin.bias is not None:
            self.gcn_conv_imag1.lin.bias = self.gcn_conv_real1.lin.bias

        # imag2 与 real2 对称
        self.gcn_conv_imag2.lin.weight = self.gcn_conv_real2.lin.weight
        if self.gcn_conv_real2.lin.bias is not None:
            self.gcn_conv_imag2.lin.bias = self.gcn_conv_real2.lin.bias

    def forward(self, x_r1, x_r2, x_i1, x_i2, edge_index):
        h_r1 = self.gcn_conv_real1(x_r1, edge_index)
        h_r2 = self.gcn_conv_real2(x_r2, edge_index)
        h_i1 = self.gcn_conv_imag1(x_i1, edge_index)
        h_i2 = self.gcn_conv_imag2(x_i2, edge_index)

        return h_r1, h_r2, h_i1, h_i2


# ==============================================================================
# 2. 修改后的 DrugEncoder (使用 TwoRealTwoImaginaryGCNLayer)
# ==============================================================================
class DrugEncoder2Real2Imag(nn.Module):
    """药物编码器，GCN底层采用两实两虚对称结构"""

    def __init__(self, input_atom_dim=15,
                 component_hidden_dim=64,
                 component_output_dim=128,
                 num_layers=3, dropout=0.2):
        super(DrugEncoder2Real2Imag, self).__init__()
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(p=dropout)

        # 初始投影层：将原子特征映射到初始的4个分量
        self.initial_proj_real1 = nn.Linear(input_atom_dim, component_hidden_dim)
        self.initial_proj_real2 = nn.Linear(input_atom_dim, component_hidden_dim)
        self.initial_proj_imag1 = nn.Linear(input_atom_dim, component_hidden_dim)
        self.initial_proj_imag2 = nn.Linear(input_atom_dim, component_hidden_dim)

        # 实现初始投影的对称性
        self.initial_proj_imag1.weight = self.initial_proj_real1.weight
        if self.initial_proj_real1.bias is not None:
            self.initial_proj_imag1.bias = self.initial_proj_real1.bias
        self.initial_proj_imag2.weight = self.initial_proj_real2.weight
        if self.initial_proj_real2.bias is not None:
            self.initial_proj_imag2.bias = self.initial_proj_real2.bias

        self.comp_gcn_layers = nn.ModuleList()
        self.batch_norms_r1 = nn.ModuleList()
        self.batch_norms_r2 = nn.ModuleList()
        self.batch_norms_i1 = nn.ModuleList()
        self.batch_norms_i2 = nn.ModuleList()

        current_dim = component_hidden_dim
        for i in range(num_layers):
            out_dim = component_output_dim if i == num_layers - 1 else component_hidden_dim
            self.comp_gcn_layers.append(TwoRealTwoImaginaryGCNLayer(current_dim, out_dim))
            self.batch_norms_r1.append(nn.BatchNorm1d(out_dim))
            self.batch_norms_r2.append(nn.BatchNorm1d(out_dim))
            self.batch_norms_i1.append(nn.BatchNorm1d(out_dim))
            self.batch_norms_i2.append(nn.BatchNorm1d(out_dim))
            # BN层的对称性也可以考虑，但通常独立学习
            current_dim = out_dim

    def forward(self, x_atom_features, edge_index, batch):
        # 1. 初始投影
        x_r1 = self.initial_proj_real1(x_atom_features)
        x_r2 = self.initial_proj_real2(x_atom_features)
        x_i1 = self.initial_proj_imag1(x_atom_features)
        x_i2 = self.initial_proj_imag2(x_atom_features)

        # 2. 通过双实双虚GCN层
        for i in range(self.num_layers):
            x_r1, x_r2, x_i1, x_i2 = self.comp_gcn_layers[i](x_r1, x_r2, x_i1, x_i2, edge_index)

            x_r1 = self.batch_norms_r1[i](x_r1)
            x_r2 = self.batch_norms_r2[i](x_r2)
            x_i1 = self.batch_norms_i1[i](x_i1)
            x_i2 = self.batch_norms_i2[i](x_i2)

            x_r1 = F.relu(x_r1)
            x_r2 = F.relu(x_r2)
            # 虚部激活函数选择
            x_i1 = F.relu(x_i1)
            x_i2 = F.relu(x_i2)

            x_r1 = self.dropout_layer(x_r1)
            x_r2 = self.dropout_layer(x_r2)
            x_i1 = self.dropout_layer(x_i1)
            x_i2 = self.dropout_layer(x_i2)

        # 3. 全局池化
        g_r1 = global_mean_pool(x_r1, batch)
        g_r2 = global_mean_pool(x_r2, batch)
        g_i1 = global_mean_pool(x_i1, batch)
        g_i2 = global_mean_pool(x_i2, batch)

        return g_r1, g_r2, g_i1, g_i2


# ==============================================================================
# 3. 原有的 ProteinEncoder (保持不变)
# ==============================================================================
class ProteinEncoder(nn.Module):
    """蛋白质编码器，直接使用预计算的 ESM-2 嵌入"""

    def __init__(self, input_dim=480, output_channels=128, dropout=0.2):
        super(ProteinEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        return x


# ==============================================================================
# 4. 修改后的 DTIPredictor
# ==============================================================================
class DTIPredictor(nn.Module):
    """药物-靶点相互作用预测模型，药物编码器GCN底层采用两实两虚对称，蛋白质编码器对应调整"""

    def __init__(self, config):
        super(DTIPredictor, self).__init__()

        drug_input_dim = config.get('drug_input_dim', 15)
        drug_comp_hidden_dim = config.get('drug_hidden_channels', 64)
        drug_comp_output_dim = config.get('drug_output_channels_component', 64)  # 每个分量的输出维度
        # 假设总共期望药物输出 4 * 64 = 256维. 配置文件中此项需要对应单个分量的维度

        drug_num_layers = config.get('drug_num_layers', 3)
        drug_dropout = config.get('drug_dropout', 0.2)

        protein_input_dim = config.get('protein_input_dim', 480)
        protein_comp_output_dim = config.get('protein_output_channels_component', 64)  # 每个分量的输出维度
        protein_dropout = config.get('protein_dropout', 0.2)

        self.drug_encoder = DrugEncoder2Real2Imag(
            input_atom_dim=drug_input_dim,
            component_hidden_dim=drug_comp_hidden_dim,
            component_output_dim=drug_comp_output_dim,
            num_layers=drug_num_layers,
            dropout=drug_dropout
        )

        # 蛋白质编码器：需要4个实例，两实两虚对称
        self.protein_encoder_real1 = ProteinEncoder(protein_input_dim, protein_comp_output_dim, protein_dropout)
        self.protein_encoder_real2 = ProteinEncoder(protein_input_dim, protein_comp_output_dim, protein_dropout)
        self.protein_encoder_imag1 = ProteinEncoder(protein_input_dim, protein_comp_output_dim, protein_dropout)
        self.protein_encoder_imag2 = ProteinEncoder(protein_input_dim, protein_comp_output_dim, protein_dropout)
        self._share_protein_encoder_weights()

        # 拼接后的维度
        # 药物贡献: 4 * drug_comp_output_dim
        # 蛋白质贡献: 4 * protein_comp_output_dim
        combined_dim = (drug_comp_output_dim + protein_comp_output_dim) * 4  # 注意乘以4

        predictor_hidden_dim1 = config.get('predictor_hidden_dim1', combined_dim // 2)
        predictor_hidden_dim2 = config.get('predictor_hidden_dim2', combined_dim // 4)
        predictor_dropout = config.get('predictor_dropout', 0.2)

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, predictor_hidden_dim1),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_dim1, predictor_hidden_dim2),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_dim2, 1)
        )

    def _share_protein_encoder_weights(self):
        self.protein_encoder_imag1.load_state_dict(self.protein_encoder_real1.state_dict())
        self.protein_encoder_imag2.load_state_dict(self.protein_encoder_real2.state_dict())

    def forward(self, drug_data, protein_data_input):
        if isinstance(drug_data, Data) and not isinstance(drug_data, Batch):
            drug_x = drug_data.x
            drug_edge_index = drug_data.edge_index
            drug_batch = torch.zeros(drug_data.x.size(0), dtype=torch.long, device=drug_data.x.device)
        elif isinstance(drug_data, Batch):
            drug_x = drug_data.x
            drug_edge_index = drug_data.edge_index
            drug_batch = drug_data.batch
        else:
            raise TypeError(f"drug_data must be Data or Batch object, got {type(drug_data)}")

        dr1, dr2, di1, di2 = self.drug_encoder(drug_x, drug_edge_index, drug_batch)

        pr1 = self.protein_encoder_real1(protein_data_input)
        pr2 = self.protein_encoder_real2(protein_data_input)
        pi1 = self.protein_encoder_imag1(protein_data_input)
        pi2 = self.protein_encoder_imag2(protein_data_input)

        combined = torch.cat([
            dr1, dr2, di1, di2,
            pr1, pr2, pi1, pi2
        ], dim=1)

        interaction = self.predictor(combined)
        return interaction

    def predict_interaction(self, drug_data, protein_data):
        self.eval()
        with torch.no_grad():
            return self.forward(drug_data, protein_data)

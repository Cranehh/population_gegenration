"""
分层人口VAE - 针对家庭-成员分层结构的变分自编码器
解决变长家庭数据的固定长度编码问题，为贝叶斯网络约束提供潜在表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import pickle
from collections import defaultdict


class PopulationDataLoader:
    """人口数据加载器 - 整合dataset.py和生成数据的加载逻辑"""
    
    def __init__(self, data_dir: str = "数据", generated_data_dir: str = "生成数据"):
        self.data_dir = data_dir
        self.generated_data_dir = generated_data_dir
        self.scaler_family = StandardScaler()
        self.scaler_member = StandardScaler()
        
    def load_numpy_data(self):
        """加载NPY格式的处理后数据"""
        try:
            family_data = np.load(os.path.join(self.data_dir, 'family_sample_improved_cluster.npy'))
            member_data = np.load(os.path.join(self.data_dir, 'family_member_sample_improved_cluster.npy'))
            adj_data = np.load(os.path.join(self.data_dir, 'family_adj.npy'))
            edge_data = np.load(os.path.join(self.data_dir, 'familymember_relationship.npy'))
            node_data = np.load(os.path.join(self.data_dir, 'familymember_type.npy'))
            
            print(f"已加载NPY数据:")
            print(f"  家庭数据: {family_data.shape}")
            print(f"  成员数据: {member_data.shape}")
            print(f"  邻接矩阵: {adj_data.shape}")
            print(f"  边特征: {edge_data.shape}")
            print(f"  节点特征: {node_data.shape}")
            
            return {
                'family': family_data,
                'member': member_data,
                'adj': adj_data,
                'edge': edge_data,
                'node': node_data
            }
        except FileNotFoundError as e:
            print(f"NPY数据文件未找到: {e}")
            return None
    
    def load_csv_data(self, max_samples_per_grid: int = 1000):
        """加载CSV格式的生成数据"""
        family_data_list = []
        member_data_list = []
        
        # 遍历生成数据文件夹
        if not os.path.exists(self.generated_data_dir):
            print(f"生成数据文件夹不存在: {self.generated_data_dir}")
            return None
            
        csv_files = [f for f in os.listdir(self.generated_data_dir) if f.endswith('.csv')]
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 按栅格分组
        grid_files = defaultdict(lambda: {'family': None, 'member': None})
        
        for file in csv_files:
            if '家庭数据' in file:
                grid_id = self._extract_grid_id(file)
                grid_files[grid_id]['family'] = file
            elif '个人数据' in file:
                grid_id = self._extract_grid_id(file)
                grid_files[grid_id]['member'] = file
        
        print(f"识别出 {len(grid_files)} 个栅格的数据")
        
        # 加载每个栅格的数据
        for grid_id, files in list(grid_files.items())[:10]:  # 限制加载数量用于测试
            if files['family'] and files['member']:
                try:
                    # 加载家庭数据
                    family_df = pd.read_csv(os.path.join(self.generated_data_dir, files['family']))
                    member_df = pd.read_csv(os.path.join(self.generated_data_dir, files['member']))
                    
                    # 限制样本数量
                    if len(family_df) > max_samples_per_grid:
                        family_df = family_df.sample(n=max_samples_per_grid, random_state=42)
                    if len(member_df) > max_samples_per_grid * 8:  # 假设平均每家庭8人
                        member_df = member_df.sample(n=max_samples_per_grid * 8, random_state=42)
                    
                    family_data_list.append(family_df)
                    member_data_list.append(member_df)
                    
                except Exception as e:
                    print(f"加载栅格 {grid_id} 数据失败: {e}")
                    continue
        
        if family_data_list and member_data_list:
            # 合并所有数据
            all_family_data = pd.concat(family_data_list, ignore_index=True)
            all_member_data = pd.concat(member_data_list, ignore_index=True)
            
            print(f"CSV数据加载完成:")
            print(f"  家庭数据: {all_family_data.shape}")
            print(f"  个人数据: {all_member_data.shape}")
            
            return {
                'family_df': all_family_data,
                'member_df': all_member_data
            }
        else:
            print("未能成功加载CSV数据")
            return None
    
    def _extract_grid_id(self, filename):
        """从文件名提取栅格ID"""
        try:
            # 例如: "人口栅格_东城区_栅格0_家庭数据.csv"
            parts = filename.split('_')
            for part in parts:
                if '栅格' in part and part != '人口栅格':
                    return part.replace('栅格', '').split('_')[0]
            return filename.split('_')[2] if len(filename.split('_')) > 2 else "unknown"
        except:
            return "unknown"
    
    def create_hierarchical_dataset(self, use_numpy: bool = True):
        """创建分层数据集用于VAE训练"""
        
        if use_numpy:
            data = self.load_numpy_data()
            if data is None:
                print("尝试加载CSV数据...")
                use_numpy = False
        
        if not use_numpy:
            data = self.load_csv_data()
            if data is None:
                raise ValueError("无法加载任何数据文件")
                
        if use_numpy:
            return self._process_numpy_data(data)
        else:
            return self._process_csv_data(data)
    
    def _process_numpy_data(self, data):
        """处理NPY格式数据"""
        family_data = data['family']
        member_data = data['member']
        adj_data = data['adj']
        edge_data = data['edge']
        node_data = data['node']
        
        # 基本信息
        num_samples = family_data.shape[0]
        max_family_size = member_data.shape[1]
        family_feature_dim = family_data.shape[1] if len(family_data.shape) > 1 else 10
        member_feature_dim = member_data.shape[2] if len(member_data.shape) > 2 else 51
        
        print(f"数据维度信息:")
        print(f"  样本数量: {num_samples}")
        print(f"  最大家庭规模: {max_family_size}")
        print(f"  家庭特征维度: {family_feature_dim}")
        print(f"  成员特征维度: {member_feature_dim}")
        
        # 创建数据集
        dataset = HierarchicalPopulationDataset(
            family_data=family_data,
            member_data=member_data,
            adj_data=adj_data,
            edge_data=edge_data,
            node_data=node_data
        )
        
        return dataset
    
    def _process_csv_data(self, data):
        """处理CSV格式数据"""
        family_df = data['family_df']
        member_df = data['member_df']
        
        # 提取特征列
        family_features = family_df.iloc[:, :10].values  # 前10列为家庭特征
        member_features = member_df.iloc[:, :51].values  # 前51列为成员特征
        
        print(f"CSV数据转换:")
        print(f"  家庭特征形状: {family_features.shape}")
        print(f"  成员特征形状: {member_features.shape}")
        
        # 重组成员数据为家庭-成员结构
        max_family_size = 8  # 假设最大家庭规模
        
        # 简化处理：将成员数据重新组织为家庭结构
        # 这里需要根据实际数据结构调整
        num_families = min(len(family_features), len(member_features) // max_family_size)
        
        # 重塑成员数据
        member_data_reshaped = member_features[:num_families * max_family_size].reshape(
            num_families, max_family_size, -1
        )
        
        # 创建简化的邻接矩阵（可以后续改进）
        adj_data = np.zeros((num_families, max_family_size, max_family_size))
        edge_data = np.zeros((num_families, max_family_size, max_family_size, 5))  # 假设5种关系类型
        node_data = np.zeros((num_families, max_family_size, 6))  # 假设6种节点类型
        
        dataset = HierarchicalPopulationDataset(
            family_data=family_features[:num_families],
            member_data=member_data_reshaped,
            adj_data=adj_data,
            edge_data=edge_data,
            node_data=node_data
        )
        
        return dataset


class HierarchicalPopulationDataset(Dataset):
    """分层人口数据集"""
    
    def __init__(self, family_data, member_data, adj_data, edge_data, node_data):
        self.family_data = torch.FloatTensor(family_data)
        self.member_data = torch.FloatTensor(member_data)
        self.adj_data = torch.FloatTensor(adj_data)
        self.edge_data = torch.FloatTensor(edge_data)
        self.node_data = torch.FloatTensor(node_data)
        
        # 计算有效成员掩码
        self.member_mask = self._compute_member_mask()
        
        # 数据统计信息
        self.num_samples = len(self.family_data)
        self.max_family_size = self.member_data.shape[1]
        self.family_feature_dim = self.family_data.shape[1]
        self.member_feature_dim = self.member_data.shape[2]
        
    def _compute_member_mask(self):
        """计算有效成员的掩码"""
        # 假设全零向量表示无效成员
        member_sum = torch.sum(self.member_data, dim=-1)  # [N, max_family_size]
        mask = (member_sum != 0).float()
        return mask
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'family': self.family_data[idx],         # [family_feature_dim]
            'member': self.member_data[idx],         # [max_family_size, member_feature_dim]
            'adj': self.adj_data[idx],               # [max_family_size, max_family_size]
            'edge': self.edge_data[idx],             # [max_family_size, max_family_size, edge_types]
            'node': self.node_data[idx],             # [max_family_size, node_types]
            'mask': self.member_mask[idx]            # [max_family_size]
        }
    
    def get_flattened_data(self, idx):
        """获取展平后的数据用于VAE"""
        sample = self.__getitem__(idx)
        
        # 展平策略：结构感知的展平
        flattened = self._structure_aware_flatten(sample)
        
        return flattened
    
    def _structure_aware_flatten(self, sample):
        """结构感知的数据展平"""
        family_features = sample['family']  # [family_dim]
        member_features = sample['member']  # [max_family_size, member_dim]
        adj_matrix = sample['adj']          # [max_family_size, max_family_size]
        mask = sample['mask']               # [max_family_size]
        
        # 1. 家庭特征广播到每个成员位置
        family_broadcast = family_features.unsqueeze(0).expand(self.max_family_size, -1)
        
        # 2. 位置编码
        position_encoding = self._create_position_encoding()
        
        # 3. 关系邻接信息编码
        adj_encoding = adj_matrix.sum(dim=-1, keepdim=True)  # 每个成员的连接度
        
        # 4. 掩码信息
        mask_encoding = mask.unsqueeze(-1)
        
        # 5. 融合所有信息
        contextualized_members = torch.cat([
            member_features,        # 原始成员特征
            family_broadcast,       # 家庭上下文
            position_encoding,      # 位置信息
            adj_encoding,          # 关系连接度
            mask_encoding          # 有效性掩码
        ], dim=-1)
        
        # 6. 展平为一维向量
        flattened = contextualized_members.flatten()
        
        return flattened
    
    def _create_position_encoding(self):
        """创建位置编码"""
        # 简单的位置编码
        positions = torch.arange(self.max_family_size, dtype=torch.float)
        pos_encoding = torch.zeros(self.max_family_size, 8)  # 8维位置编码
        
        for i in range(4):
            pos_encoding[:, 2*i] = torch.sin(positions / (10000 ** (2*i / 8)))
            pos_encoding[:, 2*i+1] = torch.cos(positions / (10000 ** (2*i / 8)))
            
        return pos_encoding


class FamilyEncoder(nn.Module):
    """家庭级别编码器"""
    
    def __init__(self, family_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(family_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
    def forward(self, family_features):
        hidden = self.encoder(family_features)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar


class MemberEncoder(nn.Module):
    """成员级别编码器 - 处理变长序列"""
    
    def __init__(self, member_dim: int, hidden_dim: int, latent_dim: int, max_family_size: int):
        super().__init__()
        
        self.member_dim = member_dim
        self.max_family_size = max_family_size
        
        # 个体成员编码器
        self.member_encoder = nn.Sequential(
            nn.Linear(member_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 序列级别编码器（LSTM处理变长序列）
        self.sequence_encoder = nn.LSTM(
            hidden_dim // 2, 
            hidden_dim // 4, 
            batch_first=True,
            bidirectional=True
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
    def forward(self, member_features, member_mask):
        batch_size = member_features.shape[0]
        
        # 编码每个成员
        member_encoded = self.member_encoder(member_features)  # [B, max_family_size, hidden//2]
        
        # 计算实际序列长度
        seq_lengths = member_mask.sum(dim=1).long()  # [B]
        
        # 打包序列用于LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            member_encoded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM编码
        lstm_out, (hidden, cell) = self.sequence_encoder(packed)
        
        # 使用最终隐状态作为序列表示
        # hidden: [2, B, hidden//4] -> [B, hidden//2]
        sequence_repr = torch.cat([hidden[0], hidden[1]], dim=1)
        
        mu = self.mu_layer(sequence_repr)
        logvar = self.logvar_layer(sequence_repr)
        
        return mu, logvar


class GraphEncoder(nn.Module):
    """图结构编码器 - 编码家庭关系图"""
    
    def __init__(self, graph_feature_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        # 简化的图编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.graph_aggregator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 4, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 4, latent_dim)
        
    def forward(self, adj_matrix, edge_features, member_mask):
        batch_size = adj_matrix.shape[0]
        
        # 提取有效边
        edge_mask = (adj_matrix > 0).float()
        
        # 简化处理：使用邻接矩阵作为图特征
        graph_features = adj_matrix.view(batch_size, -1)
        
        # 编码图特征
        encoded = self.edge_encoder(graph_features)
        aggregated = self.graph_aggregator(encoded)
        
        mu = self.mu_layer(aggregated)
        logvar = self.logvar_layer(aggregated)
        
        return mu, logvar


class HierarchicalPopulationVAE(nn.Module):
    """分层人口VAE主模型"""
    
    def __init__(self, 
                 family_feature_dim: int = 10,
                 member_feature_dim: int = 51,
                 max_family_size: int = 8,
                 hidden_dim: int = 256,
                 family_latent_dim: int = 32,
                 member_latent_dim: int = 64,
                 graph_latent_dim: int = 16):
        super().__init__()
        
        self.family_feature_dim = family_feature_dim
        self.member_feature_dim = member_feature_dim
        self.max_family_size = max_family_size
        self.total_latent_dim = family_latent_dim + member_latent_dim + graph_latent_dim
        
        # 分层编码器
        self.family_encoder = FamilyEncoder(family_feature_dim, hidden_dim, family_latent_dim)
        
        # 成员编码器输入维度包括上下文信息
        contextualized_member_dim = member_feature_dim + family_feature_dim + 8 + 1 + 1  # 成员+家庭+位置+邻接+掩码
        self.member_encoder = MemberEncoder(contextualized_member_dim, hidden_dim, member_latent_dim, max_family_size)
        
        # 图编码器
        graph_feature_dim = max_family_size * max_family_size  # 简化的邻接矩阵特征
        self.graph_encoder = GraphEncoder(graph_feature_dim, hidden_dim, graph_latent_dim)
        
        # 解码器
        self.decoder = HierarchicalDecoder(
            self.total_latent_dim, 
            hidden_dim,
            family_feature_dim,
            member_feature_dim,
            max_family_size
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def encode(self, batch):
        """编码阶段"""
        family_features = batch['family']
        member_features = batch['member']
        adj_matrix = batch['adj']
        edge_features = batch['edge']
        member_mask = batch['mask']
        
        # 创建上下文化的成员特征
        batch_size = family_features.shape[0]
        
        # 家庭特征广播
        family_broadcast = family_features.unsqueeze(1).expand(-1, self.max_family_size, -1)
        
        # 位置编码
        pos_encoding = self._create_position_encoding(batch_size, family_features.device)
        
        # 邻接度编码
        adj_encoding = adj_matrix.sum(dim=-1, keepdim=True)
        
        # 掩码编码
        mask_encoding = member_mask.unsqueeze(-1)
        
        # 融合成员特征
        contextualized_members = torch.cat([
            member_features,
            family_broadcast,
            pos_encoding,
            adj_encoding,
            mask_encoding
        ], dim=-1)
        
        # 分层编码
        family_mu, family_logvar = self.family_encoder(family_features)
        member_mu, member_logvar = self.member_encoder(contextualized_members, member_mask)
        graph_mu, graph_logvar = self.graph_encoder(adj_matrix, edge_features, member_mask)
        
        # 合并潜在变量
        mu = torch.cat([family_mu, member_mu, graph_mu], dim=1)
        logvar = torch.cat([family_logvar, member_logvar, graph_logvar], dim=1)
        
        return mu, logvar
    
    def _create_position_encoding(self, batch_size, device):
        """创建批量的位置编码"""
        positions = torch.arange(self.max_family_size, dtype=torch.float, device=device)
        pos_encoding = torch.zeros(self.max_family_size, 8, device=device)
        
        for i in range(4):
            pos_encoding[:, 2*i] = torch.sin(positions / (10000 ** (2*i / 8)))
            pos_encoding[:, 2*i+1] = torch.cos(positions / (10000 ** (2*i / 8)))
        
        return pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码阶段"""
        return self.decoder(z)
    
    def forward(self, batch):
        """完整的前向传播"""
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decode(z)
        
        return recon_batch, mu, logvar
    
    def generate(self, num_samples, device):
        """生成新样本"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.total_latent_dim, device=device)
            generated = self.decode(z)
        return generated


class HierarchicalDecoder(nn.Module):
    """分层解码器"""
    
    def __init__(self, latent_dim, hidden_dim, family_feature_dim, member_feature_dim, max_family_size):
        super().__init__()
        
        self.max_family_size = max_family_size
        self.member_feature_dim = member_feature_dim
        
        # 家庭特征解码器
        self.family_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, family_feature_dim)
        )
        
        # 成员特征解码器
        self.member_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_family_size * member_feature_dim)
        )
        
        # 邻接矩阵解码器
        self.adj_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_family_size * max_family_size),
            nn.Sigmoid()  # 确保输出在[0,1]范围
        )
        
        # 成员存在性解码器
        self.mask_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_family_size),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        """解码潜在变量为分层结构数据"""
        batch_size = z.shape[0]
        
        # 解码各个组件
        family_features = self.family_decoder(z)
        member_features = self.member_decoder(z).view(batch_size, self.max_family_size, self.member_feature_dim)
        adj_matrix = self.adj_decoder(z).view(batch_size, self.max_family_size, self.max_family_size)
        member_mask = self.mask_decoder(z)
        
        return {
            'family': family_features,
            'member': member_features,
            'adj': adj_matrix,
            'mask': member_mask
        }


def vae_loss_function(recon_batch, batch, mu, logvar, beta=1.0):
    """VAE损失函数"""
    
    # 重建损失 - 分层计算
    family_recon_loss = F.mse_loss(recon_batch['family'], batch['family'], reduction='sum')
    
    # 成员特征重建损失（考虑掩码）
    member_mask = batch['mask'].unsqueeze(-1)  # [B, max_family_size, 1]
    member_recon_loss = F.mse_loss(
        recon_batch['member'] * member_mask,
        batch['member'] * member_mask,
        reduction='sum'
    )
    
    # 邻接矩阵重建损失
    adj_recon_loss = F.binary_cross_entropy(
        recon_batch['adj'],
        batch['adj'],
        reduction='sum'
    )
    
    # 掩码重建损失
    mask_recon_loss = F.binary_cross_entropy(
        recon_batch['mask'],
        batch['mask'],
        reduction='sum'
    )
    
    # 总重建损失
    reconstruction_loss = (
        family_recon_loss + 
        member_recon_loss + 
        0.5 * adj_recon_loss +
        0.3 * mask_recon_loss
    )
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = reconstruction_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kl_loss': kl_loss,
        'family_recon_loss': family_recon_loss,
        'member_recon_loss': member_recon_loss,
        'adj_recon_loss': adj_recon_loss,
        'mask_recon_loss': mask_recon_loss
    }


def train_hierarchical_vae(data_dir="数据", generated_data_dir="生成数据",
                          batch_size=32, num_epochs=100, lr=1e-3, beta=1.0):
    """训练分层人口VAE"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载
    data_loader = PopulationDataLoader(data_dir, generated_data_dir)
    dataset = data_loader.create_hierarchical_dataset(use_numpy=True)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"数据集信息:")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数量: {len(dataloader)}")
    print(f"  最大家庭规模: {dataset.max_family_size}")
    print(f"  家庭特征维度: {dataset.family_feature_dim}")
    print(f"  成员特征维度: {dataset.member_feature_dim}")
    
    # 创建模型
    model = HierarchicalPopulationVAE(
        family_feature_dim=dataset.family_feature_dim,
        member_feature_dim=dataset.member_feature_dim,
        max_family_size=dataset.max_family_size
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"\\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"总潜在维度: {model.total_latent_dim}")
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            
            # 计算损失
            loss_dict = vae_loss_function(recon_batch, batch, mu, logvar, beta)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.6f}')
        
        # 平均损失
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        print(f'Epoch {epoch} 完成 - 总损失: {avg_loss:.6f}, '
              f'重建损失: {avg_recon_loss:.6f}, KL损失: {avg_kl_loss:.6f}')
        
        # 保存检查点
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'hierarchical_vae_checkpoint_epoch_{epoch}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'hierarchical_population_vae.pth')
    print("\\n✅ VAE训练完成！")
    
    return model


# 测试和使用示例
if __name__ == "__main__":
    # 训练VAE
    print("开始训练分层人口VAE...")
    model = train_hierarchical_vae(
        batch_size=16,
        num_epochs=50,
        lr=1e-3,
        beta=1.0
    )
    
    print("\\n🎯 VAE训练完成，可以用于：")
    print("1. 生成固定长度的人口潜在表示")
    print("2. 为贝叶斯网络提供约束建模的基础")
    print("3. 在扩散模型中进行classifier guidance")
    print("4. 解决变长数据的表示学习问题")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import random

class FamilyGraphGenerator(nn.Module):
    """基于家庭特征生成家庭成员图的邻接矩阵和特征矩阵"""
    
    def __init__(self, max_family_size=8, node_feature_dim=16):
        super().__init__()
        self.max_family_size = max_family_size
        self.node_feature_dim = node_feature_dim
        
        # 关系类型编码
        self.relation_types = {
            'household_head': 0,    # 户主
            'spouse': 1,           # 配偶
            'child': 2,            # 子女
            'parent': 3,           # 父母
            'sibling': 4,          # 兄弟姐妹
            'grandparent': 5,      # 祖父母
            'grandchild': 6,       # 孙辈
            'other_relative': 7,   # 其他亲属
            'non_relative': 8      # 非亲属
        }
        
        # 边类型（关系类型）
        self.edge_types = {
            'spouse': 0,           # 夫妻关系
            'parent_child': 1,     # 父子/母子关系
            'sibling': 2,          # 兄弟姐妹关系
            'grandparent': 3,      # 祖孙关系
            'other': 4             # 其他关系
        }
        
        # 年龄分组
        self.age_groups = {
            'infant': 0,      # 0-3岁
            'child': 1,       # 4-12岁
            'teen': 2,        # 13-17岁
            'young_adult': 3, # 18-35岁
            'middle_aged': 4, # 36-60岁
            'elderly': 5      # 61岁以上
        }
        
        # 图生成网络
        self.family_encoder = nn.Sequential(
            nn.Linear(19, 128),  # family_final_out的维度
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 节点特征生成器
        self.node_generator = nn.Sequential(
            nn.Linear(128 + 4, 64),  # 家庭编码 + 节点索引 + 关系类型 + 年龄组
            nn.ReLU(),
            nn.Linear(64, node_feature_dim)
        )
        
        # 邻接矩阵生成器
        self.adjacency_generator = nn.Sequential(
            nn.Linear(128 + 2, 32),  # 家庭编码 + 两个节点的索引
            nn.ReLU(),
            nn.Linear(32, len(self.edge_types))
        )
    
    def forward(self, family_final_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成家庭成员图的邻接矩阵和特征矩阵
        
        Args:
            family_final_out: [batch_size, 19] 第一个DiT的家庭特征输出
            
        Returns:
            adjacency_matrices: [batch_size, max_family_size, max_family_size, edge_types] 邻接矩阵
            node_features: [batch_size, max_family_size, node_feature_dim] 节点特征矩阵
            family_masks: [batch_size, max_family_size] 有效成员掩码
        """
        batch_size = family_final_out.shape[0]
        device = family_final_out.device
        
        # 解析家庭特征
        family_info = self._parse_family_features(family_final_out)
        
        # 编码家庭特征
        family_encoding = self.family_encoder(family_final_out)  # [batch_size, 128]
        
        # 生成图结构
        adjacency_matrices, node_features, family_masks = self._generate_family_graphs(
            family_encoding, family_info, device
        )
        
        return adjacency_matrices, node_features, family_masks
    
    def _parse_family_features(self, family_final_out: torch.Tensor) -> Dict:
        """解析家庭特征"""
        batch_size = family_final_out.shape[0]
        
        # 提取家庭规模（第一维）
        family_sizes = torch.clamp(torch.round(family_final_out[:, 0]), 1, self.max_family_size)
        
        # 提取其他特征
        working_members = torch.clamp(torch.round(family_final_out[:, 1]), 0, family_sizes)
        vehicles = family_final_out[:, 2:7]  # 各种交通工具
        
        # 解析概率特征
        have_student_probs = F.softmax(family_final_out[:, 7:9], dim=1)
        income_probs = F.softmax(family_final_out[:, 9:19], dim=1)
        
        return {
            'family_sizes': family_sizes.long(),
            'working_members': working_members.long(),
            'vehicles': vehicles,
            'have_student_probs': have_student_probs,
            'income_probs': income_probs
        }
    
    def _generate_family_graphs(self, family_encoding: torch.Tensor, family_info: Dict, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成家庭图结构"""
        batch_size = family_encoding.shape[0]
        
        # 初始化输出张量
        adjacency_matrices = torch.zeros(
            batch_size, self.max_family_size, self.max_family_size, len(self.edge_types),
            device=device
        )
        node_features = torch.zeros(
            batch_size, self.max_family_size, self.node_feature_dim,
            device=device
        )
        family_masks = torch.zeros(batch_size, self.max_family_size, device=device)
        
        for i in range(batch_size):
            family_size = family_info['family_sizes'][i].item()
            family_masks[i, :family_size] = 1.0
            
            # 生成家庭结构模板
            family_structure = self._generate_family_structure_template(
                family_size, family_info, i
            )
            
            # 生成节点特征
            node_features[i] = self._generate_node_features(
                family_encoding[i], family_structure, device
            )
            
            # 生成邻接矩阵
            adjacency_matrices[i] = self._generate_adjacency_matrix(
                family_encoding[i], family_structure, device
            )
        
        return adjacency_matrices, node_features, family_masks
    
    def _generate_family_structure_template(self, family_size: int, family_info: Dict, batch_idx: int) -> List[Dict]:
        """生成家庭结构模板"""
        have_student_prob = family_info['have_student_probs'][batch_idx]
        have_student = torch.multinomial(have_student_prob, 1).item()
        
        # 基于家庭规模选择结构模板
        if family_size == 1:
            structure = [{'relation': 'household_head', 'age_group': 'middle_aged'}]
        elif family_size == 2:
            if random.random() < 0.7:  # 夫妻
                structure = [
                    {'relation': 'household_head', 'age_group': 'middle_aged'},
                    {'relation': 'spouse', 'age_group': 'middle_aged'}
                ]
            else:  # 父子或母子
                structure = [
                    {'relation': 'household_head', 'age_group': 'middle_aged'},
                    {'relation': 'child', 'age_group': 'young_adult' if not have_student else 'teen'}
                ]
        elif family_size == 3:
            structure = [
                {'relation': 'household_head', 'age_group': 'middle_aged'},
                {'relation': 'spouse', 'age_group': 'middle_aged'},
                {'relation': 'child', 'age_group': 'teen' if have_student else 'child'}
            ]
        elif family_size == 4:
            structure = [
                {'relation': 'household_head', 'age_group': 'middle_aged'},
                {'relation': 'spouse', 'age_group': 'middle_aged'},
                {'relation': 'child', 'age_group': 'teen'},
                {'relation': 'child', 'age_group': 'child'}
            ]
        else:  # 5人以上大家庭
            structure = [
                {'relation': 'household_head', 'age_group': 'middle_aged'},
                {'relation': 'spouse', 'age_group': 'middle_aged'}
            ]
            
            # 添加子女
            num_children = min(3, family_size - 2)
            for i in range(num_children):
                age_group = ['teen', 'child', 'young_adult'][i % 3]
                structure.append({'relation': 'child', 'age_group': age_group})
            
            # 如果还有空位，添加父母或其他亲属
            remaining = family_size - len(structure)
            for i in range(remaining):
                if i == 0:
                    structure.append({'relation': 'parent', 'age_group': 'elderly'})
                else:
                    structure.append({'relation': 'other_relative', 'age_group': 'middle_aged'})
        
        # 确保达到指定的家庭规模
        while len(structure) < family_size:
            structure.append({'relation': 'other_relative', 'age_group': 'middle_aged'})
        
        return structure[:family_size]
    
    def _generate_node_features(self, family_encoding: torch.Tensor, family_structure: List[Dict], device) -> torch.Tensor:
        """生成节点特征矩阵"""
        node_features = torch.zeros(self.max_family_size, self.node_feature_dim, device=device)
        
        for i, member in enumerate(family_structure):
            # 构建输入特征：家庭编码 + 节点索引 + 关系类型 + 年龄组
            node_idx = torch.tensor([i / self.max_family_size], device=device)  # 归一化的节点索引
            relation_type = torch.tensor([self.relation_types[member['relation']] / len(self.relation_types)], device=device)
            age_group = torch.tensor([self.age_groups[member['age_group']] / len(self.age_groups)], device=device)
            
            node_input = torch.cat([
                family_encoding,
                node_idx,
                relation_type,
                age_group
            ])
            
            # 生成节点特征
            node_features[i] = self.node_generator(node_input)
        
        return node_features
    
    def _generate_adjacency_matrix(self, family_encoding: torch.Tensor, family_structure: List[Dict], device) -> torch.Tensor:
        """生成邻接矩阵"""
        family_size = len(family_structure)
        adj_matrix = torch.zeros(self.max_family_size, self.max_family_size, len(self.edge_types), device=device)
        
        # 基于家庭结构规则生成边
        relations = [member['relation'] for member in family_structure]
        
        # 找到关键成员的索引
        household_head_idx = None
        spouse_idx = None
        children_idx = []
        parent_idx = None
        
        for i, relation in enumerate(relations):
            if relation == 'household_head':
                household_head_idx = i
            elif relation == 'spouse':
                spouse_idx = i
            elif relation == 'child':
                children_idx.append(i)
            elif relation == 'parent':
                parent_idx = i
        
        # 生成关系边
        edges_to_generate = []
        
        # 夫妻关系
        if household_head_idx is not None and spouse_idx is not None:
            edges_to_generate.append((household_head_idx, spouse_idx, 'spouse'))
        
        # 父子关系
        if household_head_idx is not None:
            for child_idx in children_idx:
                edges_to_generate.append((household_head_idx, child_idx, 'parent_child'))
            if parent_idx is not None:
                edges_to_generate.append((parent_idx, household_head_idx, 'parent_child'))
        
        if spouse_idx is not None:
            for child_idx in children_idx:
                edges_to_generate.append((spouse_idx, child_idx, 'parent_child'))
        
        # 兄弟姐妹关系
        for i in range(len(children_idx)):
            for j in range(i+1, len(children_idx)):
                edges_to_generate.append((children_idx[i], children_idx[j], 'sibling'))
        
        # 使用神经网络生成边的权重
        for i, j, edge_type in edges_to_generate:
            # 构建边的输入特征
            edge_input = torch.cat([
                family_encoding,
                torch.tensor([i / self.max_family_size, j / self.max_family_size], device=device)
            ])
            
            # 生成边的概率分布
            edge_probs = F.softmax(self.adjacency_generator(edge_input), dim=0)
            
            # 设置对应边类型的权重
            edge_type_idx = self.edge_types[edge_type]
            adj_matrix[i, j, edge_type_idx] = edge_probs[edge_type_idx]
            adj_matrix[j, i, edge_type_idx] = edge_probs[edge_type_idx]  # 对称矩阵
        
        return adj_matrix
    
    def sample_family_graphs(self, family_final_out: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样模式生成家庭图（用于推理时）"""
        with torch.no_grad():
            adjacency_matrices, node_features, family_masks = self.forward(family_final_out)
            
            # 对邻接矩阵进行采样
            if temperature != 1.0:
                adjacency_matrices = adjacency_matrices / temperature
            
            # 二值化邻接矩阵（可选）
            adjacency_binary = (adjacency_matrices > 0.5).float()
            
            return adjacency_binary, node_features, family_masks


class FamilyGraphVAE(nn.Module):
    """基于VAE的家庭图生成模型"""
    
    def __init__(self, max_family_size=8, node_feature_dim=16, latent_dim=64):
        super().__init__()
        self.max_family_size = max_family_size
        self.node_feature_dim = node_feature_dim
        self.latent_dim = latent_dim
        
        # 编码器：将家庭特征编码为潜在表示
        self.encoder = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        # 解码器：从潜在表示生成图结构
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # 节点特征解码器
        self.node_decoder = nn.Linear(512, max_family_size * node_feature_dim)
        
        # 邻接矩阵解码器
        self.adj_decoder = nn.Linear(512, max_family_size * max_family_size * 5)  # 5种边类型
    
    def encode(self, family_final_out):
        """编码"""
        h = self.encoder(family_final_out)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码"""
        h = self.decoder(z)
        
        # 生成节点特征
        node_features = self.node_decoder(h)
        node_features = node_features.view(-1, self.max_family_size, self.node_feature_dim)
        
        # 生成邻接矩阵
        adj_logits = self.adj_decoder(h)
        adj_logits = adj_logits.view(-1, self.max_family_size, self.max_family_size, 5)
        adjacency_matrices = torch.sigmoid(adj_logits)
        
        return adjacency_matrices, node_features
    
    def forward(self, family_final_out):
        """前向传播"""
        mu, logvar = self.encode(family_final_out)
        z = self.reparameterize(mu, logvar)
        adjacency_matrices, node_features = self.decode(z)
        
        return adjacency_matrices, node_features, mu, logvar
    
    def loss_function(self, adjacency_matrices, node_features, mu, logvar, target_adj=None, target_nodes=None):
        """VAE损失函数"""
        # 重构损失
        recon_loss = 0
        if target_adj is not None:
            recon_loss += F.mse_loss(adjacency_matrices, target_adj)
        if target_nodes is not None:
            recon_loss += F.mse_loss(node_features, target_nodes)
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + 0.001 * kl_loss, recon_loss, kl_loss


# 使用示例和工具函数
def create_family_graph_pipeline(family_final_out: torch.Tensor, use_vae: bool = False):
    """创建家庭图生成管道"""
    
    if use_vae:
        generator = FamilyGraphVAE()
        adjacency_matrices, node_features, mu, logvar = generator(family_final_out)
        family_masks = torch.ones(family_final_out.shape[0], 8)  # 简化的掩码
    else:
        generator = FamilyGraphGenerator()
        adjacency_matrices, node_features, family_masks = generator(family_final_out)
    
    return adjacency_matrices, node_features, family_masks


def prepare_conditional_input_for_second_dit(adjacency_matrices: torch.Tensor, 
                                           node_features: torch.Tensor, 
                                           family_masks: torch.Tensor) -> torch.Tensor:
    """为第二个DiT准备条件输入"""
    batch_size, max_family_size, feature_dim = node_features.shape
    
    # 展平邻接矩阵
    adj_flat = adjacency_matrices.reshape(batch_size, -1)
    
    # 展平节点特征
    node_flat = node_features.reshape(batch_size, -1)
    
    # 展平掩码
    mask_flat = family_masks.reshape(batch_size, -1)
    
    # 拼接所有条件信息
    conditional_input = torch.cat([adj_flat, node_flat, mask_flat], dim=1)
    
    return conditional_input


if __name__ == "__main__":
    # 测试示例
    batch_size = 4
    
    # 模拟第一个DiT的输出
    family_final_out = torch.randn(batch_size, 19)
    family_final_out[:, 0] = torch.clamp(torch.normal(3.5, 1.5, (batch_size,)), 1, 8)  # 家庭规模
    
    print("第一个DiT输出形状:", family_final_out.shape)
    
    # 生成家庭图结构
    adjacency_matrices, node_features, family_masks = create_family_graph_pipeline(family_final_out)
    
    print("邻接矩阵形状:", adjacency_matrices.shape)
    print("节点特征矩阵形状:", node_features.shape)
    print("家庭掩码形状:", family_masks.shape)
    
    # 为第二个DiT准备条件输入
    conditional_input = prepare_conditional_input_for_second_dit(adjacency_matrices, node_features, family_masks)
    
    print("第二个DiT的条件输入形状:", conditional_input.shape)
    print("成功生成家庭图结构！")
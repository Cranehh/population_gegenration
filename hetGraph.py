import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
def convert_graphvae_to_differentiable(decoder, use_gumbel_softmax=True, temperature=1.0, hard=False):
    """
    将GraphVAE的logits输出转换为可微分的软图结构

    Args:
        decoder: GraphVAE decoder对象
        use_gumbel_softmax: 是否使用Gumbel Softmax技巧
        temperature: Gumbel Softmax温度参数
        hard: 是否使用hard模式（前向传播时离散，反向传播时连续）

    Returns:
        adj_soft: 软邻接矩阵 [batch, 8, 8]
        edge_soft: 软边类型分布 [batch, 8, 8, 5]
        node_soft: 软节点类型分布 [batch, 8, 6]
    """
    # 获取logits
    adj_logits = decoder._tilde_structure.adj_matrices_special_diag  # [batch, 8, 8]
    edge_logits = decoder._tilde_structure.edge_atr_tensors  # [batch, 8, 8, 5]
    node_logits = decoder._tilde_structure.node_atr_matrices  # [batch, 8, 6]

    # 可微分转换
    if use_gumbel_softmax:
        # 使用Gumbel Softmax保持可微分性
        adj_soft = torch.sigmoid(adj_logits)  # 边存在概率保持sigmoid

        # 边类型使用Gumbel Softmax
        edge_soft = F.gumbel_softmax(edge_logits, tau=temperature, hard=hard, dim=-1)

        # 节点类型使用Gumbel Softmax
        node_soft = F.gumbel_softmax(node_logits, tau=temperature, hard=hard, dim=-1)
    else:
        # 直接使用软概率分布
        adj_soft = torch.sigmoid(adj_logits)
        edge_soft = F.softmax(edge_logits, dim=-1)
        node_soft = F.softmax(node_logits, dim=-1)


    # 1. 找出有效连接（adj > 0.8）[batch, 8, 8]
    valid_edge_mask = (adj_soft > 0.8).float()

    # 2. 找出有效节点（行和 > 0.8）[batch, 8]
    node_degree = adj_soft.sum(dim=2)  # [batch, 8]
    valid_node_mask = (node_degree > 0.8).float()  # [batch, 8]

    # 3. 处理没有有效节点的情况：保留第一个节点
    # 检查每个batch是否至少有一个有效节点
    has_valid_nodes = valid_node_mask.sum(dim=1) > 0  # [batch]
    # 对于没有有效节点的batch，强制第一个节点为有效
    valid_node_mask[:, 0] = torch.where(has_valid_nodes, valid_node_mask[:, 0], torch.ones_like(valid_node_mask[:, 0]))

    # 4. 过滤邻接矩阵：只保留有效节点之间的有效连接
    # 创建节点掩码的广播版本 [batch, 8, 1] 和 [batch, 1, 8]
    node_mask_i = valid_node_mask.unsqueeze(2)  # [batch, 8, 1]
    node_mask_j = valid_node_mask.unsqueeze(1)  # [batch, 1, 8]

    # 边必须连接两个有效节点，且边权重 > 0.8
    valid_edge_mask = valid_edge_mask * node_mask_i * node_mask_j  # [batch, 8, 8]


    # 5. 过滤边特征：使用掩码加权
    # edge_soft: [batch, 8, 8, 5]
    # valid_edge_mask: [batch, 8, 8] -> [batch, 8, 8, 1]
    edge_mask_expanded = valid_edge_mask.unsqueeze(-1)  # [batch, 8, 8, 1]
    edge_filtered = edge_soft * edge_mask_expanded  # [batch, 8, 8, 5]

    # 6. 过滤节点特征：使用掩码加权
    # node_soft: [batch, 8, 6]
    # valid_node_mask: [batch, 8] -> [batch, 8, 1]
    node_mask_expanded = valid_node_mask.unsqueeze(-1)  # [batch, 8, 1]
    node_filtered = node_soft * node_mask_expanded  # [batch, 8, 6]


    return adj_soft, edge_filtered, node_filtered


def create_differentiable_hgt_data(decoder, family_features=None, temperature=1.0, hard=False):
    """
    创建可微分的HGT输入数据（保持梯度流）

    Args:
        decoder: GraphVAE decoder对象
        family_features: 家庭级特征 [batch_size, feature_dim]
        temperature: Gumbel Softmax温度
        hard: 是否使用hard模式

    Returns:
        dict: 包含软概率分布的HGT输入数据
    """
    adj_soft, edge_soft, node_soft = convert_graphvae_to_differentiable(
        decoder, use_gumbel_softmax=True, temperature=temperature, hard=hard)
    # adj_soft = adj_soft[:100]
    # edge_soft = edge_soft[:100]
    # node_soft = node_soft[:100]
    batch_size, max_nodes = adj_soft.shape[:2]

    # 构建可微分的图数据结构
    hgt_data = {
        'adj_matrix': adj_soft,  # [batch, 8, 8] 软邻接矩阵
        'edge_types': edge_soft,  # [batch, 8, 8, 5] 软边类型分布
        'node_types': node_soft,  # [batch, 8, 6] 软节点类型分布
        'family_features': family_features,  # [batch, feature_dim] 家庭特征
        'batch_size': batch_size,
        'max_nodes': max_nodes
    }

    return hgt_data


import math


class DifferentiableDenseHGTConv(nn.Module):
    """
    基于原版HGTConv的可微分密集版本，能够处理软概率分布输入
    保持与原版相同的异构注意力机制和关系感知变换
    """

    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=True, use_RTE=False):
        super(DifferentiableDenseHGTConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE

        # 与原版相同的类型特定线性层
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        # 与原版相同的关系感知参数
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        # 时序编码（如果需要）
        # if self.use_RTE:
        #     self.emb = RelTemporalEncoding(in_dim)

        # 初始化参数
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, node_features, node_types_soft, adj_matrix_soft, edge_types_soft, edge_time=None):
        """
        可微分前向传播

        Args:
            node_features: [batch, max_nodes, in_dim] 节点特征
            node_types_soft: [batch, max_nodes, num_types] 软节点类型分布
            adj_matrix_soft: [batch, max_nodes, max_nodes] 软邻接矩阵
            edge_types_soft: [batch, max_nodes, max_nodes, num_relations] 软边类型分布
            edge_time: [batch, max_nodes, max_nodes] 边时间（可选）

        Returns:
            output: [batch, max_nodes, out_dim] 输出节点特征
        """
        batch_size, max_nodes, _ = node_features.shape
        device = node_features.device

        # 初始化输出
        output = torch.zeros(batch_size, max_nodes, self.out_dim, device=device)

        # 对每个批次处理
        for b in range(batch_size):
            batch_output = self._forward_single_batch(
                node_features[b], node_types_soft[b],
                adj_matrix_soft[b], edge_types_soft[b],
                edge_time[b] if edge_time is not None else None
            )
            output[b] = batch_output

        return output

    def _forward_single_batch(self, node_features, node_types_soft, adj_matrix_soft, edge_types_soft, edge_time=None):
        """
        单个批次的可微分前向传播（保持原版HGTConv的逻辑）
        """
        max_nodes = node_features.shape[0]
        device = node_features.device

        # 步骤1: 计算异构互相注意力和消息
        res_att = torch.zeros(max_nodes, max_nodes, self.n_heads, device=device)
        res_msg = torch.zeros(max_nodes, max_nodes, self.n_heads, self.d_k, device=device)

        # 遍历所有源节点类型
        for source_type in range(self.num_types):
            # 软选择源节点
            source_mask = node_types_soft[:, source_type]  # [max_nodes]
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]

            # 遍历所有目标节点类型
            for target_type in range(self.num_types):
                # 软选择目标节点
                target_mask = node_types_soft[:, target_type]  # [max_nodes]
                q_linear = self.q_linears[target_type]

                # 遍历所有关系类型
                for relation_type in range(self.num_relations):
                    # 软选择关系类型
                    relation_mask = edge_types_soft[:, :, relation_type]  # [max_nodes, max_nodes]

                    # 组合掩码：源类型 × 目标类型 × 关系类型 × 邻接
                    combined_mask = (source_mask.unsqueeze(0) * target_mask.unsqueeze(1) *
                                     relation_mask * adj_matrix_soft)  # [max_nodes, max_nodes]

                    if combined_mask.sum() < 1e-6:
                        continue

                    # 计算Q, K, V
                    target_features = node_features  # [max_nodes, in_dim]
                    source_features = node_features  # [max_nodes, in_dim]

                    # 时序编码（如果启用）
                    if self.use_RTE and edge_time is not None:
                        source_features = self.emb(source_features, edge_time)

                    # 计算Q, K, V矩阵
                    q_mat = q_linear(target_features).view(max_nodes, self.n_heads,
                                                           self.d_k)  # [max_nodes, n_heads, d_k]
                    k_mat = k_linear(source_features).view(max_nodes, self.n_heads,
                                                           self.d_k)  # [max_nodes, n_heads, d_k]
                    v_mat = v_linear(source_features).view(max_nodes, self.n_heads,
                                                           self.d_k)  # [max_nodes, n_heads, d_k]

                    # 关系感知变换

                    k_mat_transformed = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    # k_mat_transformed = torch.einsum('nhd,rhd->nrh', k_mat, self.relation_att[relation_type])
                    # k_mat_transformed = k_mat_transformed[:, relation_type, :]  # [max_nodes, n_heads, d_k]

                    v_mat_transformed = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
                    # v_mat_transformed = v_mat_transformed[:, relation_type, :]  # [max_nodes, n_heads, d_k]

                    # 计算注意力分数
                    att_scores = torch.einsum('ihd,jhd->ijh', q_mat, k_mat_transformed) / self.sqrt_dk
                    att_scores = att_scores * self.relation_pri[relation_type].unsqueeze(0)

                    # 应用软掩码
                    att_scores = att_scores * combined_mask.unsqueeze(-1)

                    # 累积注意力和消息
                    res_att += att_scores
                    res_msg += torch.einsum('ijh,jhd->ijhd', att_scores, v_mat_transformed)

        # 步骤2: Softmax注意力归一化
        # 对每个目标节点进行归一化
        attention_weights = torch.zeros_like(res_att)
        for i in range(max_nodes):
            if res_att[i].sum() > 1e-6:
                attention_weights[i] = F.softmax(res_att[i], dim=0)

        # 步骤3: 加权聚合消息
        aggregated = torch.einsum('ijh,ijhd->ihd', attention_weights, res_msg)
        aggregated = aggregated.view(max_nodes, self.out_dim)  # [max_nodes, out_dim]

        # 步骤4: 类型特定输出变换
        output = torch.zeros(max_nodes, self.out_dim, device=device)
        for target_type in range(self.num_types):
            type_mask = node_types_soft[:, target_type]  # [max_nodes]

            if type_mask.sum() < 1e-6:
                continue

            # 类型特定变换
            transformed = self.drop(self.a_linears[target_type](F.gelu(aggregated)))

            # 跳跃连接
            alpha = torch.sigmoid(self.skip[target_type])
            residual_output = transformed * alpha + node_features * (1 - alpha)

            # 归一化
            if self.use_norm:
                residual_output = self.norms[target_type](residual_output)

            # 软组合
            output += residual_output * type_mask.unsqueeze(-1)

        return output


class DifferentiableHGT(nn.Module):
    """
    完整的可微分HGT模型（基于原版架构）
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_node_types, num_relations,
                 n_heads=4, n_layers=2, dropout=0.2, use_norm=True):
        super(DifferentiableHGT, self).__init__()

        self.input_projection = nn.Linear(num_node_types, in_dim)  # 将one-hot投影到特征空间

        # HGT层
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in_dim = in_dim if i == 0 else hidden_dim
            layer_out_dim = hidden_dim if i < n_layers - 1 else out_dim

            self.layers.append(DifferentiableDenseHGTConv(
                layer_in_dim, layer_out_dim, num_node_types, num_relations,
                n_heads, dropout, use_norm
            ))

    def forward(self, hgt_data):
        """
        Args:
            hgt_data: 包含软概率分布的数据字典
        Returns:
            node_embeddings: [batch, max_nodes, out_dim]
        """
        node_types_soft = hgt_data['node_types']  # [batch, max_nodes, num_types]
        adj_matrix_soft = hgt_data['adj_matrix']  # [batch, max_nodes, max_nodes]
        edge_types_soft = hgt_data['edge_types']  # [batch, max_nodes, max_nodes, num_relations]

        # 初始节点特征
        x = self.input_projection(node_types_soft)  # [batch, max_nodes, in_dim]

        # 通过HGT层
        for layer in self.layers:
            x = layer(x, node_types_soft, adj_matrix_soft, edge_types_soft)

        return x
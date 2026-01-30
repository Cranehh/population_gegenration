##total_member的loss也绑定成和mask相关
## income 和 age reclass的损失函数
## mask中只考虑有效的
## 分类损失都改为focal

## 添加户主和配偶的唯一约束
import torch
import torch.nn.functional as F

def focal_loss(input, target, alpha=1.0, gamma=2.0, reduction='none'):
    """
    Focal Loss 实现（支持与 F.cross_entropy 一致的输入）
    
    Args:
        input: [N, C] logits (未经softmax)
        target: [N] 类别索引（非one-hot）
        alpha: 类别平衡参数
        gamma: 聚焦参数
        reduction: 'none' | 'mean' | 'sum'
    """
    ce_loss = F.cross_entropy(input, target, reduction='none')  # 标准交叉熵
    pt = torch.exp(-ce_loss)  # pt = softmax概率
    focal_term = (1 - pt) ** gamma
    loss = alpha * focal_term * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def compute_family_losses(family_pred, family_true):
    """
    计算家庭层面的损失函数
    
    Args:
        family_pred: [batch_size, family_feature_dim] 预测的家庭特征
        family_true: [batch_size, family_feature_dim] 真实的家庭特征
    
    Returns:
        dict: 包含各种损失的字典
    """
    losses = {}
    
    # 连续变量损失 (前7个特征: 成员数量,工作人口数,机动车数量等)
    family_continuous_loss = F.mse_loss(
        family_pred[:, :8], 
        family_true[:, :8],
        reduction="none"
    )
    losses['family_continuous'] = torch.sum(family_continuous_loss, dim=-1)
    
    # 学生状态分类损失 (第8-9个特征)
    losses['family_student'] = focal_loss(
        family_pred[:, 8:10],
        family_true[:, 8:10].argmax(dim=-1),
        reduction="none"
    )
    
    # 收入类别分类损失 (第10-19个特征)
    # losses['family_income'] = F.cross_entropy(
    #     family_pred[:, 9:19],
    #     family_true[:, 9:19].argmax(dim=-1),
    #     reduction="none"
    # )
    
    return losses


def compute_person_losses(person_pred, person_true, person_mask, family_pred):
    """
    计算个人层面的损失函数
    
    Args:
        person_pred: [batch_size, max_family_size, person_feature_dim] 预测的个人特征
        person_true: [batch_size, max_family_size, person_feature_dim] 真实的个人特征
    
    Returns:
        dict: 包含各种损失的字典
    """
    batch_size, max_nodes = person_mask.shape
    person_idx = torch.arange(batch_size, device=person_mask.device).repeat_interleave(max_nodes)
    graph_idx_associated_with_considered_person = person_idx[person_mask.view(-1)]
    losses = {}
    
    # 年龄连续变量损失 (第1个特征)
    losses['person_age'] = F.mse_loss(
        person_pred[:, :, :1][person_mask], 
        person_true[:, :, :1][person_mask],
        reduction="none"
    ).mean(dim=-1)

    losses['person_age'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_age'].dtype, device=losses['person_age'].device), 0, graph_idx_associated_with_considered_person, losses['person_age'])
    
    # 性别分类损失 (第2-3个特征)
    losses['person_gender'] = focal_loss(
        person_pred[:, :, 1:3][person_mask],
        person_true[:, :, 1:3][person_mask].argmax(dim=-1),
        reduction="none"
    )
    losses['person_gender'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_gender'].dtype, device=losses['person_gender'].device), 0, graph_idx_associated_with_considered_person, losses['person_gender'])
    
    # 驾照状态分类损失 (第4-5个特征)
    losses['person_license'] = focal_loss(
        person_pred[:, :, 3:5][person_mask],
        person_true[:, :, 3:5][person_mask].argmax(dim=-1),
        reduction="none"
    )
    losses['person_license'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_license'].dtype, device=losses['person_license'].device), 0, graph_idx_associated_with_considered_person, losses['person_license'])
    
    # 关系分类损失 (第6-21个特征)
    losses['person_relation'] = focal_loss(
        person_pred[:, :, 5:21][person_mask],
        person_true[:, :, 5:21][person_mask].argmax(dim=-1),
        reduction="none"
    )
    losses['person_relation'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_relation'].dtype, device=losses['person_relation'].device), 0, graph_idx_associated_with_considered_person, losses['person_relation'])
    
    # 教育程度分类损失 (第22-30个特征)
    losses['person_education'] = focal_loss(
        person_pred[:, :, 21:30][person_mask],
        person_true[:, :, 21:30][person_mask].argmax(dim=-1),
        reduction="none"
    )
    losses['person_education'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_education'].dtype, device=losses['person_education'].device), 0, graph_idx_associated_with_considered_person, losses['person_education'])
    
    # 职业分类损失 (第31-50个特征)
    losses['person_occupation'] = focal_loss(
        person_pred[:, :, 30:50][person_mask],
        person_true[:, :, 30:50][person_mask].argmax(dim=-1),
        reduction="none"
    )
    losses['person_occupation'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_occupation'].dtype, device=losses['person_occupation'].device), 0, graph_idx_associated_with_considered_person, losses['person_occupation'])

    # invalid_indices = ~person_mask
    # person_invalid_idx = torch.arange(batch_size, device=person_mask.device).repeat_interleave(max_nodes)
    # graph_idx_associated_with_considered_invalid_person = person_invalid_idx[invalid_indices.view(-1)]
    #
    # pred_invalid = person_pred[invalid_indices]
    # zero_target = torch.zeros_like(pred_invalid)
    # zero_loss = F.mse_loss(pred_invalid, zero_target, reduction='none').mean(dim=-1)
    #
    # zero_loss = torch.scatter_add(torch.zeros(batch_size, dtype=zero_loss.dtype, device=zero_loss.device), 0, graph_idx_associated_with_considered_invalid_person, zero_loss)
    #
    # losses['invalid_person'] = zero_loss


    ## mask loss

    pred_person_mask = person_pred[:, :, -1]  # 预测的掩码概率
    true_person_mask = person_mask.float()  # 真实的掩码标签
    bce_loss = F.binary_cross_entropy_with_logits(pred_person_mask, true_person_mask, reduction='none')
    bce_loss = torch.sum(bce_loss, dim=-1) / max_nodes
    losses['mask_loss'] = bce_loss


    ## total member loss
    pred_person_number_mask = pred_person_mask[person_mask]  # 只考虑存在的成员
    pred_total_member = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_age'].dtype, device=pred_person_number_mask.device), 0, graph_idx_associated_with_considered_person, pred_person_number_mask)

    losses['total_member_loss'] = F.mse_loss(pred_total_member, family_pred[:, 0] * 0.88397094 + 2.38862088, reduction='none')

    # losses['total_member_loss'] = torch.scatter_add(torch.zeros(batch_size, dtype=losses['person_age'].dtype, device=losses['person_age'].device), 0, graph_idx_associated_with_considered_person, losses['person_age'])

    ## 户主和配偶唯一约束
    pred_head_num = person_pred[:,:,5]
    loss_head_count = (pred_head_num.sum(dim=1) - 1).abs()

    loss_head_multi_peak = pred_head_num.topk(2, dim=1).values[:, 1]

    pred_sprouse_num = person_pred[:,:,20]
    loss_spouse_count = F.relu(pred_sprouse_num.sum(dim=1) - 1)
    loss_spouse_multi_peak = pred_sprouse_num.topk(2, dim=1).values[:, 1]

    losses['unique_loss'] = loss_head_count + loss_head_multi_peak + loss_spouse_count + loss_spouse_multi_peak
    
    return losses


def compute_adjacency_loss(decoder_adj_logits, true_adj_matrix):
    """
    邻接矩阵二元交叉熵损失
    
    Args:
        decoder_adj_logits: [batch_size, max_family_size, max_family_size] 预测的邻接矩阵logits
        true_adj_matrix: [batch_size, max_family_size, max_family_size] 真实邻接矩阵
    
    Returns:
        torch.Tensor: [batch_size] 每个样本的邻接矩阵损失
    """
    batch_size, max_nodes, _ = decoder_adj_logits.shape
    
    # 展平为向量形式
    adj_logits_flat = decoder_adj_logits.view(-1)
    adj_truth_flat = true_adj_matrix.view(-1)
    
    # 计算二元交叉熵损失
    bce_loss = F.binary_cross_entropy_with_logits(
        adj_logits_flat, adj_truth_flat, reduction='none'
    )
    
    loss = torch.sum(bce_loss.view(batch_size, -1), dim=1) / (max_nodes ** 2)
    
    return loss


def compute_node_type_loss(decoder_node_logits, true_node_types, node_mask):
    """
    节点类型交叉熵损失
    
    Args:
        decoder_node_logits: [batch_size, max_family_size, num_node_types] 预测的节点类型
        true_node_types: [batch_size, max_family_size, num_node_types] 真实节点类型(one-hot)
        node_mask: [batch_size, max_family_size] 节点存在掩码
    
    Returns:
        torch.Tensor: [batch_size] 每个样本的节点类型损失
    """
    batch_size, max_nodes, num_classes = decoder_node_logits.shape
    
    # 创建图索引
    node_idx = torch.arange(batch_size, device=decoder_node_logits.device).repeat_interleave(max_nodes)
    graph_idx_associated_with_considered_node = node_idx[node_mask.view(-1)]
    num_nodes_active = torch.sum(node_mask, dim=-1, dtype=torch.float32)
    
    # 只对存在的节点计算损失
    active_node_logits = decoder_node_logits[node_mask]  # [num_active_nodes, num_classes]
    active_node_types = true_node_types[node_mask].argmax(dim=-1)  # [num_active_nodes]
    
    # 交叉熵损失
    node_loss = focal_loss(active_node_logits, active_node_types, reduction='none')
    node_loss = (torch.scatter_add(
        torch.zeros(batch_size, dtype=node_loss.dtype, device=node_loss.device), 
        0, graph_idx_associated_with_considered_node, node_loss
    ) / (num_nodes_active + 1e-6))
    
    return node_loss


def compute_edge_type_loss(decoder_edge_logits, true_edge_types, a_norm_one, valid_edge):
    """
    边类型交叉熵损失
    
    Args:
        decoder_edge_logits: [batch_size, max_family_size, max_family_size, num_edge_types]
        true_edge_types: [batch_size, max_family_size, max_family_size, num_edge_types] 真实边类型
        a_norm_one: [batch_size] 每个图的边数归一化因子
        valid_edge: [batch_size, max_family_size, max_family_size] 边存在掩码
    
    Returns:
        torch.Tensor: [batch_size] 每个样本的边类型损失
    """
    batch_size, max_nodes, _, _ = decoder_edge_logits.shape
    
    edge_mask = valid_edge.view(-1)
    edge_idx = torch.arange(batch_size, device=decoder_edge_logits.device).repeat_interleave(max_nodes * max_nodes)
    graph_idx_associated_with_considered_edge = edge_idx[edge_mask]
    
    # 提取存在边的logits和标签
    active_edge_logits = decoder_edge_logits[valid_edge]  # [num_active_edges, num_edge_types]
    active_edge_types = true_edge_types[valid_edge].argmax(dim=-1)  # [num_active_edges]
    
    # 交叉熵损失
    edge_loss = focal_loss(active_edge_logits, active_edge_types, reduction='none')
    edge_loss = torch.scatter_add(
        torch.zeros(batch_size, dtype=edge_loss.dtype, device=edge_loss.device), 
        0, graph_idx_associated_with_considered_edge, edge_loss
    )
    edge_loss = edge_loss / (a_norm_one + 1e-6)
    
    return edge_loss


def compute_total_loss(family_pred, family_true, person_pred, person_true, person_mask,
                      relation_graph, source_data_adj, source_data_edge, 
                      source_data_node, weights=None):
    """
    计算总损失
    
    Args:
        family_pred: 预测的家庭特征
        family_true: 真实的家庭特征  
        person_pred: 预测的个人特征
        person_true: 真实的个人特征
        relation_graph: 预测的关系图
        source_data_adj: 真实邻接矩阵
        source_data_edge: 真实边特征
        source_data_node: 真实节点特征
        weights: 损失权重字典
    
    Returns:
        dict: 包含总损失和各分项损失的字典
    """
    if weights is None:
        weights = {
            'family_continuous': 1.0,
            'family_student': 1.0,
            # 'family_income': 1.0,
            'person_age': 1.0,
            'person_gender': 1.0,
            'person_license': 1.0,
            'person_relation': 1.0,
            'person_education': 1.0,
            'person_occupation': 1.0,
            'invalid_person': 0.5,
            'graph_adj': 1.0,
            'graph_node': 1.0,
            'graph_edge': 1.0
        }
    
    # 计算各类损失
    family_losses = compute_family_losses(family_pred, family_true)
    person_losses = compute_person_losses(person_pred, person_true, person_mask, family_pred)
    
    # 计算掩码
    valid_person = torch.sum(source_data_node, dim=-1) != 0
    a_norm_one = torch.sum(source_data_adj, dim=[1, 2], dtype=torch.float32)
    valid_edge = torch.sum(source_data_edge, dim=-1) != 0
    
    # 图结构损失
    graph_adj_loss = compute_adjacency_loss(relation_graph['adj_matrix'], source_data_adj)
    graph_node_loss = compute_node_type_loss(relation_graph['node_types'], source_data_node, valid_person)
    graph_edge_loss = compute_edge_type_loss(relation_graph['edge_types'], source_data_edge, a_norm_one, valid_edge)
    
    # 汇总所有损失
    total_loss = (
        weights['family_continuous'] * family_losses['family_continuous'] +
        weights['family_student'] * family_losses['family_student'] +
        # weights['family_income'] * family_losses['family_income'] +
        weights['person_age'] * person_losses['person_age'] +
        weights['person_gender'] * person_losses['person_gender'] +
        weights['person_license'] * person_losses['person_license'] +
        weights['person_relation'] * person_losses['person_relation'] +
        weights['person_education'] * person_losses['person_education'] +
        weights['person_occupation'] * person_losses['person_occupation'] +
        # weights['invalid_person'] * person_losses['invalid_person'] +
        weights['mask_loss'] * person_losses['mask_loss'] +
        weights['total_member_loss'] * person_losses['total_member_loss'] +
        weights['unique_loss'] * person_losses['unique_loss'] +
        weights['graph_adj'] * graph_adj_loss +
        weights['graph_node'] * graph_node_loss +
        weights['graph_edge'] * graph_edge_loss
    )
    
    loss_dict = {
        'total_loss': total_loss,
        'family_continuous': family_losses['family_continuous'],
        'family_student': family_losses['family_student'],
        # 'family_income': family_losses['family_income'],
        'person_age': person_losses['person_age'],
        'person_gender': person_losses['person_gender'],
        'person_license': person_losses['person_license'],
        'person_relation': person_losses['person_relation'],
        'person_education': person_losses['person_education'],
        'person_occupation': person_losses['person_occupation'],
        # 'invalid_person': person_losses['invalid_person'],
        'mask_loss': person_losses['mask_loss'],
        'total_member_loss': person_losses['total_member_loss'],
        'unique_loss': person_losses['unique_loss'],
        'graph_adj': graph_adj_loss,
        'graph_node': graph_node_loss,
        'graph_edge': graph_edge_loss
    }
    
    return loss_dict
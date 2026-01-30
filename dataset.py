import torch
from torch.utils.data import Dataset
import numpy as np


class PopulationDataset(Dataset):
    """
    人口合成数据集类
    """
    def __init__(self, family_data, member_data, adj_data, edge_data, node_data):
        """
        Args:
            family_data: [N, family_feature_dim] 家庭特征数据
            member_data: [N, max_family_size, member_feature_dim] 成员特征数据
            adj_data: [N, max_family_size, max_family_size] 邻接矩阵数据
            edge_data: [N, max_family_size, max_family_size, num_edge_types] 边特征数据
            node_data: [N, max_family_size, num_node_types] 节点特征数据
        """
        self.family_data = torch.FloatTensor(family_data)
        self.member_data = torch.FloatTensor(member_data) 
        self.adj_data = torch.FloatTensor(adj_data)
        self.edge_data = torch.FloatTensor(edge_data)
        self.node_data = torch.FloatTensor(node_data)
        
        assert len(family_data) == len(member_data) == len(adj_data) == len(edge_data) == len(node_data)
        
    def __len__(self):
        return len(self.family_data)
    
    def __getitem__(self, idx):
        return {
            'family': self.family_data[idx, :10],
            'member': self.member_data[idx],
            'adj': self.adj_data[idx],
            'edge': self.edge_data[idx],
            'node': self.node_data[idx],
            'cluster': self.family_data[idx, 10],
            'cluster_profile': self.family_data[idx, 11:]  # 假设最后11维是聚类特征
        }


def load_population_data(data_dir="数据"):
    """
    加载人口数据
    
    Args:
        data_dir: 数据文件夹路径
    
    Returns:
        PopulationDataset: 人口数据集对象
    """
    family_data = np.load(f'{data_dir}/family_sample_improved_cluster.npy')
    member_data = np.load(f'{data_dir}/family_member_sample_improved_cluster.npy')
    adj_data = np.load(f'{data_dir}/family_adj.npy')
    edge_data = np.load(f'{data_dir}/familymember_relationship.npy')
    node_data = np.load(f'{data_dir}/familymember_type.npy')

    
    return PopulationDataset(family_data, member_data, adj_data, edge_data, node_data)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True):
    """
    创建数据加载器
    
    Args:
        dataset: PopulationDataset对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否固定内存
    
    Returns:
        DataLoader: 数据加载器
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
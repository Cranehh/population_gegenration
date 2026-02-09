"""
互信息先验模块

基于数据的互信息结构构建超参数采样分布：
- 计算变量间互信息矩阵
- 从互信息结构映射到Loss权重的先验分布（均值和方差）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

try:
    from sklearn.metrics import mutual_info_score
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import entropy
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn未安装，部分功能不可用")


class MutualInformationPrior:
    """
    基于互信息的超参数先验分布构建器
    
    核心思想：
    - 变量依赖强 → 对应Loss权重的先验均值高
    - 结构复杂/不确定 → 先验方差大
    """
    
    def __init__(
        self,
        n_bins: int = 20,
        base_mean: float = 1.0,
        base_std: float = 0.3,
        complexity_sensitivity: float = 1.0
    ):
        """
        初始化互信息先验构建器
        
        Args:
            n_bins: 离散化时的bin数量
            base_mean: 基础先验均值
            base_std: 基础先验标准差
            complexity_sensitivity: 复杂度对方差的影响系数
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("请安装sklearn: pip install scikit-learn")
        
        self.n_bins = n_bins
        self.base_mean = base_mean
        self.base_std = base_std
        self.complexity_sensitivity = complexity_sensitivity
        
        # 存储计算结果
        self.mi_matrix: Optional[np.ndarray] = None
        self.variable_groups: Dict[str, List[int]] = {}
        self.group_mi_stats: Dict[str, Dict] = {}
        self.prior_distribution: Dict[str, Dict[str, float]] = {}
    
    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """将连续变量离散化"""
        if len(np.unique(x)) <= self.n_bins:
            return x.astype(int)
        return np.digitize(x, np.histogram_bin_edges(x, bins=self.n_bins)[:-1])
    
    def _compute_entropy(self, x: np.ndarray) -> float:
        """计算单变量熵"""
        x_disc = self._discretize(x)
        _, counts = np.unique(x_disc, return_counts=True)
        return entropy(counts / counts.sum())
    
    def _compute_mutual_info(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两变量间的互信息"""
        x_disc = self._discretize(x)
        y_disc = self._discretize(y)
        return mutual_info_score(x_disc, y_disc)
    
    def _compute_normalized_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算归一化互信息"""
        mi = self._compute_mutual_info(x, y)
        h_x = self._compute_entropy(x)
        h_y = self._compute_entropy(y)
        
        if h_x == 0 or h_y == 0:
            return 0.0
        
        return mi / np.sqrt(h_x * h_y)
    
    def compute_mi_matrix(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        计算变量间的归一化互信息矩阵
        
        Args:
            data: 数据矩阵 [n_samples, n_variables]
            variable_names: 变量名列表
        
        Returns:
            归一化互信息矩阵 M, M_ij ∈ [0, 1]
        """
        n_vars = data.shape[1]
        self.mi_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(i, n_vars):
                if i == j:
                    self.mi_matrix[i, j] = 1.0
                else:
                    nmi = self._compute_normalized_mi(data[:, i], data[:, j])
                    self.mi_matrix[i, j] = nmi
                    self.mi_matrix[j, i] = nmi
        
        self.variable_names = variable_names or [f"var_{i}" for i in range(n_vars)]
        return self.mi_matrix
    
    def set_variable_groups(self, groups: Dict[str, List[int]]):
        """
        设置变量分组
        
        Args:
            groups: {组名: [变量索引列表]}
                例如: {'family': [0,1,2], 'person': [3,4,5,6], 'graph': [7,8,9]}
        """
        self.variable_groups = groups
    
    def compute_group_mi_statistics(self) -> Dict[str, Dict]:
        """
        计算各分组的互信息统计量
        
        Returns:
            {组名: {'within': 块内平均MI, 'cross': 跨块平均MI, 'spectral_entropy': 谱熵}}
        """
        if self.mi_matrix is None:
            raise ValueError("请先调用compute_mi_matrix()")
        
        if not self.variable_groups:
            raise ValueError("请先调用set_variable_groups()")
        
        self.group_mi_stats = {}
        group_names = list(self.variable_groups.keys())
        
        for group_name, indices in self.variable_groups.items():
            indices = np.array(indices)
            
            # 块内互信息（排除对角线）
            within_mi = []
            for i in indices:
                for j in indices:
                    if i != j:
                        within_mi.append(self.mi_matrix[i, j])
            avg_within = np.mean(within_mi) if within_mi else 0.0
            
            # 跨块互信息
            cross_mi = []
            other_indices = []
            for other_name, other_idx in self.variable_groups.items():
                if other_name != group_name:
                    other_indices.extend(other_idx)
            
            for i in indices:
                for j in other_indices:
                    cross_mi.append(self.mi_matrix[i, j])
            avg_cross = np.mean(cross_mi) if cross_mi else 0.0
            
            # 块的谱熵（衡量结构复杂度）
            block = self.mi_matrix[np.ix_(indices, indices)]
            eigenvalues = np.linalg.eigvalsh(block)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                eigenvalues = eigenvalues / eigenvalues.sum()
                spectral_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
            else:
                spectral_entropy = 0.0
            
            self.group_mi_stats[group_name] = {
                'within': avg_within,
                'cross': avg_cross,
                'spectral_entropy': spectral_entropy,
                'n_vars': len(indices)
            }
        
        return self.group_mi_stats
    
    def build_prior_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        从互信息统计量构建先验分布
        
        映射逻辑：
        - 均值 μ ∝ (块内依赖 + 跨块依赖)
        - 方差 σ² ∝ 谱熵（复杂度高则不确定性大）
        
        Returns:
            {参数名: {'mean': μ, 'std': σ}}
        """
        if not self.group_mi_stats:
            self.compute_group_mi_statistics()
        
        # 计算归一化因子
        total_importance = sum(
            stats['within'] + stats['cross']
            for stats in self.group_mi_stats.values()
        )
        if total_importance == 0:
            total_importance = 1.0
        
        max_spectral_entropy = max(
            stats['spectral_entropy'] for stats in self.group_mi_stats.values()
        )
        if max_spectral_entropy == 0:
            max_spectral_entropy = 1.0
        
        self.prior_distribution = {}
        
        # Loss权重参数的先验
        weight_param_map = {
            'family': 'family_weight_scale',
            'person': 'person_weight_scale',
            'graph': 'graph_weight_scale',
            'constraint': 'constraint_weight_scale'
        }
        
        for group_name, stats in self.group_mi_stats.items():
            param_name = weight_param_map.get(group_name, f'{group_name}_weight_scale')
            
            # 均值：依赖强度越高，权重越大
            importance = stats['within'] + stats['cross']
            mean = self.base_mean * (1 + importance / total_importance)
            
            # 方差：复杂度越高，不确定性越大
            complexity = stats['spectral_entropy'] / max_spectral_entropy
            std = self.base_std * (1 + self.complexity_sensitivity * complexity)
            
            self.prior_distribution[param_name] = {
                'mean': mean,
                'std': std,
                'mi_within': stats['within'],
                'mi_cross': stats['cross'],
                'complexity': complexity
            }
        
        # 约束权重：基于跨层耦合强度
        if len(self.group_mi_stats) > 1:
            total_cross = sum(stats['cross'] for stats in self.group_mi_stats.values())
            avg_cross = total_cross / len(self.group_mi_stats)
            
            self.prior_distribution['constraint_weight_scale'] = {
                'mean': self.base_mean * (1 + 2 * avg_cross),  # 跨层耦合强则约束重要
                'std': self.base_std * 1.5,  # 约束权重通常不确定性较大
                'mi_cross': avg_cross
            }
        
        # 噪声相关系数rho的先验
        if 'family' in self.group_mi_stats and 'person' in self.group_mi_stats:
            cross_fp = self._compute_cross_group_mi('family', 'person')
            self.prior_distribution['rho'] = {
                'mean': 0.7 + 0.25 * cross_fp,  # 跨层依赖强则rho高
                'std': 0.1,
                'mi_cross_fp': cross_fp
            }
        
        return self.prior_distribution
    
    def _compute_cross_group_mi(self, group1: str, group2: str) -> float:
        """计算两组之间的平均互信息"""
        idx1 = np.array(self.variable_groups[group1])
        idx2 = np.array(self.variable_groups[group2])
        
        cross_mi = []
        for i in idx1:
            for j in idx2:
                cross_mi.append(self.mi_matrix[i, j])
        
        return np.mean(cross_mi) if cross_mi else 0.0
    
    def sample_initial_configs(self, n_samples: int) -> List[Dict]:
        """
        从先验分布中采样初始配置
        
        Args:
            n_samples: 采样数量
        
        Returns:
            配置列表
        """
        if not self.prior_distribution:
            self.build_prior_distribution()
        
        configs = []
        for _ in range(n_samples):
            config = {}
            for param_name, dist in self.prior_distribution.items():
                # 从高斯分布采样，并裁剪到合理范围
                value = np.random.normal(dist['mean'], dist['std'])
                
                if param_name == 'rho':
                    value = np.clip(value, 0.5, 0.95)
                else:
                    value = np.clip(value, 0.1, 5.0)
                
                config[param_name] = value
            
            configs.append(config)
        
        return configs
    
    def get_prior_summary(self) -> str:
        """获取先验分布摘要"""
        if not self.prior_distribution:
            return "先验分布尚未构建"
        
        lines = ["=" * 50, "互信息先验分布摘要", "=" * 50]
        
        for param_name, dist in self.prior_distribution.items():
            lines.append(f"\n{param_name}:")
            lines.append(f"  均值: {dist['mean']:.4f}")
            lines.append(f"  标准差: {dist['std']:.4f}")
            if 'mi_within' in dist:
                lines.append(f"  块内MI: {dist['mi_within']:.4f}")
            if 'mi_cross' in dist:
                lines.append(f"  跨块MI: {dist['mi_cross']:.4f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def extract_mi_features_from_dataset(dataset, max_samples: int = 5000) -> MutualInformationPrior:
    """
    从PopulationDataset中提取互信息特征并构建先验
    
    Args:
        dataset: PopulationDataset实例
        max_samples: 最大采样数量（加速计算）
    
    Returns:
        配置好的MutualInformationPrior实例
    """
    # 采样数据
    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    family_data = []
    person_data = []
    
    for idx in indices:
        batch = dataset[idx]
        family_data.append(batch['family'].numpy())
        # 将person数据展平（只取有效成员）
        person = batch['member'].numpy()
        mask = np.sum(np.abs(person), axis=-1) > 0
        if mask.any():
            person_data.append(person[mask].mean(axis=0))
    
    family_data = np.array(family_data)
    person_data = np.array(person_data) if person_data else np.zeros((n_samples, 1))
    
    # 合并数据
    combined_data = np.hstack([family_data, person_data])
    
    # 构建先验
    mi_prior = MutualInformationPrior()
    mi_prior.compute_mi_matrix(combined_data)
    
    # 设置变量分组
    n_family = family_data.shape[1]
    n_person = person_data.shape[1]
    
    mi_prior.set_variable_groups({
        'family': list(range(n_family)),
        'person': list(range(n_family, n_family + n_person))
    })
    
    mi_prior.build_prior_distribution()
    
    return mi_prior

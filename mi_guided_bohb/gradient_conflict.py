"""
梯度冲突检测模块

在训练过程中检测各Loss分项之间的梯度冲突：
- 计算梯度余弦相似度
- 量化冲突暴露度
- 基于冲突信息更新采样分布
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class GradientConflictInfo:
    """梯度冲突信息"""
    conflict_matrix: np.ndarray  # 冲突矩阵 C_ij
    exposure: Dict[str, float]   # 各Loss的冲突暴露度
    magnitude: Dict[str, float]  # 各Loss的梯度量级
    imbalance: float             # 量级失衡系数


class GradientConflictDetector:
    """
    梯度冲突检测器
    
    核心功能：
    1. 计算各Loss分项的梯度
    2. 检测梯度间的冲突（余弦相似度为负）
    3. 计算冲突暴露度（某Loss被其他Loss压制的程度）
    """
    
    def __init__(
        self,
        loss_names: List[str],
        ema_decay: float = 0.9,
        conflict_threshold: float = 0.0
    ):
        """
        初始化梯度冲突检测器
        
        Args:
            loss_names: Loss分项名称列表
            ema_decay: 指数移动平均衰减系数
            conflict_threshold: 冲突判定阈值（cos < threshold视为冲突）
        """
        self.loss_names = loss_names
        self.n_losses = len(loss_names)
        self.ema_decay = ema_decay
        self.conflict_threshold = conflict_threshold
        
        # 历史记录（使用EMA平滑）
        self.ema_conflict_matrix = np.zeros((self.n_losses, self.n_losses))
        self.ema_magnitudes = np.ones(self.n_losses)
        self.history: List[GradientConflictInfo] = []
        
        # 名称到索引的映射
        self.name_to_idx = {name: i for i, name in enumerate(loss_names)}
    
    def compute_gradients(
        self,
        model: nn.Module,
        loss_dict: Dict[str, torch.Tensor],
        create_graph: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算各Loss分项对模型参数的梯度
        
        Args:
            model: 模型
            loss_dict: {loss_name: loss_tensor}
            create_graph: 是否创建计算图（用于二阶导数）
        
        Returns:
            {loss_name: flattened_gradient}
        """
        gradients = {}
        
        # 获取所有需要梯度的参数
        params = [p for p in model.parameters() if p.requires_grad]
        
        for loss_name, loss_tensor in loss_dict.items():
            if loss_name not in self.name_to_idx:
                continue
            
            # 计算梯度
            if loss_tensor.requires_grad:
                grads = torch.autograd.grad(
                    loss_tensor.mean(),
                    params,
                    retain_graph=True,
                    create_graph=create_graph,
                    allow_unused=True
                )
                
                # 展平并拼接所有参数的梯度
                flat_grad = torch.cat([
                    g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
                    for g, p in zip(grads, params)
                ])
                
                gradients[loss_name] = flat_grad
        
        return gradients
    
    def compute_conflict_matrix(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        计算梯度冲突矩阵
        
        C_ij = max(0, -cos(g_i, g_j))
        
        Args:
            gradients: {loss_name: gradient_tensor}
        
        Returns:
            冲突矩阵 [n_losses, n_losses]
        """
        conflict_matrix = np.zeros((self.n_losses, self.n_losses))
        
        grad_list = []
        for name in self.loss_names:
            if name in gradients:
                grad_list.append(gradients[name])
            else:
                # 如果某个Loss没有梯度，用零向量代替
                grad_list.append(torch.zeros_like(list(gradients.values())[0]))
        
        # 计算所有对之间的余弦相似度
        for i in range(self.n_losses):
            for j in range(i + 1, self.n_losses):
                g_i = grad_list[i]
                g_j = grad_list[j]
                
                norm_i = torch.norm(g_i)
                norm_j = torch.norm(g_j)
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = torch.dot(g_i, g_j) / (norm_i * norm_j)
                    # 冲突度 = max(0, -cos)
                    conflict = max(0, -cos_sim.item())
                else:
                    conflict = 0.0
                
                conflict_matrix[i, j] = conflict
                conflict_matrix[j, i] = conflict
        
        return conflict_matrix
    
    def compute_magnitudes(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算各Loss的梯度量级"""
        magnitudes = {}
        for name in self.loss_names:
            if name in gradients:
                magnitudes[name] = torch.norm(gradients[name]).item()
            else:
                magnitudes[name] = 0.0
        return magnitudes
    
    def compute_exposure(
        self,
        conflict_matrix: np.ndarray,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算冲突暴露度
        
        E_k = Σ_{j≠k} λ_j · C_kj
        
        表示第k个Loss被其他Loss"压制"的程度
        
        Args:
            conflict_matrix: 冲突矩阵
            weights: 当前Loss权重 {loss_name: weight}
        
        Returns:
            {loss_name: exposure}
        """
        exposure = {}
        
        for i, name_i in enumerate(self.loss_names):
            exp_i = 0.0
            for j, name_j in enumerate(self.loss_names):
                if i != j:
                    w_j = weights.get(name_j, 1.0)
                    exp_i += w_j * conflict_matrix[i, j]
            exposure[name_i] = exp_i
        
        return exposure
    
    def detect(
        self,
        model: nn.Module,
        loss_dict: Dict[str, torch.Tensor],
        weights: Dict[str, float]
    ) -> GradientConflictInfo:
        """
        执行完整的梯度冲突检测
        
        Args:
            model: 模型
            loss_dict: 各Loss分项
            weights: 当前权重配置
        
        Returns:
            GradientConflictInfo
        """
        # 计算梯度
        gradients = self.compute_gradients(model, loss_dict)
        
        # 计算冲突矩阵
        conflict_matrix = self.compute_conflict_matrix(gradients)
        
        # 计算梯度量级
        magnitudes = self.compute_magnitudes(gradients)
        
        # 计算冲突暴露度
        exposure = self.compute_exposure(conflict_matrix, weights)
        
        # 计算量级失衡系数
        mag_values = list(magnitudes.values())
        if min(mag_values) > 1e-8:
            imbalance = max(mag_values) / min(mag_values)
        else:
            imbalance = float('inf')
        
        # 更新EMA
        self.ema_conflict_matrix = (
            self.ema_decay * self.ema_conflict_matrix +
            (1 - self.ema_decay) * conflict_matrix
        )
        
        for i, name in enumerate(self.loss_names):
            self.ema_magnitudes[i] = (
                self.ema_decay * self.ema_magnitudes[i] +
                (1 - self.ema_decay) * magnitudes.get(name, 0.0)
            )
        
        info = GradientConflictInfo(
            conflict_matrix=conflict_matrix,
            exposure=exposure,
            magnitude=magnitudes,
            imbalance=imbalance
        )
        
        self.history.append(info)
        
        return info
    
    def get_aggregated_exposure(self, window: int = 10) -> Dict[str, float]:
        """
        获取最近window步的聚合冲突暴露度
        
        Args:
            window: 聚合窗口大小
        
        Returns:
            {loss_name: avg_exposure}
        """
        if not self.history:
            return {name: 0.0 for name in self.loss_names}
        
        recent = self.history[-window:]
        
        aggregated = defaultdict(float)
        for info in recent:
            for name, exp in info.exposure.items():
                aggregated[name] += exp
        
        return {name: aggregated[name] / len(recent) for name in self.loss_names}
    
    def get_ema_conflict_matrix(self) -> np.ndarray:
        """获取EMA平滑后的冲突矩阵"""
        return self.ema_conflict_matrix.copy()
    
    def clear_history(self):
        """清空历史记录"""
        self.history.clear()
        self.ema_conflict_matrix = np.zeros((self.n_losses, self.n_losses))
        self.ema_magnitudes = np.ones(self.n_losses)


class ConflictAwareDistributionUpdater:
    """
    基于冲突信息更新采样分布
    
    核心逻辑：
    - 高冲突暴露 → 方差膨胀 → 多探索
    - 低冲突 → 保持当前均值 → 多利用
    """
    
    def __init__(
        self,
        param_names: List[str],
        loss_to_param_map: Dict[str, str],
        variance_inflation_rate: float = 0.5,
        mean_shift_rate: float = 0.3,
        min_std: float = 0.05,
        max_std: float = 2.0
    ):
        """
        初始化分布更新器
        
        Args:
            param_names: 超参数名称列表
            loss_to_param_map: Loss名称到参数名称的映射
            variance_inflation_rate: 方差膨胀率 β
            mean_shift_rate: 均值偏移率
            min_std: 最小标准差
            max_std: 最大标准差
        """
        self.param_names = param_names
        self.loss_to_param_map = loss_to_param_map
        self.beta = variance_inflation_rate
        self.alpha = mean_shift_rate
        self.min_std = min_std
        self.max_std = max_std
        
        # 当前分布参数
        self.current_mean: Dict[str, float] = {}
        self.current_std: Dict[str, float] = {}
    
    def initialize(self, prior_distribution: Dict[str, Dict[str, float]]):
        """
        从先验初始化分布参数
        
        Args:
            prior_distribution: {param_name: {'mean': μ, 'std': σ}}
        """
        for param_name in self.param_names:
            if param_name in prior_distribution:
                self.current_mean[param_name] = prior_distribution[param_name]['mean']
                self.current_std[param_name] = prior_distribution[param_name]['std']
            else:
                self.current_mean[param_name] = 1.0
                self.current_std[param_name] = 0.3
    
    def update(
        self,
        exposure: Dict[str, float],
        surviving_configs: List[Dict],
        surviving_losses: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """
        基于冲突暴露度和存活配置更新分布
        
        Args:
            exposure: {loss_name: exposure_value}
            surviving_configs: 存活的配置列表
            surviving_losses: 存活配置的损失值
        
        Returns:
            更新后的分布 {param_name: {'mean': μ, 'std': σ}}
        """
        # 归一化暴露度
        total_exposure = sum(exposure.values())
        if total_exposure > 0:
            normalized_exposure = {k: v / total_exposure for k, v in exposure.items()}
        else:
            normalized_exposure = {k: 1.0 / len(exposure) for k in exposure}
        
        # 1. 基于冲突暴露度膨胀方差
        for loss_name, param_name in self.loss_to_param_map.items():
            if param_name in self.current_std:
                prex = loss_name.split('_')[0]
                n = 0
                exp = 0
                if prex =='family':
                    for i, j in normalized_exposure.items():
                        if prex in i:
                            exp += j
                            n += 1
                elif prex == 'person':
                    for i, j in normalized_exposure.items():
                        if prex in i:
                            exp += j
                            n += 1
                elif prex == 'graph':
                    for i, j in normalized_exposure.items():
                        if prex in i:
                            exp += j
                            n += 1
                else:
                    exp = normalized_exposure['mask_loss'] + normalized_exposure['total_member_loss']
                    n = 2
                exp = exp / n
                # σ_new = σ_old × (1 + β × E)
                inflation_factor = 1 + self.beta * exp * len(exposure)
                new_std = self.current_std[param_name] * inflation_factor
                self.current_std[param_name] = np.clip(new_std, self.min_std, self.max_std)
        
        # 2. 基于存活配置偏移均值（向好配置靠拢）
        if surviving_configs and surviving_losses:
            # 计算加权均值（损失越小权重越大）
            losses = np.array(surviving_losses)
            weights = np.exp(-losses / (losses.mean() + 1e-8))
            weights = weights / weights.sum()
            
            for param_name in self.param_names:
                if param_name in self.current_mean:
                    values = [c.get(param_name, self.current_mean[param_name]) 
                              for c in surviving_configs]
                    weighted_mean = np.sum(weights * np.array(values))
                    
                    # μ_new = (1-α) × μ_old + α × weighted_mean
                    self.current_mean[param_name] = (
                        (1 - self.alpha) * self.current_mean[param_name] +
                        self.alpha * weighted_mean
                    )
        
        # 返回更新后的分布
        return {
            param_name: {
                'mean': self.current_mean.get(param_name, 1.0),
                'std': self.current_std.get(param_name, 0.3)
            }
            for param_name in self.param_names
        }
    
    def sample(self, n_samples: int) -> List[Dict]:
        """
        从当前分布采样配置
        
        Args:
            n_samples: 采样数量
        
        Returns:
            配置列表
        """
        configs = []
        
        for _ in range(n_samples):
            config = {}
            for param_name in self.param_names:
                mean = self.current_mean.get(param_name, 1.0)
                std = self.current_std.get(param_name, 0.3)
                
                value = np.random.normal(mean, std)
                
                # 根据参数类型裁剪
                if param_name == 'rho':
                    value = np.clip(value, 0.5, 0.95)
                elif 'weight' in param_name:
                    value = np.clip(value, 0.1, 5.0)
                else:
                    value = max(0.01, value)
                
                config[param_name] = value
            
            configs.append(config)
        
        return configs
    
    def get_distribution_summary(self) -> str:
        """获取当前分布摘要"""
        lines = ["当前采样分布:"]
        for param_name in self.param_names:
            mean = self.current_mean.get(param_name, 0)
            std = self.current_std.get(param_name, 0)
            lines.append(f"  {param_name}: μ={mean:.4f}, σ={std:.4f}")
        return "\n".join(lines)

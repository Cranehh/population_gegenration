"""
自适应带宽核密度估计模块

根据参数重要性动态调整KDE带宽：
- 重要参数：使用较小带宽，保持精细分辨率
- 次要参数：使用较大带宽，增加平滑性

支持：
1. 自适应带宽计算
2. 基于重要性的采样
3. TPE风格的EI采样
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings

try:
    from scipy.stats import gaussian_kde, norm
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy未安装，部分功能不可用")


class AdaptiveBandwidthKDE:
    """
    基于参数重要性的自适应带宽核密度估计

    核心思想：
    - 重要参数使用较小带宽（高分辨率）
    - 不重要参数使用较大带宽（高平滑性）

    带宽调整公式：
        adjusted_bw = base_bw * exp(-alpha * (importance - baseline))

    Attributes:
        importance_scores: 参数重要性字典
        kdes: 各参数的KDE模型
        bandwidths: 各参数的带宽值
    """

    def __init__(
        self,
        importance_scores: Optional[Dict[str, float]] = None,
        base_bandwidth_factor: float = 1.0,
        importance_sensitivity: float = 2.0,
        min_bandwidth_ratio: float = 0.3,
        max_bandwidth_ratio: float = 3.0
    ):
        """
        初始化自适应带宽KDE

        Args:
            importance_scores: 参数重要性字典 {param_name: importance}
            base_bandwidth_factor: 基础带宽缩放因子
            importance_sensitivity: 重要性敏感度系数 (alpha)
            min_bandwidth_ratio: 最小带宽比例
            max_bandwidth_ratio: 最大带宽比例
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("请安装scipy: pip install scipy")

        self.importance_scores = importance_scores or {}
        self.base_factor = base_bandwidth_factor
        self.alpha = importance_sensitivity
        self.min_ratio = min_bandwidth_ratio
        self.max_ratio = max_bandwidth_ratio

        # 存储各参数的KDE和带宽
        self.kdes: Dict[str, gaussian_kde] = {}
        self.bandwidths: Dict[str, float] = {}
        self.data_ranges: Dict[str, Tuple[float, float]] = {}

    def update_importance(self, importance_scores: Dict[str, float]):
        """
        更新参数重要性分数

        Args:
            importance_scores: 新的重要性分数
        """
        self.importance_scores = importance_scores

    def compute_adaptive_bandwidth(
        self,
        param_name: str,
        data: np.ndarray,
        method: str = 'scott'
    ) -> float:
        """
        计算自适应带宽

        Args:
            param_name: 参数名
            data: 该参数的观测值
            method: 基础带宽估计方法 ('scott' 或 'silverman')

        Returns:
            调整后的带宽值
        """
        n = len(data)
        if n < 2:
            return 1.0

        std = np.std(data)
        if std == 0:
            std = 1.0

        # 基础带宽计算
        if method == 'scott':
            # Scott's Rule: h = n^(-1/5) * std
            base_bw = n ** (-1 / 5) * std
        elif method == 'silverman':
            # Silverman's Rule
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0:
                iqr = std
            base_bw = 0.9 * min(std, iqr / 1.34) * n ** (-1 / 5)
        else:
            base_bw = std * 0.5

        # 根据重要性调整带宽
        importance = self.importance_scores.get(param_name, 0.1)

        # 计算重要性的基线（所有参数的平均重要性）
        if self.importance_scores:
            baseline = np.mean(list(self.importance_scores.values()))
        else:
            baseline = 0.1

        # 带宽调整：重要性越高，带宽越小
        # adjusted_bw = base_bw * exp(-alpha * (importance - baseline))
        adjustment = np.exp(-self.alpha * (importance - baseline))
        adjusted_bw = base_bw * adjustment

        # 限制带宽范围
        min_bw = base_bw * self.min_ratio
        max_bw = base_bw * self.max_ratio
        adjusted_bw = np.clip(adjusted_bw, min_bw, max_bw)

        return adjusted_bw * self.base_factor

    def fit(
        self,
        param_name: str,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Optional['gaussian_kde']:
        """
        为指定参数拟合KDE

        Args:
            param_name: 参数名
            data: 观测数据
            weights: 样本权重（可选）

        Returns:
            拟合的KDE对象，或None（如果数据不足）
        """
        data = np.asarray(data).flatten()

        if len(data) < 3:
            return None

        # 记录数据范围
        self.data_ranges[param_name] = (data.min(), data.max())

        # 计算自适应带宽
        bw = self.compute_adaptive_bandwidth(param_name, data)
        self.bandwidths[param_name] = bw

        # 创建KDE
        try:
            # scipy的KDE带宽是相对于数据标准差的
            std = np.std(data)
            if std == 0:
                std = 1.0
            relative_bw = bw / std

            if weights is not None:
                # 加权KDE（scipy不直接支持，需要重采样）
                kde = gaussian_kde(data, bw_method=relative_bw)
            else:
                kde = gaussian_kde(data, bw_method=relative_bw)

            self.kdes[param_name] = kde
            return kde

        except Exception as e:
            warnings.warn(f"拟合KDE失败 ({param_name}): {e}")
            return None

    def fit_all(
        self,
        data_dict: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None
    ):
        """
        为所有参数拟合KDE

        Args:
            data_dict: {param_name: data_array}
            weights: 样本权重
        """
        for param_name, data in data_dict.items():
            self.fit(param_name, data, weights)

    def sample(
        self,
        param_name: str,
        n_samples: int = 1,
        bounds: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        从KDE中采样

        Args:
            param_name: 参数名
            n_samples: 采样数量
            bounds: 值域范围 (min, max)

        Returns:
            采样值数组
        """
        if param_name not in self.kdes:
            raise ValueError(f"参数 {param_name} 未拟合KDE")

        kde = self.kdes[param_name]

        # 采样（可能需要多次尝试以满足边界约束）
        samples = []
        max_attempts = n_samples * 10

        for _ in range(max_attempts):
            if len(samples) >= n_samples:
                break

            s = kde.resample(1)[0, 0]

            # 检查边界
            if bounds is not None:
                if bounds[0] <= s <= bounds[1]:
                    samples.append(s)
            else:
                samples.append(s)

        # 如果采样不足，使用边界反射
        while len(samples) < n_samples:
            s = kde.resample(1)[0, 0]
            if bounds is not None:
                s = np.clip(s, bounds[0], bounds[1])
            samples.append(s)

        return np.array(samples[:n_samples])

    def pdf(self, param_name: str, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        计算概率密度

        Args:
            param_name: 参数名
            x: 查询点

        Returns:
            概率密度值
        """
        if param_name not in self.kdes:
            raise ValueError(f"参数 {param_name} 未拟合KDE")

        x = np.atleast_1d(x)
        return self.kdes[param_name].pdf(x)

    def get_bandwidth(self, param_name: str) -> float:
        """获取参数的带宽"""
        return self.bandwidths.get(param_name, 1.0)

    def get_bandwidth_summary(self) -> Dict[str, Dict]:
        """
        获取带宽摘要信息

        Returns:
            {param_name: {'bandwidth': bw, 'importance': imp, 'adjustment': adj}}
        """
        summary = {}
        baseline = np.mean(list(self.importance_scores.values())) if self.importance_scores else 0.1

        for param_name, bw in self.bandwidths.items():
            imp = self.importance_scores.get(param_name, 0.1)
            adjustment = np.exp(-self.alpha * (imp - baseline))
            summary[param_name] = {
                'bandwidth': bw,
                'importance': imp,
                'adjustment_factor': adjustment
            }

        return summary


class ImportanceAwareTPE:
    """
    重要性感知的TPE采样器

    改进点：
    1. 根据参数重要性分配采样预算
    2. 对重要参数进行更精细的搜索
    3. 对不重要参数使用更宽松的采样
    4. 支持领域约束

    Attributes:
        configspace: 配置空间
        gamma: 好配置的比例阈值
        importance_scores: 参数重要性分数
        good_kde: 好配置的KDE
        bad_kde: 坏配置的KDE
    """

    def __init__(
        self,
        configspace,
        gamma: float = 0.15,
        n_candidates: int = 64,
        min_points_in_model: int = 15,
        random_fraction: float = 0.1
    ):
        """
        初始化TPE采样器

        Args:
            configspace: ConfigSpace配置空间
            gamma: 好配置的比例阈值（默认15%）
            n_candidates: 候选采样数
            min_points_in_model: 模型训练最小样本数
            random_fraction: 随机采样比例
        """
        self.cs = configspace
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.min_points = min_points_in_model
        self.random_fraction = random_fraction

        # 参数信息
        self.param_names = list(configspace.get_hyperparameter_names())
        self.param_bounds = self._get_param_bounds()

        # 观测数据
        self.observations: List[Tuple[Dict, float]] = []

        # 重要性分数
        self.importance_scores: Dict[str, float] = {}

        # KDE模型
        self.good_kde: Optional[AdaptiveBandwidthKDE] = None
        self.bad_kde: Optional[AdaptiveBandwidthKDE] = None

        # 领域约束
        self.constraints: List[callable] = []

    def _get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取各参数的边界"""
        bounds = {}
        for param_name in self.param_names:
            hp = self.cs.get_hyperparameter(param_name)
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                bounds[param_name] = (hp.lower, hp.upper)
            elif hasattr(hp, 'choices'):
                # 分类参数，使用索引作为边界
                bounds[param_name] = (0, len(hp.choices) - 1)
            else:
                bounds[param_name] = (0, 1)
        return bounds

    def add_constraint(self, constraint_fn: callable):
        """
        添加领域约束

        Args:
            constraint_fn: 约束函数，接受config字典，返回bool
        """
        self.constraints.append(constraint_fn)

    def check_constraints(self, config: Dict) -> bool:
        """检查配置是否满足所有约束"""
        for constraint in self.constraints:
            try:
                if not constraint(config):
                    return False
            except Exception:
                pass
        return True

    def update(self, config: Dict, loss: float):
        """
        添加新观测

        Args:
            config: 超参数配置
            loss: 损失值
        """
        self.observations.append((config, loss))

    def update_importance(self, importance_scores: Dict[str, float]):
        """
        更新参数重要性分数

        Args:
            importance_scores: {param_name: importance}
        """
        self.importance_scores = importance_scores

    def _build_kde_models(self):
        """构建好/坏配置的KDE模型"""
        if len(self.observations) < self.min_points:
            return False

        # 提取数据
        configs = [c for c, _ in self.observations]
        losses = np.array([l for _, l in self.observations])

        # 分割好/坏配置
        threshold = np.percentile(losses, self.gamma * 100)
        good_mask = losses <= threshold
        bad_mask = ~good_mask

        # 确保两组都有足够样本
        n_good = np.sum(good_mask)
        n_bad = np.sum(bad_mask)

        if n_good < 3 or n_bad < 3:
            return False

        # 转换为数组格式
        good_data = {}
        bad_data = {}

        for param_name in self.param_names:
            good_values = [configs[i].get(param_name, 0) for i in range(len(configs)) if good_mask[i]]
            bad_values = [configs[i].get(param_name, 0) for i in range(len(configs)) if bad_mask[i]]

            good_data[param_name] = np.array(good_values)
            bad_data[param_name] = np.array(bad_values)

        # 创建自适应带宽KDE
        self.good_kde = AdaptiveBandwidthKDE(self.importance_scores)
        self.bad_kde = AdaptiveBandwidthKDE(self.importance_scores)

        # 拟合KDE
        self.good_kde.fit_all(good_data)
        self.bad_kde.fit_all(bad_data)

        return True

    def sample(self) -> Dict:
        """
        采样新配置

        Returns:
            新的超参数配置字典
        """
        # 随机探索
        if np.random.random() < self.random_fraction or \
           len(self.observations) < self.min_points:
            return self._random_sample()

        # 构建KDE模型
        if not self._build_kde_models():
            return self._random_sample()

        # 生成候选配置并选择最佳
        best_config = None
        best_ei = -np.inf

        for _ in range(self.n_candidates):
            config = self._sample_from_good_kde()

            if not self.check_constraints(config):
                continue

            ei = self._compute_ei(config)

            if ei > best_ei:
                best_ei = ei
                best_config = config

        if best_config is None:
            return self._random_sample()

        return best_config

    def _random_sample(self) -> Dict:
        """随机采样配置"""
        for _ in range(100):
            config = self.cs.sample_configuration().get_dictionary()
            if self.check_constraints(config):
                return config
        return self.cs.sample_configuration().get_dictionary()

    def _sample_from_good_kde(self) -> Dict:
        """从好配置的KDE中采样，确保类型正确"""
        config = {}
        
        # 定义哪些参数必须是整数
        INT_PARAMS = {'batch_size', 'num_timesteps', 'hidden_dim', 'num_layers', 'num_heads'}

        for param_name in self.param_names:
            hp = self.cs.get_hyperparameter(param_name)
            bounds = self.param_bounds.get(param_name)

            if param_name in self.good_kde.kdes:
                value = self.good_kde.sample(param_name, 1, bounds)[0]

                if hasattr(hp, 'choices'):
                    # 分类参数
                    idx = int(np.clip(np.round(value), 0, len(hp.choices) - 1))
                    config[param_name] = hp.choices[idx]
                elif param_name in INT_PARAMS:
                    # 【关键】已知的整数参数
                    value = np.clip(value, hp.lower, hp.upper)
                    config[param_name] = int(np.round(value))
                elif hasattr(hp, 'lower'):
                    # 浮点数值参数
                    value = np.clip(value, hp.lower, hp.upper)
                    config[param_name] = float(value)
                else:
                    config[param_name] = float(value)
            else:
                config[param_name] = hp.sample(self.cs.random)

        return config

    def _compute_ei(self, config: Dict) -> float:
        """
        计算Expected Improvement (EI)

        EI ∝ l(x) / g(x)

        其中l(x)是好配置的密度，g(x)是坏配置的密度

        Args:
            config: 配置字典

        Returns:
            EI值
        """
        l_score = 1.0  # 好配置的密度
        g_score = 1.0  # 坏配置的密度

        for param_name in self.param_names:
            value = config.get(param_name)

            # 处理分类参数
            hp = self.cs.get_hyperparameter(param_name)
            if hasattr(hp, 'choices'):
                # 将分类值转换为索引
                if value in hp.choices:
                    value = hp.choices.index(value)
                else:
                    value = 0

            # 计算密度
            if param_name in self.good_kde.kdes:
                l_pdf = self.good_kde.pdf(param_name, value)
                l_score *= (l_pdf + 1e-10)

            if param_name in self.bad_kde.kdes:
                g_pdf = self.bad_kde.pdf(param_name, value)
                g_score *= (g_pdf + 1e-10)

        # EI = l(x) / g(x)
        return l_score / g_score

    def get_statistics(self) -> Dict:
        """
        获取采样器统计信息

        Returns:
            统计信息字典
        """
        if not self.observations:
            return {}

        losses = [l for _, l in self.observations]

        return {
            'n_observations': len(self.observations),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'threshold': np.percentile(losses, self.gamma * 100) if losses else None,
            'importance_scores': self.importance_scores.copy(),
            'bandwidths': self.good_kde.bandwidths if self.good_kde else {}
        }

    def get_best_config(self) -> Tuple[Dict, float]:
        """
        获取最佳配置

        Returns:
            (best_config, best_loss)
        """
        if not self.observations:
            return {}, float('inf')

        best_idx = np.argmin([l for _, l in self.observations])
        return self.observations[best_idx]


class MultiBandwidthKDE:
    """
    多带宽KDE采样器

    对每个参数使用不同的带宽策略，并支持联合采样
    """

    def __init__(self, param_names: List[str]):
        """
        初始化多带宽KDE

        Args:
            param_names: 参数名列表
        """
        self.param_names = param_names
        self.individual_kdes: Dict[str, AdaptiveBandwidthKDE] = {}
        self.joint_kde = None  # 可选的联合KDE

    def fit(
        self,
        data: np.ndarray,
        importance_scores: Dict[str, float],
        use_joint: bool = False
    ):
        """
        拟合多带宽KDE

        Args:
            data: 数据矩阵 (n_samples, n_params)
            importance_scores: 参数重要性
            use_joint: 是否使用联合KDE
        """
        # 为每个参数拟合独立KDE
        for i, param_name in enumerate(self.param_names):
            kde = AdaptiveBandwidthKDE(importance_scores)
            kde.fit(param_name, data[:, i])
            self.individual_kdes[param_name] = kde

        # 可选：拟合联合KDE
        if use_joint and len(data) >= 10:
            try:
                self.joint_kde = gaussian_kde(data.T)
            except Exception:
                self.joint_kde = None

    def sample(self, n_samples: int = 1, use_joint: bool = False) -> np.ndarray:
        """
        采样

        Args:
            n_samples: 采样数量
            use_joint: 是否使用联合采样

        Returns:
            采样矩阵 (n_samples, n_params)
        """
        if use_joint and self.joint_kde is not None:
            return self.joint_kde.resample(n_samples).T

        # 独立采样
        samples = np.zeros((n_samples, len(self.param_names)))
        for i, param_name in enumerate(self.param_names):
            if param_name in self.individual_kdes:
                kde = self.individual_kdes[param_name]
                if param_name in kde.kdes:
                    samples[:, i] = kde.sample(param_name, n_samples)

        return samples

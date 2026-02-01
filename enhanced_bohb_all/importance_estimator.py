"""
参数重要性估计模块

使用fANOVA (functional ANOVA) 方法估计超参数的重要性。
基于随机森林实现，支持：
1. 单参数主效应重要性
2. 参数对交互效应重要性
3. 重要性可视化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn未安装，部分功能不可用")


class fANOVAImportance:
    """
    基于随机森林的fANOVA参数重要性估计

    原理：
    - 使用随机森林拟合超参数-性能的映射
    - 通过特征重要性估计各参数的贡献
    - 支持不纯度重要性和置换重要性两种方法

    Attributes:
        rf: 随机森林模型
        param_names: 参数名列表
        importance_scores: 重要性分数字典
        is_fitted: 是否已拟合
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 3,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        初始化fANOVA重要性估计器

        Args:
            n_estimators: 随机森林中树的数量
            max_depth: 树的最大深度，None表示不限制
            min_samples_leaf: 叶节点最小样本数
            random_state: 随机种子
            n_jobs: 并行任务数，-1表示使用所有CPU
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("请安装sklearn: pip install scikit-learn")

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.param_names: List[str] = []
        self.importance_scores: Dict[str, float] = {}
        self.interaction_scores: Dict[Tuple[str, str], float] = {}
        self.is_fitted: bool = False

        # 数据预处理器
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.categorical_params: set = set()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_names: List[str],
        categorical_params: Optional[List[str]] = None
    ) -> 'fANOVAImportance':
        """
        拟合随机森林并计算重要性

        Args:
            X: 超参数配置矩阵, shape (n_samples, n_params)
            y: 目标值（损失）, shape (n_samples,)
            param_names: 参数名称列表
            categorical_params: 分类参数列表（可选）

        Returns:
            self
        """
        self.param_names = param_names
        self.categorical_params = set(categorical_params) if categorical_params else set()

        # 数据预处理
        X_processed = self._preprocess_features(X, fit=True)

        # 拟合随机森林
        self.rf.fit(X_processed, y)
        self.is_fitted = True

        # 计算基于不纯度的重要性
        self.importance_scores = self._compute_fanova_importance(X_processed, y)

        return self

    def _compute_fanova_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        基于方差分解计算参数重要性

        V(E[Y|X_i]) / V(Y) 表示参数 X_i 的主效应贡献
        """
        n_samples, n_params = X.shape
        total_variance = np.var(y)

        if total_variance == 0:
            return {p: 1.0 / n_params for p in self.param_names}

        importance_scores = {}

        for i, param_name in enumerate(self.param_names):
            # 对参数i的取值进行分组
            unique_values = np.unique(X[:, i])

            if len(unique_values) > 20:
                # 连续参数：分成20个bin
                percentiles = np.percentile(X[:, i], np.linspace(0, 100, 21))
                bins = np.digitize(X[:, i], percentiles[1:-1])
            else:
                # 离散参数：直接使用唯一值
                bins = np.searchsorted(unique_values, X[:, i])

            # 计算条件期望 E[Y|X_i = x]
            conditional_means = []
            bin_counts = []
            for b in np.unique(bins):
                mask = bins == b
                if np.sum(mask) > 0:
                    conditional_means.append(np.mean(y[mask]))
                    bin_counts.append(np.sum(mask))

            conditional_means = np.array(conditional_means)
            bin_counts = np.array(bin_counts)

            # 计算条件期望的方差 V(E[Y|X_i])
            # 使用加权方差，权重为各bin的样本比例
            weights = bin_counts / np.sum(bin_counts)
            weighted_mean = np.sum(weights * conditional_means)
            variance_of_conditional_mean = np.sum(weights * (conditional_means - weighted_mean) ** 2)

            # 重要性 = V(E[Y|X_i]) / V(Y)
            importance_scores[param_name] = variance_of_conditional_mean / total_variance

        # 归一化使总和为1
        total_imp = sum(importance_scores.values())
        if total_imp > 0:
            importance_scores = {k: v / total_imp for k, v in importance_scores.items()}

        return importance_scores

    def _preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        预处理特征

        Args:
            X: 原始特征矩阵
            fit: 是否拟合预处理器

        Returns:
            预处理后的特征矩阵
        """
        X_processed = X.copy().astype(float)

        # 对分类参数进行编码（如果需要）
        for i, param_name in enumerate(self.param_names):
            if param_name in self.categorical_params:
                if fit:
                    self.label_encoders[param_name] = LabelEncoder()
                    X_processed[:, i] = self.label_encoders[param_name].fit_transform(
                        X_processed[:, i].astype(str)
                    )
                elif param_name in self.label_encoders:
                    X_processed[:, i] = self.label_encoders[param_name].transform(
                        X_processed[:, i].astype(str)
                    )

        return X_processed

    def get_importance(self, method: str = 'fanova', X=None, y=None, n_repeats=10):
        """
        获取参数重要性

        Args:
            method: 'fanova' - 方差分解（默认）
                    'impurity' - 随机森林不纯度
                    'permutation' - 置换重要性
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit()")

        if method == 'fanova':
            return self.importance_scores.copy()
        elif method == 'impurity':
            return dict(zip(self.param_names, self.rf.feature_importances_))
        elif method == 'permutation':
            if X is None or y is None:
                raise ValueError("置换方法需要提供X和y")

            X_processed = self._preprocess_features(X, fit=False)
            perm_importance = permutation_importance(
                self.rf, X_processed, y,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )
            return dict(zip(self.param_names, perm_importance.importances_mean))

        else:
            raise ValueError(f"未知方法: {method}，支持 'impurity' 或 'permutation'")

    def get_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        获取按重要性排序的参数列表

        Returns:
            [(param_name, importance), ...] 按重要性降序排列
        """
        if not self.importance_scores:
            return []

        return sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def compute_interaction_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        top_k: int = 10,
        n_grid: int = 20
    ) -> Dict[Tuple[str, str], float]:
        """
        计算参数对的交互重要性

        使用H-statistic方法估计两个参数间的交互效应

        Args:
            X: 特征矩阵
            y: 目标值
            top_k: 返回前k个最强交互
            n_grid: 网格点数量

        Returns:
            交互重要性字典 {(param_i, param_j): interaction_score}
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit()")

        n_params = len(self.param_names)
        interactions = {}

        X_processed = self._preprocess_features(X, fit=False)

        # 计算所有参数对的交互
        for i in range(n_params):
            for j in range(i + 1, n_params):
                h_stat = self._compute_h_statistic(X_processed, y, i, j, n_grid)
                param_pair = (self.param_names[i], self.param_names[j])
                interactions[param_pair] = h_stat

        # 按交互强度排序
        sorted_interactions = sorted(
            interactions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.interaction_scores = dict(sorted_interactions[:top_k])
        return self.interaction_scores

    def _compute_h_statistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        i: int,
        j: int,
        n_grid: int = 20
    ) -> float:
        """
        计算两个参数间的H-statistic

        H^2_{ij} 衡量两个参数的联合效应与各自独立效应之差

        Args:
            X: 预处理后的特征矩阵
            y: 目标值
            i, j: 参数索引
            n_grid: 网格点数量

        Returns:
            H-statistic值
        """
        # 创建网格
        xi_values = np.linspace(X[:, i].min(), X[:, i].max(), n_grid)
        xj_values = np.linspace(X[:, j].min(), X[:, j].max(), n_grid)

        # 计算边际效应
        X_mean = X.mean(axis=0)
        f_0 = y.mean()

        f_i = np.zeros(n_grid)
        f_j = np.zeros(n_grid)
        f_ij = np.zeros((n_grid, n_grid))

        # f_i: 参数i的边际效应
        for ii, xi in enumerate(xi_values):
            X_temp = np.tile(X_mean, (len(X), 1))
            X_temp[:, i] = xi
            f_i[ii] = self.rf.predict(X_temp).mean()

        # f_j: 参数j的边际效应
        for jj, xj in enumerate(xj_values):
            X_temp = np.tile(X_mean, (len(X), 1))
            X_temp[:, j] = xj
            f_j[jj] = self.rf.predict(X_temp).mean()

        # f_ij: 联合效应
        for ii, xi in enumerate(xi_values):
            for jj, xj in enumerate(xj_values):
                X_temp = np.tile(X_mean, (len(X), 1))
                X_temp[:, i] = xi
                X_temp[:, j] = xj
                f_ij[ii, jj] = self.rf.predict(X_temp).mean()

        # 计算H-statistic
        numerator = 0.0
        for ii in range(n_grid):
            for jj in range(n_grid):
                interaction = f_ij[ii, jj] - f_i[ii] - f_j[jj] + f_0
                numerator += interaction ** 2

        denominator = np.sum((y - f_0) ** 2)

        if denominator == 0:
            return 0.0

        return numerator / (n_grid * n_grid) / (denominator / len(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合的随机森林预测性能

        Args:
            X: 特征矩阵

        Returns:
            预测的性能值
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合")

        X_processed = self._preprocess_features(X, fit=False)
        return self.rf.predict(X_processed)


class ParameterImportanceAnalyzer:
    """
    参数重要性分析器

    提供更高级的分析功能：
    1. 动态更新重要性估计
    2. 重要性变化追踪
    3. 置信区间估计
    """

    def __init__(
        self,
        configspace,
        update_frequency: int = 10,
        min_samples: int = 20,
        n_bootstrap: int = 10
    ):
        """
        初始化分析器

        Args:
            configspace: 配置空间
            update_frequency: 更新频率（每多少次观测更新一次）
            min_samples: 最小样本数
            n_bootstrap: Bootstrap采样次数（用于置信区间）
        """
        self.cs = configspace
        self.update_frequency = update_frequency
        self.min_samples = min_samples
        self.n_bootstrap = n_bootstrap

        self.param_names = list(configspace.get_hyperparameter_names())

        # 观测数据
        self.configs: List[Dict] = []
        self.losses: List[float] = []
        self.budgets: List[int] = []

        # 重要性估计器
        self.importance_estimator = fANOVAImportance()

        # 重要性历史
        self.importance_history: List[Dict[str, float]] = []
        self.importance_confidence: Dict[str, Tuple[float, float]] = {}

        # 当前重要性
        self.current_importance: Dict[str, float] = {}

    def add_observation(
        self,
        config: Dict,
        loss: float,
        budget: Optional[int] = None
    ):
        """
        添加观测

        Args:
            config: 超参数配置
            loss: 损失值
            budget: 使用的预算（可选）
        """
        self.configs.append(config)
        self.losses.append(loss)
        self.budgets.append(budget if budget else 1)

        # 检查是否需要更新
        if len(self.configs) >= self.min_samples and \
           len(self.configs) % self.update_frequency == 0:
            self._update_importance()

    def _update_importance(self):
        """更新重要性估计"""
        # 转换为数组
        X = self._configs_to_array()
        y = np.array(self.losses)

        # 如果有预算信息，对损失进行加权
        if any(b > 1 for b in self.budgets):
            weights = np.array(self.budgets) / max(self.budgets)
            # 高预算的观测更可信
            y_weighted = y  # 可以考虑加权

        # 拟合并计算重要性
        self.importance_estimator.fit(X, y, self.param_names)
        self.current_importance = self.importance_estimator.get_importance()

        # 记录历史
        self.importance_history.append(self.current_importance.copy())

        # Bootstrap置信区间
        self._compute_confidence_intervals(X, y)

    def _configs_to_array(self) -> np.ndarray:
        """将配置列表转换为数组"""
        X = []
        for config in self.configs:
            row = [config.get(p, 0) for p in self.param_names]
            X.append(row)
        return np.array(X)

    def _compute_confidence_intervals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05
    ):
        """
        使用Bootstrap计算重要性的置信区间

        Args:
            X: 特征矩阵
            y: 目标值
            alpha: 显著性水平
        """
        n_samples = len(y)
        bootstrap_importances = defaultdict(list)

        for _ in range(self.n_bootstrap):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # 拟合并计算重要性
            estimator = fANOVAImportance(n_estimators=50)
            estimator.fit(X_boot, y_boot, self.param_names)
            importance = estimator.get_importance()

            for param, score in importance.items():
                bootstrap_importances[param].append(score)

        # 计算置信区间
        for param in self.param_names:
            scores = bootstrap_importances[param]
            lower = np.percentile(scores, alpha / 2 * 100)
            upper = np.percentile(scores, (1 - alpha / 2) * 100)
            self.importance_confidence[param] = (lower, upper)

    def get_importance(self) -> Dict[str, float]:
        """获取当前重要性估计"""
        return self.current_importance.copy()

    def get_importance_with_confidence(self) -> Dict[str, Tuple[float, float, float]]:
        """
        获取带置信区间的重要性

        Returns:
            {param: (importance, lower, upper)}
        """
        result = {}
        for param in self.param_names:
            imp = self.current_importance.get(param, 0)
            lower, upper = self.importance_confidence.get(param, (0, 0))
            result[param] = (imp, lower, upper)
        return result

    def get_importance_trend(self, param_name: str) -> List[float]:
        """
        获取参数重要性的变化趋势

        Args:
            param_name: 参数名

        Returns:
            重要性值列表（按时间顺序）
        """
        return [h.get(param_name, 0) for h in self.importance_history]

    def get_top_params(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        获取最重要的k个参数

        Args:
            k: 返回数量

        Returns:
            [(param_name, importance), ...]
        """
        sorted_params = sorted(
            self.current_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_params[:k]

    def print_summary(self):
        """打印重要性摘要"""
        print("\n" + "=" * 60)
        print("参数重要性分析摘要")
        print("=" * 60)
        print(f"观测数量: {len(self.configs)}")
        print(f"更新次数: {len(self.importance_history)}")
        print("\n参数重要性排名:")
        print("-" * 40)

        for i, (param, imp) in enumerate(self.get_top_params(len(self.param_names)), 1):
            conf = self.importance_confidence.get(param, (0, 0))
            print(f"{i:2d}. {param:25s}: {imp:.4f} [{conf[0]:.4f}, {conf[1]:.4f}]")

        print("=" * 60)

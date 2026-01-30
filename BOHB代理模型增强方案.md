# BOHB代理模型增强方案：面向合成人口生成的定制化TPE

## 1. 背景与动机

### 1.1 标准TPE的局限性

BOHB中的标准TPE（Tree-structured Parzen Estimator）存在以下局限：

| 局限性 | 描述 | 对您问题的影响 |
|--------|------|----------------|
| **参数独立性假设** | 假设各超参数相互独立 | 忽略hidden_dim与num_heads的约束关系 |
| **固定带宽KDE** | 所有参数使用相同的带宽策略 | 敏感参数（lr）和不敏感参数（grad_clip）同等对待 |
| **无重要性感知** | 不区分参数重要性 | 在不重要的参数上浪费搜索预算 |
| **历史信息利用不足** | 仅使用当前观测 | 未利用参数间的交互模式 |

### 1.2 增强方案目标

```
┌─────────────────────────────────────────────────────────────────┐
│                    增强TPE的核心目标                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 参数重要性感知：自动识别关键超参数，集中搜索资源              │
│  2. 参数交互建模：捕获hidden_dim与num_heads等参数间的依赖         │
│  3. 自适应带宽：根据参数敏感性动态调整KDE带宽                     │
│  4. 多保真度信息融合：利用不同预算下的观测信息                    │
│  5. 领域知识注入：融入扩散模型训练的先验知识                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心技术：参数重要性估计

### 2.1 fANOVA（功能方差分解）原理

fANOVA将目标函数分解为各参数及其交互的贡献：

$$f(\mathbf{x}) = f_0 + \sum_i f_i(x_i) + \sum_{i<j} f_{ij}(x_i, x_j) + \cdots$$

其中：
- $f_0$：全局均值
- $f_i(x_i)$：参数$i$的主效应
- $f_{ij}(x_i, x_j)$：参数$i$和$j$的二阶交互效应

**参数重要性**定义为方差贡献比：

$$\text{Importance}(x_i) = \frac{\text{Var}[f_i(x_i)]}{\text{Var}[f(\mathbf{x})]}$$

### 2.2 基于随机森林的近似实现

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings

class fANOVAImportance:
    """
    基于随机森林的fANOVA参数重要性估计

    优点：
    - 天然处理非线性关系
    - 可捕获参数交互
    - 计算效率高
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.param_names = None

    def fit(self, X, y, param_names):
        """
        拟合随机森林并计算重要性

        Args:
            X: np.ndarray, shape (n_samples, n_params) - 超参数配置
            y: np.ndarray, shape (n_samples,) - 目标值（损失）
            param_names: List[str] - 参数名称列表
        """
        self.param_names = param_names
        self.rf.fit(X, y)
        self.is_fitted = True

    def get_importance(self, X=None, y=None, method='impurity'):
        """
        获取参数重要性

        Args:
            method: 'impurity' - 基于不纯度减少
                   'permutation' - 基于置换（更准确但更慢）

        Returns:
            dict: {param_name: importance_score}
        """
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit()")

        if method == 'impurity':
            # 基于树的不纯度减少（Gini importance）
            importance = self.rf.feature_importances_
        elif method == 'permutation':
            # 基于置换的重要性（更可靠）
            if X is None or y is None:
                raise ValueError("置换重要性需要提供X和y")
            perm_importance = permutation_importance(
                self.rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            importance = perm_importance.importances_mean
        else:
            raise ValueError(f"未知方法: {method}")

        return dict(zip(self.param_names, importance))

    def get_interaction_importance(self, X, y, top_k=5):
        """
        估计参数间的交互重要性

        使用H-statistic近似方法

        Returns:
            dict: {(param_i, param_j): interaction_score}
        """
        n_params = X.shape[1]
        interactions = {}

        # 计算所有参数对的交互
        for i in range(n_params):
            for j in range(i + 1, n_params):
                # H-statistic: 交互效应的方差占比
                h_stat = self._compute_h_statistic(X, y, i, j)
                interactions[(self.param_names[i], self.param_names[j])] = h_stat

        # 返回top-k交互
        sorted_interactions = sorted(
            interactions.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_interactions[:top_k])

    def _compute_h_statistic(self, X, y, i, j, n_grid=20):
        """
        计算两个参数间的H-statistic

        H^2_{ij} = sum[(f_ij - f_i - f_j + f_0)^2] / sum[(f - f_0)^2]
        """
        # 创建网格点
        xi_values = np.linspace(X[:, i].min(), X[:, i].max(), n_grid)
        xj_values = np.linspace(X[:, j].min(), X[:, j].max(), n_grid)

        # 计算部分依赖
        f_ij = np.zeros((n_grid, n_grid))
        f_i = np.zeros(n_grid)
        f_j = np.zeros(n_grid)

        X_temp = X.mean(axis=0, keepdims=True).repeat(len(X), axis=0)

        for ii, xi in enumerate(xi_values):
            X_temp[:, i] = xi
            f_i[ii] = self.rf.predict(X_temp).mean()

            for jj, xj in enumerate(xj_values):
                X_temp[:, j] = xj
                f_ij[ii, jj] = self.rf.predict(X_temp).mean()

        for jj, xj in enumerate(xj_values):
            X_temp = X.mean(axis=0, keepdims=True).repeat(len(X), axis=0)
            X_temp[:, j] = xj
            f_j[jj] = self.rf.predict(X_temp).mean()

        f_0 = y.mean()

        # 计算H-statistic
        numerator = 0
        for ii in range(n_grid):
            for jj in range(n_grid):
                interaction = f_ij[ii, jj] - f_i[ii] - f_j[jj] + f_0
                numerator += interaction ** 2

        denominator = ((y - f_0) ** 2).sum()

        if denominator == 0:
            return 0

        return numerator / denominator
```

### 2.3 针对您问题的预期重要性排序

基于扩散模型训练的先验知识，预期参数重要性排序：

```
高重要性 (预期 > 0.15):
├── lr (学习率) - 对收敛速度和最终性能影响最大
├── hidden_dim - 模型容量的核心决定因素
└── num_layers - 直接影响模型表达能力

中等重要性 (预期 0.05-0.15):
├── batch_size - 影响梯度估计质量
├── weight_decay - 正则化强度
├── rho (噪声相关系数) - 扩散过程的关键参数
└── num_heads - 注意力机制的并行度

低重要性 (预期 < 0.05):
├── grad_clip - 主要防止梯度爆炸
└── 各损失权重 - 在合理范围内影响较小
```

---

## 3. 自适应带宽KDE

### 3.1 标准KDE回顾

核密度估计：

$$\hat{p}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

其中$h$是带宽，$K$是核函数（通常为高斯核）。

**问题**：固定带宽无法适应不同参数的敏感性

### 3.2 重要性感知的自适应带宽

```python
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

class AdaptiveBandwidthKDE:
    """
    基于参数重要性的自适应带宽KDE

    核心思想：
    - 重要参数：使用较小带宽，保持精细分辨率
    - 不重要参数：使用较大带宽，增加平滑性
    """

    def __init__(self, importance_scores, base_bandwidth_factor=1.0):
        """
        Args:
            importance_scores: dict, {param_name: importance}
            base_bandwidth_factor: float, 基础带宽缩放因子
        """
        self.importance = importance_scores
        self.base_factor = base_bandwidth_factor
        self.kdes = {}  # 每个参数的KDE模型

    def compute_adaptive_bandwidth(self, param_name, data, method='scott'):
        """
        计算自适应带宽

        Args:
            param_name: 参数名
            data: 该参数的观测值
            method: 'scott' 或 'silverman' 基础带宽估计方法

        Returns:
            float: 调整后的带宽
        """
        n = len(data)
        std = np.std(data)

        # 基础带宽（Scott's rule）
        if method == 'scott':
            base_bw = n ** (-1/5) * std
        elif method == 'silverman':
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            base_bw = 0.9 * min(std, iqr/1.34) * n ** (-1/5)
        else:
            base_bw = std * 0.5

        # 根据重要性调整带宽
        importance = self.importance.get(param_name, 0.1)

        # 重要性越高，带宽越小（分辨率越高）
        # 使用指数映射：bw = base_bw * exp(-alpha * importance)
        alpha = 2.0  # 调节因子
        adjusted_bw = base_bw * np.exp(-alpha * (importance - 0.1))

        # 限制带宽范围
        min_bw = base_bw * 0.3
        max_bw = base_bw * 3.0
        adjusted_bw = np.clip(adjusted_bw, min_bw, max_bw)

        return adjusted_bw * self.base_factor

    def fit(self, param_name, data):
        """
        为指定参数拟合KDE
        """
        if len(data) < 3:
            return None

        bw = self.compute_adaptive_bandwidth(param_name, data)

        # 使用scipy的gaussian_kde，设置带宽
        kde = gaussian_kde(data, bw_method=bw / np.std(data))
        self.kdes[param_name] = kde

        return kde

    def sample(self, param_name, n_samples=1):
        """
        从KDE中采样
        """
        if param_name not in self.kdes:
            raise ValueError(f"参数 {param_name} 未拟合")

        return self.kdes[param_name].resample(n_samples).flatten()

    def pdf(self, param_name, x):
        """
        计算概率密度
        """
        if param_name not in self.kdes:
            raise ValueError(f"参数 {param_name} 未拟合")

        return self.kdes[param_name].pdf(x)


class ImportanceAwareTPE:
    """
    重要性感知的TPE采样器

    改进点：
    1. 根据参数重要性分配采样预算
    2. 对重要参数进行更精细的搜索
    3. 对不重要参数使用更宽松的采样
    """

    def __init__(self, configspace, gamma=0.15, n_candidates=64):
        """
        Args:
            configspace: 配置空间
            gamma: 好配置的比例阈值
            n_candidates: 候选采样数
        """
        self.cs = configspace
        self.gamma = gamma
        self.n_candidates = n_candidates

        self.observations = []
        self.importance_estimator = fANOVAImportance()
        self.importance_scores = {}
        self.good_kde = None
        self.bad_kde = None

    def update(self, config, loss):
        """
        添加新观测
        """
        self.observations.append((config, loss))

        # 当观测足够多时，更新重要性估计
        if len(self.observations) >= 20 and len(self.observations) % 5 == 0:
            self._update_importance()

    def _update_importance(self):
        """
        更新参数重要性估计
        """
        X, y = self._observations_to_array()
        param_names = list(self.cs.get_hyperparameter_names())

        self.importance_estimator.fit(X, y, param_names)
        self.importance_scores = self.importance_estimator.get_importance()

        print(f"参数重要性更新: {self.importance_scores}")

    def _observations_to_array(self):
        """
        将观测转换为数组
        """
        param_names = list(self.cs.get_hyperparameter_names())
        X = []
        y = []

        for config, loss in self.observations:
            x = [config[p] for p in param_names]
            X.append(x)
            y.append(loss)

        return np.array(X), np.array(y)

    def sample(self):
        """
        采样新配置

        Returns:
            dict: 新的超参数配置
        """
        if len(self.observations) < 10:
            # 观测不足，随机采样
            return self.cs.sample_configuration().get_dictionary()

        X, y = self._observations_to_array()

        # 分割好/坏配置
        threshold = np.percentile(y, self.gamma * 100)
        good_mask = y <= threshold
        bad_mask = ~good_mask

        X_good = X[good_mask]
        X_bad = X[bad_mask]

        # 使用自适应带宽KDE
        good_kde = AdaptiveBandwidthKDE(self.importance_scores)
        bad_kde = AdaptiveBandwidthKDE(self.importance_scores)

        param_names = list(self.cs.get_hyperparameter_names())

        # 为每个参数拟合KDE
        for i, param_name in enumerate(param_names):
            if len(X_good) >= 3:
                good_kde.fit(param_name, X_good[:, i])
            if len(X_bad) >= 3:
                bad_kde.fit(param_name, X_bad[:, i])

        # 采样候选配置
        candidates = []
        for _ in range(self.n_candidates):
            config = {}
            for i, param_name in enumerate(param_names):
                if param_name in good_kde.kdes:
                    config[param_name] = good_kde.sample(param_name, 1)[0]
                else:
                    # 随机采样
                    hp = self.cs.get_hyperparameter(param_name)
                    config[param_name] = hp.sample(self.cs.random)
            candidates.append(config)

        # 计算EI分数并选择最佳
        best_config = None
        best_score = -np.inf

        for config in candidates:
            score = self._compute_ei_score(config, good_kde, bad_kde, param_names)
            if score > best_score:
                best_score = score
                best_config = config

        return best_config

    def _compute_ei_score(self, config, good_kde, bad_kde, param_names):
        """
        计算Expected Improvement分数

        EI ∝ l(x) / g(x)
        """
        l_score = 1.0
        g_score = 1.0

        for param_name in param_names:
            x = config[param_name]

            if param_name in good_kde.kdes:
                l_score *= good_kde.pdf(param_name, x) + 1e-10
            if param_name in bad_kde.kdes:
                g_score *= bad_kde.pdf(param_name, x) + 1e-10

        return l_score / g_score
```

---

## 4. 参数交互建模

### 4.1 您问题中的关键参数交互

```
┌─────────────────────────────────────────────────────────────────┐
│                    参数交互关系图                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   hidden_dim ←──────→ num_heads                                │
│       │         约束: hidden_dim % num_heads == 0              │
│       │                                                         │
│       ↓                                                         │
│   num_layers ←──────→ lr                                       │
│       │         深网络需要更小学习率                             │
│       │                                                         │
│       ↓                                                         │
│   batch_size ←──────→ lr                                       │
│                 大batch可用更大学习率（线性缩放）                │
│                                                                 │
│   rho ←──────→ num_timesteps                                   │
│         噪声相关性与扩散步数相关                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 交互感知的联合采样

```python
import numpy as np
from scipy.stats import multivariate_normal

class InteractionAwareTPE:
    """
    考虑参数交互的TPE采样器

    核心改进：
    1. 识别强交互的参数对
    2. 对强交互参数使用联合分布建模
    3. 融入领域约束
    """

    def __init__(self, configspace, interaction_threshold=0.05):
        self.cs = configspace
        self.interaction_threshold = interaction_threshold
        self.param_names = list(configspace.get_hyperparameter_names())

        # 存储观测
        self.observations = []

        # 参数交互信息
        self.interactions = {}
        self.interaction_groups = []  # 强交互的参数组

        # 领域约束
        self.domain_constraints = self._define_domain_constraints()

    def _define_domain_constraints(self):
        """
        定义领域约束
        """
        constraints = []

        # 约束1: hidden_dim必须是num_heads的倍数
        def hidden_heads_constraint(config):
            if 'hidden_dim' in config and 'num_heads' in config:
                return config['hidden_dim'] % config['num_heads'] == 0
            return True
        constraints.append(hidden_heads_constraint)

        # 约束2: 深网络学习率不宜过大
        def depth_lr_constraint(config):
            if 'num_layers' in config and 'lr' in config:
                if config['num_layers'] > 24 and config['lr'] > 5e-4:
                    return False
            return True
        constraints.append(depth_lr_constraint)

        # 约束3: 大batch需要适当调大学习率
        def batch_lr_constraint(config):
            if 'batch_size' in config and 'lr' in config:
                # 线性缩放规则的软约束
                base_batch = 1024
                base_lr = 1e-4
                expected_lr = base_lr * (config['batch_size'] / base_batch)
                actual_lr = config['lr']
                # 允许2倍范围内的偏差
                if actual_lr < expected_lr * 0.3 or actual_lr > expected_lr * 3:
                    return False
            return True
        constraints.append(batch_lr_constraint)

        return constraints

    def _check_constraints(self, config):
        """
        检查配置是否满足所有约束
        """
        for constraint in self.domain_constraints:
            if not constraint(config):
                return False
        return True

    def _identify_interaction_groups(self):
        """
        识别强交互的参数组
        """
        if len(self.observations) < 30:
            return

        X, y = self._observations_to_array()

        # 计算交互重要性
        importance_estimator = fANOVAImportance()
        importance_estimator.fit(X, y, self.param_names)

        try:
            self.interactions = importance_estimator.get_interaction_importance(X, y, top_k=10)
        except Exception as e:
            print(f"交互分析失败: {e}")
            return

        # 识别强交互的参数对
        self.interaction_groups = []
        for (p1, p2), score in self.interactions.items():
            if score > self.interaction_threshold:
                self.interaction_groups.append((p1, p2))

        print(f"识别到的强交互参数对: {self.interaction_groups}")

    def _joint_sample(self, param_pair, X_good):
        """
        对强交互的参数对进行联合采样

        使用二维高斯KDE
        """
        p1, p2 = param_pair
        i1 = self.param_names.index(p1)
        i2 = self.param_names.index(p2)

        data = X_good[:, [i1, i2]]

        if len(data) < 5:
            return None, None

        # 使用二维高斯KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data.T)
            samples = kde.resample(1).T[0]
            return samples[0], samples[1]
        except Exception:
            return None, None

    def sample(self):
        """
        采样新配置，考虑参数交互
        """
        if len(self.observations) < 10:
            config = self.cs.sample_configuration().get_dictionary()
            # 确保满足约束
            for _ in range(100):
                if self._check_constraints(config):
                    return config
                config = self.cs.sample_configuration().get_dictionary()
            return config

        # 更新交互分析
        if len(self.observations) % 10 == 0:
            self._identify_interaction_groups()

        X, y = self._observations_to_array()
        threshold = np.percentile(y, 15)
        X_good = X[y <= threshold]

        # 生成候选配置
        best_config = None
        best_score = -np.inf

        for _ in range(64):
            config = {}
            sampled_params = set()

            # 1. 先对强交互的参数对进行联合采样
            for p1, p2 in self.interaction_groups:
                if p1 not in sampled_params and p2 not in sampled_params:
                    v1, v2 = self._joint_sample((p1, p2), X_good)
                    if v1 is not None:
                        config[p1] = v1
                        config[p2] = v2
                        sampled_params.add(p1)
                        sampled_params.add(p2)

            # 2. 对剩余参数独立采样
            for i, param_name in enumerate(self.param_names):
                if param_name not in sampled_params:
                    if len(X_good) >= 3:
                        # 从好配置的分布中采样
                        good_values = X_good[:, i]
                        kde = gaussian_kde(good_values)
                        config[param_name] = kde.resample(1)[0][0]
                    else:
                        hp = self.cs.get_hyperparameter(param_name)
                        config[param_name] = hp.sample(self.cs.random)

            # 3. 检查约束并修复
            config = self._repair_config(config)

            # 4. 评估配置质量
            score = self._evaluate_config(config, X_good, y)
            if score > best_score and self._check_constraints(config):
                best_score = score
                best_config = config

        return best_config if best_config else self.cs.sample_configuration().get_dictionary()

    def _repair_config(self, config):
        """
        修复不满足约束的配置
        """
        # 修复hidden_dim和num_heads的约束
        if 'hidden_dim' in config and 'num_heads' in config:
            hidden_dim = config['hidden_dim']
            num_heads = config['num_heads']

            # 找到最近的有效hidden_dim
            valid_dims = [d for d in [128, 192, 256, 320, 384, 448, 512]
                         if d % num_heads == 0]
            if valid_dims:
                config['hidden_dim'] = min(valid_dims, key=lambda x: abs(x - hidden_dim))

        return config

    def _evaluate_config(self, config, X_good, y):
        """
        评估配置的质量分数
        """
        # 基于与好配置的相似度
        config_vec = np.array([config.get(p, 0) for p in self.param_names])

        if len(X_good) == 0:
            return 0

        # 计算与好配置的平均距离（归一化）
        X_good_normalized = (X_good - X_good.mean(0)) / (X_good.std(0) + 1e-10)
        config_normalized = (config_vec - X_good.mean(0)) / (X_good.std(0) + 1e-10)

        distances = np.linalg.norm(X_good_normalized - config_normalized, axis=1)
        return -distances.mean()  # 距离越小，分数越高

    def _observations_to_array(self):
        X = []
        y = []
        for config, loss in self.observations:
            x = [config.get(p, 0) for p in self.param_names]
            X.append(x)
            y.append(loss)
        return np.array(X), np.array(y)

    def update(self, config, loss):
        self.observations.append((config, loss))
```

---

## 5. 多保真度信息融合

### 5.1 问题描述

BOHB使用不同预算（epoch数）评估配置，但标准实现中：
- 低预算观测和高预算观测同等对待
- 未考虑学习曲线信息

### 5.2 多保真度TPE

```python
import numpy as np
from collections import defaultdict

class MultiFidelityTPE:
    """
    多保真度TPE：融合不同预算下的观测信息

    核心思想：
    1. 对不同预算的观测赋予不同权重
    2. 利用学习曲线预测最终性能
    3. 根据预算调整好/坏配置的划分阈值
    """

    def __init__(self, configspace, min_budget=10, max_budget=200):
        self.cs = configspace
        self.min_budget = min_budget
        self.max_budget = max_budget

        # 按预算分组存储观测
        self.observations_by_budget = defaultdict(list)

        # 学习曲线模型
        self.curve_predictor = LearningCurvePredictor()

    def update(self, config, loss, budget):
        """
        添加观测

        Args:
            config: 超参数配置
            loss: 验证损失
            budget: 使用的预算（epoch数）
        """
        self.observations_by_budget[budget].append((config, loss))

        # 更新学习曲线模型
        config_key = self._config_to_key(config)
        self.curve_predictor.add_observation(config_key, budget, loss)

    def _config_to_key(self, config):
        """将配置转换为可哈希的key"""
        return tuple(sorted(config.items()))

    def sample(self, target_budget=None):
        """
        采样新配置

        Args:
            target_budget: 目标预算，用于调整采样策略
        """
        if target_budget is None:
            target_budget = self.max_budget

        # 收集所有观测并转换为估计的最终性能
        all_observations = []

        for budget, obs_list in self.observations_by_budget.items():
            for config, loss in obs_list:
                # 预测在max_budget下的性能
                predicted_loss = self.curve_predictor.predict(
                    self._config_to_key(config),
                    self.max_budget,
                    current_loss=loss,
                    current_budget=budget
                )

                # 计算置信度（预算越高，置信度越高）
                confidence = budget / self.max_budget

                all_observations.append({
                    'config': config,
                    'loss': predicted_loss,
                    'confidence': confidence,
                    'original_budget': budget
                })

        if len(all_observations) < 10:
            return self.cs.sample_configuration().get_dictionary()

        # 加权分割好/坏配置
        return self._weighted_sample(all_observations, target_budget)

    def _weighted_sample(self, observations, target_budget):
        """
        基于置信度加权的采样
        """
        # 根据预测损失和置信度排序
        # 高置信度的低损失配置排在前面
        scored_obs = []
        for obs in observations:
            score = obs['loss'] - 0.5 * obs['confidence']  # 低损失高置信度得分低
            scored_obs.append((score, obs))
        scored_obs.sort(key=lambda x: x[0])

        # 选择top 15%作为好配置
        n_good = max(3, int(len(scored_obs) * 0.15))
        good_configs = [obs['config'] for _, obs in scored_obs[:n_good]]
        bad_configs = [obs['config'] for _, obs in scored_obs[n_good:]]

        # 根据目标预算调整采样策略
        if target_budget < self.max_budget * 0.3:
            # 低预算：更注重探索
            exploration_ratio = 0.4
        else:
            # 高预算：更注重利用
            exploration_ratio = 0.1

        if np.random.random() < exploration_ratio:
            return self.cs.sample_configuration().get_dictionary()
        else:
            return self._sample_from_good(good_configs)

    def _sample_from_good(self, good_configs):
        """从好配置分布中采样"""
        if not good_configs:
            return self.cs.sample_configuration().get_dictionary()

        # 简化实现：随机选择一个好配置并添加噪声
        base_config = dict(good_configs[np.random.randint(len(good_configs))])

        # 对数值参数添加小扰动
        for param_name in base_config:
            hp = self.cs.get_hyperparameter(param_name)
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                range_size = hp.upper - hp.lower
                noise = np.random.normal(0, range_size * 0.1)
                base_config[param_name] = np.clip(
                    base_config[param_name] + noise,
                    hp.lower, hp.upper
                )

        return base_config


class LearningCurvePredictor:
    """
    学习曲线预测器

    预测模型在更多epoch后的性能
    """

    def __init__(self):
        self.curves = defaultdict(list)  # config_key -> [(budget, loss), ...]

    def add_observation(self, config_key, budget, loss):
        self.curves[config_key].append((budget, loss))
        # 按budget排序
        self.curves[config_key].sort(key=lambda x: x[0])

    def predict(self, config_key, target_budget, current_loss=None, current_budget=None):
        """
        预测在target_budget下的损失

        使用幂律外推：loss(t) = a * t^(-b) + c
        """
        curve = self.curves.get(config_key, [])

        if len(curve) >= 3:
            # 有足够观测，拟合幂律曲线
            return self._power_law_predict(curve, target_budget)
        elif current_loss is not None and current_budget is not None:
            # 使用简单外推
            return self._simple_extrapolate(current_loss, current_budget, target_budget)
        else:
            return current_loss if current_loss else float('inf')

    def _power_law_predict(self, curve, target_budget):
        """
        幂律曲线拟合和预测

        loss(t) = a * t^(-b) + c
        """
        from scipy.optimize import curve_fit

        budgets = np.array([b for b, _ in curve])
        losses = np.array([l for _, l in curve])

        def power_law(t, a, b, c):
            return a * np.power(t, -b) + c

        try:
            # 拟合参数
            popt, _ = curve_fit(
                power_law, budgets, losses,
                p0=[1, 0.5, losses[-1]],
                bounds=([0, 0, 0], [np.inf, 2, losses.min()]),
                maxfev=1000
            )
            return power_law(target_budget, *popt)
        except Exception:
            # 拟合失败，返回最后观测值
            return losses[-1]

    def _simple_extrapolate(self, current_loss, current_budget, target_budget):
        """
        简单外推（假设对数线性下降）
        """
        # 假设每翻倍预算，损失下降10%
        budget_ratio = target_budget / current_budget
        decay_factor = 0.9 ** np.log2(budget_ratio)
        return current_loss * decay_factor
```

---

## 6. 完整增强BOHB实现

### 6.1 整合所有组件

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Observation:
    """观测记录"""
    config: Dict
    loss: float
    budget: int
    info: Dict = None


class EnhancedBOHB:
    """
    增强版BOHB：融合所有改进组件

    改进点：
    1. 参数重要性感知
    2. 自适应带宽KDE
    3. 参数交互建模
    4. 多保真度信息融合
    5. 领域约束
    """

    def __init__(
        self,
        configspace,
        min_budget: int = 10,
        max_budget: int = 200,
        eta: int = 3,
        min_points_in_model: int = 20,
        importance_update_freq: int = 10,
        interaction_threshold: float = 0.05
    ):
        self.cs = configspace
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.min_points_in_model = min_points_in_model
        self.importance_update_freq = importance_update_freq
        self.interaction_threshold = interaction_threshold

        # 参数名列表
        self.param_names = list(configspace.get_hyperparameter_names())

        # 观测存储
        self.all_observations: List[Observation] = []
        self.observations_by_budget: Dict[int, List[Observation]] = defaultdict(list)

        # 组件初始化
        self.importance_estimator = fANOVAImportance()
        self.importance_scores: Dict[str, float] = {}
        self.interaction_groups: List[Tuple[str, str]] = []
        self.curve_predictor = LearningCurvePredictor()

        # 领域约束
        self.domain_constraints = self._define_constraints()

        # 计算Hyperband参数
        self._init_hyperband_params()

    def _init_hyperband_params(self):
        """初始化Hyperband参数"""
        self.s_max = int(np.log(self.max_budget / self.min_budget) / np.log(self.eta))
        self.budgets = [
            self.min_budget * (self.eta ** i)
            for i in range(self.s_max + 1)
        ]

    def _define_constraints(self):
        """定义领域约束"""
        constraints = []

        # hidden_dim % num_heads == 0
        def hidden_heads(c):
            if 'hidden_dim' in c and 'num_heads' in c:
                return c['hidden_dim'] % c['num_heads'] == 0
            return True
        constraints.append(hidden_heads)

        return constraints

    def _check_constraints(self, config):
        for constraint in self.domain_constraints:
            if not constraint(config):
                return False
        return True

    def _repair_config(self, config):
        """修复配置以满足约束"""
        if 'hidden_dim' in config and 'num_heads' in config:
            hd = config['hidden_dim']
            nh = config['num_heads']
            valid_dims = [d for d in [128, 192, 256, 320, 384, 448, 512] if d % nh == 0]
            if valid_dims:
                config['hidden_dim'] = min(valid_dims, key=lambda x: abs(x - hd))
        return config

    def update(self, config: Dict, loss: float, budget: int, info: Dict = None):
        """
        添加新观测
        """
        obs = Observation(config=config, loss=loss, budget=budget, info=info)
        self.all_observations.append(obs)
        self.observations_by_budget[budget].append(obs)

        # 更新学习曲线
        config_key = tuple(sorted(config.items()))
        self.curve_predictor.add_observation(config_key, budget, loss)

        # 定期更新重要性和交互
        if len(self.all_observations) % self.importance_update_freq == 0:
            self._update_importance_and_interactions()

    def _update_importance_and_interactions(self):
        """更新参数重要性和交互分析"""
        if len(self.all_observations) < self.min_points_in_model:
            return

        # 转换为数组
        X = np.array([
            [obs.config.get(p, 0) for p in self.param_names]
            for obs in self.all_observations
        ])

        # 使用预测的最终性能
        y = np.array([
            self.curve_predictor.predict(
                tuple(sorted(obs.config.items())),
                self.max_budget,
                obs.loss,
                obs.budget
            )
            for obs in self.all_observations
        ])

        # 更新重要性
        self.importance_estimator.fit(X, y, self.param_names)
        self.importance_scores = self.importance_estimator.get_importance()

        # 更新交互
        try:
            interactions = self.importance_estimator.get_interaction_importance(X, y, top_k=5)
            self.interaction_groups = [
                pair for pair, score in interactions.items()
                if score > self.interaction_threshold
            ]
        except Exception as e:
            print(f"交互分析失败: {e}")

        print(f"[EnhancedBOHB] 重要性更新: {self.importance_scores}")
        print(f"[EnhancedBOHB] 强交互参数对: {self.interaction_groups}")

    def sample(self, target_budget: Optional[int] = None) -> Dict:
        """
        采样新配置
        """
        if len(self.all_observations) < self.min_points_in_model:
            # 观测不足，随机采样
            for _ in range(100):
                config = self.cs.sample_configuration().get_dictionary()
                config = self._repair_config(config)
                if self._check_constraints(config):
                    return config
            return self.cs.sample_configuration().get_dictionary()

        # 准备数据
        X = np.array([
            [obs.config.get(p, 0) for p in self.param_names]
            for obs in self.all_observations
        ])

        y = np.array([
            self.curve_predictor.predict(
                tuple(sorted(obs.config.items())),
                self.max_budget,
                obs.loss,
                obs.budget
            )
            for obs in self.all_observations
        ])

        # 分割好/坏配置
        threshold = np.percentile(y, 15)
        good_mask = y <= threshold
        X_good = X[good_mask]
        X_bad = X[~good_mask]

        # 创建自适应带宽KDE
        good_kde = AdaptiveBandwidthKDE(self.importance_scores)
        bad_kde = AdaptiveBandwidthKDE(self.importance_scores)

        for i, pname in enumerate(self.param_names):
            if len(X_good) >= 3:
                good_kde.fit(pname, X_good[:, i])
            if len(X_bad) >= 3:
                bad_kde.fit(pname, X_bad[:, i])

        # 生成候选配置
        best_config = None
        best_score = -np.inf

        for _ in range(64):
            config = self._generate_candidate(X_good, good_kde)
            config = self._repair_config(config)

            if not self._check_constraints(config):
                continue

            score = self._compute_acquisition(config, good_kde, bad_kde)
            if score > best_score:
                best_score = score
                best_config = config

        if best_config is None:
            return self.cs.sample_configuration().get_dictionary()

        return best_config

    def _generate_candidate(self, X_good, good_kde) -> Dict:
        """生成候选配置"""
        config = {}
        sampled = set()

        # 1. 联合采样强交互参数
        for p1, p2 in self.interaction_groups:
            if p1 not in sampled and p2 not in sampled:
                i1 = self.param_names.index(p1)
                i2 = self.param_names.index(p2)
                if len(X_good) >= 5:
                    try:
                        from scipy.stats import gaussian_kde
                        joint_kde = gaussian_kde(X_good[:, [i1, i2]].T)
                        samples = joint_kde.resample(1).T[0]
                        config[p1] = samples[0]
                        config[p2] = samples[1]
                        sampled.add(p1)
                        sampled.add(p2)
                    except:
                        pass

        # 2. 独立采样其余参数
        for pname in self.param_names:
            if pname not in sampled:
                if pname in good_kde.kdes:
                    config[pname] = good_kde.sample(pname, 1)[0]
                else:
                    hp = self.cs.get_hyperparameter(pname)
                    if hasattr(hp, 'sample'):
                        config[pname] = hp.sample(self.cs.random)
                    else:
                        config[pname] = hp.default_value

        return config

    def _compute_acquisition(self, config, good_kde, bad_kde) -> float:
        """计算采集函数值 (EI)"""
        l_score = 1.0
        g_score = 1.0

        for pname in self.param_names:
            x = config.get(pname, 0)
            if pname in good_kde.kdes:
                l_score *= (good_kde.pdf(pname, x) + 1e-10)
            if pname in bad_kde.kdes:
                g_score *= (bad_kde.pdf(pname, x) + 1e-10)

        return l_score / g_score

    def get_best_config(self) -> Tuple[Dict, float]:
        """获取最佳配置"""
        if not self.all_observations:
            return None, float('inf')

        # 使用高预算观测
        high_budget_obs = [
            obs for obs in self.all_observations
            if obs.budget >= self.max_budget * 0.5
        ]

        if not high_budget_obs:
            high_budget_obs = self.all_observations

        best_obs = min(high_budget_obs, key=lambda x: x.loss)
        return best_obs.config, best_obs.loss

    def get_importance_ranking(self) -> List[Tuple[str, float]]:
        """获取参数重要性排名"""
        if not self.importance_scores:
            return []
        return sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
```

### 6.2 使用示例

```python
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)

# 定义搜索空间
cs = ConfigurationSpace()
cs.add_hyperparameter(CategoricalHyperparameter(
    'hidden_dim', choices=[128, 192, 256, 320, 384, 448, 512]
))
cs.add_hyperparameter(UniformIntegerHyperparameter(
    'num_layers', lower=12, upper=36
))
cs.add_hyperparameter(CategoricalHyperparameter(
    'num_heads', choices=[8, 16, 32]
))
cs.add_hyperparameter(UniformFloatHyperparameter(
    'lr', lower=1e-5, upper=1e-3, log=True
))
cs.add_hyperparameter(UniformFloatHyperparameter(
    'weight_decay', lower=1e-6, upper=1e-2, log=True
))
cs.add_hyperparameter(CategoricalHyperparameter(
    'batch_size', choices=[512, 1024, 2048]
))
cs.add_hyperparameter(UniformFloatHyperparameter(
    'rho', lower=0.7, upper=0.95
))

# 创建增强BOHB
optimizer = EnhancedBOHB(
    configspace=cs,
    min_budget=10,
    max_budget=200,
    eta=3,
    min_points_in_model=20
)

# 优化循环
for iteration in range(100):
    # 采样配置
    config = optimizer.sample()

    # 选择预算（按Hyperband调度）
    budget = optimizer.budgets[iteration % len(optimizer.budgets)]

    # 训练模型并获取损失
    loss = train_population_dit(config, epochs=budget)

    # 更新优化器
    optimizer.update(config, loss, budget)

    # 打印进度
    if iteration % 10 == 0:
        best_config, best_loss = optimizer.get_best_config()
        print(f"Iteration {iteration}: Best loss = {best_loss:.4f}")
        print(f"参数重要性: {optimizer.get_importance_ranking()[:5]}")

# 最终结果
best_config, best_loss = optimizer.get_best_config()
print(f"\n最佳配置: {best_config}")
print(f"最佳损失: {best_loss}")
print(f"\n参数重要性排名:")
for param, importance in optimizer.get_importance_ranking():
    print(f"  {param}: {importance:.4f}")
```

---

## 7. 实验验证建议

### 7.1 对比实验设计

```python
# 实验1: 标准BOHB vs 增强BOHB
experiments = {
    'standard_bohb': StandardBOHB(cs, min_budget=10, max_budget=200),
    'enhanced_bohb': EnhancedBOHB(cs, min_budget=10, max_budget=200),
    'random_search': RandomSearch(cs),
}

# 运行100次迭代，记录：
# 1. 最佳损失的收敛曲线
# 2. 达到目标损失所需的迭代次数
# 3. 总计算预算消耗

# 实验2: 消融实验
ablation_experiments = {
    'full': EnhancedBOHB(...),  # 完整版
    'no_importance': EnhancedBOHB(..., use_importance=False),
    'no_interaction': EnhancedBOHB(..., use_interaction=False),
    'no_multifidelity': EnhancedBOHB(..., use_multifidelity=False),
}
```

### 7.2 预期改进

| 指标 | 标准BOHB | 增强BOHB | 预期提升 |
|------|---------|---------|---------|
| 达到目标损失的迭代数 | 100 | 60-70 | 30-40% |
| 最终最佳损失 | 1.0 | 0.85-0.95 | 5-15% |
| 参数重要性识别准确率 | N/A | >80% | - |

---

## 8. 总结

### 8.1 核心改进点

1. **参数重要性感知**：使用fANOVA识别关键参数，集中搜索资源
2. **自适应带宽KDE**：重要参数用小带宽（高分辨率），次要参数用大带宽
3. **参数交互建模**：对强交互参数进行联合采样
4. **多保真度融合**：利用学习曲线预测最终性能
5. **领域约束**：融入hidden_dim/num_heads等约束

### 8.2 实现复杂度

| 组件 | 实现难度 | 收益 | 推荐优先级 |
|------|---------|------|-----------|
| 参数重要性 | 低 | 高 | ⭐⭐⭐ 最高 |
| 自适应带宽 | 中 | 中 | ⭐⭐ 高 |
| 交互建模 | 中 | 中 | ⭐⭐ 高 |
| 多保真度融合 | 高 | 中 | ⭐ 中 |
| 领域约束 | 低 | 高 | ⭐⭐⭐ 最高 |

### 8.3 快速开始建议

如果时间有限，建议按以下顺序实现：

1. **第一步**：添加领域约束（1-2小时）
2. **第二步**：添加参数重要性估计（2-3小时）
3. **第三步**：添加自适应带宽（2-3小时）
4. **第四步**：添加交互建模（可选，3-4小时）

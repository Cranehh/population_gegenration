# 合成人口生成模型超参数优化分析

## 1. 项目背景与问题特点

### 1.1 模型概述

您的项目是一个**合成人口生成模型**，基于Diffusion Transformer (DiT) 架构，具有以下特点：

- **多层级生成**：同时生成家庭级别和个人级别的数据
- **混合数据类型**：包含连续变量（年龄、收入等）和离散变量（性别、职业等）
- **图结构生成**：包含家庭关系图的邻接矩阵、节点类型、边类型
- **条件生成**：基于聚类信息和聚类配置文件进行条件生成

### 1.2 当前超参数空间

从 `train.py` 中识别的主要超参数：

| 类别 | 超参数 | 当前值 | 类型 |
|------|--------|--------|------|
| **模型架构** | hidden_dim | 320 | 离散/连续 |
| | num_layers | 30 | 离散 |
| | num_heads | 16 | 离散 |
| | proj_dim | 24 | 离散 |
| **训练参数** | lr | 1e-4 | 对数连续 |
| | weight_decay | 1e-4 | 对数连续 |
| | batch_size | 1024 | 离散 |
| | grad_clip | 1.0 | 连续 |
| **扩散参数** | num_timesteps | 200 | 离散 |
| | rho (噪声相关) | 0.85 | 连续 |
| **损失权重** | family_continuous_weight | 1.0 | 连续 |
| | person_age_weight | 2.0 | 连续 |
| | person_gender_weight | 2.0 | 连续 |
| | ... (共14个损失权重) | | |

### 1.3 问题特点分析

| 特点 | 描述 | 对HPO的影响 |
|------|------|------------|
| **高维搜索空间** | 20+个超参数 | 需要高效的搜索策略 |
| **训练成本高** | 500 epochs，大batch | 需要early stopping机制 |
| **多目标损失** | 14个子损失函数 | 可考虑多目标优化 |
| **参数敏感性差异** | 架构参数 vs 损失权重 | 分层搜索可能有效 |
| **离散+连续混合** | 层数(离散) vs 学习率(连续) | 需要处理混合空间 |

---

## 2. 超参数优化算法对比

### 2.1 主流算法概览

| 算法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **Grid Search** | 网格穷举 | 简单可靠 | 指数爆炸 | 低维空间 |
| **Random Search** | 随机采样 | 简单高效 | 无学习能力 | 中等维度 |
| **Bayesian Opt (BO)** | 高斯过程代理 | 样本效率高 | 计算开销大 | 低维、贵评估 |
| **TPE** | Tree Parzen Estimator | 处理条件好 | 并行困难 | 条件超参数 |
| **Hyperband** | 多保真度+早停 | 资源高效 | 无模型指导 | 资源受限 |
| **BOHB** | BO + Hyperband | 综合优势 | 实现复杂 | **您的场景** |
| **PBT** | 进化+热启动 | 动态调整 | 不确定收敛 | 强化学习 |

### 2.2 为什么推荐BOHB

对于您的合成人口生成问题，**BOHB (Bayesian Optimization and Hyperband)** 是最佳选择，原因如下：

1. **多保真度优化**：可以使用少量epoch快速淘汰差配置
2. **模型指导搜索**：TPE模型学习参数分布，不是纯随机
3. **处理混合空间**：天然支持连续+离散+条件参数
4. **并行友好**：支持异步并行评估
5. **鲁棒性强**：即使代理模型不准确也不会比Hyperband差

---

## 3. BOHB算法深度分析

### 3.1 算法原理

BOHB结合了两个核心组件：

#### 3.1.1 Hyperband组件

Hyperband使用**连续减半 (Successive Halving)** 策略：

```
输入：最大预算R, 缩减因子η (通常为3)
对于每个bracket s ∈ {s_max, s_max-1, ..., 0}:
    n = ⌈(s_max+1)/(s+1) × η^s⌉  # 初始配置数
    r = R × η^(-s)                # 初始预算

    对于每轮 i ∈ {0, 1, ..., s}:
        评估 n_i 个配置，每个使用预算 r_i
        保留 top 1/η 的配置
        n_{i+1} = ⌊n_i / η⌋
        r_{i+1} = r_i × η
```

**示例** (R=81 epochs, η=3):

| Bracket | 初始配置数 | 初始预算 | 轮次 |
|---------|-----------|---------|------|
| s=4 | 81 | 1 epoch | 5轮 |
| s=3 | 27 | 3 epochs | 4轮 |
| s=2 | 9 | 9 epochs | 3轮 |
| s=1 | 3 | 27 epochs | 2轮 |
| s=0 | 1 | 81 epochs | 1轮 |

#### 3.1.2 TPE模型指导

BOHB使用**核密度估计 (KDE)** 替代高斯过程：

```python
# 将观测分为好/坏两组
threshold = np.percentile(losses, 15)  # 默认15%分位
good_configs = configs[losses < threshold]
bad_configs = configs[losses >= threshold]

# 构建KDE模型
l(x) = KDE(good_configs)  # 好配置的密度
g(x) = KDE(bad_configs)   # 坏配置的密度

# 采样策略：最大化 l(x)/g(x)
# 实现：从l(x)采样，按l(x)/g(x)排序选择
```

### 3.2 BOHB工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                      BOHB主循环                              │
├─────────────────────────────────────────────────────────────┤
│  1. 选择当前bracket (循环遍历)                               │
│  2. 确定需要的配置数量n和初始预算r                            │
│  3. 配置采样：                                               │
│     ├─ 如果观测不足 → 随机采样                               │
│     └─ 如果观测充足 → TPE模型采样 (最大化l(x)/g(x))          │
│  4. 评估配置：以预算r训练模型，记录损失                       │
│  5. 连续减半：保留top-k配置，增加预算                         │
│  6. 更新TPE模型（所有观测数据）                              │
│  7. 重复直到预算耗尽                                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 关键超参数

| BOHB参数 | 含义 | 推荐值 | 对您问题的建议 |
|----------|------|--------|----------------|
| `min_budget` | 最小评估预算 | - | 5-10 epochs |
| `max_budget` | 最大评估预算 | - | 100-200 epochs |
| `eta` | 缩减因子 | 3 | 3 (平衡探索与利用) |
| `min_points_in_model` | 模型训练最小样本 | d+1 | 25-30 (参数数+1) |
| `top_n_percent` | 好配置百分比 | 15 | 15-20 |
| `num_samples` | TPE采样数 | 64 | 64-128 |
| `bandwidth_factor` | KDE带宽系数 | 3 | 3 |

---

## 4. 针对您问题的BOHB定制化方案

### 4.1 搜索空间设计

```python
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter
)

def get_configspace():
    cs = ConfigurationSpace()

    # ============ 模型架构参数 ============
    # hidden_dim: 必须是num_heads的倍数
    cs.add_hyperparameter(OrdinalHyperparameter(
        'hidden_dim',
        sequence=[128, 192, 256, 320, 384, 448, 512]
    ))

    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'num_layers', lower=12, upper=36, default_value=30
    ))

    cs.add_hyperparameter(CategoricalHyperparameter(
        'num_heads', choices=[8, 16, 32], default_value=16
    ))

    # ============ 训练参数 ============
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'lr', lower=1e-5, upper=1e-3, log=True, default_value=1e-4
    ))

    cs.add_hyperparameter(UniformFloatHyperparameter(
        'weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=1e-4
    ))

    cs.add_hyperparameter(CategoricalHyperparameter(
        'batch_size', choices=[512, 1024, 2048], default_value=1024
    ))

    cs.add_hyperparameter(UniformFloatHyperparameter(
        'grad_clip', lower=0.5, upper=2.0, default_value=1.0
    ))

    # ============ 扩散参数 ============
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'rho', lower=0.7, upper=0.95, default_value=0.85
    ))

    # ============ 损失权重 (分组优化) ============
    # 家庭级别权重
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'family_weight_scale', lower=0.5, upper=2.0, default_value=1.0
    ))

    # 人员级别权重
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'person_weight_scale', lower=0.5, upper=3.0, default_value=2.0
    ))

    # 图结构权重
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'graph_weight_scale', lower=0.2, upper=1.0, default_value=0.5
    ))

    # 约束损失权重
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'constraint_weight_scale', lower=0.5, upper=2.0, default_value=1.0
    ))

    return cs
```

### 4.2 多目标优化设计

您的问题有多个子损失，可以考虑**多目标BOHB**：

```python
def compute_objectives(loss_dict):
    """
    将14个子损失聚合为3-4个主要目标
    """
    objectives = {}

    # 目标1: 家庭属性重建质量
    objectives['family_quality'] = (
        loss_dict['family_continuous'].mean() +
        loss_dict['family_student'].mean()
    )

    # 目标2: 人员属性重建质量
    objectives['person_quality'] = (
        loss_dict['person_age'].mean() +
        loss_dict['person_gender'].mean() +
        loss_dict['person_license'].mean() +
        loss_dict['person_relation'].mean() +
        loss_dict['person_education'].mean() +
        loss_dict['person_occupation'].mean()
    ) / 6

    # 目标3: 图结构质量
    objectives['graph_quality'] = (
        loss_dict['graph_adj'].mean() +
        loss_dict['graph_node'].mean() +
        loss_dict['graph_edge'].mean()
    ) / 3

    # 目标4: 约束满足度
    objectives['constraint_satisfaction'] = (
        loss_dict['mask_loss'].mean() +
        loss_dict['total_member_loss'].mean() +
        loss_dict['unique_loss'].mean()
    )

    return objectives
```

### 4.3 自定义预算定义

对于扩散模型，可以使用**复合预算**：

```python
class DiffusionBudget:
    """
    复合预算策略：结合epoch数和时间步数
    """
    def __init__(self, epochs, timesteps_fraction=1.0):
        self.epochs = epochs
        self.timesteps_fraction = timesteps_fraction  # 使用部分扩散步骤

    @staticmethod
    def from_bohb_budget(budget, max_epochs=200):
        """
        将BOHB预算映射到训练配置

        budget范围: [min_budget, max_budget]

        策略:
        - 小预算: 少epoch + 部分timesteps
        - 大预算: 完整训练
        """
        if budget < 20:
            return DiffusionBudget(
                epochs=int(budget),
                timesteps_fraction=0.5  # 只用50%的扩散步骤
            )
        else:
            return DiffusionBudget(
                epochs=int(budget),
                timesteps_fraction=1.0
            )
```

### 4.4 早停准则定制

```python
class PopulationDiTEarlyStopping:
    """
    针对合成人口模型的早停策略
    """
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

        # 多指标监控
        self.loss_history = {
            'total': [],
            'family': [],
            'person': [],
            'graph': [],
            'constraint': []
        }

    def should_stop(self, loss_dict, epoch):
        """
        综合判断是否应该早停
        """
        # 主要指标：总损失
        total_loss = loss_dict['total_loss'].mean().item()

        # 计算各部分损失
        family_loss = (loss_dict['family_continuous'].mean() +
                      loss_dict['family_student'].mean()).item()
        person_loss = sum(loss_dict[k].mean().item()
                        for k in ['person_age', 'person_gender', 'person_license',
                                 'person_relation', 'person_education', 'person_occupation'])
        graph_loss = sum(loss_dict[k].mean().item()
                        for k in ['graph_adj', 'graph_node', 'graph_edge'])
        constraint_loss = sum(loss_dict[k].mean().item()
                             for k in ['mask_loss', 'total_member_loss', 'unique_loss'])

        # 记录历史
        self.loss_history['total'].append(total_loss)
        self.loss_history['family'].append(family_loss)
        self.loss_history['person'].append(person_loss)
        self.loss_history['graph'].append(graph_loss)
        self.loss_history['constraint'].append(constraint_loss)

        # 判断是否改进
        if total_loss < self.best_loss - self.min_delta:
            self.best_loss = total_loss
            self.counter = 0
            self.best_epoch = epoch
            return False

        self.counter += 1

        # 额外条件：如果约束损失持续上升，也早停
        if len(self.loss_history['constraint']) > 5:
            recent_constraint = self.loss_history['constraint'][-5:]
            if all(recent_constraint[i] < recent_constraint[i+1]
                   for i in range(len(recent_constraint)-1)):
                return True

        return self.counter >= self.patience
```

---

## 5. BOHB修改可行性分析

### 5.1 可行的修改方向

#### 5.1.1 目标函数定制 ✅ 高度可行

```python
class PopulationDiTObjective:
    """
    自定义目标函数，考虑生成质量的多个维度
    """
    def __init__(self, weights=None):
        self.weights = weights or {
            'reconstruction': 0.4,    # 重建损失
            'constraint': 0.3,        # 约束满足
            'distribution': 0.3       # 分布匹配
        }

    def __call__(self, loss_dict, generated_samples=None, real_samples=None):
        """
        综合目标函数
        """
        # 1. 重建损失（训练期间可用）
        reconstruction = loss_dict['total_loss'].mean()

        # 2. 约束损失
        constraint = (
            loss_dict['unique_loss'].mean() +
            loss_dict['mask_loss'].mean() +
            loss_dict['total_member_loss'].mean()
        )

        # 3. 分布匹配（需要生成样本）
        if generated_samples is not None and real_samples is not None:
            distribution = self.compute_distribution_distance(
                generated_samples, real_samples
            )
        else:
            distribution = 0

        return (
            self.weights['reconstruction'] * reconstruction +
            self.weights['constraint'] * constraint +
            self.weights['distribution'] * distribution
        )

    def compute_distribution_distance(self, gen, real):
        """
        计算生成样本与真实样本的分布距离
        可使用MMD、Wasserstein距离等
        """
        # 示例：简单的均值方差距离
        mean_dist = torch.norm(gen.mean(0) - real.mean(0))
        std_dist = torch.norm(gen.std(0) - real.std(0))
        return mean_dist + std_dist
```

#### 5.1.2 搜索空间结构化 ✅ 高度可行

```python
from ConfigSpace.conditions import InCondition, EqualsCondition

def get_hierarchical_configspace():
    """
    层次化搜索空间：先搜索架构，再搜索训练参数
    """
    cs = ConfigurationSpace()

    # 第一层：架构搜索
    model_scale = CategoricalHyperparameter(
        'model_scale', choices=['small', 'medium', 'large']
    )
    cs.add_hyperparameter(model_scale)

    # 条件参数：根据模型规模确定具体配置
    hidden_dim_small = OrdinalHyperparameter(
        'hidden_dim_small', sequence=[128, 192, 256]
    )
    hidden_dim_medium = OrdinalHyperparameter(
        'hidden_dim_medium', sequence=[256, 320, 384]
    )
    hidden_dim_large = OrdinalHyperparameter(
        'hidden_dim_large', sequence=[384, 448, 512]
    )

    cs.add_hyperparameters([hidden_dim_small, hidden_dim_medium, hidden_dim_large])

    # 添加条件
    cs.add_condition(EqualsCondition(hidden_dim_small, model_scale, 'small'))
    cs.add_condition(EqualsCondition(hidden_dim_medium, model_scale, 'medium'))
    cs.add_condition(EqualsCondition(hidden_dim_large, model_scale, 'large'))

    return cs
```

#### 5.1.3 多保真度策略定制 ✅ 可行

```python
class AdaptiveBudgetAllocation:
    """
    自适应预算分配：根据配置特点调整评估预算
    """
    def __init__(self):
        self.config_performance = {}  # 记录配置性能

    def get_budget(self, config, base_budget):
        """
        根据配置特点调整预算

        - 大模型：可能需要更多预算才能收敛
        - 小学习率：需要更多epoch
        - 高batch_size：每epoch信息量更大
        """
        multiplier = 1.0

        # 大模型需要更多预算
        if config['hidden_dim'] >= 384:
            multiplier *= 1.2

        # 小学习率需要更多epoch
        if config['lr'] < 5e-5:
            multiplier *= 1.3

        # 大batch可以用更少epoch
        if config['batch_size'] >= 2048:
            multiplier *= 0.8

        return int(base_budget * multiplier)
```

#### 5.1.4 代理模型增强 ⚠️ 中等可行

```python
class EnhancedTPE:
    """
    增强的TPE模型，考虑参数间的相互作用
    """
    def __init__(self, configspace, min_points=30):
        self.cs = configspace
        self.min_points = min_points
        self.observations = []

        # 添加参数重要性学习
        self.param_importance = {}

    def fit(self, configs, losses):
        """
        拟合TPE模型，同时学习参数重要性
        """
        self.observations.extend(zip(configs, losses))

        if len(self.observations) >= self.min_points:
            # 使用fANOVA估计参数重要性
            self.param_importance = self._compute_importance()

            # 根据重要性调整KDE带宽
            self._adjust_bandwidth()

    def _compute_importance(self):
        """
        使用功能方差分解估计参数重要性
        """
        # 简化实现：使用随机森林特征重要性
        from sklearn.ensemble import RandomForestRegressor

        X = np.array([self._config_to_vector(c) for c, _ in self.observations])
        y = np.array([l for _, l in self.observations])

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)

        importance = dict(zip(self.cs.get_hyperparameter_names(),
                             rf.feature_importances_))
        return importance
```

### 5.2 不建议修改的部分

| 组件 | 原因 |
|------|------|
| Successive Halving逻辑 | 理论保证的核心，修改可能破坏收敛性 |
| bracket循环策略 | 平衡探索与利用的关键 |
| 基本KDE采样 | 经过大量验证，修改风险高 |

---

## 6. 实现方案

### 6.1 使用HEBO库实现（推荐）

```python
# pip install hebo

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

# 定义搜索空间
space = DesignSpace().parse([
    {'name': 'hidden_dim', 'type': 'int', 'lb': 128, 'ub': 512},
    {'name': 'num_layers', 'type': 'int', 'lb': 12, 'ub': 36},
    {'name': 'num_heads', 'type': 'cat', 'categories': [8, 16, 32]},
    {'name': 'lr', 'type': 'pow', 'lb': 1e-5, 'ub': 1e-3},
    {'name': 'weight_decay', 'type': 'pow', 'lb': 1e-6, 'ub': 1e-2},
    {'name': 'batch_size', 'type': 'cat', 'categories': [512, 1024, 2048]},
    {'name': 'rho', 'type': 'num', 'lb': 0.7, 'ub': 0.95},
])

opt = HEBO(space)

for i in range(100):
    rec = opt.suggest(n_suggestions=1)
    # 训练模型并获取损失
    loss = train_and_evaluate(rec)
    opt.observe(rec, loss)
```

### 6.2 使用HpBandSter库实现BOHB

```python
# pip install hpbandster

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

class PopulationDiTWorker(Worker):
    def __init__(self, *args, data_dir='数据', **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def compute(self, config, budget, **kwargs):
        """
        训练模型并返回验证损失

        Args:
            config: 超参数配置
            budget: 训练epochs数
        """
        # 创建模型
        model = PopulationDiT(
            hidden_size=config['hidden_dim'],
            depth=config['num_layers'],
            num_heads=config['num_heads']
        ).cuda()

        # 训练
        loss = train_model(
            model=model,
            epochs=int(budget),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            batch_size=config['batch_size'],
            rho=config['rho'],
            data_dir=self.data_dir
        )

        return {
            'loss': loss,
            'info': {
                'epochs': int(budget),
                'config': config
            }
        }

    @staticmethod
    def get_configspace():
        cs = ConfigurationSpace()
        # ... 添加超参数 (见4.1节)
        return cs

# 运行BOHB
NS = hpns.NameServer(run_id='population_dit', host='127.0.0.1', port=None)
NS.start()

worker = PopulationDiTWorker(nameserver='127.0.0.1', run_id='population_dit')
worker.run(background=True)

bohb = BOHB(
    configspace=worker.get_configspace(),
    run_id='population_dit',
    min_budget=10,      # 最少10个epoch
    max_budget=200,     # 最多200个epoch
    eta=3,              # 缩减因子
    nameserver='127.0.0.1'
)

res = bohb.run(n_iterations=100)

# 获取最佳配置
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
best_config = id2config[incumbent]['config']
print(f"Best config: {best_config}")
```

### 6.3 使用Ray Tune实现（推荐用于分布式）

```python
# pip install "ray[tune]"

from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

def train_population_dit(config):
    """Ray Tune训练函数"""
    model = PopulationDiT(
        hidden_size=config['hidden_dim'],
        depth=config['num_layers'],
        num_heads=config['num_heads']
    ).cuda()

    for epoch in range(config['max_epochs']):
        loss = train_one_epoch(model, config)

        # 报告中间结果
        tune.report(loss=loss, epoch=epoch)

# 配置搜索空间
config_space = {
    'hidden_dim': tune.choice([128, 192, 256, 320, 384, 448, 512]),
    'num_layers': tune.randint(12, 37),
    'num_heads': tune.choice([8, 16, 32]),
    'lr': tune.loguniform(1e-5, 1e-3),
    'weight_decay': tune.loguniform(1e-6, 1e-2),
    'batch_size': tune.choice([512, 1024, 2048]),
    'rho': tune.uniform(0.7, 0.95),
    'max_epochs': 200
}

# BOHB调度器
scheduler = HyperBandForBOHB(
    time_attr='epoch',
    max_t=200,
    reduction_factor=3
)

# BOHB搜索算法
search_alg = TuneBOHB(metric='loss', mode='min')

# 运行优化
analysis = tune.run(
    train_population_dit,
    config=config_space,
    scheduler=scheduler,
    search_alg=search_alg,
    num_samples=100,
    resources_per_trial={'gpu': 1}
)

print("Best config:", analysis.best_config)
```

---

## 7. 实验建议

### 7.1 分阶段优化策略

```
阶段1: 架构搜索 (20-30次试验)
├── 固定: 损失权重=默认, lr=1e-4
├── 搜索: hidden_dim, num_layers, num_heads
└── 预算: 10-50 epochs

阶段2: 训练参数搜索 (30-50次试验)
├── 固定: 最佳架构
├── 搜索: lr, weight_decay, batch_size, grad_clip, rho
└── 预算: 20-100 epochs

阶段3: 损失权重调优 (20-30次试验)
├── 固定: 最佳架构+训练参数
├── 搜索: 各损失权重
└── 预算: 50-200 epochs

阶段4: 联合微调 (10-20次试验)
├── 在最佳配置附近微调
└── 预算: 完整训练
```

### 7.2 预算分配建议

| 总GPU小时 | min_budget | max_budget | n_iterations | 预期效果 |
|-----------|------------|------------|--------------|----------|
| 100 | 5 | 50 | 50 | 初步筛选 |
| 500 | 10 | 100 | 100 | 良好优化 |
| 1000+ | 10 | 200 | 150+ | 充分优化 |

### 7.3 监控指标

```python
# 建议监控的指标
metrics_to_track = {
    # 主要指标
    'total_loss': 'minimize',

    # 子任务指标
    'family_reconstruction': 'minimize',
    'person_reconstruction': 'minimize',
    'graph_reconstruction': 'minimize',

    # 约束指标
    'unique_constraint_violation': 'minimize',
    'member_count_error': 'minimize',

    # 生成质量指标（在完整评估时）
    'age_distribution_wasserstein': 'minimize',
    'relation_type_accuracy': 'maximize',
    'family_structure_validity': 'maximize'
}
```

---

## 8. 总结

### 8.1 推荐方案

对于您的合成人口生成问题，推荐采用**BOHB算法**，配合以下定制：

1. **多目标聚合**：将14个子损失聚合为4个主要目标
2. **分层搜索空间**：架构参数→训练参数→损失权重
3. **自定义早停**：结合总损失和约束满足度
4. **分阶段优化**：先架构后训练参数

### 8.2 修改可行性总结

| 修改方向 | 可行性 | 建议 |
|----------|--------|------|
| 目标函数定制 | ✅ 高 | 强烈建议，考虑多目标聚合 |
| 搜索空间结构化 | ✅ 高 | 建议使用条件参数 |
| 预算策略定制 | ✅ 高 | 可考虑复合预算 |
| 代理模型增强 | ⚠️ 中 | 可尝试添加参数重要性 |
| 核心算法修改 | ❌ 低 | 不建议修改SH和bracket逻辑 |

### 8.3 预期收益

- **搜索效率提升**：相比随机搜索，样本效率提升3-10倍
- **计算资源节省**：通过早停节省50-70%的计算资源
- **更好的超参数配置**：预期损失降低10-30%

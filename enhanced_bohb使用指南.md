# Enhanced BOHB 使用指南

## 1. 概述

本模块实现了增强版BOHB超参数优化，专门针对您的PopulationDiT合成人口生成模型进行了定制。

### 核心特性

| 特性 | 说明 |
|------|------|
| **参数重要性估计** | 使用fANOVA自动识别关键超参数，优先搜索重要参数 |
| **自适应带宽KDE** | 重要参数使用小带宽（高分辨率），次要参数使用大带宽 |
| **领域约束** | 自动满足`hidden_dim % num_heads == 0`等约束 |
| **Successive Halving** | 使用少量epoch快速淘汰差配置，节省计算资源 |

---

## 2. 安装依赖

```bash
# 必需依赖
pip install numpy scipy scikit-learn ConfigSpace

# 可选：用于可视化
pip install matplotlib seaborn
```

---

## 3. 文件结构

```
enhanced_bohb/
├── __init__.py                 # 模块入口
├── importance_estimator.py     # 参数重要性估计
├── adaptive_kde.py             # 自适应带宽KDE
├── config_space.py             # 配置空间定义
├── enhanced_bohb.py            # 主优化器
└── worker.py                   # 训练Worker

run_enhanced_bohb.py            # 主运行脚本
```

---

## 4. 快速开始

### 4.1 命令行运行（推荐）

```bash
# 基本运行
python run_enhanced_bohb.py --n_iterations 100

# 指定预算范围
python run_enhanced_bohb.py --n_iterations 100 --min_budget 10 --max_budget 100

# 使用简化版（不使用Successive Halving）
python run_enhanced_bohb.py --n_iterations 50 --simplified

# 包含损失权重搜索
python run_enhanced_bohb.py --n_iterations 100 --include_loss_weights

# 完整参数
python run_enhanced_bohb.py \
    --data_dir 数据 \
    --result_dir bohb_results \
    --n_iterations 100 \
    --min_budget 10 \
    --max_budget 100 \
    --eta 3 \
    --min_points 20 \
    --gamma 0.15
```

### 4.2 Python脚本运行

```python
from enhanced_bohb import (
    EnhancedBOHBOptimizer,
    create_population_dit_configspace,
    PopulationDiTWorker
)

# 1. 创建配置空间
configspace = create_population_dit_configspace(
    include_loss_weights=False,  # 是否包含损失权重
    simplified=False              # 是否使用简化空间
)

# 2. 创建训练Worker
worker = PopulationDiTWorker(
    data_dir='数据',
    device='cuda',
    validation_split=0.1,
    early_stopping_patience=10
)

# 3. 创建优化器
optimizer = EnhancedBOHBOptimizer(
    configspace=configspace,
    min_budget=10,          # 最小epoch
    max_budget=100,         # 最大epoch
    eta=3,                  # 缩减因子
    min_points_in_model=20, # TPE最小样本数
    result_dir='bohb_results'
)

# 4. 定义评估函数
def evaluate_fn(config, budget):
    return worker.evaluate(config, budget)

# 5. 运行优化
best_config, best_loss = optimizer.optimize(
    evaluate_fn=evaluate_fn,
    n_iterations=100,
    verbose=True
)

# 6. 查看结果
print(f"最佳损失: {best_loss}")
print(f"最佳配置: {best_config}")
print(f"参数重要性: {optimizer.get_importance_ranking()}")
```

---

## 5. 核心组件详解

### 5.1 参数重要性估计

```python
from enhanced_bohb import fANOVAImportance, ParameterImportanceAnalyzer

# 方式1：直接使用fANOVA
importance_estimator = fANOVAImportance(n_estimators=100)

# 准备数据（从已有观测）
X = np.array([...])  # (n_samples, n_params)
y = np.array([...])  # (n_samples,) 损失值
param_names = ['hidden_dim', 'num_layers', 'lr', ...]

# 拟合
importance_estimator.fit(X, y, param_names)

# 获取重要性
importance = importance_estimator.get_importance()
# {'hidden_dim': 0.25, 'lr': 0.35, 'num_layers': 0.15, ...}

# 获取排名
ranking = importance_estimator.get_importance_ranking()
# [('lr', 0.35), ('hidden_dim', 0.25), ('num_layers', 0.15), ...]

# 方式2：使用分析器（自动更新）
analyzer = ParameterImportanceAnalyzer(
    configspace=configspace,
    update_frequency=10,  # 每10次观测更新一次
    min_samples=20
)

# 添加观测（会自动更新重要性）
for config, loss in observations:
    analyzer.add_observation(config, loss)

# 获取带置信区间的重要性
importance_with_ci = analyzer.get_importance_with_confidence()
# {'lr': (0.35, 0.30, 0.40), ...}  # (mean, lower, upper)

# 打印摘要
analyzer.print_summary()
```

**输出示例：**
```
============================================================
参数重要性分析摘要
============================================================
观测数量: 100
更新次数: 10

参数重要性排名:
----------------------------------------
 1. lr                       : 0.3521 [0.3012, 0.4030]
 2. hidden_dim               : 0.2456 [0.1987, 0.2925]
 3. num_layers               : 0.1523 [0.1102, 0.1944]
 4. batch_size               : 0.0987 [0.0654, 0.1320]
 5. weight_decay             : 0.0756 [0.0423, 0.1089]
 ...
============================================================
```

### 5.2 自适应带宽KDE

```python
from enhanced_bohb import AdaptiveBandwidthKDE, ImportanceAwareTPE

# 使用自适应带宽KDE
importance_scores = {'lr': 0.35, 'hidden_dim': 0.25, 'num_layers': 0.15}

kde = AdaptiveBandwidthKDE(
    importance_scores=importance_scores,
    base_bandwidth_factor=1.0,
    importance_sensitivity=2.0,  # alpha参数
    min_bandwidth_ratio=0.3,
    max_bandwidth_ratio=3.0
)

# 为各参数拟合KDE
data = {'lr': np.array([...]), 'hidden_dim': np.array([...])}
kde.fit_all(data)

# 查看带宽
print(kde.get_bandwidth_summary())
# lr: 带宽较小（重要参数）
# hidden_dim: 带宽中等
# num_layers: 带宽较大

# 从KDE采样
samples = kde.sample('lr', n_samples=10, bounds=(1e-5, 1e-3))
```

**带宽计算原理：**

```
adjusted_bw = base_bw * exp(-alpha * (importance - baseline))

- 重要性高 → 带宽小 → 分辨率高 → 精细搜索
- 重要性低 → 带宽大 → 平滑性高 → 粗略搜索
```

### 5.3 使用TPE采样器

```python
from enhanced_bohb import ImportanceAwareTPE

# 创建TPE采样器
tpe = ImportanceAwareTPE(
    configspace=configspace,
    gamma=0.15,           # 好配置比例
    n_candidates=64,      # 候选数
    min_points_in_model=15,
    random_fraction=0.1   # 随机探索比例
)

# 添加领域约束
def hidden_heads_constraint(config):
    return config['hidden_dim'] % config['num_heads'] == 0

tpe.add_constraint(hidden_heads_constraint)

# 更新重要性
tpe.update_importance({'lr': 0.35, 'hidden_dim': 0.25})

# 添加观测
for config, loss in historical_data:
    tpe.update(config, loss)

# 采样新配置
new_config = tpe.sample()
print(new_config)
# {'hidden_dim': 320, 'num_layers': 28, 'lr': 8.5e-5, ...}
```

---

## 6. 配置空间

### 6.1 默认超参数

| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `hidden_dim` | 分类 | [128,192,256,320,384,448,512] | 320 | 隐藏层维度 |
| `num_layers` | 整数 | [12, 36] | 30 | Transformer层数 |
| `num_heads` | 分类 | [8, 16, 32] | 16 | 注意力头数 |
| `lr` | 对数连续 | [1e-5, 1e-3] | 1e-4 | 学习率 |
| `weight_decay` | 对数连续 | [1e-6, 1e-2] | 1e-4 | 权重衰减 |
| `batch_size` | 分类 | [512, 1024, 2048] | 1024 | 批次大小 |
| `grad_clip` | 连续 | [0.5, 2.0] | 1.0 | 梯度裁剪 |
| `rho` | 连续 | [0.7, 0.95] | 0.85 | 噪声相关系数 |

### 6.2 自定义配置空间

```python
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)

# 创建自定义配置空间
cs = ConfigurationSpace()

# 添加超参数
cs.add_hyperparameter(CategoricalHyperparameter(
    'hidden_dim', choices=[256, 320, 384], default_value=320
))

cs.add_hyperparameter(UniformIntegerHyperparameter(
    'num_layers', lower=20, upper=32, default_value=30
))

cs.add_hyperparameter(UniformFloatHyperparameter(
    'lr', lower=5e-5, upper=5e-4, log=True, default_value=1e-4
))

# 使用自定义配置空间
optimizer = EnhancedBOHBOptimizer(
    configspace=cs,
    min_budget=10,
    max_budget=100
)
```

---

## 7. 领域约束

### 7.1 内置约束

模块内置以下约束：

```python
# 约束1: hidden_dim必须是num_heads的倍数
def hidden_dim_heads_constraint(config):
    return config['hidden_dim'] % config['num_heads'] == 0

# 约束2: 深网络应使用较小学习率
def depth_lr_constraint(config):
    if config['num_layers'] > 24 and config['lr'] > 5e-4:
        return False
    return True
```

### 7.2 添加自定义约束

```python
from enhanced_bohb import EnhancedBOHBOptimizer

optimizer = EnhancedBOHBOptimizer(configspace=cs, ...)

# 添加自定义约束
def my_constraint(config):
    # 大batch需要足够大的hidden_dim
    if config['batch_size'] >= 2048 and config['hidden_dim'] < 256:
        return False
    return True

optimizer.tpe_sampler.add_constraint(my_constraint)
```

---

## 8. 结果分析

### 8.1 查看优化结果

```python
# 获取最佳配置
best_config, best_loss = optimizer.get_best_config()

# 获取参数重要性排名
importance_ranking = optimizer.get_importance_ranking()
for param, score in importance_ranking:
    print(f"{param}: {score:.4f}")

# 获取统计信息
stats = optimizer.get_statistics()
print(f"总观测数: {stats['n_observations']}")
print(f"最小损失: {stats['min_loss']:.4f}")
print(f"各预算观测数: {stats['observations_by_budget']}")
```

### 8.2 结果文件

优化完成后，结果保存在 `result_dir` 目录：

```
bohb_results/run_20240101_120000/
├── config.json         # 运行配置
├── checkpoint.json     # 检查点（可用于恢复）
└── final_results.json  # 最终结果
```

**final_results.json 格式：**
```json
{
  "best_config": {
    "hidden_dim": 320,
    "num_layers": 28,
    "num_heads": 16,
    "lr": 8.5e-05,
    "weight_decay": 1.2e-04,
    "batch_size": 1024,
    "grad_clip": 1.0,
    "rho": 0.87
  },
  "best_loss": 0.4523,
  "importance_ranking": [
    ["lr", 0.3521],
    ["hidden_dim", 0.2456],
    ["num_layers", 0.1523]
  ],
  "statistics": {
    "n_observations": 100,
    "min_loss": 0.4523,
    "mean_loss": 0.6234
  }
}
```

### 8.3 从检查点恢复

```bash
# 命令行
python run_enhanced_bohb.py --resume bohb_results/run_xxx/checkpoint.json
```

```python
# Python
optimizer.load_checkpoint('bohb_results/run_xxx/checkpoint.json')
```

---

## 9. 高级用法

### 9.1 分阶段优化

```python
from enhanced_bohb import (
    create_architecture_only_configspace,
    create_training_only_configspace
)

# 阶段1: 架构搜索
print("阶段1: 架构搜索")
arch_cs = create_architecture_only_configspace()
arch_optimizer = EnhancedBOHBOptimizer(
    configspace=arch_cs,
    min_budget=10,
    max_budget=50
)
best_arch, _ = arch_optimizer.optimize(evaluate_fn, n_iterations=30)

# 阶段2: 训练参数搜索（固定架构）
print("阶段2: 训练参数搜索")
train_cs = create_training_only_configspace()
train_optimizer = EnhancedBOHBOptimizer(
    configspace=train_cs,
    min_budget=20,
    max_budget=100
)

def evaluate_with_fixed_arch(config, budget):
    full_config = {**best_arch, **config}
    return worker.evaluate(full_config, budget)

best_train, _ = train_optimizer.optimize(evaluate_with_fixed_arch, n_iterations=50)

# 合并最佳配置
final_config = {**best_arch, **best_train}
```

### 9.2 并行评估

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def parallel_evaluate(configs_and_budgets):
    """并行评估多个配置"""
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(worker.evaluate, config, budget)
            for config, budget in configs_and_budgets
        ]
        for future in futures:
            results.append(future.result())
    return results
```

### 9.3 自定义评估函数

```python
def custom_evaluate_fn(config, budget):
    """
    自定义评估函数

    可以添加：
    - 多目标优化
    - 自定义指标
    - 额外约束检查
    """
    # 训练模型
    loss, info = worker.evaluate(config, budget)

    # 添加额外惩罚项
    n_params = info.get('n_params', 0)
    param_penalty = 0.01 * (n_params / 1e6)  # 参数量惩罚

    # 组合损失
    combined_loss = loss + param_penalty

    info['original_loss'] = loss
    info['param_penalty'] = param_penalty

    return combined_loss, info

# 使用自定义评估函数
optimizer.optimize(custom_evaluate_fn, n_iterations=100)
```

---

## 10. 常见问题

### Q1: 如何选择min_budget和max_budget？

**建议：**
- `min_budget`: 能够区分明显差配置的最小epoch数（通常5-10）
- `max_budget`: 完整训练所需的epoch数（通常100-200）
- 比值 `max_budget/min_budget` 建议在10-30之间

### Q2: eta参数如何选择？

- `eta=3`（默认）：平衡探索与利用
- `eta=2`：更多配置被完整评估（计算量大）
- `eta=4`：更激进的淘汰（可能错过好配置）

### Q3: 观测到参数重要性变化怎么办？

这是正常的。早期观测不足时，重要性估计可能不准确。建议：
- 至少收集20-30个观测后再参考重要性
- 关注置信区间，重叠多的参数重要性差异不显著

### Q4: 如何处理离散参数？

模块自动处理：
- 分类参数：使用索引编码，采样后四舍五入
- 整数参数：采样后取整
- 对数参数：在对数空间采样

### Q5: 内存不足怎么办？

1. 减小 `batch_size`
2. 使用 `--simplified` 模式
3. 减小 `max_budget`

---

## 11. 性能预期

基于您的PopulationDiT模型，预期性能：

| 配置 | 迭代数 | 预计时间 | 预期效果 |
|------|--------|----------|----------|
| 快速测试 | 20 | 2-4小时 | 初步筛选 |
| 标准优化 | 50-100 | 8-20小时 | 良好结果 |
| 深度优化 | 150+ | 30+小时 | 充分优化 |

**预期收益：**
- 相比随机搜索，样本效率提升3-5倍
- 通过早停节省50-70%计算资源
- 自动识别lr、hidden_dim等关键参数

---

## 12. 参考文献

1. Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. ICML.
2. Bergstra, J., et al. (2011). Algorithms for Hyper-Parameter Optimization. NeurIPS.
3. Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An Efficient Approach for Assessing Hyperparameter Importance. ICML.

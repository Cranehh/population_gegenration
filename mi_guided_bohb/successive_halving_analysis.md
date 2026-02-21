# MI-Guided BOHB — Successive Halving 算法 Budget 分配机制分析

## 1. 总体架构

本项目实现了一个**互信息引导的 BOHB**（Bayesian Optimization + Hyperband）超参数优化框架，核心由三个阶段构成：

```
Phase 1: 互信息先验 (MI Prior)
    ↓  从数据计算变量间依赖结构，构建高斯先验分布
Hyperband 循环 (n_brackets 次)
    ├─ 采样 → 从当前分布采样候选配置
    ├─ Successive Halving → 逐轮评估 + 淘汰
    └─ Phase 2: 梯度冲突感知 → 更新采样分布
```

相关文件：
- `mi_guided_optimizer.py`：核心 `MIGuidedBOHB` 类（单机）
- `parallel_optimizer.py`：`ParallelMIGuidedBOHB`（多 GPU）
- `mutual_information_prior.py`：MI 先验计算
- `gradient_conflict.py`：梯度冲突检测 + 分布更新
- `conflict_aware_worker.py`：训练 Worker（实际评估函数）
- `run_mi_guided_bohb.py`：主运行脚本

---

## 2. eta 参数的作用

`eta`（缩减因子，默认 = 3）是 Successive Halving 的核心控制参数，同时影响三个层面：

| 层面 | 公式 | 含义 |
|------|------|------|
| Bracket 总数 | `s_max = floor(log(B_max / B_min) / log(eta))` | 决定 Hyperband 包含多少个 bracket |
| 每轮淘汰率 | 每轮保留 `1/eta` 的配置 | eta=3 → 每轮淘汰 2/3 |
| Budget 倍增 | 每轮 budget 乘以 eta | 资源分配的几何级数增长 |

**默认参数下 (min_budget=5, max_budget=50, eta=3) 的计算**：

```
s_max = floor(log(50/5) / log(3)) = floor(2.096) = 2
```

因此共 3 个 bracket（s = 0, 1, 2）。

---

## 3. Budget 分配逻辑

### 3.1 核心公式（`_init_hyperband_params`，`mi_guided_optimizer.py:132`）

```python
s_max = floor(log(max_budget / min_budget) / log(eta))

for s in range(s_max + 1):
    n_configs   = ceil((s_max + 1) / (s + 1) * eta^s)    # 初始配置数
    min_budget_s = max_budget * eta^(-s)                   # 本 bracket 起始 budget
    budgets     = [int(min_budget_s * eta^i) for i in range(s+1)]  # budget 序列
```

### 3.2 默认参数下的 Bracket 配置表

| Bracket (s) | 初始配置数 | Budget 序列 (epoch) | SHA 轮数 |
|-------------|------------|---------------------|----------|
| 0 | 3  | [50]          | 1（直接全量训练）|
| 1 | 5  | [16, 50]      | 2 |
| 2 | 9  | [5, 16, 50]   | 3 |

**设计意图**：
- Bracket 0：少量配置，充分训练（最可靠）
- Bracket 2：大量配置，快速筛选（最高效）
- 每个 bracket 的总计算量大致相等（Hyperband 的等计算量设计）

### 3.3 Successive Halving 内部流程（`run_successive_halving`，`mi_guided_optimizer.py:190`）

```
输入：bracket s，当前采样分布

1. 从分布中采样 n_configs 个配置
2. for round_idx, budget in enumerate(budgets):
     a. 评估所有当前存活配置（用 budget 个 epoch 训练）
     b. 收集梯度冲突暴露度
     c. if not 最后一轮:
          n_keep = max(1, len(configs) / eta)    # ← 淘汰：仅保留 top 1/eta
          configs = sorted_by_loss[:n_keep]
3. 聚合冲突暴露度（所有轮的平均）
4. 返回 BracketResult（含最佳配置、所有评估、聚合暴露度）
```

**以 Bracket 2 为例**（9 个初始配置，budgets=[5,16,50]）：

```
Round 1: 9 configs × 5 epochs  → 保留 top 3（淘汰 6 个）
Round 2: 3 configs × 16 epochs → 保留 top 1（淘汰 2 个）
Round 3: 1 config  × 50 epochs → 最终结果
```

### 3.4 并行模式的调整（`parallel_optimizer.py:146`）

并行模式下，配置数会向上取整为 GPU 数量的整数倍：

```python
adjusted = ceil(n_configs / n_gpus) * n_gpus
```

淘汰时也做类似对齐，确保每批评估恰好占满所有 GPU：

```python
n_keep = max(n_gpus, ceil(n_keep / n_gpus) * n_gpus)
```

---

## 4. Hyperband 外层循环

### 4.1 Bracket 循环顺序（`optimize`，`mi_guided_optimizer.py:346`）

```python
for bracket_idx in range(n_brackets):
    bracket = bracket_idx % (s_max + 1)   # 循环遍历所有 bracket
    result  = run_successive_halving(bracket=bracket)
    update_distribution(result)            # Phase 2
```

以默认 `n_brackets=10`、`s_max=2` 为例，执行顺序为：
```
Iter 1→s=0, Iter 2→s=1, Iter 3→s=2, Iter 4→s=0, ..., Iter 10→s=0
```

这与标准 Hyperband 一致，保证每种 bracket 被多次执行。

### 4.2 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1：构建 MI 先验（仅执行一次）                              │
│  ① compute_mi_matrix(data) → 归一化互信息矩阵 M[i,j]           │
│  ② compute_group_mi_statistics() →                             │
│     ∙ within_MI：块内平均 MI（依赖强度）                         │
│     ∙ cross_MI：跨块平均 MI（耦合强度）                          │
│     ∙ spectral_entropy：谱熵（结构复杂度）                       │
│  ③ build_prior_distribution() →                                 │
│     ∙ μ_weight ∝ (within + cross) / total_importance           │
│     ∙ σ_weight ∝ spectral_entropy（不确定性高 → 方差大）         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Hyperband 循环（n_brackets 次）                                  │
│                                                                  │
│  for bracket_idx in range(n_brackets):                           │
│    bracket = bracket_idx % (s_max + 1)                           │
│                                                                  │
│    ┌─────────────────────────────────────────────────────────┐   │
│    │  Successive Halving（bracket s）                         │   │
│    │  采样 n_configs 个配置                                   │   │
│    │  for budget in budgets:                                  │   │
│    │    ① 评估所有配置（ConflictAwareWorker.evaluate）        │   │
│    │       ∙ 训练 budget 个 epoch                             │   │
│    │       ∙ 每 conflict_detection_freq 步检测梯度冲突        │   │
│    │       ∙ 验证集评估（SRMSE）                              │   │
│    │    ② 淘汰：保留 top 1/eta                               │   │
│    │  → 输出：BracketResult（含 aggregated_exposure）         │   │
│    └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│    ┌─────────────────────────────────────────────────────────┐   │
│    │  Phase 2：冲突感知分布更新                               │   │
│    │  ① 归一化冲突暴露度 E_k = Σ_{j≠k} λ_j · C_kj          │   │
│    │  ② 方差膨胀：σ_new = σ_old × (1 + β × E)              │   │
│    │     （冲突高 → 方差大 → 更多探索）                      │   │
│    │  ③ 均值偏移：μ_new = (1-α)×μ_old + α×weighted_mean    │   │
│    │     （向损失小的存活配置靠拢）                           │   │
│    └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        返回最佳配置
```

---

## 5. 梯度冲突检测机制

### 5.1 冲突矩阵（`gradient_conflict.py:110`）

```
C_ij = max(0, -cos(g_i, g_j))
```

当梯度余弦相似度为负（两 Loss 的梯度方向相反）时，冲突度 > 0。

### 5.2 冲突暴露度（`gradient_conflict.py:169`）

```
E_k = Σ_{j≠k} λ_j · C_kj
```

表示第 k 个 Loss 被其他 Loss（权重 λ_j）压制的程度。

### 5.3 分布更新规则（`gradient_conflict.py:347`）

```python
# 方差膨胀（鼓励探索高冲突的参数空间）
inflation_factor = 1 + β × E_normalized × n_losses
σ_new = clip(σ_old × inflation_factor, min_std=0.05, max_std=2.0)

# 均值偏移（向好配置学习）
weights = softmax(-losses / mean_loss)   # 损失越小权重越大
weighted_mean = Σ weights_i × config_i[param]
μ_new = (1 - α) × μ_old + α × weighted_mean
```

默认超参：`β=0.5`（方差膨胀率），`α=0.3`（均值偏移率）

---

## 6. 关键参数总结

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `eta` | 3 | SHA 缩减因子：每轮保留 1/3；budget 倍增 3× |
| `min_budget` | 5 | 最短训练 epoch（Bracket 2 第一轮）|
| `max_budget` | 50 | 最长训练 epoch（全量评估）|
| `n_brackets` | 10 | Hyperband 总迭代次数（循环遍历 s_max+1 个 bracket）|
| `variance_inflation_rate` (β) | 0.5 | 冲突越高 → 方差膨胀越快 → 探索范围越大 |
| `mean_shift_rate` (α) | 0.3 | 向最优存活配置偏移的速率 |
| `conflict_detection_freq` | 20 | 每 20 步检测一次梯度冲突 |

---

## 7. 与标准 BOHB 的区别

| 特性 | 标准 BOHB | MI-Guided BOHB |
|------|-----------|----------------|
| 初始先验 | 均匀 / 随机 | **互信息先验**（数据驱动）|
| 采样分布 | KDE（基于历史评估）| **高斯 + 梯度冲突驱动更新** |
| 分布更新信号 | 历史 loss 排名 | **梯度冲突暴露度 + 存活配置加权均值** |
| 多 GPU 支持 | 否（原版）| **ProcessPoolExecutor 并行评估** |
| 评估函数 | 黑盒 loss | **SRMSE（合成人口统计相似度）** |

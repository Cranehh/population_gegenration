# 代码实现角度联合审查报告

**审查对象**: `budget_allocation_proposals.md` 5 个方案
**对照基准**: `parallel_optimizer.py`、`mi_guided_optimizer.py`、`conflict_aware_worker.py`、`gradient_conflict.py`
**审查日期**: 2026-02-21

---

## 总体审查框架

每个方案从四个维度评分（1–5 分）：

| 维度 | 说明 |
|------|------|
| **伪代码可实现性** | 逻辑闭合性、边界条件、依赖函数是否已存在 |
| **与现有代码兼容性** | 修改量、接口变动程度 |
| **多 GPU 并行可行性** | 是否破坏 `_parallel_evaluate` 的批量统一 budget 设计 |
| **实现代价** | 核心修改点数量与风险 |

---

## 方案 1：OCBA-m 软性淘汰

### 1.1 伪代码可实现性 ★★★☆☆

**问题 1 — IndexError 边界缺失**
```python
k = max(1, ceil(len(alive) / eta))
c = (means[sorted_alive[k-1]] + means[sorted_alive[k]]) / 2  # ← 越界风险
```
当 `len(alive) == eta`（如只剩 3 个，k=1，但 `sorted_alive[1]` 是第 2 名，OK）；但当 `len(alive) == k`（如只剩 1 个）时，`sorted_alive[k]` 会越界。需要加护栏：
```python
if k >= len(alive):
    k = len(alive) - 1  # 或直接跳过本轮分配
```

**问题 2 — 方差冷启动不稳定**
```python
sigmas = {c: std(loss_history[c]) + 1e-6 for c in alive}
```
每个配置在第一轮只有 1 次历史评估，`std([single_value]) = 0`，退化为 `1e-6`，导致权重仅由 `delta` 决定，OCBA 信号失效。需要至少 2 轮评估或使用 MC Dropout 估计方差。

**问题 3 — 不等量 budget 的整数化误差**
```python
alloc = {c: round(budgets[round_idx+1] * weights[c] / total_w) for c in alive}
alloc = {c: max(alloc[c], budgets[0]) for c in alive}  # 保底 min_budget
```
四舍五入后各配置 budget 之和不等于 `budgets[round_idx+1]`，总计算量不可控。需要用 Hamilton 方法（largest remainder）精确分配。

**可实现结论**: 逻辑闭合，但边界条件和数值稳定性需修复后才能运行。

### 1.2 与 `parallel_optimizer.py` 的兼容性 ★★☆☆☆

核心冲突：`_parallel_evaluate(configs, budget)` 接收**统一**的 `budget`（`parallel_optimizer.py:153`），而方案 1 要求各配置获得不同 epoch 数。

```python
# 现有接口（parallel_optimizer.py:153）
def _parallel_evaluate(self, configs: List[Dict], budget: int) -> List[Tuple]:
```

不等量 budget 的适配方案：按 budget 值分组，每组单独调用一次 `_parallel_evaluate`，但这会产生 GPU 空闲等待。例如 4 GPU、3 种不同 budget，需分 3 批次调用，GPU 利用率降至 33%。

### 1.3 多 GPU 并行可行性 ★★☆☆☆

- **批次对齐破坏**: 不等量 budget 无法填满所有 GPU slots
- **一种折中方案**: 将各配置 budget 取整到 `min_budget` 的倍数，按 budget 桶分组并行，但会引入离散误差
- **实际可行性**: 低，需要重构批次调度逻辑

### 1.4 需要修改的函数

| 函数 | 文件 | 修改类型 |
|------|------|----------|
| `run_successive_halving` | `mi_guided_optimizer.py:190` | 重写淘汰逻辑（中等） |
| `run_successive_halving` | `parallel_optimizer.py:205` | 同步修改（中等） |
| `_parallel_evaluate` | `parallel_optimizer.py:153` | 扩展支持 budget 字典（较大） |
| 新增 `_ocba_weights` | `mi_guided_optimizer.py` | 新增工具方法（小） |
| 新增 `_hamilton_alloc` | `mi_guided_optimizer.py` | 整数分配工具（小） |

---

## 方案 2：MOCBA 多目标 Pareto 配置选择

### 2.1 伪代码可实现性 ★★☆☆☆

**问题 1 — 关键 Bug：未传入变量**
```python
def compute_domination_matrix(history):
    for k in range(m_objectives):  # ← m_objectives 未传入，运行时 NameError
        ...
        std_diff = sqrt(var_ik / N[i] + var_jk / N[j]) + 1e-8  # ← N 未传入
```
`m_objectives` 和 `N` 均为外部变量，函数签名不完整，直接运行会报 `NameError`。

**问题 2 — `marginal_gain` 未实现**
```python
delta_psi[d] = marginal_gain(d, S_p, history, delta_alloc)
```
这是整个方案的核心计算函数，但伪代码中完全没有实现。需要数值微分：
```python
# 正确实现需要模拟 N[d] += delta_alloc 后重新计算 psi 的变化
```

**问题 3 — 序贯循环与增量评估冲突**
```python
while sum(N.values()) < max_budget:
    for c in configs:
        history[c] = evaluate(c, epochs=N[c])  # 每轮对所有配置重新评估
```
每次循环都重新训练所有配置 `N[c]` 个 epoch（从头开始），而不是增量训练。对 50 epoch 的模型，这是 O(N²) 的总计算量。

**问题 4 — 评估接口不兼容**
当前 `ConflictAwareWorker.evaluate()` 在 `_validate()` 中只返回单一 `srmse_results`（标量），不分目标。需要拆分 SRMSE 为多个分项（年龄、性别、职业、学历、许可证、就业等）。

**可实现结论**: 存在关键 Bug 和未实现的核心函数，当前伪代码无法直接运行，需要大量补全。

### 2.2 与 `parallel_optimizer.py` 的兼容性 ★☆☆☆☆

- MOCBA 的序贯增量循环（每次只给一个配置追加 budget）从根本上与批量并行评估不兼容
- `_parallel_evaluate` 无法在此框架下使用
- `BracketResult` dataclass 不支持多目标输出（只有 `best_loss: float`）

### 2.3 多 GPU 并行可行性 ★☆☆☆☆

- 序贯单步分配无法并行，GPU 闲置
- 如果批量化（每步选 top-k，并行训练），偏离 MOCBA 严格最优性理论保证
- 实际可行性极低

### 2.4 需要修改的函数

| 函数 | 文件 | 修改类型 |
|------|------|----------|
| `_validate` | `conflict_aware_worker.py:565` | 拆分 SRMSE 为多目标（较大） |
| `evaluate` | `conflict_aware_worker.py:394` | 返回 `per_objective_losses`（中等） |
| `ConfigEvaluation` | `mi_guided_optimizer.py:27` | 扩展 `loss_components` 结构（小） |
| `BracketResult` | `mi_guided_optimizer.py:37` | 支持 Pareto 集输出（小） |
| `run_successive_halving` | `mi_guided_optimizer.py:190` | 替换为 MOCBA 循环（重写） |
| 新建 `mocba_selector.py` | — | `compute_domination_matrix`, `marginal_gain`, `compute_psi`（大） |
| `parallel_optimizer.py` | — | 大幅修改或基本弃用（大） |

---

## 方案 3：冲突增强型 OCBA（CA-OCBA）— 推荐首选

### 3.1 伪代码可实现性 ★★★★☆

**逻辑总体闭合**。仅有两个需要注意的地方：

**问题 1 — per-config exposure 未按配置存储**
伪代码中：
```python
conflict_hist = {c: [] for c in configs}
for cfg in alive:
    loss, exposure = evaluate_with_conflict(cfg, epochs=budget)
    conflict_hist[cfg].append(exposure)
```
但现有代码中，exposure 是 bracket 级别的聚合值（`all_exposures` 列表，`parallel_optimizer.py:290-300`），没有做 per-config 映射。实际 `_worker_evaluate` 返回 `conflict_exposure` 是对应单个 config 的，可以直接用，但需要修改收集逻辑：
```python
# 现有（parallel_optimizer.py:260-271）
for config, loss, loss_components, conflict_exposure in round_results:
    if conflict_exposure is not None:
        all_exposures.append(conflict_exposure)   # ← 没有 config 标识

# 需改为
config_exposures = {}
for config, loss, loss_components, conflict_exposure in round_results:
    config_key = str(sorted(config.items()))
    config_exposures[config_key] = conflict_exposure
```

**问题 2 — 方差冷启动（与方案 1 相同）**
第一轮只有一次评估，`std([x]) = 0`。可以用初始标准差占位符（如 `sigma_init = 0.1`）或者在第一轮跳过 CA-OCBA，回退到标准 top-k 淘汰。

**`weighted_top_k` 实现正确**：确定性 top-k 即直接按权重排序，等价于权重最高的 k 个存活，逻辑清晰。

**可实现结论**: 伪代码逻辑完整，修复两个小问题后可运行。

### 3.2 与 `parallel_optimizer.py` 的兼容性 ★★★★★

这是 5 个方案中兼容性最高的：
- **每轮 budget 仍然统一**：CA-OCBA 只改变"谁存活"，不改变"存活者训练多少 epoch"，`_parallel_evaluate(configs, budget)` 接口完全不变
- **n_keep 对齐 GPU 数量的逻辑已存在**：`parallel_optimizer.py:277-278` 已经处理了 `n_keep` 向 GPU 数量对齐的逻辑，CA-OCBA 的 `weighted_top_k` 结果可直接接入：
  ```python
  # parallel_optimizer.py:274（现有）
  n_keep = max(len(self.gpu_ids), int(np.ceil(n_keep / len(self.gpu_ids)) * len(self.gpu_ids)))
  ```
- `BracketResult` 的 `aggregated_exposure` 字段已存在，无需修改数据结构

### 3.3 多 GPU 并行可行性 ★★★★★

- 批量评估接口不变
- `_parallel_evaluate` 按 GPU 数量分批次调用，接口签名不变
- 仅在每轮批次评估结束后新增权重计算步骤（纯 CPU，毫秒级）
- 并行效率无损失

### 3.4 需要修改的函数

| 函数 | 文件 | 位置 | 修改类型 |
|------|------|------|----------|
| `run_successive_halving`（淘汰步骤） | `mi_guided_optimizer.py` | 249–256 行 | 替换 4 行为 CA-OCBA 权重计算（小） |
| `run_successive_halving`（并行版） | `parallel_optimizer.py` | 273–285 行 | 同步修改（小） |
| exposure 收集逻辑 | `parallel_optimizer.py` | 260–271 行 | 改为 per-config 字典（小） |
| 新增 `_ca_ocba_weights` | `mi_guided_optimizer.py` | — | 新增工具方法，约 15 行（小） |
| `MIGuidedBOHB.__init__` | `mi_guided_optimizer.py` | 57 行 | 添加 `gamma` 参数（微小） |

**实际修改代码量估计：< 50 行**

---

## 方案 4：自适应 η + UCB Bracket 选择

### 4.1 伪代码可实现性 ★★★☆☆

**问题 1 — 协方差矩阵不存在**
```python
H_t = spectral_entropy(current_distribution.covariance)
```
`ConflictAwareDistributionUpdater` 在 `gradient_conflict.py:292` 中维护的是独立的 `current_mean: Dict` 和 `current_std: Dict`，**没有协方差矩阵**。谱熵计算需要从对角协方差矩阵推导：
```python
# 实际可行的替代方案
stds = np.array([self.current_std[p] for p in self.param_names])
eigenvalues = stds ** 2  # 对角协方差矩阵的特征值就是方差
eigenvalues /= eigenvalues.sum()
H_t = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
```
注意：`mutual_information_prior.py` 中已有 `spectral_entropy` 的计算逻辑（第 177 行），但那是针对 MI 矩阵块的，两者语义不同，不能直接复用。

**问题 2 — `run_successive_halving` 不接受 `eta` 参数**
```python
result = run_successive_halving(bracket, eta=eta_t)  # ← eta 是实例变量
```
现有 `run_successive_halving` 使用 `self.eta`（`mi_guided_optimizer.py:251`），不接受 `eta` 参数。需修改函数签名：
```python
def run_successive_halving(self, ..., eta: Optional[float] = None):
    eta = eta or self.eta
    n_keep = max(1, int(len(configs) / eta))
```

**问题 3 — UCB 初期不稳定**
```python
n_s = max(ucb_stats[s]['n_runs'], 1)  # 初始 n_s=1 使 bonus 极大
ucb_scores[s] = mu - bonus
```
当某个 bracket 从未运行过时，`best_loss = inf`，`ucb_scores = inf - bonus = inf - large = nan` 可能出现。需要先轮询所有 bracket 各一次再启用 UCB，或对 `best_loss` 做初始化：
```python
# 建议：前 s_max+1 次迭代轮询所有 bracket，之后再用 UCB
```

**可实现结论**: 主逻辑合理，但需修复 2 个接口问题和 UCB 冷启动处理后才能运行。

### 4.2 与 `parallel_optimizer.py` 的兼容性 ★★★★☆

- UCB bracket 选择只修改 `optimize()` 外层循环，对并行无影响
- 自适应 η 需要修改 `run_successive_halving` 函数签名，父类和子类需同步
- `parallel_optimizer.py:run_successive_halving` 也需要接受 `eta` 参数并向下传递
- `_adjust_bracket_configs()` 中的 n_configs 基于固定 `self.eta` 计算（`parallel_optimizer.py:146`），若 η 运行时变化，`n_configs` 也应动态调整（但这又影响 GPU 对齐逻辑）

### 4.3 多 GPU 并行可行性 ★★★★☆

- 每轮 budget 序列和每轮评估批量不变，并行结构不受影响
- 唯一需注意的是：自适应 η 改变 `n_keep`，`parallel_optimizer.py:277-278` 的 GPU 对齐逻辑已经处理了这种情况，无需额外修改

### 4.4 需要修改的函数

| 函数 | 文件 | 位置 | 修改类型 |
|------|------|------|----------|
| `__init__` | `mi_guided_optimizer.py` | 57 行 | 添加 UCB 相关参数（小） |
| `optimize` | `mi_guided_optimizer.py` | 319 行 | 替换外层 bracket 循环（中等） |
| `optimize` | `parallel_optimizer.py` | 318 行 | 同步 UCB 逻辑（中等） |
| `run_successive_halving` | `mi_guided_optimizer.py` | 190 行 | 添加 `eta` 参数（小） |
| `run_successive_halving` | `parallel_optimizer.py` | 205 行 | 同步添加 `eta` 参数（小） |
| 新增 `_compute_distribution_entropy` | `mi_guided_optimizer.py` | — | 从 `current_std` 计算谱熵（小） |
| `_adjust_bracket_configs` | `parallel_optimizer.py` | 146 行 | 支持动态 η（选做，中等） |

---

## 方案 5：序贯边际 Budget 分配（SMA）

### 5.1 伪代码可实现性 ★★☆☆☆

**问题 1 — 增量训练未实现**
```python
new_result = evaluate(d_star, epochs=N[d_star] + delta)
```
当前 `ConflictAwareWorker.evaluate(config, budget)` **每次从头训练** `budget` 个 epoch（`conflict_aware_worker.py:450`），没有检查点续训功能。若 `N[d_star]` 从 5 增长到 50，第 10 次迭代需要训练 50 epoch（而非增量的 5 epoch），总计算量为 O(N²)，对 50 epoch 的任务会产生 `5+10+15+...+50 = 275` epoch 的实际训练量，而非 50。

**问题 2 — `update_rolling_std` 未定义**
```python
sigmas[d_star] = update_rolling_std(sigmas[d_star], old_mean, new_result.loss, N[d_star], delta)
```
此函数在伪代码中未实现，且增量更新标准差需要额外存储平方和（Welford 算法），不能仅凭 `old_mean` 和 `new_result.loss` 计算。

**问题 3 — `norm_cdf` 未定义**
```python
apcs *= norm_cdf((means[c] - means[i_star]) / std_diff)
```
需要 `from scipy.stats import norm; norm_cdf = norm.cdf`。虽然是小问题，但说明伪代码未完整考虑依赖。

**问题 4 — 单步分配粒度极细**
每次只追加 `delta=5` epoch 到一个配置，需要 `(B_total - n0 × n_configs) / delta` 轮迭代，对 `n_configs=9, n0=5, B_total=450` 的 Bracket 2 场景，需要 `(450-45)/5 = 81` 轮迭代，严重影响效率。

**可实现结论**: 存在未实现的关键函数和 O(N²) 训练代价的根本性问题，在没有检查点机制的情况下不可实用。

### 5.2 与 `parallel_optimizer.py` 的兼容性 ★☆☆☆☆

- 完全替换 SHA 结构，`run_successive_halving` 基本失效
- `_parallel_evaluate` 每次批量评估全部配置，而 SMA 每次只更新一个配置，两者逻辑根本冲突
- `BracketResult` 的概念在 SMA 中不存在（SMA 没有 bracket，是连续序贯过程）

### 5.3 多 GPU 并行可行性 ★★☆☆☆

- 严格 SMA 每步只给一个配置追加 budget，无法并行
- 批量化变体（选 top-k 边际增量配置同时训练）可以利用多 GPU，但：
  - 批量化后失去严格的 OCBA 最优性保证
  - 需要重新设计整个外层循环

### 5.4 需要修改的函数

| 函数 | 文件 | 修改类型 |
|------|------|----------|
| `evaluate` | `conflict_aware_worker.py:394` | 添加检查点续训支持（大） |
| `optimize` | `mi_guided_optimizer.py:319` | 完全重写（大） |
| `run_successive_halving` | `mi_guided_optimizer.py:190` | 弃用或保留作回退（大） |
| `optimize` | `parallel_optimizer.py:318` | 大幅修改（大） |
| 新增 `_welford_std_update` | — | Welford 滚动方差（小） |
| 新增 `compute_apcs` | — | APCS 计算（中等） |
| 新增检查点管理 | `conflict_aware_worker.py` | 模型保存/恢复（大） |

---

## 综合评分矩阵

| 方案 | 伪代码可实现性 | 兼容性 | 多GPU可行性 | 实现代价 | **综合** |
|------|:---:|:---:|:---:|:---:|:---:|
| 方案1 OCBA-m软淘汰 | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | 中 | ★★★☆☆ |
| 方案2 MOCBA多目标 | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | 大 | ★★☆☆☆ |
| **方案3 CA-OCBA** | **★★★★☆** | **★★★★★** | **★★★★★** | **小** | **★★★★★** |
| 方案4 自适应η+UCB | ★★★☆☆ | ★★★★☆ | ★★★★☆ | 中 | ★★★★☆ |
| 方案5 SMA序贯 | ★★☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ | 大 | ★★☆☆☆ |

---

## 优先实现路径（代码工程角度）

### 阶段 1：方案 3（CA-OCBA）— 最小可行实现

**核心改动（3 处，< 50 行）**：

```python
# 改动 1：mi_guided_optimizer.py:249–256（替换淘汰逻辑）
# 原始代码：
n_keep = max(1, int(len(configs) / self.eta))
sorted_evals = sorted(round_evals, key=lambda x: x.loss)
configs = [e.config for e in sorted_evals[:n_keep]]

# 替换为：
configs = self._ca_ocba_select(round_evals, self.eta, self.gamma)
```

```python
# 改动 2：新增工具方法
def _ca_ocba_select(self, evals, eta, gamma):
    n_keep = max(1, int(len(evals) / eta))
    k = n_keep  # 保留边界
    if k >= len(evals):
        return [e.config for e in evals]
    means = {e.config_key: e.loss for e in evals}
    sorted_e = sorted(evals, key=lambda x: x.loss)
    c_thresh = (sorted_e[k-1].loss + sorted_e[k].loss) / 2
    weights = {}
    for e in evals:
        exposure = e.conflict_info.exposure if e.conflict_info else {}
        E_i = np.mean(list(exposure.values())) if exposure else 0.0
        delta_eff = abs(e.loss - c_thresh) / (1 + gamma * E_i) + 1e-8
        sigma = 0.1  # 冷启动占位（后续可用滚动方差）
        weights[id(e)] = (sigma / delta_eff) ** 2
    top_ids = sorted(weights, key=weights.get, reverse=True)[:n_keep]
    return [e.config for e in evals if id(e) in top_ids]
```

```python
# 改动 3：parallel_optimizer.py:260–271（记录 per-config exposure）
# 在 round_results 收集时维护 config → exposure 映射
config_exposure_map = {}
for config, loss, loss_components, conflict_exposure in round_results:
    config_key = str(sorted(config.items()))
    config_exposure_map[config_key] = conflict_exposure
```

### 阶段 2：方案 4（自适应 η + UCB）— 叠加在方案 3 之上

**核心改动（2 处，约 40 行）**：

```python
# 改动 1：run_successive_halving 添加 eta 参数
def run_successive_halving(self, ..., eta: Optional[float] = None):
    eta = eta if eta is not None else self.eta
    ...
    n_keep = max(1, int(len(configs) / eta))

# 改动 2：optimize 中添加 UCB 逻辑
ucb_stats = {s: {'best_loss': float('inf'), 'n_runs': 0}
             for s in range(self.s_max + 1)}
for t in range(1, n_brackets + 1):
    if t <= self.s_max + 1:  # 冷启动：先轮询
        bracket = (t - 1) % (self.s_max + 1)
    else:
        bracket = self._ucb_select_bracket(ucb_stats, t, beta_ucb)
    eta_t = self._adaptive_eta(eta_base=self.eta)
    result = self.run_successive_halving(bracket=bracket, eta=eta_t)
    ucb_stats[bracket]['best_loss'] = min(
        ucb_stats[bracket]['best_loss'], result.best_loss)
    ucb_stats[bracket]['n_runs'] += 1
```

### 阶段 3：方案 2（MOCBA）— 仅在 SRMSE 分项可用后

前置条件：
1. `conflict_aware_worker.py:_validate()` 必须先拆分为多目标输出
2. 需要新建 `mocba_selector.py` 并完整实现 `marginal_gain`
3. 建议先在单 GPU 模式下验证，再扩展并行

---

## 关键风险提示

| 风险 | 影响方案 | 严重程度 |
|------|----------|----------|
| 方差冷启动（第一轮只有 1 次评估） | 方案 1、3 | 中（可用占位值规避） |
| 检查点续训缺失 | 方案 5 | 高（O(N²) 计算量，必须解决） |
| `compute_domination_matrix` 存在 NameError Bug | 方案 2 | 高（直接运行崩溃） |
| 自适应 η 破坏 `_adjust_bracket_configs` 的 GPU 对齐 | 方案 4 | 低（GPU 对齐逻辑已有兜底） |
| UCB 冷启动时 `best_loss=inf` 导致 NaN | 方案 4 | 中（前 s_max+1 轮轮询可规避） |

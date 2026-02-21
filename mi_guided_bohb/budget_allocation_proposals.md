# MI-Guided BOHB — Budget 分配改进方案

**背景**: 当前 MI-Guided BOHB 采用标准 Successive Halving（SHA）进行 budget 分配：每轮保留 top-1/η 配置，其余硬性淘汰，配置间分配的 budget 完全相等，不利用方差信息也不考虑多目标结构。OCBA-m 和 MOCBA 两篇论文提供了贝叶斯最优的预算分配理论，本文档基于两者提出 5 个改进方案。

---

## 方案 1：OCBA-m 软性淘汰（Uncertainty-Weighted Budget Redistribution）

### 核心思想

将 SHA 的**硬性排名淘汰**替换为 OCBA-m 的**不确定性加权软性再分配**。在每轮 SHA 结束时，不再按排名切断 bottom-1/η 的配置，而是：
1. 以当前保留/淘汰边界的**中点** `c` 作为阈值；
2. 按 `(σ̂_i / |J̄_i - c|)²` 分配下一轮的计算资源；
3. 权重低于 `ε` 的配置自然被"饿死"（软性淘汰）。

**直觉**: 越接近边界（难以区分）且方差越大（评估噪声大）的配置越需要更多 epoch 来确认表现，而非在信息不足时就被淘汰。

### 数学公式

设第 `t` 轮结束时有 `n_t` 个配置，按损失排序后 k = ⌈n_t / η⌉：

$$c = \frac{\bar{J}_{(k)} + \bar{J}_{(k+1)}}{2}$$

配置 `i` 的下轮 budget 分配权重：

$$w_i = \left(\frac{\hat{\sigma}_i}{\left|\bar{J}_i - c\right|}\right)^2$$

标准化后，下轮分配给配置 `i` 的 epoch 数：

$$B_i^{(t+1)} = B_{total}^{(t+1)} \cdot \frac{w_i}{\sum_j w_j}$$

软性淘汰条件（权重过低则丢弃）：

$$\text{drop config } i \iff w_i < \frac{\varepsilon}{\sum_j w_j}, \quad \varepsilon = \frac{1}{\eta \cdot n_t}$$

其中 `σ̂_i` 由跨 bracket 历史评估的滚动方差估计，或由 Dropout 前向传播得到。

### 伪代码

```python
def ocba_soft_halving(configs, budgets, eta, eps_factor=1.0):
    """
    Replace hard rank-based elimination with OCBA-m soft redistribution.
    """
    loss_history = {cfg: [] for cfg in configs}
    alive = list(configs)

    for round_idx, budget in enumerate(budgets):
        # 1. 评估所有存活配置
        results = evaluate_all(alive, epochs=budget)
        for cfg, (loss, _) in zip(alive, results):
            loss_history[cfg].append(loss)

        if round_idx == len(budgets) - 1:
            break  # 最后一轮无需分配

        # 2. 统计均值和方差
        means  = {c: mean(loss_history[c])  for c in alive}
        sigmas = {c: std(loss_history[c]) + 1e-6 for c in alive}

        # 3. 计算边界阈值 c（第 k 名与第 k+1 名中点）
        sorted_alive = sorted(alive, key=lambda c: means[c])
        k = max(1, ceil(len(alive) / eta))
        c = (means[sorted_alive[k-1]] + means[sorted_alive[k]]) / 2

        # 4. 计算 OCBA-m 权重
        weights = {}
        for cfg in alive:
            delta = abs(means[cfg] - c) + 1e-8
            weights[cfg] = (sigmas[cfg] / delta) ** 2

        total_w = sum(weights.values())
        eps = eps_factor / (eta * len(alive))

        # 5. 软性淘汰 + 不等量分配
        alive = [c for c in alive if weights[c] / total_w > eps]
        # 下一轮各配置 budget 按权重缩放（整合到 budget 序列中）
        # 实际实现：在 GPU 并行中，分配到最近的整数倍
        alloc = {c: round(budgets[round_idx+1] * weights[c] / total_w)
                 for c in alive}
        alloc = {c: max(alloc[c], budgets[0]) for c in alive}  # 保底 min_budget

    return best_config(alive, means)
```

### 优缺点

**优点**:
- 理论上渐近最优（OCBA-m Theorem 1）；精确解决"边界模糊"问题；避免因评估噪声误杀好配置
- 充分利用已有评估历史（方差估计），Budget 利用率更高

**缺点**:
- 需要多次历史评估才能得到可靠 σ̂（冷启动时不稳定）
- 各配置 budget 不等 → GPU 批次无法对齐（并行效率降低）
- 引入额外超参 `ε`（软淘汰阈值）

### 集成方式

- 修改 `mi_guided_optimizer.py:82`（`run_successive_halving` 中的淘汰步骤）
- 新增 `_ocba_weights(means, sigmas, eta)` 工具函数
- 并行模式下：将不等量 budget 转换为"批次数"（`alloc[c] / min_budget`），对齐到 GPU 数量整数倍

---

## 方案 2：MOCBA 多目标 Pareto 配置选择

### 核心思想

将合成人口优化问题的多个 SRMSE 分项（如年龄结构、家庭结构、性别分布等）作为**独立目标**，采用 MOCBA 框架识别**Pareto 最优配置集合**，并将 budget 分配给能最大程度提升 Pareto 集确信度的配置。

**直觉**: 当前 SHA 仅看总损失排名，可能淘汰某些在特定人口特征上表现优异但总分偏差的配置。MOCBA 允许这些配置存活并在最终输出多样化的 Pareto 解。

### 数学公式

设配置 `i` 的第 `k` 个目标损失估计为 `F̃_{ik} ~ N(f̄_{ik}, σ²_{ik}/δ_i)`，目标 `j` 支配 `i` 的概率：

$$P(\mu_j \preceq \mu_i) = \prod_{k=1}^{m} P(F_{jk} \leq F_{ik}) = \prod_{k=1}^{m} \Phi\left(\frac{f̄_{ik} - f̄_{jk}}{\sqrt{\sigma^2_{ik}/\delta_i + \sigma^2_{jk}/\delta_j}}\right)$$

配置 `i` 的性能指数（被支配概率之和）：

$$\psi_i = \sum_{j \neq i} P(\mu_j \preceq \mu_i)$$

对配置 `d` 额外分配 `Δ` 个 epoch 后，Pareto 集 `S_p` 中配置 `i` 的 ψ 改变量：

$$\Delta\psi_d = \sum_{i \in S_p} \Delta\psi_{id}$$

总边际收益最大化分配：

$$d^* = \arg\max_d \Delta\psi_d, \quad N_{d^*} \mathrel{+}= \Delta$$

### 伪代码

```python
def mocba_bracket(configs, delta_alloc, max_budget, n_pareto=3, m_objectives=4):
    """
    MOCBA-guided budget allocation: identify Pareto-optimal configs.
    Requires evaluate() to return per-objective losses.
    """
    N = {c: min_budget for c in configs}  # initial allocation
    history = {c: {'means': zeros(m_objectives), 'vars': ones(m_objectives)}
               for c in configs}

    while sum(N.values()) < max_budget:
        # 1. 评估（或更新统计）
        for c in configs:
            history[c] = evaluate(c, epochs=N[c])  # 返回各目标均值和方差

        # 2. 计算支配概率矩阵
        P_dom = compute_domination_matrix(history)  # P_dom[j][i] = P(j ≼ i)

        # 3. 计算性能指数
        psi = {i: sum(P_dom[j][i] for j in configs if j != i) for i in configs}

        # 4. 确定 Pareto 候选集（ψ 最小的 K 个）
        S_p = sorted(configs, key=lambda x: psi[x])[:n_pareto]

        # 5. 计算每个候选的边际收益 Δψ_d
        delta_psi = {}
        for d in configs:
            delta_psi[d] = marginal_gain(d, S_p, history, delta_alloc)

        # 6. 分配给边际收益最大的配置
        d_star = max(delta_psi, key=delta_psi.get)
        N[d_star] += delta_alloc

    return {c: psi[c] for c in S_p}  # 返回 Pareto 集及其置信度


def compute_domination_matrix(history):
    """P(j dominates i) = product over objectives of Φ(...)"""
    from scipy.stats import norm
    configs = list(history.keys())
    P = {j: {} for j in configs}
    for j in configs:
        for i in configs:
            if i == j:
                continue
            prob = 1.0
            for k in range(m_objectives):
                mu_ik, var_ik = history[i]['means'][k], history[i]['vars'][k]
                mu_jk, var_jk = history[j]['means'][k], history[j]['vars'][k]
                std_diff = sqrt(var_ik / N[i] + var_jk / N[j]) + 1e-8
                prob *= norm.cdf((mu_ik - mu_jk) / std_diff)
            P[j][i] = prob
    return P
```

### 优缺点

**优点**:
- 天然支持多目标（符合合成人口的多维拟合目标）；输出 Pareto 集（多样化解）
- 理论上以最小 budget 达到最高置信度（MOCBA 理论保证）

**缺点**:
- O(n²) 计算复杂度随配置数快速增长
- 需要 `evaluate()` 分别返回各目标损失（当前 `ConflictAwareWorker` 仅返回总 SRMSE）
- 各目标独立性假设在实际中可能不成立

### 集成方式

- 修改 `conflict_aware_worker.py:evaluate()` 返回 `per_objective_losses` 字典
- 新增 `mocba_selector.py` 模块实现 `compute_domination_matrix` 和 `marginal_gain`
- 在 `run_successive_halving` 中，当 bracket 含多轮时，将第 1 轮改为 MOCBA 增量分配模式

---

## 方案 3：冲突增强型 OCBA（Conflict-Augmented OCBA, CA-OCBA）

### 核心思想

将 MI-Guided BOHB 已有的**梯度冲突暴露度** `E_i` 融入 OCBA-m 的分配公式。高冲突配置在当前 budget 下损失估计不稳定（各 Loss 相互拉扯），需要更多 epoch 才能给出可靠的性能评估——即高冲突等价于"有效距离阈值更近"。

**直觉**: 冲突暴露度 `E_i` 越高，配置 `i` 的损失评估越噪声（梯度方向不稳定），可视为该配置在决策边界附近的"等效不确定性"增大，应分配更多 budget。

### 数学公式

定义冲突调整后的有效阈值距离（`γ > 0` 为冲突敏感系数）：

$$\tilde{\delta}_i = \frac{|\bar{J}_i - c|}{1 + \gamma \cdot E_i}$$

其中梯度冲突暴露度（来自 `gradient_conflict.py:169`）：

$$E_i = \sum_{j \neq i} \lambda_j \cdot C_{ij}, \quad C_{ij} = \max(0, -\cos(\mathbf{g}_i, \mathbf{g}_j))$$

代入 OCBA-m 公式，配置 `i` 的分配权重变为：

$$w_i = \left(\frac{\hat{\sigma}_i}{\tilde{\delta}_i}\right)^2 = \hat{\sigma}_i^2 \cdot \frac{(1 + \gamma E_i)^2}{|\bar{J}_i - c|^2}$$

归一化分配：

$$N_i^{(t+1)} = N_{total}^{(t+1)} \cdot \frac{w_i}{\sum_j w_j}$$

### 伪代码

```python
def ca_ocba_halving(configs, budgets, eta, gamma=0.5):
    """
    Conflict-Augmented OCBA: integrate gradient conflict into OCBA-m weights.
    Requires ConflictAwareWorker to return (loss, conflict_exposure).
    """
    alive = list(configs)
    loss_history   = {c: [] for c in configs}
    conflict_hist  = {c: [] for c in configs}  # E_i from ConflictAwareWorker

    for round_idx, budget in enumerate(budgets):
        # 1. 评估：ConflictAwareWorker 返回 (loss, E_i)
        for cfg in alive:
            loss, exposure = evaluate_with_conflict(cfg, epochs=budget)
            loss_history[cfg].append(loss)
            conflict_hist[cfg].append(exposure)

        if round_idx == len(budgets) - 1:
            break

        # 2. 统计
        means    = {c: mean(loss_history[c])    for c in alive}
        sigmas   = {c: std(loss_history[c]) + 1e-6 for c in alive}
        exposures = {c: mean(conflict_hist[c])  for c in alive}  # 滚动平均 E_i

        # 3. 边界阈值
        sorted_alive = sorted(alive, key=lambda c: means[c])
        k = max(1, ceil(len(alive) / eta))
        c = (means[sorted_alive[k-1]] + means[sorted_alive[k]]) / 2

        # 4. 冲突增强权重
        weights = {}
        for cfg in alive:
            delta_tilde = abs(means[cfg] - c) / (1 + gamma * exposures[cfg]) + 1e-8
            weights[cfg] = (sigmas[cfg] / delta_tilde) ** 2

        total_w = sum(weights.values())

        # 5. 按权重保留 top-k 配置（加权随机抽样 or 直接排名）
        n_keep = max(1, ceil(len(alive) / eta))
        # 加权存活：权重越大越可能保留（兼顾确定性与随机性）
        probs = [weights[c] / total_w for c in alive]
        survivors_idx = weighted_top_k(probs, n_keep)
        alive = [alive[i] for i in survivors_idx]

    return min(alive, key=lambda c: mean(loss_history[c]))


def weighted_top_k(probs, k):
    """Deterministic top-k by weight (can be replaced with stochastic sampling)."""
    return sorted(range(len(probs)), key=lambda i: -probs[i])[:k]
```

### 优缺点

**优点**:
- **最小改动集成**: 冲突暴露度 `E_i` 已由 `gradient_conflict.py` 计算，无需新增评估接口
- 将已有的 MI 冲突信号升级为 budget 分配的一阶公民
- 可提升对"冲突高但潜力大"配置的识别能力

**缺点**:
- 引入超参 `γ`（冲突敏感系数），需网格搜索或自适应调整
- 冲突暴露度与损失方差可能重叠信息，导致某些配置过度加权

### 集成方式

- **几乎零侵入**: 仅修改 `run_successive_halving` 的淘汰逻辑（`mi_guided_optimizer.py:82`），调用已有 `aggregated_exposure` 字段（`BracketResult` 中已存储）
- 新增 `_ca_ocba_weights(means, sigmas, exposures, eta, gamma)` 工具方法
- 在 `parallel_optimizer.py` 中对齐加权选择后的 `n_keep` 至 GPU 整数倍

---

## 方案 4：自适应 η + UCB Bracket 选择（Adaptive-η UCB）

### 核心思想

当前 Hyperband 外层以**固定循环顺序**（s=0,1,2,0,1,2...）运行 bracket，且缩减因子 `η` 固定为 3。本方案在两个层面引入自适应：
1. **UCB Bracket 选择**: 用 UCB（上置信界）策略动态选择最值得运行的 bracket；
2. **自适应 η**: 根据当前采样分布的不确定性（谱熵）动态调整每轮淘汰率。

**直觉**: 搜索早期不确定性高，η 应较小（保留更多候选，多探索）；搜索后期不确定性低，η 可较大（大胆淘汰，快速收敛）。UCB 选择 bracket 则让算法自动偏向"历史最佳 bracket 类型"。

### 数学公式

**自适应 η**（基于分布谱熵）：

$$H_t = -\sum_k \lambda_k \log \lambda_k, \quad \lambda_k = \text{eigenvalues of } \Sigma_t / \text{tr}(\Sigma_t)$$

$$\eta_t = \eta_{base} \cdot \exp\left(-\alpha_\eta \cdot \frac{H_{max} - H_t}{H_{max}}\right), \quad \eta_t \in [\eta_{min}, \eta_{max}]$$

不确定性越高（H_t 大）→ η_t 接近 η_min（温柔淘汰，保留更多）

**UCB Bracket 选择**（最小化损失）：

$$\text{UCB}(s, t) = \mu_s^* - \beta \sqrt{\frac{\ln t}{n_s}}, \quad s^* = \arg\min_s \text{UCB}(s, t)$$

其中 `μ_s*` 为 bracket `s` 历史最佳损失，`n_s` 为 bracket `s` 已运行次数，`β` 为探索权重。

### 伪代码

```python
def adaptive_eta_ucb_optimize(n_brackets, eta_base=3, beta_ucb=1.0, alpha_eta=2.0):
    """
    Hyperband with UCB bracket selection and variance-adaptive eta.
    """
    s_max = compute_s_max(min_budget, max_budget, eta_base)
    ucb_stats = {s: {'best_loss': float('inf'), 'n_runs': 0}
                 for s in range(s_max + 1)}

    for t in range(1, n_brackets + 1):
        # 1. UCB Bracket 选择（最小化损失 → 负 UCB 最小）
        ucb_scores = {}
        for s in range(s_max + 1):
            mu    = ucb_stats[s]['best_loss']
            n_s   = max(ucb_stats[s]['n_runs'], 1)
            bonus = beta_ucb * sqrt(log(t) / n_s)
            ucb_scores[s] = mu - bonus  # 越小越值得选
        bracket = min(ucb_scores, key=ucb_scores.get)

        # 2. 自适应 η：基于当前分布谱熵
        H_t   = spectral_entropy(current_distribution.covariance)  # 已在 Phase 1/2 计算
        H_max = log(n_params)  # 最大熵（均匀分布）
        eta_t = eta_base * exp(-alpha_eta * (H_max - H_t) / H_max)
        eta_t = clip(eta_t, eta_min=2.0, eta_max=eta_base * 2)

        # 3. 用自适应 η 运行 SHA
        result = run_successive_halving(bracket, eta=eta_t)

        # 4. 更新 UCB 统计
        ucb_stats[bracket]['best_loss'] = min(
            ucb_stats[bracket]['best_loss'], result.best_loss)
        ucb_stats[bracket]['n_runs'] += 1

        # 5. Phase 2：更新采样分布（不变）
        update_distribution(result)

    return global_best_config
```

### 优缺点

**优点**:
- 自适应 η 使算法在探索-利用之间动态平衡，无需手动调参
- UCB bracket 选择比轮询更智能，节省被浪费在低效 bracket 上的 budget
- 谱熵已在 Phase 1 计算（`spectral_entropy` in `mutual_information_prior.py`），可复用

**缺点**:
- UCB 的 β 仍是超参；且早期 bracket 统计量少，UCB 置信区间宽（稳定性差）
- η 变化使 `_init_hyperband_params` 中的 bracket 配置需动态重算
- 自适应 η 改变了"各 bracket 等计算量"的 Hyperband 设计原则

### 集成方式

- 修改 `optimize()` 外层循环（`mi_guided_optimizer.py:346`），增加 UCB 统计字典
- 将 `run_successive_halving` 的 `eta` 参数从固定值改为每次调用时传入
- 从 `MutualInformationPrior` 的已有谱熵计算中提取 `H_t`（`mutual_information_prior.py`）

---

## 方案 5：序贯边际 Budget 分配（Sequential Marginal Allocation, SMA）

### 核心思想

彻底抛弃 SHA 的"轮次"结构，采用 **OCBA 的序贯精神**：每次将固定的 `Δ` 个 epoch 分配给能**最大程度提升正确选择概率**的配置。以 APCS（近似正确选择概率）为目标函数，贪心地逐步增量分配，直到满足置信阈值或耗尽总 budget。

**直觉**: 这是 OCBA-m 序贯算法的直接移植——每 `Δ` 个 epoch 的边际价值最高的配置优先获得训练机会，配置不会被"提前淘汰"直到我们有足够统计信据。

### 数学公式

维护每个配置 `i` 的 loss 均值 `J̄_i`、方差 `σ̂²_i`、训练量 `N_i`（epoch 数），对应后验：

$$\tilde{J}_i \sim \mathcal{N}\!\left(\bar{J}_i,\ \frac{\hat{\sigma}_i^2}{N_i}\right)$$

近似正确选择概率（找最优单个配置 m=1）：

$$\text{APCS} = \prod_{i \neq i^*} \Phi\!\left(\frac{\bar{J}_i - \bar{J}_{i^*}}{\sqrt{\hat{\sigma}_i^2/N_i + \hat{\sigma}_{i^*}^2/N_{i^*}}}\right)$$

对配置 `d` 追加 `Δ` epoch 后 APCS 的边际增量（数值微分近似）：

$$\Delta\text{APCS}_d \approx \text{APCS}(N_d + \Delta) - \text{APCS}(N_d)$$

选择边际增量最大者：

$$d^* = \arg\max_d\ \Delta\text{APCS}_d, \quad N_{d^*} \mathrel{+}= \Delta$$

终止条件：`APCS > τ`（置信阈值，如 0.95）或 `ΣN_i > B_total`

### 伪代码

```python
def sequential_marginal_allocation(configs, total_budget, delta=5,
                                    tau=0.95, n0=min_budget):
    """
    OCBA-m sequential allocation: greedily assign delta epochs to the
    configuration with the highest marginal APCS improvement.
    """
    # 初始化：每个配置先做 n0 epoch
    N      = {c: n0      for c in configs}
    means  = {c: evaluate(c, n0).loss for c in configs}
    sigmas = {c: 1.0     for c in configs}  # 初始方差占位（后续估计）
    total_used = n0 * len(configs)

    while total_used < total_budget:
        # 1. 计算当前 APCS
        apcs_now = compute_apcs(means, sigmas, N)
        if apcs_now >= tau:
            break  # 已达置信阈值

        # 2. 计算每个配置的边际 APCS 增量
        delta_apcs = {}
        for d in configs:
            N_trial = dict(N); N_trial[d] += delta
            delta_apcs[d] = compute_apcs(means, sigmas, N_trial) - apcs_now

        # 3. 选择边际增量最大的配置
        d_star = max(delta_apcs, key=delta_apcs.get)

        # 4. 追加 delta epoch 训练并更新统计
        old_mean = means[d_star]
        new_result = evaluate(d_star, epochs=N[d_star] + delta)
        # 滚动更新均值和方差
        means[d_star]  = new_result.loss
        sigmas[d_star] = update_rolling_std(sigmas[d_star], old_mean,
                                            new_result.loss, N[d_star], delta)
        N[d_star]     += delta
        total_used    += delta

    return min(configs, key=lambda c: means[c])


def compute_apcs(means, sigmas, N):
    """Approximate P{CS} = product of P(J̃_i > J̃_{i*}) for all i ≠ i*."""
    i_star = min(means, key=means.get)
    apcs = 1.0
    for i, c in enumerate(means):
        if c == i_star:
            continue
        std_diff = sqrt(sigmas[c]**2 / N[c] + sigmas[i_star]**2 / N[i_star]) + 1e-8
        apcs *= norm_cdf((means[c] - means[i_star]) / std_diff)
    return apcs
```

### 优缺点

**优点**:
- **最接近 OCBA-m 理论最优**: 直接最大化 P{CS}，理论保证最强
- 完全序贯，无"淘汰"概念，不会因噪声误杀有潜力的配置
- 天然适应任意 budget 大小（不依赖 bracket 结构）

**缺点**:
- **评估粒度细**: 每次仅追加 `Δ=5` epoch，GPU 利用率低（频繁切换模型）
- **不支持批量并行**: 每步只选一个配置，无法同时跑多 GPU
- 冷启动阶段（N_i 小）方差估计不稳定，APCS 计算不可靠
- 完全替换 SHA/Hyperband 结构，集成代价大

### 集成方式

- 作为**独立外层策略**接入 `optimize()` 的外层循环（`mi_guided_optimizer.py:346`）
- 将 `ConflictAwareWorker.evaluate(config, epochs=N)` 的参数改为支持**增量训练**（`resume_from_checkpoint`），避免每次从头训练
- 推荐先在单 GPU 模式下验证，再考虑并行化（每步批量运行 top-k 边际增量配置）

---

## 方案对比总结

| 维度 | 方案1 OCBA-m软淘汰 | 方案2 MOCBA多目标 | 方案3 CA-OCBA | 方案4 自适应η+UCB | 方案5 SMA序贯 |
|------|------------------|-----------------|--------------|-----------------|--------------|
| 理论基础 | OCBA-m | MOCBA | OCBA-m + 冲突信号 | UCB + 谱熵 | OCBA-m（最纯） |
| 多目标支持 | 否 | **是** | 否 | 否 | 否（可扩展） |
| 代码改动量 | 中 | 大 | **小** | 中 | 大 |
| 并行友好度 | 中 | 低 | **高** | 高 | 低 |
| 利用方差信息 | **是** | **是** | **是** | 部分 | **是** |
| 利用冲突信号 | 否 | 否 | **是** | 否 | 否 |
| 推荐优先级 | ★★★★☆ | ★★★☆☆ | **★★★★★** | ★★★★☆ | ★★★☆☆ |

**推荐路径**:
1. 首先实现**方案 3（CA-OCBA）**: 改动最小，直接利用已有冲突信号，可快速验证效果
2. 在方案 3 基础上叠加**方案 4（自适应 η）**: UCB bracket 选择进一步提升 bracket 利用效率
3. 待多目标 SRMSE 分项数据可用后，引入**方案 2（MOCBA）** 实现多目标 Pareto 选择

---

*生成日期: 2026-02-21*
*参考: OCBA-m (Chen et al. 2006), MOCBA (Lee et al. 2004), successive_halving_analysis.md*

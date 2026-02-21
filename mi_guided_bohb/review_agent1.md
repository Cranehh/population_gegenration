# 联合审查报告：Budget 分配方案 vs. OCBA-m / MOCBA 论文

**审查人**: Agent 1（OCBA-m & MOCBA 文献视角）
**审查对象**: `budget_allocation_proposals.md` 5 个方案
**参考文献**:
- Paper 1: Chen, He, Fu — *Efficient Simulation Budget Allocation for Selecting an Optimal Subset* (OCBA-m)
- Paper 2: Lee, Chew, Teng, Goldsman — *Optimal Computing Budget Allocation for Multi-Objective Simulation Models* (MOCBA, WSC 2004)

---

## 总体评价

5 个方案均体现了对两篇论文的理解，核心框架转化思路合理。但存在若干**数学公式错误**（方案 4、5 各有严重缺陷）、**理论假设越界**（方案 3 无文献支撑的扩展）以及**实现级错误**（方案 2 代码作用域 bug）。以下逐方案详述。

---

## 方案 1：OCBA-m 软性淘汰

### 1.1 数学公式正确性

| 公式 | 评价 | 依据 |
|------|------|------|
| `c = (J̄_{(k)} + J̄_{(k+1)}) / 2` | ✅ 完全正确 | Paper 1 §3.3，c 取第 m 名与 m+1 名均值中点，直接对应 |
| `w_i = (σ̂_i / |J̄_i - c|)²` | ✅ 正确 | Paper 1 Theorem 1 (Eq.12)，`N_i/N_j = (σ_i/δ_i)²/(σ_j/δ_j)²`，归一化后等价 |
| `B_i = B_total * w_i / Σw_j` | ✅ 正确 | 归一化分配，与论文相容 |
| 软淘汰阈值 `ε = 1/(η * n_t)` | ⚠️ 无文献依据 | OCBA-m **不含任何淘汰机制**，该阈值为方案自设，合理但须单独验证 |

**细节澄清**：论文 Eq.(12) 中 `δ_i = J̄_i - c`（有符号），方案取 `|J̄_i - c|`（绝对值）——因为权重公式对 δ_i 取平方，两者结果相同，无误。

### 1.2 理论假设合理性

**问题 1（严重）：σ̂_i 估计方法与论文定义不符**

- 论文假设：`X_{ij} ~ N(J_i, σ_i²)`，每个方案 i 有多次**独立同分布**仿真复制，`σ_i²` 是**同一参数配置在固定 budget 下的输出噪声方差**。
- 方案实现：`sigmas = {c: std(loss_history[c]) + 1e-6}` 是对**不同 epoch 轮次**历史损失取标准差，混淆了**时序学习曲线波动**与**随机输出噪声**。这两类方差语义不同，该估计会高估早期 σ 并低估后期 σ。

**问题 2（中等）：局部最优性被标注为"渐近最优"**

方案"优点"中写"理论上渐近最优（OCBA-m Theorem 1）"——但论文定理的原文是：

> "The allocation given by (12) is asymptotically (as T → ∞) **a locally optimal** solution for OCBA problem (5)."

这是**局部**最优，非全局最优，且仅在 T → ∞ 渐近意义下成立。在 HPO 的有限 budget 情景下，该性质可能不保持，建议措辞修正。

**问题 3（轻）：i.i.d. 正态性假设**

论文假设每次仿真复制独立同正态分布。HPO 中每个 epoch 是带状态的序贯训练，不满足 i.i.d.。这不影响方案的工程价值，但应在论文写作时说明为"启发式应用"。

### 1.3 与论文一致性

- 核心分配公式与 Paper 1 完全一致，是 OCBA-m 最直接的工程移植。
- 软淘汰阈值是合理扩展但超出原文范围，须自行验证。
- **建议**：在论文或技术报告中，明确区分"Paper 1 推导部分"与"本方案扩展部分"。

---

## 方案 2：MOCBA 多目标 Pareto 配置选择

### 2.1 数学公式正确性

| 公式 | 评价 | 依据 |
|------|------|------|
| `P(μ_j ≼ μ_i) = ∏_k Φ(...)` | ✅ 正确 | Paper 2 Eq.(1) + Section 3 后验 `F̃_{ik} ~ N(f̄_{ik}, σ²_{ik}/δ_i)` |
| `ψ_i = Σ_{j≠i} P(μ_j ≼ μ_i)` | ✅ 正确 | Paper 2 Eq.(2) |
| `Δψ_d = Σ_{i∈S_p} Δψ_{id}` | ✅ 正确 | Paper 2 Section 4.1, Procedure II Step 2 |
| `d* = argmax Δψ_d` | ✅ 正确 | Paper 2 Procedure II Step 3（p=1 情形，与论文计算实验一致） |

**代码 Bug（严重）**：`compute_domination_matrix` 函数内引用了 `m_objectives`，但该变量定义在外层函数 `mocba_bracket` 的作用域中。在 Python 中这会在运行时抛出 `NameError`：

```python
# Bug: m_objectives 未作为参数传入
def compute_domination_matrix(history):
    for k in range(m_objectives):  # <- NameError: m_objectives not defined here
        ...
```

**修复方式**：将 `m_objectives` 作为参数传入，或定义为模块级常量。

### 2.2 理论假设合理性

**假设 1（必须明确）：K 已知**

Paper 2 明确陈述：*"we assume that the number of non-dominated designs (K) in the space is known in advance."*

方案通过 `n_pareto=3` 参数实现此假设，在实际中 K 未知。论文作者在结论中指出这是重要局限并建议未来放松——方案继承了此约束，**必须在文档和代码注释中明确标注**。

**假设 2（关键）：各目标独立**

公式 `P(μ_j ≼ μ_i) = ∏_k P(F_{jk} ≤ F_{ik})` 要求各目标独立（Paper 2 原文："Under the condition that the performance measures are **independent** from one another"）。

对于合成人口优化，年龄结构 SRMSE、家庭结构 SRMSE 等目标之间很可能存在相关性（同一人口模型参数同时影响多个维度）。若违反独立性：
- 高相关情形下，∏ 会低估真实支配概率
- Pareto 集可能因系统性偏差而产生错误

**建议**：在数值实验中验证目标间的实际相关性；若相关性高，须使用多元正态版本（Dudewicz and Taneja 1978 框架）。

### 2.3 与论文一致性

- 数学框架与 Paper 2 高度一致，是 MOCBA 在 HPO 场景最直接的移植。
- 代码 bug 需修复，独立性假设需在文档中说明。

---

## 方案 3：冲突增强型 OCBA（CA-OCBA）

### 3.1 数学公式正确性

| 公式 | 评价 | 依据 |
|------|------|------|
| `δ̃_i = |J̄_i - c| / (1 + γ * E_i)` | ⚠️ 内部自洽但无文献依据 | OCBA-m 未涉及冲突信号 |
| `w_i = σ̂_i² * (1+γE_i)² / |J̄_i - c|²` | ✅ 代数展开正确 | 代入 OCBA-m 权重公式后正确展开 |
| `E_i = Σ_{j≠i} λ_j * max(0, -cos(g_i, g_j))` | ⚠️ 来自项目自身，非 Paper 1/2 | gradient_conflict.py 中定义 |

### 3.2 理论假设合理性

**核心问题（严重）：理论依据缺失**

两篇论文均未涉及梯度冲突信号。`δ̃_i = δ_i / (1 + γ*E_i)` 将冲突暴露度解释为"有效距离阈值变小"，但这一等效仅是**类比推理**，不是理论推导。

**概念混淆问题（严重）**：在 OCBA-m 的框架中：
- `σ_i²` = 同一配置多次独立评估的**输出噪声**（随机性来源）
- `E_i` = 多个 loss 梯度方向之间的**冲突程度**（确定性的结构特性）

两者性质不同，不能通过简单乘法混合。具体而言：

> E_i 高（梯度冲突大）+ |J̄_i - c| 大（远离边界）→ w_i 被冲突信号大幅放大，但该配置其实明显劣于 top-k，不需要更多 budget。

这会导致 budget 浪费在确定性低质配置上。γ 的量级与 `|J̄_i - c|` 及 `E_i` 的量级强相关，难以迁移到不同问题设置。

### 3.3 与论文一致性

- 结构上从 Paper 1 出发但做了无文献支撑的扩展。
- 若要声称"OCBA-m 扩展"，需在论文中明确说明这是工程启发式方法，而非理论推导。
- **建议**：将 E_i 作为独立的"pre-filtering"信号（先淘汰高冲突+低质配置），再对剩余配置用标准 OCBA-m 分配，这样两个机制各自保持理论独立性。

---

## 方案 4：自适应 η + UCB Bracket 选择

### 4.1 数学公式正确性

**严重错误：自适应 η 公式方向反了**

方案中公式：
```
η_t = η_base * exp(-α_η * (H_max - H_t) / H_max)
```

设计意图（方案文档明确陈述）：
> "不确定性越高（H_t 大）→ η_t 接近 η_min（温柔淘汰，保留更多）"

验证：
- 当 H_t = H_max（最大熵，最高不确定性）时：指数项 = exp(0) = 1，`η_t = η_base`（无变化！）
- 当 H_t = 0（最小熵，最低不确定性）时：`η_t = η_base * exp(-α_η)`（最小值！）

**与设计意图完全相反**。正确公式应为：

```
η_t = η_base * exp(-α_η * H_t / H_max)
```

此时：H_t = H_max → η_t = η_base * exp(-α_η)（小 η，保守淘汰）；H_t = 0 → η_t = η_base（大 η，激进淘汰）。

**UCB 命名混淆（中等）**

方案公式 `UCB(s,t) = μ_s* - β√(ln(t)/n_s)` 配合 `argmin` 选择，实际上是 **LCB（Lower Confidence Bound）**——对最小化问题的乐观估计。UCB 通常指 Upper Confidence Bound，用于最大化问题。命名应修正为 LCB，或明确说明"这是 LCB for minimization"。

### 4.2 理论假设合理性

本方案不基于 OCBA-m 或 MOCBA 论文的理论，来自 MAB 文献和信息论，方案对比表中已如实标注"UCB + 谱熵"，这是诚实的。

**结构性问题**：Hyperband 的核心设计原则是"各 bracket 等计算量"，自适应 η 打破了这一设计前提，可能导致 bracket 之间 budget 不可比，影响 LCB 统计的一致性。

### 4.3 与论文一致性

- 本方案与 OCBA-m/MOCBA 基本无关，是独立的 Hyperband 扩展。
- **η 公式符号错误需立即修正**。

---

## 方案 5：序贯边际 Budget 分配（SMA）

### 5.1 数学公式正确性

**严重错误：APCS 公式在数学上不正确**

方案中的 APCS 公式：
```
APCS = ∏_{i≠i*} Φ((J̄_i - J̄_{i*}) / √(σ̂_i²/N_i + σ̂_{i*}²/N_{i*}))
```

这等于 `∏_{i≠i*} P{J̃_{i*} < J̃_i}`，而真实正确选择概率是：
```
P{CS_1} = P{ ∩_{i≠i*} {J̃_{i*} < J̃_i} }
```

这些事件 `{J̃_{i*} < J̃_1}`, `{J̃_{i*} < J̃_2}`, ... **不是独立的**（它们共享 J̃_{i*}），因此：
```
P{ ∩_{i≠i*} {J̃_{i*} < J̃_i} } ≠ ∏_{i≠i*} P{J̃_{i*} < J̃_i}
```

**Paper 1 的正确 APCS 方法**（引入阈值 c 使各项独立，§3.1 Eq.4）：

对 m=1，c = (J̄_{i*} + J̄_{(2)}) / 2：
```
APCS_1 = P{J̃_{i*} < c} · ∏_{j≠i*} P{J̃_j > c}
```

这里条件 `J̃_i < c` 与 `J̃_j > c` 关于不同设计，各项相互独立，乘积分解在理论上是正确的。方案绕过了这个关键步骤。

**建议修正 compute_apcs**：

```python
def compute_apcs_correct(means, sigmas, N):
    from math import sqrt
    from scipy.stats import norm
    i_star = min(means, key=means.get)
    sorted_others = sorted([c for c in means if c != i_star], key=lambda c: means[c])
    second_best = sorted_others[0]
    c = (means[i_star] + means[second_best]) / 2  # Paper 1 §3.3

    se = {i: sigmas[i] / sqrt(N[i]) + 1e-8 for i in means}
    apcs = norm.cdf((c - means[i_star]) / se[i_star])
    for i in means:
        if i == i_star:
            continue
        apcs *= norm.cdf((means[i] - c) / se[i])
    return apcs
```

**边际 ΔAPCS 计算（中等问题）**

方案使用数值差分 `APCS(N_d+Δ) - APCS(N_d)`。这是可行的数值方法，但 Paper 1 给出了**解析的最优分配公式**（Eq.12），理论上更高效且无近似误差。

### 5.2 理论假设合理性

- 序贯精神与 Paper 1 算法（§3.4）高度一致 ✅
- 方案说明"最接近 OCBA-m 理论最优"——此说法成立，但前提是 APCS 公式正确；当前公式有误，会导致次优的边际分配决策

### 5.3 与论文一致性

- 设计思路完全遵循 Paper 1 的 OCBA-m 序贯框架
- **APCS 公式需修正**，是阻碍理论一致性的核心错误

---

## 汇总：关键问题清单

### 必须修正（影响正确性）

| 编号 | 方案 | 问题描述 |
|------|------|---------|
| M1 | 方案 4 | η_t 公式方向与设计意图相反：高熵时应小 η，但公式给出大 η |
| M2 | 方案 5 | APCS = ∏ P{...} 不是联合概率（事件通过 J̃_{i*} 相关），应使用 Paper 1 的阈值 c 分解方法 |
| M3 | 方案 2 | `compute_domination_matrix` 中 `m_objectives` 未定义（Python 作用域 bug） |

### 理论假设需明确声明

| 编号 | 方案 | 需要声明的假设 |
|------|------|-------------|
| A1 | 方案 1/2/5 | σ_i 使用跨轮方差估计，违反 OCBA/MOCBA 的 i.i.d. 复制假设；须说明为启发式应用 |
| A2 | 方案 1 | Theorem 1 是局部最优，非全局最优；"渐近最优"表述需加限定 |
| A3 | 方案 2 | K（Pareto 集大小）假设已知；目标独立性假设须验证 |
| A4 | 方案 3 | 冲突信号 E_i 与损失方差 σ_i 性质不同，混合无理论支撑；须标注为"启发式扩展" |

### 建议改进

| 编号 | 方案 | 改进建议 |
|------|------|---------|
| I1 | 方案 1 | 软淘汰阈值 ε 可设 ε=0（纯 budget 重分配，无淘汰）作为基线进行消融实验 |
| I2 | 方案 3 | 将冲突信号作为独立 pre-filter，而非直接混入 OCBA 权重公式 |
| I3 | 方案 4 | 将 LCB bracket 选择与 η 自适应解耦，独立验证各自效果 |
| I4 | 方案 5 | 将 compute_apcs 替换为 Paper 1 §3.1 的正确阈值 c 下界公式 |

---

## 最终一致性评分

| 方案 | 公式正确性 | 假设合理性 | 与 Paper 1 (OCBA-m) 一致性 | 与 Paper 2 (MOCBA) 一致性 |
|------|-----------|-----------|--------------------------|--------------------------|
| 方案 1 OCBA-m 软淘汰 | 4/5 | 3/5 | 4/5 | 1/5 |
| 方案 2 MOCBA 多目标 | 4/5 | 3/5 | 2/5 | 4/5 |
| 方案 3 CA-OCBA | 3/5 | 2/5 | 2/5 | 1/5 |
| 方案 4 自适应 η+UCB | 1/5 (η公式错) | 3/5 | 1/5 | 1/5 |
| 方案 5 SMA 序贯 | 2/5 (APCS错) | 4/5 | 3/5 | 1/5 |

---

*审查日期: 2026-02-21*
*审查依据: Paper 1 (Chen, He, Fu — OCBA-m), Paper 2 (Lee, Chew, Teng, Goldsman — MOCBA WSC 2004)*

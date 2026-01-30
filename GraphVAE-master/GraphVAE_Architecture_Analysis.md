# GraphVAE 项目架构详细分析

## 项目概述

GraphVAE 是一个基于变分自编码器的分子图生成模型，专门用于生成小分子图结构。该实现基于 Simonovsky & Komodakis (2018) 的论文，使用 PyTorch 1.1 和 Python 3.6 开发。

## 1. 原始图的表示结构

### 文件位置：`graph_vae/graph_datastructure.py`

### 核心数据结构：`OneHotMolecularGraphs` 类

分子图通过三个主要张量表示：

#### 1.1 邻接矩阵 (`adj_matrices_special_diag`)
- **形状**: `[b, v, v]` 
- **含义**: 对角线表示节点是否存在（1=存在，0=不存在）
- **数据类型**: Float tensor
- **特点**: 对称矩阵，上三角和下三角镜像

#### 1.2 边特征张量 (`edge_atr_tensors`)
- **形状**: `[b, v, v, h_e]`
- **含义**: 边的 one-hot 编码，表示边的类型
- **支持的边类型**: 
  - 单键 (`Chem.BondType.SINGLE`)
  - 双键 (`Chem.BondType.DOUBLE`) 
  - 三键 (`Chem.BondType.TRIPLE`)
  - 芳香键 (`Chem.BondType.AROMATIC`)
- **数据位置**: `graph_datastructure.py:16`

#### 1.3 节点特征矩阵 (`node_atr_matrices`)
- **形状**: `[b, v, h_n]`
- **含义**: 节点的 one-hot 编码，表示原子类型
- **支持的原子类型**: 
  - QM9 数据集: `['C', 'N', 'O', 'S', 'Se', 'Si', 'I', 'F', 'Cl', 'Br']`
  - ZINC 数据集: `['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']`
- **数据位置**: `graph_datastructure.py:42-50`

### 关键代码实现：

```python
# SMILES 到图的转换过程 (graph_datastructure.py:309-345)
@classmethod
def create_from_smiles_list(cls, smiles_list, padding_size):
    # 创建空张量
    adj_mat = torch.zeros(batch_size, padding_size, padding_size)
    node_attr = torch.zeros(batch_size, padding_size, CHEM_DETAILS.num_node_types)
    edge_attr = torch.zeros(batch_size, padding_size, padding_size, CHEM_DETAILS.num_bond_types)
    
    # 逐个转换 SMILES
    for batch_idx, mol_smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(mol_smi)
        # 处理原子和键的特征
```

## 2. 隐空间结构

### 文件位置：`graph_vae/graph_vae_model.py`

### 2.1 编码器架构 (`EncoderNet` 类)

#### GGNN (Graph Gated Neural Network) 处理
- **输入**: 邻接矩阵和节点特征
- **处理层数**: T=2 (默认)
- **隐藏层维度**: 64 (默认)
- **代码位置**: `graph_vae_model.py:44-70`

#### 图聚合层 (`GrapTopGGNN`)
- **投影层**: `h' → 128` 维
- **门控机制**: `h' → 1` 维 sigmoid 激活
- **最终输出**: `128 → latent_dim*2` (均值和方差)
- **代码位置**: `graph_vae_model.py:21-42`

### 2.2 隐空间参数
- **维度**: 40 维 (默认)
- **分布**: 独立高斯分布 `N(μ, σ²)`
- **输出**: 编码器输出 80 维向量 (40维均值 + 40维对数方差)
- **先验**: 标准正态分布 `N(0, I)`

```python
# 隐空间分布构建 (graph_vae_model.py:129-130)
encoder = nn_paramterised_dists.NNParamterisedDistribution(
    EncoderNet(..., out_dim=2 * latent_space_dim),
    shallow_distributions.IndependentGaussianDistribution()
)
```

## 3. 解码器架构和生成图结构

### 文件位置：`graph_vae/graph_vae_model.py:73-123`

### 3.1 解码器网络架构
```python
self.parameterizing_net = nn.Sequential(
    nn.Linear(latent_space_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
    nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Linear(512, final_hidden_dim)
)
```

### 3.2 输出打包结构

解码器输出一个 **打包向量** (`packed_tensor`)，包含三部分：

1. **邻接矩阵 logits**: `v*(v+1)/2` 个元素（上三角矩阵）
2. **边特征 logits**: `v*(v-1)/2 * num_bond_types` 个元素
3. **节点特征 logits**: `v * num_node_types` 个元素

**代码位置**: `graph_datastructure.py:95-106`

### 3.3 生成图的分布表示 (`LogitMolecularGraphs`)

#### 解包过程 (`create_from_nn_prediction`)
- **邻接矩阵**: 从 logits 重构对称矩阵
- **边特征**: 4维张量 `[b, v, v, num_bond_types]`
- **节点特征**: 3维张量 `[b, v, num_node_types]`
- **代码位置**: `graph_datastructure.py:544-581`

#### 模式推断 (`calc_distributions_mode`)
- **节点存在**: `sigmoid(adj_diag) ≥ 0.5`
- **边连接**: 最小生成树 + 阈值 `> 0.5`
- **特征类型**: `argmax` 操作确定类型
- **代码位置**: `graph_datastructure.py:372-459`

## 4. 训练损失函数详细分析

### 4.1 训练主流程损失计算

**位置**: `scripts/train_graphvae.py:110-121`

```python
def train_one_epoch(vae, optimizer, dataloader, torch_device, add_step_func, beta):
    for data in dataloader:
        data.to_inplace(torch_device)
        optimizer.zero_grad()
        elbo = vae.elbo(data, beta=beta).mean()  # 计算 ELBO
        loss = -elbo  # 最小化负 ELBO
        loss.backward()
        optimizer.step()
```

**关键点**: 
- 训练目标是最大化 ELBO (Evidence Lower Bound)
- 实际优化时最小化 `-elbo`
- β参数控制KL散度权重 (`β = 1/40`)

### 4.2 ELBO 损失构成 (`VAE.elbo`)

**位置**: `graph_vae/autoencoders/variational.py:43-70`

```python
def elbo(self, x, beta=1., return_extra_vals=False):
    # 1. 编码器前向传播
    self.encoder.update(x)
    z_sample = self.encoder.sample_via_reparam(1)[0]  # 重参数化采样
    
    # 2. 解码器前向传播
    self.decoder.update(z_sample)
    log_like = -self.decoder.nlog_like_of_obs(x)  # 重构损失
    
    elbo = log_like  # 初始化为重构项
    
    # 3. KL散度项 (如果 β ≠ 0)
    if beta != 0.:
        kl_term = -self.encoder.kl_with_other(self.latent_prior)
        elbo += beta * kl_term
    
    return elbo
```

**ELBO 公式**: 
```
ELBO = E[log p(x|z)] - β·KL(q(z|x) || p(z))
     = -重构损失 - β·KL散度
```

### 4.3 重构损失详细构成 (`Decoder.nlog_like_of_obs`)

**位置**: `graph_vae/graph_vae_model.py:93-115`

```python
def nlog_like_of_obs(self, obs: graph_datastructure.OneHotMolecularGraphs) -> torch.Tensor:
    # 图匹配处理
    if self.run_graph_matching_flag:
        this_graph_matched = self._tilde_structure.return_matched_version_to_other(obs)
    else:
        this_graph_matched = self._tilde_structure
    
    # 计算负对数似然
    nll = this_graph_matched.neg_log_like(obs)
    
    # 可选：记录匹配和未匹配的损失对比
    if self._logger is not None:
        matched_nll = this_graph_matched.neg_log_like(obs) 
        unmatched_nll = self._tilde_structure.neg_log_like(obs)
        # 记录统计信息...
    
    return nll
```

### 4.4 图重构损失三个分量 (`LogitMolecularGraphs.neg_log_like`)

**位置**: `graph_datastructure.py:461-528`

```python
def neg_log_like(self, other_molecular_graphs: OneHotMolecularGraphs) -> torch.Tensor:
    # 1. 邻接矩阵损失 (节点存在性 + 边连接性)
    adj_loss = F.binary_cross_entropy_with_logits(
        self.adj_matrices_special_diag_logits, 
        other_molecular_graphs.adj_matrices_special_diag,
        reduction='none'
    ).sum(dim=[1,2])  # [b]
    
    # 2. 节点特征损失 (原子类型分类)
    node_loss = F.cross_entropy(
        node_logits_reshaped,           # [b*v, num_node_types]
        true_node_labels,               # [b*v]
        reduction='none'
    ).view(batch_size, -1).sum(dim=1)  # [b]
    
    # 3. 边特征损失 (键类型分类)
    edge_loss = F.cross_entropy(
        edge_logits_reshaped,           # [num_edges, num_bond_types]
        true_edge_labels,               # [num_edges]
        reduction='none'
    )  # 聚合到批次维度 [b]
    
    # 总重构损失 (等权重组合)
    total_loss = adj_loss + node_loss + edge_loss  # [b]
    return total_loss
```

#### 损失分量详解：

1. **邻接矩阵损失**: 
   - **目标**: 预测节点是否存在、边是否连接
   - **形式**: 二元交叉熵 `BCE(logits, targets)`
   - **权重**: λ_a = 1.0

2. **节点特征损失**:
   - **目标**: 预测原子类型 (C, N, O, S 等)
   - **形式**: 多类交叉熵 `CE(logits, labels)`
   - **权重**: λ_f = 1.0

3. **边特征损失**:
   - **目标**: 预测键类型 (单键、双键、三键、芳香键)
   - **形式**: 多类交叉熵 `CE(logits, labels)`
   - **权重**: λ_e = 1.0

### 4.5 KL散度损失

**计算位置**: `autoencoders/dist_parameterisers/shallow_distributions.py`

```python
def kl_with_other(self, other):
    # 对于独立高斯分布 q(z|x) = N(μ_q, σ_q²) 和 p(z) = N(μ_p, σ_p²)
    # KL(q||p) = 0.5 * Σ[log(σ_p²/σ_q²) + (σ_q² + (μ_q-μ_p)²)/σ_p² - 1]
    mean_q, log_var_q = self.mean_log_var.chunk(2, dim=1)
    mean_p, log_var_p = other.mean_log_var.chunk(2, dim=1)
    
    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)
    
    kl = 0.5 * torch.sum(
        log_var_p - log_var_q + 
        (var_q + (mean_q - mean_p).pow(2)) / var_p - 1,
        dim=1
    )
    return kl
```

**参数设置**:
- **先验分布**: `p(z) = N(0, I)` (标准正态分布)
- **后验分布**: `q(z|x) = N(μ_encoder, σ_encoder²)`
- **β权重**: `1/40` (β-VAE 技术)

### 4.6 图匹配算法影响

**MPM (Maximum Pooling Matching)** 对损失的影响：

1. **启用匹配** (`--mpm` 标志):
   - 在计算重构损失前，先找到最优节点对应关系
   - 通过75次迭代优化匹配矩阵
   - **优点**: 处理图同构问题，提高重构质量
   - **缺点**: 计算开销大

2. **禁用匹配** (NoGM版本):
   - 直接比较预测图和目标图
   - **优点**: 计算速度快
   - **缺点**: 可能受节点排序影响

### 4.7 训练监控指标

**TensorBoard 记录的损失组件**:
- `elbo(larger_better)`: 总ELBO值
- `reconstruction_term(larger_better)`: 重构项 
- `neg_kl_(no_beta)(larger_better)`: KL项(未加权)
- `nll_matched`: 匹配后的负对数似然
- `nll_unmatched`: 未匹配的负对数似然

**最终训练目标**:
```
minimize: L = -E[log p(x|z)] + β·KL(q(z|x) || p(z))
        = 重构损失 + β·KL散度
```

## 5. 数据流处理

### 5.1 SMILES 数据处理

**文件位置**: `graph_vae/smiles_data.py`

```python
class SmilesDataset(data.Dataset):
    def __init__(self, filename, transforms=None):
        with open(filename, 'r') as fo:
            data = [x.strip() for x in fo.readlines()]
```

#### 数据集分割：
- **训练集**: 除最后 20000 个样本
- **验证集**: 最后 20000-15000 个样本  
- **测试集**: 最后 10000 个样本

### 5.2 图转换管道

1. **SMILES → RDKit Mol**: `Chem.MolFromSmiles()`
2. **Mol → 图张量**: `create_from_smiles_list()`
3. **图张量 → 隐空间**: 编码器网络
4. **隐空间 → 图分布**: 解码器网络
5. **图分布 → 确定图**: 模式推断
6. **确定图 → SMILES**: `to_smiles_strings()`

## 6. 关键超参数

### 6.1 模型参数
- **隐空间维度**: 40
- **最大节点数**: 9 (QM9), 可配置
- **图隐藏层**: 64
- **GGNN 层数**: 2

### 6.2 训练参数
- **学习率**: 1e-3
- **批次大小**: 32
- **训练轮数**: 25
- **β 权重**: 1/40
- **优化器**: Adam (β1=0.5)

### 6.3 数据集支持
- **QM9**: 小有机分子，最多 9 个重原子
- **ZINC**: 药物分子，最多 20 个重原子

## 7. 实验和评估

### 7.1 生成质量评估
- **有效性**: SMILES 能否被 RDKit 解析
- **多样性**: 生成分子的结构差异
- **重构精度**: 编码-解码后的一致性

### 7.2 可视化工具
- **TensorBoard**: 损失曲线和生成样本
- **RDKit 绘图**: 分子结构可视化
- **样本对比**: 原始vs重构分子

这个详细分析涵盖了 GraphVAE 项目的完整架构，从原始数据表示到最终生成结果的整个流程，以及相应的代码实现位置。
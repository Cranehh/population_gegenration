# 贝叶斯网络Classifier Guidance解决零和结构零问题设计

## 1. 核心思想

### 1.1 问题定义
- **零问题**: 某些属性组合在训练数据中出现频次为0
- **结构零问题**: 逻辑上不可能的属性组合（如5岁博士、120岁工作者）
- **目标**: 生成多样化但逻辑合理的人口样本

### 1.2 贝叶斯网络优势
```
传统方法 → 硬约束规则 → 缺乏灵活性
贝叶斯网络 → 概率约束 → 软引导 + 逻辑推理
```

## 2. 架构设计

### 2.1 贝叶斯网络结构
```python
# 人口属性依赖关系建模
BayesianNetwork = {
    'age': [],  # 根节点
    'education': ['age'],  # 教育依赖年龄
    'occupation': ['age', 'education'],  # 职业依赖年龄和教育
    'income': ['age', 'education', 'occupation'],  # 收入依赖多个因素
    'license': ['age'],  # 驾照依赖年龄
    'gender': [],  # 独立节点
    'relation': ['age', 'gender'],  # 家庭关系依赖年龄性别
    'family_size': ['age', 'income'],  # 家庭规模依赖年龄收入
}
```

### 2.2 概率表定义
```python
# 条件概率表 (CPT)
conditional_probability_tables = {
    'P(education|age)': {
        'age_0_6': {'no_education': 0.95, 'primary': 0.05, 'secondary': 0, 'higher': 0},
        'age_7_18': {'no_education': 0.2, 'primary': 0.6, 'secondary': 0.2, 'higher': 0},
        'age_19_25': {'no_education': 0.05, 'primary': 0.15, 'secondary': 0.4, 'higher': 0.4},
        'age_26_65': {'no_education': 0.1, 'primary': 0.2, 'secondary': 0.3, 'higher': 0.4},
        'age_65_plus': {'no_education': 0.3, 'primary': 0.4, 'secondary': 0.2, 'higher': 0.1}
    },
    'P(occupation|age,education)': {
        # 复合条件概率
        ('age_0_18', 'any'): {'student': 0.8, 'unemployed': 0.2, 'worker': 0},
        ('age_19_65', 'higher'): {'professional': 0.6, 'manager': 0.2, 'worker': 0.2},
        ('age_65_plus', 'any'): {'retired': 0.9, 'part_time': 0.1, 'worker': 0}
    }
}
```

## 3. Classifier Guidance实现

### 3.1 贝叶斯约束分类器
```python
class BayesianConstraintClassifier(nn.Module):
    """贝叶斯网络约束的分类器"""
    
    def __init__(self, attribute_dims, bayesian_network, cpt_tables):
        super().__init__()
        self.bn = bayesian_network
        self.cpt = cpt_tables
        
        # 每个属性的分类器
        self.classifiers = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(sum(attribute_dims.values()), 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            ) for attr, dim in attribute_dims.items()
        })
        
        # 贝叶斯网络推理模块
        self.bn_inference = BayesianNetworkInference(bayesian_network, cpt_tables)
        
    def forward(self, x, target_marginals=None):
        """
        前向传播计算约束损失
        
        Args:
            x: [batch_size, feature_dim] 输入特征
            target_marginals: 目标边际分布
            
        Returns:
            constraint_loss: 贝叶斯约束损失
            probability_guidance: 概率引导信号
        """
        batch_size = x.shape[0]
        
        # 1. 预测所有属性概率分布
        attr_predictions = {}
        for attr_name, classifier in self.classifiers.items():
            logits = classifier(x)
            probs = F.softmax(logits, dim=-1)
            attr_predictions[attr_name] = probs
            
        # 2. 贝叶斯网络一致性检查
        consistency_loss = self.compute_consistency_loss(attr_predictions)
        
        # 3. 边际分布约束
        marginal_loss = self.compute_marginal_loss(attr_predictions, target_marginals)
        
        # 4. 结构零检测和修正
        structural_penalty = self.detect_structural_zeros(attr_predictions)
        
        # 5. 计算概率引导梯度
        probability_guidance = self.compute_probability_guidance(
            attr_predictions, consistency_loss, structural_penalty
        )
        
        total_loss = consistency_loss + marginal_loss + structural_penalty
        
        return total_loss, probability_guidance
```

### 3.2 贝叶斯网络推理
```python
class BayesianNetworkInference(nn.Module):
    """贝叶斯网络推理和约束检查"""
    
    def __init__(self, network_structure, cpt_tables):
        super().__init__()
        self.network = network_structure
        self.cpt = cpt_tables
        
    def compute_joint_probability(self, attr_values):
        """计算联合概率 P(X1, X2, ..., Xn)"""
        joint_prob = 1.0
        
        for attr, value in attr_values.items():
            # 获取父节点
            parents = self.network.get(attr, [])
            
            if not parents:
                # 边际概率
                prob = self.cpt[f'P({attr})'].get(value, 1e-8)
            else:
                # 条件概率
                parent_values = tuple(attr_values[p] for p in parents)
                prob = self.cpt[f'P({attr}|{",".join(parents)})'].get(
                    (parent_values, value), 1e-8
                )
            
            joint_prob *= prob
            
        return joint_prob
    
    def compute_conditional_probability(self, target_attr, target_value, evidence):
        """计算条件概率 P(target|evidence)"""
        # 使用变量消除算法或近似推理
        numerator = self.compute_joint_probability({**evidence, target_attr: target_value})
        
        # 边际化target_attr
        denominator = sum(
            self.compute_joint_probability({**evidence, target_attr: val})
            for val in self.get_possible_values(target_attr)
        )
        
        return numerator / (denominator + 1e-8)
```

### 3.3 结构零检测
```python
def detect_structural_zeros(self, attr_predictions):
    """检测和惩罚结构零组合"""
    structural_rules = [
        # 年龄-教育不一致
        ('age_0_6', 'education_higher', 1000.0),  # 幼儿不能有高等教育
        ('age_7_12', 'education_higher', 100.0),   # 小学生不能有高等教育
        
        # 年龄-职业不一致  
        ('age_0_15', 'occupation_manager', 500.0), # 未成年不能当经理
        ('age_16_18', 'occupation_retired', 200.0), # 青少年不能退休
        
        # 教育-职业不一致
        ('education_no', 'occupation_professor', 300.0), # 无教育不能当教授
        ('education_primary', 'occupation_doctor', 250.0), # 小学学历不能当医生
        
        # 家庭关系不一致
        ('age_0_12', 'relation_spouse', 800.0),   # 儿童不能是配偶
        ('age_0_15', 'relation_parent', 600.0),   # 未成年不能是父母
    ]
    
    total_penalty = 0.0
    batch_size = list(attr_predictions.values())[0].shape[0]
    
    for age_cond, other_cond, penalty_weight in structural_rules:
        # 解析条件
        age_attr, age_val = age_cond.split('_', 1)
        other_attr, other_val = other_cond.split('_', 1)
        
        # 获取对应的概率
        age_prob = attr_predictions[age_attr][:, self.get_attr_index(age_attr, age_val)]
        other_prob = attr_predictions[other_attr][:, self.get_attr_index(other_attr, other_val)]
        
        # 计算违反概率
        violation_prob = age_prob * other_prob
        penalty = penalty_weight * violation_prob.mean()
        total_penalty += penalty
        
    return total_penalty
```

## 4. 梯度引导机制

### 4.1 概率梯度计算
```python
def compute_probability_guidance(self, attr_predictions, consistency_loss, structural_penalty):
    """计算概率引导梯度"""
    
    guidance_signals = {}
    
    for attr_name, probs in attr_predictions.items():
        # 1. 一致性引导：推向贝叶斯网络一致的状态
        consistency_grad = torch.autograd.grad(
            consistency_loss, probs, retain_graph=True, create_graph=True
        )[0]
        
        # 2. 结构约束引导：远离结构零状态
        structural_grad = torch.autograd.grad(
            structural_penalty, probs, retain_graph=True, create_graph=True
        )[0]
        
        # 3. 多样性引导：增加熵以提高多样性
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        diversity_grad = torch.autograd.grad(
            -entropy.mean(), probs, retain_graph=True, create_graph=True
        )[0]
        
        # 4. 组合引导信号
        guidance_signals[attr_name] = {
            'consistency': consistency_grad,
            'structural': structural_grad,
            'diversity': diversity_grad,
            'combined': consistency_grad - 0.5 * structural_grad + 0.2 * diversity_grad
        }
        
    return guidance_signals
```

### 4.2 扩散采样中的引导
```python
def guided_diffusion_sampling_step(self, model, x_t, t, guidance_classifier, 
                                 target_marginals, guidance_scale=1.0):
    """在扩散采样中应用贝叶斯引导"""
    
    # 1. 标准扩散预测
    with torch.enable_grad():
        x_t.requires_grad_(True)
        
        # 模型预测噪声
        noise_pred = model(x_t, t)
        
        # 2. 贝叶斯约束分类
        constraint_loss, probability_guidance = guidance_classifier(
            x_t, target_marginals
        )
        
        # 3. 计算引导梯度
        guidance_grad = torch.autograd.grad(
            constraint_loss, x_t, retain_graph=True
        )[0]
        
        # 4. 应用引导到噪声预测
        guided_noise = noise_pred - guidance_scale * guidance_grad
        
    # 5. 去噪步骤
    x_prev = self.diffusion_scheduler.step(guided_noise, t, x_t).prev_sample
    
    return x_prev, {
        'constraint_loss': constraint_loss.item(),
        'guidance_norm': torch.norm(guidance_grad).item(),
        'probability_guidance': probability_guidance
    }
```

## 5. 训练策略

### 5.1 两阶段训练
```python
# 阶段1: 预训练贝叶斯分类器
def pretrain_bayesian_classifier():
    """使用真实数据预训练贝叶斯约束分类器"""
    for epoch in range(pretrain_epochs):
        for batch in dataloader:
            # 从真实数据学习条件概率表
            family_data, person_data = batch
            
            # 计算经验条件概率
            empirical_cpt = compute_empirical_cpt(family_data, person_data)
            
            # 更新贝叶斯网络参数
            cpt_loss = update_cpt_parameters(empirical_cpt)
            
            # 训练分类器匹配经验分布
            classifier_loss = train_classifiers_on_real_data(batch)
            
            total_loss = cpt_loss + classifier_loss
            total_loss.backward()
            optimizer.step()

# 阶段2: 联合训练扩散模型和引导分类器
def joint_training():
    """联合训练扩散模型和贝叶斯引导"""
    for epoch in range(joint_epochs):
        for batch in dataloader:
            # 扩散模型训练
            diffusion_loss = train_diffusion_model(batch)
            
            # 引导分类器训练
            guidance_loss = train_guidance_classifier(batch)
            
            # 端到端引导采样训练
            guided_samples = guided_sampling_during_training()
            consistency_loss = evaluate_bayesian_consistency(guided_samples)
            
            total_loss = diffusion_loss + guidance_loss + consistency_loss
            total_loss.backward()
            optimizer.step()
```

### 5.2 自适应引导强度
```python
def adaptive_guidance_scale(self, current_sample, target_marginals, base_scale=1.0):
    """根据样本质量自适应调整引导强度"""
    
    # 1. 计算当前样本的约束违反程度
    violation_score = self.compute_constraint_violations(current_sample)
    
    # 2. 计算与目标分布的偏差
    marginal_deviation = self.compute_marginal_deviation(current_sample, target_marginals)
    
    # 3. 计算样本多样性
    diversity_score = self.compute_sample_diversity(current_sample)
    
    # 4. 自适应调整
    if violation_score > threshold_high:
        # 高违反：增强约束引导
        guidance_scale = base_scale * 2.0
    elif diversity_score < threshold_low:
        # 低多样性：减弱约束，增强多样性
        guidance_scale = base_scale * 0.5
    else:
        # 正常情况
        guidance_scale = base_scale
        
    return guidance_scale
```

## 6. 实验验证

### 6.1 多样性指标
```python
def evaluate_diversity_metrics(generated_samples):
    """评估生成样本的多样性"""
    metrics = {}
    
    # 1. 香农熵
    for attr in attributes:
        attr_dist = compute_attribute_distribution(generated_samples, attr)
        entropy = -np.sum(attr_dist * np.log(attr_dist + 1e-8))
        metrics[f'{attr}_entropy'] = entropy
    
    # 2. 覆盖率
    unique_combinations = len(np.unique(generated_samples, axis=0))
    total_samples = len(generated_samples)
    metrics['coverage_rate'] = unique_combinations / total_samples
    
    # 3. 新奇度（与训练数据的不同程度）
    novelty_score = compute_novelty_score(generated_samples, training_data)
    metrics['novelty_score'] = novelty_score
    
    return metrics
```

### 6.2 约束满足度
```python
def evaluate_constraint_satisfaction(generated_samples):
    """评估约束满足度"""
    
    # 1. 硬约束违反率
    hard_violations = detect_hard_constraint_violations(generated_samples)
    hard_violation_rate = len(hard_violations) / len(generated_samples)
    
    # 2. 软约束偏差
    soft_deviations = compute_soft_constraint_deviations(generated_samples)
    avg_soft_deviation = np.mean(soft_deviations)
    
    # 3. 贝叶斯一致性评分
    consistency_scores = []
    for sample in generated_samples:
        score = compute_bayesian_consistency_score(sample)
        consistency_scores.append(score)
    avg_consistency = np.mean(consistency_scores)
    
    return {
        'hard_violation_rate': hard_violation_rate,
        'soft_deviation': avg_soft_deviation,
        'bayesian_consistency': avg_consistency
    }
```

## 7. 优势总结

### 7.1 解决零问题
- **概率插值**: 通过贝叶斯推理为零频率组合分配合理概率
- **平滑引导**: 软约束而非硬约束，避免采样困难

### 7.2 处理结构零
- **逻辑推理**: 贝叶斯网络编码领域知识和逻辑约束
- **梯度引导**: 引导采样远离不可能的组合

### 7.3 提升多样性
- **熵正则化**: 在引导中加入多样性项
- **自适应控制**: 根据当前状态调整引导强度

### 7.4 保持真实性
- **经验学习**: 从真实数据学习条件概率表
- **软约束**: 平衡约束满足和分布匹配
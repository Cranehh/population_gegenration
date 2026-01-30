"""
贝叶斯网络约束分类器 - 解决人口合成中的零和结构零问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import defaultdict


class BayesianNetworkStructure:
    """贝叶斯网络结构定义"""
    
    def __init__(self):
        # 人口属性依赖关系
        self.network_structure = {
            'age': [],  # 根节点
            'gender': [],  # 独立节点
            'education': ['age'],  # 教育依赖年龄
            'occupation': ['age', 'education'],  # 职业依赖年龄和教育
            'license': ['age'],  # 驾照依赖年龄
            'relation': ['age', 'gender'],  # 家庭关系依赖年龄性别
            'family_size': ['age'],  # 家庭规模依赖年龄
            'income': ['age', 'education', 'occupation']  # 收入依赖多个因素
        }
        
        # 属性值域定义
        self.attribute_domains = {
            'age': ['0-5', '6-12', '13-17', '18-25', '26-35', '36-50', '51-65', '65+'],
            'gender': ['male', 'female'],
            'education': ['none', 'primary', 'secondary', 'bachelor', 'master', 'phd'],
            'occupation': ['student', 'unemployed', 'worker', 'professional', 'manager', 
                          'retired', 'self_employed', 'civil_servant'],
            'license': ['no', 'yes'],
            'relation': ['head', 'spouse', 'child', 'parent', 'sibling', 'other'],
            'family_size': ['1', '2', '3', '4', '5', '6', '7', '8+'],
            'income': ['very_low', 'low', 'medium_low', 'medium', 'medium_high', 
                      'high', 'very_high']
        }
        
        # 结构零规则定义
        self.structural_zero_rules = [
            # 年龄-教育不一致
            (['0-5'], ['bachelor', 'master', 'phd'], 1000.0),
            (['6-12'], ['bachelor', 'master', 'phd'], 500.0),
            (['13-17'], ['master', 'phd'], 200.0),
            
            # 年龄-职业不一致
            (['0-5', '6-12'], ['manager', 'professional', 'retired'], 800.0),
            (['13-17'], ['manager', 'retired'], 300.0),
            (['65+'], ['student'], 400.0),
            
            # 教育-职业不一致
            (['none', 'primary'], ['professional', 'manager'], 600.0),
            (['none'], ['civil_servant'], 400.0),
            
            # 年龄-关系不一致
            (['0-5', '6-12'], ['spouse', 'head'], 1000.0),
            (['13-17'], ['spouse'], 500.0),
            (['0-5', '6-12', '13-17'], ['parent'], 800.0),
            
            # 年龄-驾照不一致
            (['0-5', '6-12', '13-17'], ['yes'], 300.0),
            
            # 年龄-收入不一致
            (['0-5', '6-12'], ['high', 'very_high'], 400.0),
        ]


class ConditionalProbabilityTable:
    """条件概率表管理"""
    
    def __init__(self, network_structure: BayesianNetworkStructure):
        self.network = network_structure
        self.cpt_tables = {}
        self._initialize_cpt_tables()
        
    def _initialize_cpt_tables(self):
        """初始化条件概率表"""
        
        # P(age) - 边际概率
        self.cpt_tables['age'] = {
            (): {  # 无条件
                '0-5': 0.08, '6-12': 0.10, '13-17': 0.08, '18-25': 0.12,
                '26-35': 0.18, '36-50': 0.20, '51-65': 0.16, '65+': 0.08
            }
        }
        
        # P(gender) - 边际概率
        self.cpt_tables['gender'] = {
            (): {'male': 0.51, 'female': 0.49}
        }
        
        # P(education|age) - 条件概率
        self.cpt_tables['education'] = {
            ('0-5',): {'none': 0.95, 'primary': 0.05, 'secondary': 0.0, 'bachelor': 0.0, 'master': 0.0, 'phd': 0.0},
            ('6-12',): {'none': 0.2, 'primary': 0.75, 'secondary': 0.05, 'bachelor': 0.0, 'master': 0.0, 'phd': 0.0},
            ('13-17',): {'none': 0.05, 'primary': 0.25, 'secondary': 0.68, 'bachelor': 0.02, 'master': 0.0, 'phd': 0.0},
            ('18-25',): {'none': 0.02, 'primary': 0.08, 'secondary': 0.35, 'bachelor': 0.45, 'master': 0.08, 'phd': 0.02},
            ('26-35',): {'none': 0.05, 'primary': 0.15, 'secondary': 0.25, 'bachelor': 0.35, 'master': 0.15, 'phd': 0.05},
            ('36-50',): {'none': 0.08, 'primary': 0.20, 'secondary': 0.30, 'bachelor': 0.28, 'master': 0.10, 'phd': 0.04},
            ('51-65',): {'none': 0.12, 'primary': 0.28, 'secondary': 0.35, 'bachelor': 0.18, 'master': 0.05, 'phd': 0.02},
            ('65+',): {'none': 0.25, 'primary': 0.35, 'secondary': 0.25, 'bachelor': 0.12, 'master': 0.02, 'phd': 0.01}
        }
        
        # P(license|age)
        self.cpt_tables['license'] = {
            ('0-5',): {'no': 1.0, 'yes': 0.0},
            ('6-12',): {'no': 1.0, 'yes': 0.0},
            ('13-17',): {'no': 0.95, 'yes': 0.05},
            ('18-25',): {'no': 0.15, 'yes': 0.85},
            ('26-35',): {'no': 0.08, 'yes': 0.92},
            ('36-50',): {'no': 0.05, 'yes': 0.95},
            ('51-65',): {'no': 0.10, 'yes': 0.90},
            ('65+',): {'no': 0.20, 'yes': 0.80}
        }
        
        # P(occupation|age,education) - 简化版本
        self.cpt_tables['occupation'] = self._initialize_occupation_cpt()
        
        # P(relation|age,gender) - 简化版本  
        self.cpt_tables['relation'] = self._initialize_relation_cpt()
        
    def _initialize_occupation_cpt(self):
        """初始化职业条件概率表"""
        occupation_cpt = {}
        
        # 为每个(age, education)组合定义职业分布
        for age in self.network.attribute_domains['age']:
            for edu in self.network.attribute_domains['education']:
                if age in ['0-5', '6-12']:
                    occupation_cpt[(age, edu)] = {
                        'student': 0.9, 'unemployed': 0.1, 'worker': 0.0, 'professional': 0.0,
                        'manager': 0.0, 'retired': 0.0, 'self_employed': 0.0, 'civil_servant': 0.0
                    }
                elif age == '13-17':
                    occupation_cpt[(age, edu)] = {
                        'student': 0.85, 'unemployed': 0.1, 'worker': 0.05, 'professional': 0.0,
                        'manager': 0.0, 'retired': 0.0, 'self_employed': 0.0, 'civil_servant': 0.0
                    }
                elif age == '65+':
                    occupation_cpt[(age, edu)] = {
                        'student': 0.0, 'unemployed': 0.1, 'worker': 0.1, 'professional': 0.05,
                        'manager': 0.05, 'retired': 0.7, 'self_employed': 0.0, 'civil_servant': 0.0
                    }
                else:  # 工作年龄
                    if edu in ['bachelor', 'master', 'phd']:
                        occupation_cpt[(age, edu)] = {
                            'student': 0.02, 'unemployed': 0.05, 'worker': 0.15, 'professional': 0.45,
                            'manager': 0.20, 'retired': 0.0, 'self_employed': 0.08, 'civil_servant': 0.05
                        }
                    elif edu in ['secondary']:
                        occupation_cpt[(age, edu)] = {
                            'student': 0.05, 'unemployed': 0.10, 'worker': 0.50, 'professional': 0.15,
                            'manager': 0.08, 'retired': 0.0, 'self_employed': 0.10, 'civil_servant': 0.02
                        }
                    else:  # 低教育
                        occupation_cpt[(age, edu)] = {
                            'student': 0.02, 'unemployed': 0.20, 'worker': 0.65, 'professional': 0.02,
                            'manager': 0.01, 'retired': 0.0, 'self_employed': 0.10, 'civil_servant': 0.0
                        }
                        
        return occupation_cpt
    
    def _initialize_relation_cpt(self):
        """初始化家庭关系条件概率表"""
        relation_cpt = {}
        
        for age in self.network.attribute_domains['age']:
            for gender in self.network.attribute_domains['gender']:
                if age in ['0-5', '6-12', '13-17']:
                    relation_cpt[(age, gender)] = {
                        'head': 0.0, 'spouse': 0.0, 'child': 0.95, 'parent': 0.0, 'sibling': 0.05, 'other': 0.0
                    }
                elif age in ['18-25']:
                    relation_cpt[(age, gender)] = {
                        'head': 0.15, 'spouse': 0.10, 'child': 0.65, 'parent': 0.0, 'sibling': 0.08, 'other': 0.02
                    }
                elif age in ['26-35', '36-50']:
                    relation_cpt[(age, gender)] = {
                        'head': 0.45, 'spouse': 0.35, 'child': 0.05, 'parent': 0.0, 'sibling': 0.10, 'other': 0.05
                    }
                elif age in ['51-65']:
                    relation_cpt[(age, gender)] = {
                        'head': 0.50, 'spouse': 0.30, 'child': 0.0, 'parent': 0.15, 'sibling': 0.03, 'other': 0.02
                    }
                else:  # 65+
                    relation_cpt[(age, gender)] = {
                        'head': 0.35, 'spouse': 0.25, 'child': 0.0, 'parent': 0.35, 'sibling': 0.03, 'other': 0.02
                    }
                    
        return relation_cpt
    
    def get_probability(self, attribute: str, value: str, parents_values: Tuple = ()):
        """获取条件概率"""
        if attribute not in self.cpt_tables:
            return 1e-8  # 平滑概率
            
        cpt = self.cpt_tables[attribute]
        
        if parents_values in cpt and value in cpt[parents_values]:
            return cpt[parents_values][value]
        else:
            # 平滑处理
            return 1e-8


class BayesianConstraintClassifier(nn.Module):
    """贝叶斯网络约束的分类器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network_structure = BayesianNetworkStructure()
        self.cpt_manager = ConditionalProbabilityTable(self.network_structure)
        
        # 属性维度映射
        self.attribute_dims = {
            attr: len(domains) for attr, domains in self.network_structure.attribute_domains.items()
        }
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 每个属性的分类器
        self.attribute_classifiers = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, dim)
            ) for attr, dim in self.attribute_dims.items()
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播预测所有属性概率分布
        
        Args:
            x: [batch_size, input_dim] 输入特征
            
        Returns:
            attr_predictions: 每个属性的概率分布字典
        """
        batch_size = x.shape[0]
        
        # 共享特征提取
        features = self.feature_extractor(x)
        
        # 预测所有属性概率分布
        attr_predictions = {}
        for attr_name, classifier in self.attribute_classifiers.items():
            logits = classifier(features)
            probs = F.softmax(logits, dim=-1)
            attr_predictions[attr_name] = probs
            
        return attr_predictions
    
    def compute_bayesian_consistency_loss(self, attr_predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算贝叶斯网络一致性损失"""
        batch_size = list(attr_predictions.values())[0].shape[0]
        total_loss = torch.tensor(0.0, device=list(attr_predictions.values())[0].device)
        
        # 遍历所有有父节点的属性
        for attr, parents in self.network_structure.network_structure.items():
            if not parents:  # 跳过根节点
                continue
                
            attr_probs = attr_predictions[attr]  # [batch_size, attr_dim]
            
            # 计算条件概率一致性
            for batch_idx in range(min(batch_size, 32)):  # 限制批次大小以避免内存问题
                for attr_val_idx, attr_val in enumerate(self.network_structure.attribute_domains[attr]):
                    
                    # 计算所有父节点组合的概率
                    parent_combinations = self._get_parent_combinations(parents)
                    
                    for parent_combo in parent_combinations[:10]:  # 限制组合数量
                        # 计算父节点组合的概率
                        parent_prob = 1.0
                        parent_values = []
                        
                        for parent_idx, parent_attr in enumerate(parents):
                            parent_val = parent_combo[parent_idx]
                            parent_val_idx = self.network_structure.attribute_domains[parent_attr].index(parent_val)
                            parent_prob *= attr_predictions[parent_attr][batch_idx, parent_val_idx]
                            parent_values.append(parent_val)
                        
                        # 获取理论条件概率
                        theoretical_prob = self.cpt_manager.get_probability(
                            attr, attr_val, tuple(parent_values)
                        )
                        
                        # 实际条件概率
                        actual_prob = attr_probs[batch_idx, attr_val_idx]
                        
                        # KL散度损失
                        if parent_prob > 1e-6:  # 避免除零
                            kl_loss = theoretical_prob * torch.log(
                                torch.tensor(theoretical_prob + 1e-8) / (actual_prob + 1e-8)
                            )
                            total_loss += parent_prob * kl_loss
        
        return total_loss / batch_size
    
    def compute_structural_zero_penalty(self, attr_predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算结构零惩罚"""
        batch_size = list(attr_predictions.values())[0].shape[0]
        total_penalty = torch.tensor(0.0, device=list(attr_predictions.values())[0].device)
        
        for forbidden_vals1, forbidden_vals2, penalty_weight in self.network_structure.structural_zero_rules:
            # 假设第一个是年龄，第二个是其他属性
            if len(forbidden_vals1) > 0 and len(forbidden_vals2) > 0:
                # 解析属性 - 简化处理，假设都是年龄相关
                for val1 in forbidden_vals1:
                    for val2 in forbidden_vals2:
                        # 在所有属性中寻找匹配
                        for attr_name, domains in self.network_structure.attribute_domains.items():
                            if val1 in domains and val2 in domains and attr_name != 'age':
                                continue  # 跳过同属性内的组合
                            elif val1 in domains:
                                attr1_name = attr_name
                                val1_idx = domains.index(val1)
                            elif val2 in domains:
                                attr2_name = attr_name
                                val2_idx = domains.index(val2)
                        
                        # 计算违反概率
                        try:
                            if 'attr1_name' in locals() and 'attr2_name' in locals():
                                prob1 = attr_predictions[attr1_name][:, val1_idx] if 'val1_idx' in locals() else 0
                                prob2 = attr_predictions[attr2_name][:, val2_idx] if 'val2_idx' in locals() else 0
                                violation_prob = prob1 * prob2
                                penalty = penalty_weight * violation_prob.mean()
                                total_penalty += penalty
                        except:
                            continue  # 跳过无法处理的规则
        
        return total_penalty
    
    def compute_marginal_constraint_loss(self, attr_predictions: Dict[str, torch.Tensor], 
                                       target_marginals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算边际分布约束损失"""
        if target_marginals is None:
            return torch.tensor(0.0)
            
        total_loss = torch.tensor(0.0, device=list(attr_predictions.values())[0].device)
        
        for attr_name, target_dist in target_marginals.items():
            if attr_name in attr_predictions:
                predicted_dist = attr_predictions[attr_name].mean(dim=0)  # 批次平均
                
                # KL散度
                kl_div = F.kl_div(
                    torch.log(predicted_dist + 1e-8),
                    target_dist + 1e-8,
                    reduction='batchmean'
                )
                total_loss += kl_div
                
        return total_loss
    
    def compute_guidance_loss(self, attr_predictions: Dict[str, torch.Tensor],
                            target_marginals: Optional[Dict[str, torch.Tensor]] = None,
                            weights: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算总引导损失"""
        
        if weights is None:
            weights = {'consistency': 1.0, 'structural': 0.5, 'marginal': 0.8, 'diversity': 0.2}
        
        # 1. 贝叶斯一致性损失
        consistency_loss = self.compute_bayesian_consistency_loss(attr_predictions)
        
        # 2. 结构零惩罚
        structural_penalty = self.compute_structural_zero_penalty(attr_predictions)
        
        # 3. 边际分布约束损失
        marginal_loss = self.compute_marginal_constraint_loss(attr_predictions, target_marginals)
        
        # 4. 多样性损失（熵正则化）
        diversity_loss = torch.tensor(0.0, device=list(attr_predictions.values())[0].device)
        for attr_name, probs in attr_predictions.items():
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            diversity_loss -= entropy.mean()  # 负熵（最大化熵）
        
        # 总损失
        total_loss = (weights['consistency'] * consistency_loss +
                     weights['structural'] * structural_penalty +
                     weights['marginal'] * marginal_loss +
                     weights['diversity'] * diversity_loss)
        
        loss_components = {
            'total_loss': total_loss,
            'consistency_loss': consistency_loss,
            'structural_penalty': structural_penalty,
            'marginal_loss': marginal_loss,
            'diversity_loss': diversity_loss
        }
        
        return total_loss, loss_components
    
    def _get_parent_combinations(self, parents: List[str], max_combinations: int = 20) -> List[Tuple]:
        """获取父节点值的所有组合（限制数量）"""
        if not parents:
            return [()]
            
        combinations = []
        parent_domains = [self.network_structure.attribute_domains[parent] for parent in parents]
        
        def generate_combinations(current_combo, remaining_domains):
            if len(combinations) >= max_combinations:
                return
            if not remaining_domains:
                combinations.append(tuple(current_combo))
                return
            
            for value in remaining_domains[0][:5]:  # 限制每个属性的值数量
                generate_combinations(current_combo + [value], remaining_domains[1:])
        
        generate_combinations([], parent_domains)
        return combinations


class GuidedSampler:
    """引导采样器 - 在扩散采样中集成贝叶斯约束"""
    
    def __init__(self, constraint_classifier: BayesianConstraintClassifier):
        self.classifier = constraint_classifier
        
    def guided_diffusion_step(self, model, x_t: torch.Tensor, t: torch.Tensor,
                            target_marginals: Optional[Dict[str, torch.Tensor]] = None,
                            guidance_scale: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        在扩散采样步骤中应用贝叶斯引导
        
        Args:
            model: 扩散模型
            x_t: 当前噪声状态
            t: 时间步
            target_marginals: 目标边际分布
            guidance_scale: 引导强度
            
        Returns:
            x_prev: 去噪后的状态
            guidance_info: 引导信息
        """
        
        with torch.enable_grad():
            x_t = x_t.requires_grad_(True)
            
            # 1. 标准扩散预测
            noise_pred = model(x_t, t)
            
            # 2. 贝叶斯约束分类
            attr_predictions = self.classifier(x_t)
            constraint_loss, loss_components = self.classifier.compute_guidance_loss(
                attr_predictions, target_marginals
            )
            
            # 3. 计算引导梯度
            if constraint_loss.requires_grad:
                guidance_grad = torch.autograd.grad(
                    constraint_loss, x_t, retain_graph=True, create_graph=False
                )[0]
            else:
                guidance_grad = torch.zeros_like(x_t)
            
            # 4. 应用引导到噪声预测
            guided_noise = noise_pred - guidance_scale * guidance_grad
        
        # 5. 去噪步骤（这里需要根据具体的扩散调度器实现）
        # x_prev = diffusion_scheduler.step(guided_noise, t, x_t).prev_sample
        # 简化处理
        x_prev = x_t - guided_noise * 0.01  # 简化的去噪步骤
        
        guidance_info = {
            'constraint_loss': constraint_loss.item(),
            'guidance_norm': torch.norm(guidance_grad).item(),
            **{k: v.item() for k, v in loss_components.items()}
        }
        
        return x_prev, guidance_info


def create_target_marginals_from_data(data: np.ndarray, attribute_names: List[str],
                                     network_structure: BayesianNetworkStructure) -> Dict[str, torch.Tensor]:
    """从真实数据创建目标边际分布"""
    
    target_marginals = {}
    
    for attr_idx, attr_name in enumerate(attribute_names):
        if attr_name in network_structure.attribute_domains:
            # 计算经验分布
            attr_values = data[:, attr_idx]
            unique_values, counts = np.unique(attr_values, return_counts=True)
            
            # 转换为概率分布
            probs = counts / len(data)
            
            # 创建完整的分布向量
            full_probs = np.zeros(len(network_structure.attribute_domains[attr_name]))
            for val, prob in zip(unique_values, probs):
                if val < len(full_probs):
                    full_probs[val] = prob
            
            target_marginals[attr_name] = torch.tensor(full_probs, dtype=torch.float32)
    
    return target_marginals


# 使用示例
if __name__ == "__main__":
    # 创建贝叶斯约束分类器
    input_dim = 128  # 扩散模型的特征维度
    classifier = BayesianConstraintClassifier(input_dim)
    
    # 创建引导采样器
    guided_sampler = GuidedSampler(classifier)
    
    # 模拟数据
    batch_size = 4
    x_t = torch.randn(batch_size, input_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    # 测试分类器
    attr_predictions = classifier(x_t)
    print("属性预测形状:")
    for attr, pred in attr_predictions.items():
        print(f"{attr}: {pred.shape}")
    
    # 测试引导损失
    loss, components = classifier.compute_guidance_loss(attr_predictions)
    print(f"\n引导损失: {loss.item():.6f}")
    print("损失组件:")
    for name, value in components.items():
        print(f"{name}: {value.item():.6f}")
    
    print("\n✅ 贝叶斯约束分类器测试完成！")
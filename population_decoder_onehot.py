import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PopulationOneHotDecoder:
    """基于One-Hot编码的人口数据解码器"""
    
    def __init__(self, encoder, model):
        """
        Args:
            encoder: PopulationDataEncoder实例 (使用one-hot编码)
            model: 训练好的PopulationDiT模型
        """
        self.encoder = encoder
        self.model = model
    
    def decode_sample(self, family_sample, person_sample):
        """
        解码单个样本 (one-hot版本)
        Args:
            family_sample: [family_dim] 家庭数据
            person_sample: [max_family_size, person_dim] 个人数据
        Returns:
            decoded_family: 解码后的家庭数据
            decoded_persons: 解码后的个人数据
        """
        # 解码家庭数据
        decoded_family = self._decode_family_data(family_sample)
        
        # 解码个人数据
        decoded_persons = []
        for person_idx in range(person_sample.shape[0]):
            person_data = person_sample[person_idx]
            if self._is_valid_person_data(person_data):  # 检查是否为有效成员
                decoded_person = self._decode_person_data(person_data)
                decoded_persons.append(decoded_person)
        
        return decoded_family, decoded_persons
    
    def _decode_family_data(self, family_sample):
        """解码家庭数据 (one-hot版本)"""
        decoded = {}
        
        # 连续变量：直接反标准化
        continuous_start = 0
        continuous_end = self.encoder.family_continuous_dim
        continuous_data = family_sample[continuous_start:continuous_end]
        
        for i, col in enumerate(self.encoder.family_continuous_cols):
            # 从[-1,1]恢复到标准化值，再反标准化
            normalized_val = continuous_data[i].item() * 3  # 恢复到[-3,3]
            scaler = self.encoder.scalers[f'family_{col}']
            original_val = scaler.inverse_transform([[normalized_val]])[0][0]
            decoded[col] = max(0, round(original_val))  # 确保非负整数
        
        # 离散变量：从one-hot恢复到类别
        categorical_start = continuous_end
        idx = categorical_start
        
        for col in self.encoder.family_categorical_cols:
            ohe = self.encoder.onehot_encoders[f'family_{col}']
            num_categories = len(ohe.categories_[0])
            
            # 提取该变量的one-hot向量
            onehot_vector = family_sample[idx:idx + num_categories]
            
            # 方法1: argmax解码 (硬解码)
            predicted_class = torch.argmax(onehot_vector).item()
            
            # 方法2: 软解码 (可选，用于处理模糊情况)
            # probabilities = F.softmax(onehot_vector, dim=0)
            # predicted_class = torch.multinomial(probabilities, 1).item()
            
            # 确保索引在有效范围内
            if predicted_class >= num_categories:
                predicted_class = num_categories - 1
            
            # 反编码到原始类别
            original_category = ohe.categories_[0][predicted_class]
            decoded[col] = original_category
            
            idx += num_categories
        
        return decoded
    
    def _decode_person_data(self, person_sample):
        """解码个人数据 (one-hot版本)"""
        decoded = {}
        
        # 连续变量：年龄
        age_normalized = person_sample[0].item()  # 已经在[-1,1]范围
        age_01 = (age_normalized + 1) / 2  # 转换到[0,1]
        age_original = age_01 * 123  # 恢复到实际年龄
        decoded['age'] = max(0, min(123, round(age_original)))
        
        # 离散变量：从one-hot恢复
        idx = 1  # 跳过年龄
        
        for col in self.encoder.person_categorical_cols:
            ohe = self.encoder.onehot_encoders[f'person_{col}']
            num_categories = len(ohe.categories_[0])
            
            # 提取该变量的one-hot向量
            onehot_vector = person_sample[idx:idx + num_categories]
            
            # argmax解码
            predicted_class = torch.argmax(onehot_vector).item()
            
            # 确保索引在有效范围内
            if predicted_class >= num_categories:
                predicted_class = num_categories - 1
            
            # 反编码到原始类别
            original_category = ohe.categories_[0][predicted_class]
            decoded[col] = original_category
            
            idx += num_categories
        
        return decoded
    
    def _is_valid_person_data(self, person_data):
        """检查是否为有效的个人数据 (非填充数据)"""
        # 简单检查：如果所有值都接近0，可能是填充数据
        return torch.sum(torch.abs(person_data)) > 0.1


class OneHotLoss(nn.Module):
    """针对One-Hot编码的损失函数"""
    
    def __init__(self, encoder, continuous_weight=1.0, categorical_weight=1.0):
        super().__init__()
        self.encoder = encoder
        self.continuous_weight = continuous_weight
        self.categorical_weight = categorical_weight
        
        # 创建变量类型掩码
        self._create_variable_masks()
    
    def _create_variable_masks(self):
        """创建one-hot版本的变量掩码"""
        # 计算总维度
        family_continuous_dim = self.encoder.family_continuous_dim
        family_categorical_dim = sum(self.encoder.family_categorical_dims)
        person_continuous_dim = self.encoder.person_continuous_dim
        person_categorical_dim = sum(self.encoder.person_categorical_dims)
        
        total_dim = (family_continuous_dim + family_categorical_dim + 
                    8 * (person_continuous_dim + person_categorical_dim))  # 假设max_family_size=8
        
        # 创建掩码
        self.continuous_mask = torch.zeros(total_dim)
        self.categorical_mask = torch.zeros(total_dim)
        
        idx = 0
        
        # 家庭连续变量
        self.continuous_mask[idx:idx + family_continuous_dim] = 1.0
        idx += family_continuous_dim
        
        # 家庭离散变量
        self.categorical_mask[idx:idx + family_categorical_dim] = 1.0
        idx += family_categorical_dim
        
        # 个人变量 (重复8次)
        for _ in range(8):  # max_family_size
            # 个人连续变量
            self.continuous_mask[idx:idx + person_continuous_dim] = 1.0
            idx += person_continuous_dim
            
            # 个人离散变量
            self.categorical_mask[idx:idx + person_categorical_dim] = 1.0
            idx += person_categorical_dim
    
    def forward(self, predicted, target):
        """
        计算混合损失
        Args:
            predicted: 模型预测的噪声
            target: 真实噪声
        """
        device = predicted.device
        continuous_mask = self.continuous_mask.to(device).unsqueeze(0)
        categorical_mask = self.categorical_mask.to(device).unsqueeze(0)
        
        # 连续变量MSE损失
        continuous_loss = F.mse_loss(
            predicted * continuous_mask,
            target * continuous_mask,
            reduction='mean'
        )
        
        # one-hot变量的特殊损失
        # 对于one-hot向量，我们希望保持其概率分布特性
        categorical_predicted = predicted * categorical_mask
        categorical_target = target * categorical_mask
        
        # 使用MSE损失，但也可以考虑KL散度
        categorical_loss = F.mse_loss(
            categorical_predicted,
            categorical_target,
            reduction='mean'
        )
        
        # 也可以添加one-hot约束损失
        # 确保one-hot向量的和接近1
        # onehot_constraint_loss = self._compute_onehot_constraint(categorical_predicted)
        
        total_loss = (self.continuous_weight * continuous_loss + 
                     self.categorical_weight * categorical_loss)
        
        return total_loss, continuous_loss, categorical_loss
    
    def _compute_onehot_constraint(self, categorical_data):
        """计算one-hot约束损失 (可选)"""
        # 这里需要根据具体的one-hot分组来计算
        # 暂时返回0
        return torch.tensor(0.0, device=categorical_data.device)


class GumbelSoftmaxOneHotDecoder:
    """使用Gumbel Softmax的可微分one-hot解码器"""
    
    def __init__(self, encoder, temperature=1.0):
        self.encoder = encoder
        self.temperature = temperature
    
    def decode_onehot_differentiable(self, onehot_logits):
        """
        可微分的one-hot解码 (训练时使用)
        Args:
            onehot_logits: [..., num_classes] 未归一化的logits
        Returns:
            soft_onehot: [..., num_classes] 软one-hot向量
        """
        # 应用Gumbel Softmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(onehot_logits) + 1e-8) + 1e-8)
        logits = (onehot_logits + gumbel_noise) / self.temperature
        
        return F.softmax(logits, dim=-1)
    
    def decode_onehot_hard(self, onehot_vector):
        """
        硬one-hot解码 (推理时使用)
        Args:
            onehot_vector: [..., num_classes] one-hot向量
        Returns:
            class_indices: [...] 类别索引
        """
        return torch.argmax(onehot_vector, dim=-1)


class PopulationOneHotSampler:
    """基于One-Hot编码的人口数据采样器"""
    
    def __init__(self, trainer, encoder, decoder):
        self.trainer = trainer
        self.encoder = encoder
        self.decoder = decoder
    
    def generate_population(self, num_families, decode_to_original=True, post_process=True):
        """
        生成新的人口数据 (one-hot版本)
        Args:
            num_families: 要生成的家庭数量
            decode_to_original: 是否解码到原始类别格式
            post_process: 是否进行后处理优化
        Returns:
            DataFrame格式的家庭和个人数据
        """
        # 1. 从噪声采样
        family_samples, person_samples = self.trainer.sample(num_families)
        
        if not decode_to_original:
            return family_samples, person_samples
        
        # 2. one-hot后处理 (可选)
        if post_process:
            family_samples = self._post_process_onehot(family_samples, is_family=True)
            person_samples = self._post_process_onehot(person_samples, is_family=False)
        
        # 3. 解码到原始格式
        decoded_families = []
        decoded_persons = []
        
        for i in range(num_families):
            family_data = family_samples[i]
            person_data = person_samples[i]
            
            # 解码单个家庭
            decoded_family, decoded_family_persons = self.decoder.decode_sample(
                family_data, person_data
            )
            
            # 添加家庭ID
            decoded_family['家庭编号'] = f'generated_{i:06d}'
            decoded_families.append(decoded_family)
            
            # 处理家庭成员
            for person_idx, decoded_person in enumerate(decoded_family_persons):
                decoded_person['家庭编号'] = f'generated_{i:06d}'
                decoded_person['成员编号'] = person_idx
                decoded_persons.append(decoded_person)
        
        # 4. 转换为DataFrame
        import pandas as pd
        family_df = pd.DataFrame(decoded_families)
        person_df = pd.DataFrame(decoded_persons)
        
        return family_df, person_df
    
    def _post_process_onehot(self, samples, is_family=True):
        """
        后处理one-hot向量，使其更接近真实的one-hot分布
        """
        processed_samples = samples.clone()
        
        if is_family:
            # 处理家庭one-hot变量
            idx = self.encoder.family_continuous_dim
            for dim in self.encoder.family_categorical_dims:
                # 对每个one-hot组进行softmax归一化
                onehot_group = processed_samples[:, idx:idx + dim]
                processed_samples[:, idx:idx + dim] = F.softmax(onehot_group, dim=1)
                idx += dim
        else:
            # 处理个人one-hot变量
            batch_size, max_family_size, feature_dim = processed_samples.shape
            
            for family_idx in range(max_family_size):
                idx = 1  # 跳过年龄
                for dim in self.encoder.person_categorical_dims:
                    # 对每个one-hot组进行softmax归一化
                    onehot_group = processed_samples[:, family_idx, idx:idx + dim]
                    processed_samples[:, family_idx, idx:idx + dim] = F.softmax(onehot_group, dim=1)
                    idx += dim
        
        return processed_samples


# 使用示例
if __name__ == "__main__":
    print("基于One-Hot编码的人口数据解码器已创建!")
    print("主要改进:")
    print("1. 使用one-hot编码替代embedding")
    print("2. 直接对one-hot向量进行线性投影")
    print("3. 通过argmax进行硬解码")
    print("4. 支持软解码和后处理优化")
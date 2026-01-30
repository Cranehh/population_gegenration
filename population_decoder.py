import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PopulationDecoder:
    """人口数据解码器：从连续embedding恢复到离散类别"""
    
    def __init__(self, encoder, model):
        """
        Args:
            encoder: PopulationDataEncoder实例
            model: 训练好的PopulationDiT模型
        """
        self.encoder = encoder
        self.model = model
        
        # 提取embedding权重作为类别原型
        self._extract_category_prototypes()
    
    def _extract_category_prototypes(self):
        """提取embedding层的权重作为类别原型"""
        self.family_prototypes = []
        self.person_prototypes = []
        
        # 家庭离散变量原型
        for i, embedding in enumerate(self.model.family_categorical_embeddings):
            # embedding.weight: [num_classes, embedding_dim]
            self.family_prototypes.append(embedding.weight.data.clone())
        
        # 个人离散变量原型
        for i, embedding in enumerate(self.model.person_categorical_embeddings):
            self.person_prototypes.append(embedding.weight.data.clone())
    
    def decode_sample(self, family_sample, person_sample):
        """
        解码单个样本
        Args:
            family_sample: [family_dim] 家庭数据
            person_sample: [max_family_size, person_dim] 个人数据
        Returns:
            decoded_family: 解码后的家庭数据
            decoded_persons: 解码后的个人数据
        """
        device = family_sample.device
        
        # 解码家庭数据
        decoded_family = self._decode_family_data(family_sample, device)
        
        # 解码个人数据
        decoded_persons = []
        for person_idx in range(person_sample.shape[0]):
            person_data = person_sample[person_idx]
            decoded_person = self._decode_person_data(person_data, device)
            decoded_persons.append(decoded_person)
        
        return decoded_family, decoded_persons
    
    def _decode_family_data(self, family_sample, device):
        """解码家庭数据"""
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
        
        # 离散变量：从embedding恢复到类别
        categorical_start = continuous_end
        categorical_end = categorical_start + len(self.encoder.family_categorical_cols)
        categorical_data = family_sample[categorical_start:categorical_end]
        
        # 通过模型的前向传播获取embedding
        with torch.no_grad():
            categorical_data_int = categorical_data.long().unsqueeze(0)  # [1, num_categories]
            embeddings = []
            for i, embedding_layer in enumerate(self.model.family_categorical_embeddings):
                embed = embedding_layer(categorical_data_int[:, i])  # [1, embedding_dim]
                embeddings.append(embed)
        
        # 通过最近邻搜索恢复类别
        for i, col in enumerate(self.encoder.family_categorical_cols):
            embed = embeddings[i].squeeze(0)  # [embedding_dim]
            prototype = self.family_prototypes[i].to(device)  # [num_classes, embedding_dim]
            
            # 计算与所有类别原型的相似度
            similarities = F.cosine_similarity(embed.unsqueeze(0), prototype, dim=1)
            predicted_class = torch.argmax(similarities).item()
            
            # 反编码到原始类别
            le = self.encoder.label_encoders[f'family_{col}']
            original_category = le.inverse_transform([predicted_class])[0]
            decoded[col] = original_category
        
        return decoded
    
    def _decode_person_data(self, person_sample, device):
        """解码个人数据"""
        decoded = {}
        
        # 连续变量：年龄
        age_normalized = person_sample[0].item()  # 已经在[-1,1]范围
        age_01 = (age_normalized + 1) / 2  # 转换到[0,1]
        age_original = age_01 * 123  # 恢复到实际年龄
        decoded['age'] = max(0, min(123, round(age_original)))
        
        # 离散变量
        categorical_data = person_sample[1:].long().unsqueeze(0)  # [1, num_categories]
        
        # 获取embedding
        with torch.no_grad():
            embeddings = []
            for i, embedding_layer in enumerate(self.model.person_categorical_embeddings):
                embed = embedding_layer(categorical_data[:, i])  # [1, embedding_dim]
                embeddings.append(embed)
        
        # 恢复类别
        for i, col in enumerate(self.encoder.person_categorical_cols):
            embed = embeddings[i].squeeze(0)  # [embedding_dim]
            prototype = self.person_prototypes[i].to(device)  # [num_classes, embedding_dim]
            
            # 最近邻搜索
            similarities = F.cosine_similarity(embed.unsqueeze(0), prototype, dim=1)
            predicted_class = torch.argmax(similarities).item()
            
            # 反编码
            le = self.encoder.label_encoders[f'person_{col}']
            original_category = le.inverse_transform([predicted_class])[0]
            decoded[col] = original_category
        
        return decoded


class GumbelSoftmaxDecoder:
    """使用Gumbel Softmax的可微分解码器"""
    
    def __init__(self, encoder, temperature=1.0):
        self.encoder = encoder
        self.temperature = temperature
    
    def decode_categorical_differentiable(self, embedded_values, category_prototypes):
        """
        可微分的类别解码（训练时使用）
        Args:
            embedded_values: [batch_size, embedding_dim] 
            category_prototypes: [num_classes, embedding_dim]
        Returns:
            soft_assignment: [batch_size, num_classes] 软分配
        """
        # 计算相似度得分
        similarities = torch.mm(embedded_values, category_prototypes.T)  # [batch_size, num_classes]
        
        # 应用Gumbel Softmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarities) + 1e-8) + 1e-8)
        logits = (similarities + gumbel_noise) / self.temperature
        
        return F.softmax(logits, dim=-1)
    
    def decode_categorical_hard(self, embedded_values, category_prototypes):
        """
        硬解码（推理时使用）
        Args:
            embedded_values: [batch_size, embedding_dim]
            category_prototypes: [num_classes, embedding_dim]
        Returns:
            hard_assignment: [batch_size] 类别索引
        """
        similarities = torch.mm(embedded_values, category_prototypes.T)
        return torch.argmax(similarities, dim=-1)


class PopulationSampler:
    """完整的人口数据采样器"""
    
    def __init__(self, trainer, encoder, decoder):
        self.trainer = trainer
        self.encoder = encoder
        self.decoder = decoder
    
    def generate_population(self, num_families, decode_to_original=True):
        """
        生成新的人口数据
        Args:
            num_families: 要生成的家庭数量
            decode_to_original: 是否解码到原始类别格式
        Returns:
            如果decode_to_original=True，返回DataFrame格式
            否则返回tensor格式
        """
        # 1. 从噪声采样
        family_samples, person_samples = self.trainer.sample(num_families)
        
        if not decode_to_original:
            return family_samples, person_samples
        
        # 2. 解码到原始格式
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
                if self._is_valid_person(decoded_person):  # 过滤无效成员
                    decoded_person['家庭编号'] = f'generated_{i:06d}'
                    decoded_person['成员编号'] = person_idx
                    decoded_persons.append(decoded_person)
        
        # 3. 转换为DataFrame
        import pandas as pd
        family_df = pd.DataFrame(decoded_families)
        person_df = pd.DataFrame(decoded_persons)
        
        return family_df, person_df
    
    def _is_valid_person(self, person_data):
        """判断生成的人员数据是否有效"""
        # 基本有效性检查
        if person_data['age'] < 0 or person_data['age'] > 120:
            return False
        
        # 可以添加更多逻辑检查
        # 例如：学历与年龄的合理性、职业与年龄的匹配等
        
        return True


# 训练时的损失函数改进
class HybridLoss(nn.Module):
    """混合损失：连续变量MSE + 离散变量对比损失"""
    
    def __init__(self, encoder, model, continuous_weight=1.0, categorical_weight=1.0):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.continuous_weight = continuous_weight
        self.categorical_weight = categorical_weight
        
        # 提取embedding原型
        self.family_prototypes = [emb.weight for emb in model.family_categorical_embeddings]
        self.person_prototypes = [emb.weight for emb in model.person_categorical_embeddings]
    
    def forward(self, predicted, target, variable_masks):
        """
        Args:
            predicted: 模型预测的噪声
            target: 真实噪声
            variable_masks: (continuous_mask, categorical_mask)
        """
        continuous_mask, categorical_mask = variable_masks
        
        # 连续变量MSE损失
        continuous_loss = F.mse_loss(
            predicted * continuous_mask,
            target * continuous_mask,
            reduction='mean'
        )
        
        # 离散变量embedding对比损失
        categorical_loss = F.mse_loss(
            predicted * categorical_mask,
            target * categorical_mask,
            reduction='mean'
        )
        
        total_loss = (self.continuous_weight * continuous_loss + 
                     self.categorical_weight * categorical_loss)
        
        return total_loss, continuous_loss, categorical_loss


# 使用示例
def create_complete_pipeline(family_df, person_df):
    """创建完整的训练和采样管道"""
    from population_data_process import PopulationDataEncoder, create_population_dataset
    from population_DiT import PopulationDiT
    from population_diffusion import PopulationDiffusionTrainer, PopulationDiffusionProcess
    
    # 1. 数据编码
    encoder = PopulationDataEncoder()
    encoder.fit_family_data(family_df)
    encoder.fit_person_data(person_df)
    
    family_data, person_data = create_population_dataset(family_df, person_df, encoder)
    
    # 2. 模型和训练器
    diffusion = PopulationDiffusionProcess(
        family_categorical_dims=encoder.family_categorical_dims,
        person_categorical_dims=encoder.person_categorical_dims
    )
    
    model = PopulationDiT(
        family_categorical_dims=encoder.family_categorical_dims,
        person_categorical_dims=encoder.person_categorical_dims
    )
    
    trainer = PopulationDiffusionTrainer(model, diffusion)
    
    # 3. 解码器和采样器
    decoder = PopulationDecoder(encoder, model)
    sampler = PopulationSampler(trainer, encoder, decoder)
    
    return trainer, sampler, encoder


if __name__ == "__main__":
    print("人口数据解码器已创建!")
    print("主要功能:")
    print("1. 从连续embedding恢复到离散类别")
    print("2. 支持可微分和硬解码两种模式")
    print("3. 完整的采样和解码流程")
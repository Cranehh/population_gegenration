## 所有信息重新分组
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class PopulationDataEncoder:
    """人口数据编码器，适配Diffusion Transformer"""
    
    def __init__(self):
        # 家庭连续变量
        self.family_continuous_cols = [
            '家庭成员数量', '家庭工作人口数', '机动车数量', 
            '脚踏自行车数量', '电动自行车数量', '摩托车数量', '老年代步车数量', '家庭年收入'
        ]
        
        # 家庭离散变量
        self.family_categorical_cols = ['have_student']
        self.family_cluster_col = ['cluster']
        
        # 个人连续变量 (已归一化的年龄)
        self.person_continuous_cols = ['age']
        
        # 个人离散变量
        self.person_categorical_cols = ['性别', '是否有驾照', '关系', '最高学历', '职业']
        
        # 编码器
        self.scalers = {}
        self.onehot_encoders = {}
        
        # 变量维度信息
        self.family_continuous_dim = len(self.family_continuous_cols)
        self.family_categorical_dims = []
        self.person_continuous_dim = len(self.person_continuous_cols) 
        self.person_categorical_dims = []
    
    def fit_family_data(self, family_df):
        """拟合家庭数据编码器"""
        
        # 1. 连续变量标准化
        for col in self.family_continuous_cols:
            scaler = StandardScaler()
            scaler.fit(family_df[[col]].astype(float))
            self.scalers[f'family_{col}'] = scaler
        
        # 2. 离散变量one-hot编码
        for col in self.family_categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            data_to_fit = family_df[[col]].fillna('unknown')
            ohe.fit(data_to_fit)
            self.onehot_encoders[f'family_{col}'] = ohe
            self.family_categorical_dims.append(len(ohe.categories_[0]))
    
    def fit_person_data(self, person_df):
        """拟合个人数据编码器"""
        
        # 1. 连续变量 (年龄已经归一化，直接使用)
        # age已经在[0,1]范围内，转换到[-1,1]
        col = 'age'
        scaler = StandardScaler()
        scaler.fit(person_df[[col]].astype(float))
        self.scalers[f'person_{col}'] = scaler
        
        # 2. 离散变量one-hot编码
        for col in self.person_categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # 处理缺失值
            data_to_fit = person_df[[col]].fillna('-1').astype(int)
            ohe.fit(data_to_fit)
            self.onehot_encoders[f'person_{col}'] = ohe
            self.person_categorical_dims.append(len(ohe.categories_[0]))
    
    def encode_family(self, family_df):
        """编码家庭数据"""
        encoded_data = {}
        
        # 连续变量：标准化后映射到[-1,1]
        for col in self.family_continuous_cols:
            scaled = self.scalers[f'family_{col}'].transform(family_df[[col]].astype(float))
            # 限制到3个标准差内，然后映射到[-1,1]
            # clipped = np.clip(scaled.flatten(), -3, 3) / 3
            encoded_data[f'family_{col}'] = scaled.flatten()
        
        # 离散变量：one-hot编码
        for col in self.family_categorical_cols:
            data_to_encode = family_df[[col]].fillna('unknown')
            encoded = self.onehot_encoders[f'family_{col}'].transform(data_to_encode)
            # encoded是二维数组，每行是一个样本的one-hot向量
            encoded_data[f'family_{col}'] = encoded

        encoded_data['family_cluster'] = family_df['cluster'].astype(int).values
            
        return encoded_data
    
    def encode_person(self, person_df):
        """编码个人数据"""
        encoded_data = {}
        
        # 连续变量：年龄从[0,1]转换到[-1,1]
        encoded_data['person_age'] = self.scalers[f'person_age'].transform(person_df[['age']].astype(float)).flatten()
        
        # 离散变量：one-hot编码
        for col in self.person_categorical_cols:
            data_to_encode = person_df[[col]].fillna('-1').astype(int)
            encoded = self.onehot_encoders[f'person_{col}'].transform(data_to_encode)
            # encoded是二维数组，每行是一个样本的one-hot向量
            encoded_data[f'person_{col}'] = encoded
            
        return encoded_data
    
    def decode_family_continuous(self, encoded_data):
        """解码家庭连续变量"""
        decoded_data = {}
        
        for col in self.family_continuous_cols:
            # 从[-1,1]映射回标准化值（[-3,3]范围）
            standardized_values = encoded_data[f'family_{col}']
            
            # 使用对应的scaler逆变换回原始值
            scaler = self.scalers[f'family_{col}']
            original_values = scaler.inverse_transform(standardized_values.values.reshape(-1, 1)).flatten()
            
            # 确保非负值（对于计数类变量）
            if col in ['家庭成员数量', '家庭工作人口数', '机动车数量', 
                      '脚踏自行车数量', '电动自行车数量', '摩托车数量', '老年代步车数量', '家庭年收入']:
                original_values = np.maximum(original_values, 0)
                # 对于整数类变量进行四舍五入
                if col in ['家庭成员数量', '家庭工作人口数', '机动车数量', 
                          '脚踏自行车数量', '电动自行车数量', '摩托车数量', '老年代步车数量', '家庭年收入']:
                    original_values = np.round(original_values).astype(int)
            
            decoded_data[col] = original_values
            
        return decoded_data
    
    def decode_person_continuous(self, encoded_age):
        """解码个人连续变量（年龄）"""

        standardized_values = encoded_age
            
        # 使用对应的scaler逆变换回原始值
        scaler = self.scalers[f'person_age']
        original_values = scaler.inverse_transform(standardized_values.values.reshape(-1, 1)).flatten()

        
        # 四舍五入为整数年龄
        actual_age = np.round(original_values).astype(int)
        
        return {
            'age_normalized': encoded_age,
            'age_actual': actual_age
        }
    
    def decode_family_categorical(self, encoded_data):
        """解码家庭离散变量"""
        decoded_data = {}
        
        for col in self.family_categorical_cols:
            # 获取one-hot编码
            onehot_encoded = encoded_data[f'family_{col}']
            
            # 如果是logits，使用softmax
            if onehot_encoded.ndim == 1:
                onehot_encoded = onehot_encoded.reshape(1, -1)
                
            # 使用argmax获取最可能的类别
            predicted_indices = np.argmax(onehot_encoded, axis=1)
            
            # 使用编码器逆变换
            encoder = self.onehot_encoders[f'family_{col}']
            
            # 创建对应的one-hot向量
            decoded_onehot = np.zeros_like(onehot_encoded)
            decoded_onehot[np.arange(len(predicted_indices)), predicted_indices] = 1
            
            # 逆变换为原始类别
            decoded_categories = encoder.inverse_transform(decoded_onehot)
            
            decoded_data[col] = decoded_categories.flatten()
            decoded_data[f'{col}_probs'] = onehot_encoded  # 保留概率分布
            
        return decoded_data
    
    def decode_person_categorical(self, encoded_data):
        """解码个人离散变量"""
        decoded_data = {}
        
        for col in self.person_categorical_cols:
            # 获取one-hot编码
            onehot_encoded = encoded_data[f'person_{col}']
            
            # 如果是logits，使用softmax
            if onehot_encoded.ndim == 1:
                onehot_encoded = onehot_encoded.reshape(1, -1)
                
            # 使用argmax获取最可能的类别
            predicted_indices = np.argmax(onehot_encoded, axis=1)
            
            # 使用编码器逆变换
            encoder = self.onehot_encoders[f'person_{col}']
            
            # 创建对应的one-hot向量
            decoded_onehot = np.zeros_like(onehot_encoded)
            decoded_onehot[np.arange(len(predicted_indices)), predicted_indices] = 1
            
            # 逆变换为原始类别
            decoded_categories = encoder.inverse_transform(decoded_onehot)
            
            decoded_data[col] = decoded_categories.flatten()
            decoded_data[f'{col}_probs'] = onehot_encoded  # 保留概率分布
            
        return decoded_data

    def decode_model_output(self, family_output, person_output, max_family_size=8):
        """
        解码模型输出为原始数据格式
        
        Args:
            family_output: [batch_size, family_feature_dim] 模型输出的家庭特征
            person_output: [batch_size, max_family_size, person_feature_dim] 模型输出的个人特征
            max_family_size: 最大家庭大小
            
        Returns:
            dict: 包含解码后的家庭和个人数据
        """
        batch_size = family_output.shape[0]
        
        # 解析家庭特征维度
        family_continuous_end = self.family_continuous_dim
        family_categorical_start = family_continuous_end
        
        # 解码家庭连续变量
        family_continuous_data = {}
        for i, col in enumerate(self.family_continuous_cols):
            family_continuous_data[f'family_{col}'] = family_output[:, i]
        
        family_continuous_decoded = self.decode_family_continuous(family_continuous_data)
        
        # 解码家庭离散变量
        family_categorical_data = {}
        current_idx = family_categorical_start
        for i, col in enumerate(self.family_categorical_cols):
            dim = self.family_categorical_dims[i]
            family_categorical_data[f'family_{col}'] = family_output[:, current_idx:current_idx+dim]
            current_idx += dim
        
        family_categorical_decoded = self.decode_family_categorical(family_categorical_data)
        
        # 解码个人特征
        person_continuous_end = self.person_continuous_dim
        person_categorical_start = person_continuous_end
        
        decoded_families = []
        
        for batch_idx in range(batch_size):
            family_data = {
                'family_id': f'generated_{batch_idx}',
                **family_continuous_decoded,
                **family_categorical_decoded
            }
            
            # 提取该batch的个人数据
            person_batch = person_output[batch_idx]  # [max_family_size, person_feature_dim]
            
            members = []
            for member_idx in range(max_family_size):
                member_features = person_batch[member_idx]
                
                # 解码个人连续变量（年龄）
                age_data = self.decode_person_continuous(member_features[0].numpy())
                
                # 解码个人离散变量
                person_categorical_data = {}
                current_idx = person_continuous_end
                for i, col in enumerate(self.person_categorical_cols):
                    dim = self.person_categorical_dims[i]
                    person_categorical_data[f'person_{col}'] = member_features[current_idx:current_idx+dim].unsqueeze(0).numpy()
                    current_idx += dim
                
                person_categorical_decoded = self.decode_person_categorical(person_categorical_data)
                
                # 检查是否为有效成员（非零填充）
                if not torch.all(member_features == 0):
                    member_data = {
                        'member_id': member_idx,
                        **age_data,
                        **person_categorical_decoded
                    }
                    members.append(member_data)
            
            family_data['members'] = members
            family_data['actual_family_size'] = len(members)
            decoded_families.append(family_data)
        
        return decoded_families

    def tensor_to_dataframe(self, decoded_families):
        """
        将解码后的数据转换为DataFrame格式
        
        Args:
            decoded_families: decode_model_output的输出
            
        Returns:
            tuple: (family_df, person_df) 家庭和个人数据框
        """
        family_rows = []
        person_rows = []
        
        for family in decoded_families:
            # 家庭数据行
            family_row = {
                '家庭编号': family['family_id'],
                '实际家庭人数': family['actual_family_size']
            }
            
            # 添加家庭连续变量
            for col in self.family_continuous_cols:
                if col in family:
                    family_row[col] = family[col][0] if isinstance(family[col], np.ndarray) else family[col]
            
            # 添加家庭离散变量
            for col in self.family_categorical_cols:
                if col in family:
                    family_row[col] = family[col][0] if isinstance(family[col], np.ndarray) else family[col]
            
            family_rows.append(family_row)
            
            # 个人数据行
            for member in family['members']:
                person_row = {
                    '家庭编号': family['family_id'],
                    '成员编号': member['member_id'],
                    '年龄_归一化': member['age_normalized'],
                    '年龄_实际': member['age_actual']
                }
                
                # 添加个人离散变量
                for col in self.person_categorical_cols:
                    if col in member:
                        person_row[col] = member[col][0] if isinstance(member[col], np.ndarray) else member[col]
                
                person_rows.append(person_row)
        
        family_df = pd.DataFrame(family_rows)
        person_df = pd.DataFrame(person_rows)
        
        return family_df, person_df


def create_population_dataset(family_df, person_df, encoder, max_family_size=8):
    """创建用于训练的人口数据集"""
    
    # 编码家庭数据
    family_encoded = encoder.encode_family(family_df)
    person_encoded = encoder.encode_person(person_df)
    
    # 构建训练样本
    family_samples = []
    member_samples = []
    family_ids = family_df['家庭编号'].unique()
    
    for family_id in family_ids:
        # 家庭信息
        family_idx = family_df[family_df['家庭编号'] == family_id].index[0]
        family_features = []
        
        # 家庭连续变量
        for col in encoder.family_continuous_cols:
            family_features.append(family_encoded[f'family_{col}'][family_idx])
        
        # 家庭离散变量 (one-hot编码后是向量)
        for col in encoder.family_categorical_cols:
            onehot_vector = family_encoded[f'family_{col}'][family_idx]
            family_features.extend(onehot_vector.tolist())
        family_features.extend([family_encoded['family_cluster'][family_idx]])  # 添加聚类标签
        # 成员信息
        family_members = person_df[person_df['家庭编号'] == family_id]
        family_members = family_members.sort_values('age', ascending=False)
        member_features = []
        
        for _, member in family_members.iterrows():
            member_feature = []
            # 个人连续变量
            member_feature.append(person_encoded['person_age'][member.name])
            # 个人离散变量 (one-hot编码后是向量)
            for col in encoder.person_categorical_cols:
                onehot_vector = person_encoded[f'person_{col}'][member.name]
                member_feature.extend(onehot_vector.tolist())
            member_feature.append(1) # 标记为有效成员
            member_features.extend(member_feature)
        
        # 填充到最大家庭大小
        current_members = len(family_members)
        # 计算每个成员的特征维度：1个连续变量 + 所有离散变量的one-hot维度之和
        person_feature_dim = 1 + sum(encoder.person_categorical_dims)
        
        if current_members < max_family_size:
            # 用0填充缺失成员
            padding_size = (max_family_size - current_members) * (person_feature_dim + 1) # 标记无效成员
            member_features.extend([0] * padding_size)
        elif current_members > max_family_size:
            # 截断超出的成员
            member_features = member_features[:max_family_size * person_feature_dim]
        
        # 组合完整样本
        family_samples.append(family_features)
        member_samples.append(member_features)
    
    # 计算正确的reshape维度
    person_feature_dim = 1 + sum(encoder.person_categorical_dims) + 1  # +1 for valid member flag
    
    return (torch.tensor(family_samples, dtype=torch.float32), 
            torch.tensor(member_samples, dtype=torch.float32).view(len(member_samples), max_family_size, person_feature_dim))


# 使用示例
if __name__ == "__main__":
    # 1. 初始化编码器
    encoder = PopulationDataEncoder()
    
    # 2. 拟合数据 (需要你的实际数据)
    # encoder.fit_family_data(family2023)
    # encoder.fit_person_data(familymember_2023)
    
    # 3. 创建数据集
    # dataset = create_population_dataset(family2023, familymember_2023, encoder)
    
    # 4. 初始化模型
    model = PopulationDiT(
        family_categorical_dims=[2, 8],  # 根据实际类别数调整
        person_categorical_dims=[2, 3, 10, 8, 16]  # 根据实际类别数调整
    )
    
    print("人口合成DiT模型已初始化")
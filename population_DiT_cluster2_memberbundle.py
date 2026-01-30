## 改了噪声传播，改了person内部的信息传播
## 加了对个人生成的cluster的条件输入
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
# Add the DiT-main directory to Python path if it's not already in the path
sys.path.append('DiT-main')
# Now import DiTBlock from models
from models import DiTBlock, DiTBlockPerson
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import sys

sys.path.append('GraphVAE-master')
from graph_vae.graph_datastructure import *
from graph_vae.graph_vae_model import *
# from hetGraph import *
from hetGraph_optimized import *


diffusion = create_diffusion(timestep_respacing="")
# 为each离散特征添加Gumbel噪声
def add_gumbel_noise(one_hot_data, temperature=1.0):
    # 转换到logits空间
    logits = torch.log(one_hot_data + 1e-8)
    # 生成Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    # Gumbel-Softmax
    return F.gumbel_softmax(logits + gumbel_noise, tau=temperature, hard=False)

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class PopulationDiT(nn.Module):
    """适配人口数据的Diffusion Transformer"""
    
    def __init__(self, 
                 family_continuous_dim=7,
                 family_categorical_dims=[2, 10],  # have_student, 家庭年收入类别数
                 person_continuous_dim=1, 
                 person_categorical_dims=[2, 2, 16, 9, 20],  # 性别,驾照,关系,学历,职业
                 max_family_size=8,  # 最大家庭人数
                 proj_dim=24,
                 hidden_size=512,
                 num_classes=10,
                 class_dropout_prob=0.1,
                 depth=12,
                 num_heads=8):
        
        super().__init__()

        self.family_have_student_proj = nn.Linear(2, proj_dim)
        self.family_income_proj = nn.Linear(10, proj_dim)

        self.person_gender_proj = nn.Linear(2, proj_dim)
        self.person_license_proj = nn.Linear(2, proj_dim)
        self.person_relation_proj = nn.Linear(16, proj_dim)
        self.person_education_proj = nn.Linear(9, proj_dim)
        self.person_occupation_proj = nn.Linear(20, proj_dim)
        
        self.family_continuous_dim = family_continuous_dim
        self.family_categorical_dims = family_categorical_dims
        self.person_continuous_dim = person_continuous_dim
        self.person_categorical_dims = person_categorical_dims
        self.max_family_size = max_family_size
        self.hidden_size = hidden_size
        
        # 家庭变量投影
        self.family_continuous_proj = nn.Linear(7 + sum(family_categorical_dims), hidden_size)
        
        ## 家庭关系图生成
        self.relation_graph_decoder = Decoder(8, 7 + sum(family_categorical_dims), True)
        ## 家庭关系图处理
        self.hgt_model = OptimizedDifferentiableHGT(
            in_dim=hidden_size,  # 输入特征维度
            hidden_dim=hidden_size,  # 隐藏层维度
            out_dim=hidden_size,  # 输出维度
            num_node_types=6,  # 节点类型数量 (Male_Child, Female_Child, etc.)
            num_relations=5,  # 关系类型数量 (SPOUSE, PARENT_CHILD, etc.)
            n_heads=4,  # 注意力头数
            n_layers=2,  # HGT层数
            dropout=0.2,  # Dropout率
            use_norm=True  # 使用层归一化
        )
        ## 家庭关系图嵌入
        self.node_embedding_layer = nn.Linear(self.max_family_size * hidden_size, hidden_size)

        ## 家庭条件信息投影
        self.family_condition_proj = nn.Linear(7 + sum(family_categorical_dims), hidden_size)

        ## FinalLayer
        self.finallayer_family = FinalLayer(self.hidden_size, 7 + sum(family_categorical_dims))

        self.finallayer = FinalLayer(self.hidden_size, self.hidden_size)
        self.finallayer2 = FinalLayer(self.hidden_size, 1 + sum(person_categorical_dims) + 1)
        # 个人变量投影
        self.person_continuous_proj = nn.Linear(1 + sum(person_categorical_dims) + 1, hidden_size)
        
        # 位置嵌入：1个家庭位置 + max_family_size个人员位置
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + max_family_size, hidden_size)
        )
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.person_time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        

        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.y_embedder_person = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        # Transformer blocks (使用原DiT的DiTBlock)
        self.family_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.person_blocks = nn.ModuleList([
            DiTBlockPerson(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 输出层
        self.output_norm_family = nn.LayerNorm(hidden_size)
        self.output_proj_family = nn.Linear(hidden_size,
                                    7 + sum(family_categorical_dims))
        
        self.family_have_student_class = nn.Softmax(dim=-1)
        self.family_income_class = nn.Softmax(dim=-1)


        self.person_gender_class = nn.Softmax(dim=-1)
        self.person_license_class = nn.Softmax(dim=-1)
        self.person_relation_class = nn.Softmax(dim=-1)
        self.person_education_class = nn.Softmax(dim=-1)
        self.person_occupation_class = nn.Softmax(dim=-1)

        self.person_isTrue = nn.Sigmoid()
        
    
    def get_embedding_data(self, family_data, person_data):
        x = family_data
        x_person = person_data
        # 家庭级别的处理
        
        # 解析输入序列 (考虑one-hot编码后的维度)
        family_continuous = x[:, :self.family_continuous_dim]
        

        family_have_student_start = self.family_continuous_dim
        family_have_student = x[:, family_have_student_start:family_have_student_start + self.family_categorical_dims[0]]

        family_income_start = family_have_student_start + self.family_categorical_dims[0]
        family_income = x[:, family_income_start:family_income_start + self.family_categorical_dims[1]]

        family_have_student_embed = self.family_have_student_proj(family_have_student)
        family_income_embed = self.family_income_proj(family_income)
        
        # 个人级别的处理
        person_mask = x_person.abs().sum(dim=-1) > 0  # [batch_size, max_family_size]
        person_continuous = x_person[:, :, :self.person_continuous_dim]

        person_gender_start = self.person_continuous_dim
        person_gender = x_person[:, :, person_gender_start:person_gender_start + self.person_categorical_dims[0]]

        person_license_start = person_gender_start + self.person_categorical_dims[0]
        person_license = x_person[:, :, person_license_start:person_license_start + self.person_categorical_dims[1]]

        person_relation_start = person_license_start + self.person_categorical_dims[1]
        person_relation = x_person[:, :, person_relation_start:person_relation_start + self.person_categorical_dims[2]]

        person_education_start = person_relation_start + self.person_categorical_dims[2]
        person_education = x_person[:, :, person_education_start:person_education_start + self.person_categorical_dims[3]]

        person_occupation_start = person_education_start + self.person_categorical_dims[3]
        person_occupation = x_person[:, :, person_occupation_start:person_occupation_start + self.person_categorical_dims[4]]

        person_gender_embed = self.person_gender_proj(person_gender)
        person_license_embed = self.person_license_proj(person_license)
        person_relation_embed = self.person_relation_proj(person_relation)
        person_education_embed = self.person_education_proj(person_education)
        person_occupation_embed = self.person_occupation_proj(person_occupation)

        faminly_embedding = torch.concat([family_continuous, family_have_student_embed, family_income_embed], dim=-1)
        person_embedding = torch.concat([person_continuous, person_gender_embed, person_license_embed, 
                                        person_relation_embed, person_education_embed, person_occupation_embed], dim=-1)
        
        return faminly_embedding, person_embedding, person_mask
    
    
    def forward(self, family_data, person_data, cluster, t, t_person):
        """
        x: [batch_size, sequence_length] 编码后的人口数据 (one-hot格式)
        t: [batch_size] 时间步
        family_mask: [batch_size, max_family_size] 家庭成员掩码
        """
        ## TODO：1.成员的mask； 2.离散噪声的加入
        # family_embedding, person_embedding, person_mask = self.get_embedding_data(family_data, person_data)
        
        # 加入噪声

        # noise_family_continus = torch.randn_like(family_data)
        # family_embedding_t = diffusion.q_sample(family_data, t, noise=noise_family_continus)

        # noise_person = torch.randn_like(person_data)
        # person_embedding_t = diffusion.q_sample(person_data, t, noise=noise_person)

        # 准备输入
        family_final_out = self.forward_family(family_data, cluster, t)
        self.relation_graph_decoder.update(family_final_out)

        hgt_data = create_differentiable_hgt_data(
            self.relation_graph_decoder,
            family_features=None,
            temperature=0.5,  # 控制软化程度
            hard=False  # 训练时使用软分布
        )

        # 提取家庭成员关系图特征
        node_embeddings = self.hgt_model(hgt_data)
        node_embeddings = self.node_embedding_layer(node_embeddings.view(node_embeddings.shape[0],-1))

        person_final_out = self.forward_person(person_data, node_embeddings, family_final_out, t_person, None, cluster)

        
        return family_final_out, person_final_out, hgt_data

    def forward_family(self, family_embedding_t, cluster, t_family):
        # 编码家庭信息
        family_cont_embed = self.family_continuous_proj(family_embedding_t)
        family_cont_embed = family_cont_embed.unsqueeze(1)  # [b, 1, h]
        
        # 家庭信息嵌入
        cluster_embed = self.y_embedder(cluster, self.training)
        # 时间步嵌入
        t_embed = self.time_embed(self.timestep_embedding(t_family, self.hidden_size))

        c = t_embed + cluster_embed
        # Transformer处理
        for block in self.family_blocks:
            sequence = block(family_cont_embed, c)
        
        # 输出
        sequence = self.finallayer_family(sequence, c)
        output = sequence.flatten(start_dim=1)
        ## softmax分类输出
        family_have_student_out = self.family_have_student_class(output[:, self.family_continuous_dim:self.family_continuous_dim + self.family_categorical_dims[0]])
        family_income_out = self.family_income_class(output[:, self.family_continuous_dim + self.family_categorical_dims[0]:self.family_continuous_dim + sum(self.family_categorical_dims)])

        family_final_out = torch.cat([output[:, :self.family_continuous_dim], 
                                      family_have_student_out, 
                                      family_income_out], dim=-1)
        
        return family_final_out

    def forward_person(self, person_embedding_t, node_information, family_feature, t_person, person_mask, family_cluster):
        # 编码个人信息
        person_cont_embed = self.person_continuous_proj(person_embedding_t)
        
        # 时间步嵌入
        t_embed = self.person_time_embed(self.timestep_embedding(t_person, self.hidden_size))

        cluster_embed = self.y_embedder_person(family_cluster, self.training)
        cluster_embed = cluster_embed + t_embed
        node_information = node_information + t_embed
        family_feature = self.family_condition_proj(family_feature) + t_embed
        family_feature = family_feature.unsqueeze(1)
        # Transformer处理
        for block in self.person_blocks:
            sequence = block(person_cont_embed, family_feature, node_information)

        family_feature = family_feature.squeeze(1)
        # 输出
        output = self.finallayer(sequence, family_feature)
        output = self.finallayer2(output, cluster_embed)

        ## softmax分类输出
        person_gender_out = self.person_gender_class(output[:, :, self.person_continuous_dim:self.person_continuous_dim + self.person_categorical_dims[0]])
        person_license_out = self.person_license_class(output[:, :, self.person_continuous_dim + self.person_categorical_dims[0]:self.person_continuous_dim + sum(self.person_categorical_dims[:2])])
        person_relation_out = self.person_relation_class(output[:, :, self.person_continuous_dim + sum(self.person_categorical_dims[:2]):self.person_continuous_dim + sum(self.person_categorical_dims[:3])])
        person_education_out = self.person_education_class(output[:, :, self.person_continuous_dim + sum(self.person_categorical_dims[:3]):self.person_continuous_dim + sum(self.person_categorical_dims[:4])])
        person_occupation_out = self.person_occupation_class(output[:, :, self.person_continuous_dim + sum(self.person_categorical_dims[:4]):self.person_continuous_dim + sum(self.person_categorical_dims)])

        ## 生成的人是否为真的判断
        person_isTrue_out = self.person_isTrue(output[:, :, -1:])

        person_final_out = torch.cat([output[:, :, :self.person_continuous_dim],
                                      person_gender_out,
                                      person_license_out,
                                      person_relation_out,
                                      person_education_out,
                                      person_occupation_out,
                                      person_isTrue_out], dim=-1)

        return person_final_out
    
        
    
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=1000):
        """时间步正弦嵌入"""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.device)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape

        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x

def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


def load_population_dit_checkpoint(checkpoint_path, device='cuda'):
    """
    从检查点文件加载PopulationDiT模型
    
    Args:
        checkpoint_path: 检查点文件路径 (.pt文件)
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        dict: 包含模型、EMA模型、优化器等信息的字典
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点中获取参数
    args = checkpoint.get('args', None)
    if args is None:
        raise ValueError("Checkpoint does not contain 'args'. Cannot reconstruct model.")
    
    # 提取模型参数
    family_continuous_dim = getattr(args, 'family_continuous_dim', 7)
    family_categorical_dims = getattr(args, 'family_categorical_dims', [2, 10])
    person_continuous_dim = getattr(args, 'person_continuous_dim', 1)
    person_categorical_dims = getattr(args, 'person_categorical_dims', [2, 2, 16, 9, 20])
    max_family_size = getattr(args, 'max_family_size', 8)
    proj_dim = getattr(args, 'proj_dim', 24)
    hidden_size = getattr(args, 'hidden_dim', 128)
    depth = getattr(args, 'num_layers', 3)
    num_heads = getattr(args, 'num_heads', 8)
    
    print(f"Model parameters:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Depth: {depth}")
    print(f"  - Num heads: {num_heads}")
    print(f"  - Max family size: {max_family_size}")
    
    # 重构模型
    model = PopulationDiT(
        family_continuous_dim=family_continuous_dim,
        family_categorical_dims=family_categorical_dims,
        person_continuous_dim=person_continuous_dim,
        person_categorical_dims=person_categorical_dims,
        max_family_size=max_family_size,
        proj_dim=proj_dim,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model'])
    
    # 创建EMA模型
    ema_model = PopulationDiT(
        family_continuous_dim=family_continuous_dim,
        family_categorical_dims=family_categorical_dims,
        person_continuous_dim=person_continuous_dim,
        person_categorical_dims=person_categorical_dims,
        max_family_size=max_family_size,
        proj_dim=proj_dim,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads
    ).to(device)
    
    # 加载EMA权重
    if 'ema' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema'])
        print("✅ EMA model loaded successfully")
    else:
        print("⚠️ No EMA weights found, using regular model weights")
        ema_model.load_state_dict(checkpoint['model'])
    
    # 加载训练信息
    train_info = {
        'epoch': checkpoint.get('epoch', 0),
        'train_steps': checkpoint.get('train_steps', 0),
        'args': args
    }
    
    # 加载优化器状态（如果需要继续训练）
    optimizer_state = checkpoint.get('optimizer', None)
    lr_scheduler_state = checkpoint.get('lr_scheduler', None)
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"  - Epoch: {train_info['epoch']}")
    print(f"  - Train steps: {train_info['train_steps']}")
    
    return {
        'model': model,
        'ema_model': ema_model,
        'optimizer_state': optimizer_state,
        'lr_scheduler_state': lr_scheduler_state,
        'train_info': train_info,
        'args': args
    }


def load_model_for_inference(checkpoint_path, use_ema=True, device='cuda'):
    """
    专门用于推理的模型加载函数
    
    Args:
        checkpoint_path: 检查点文件路径
        use_ema: 是否使用EMA模型（推荐用于推理）
        device: 设备
    
    Returns:
        PopulationDiT: 加载好的模型，已设置为eval模式
    """
    checkpoint_data = load_population_dit_checkpoint(checkpoint_path, device)
    
    # 选择使用EMA模型还是普通模型
    if use_ema and checkpoint_data['ema_model'] is not None:
        model = checkpoint_data['ema_model']
        print("🎯 Using EMA model for inference")
    else:
        model = checkpoint_data['model']
        print("🎯 Using regular model for inference")
    
    # 设置为eval模式
    model.eval()
    
    return model


def resume_training_from_checkpoint(checkpoint_path, device='cuda'):
    """
    从检查点恢复训练
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 设备
    
    Returns:
        dict: 包含模型、优化器、调度器等完整训练状态的字典
    """
    checkpoint_data = load_population_dit_checkpoint(checkpoint_path, device)
    
    # 重构优化器
    if checkpoint_data['optimizer_state'] is not None:
        # 需要根据args重新创建优化器
        args = checkpoint_data['args']
        model = checkpoint_data['model']
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(args, 'lr', 1e-4),
            weight_decay=getattr(args, 'weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        
        # 重构学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=getattr(args, 'epochs', 100), 
            eta_min=getattr(args, 'lr', 1e-4) * 0.01
        )
        
        if checkpoint_data['lr_scheduler_state'] is not None:
            lr_scheduler.load_state_dict(checkpoint_data['lr_scheduler_state'])
        
        print("✅ Training state restored successfully!")
        
        return {
            'model': checkpoint_data['model'],
            'ema_model': checkpoint_data['ema_model'],
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'start_epoch': checkpoint_data['train_info']['epoch'],
            'start_step': checkpoint_data['train_info']['train_steps'],
            'args': checkpoint_data['args']
        }
    else:
        print("⚠️ No optimizer state found in checkpoint")
        return checkpoint_data


# 数据读取
import pandas as pd
pd.set_option('display.max_columns', None)
family2014 = pd.read_csv('数据/居民出行数据/2014/family_2014.csv',dtype=str)
travel2014 = pd.read_csv('数据/居民出行数据/2014/midtable_2014.csv',dtype=str)
familymember_2014 = pd.read_csv('数据/居民出行数据/2014/family_member_2014.csv',dtype=str)
family2023 = pd.read_csv('数据/居民出行数据/2023/family_total_33169.csv',dtype=str)
travel2023 = pd.read_csv('数据/居民出行数据/2023/midtable_total_33169.csv',dtype=str)
familymember_2023 = pd.read_csv('数据/居民出行数据/2023/familymember_total_33169.csv',dtype=str)
family_cluster = pd.read_csv('数据/family_cluster.csv',dtype=str)
## 家庭变量筛选
valid_member_number = familymember_2023.groupby('家庭编号').size().rename('家庭成员数量_real').reset_index()
family2023 = pd.merge(family2023, valid_member_number, on='家庭编号', how='left')
family2023 = family2023[family2023['家庭成员数量'].astype(int) == family2023['家庭成员数量_real']]
valid_family = family2023[['家庭编号']]
familymember_2023 = pd.merge(familymember_2023, valid_family, on='家庭编号', how='inner')
family2023[['家庭成员数量']].value_counts()
## 家庭连续型变量
family2023[['家庭成员数量','家庭工作人口数','机动车数量','脚踏自行车数量','电动自行车数量','摩托车数量','老年代步车数量']]
have_student_family = familymember_2023[familymember_2023['职业'] == '14'].drop_duplicates(['家庭编号'])[['家庭编号']]
have_student_family['have_student'] = 1
family2023 = pd.merge(family2023, have_student_family, on='家庭编号', how='left').fillna({'have_student':0})
## 家庭离散型变量
family2023[['have_student','家庭年收入']]
family2023['家庭年收入'].isna().sum()
## 个人变量筛选
familymember_2023['age'] = 2023 - familymember_2023['出生年份'].astype(int)
familymember_2023['age_group'] = pd.cut(familymember_2023['age'], bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], right=False, labels=['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96-100'])
familymember_2023['age'].max() , familymember_2023['age'].min()
# familymember_2023['age'] = (familymember_2023['age'] - familymember_2023['age'].min()) / (familymember_2023['age'].max() - familymember_2023['age'].min())
## 连续型变量
familymember_2023[['age']]
(familymember_2023[familymember_2023['关系']=='0']['age']).describe()
familymember_2023.loc[familymember_2023['最高学历'].isna(),'最高学历'] = familymember_2023.loc[familymember_2023['最高学历'].isna(),'教育阶段']
## 离散型变量,这里的关系有点不太对，有的户主很小
familymember_2023[['性别','是否有驾照','关系','最高学历','职业']]
familymember_2023['是否有驾照'] = familymember_2023['是否有驾照'].fillna('0')

from population_data_process_nonclip import *
## 家庭的变量编码
test = PopulationDataEncoder()
family2023 = pd.merge(family2023,family_cluster[['家庭编号','cluster']], on='家庭编号', how='left')
# 2. 拟合数据 (需要你的实际数据)
test.fit_family_data(family2023)
test.fit_person_data(familymember_2023)

from population_DiT_cluster2_memberbundle import *
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import logging
import os
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from tqdm import tqdm

from losses_personmask_memberbundle import compute_total_loss
from dataset import load_population_data, create_dataloader
model_result = load_population_dit_checkpoint("results/005-PopulationDiT-1020-个人生成加家庭信息、噪声分开加/checkpoints/final.pt")
args = model_result['args']
model = model_result['model']
class DiffusionScheduler:
    """扩散调度器"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 创建beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 为采样准备的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_start, noise, timesteps):
        """添加噪声到原始数据"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        
        # 对于成员数据，需要额外处理维度
        if len(x_start.shape) == 3:  # member data: [batch, family_size, features]
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def to(self, device):
        """移动到指定设备"""
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
print(f"Training Population DiT with {args.model_config}")
    
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建实验文件夹

# 加载数据
dataset = load_population_data(args.data_dir)
# dataset.family_data = dataset.family_data[:, :-1]
# dataset.member_data = dataset.member_data[:, :, :-1]
dataloader = create_dataloader(
    dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)



model.train()


# 训练变量
train_steps = 0
log_steps = 0
running_loss = 0
start_time = time()

scheduler = DiffusionScheduler(num_timesteps=args.num_timesteps).to(device)

epoch = 0
epoch_loss = 0
epoch_steps = 0

progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

results_family_list = []
results_member_list = []

raw_data_family_list = []
raw_data_member_list = []

for batch in progress_bar:
    # 将数据移至GPU
    family_data = batch['family'].to(device)
    member_data = batch['member'].to(device)
    adj_data = batch['adj'].to(device)
    edge_data = batch['edge'].to(device)
    node_data = batch['node'].to(device)
    family_cluster = batch['cluster'].to(torch.int).to(device)
    person_mask = torch.sum(member_data, dim=-1) != 0  # 掩码，标记有效成员
    
    # 随机时间步
    t = torch.randint(0, int(scheduler.num_timesteps/2), (family_data.shape[0],), device=device)
    t_person = t + torch.randint(0, int(scheduler.num_timesteps/2), (family_data.shape[0],), device=device)
    
    # 创建噪声
    noise_family = torch.randn_like(family_data)
    noise_member = torch.randn_like(member_data)
    
    # 添加噪声
    x_family_noisy = scheduler.add_noise(family_data, noise_family, t)
    x_member_noisy = scheduler.add_noise(member_data, noise_member, t_person)
    
    # 前向传播

    pred_family, pred_member, pred_graph = model(x_family_noisy, x_member_noisy, family_cluster, t, t_person)

    # 保存结果用于分析
    results_family_list.append(pred_family.detach().cpu().numpy())
    results_member_list.append(pred_member.detach().cpu().numpy())

    raw_data_family_list.append(family_data.detach().cpu().numpy())
    raw_data_member_list.append(member_data.detach().cpu().numpy())
    
    # 计算损失
    loss_dict = compute_total_loss(
                pred_family, family_data,
                pred_member, member_data, person_mask,
                pred_graph, adj_data, edge_data, node_data,
                weights=args.loss_weights
            )
    
    total_loss = loss_dict['total_loss'].mean()
    

    # 记录损失
    running_loss += total_loss.item()
    epoch_loss += total_loss.item()
    log_steps += 1
    train_steps += 1
    epoch_steps += 1
    
    # 更新进度条
    progress_bar.set_postfix({
        'loss': f'{total_loss.item():.4f}',
        'avg_loss': f'{epoch_loss/epoch_steps:.4f}'
    })
    
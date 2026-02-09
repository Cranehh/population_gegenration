"""
PopulationDiT训练Worker模块（支持梯度冲突检测）

提供BOHB优化所需的评估函数封装，并在训练过程中收集梯度冲突信息
"""

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import time
import pandas as pd

import itertools


def calc_srmse(raw_df, synth_df, max_value_list):
    srmse_results = {}

    # 遍历所有可能的属性组合（1维、2维、...）
    r = len(max_value_list)
    number_of_combinations = 0
    srmse_sum = 0.0
    for cols in itertools.combinations(raw_df.columns, r):
        # 计算联合分布频率
        real_counts = raw_df.groupby(list(cols)).size().reset_index(name='count')
        synth_counts = synth_df.groupby(list(cols)).size().reset_index(name='count')

        # 合并并补全缺失组合
        merged = pd.merge(real_counts, synth_counts, on=list(cols), how='outer', suffixes=('_real', '_synth')).fillna(0)

        # 转换为频率分布
        # merged['π_real'] = merged['count_real'] / merged['count_real'].sum()
        # merged['π_synth'] = merged['count_synth'] / merged['count_synth'].sum()

        merged['π_real'] = merged['count_real']
        merged['π_synth'] = merged['count_synth']
        # 计算 SRMSE
        N_cnt = len(merged)
        numerator = np.sqrt(((merged['π_synth'] - merged['π_real']) ** 2).sum() / N_cnt)
        denominator = (merged['π_real'].sum()) / N_cnt
        srmse = numerator / denominator if denominator != 0 else np.nan
        # srmse = numerator

        srmse_results[cols] = srmse
        srmse_sum += srmse
        number_of_combinations += 1
    avg_srmse = srmse_sum / number_of_combinations if number_of_combinations > 0 else np.nan

    return avg_srmse


current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上级目录（enhanced_bohb_all 的父目录）
parent_dir = os.path.dirname(current_dir)

# 构建完整的数据目录路径
data_dir = os.path.join(parent_dir, '数据')

pd.set_option('display.max_columns', None)
family2014 = pd.read_csv(f'{data_dir}/居民出行数据/2014/family_2014.csv', dtype=str)
travel2014 = pd.read_csv(f'{data_dir}/居民出行数据/2014/midtable_2014.csv', dtype=str)
familymember_2014 = pd.read_csv(f'{data_dir}/居民出行数据/2014/family_member_2014.csv', dtype=str)
family2023 = pd.read_csv(f'{data_dir}/居民出行数据/2023/family_total_33169.csv', dtype=str)
travel2023 = pd.read_csv(f'{data_dir}/居民出行数据/2023/midtable_total_33169.csv', dtype=str)
familymember_2023 = pd.read_csv(f'{data_dir}/居民出行数据/2023/familymember_total_33169.csv', dtype=str)
family_cluster = pd.read_csv(f'{data_dir}/family_cluster_improved.csv', dtype=str)
cluster_profile = pd.read_csv(f'{data_dir}/cluster_profile_improved.csv', dtype=str)
cluster_profile.iloc[:, 1:] = cluster_profile.iloc[:, 1:].astype(float)

valid_member_number = familymember_2023.groupby('家庭编号').size().rename('家庭成员数量_real').reset_index()
family2023 = pd.merge(family2023, valid_member_number, on='家庭编号', how='left')
family2023 = family2023[family2023['家庭成员数量'].astype(int) == family2023['家庭成员数量_real']]
valid_family = family2023[['家庭编号']]
familymember_2023 = pd.merge(familymember_2023, valid_family, on='家庭编号', how='inner')
## 家庭连续型变量
family2023[['家庭成员数量', '家庭工作人口数', '机动车数量', '脚踏自行车数量', '电动自行车数量', '摩托车数量',
            '老年代步车数量']]
have_student_family = familymember_2023[familymember_2023['职业'] == '14'].drop_duplicates(['家庭编号'])[['家庭编号']]
have_student_family['have_student'] = 1
family2023 = pd.merge(family2023, have_student_family, on='家庭编号', how='left').fillna({'have_student': 0})

familymember_2023['age'] = 2023 - familymember_2023['出生年份'].astype(int)
familymember_2023['age_group'] = pd.cut(familymember_2023['age'],
                                        bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
                                              95, 100], right=False,
                                        labels=['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
                                                '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
                                                '81-85', '86-90', '91-95', '96-100'])
familymember_2023.loc[familymember_2023['最高学历'].isna(), '最高学历'] = familymember_2023.loc[
    familymember_2023['最高学历'].isna(), '教育阶段']
## 离散型变量,这里的关系有点不太对，有的户主很小
familymember_2023[['性别', '是否有驾照', '关系', '最高学历', '职业']]
familymember_2023['是否有驾照'] = familymember_2023['是否有驾照'].fillna('0')

income_map = {'A': 1, 'B': 1, 'C': 2, 'D': 2, 'E': 3, 'F': 3, 'G': 4, 'I': 5, 'J': 5, 'K': 5}
family2023['家庭年收入'] = family2023['家庭年收入'].map(income_map)
familymember_2023['age_group'] = pd.cut(
    familymember_2023['age'],
    bins=range(0, familymember_2023['age'].max() + 6, 5),
    labels=False
)

familymember_2023['age_group'] = familymember_2023['age_group'].fillna(0)
familymember_2023['age'] = familymember_2023['age_group']
# relation_map = {'0':0, '17':1, '1':2, '2':2, '5':2, '6':2, '13':3, '14':3, '15':3, '16':3, '9':3, '10':3, '7':4, '8':4, '11':5, '12':5}
# education_map = {'1':1, '2':1, '3':2, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7}
# occupation_map = {'1':1, '2':1, '3':1, '4':2, '5':2, '6':3, '7':2, '8':3, '9':1, '10':4, '11':4, '12':4, '13':5, '14':6, '15':7, '16':8, '17':8, '18':1, '19':1, '20':8}

# familymember_2023['关系'] = familymember_2023['关系'].map(relation_map)
# familymember_2023['最高学历'] = familymember_2023['最高学历'].map(education_map)
# familymember_2023['职业'] = familymember_2023['职业'].map(occupation_map)
familymember_2023['关系'].value_counts().shape, familymember_2023['最高学历'].value_counts().shape, familymember_2023[
    '职业'].value_counts().shape
from population_data_process_nonclip_reclass import *

## 家庭的变量编码
test = PopulationDataEncoder()
family_cluster.rename(columns={'improved_cluster': 'cluster'}, inplace=True)
family2023 = pd.merge(family2023, family_cluster[['家庭编号', 'cluster']], on='家庭编号', how='left')
cluster_profile.rename(columns={'improved_cluster': 'cluster'}, inplace=True)
# 2. 拟合数据 (需要你的实际数据)
test.fit_family_data(family2023)
test.fit_person_data(familymember_2023)



class ConflictAwareWorker:
    """
    支持梯度冲突检测的训练Worker
    
    在训练过程中：
    1. 计算各Loss分项
    2. 检测梯度冲突
    3. 返回冲突暴露度用于分布更新
    """
    
    def __init__(
        self,
        data_dir: str = '数据',
        device: Optional[str] = None,
        gpu_id: Optional[int] = None,
        loss_names: Optional[List[str]] = None,
        num_workers: int = 4,
        validation_split: float = 0.1,
        conflict_detection_freq: int = 10,
        log_every: int = 50
    ):
        """
        初始化Worker
        
        Args:
            data_dir: 数据目录
            device: 计算设备
            gpu_id: GPU ID
            loss_names: Loss分项名称
            num_workers: DataLoader工作进程数
            validation_split: 验证集比例
            conflict_detection_freq: 冲突检测频率（每多少步检测一次）
            log_every: 日志打印频率
        """
        if gpu_id is not None:
            self.device = f'cuda:{gpu_id}'
        elif device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.conflict_detection_freq = conflict_detection_freq
        self.log_every = log_every
        
        self.loss_names = loss_names or [
            'family_continuous', 'family_student',
            'person_age', 'person_gender', 'person_license',
            'person_relation', 'person_education', 'person_occupation',
            'graph_adj', 'graph_node', 'graph_edge',
            'mask_loss', 'total_member_loss'
        ]
        
        # 数据加载器缓存
        self._dataset = None
        self._train_loader = None
        self._val_loader = None
        self._current_batch_size = None
        
        print(f"[ConflictAwareWorker] 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  冲突检测频率: 每 {conflict_detection_freq} 步")
    
    def _load_data(self, batch_size: int):
        """加载数据集"""
        if self._dataset is not None and self._current_batch_size == batch_size:
            return
        
        try:
            from dataset import load_population_data, create_dataloader
        except ImportError:
            raise ImportError("无法导入dataset模块")
        
        print(f"[Worker] 加载数据集...")
        self._dataset = load_population_data(self.data_dir)
        
        n_total = len(self._dataset)
        n_val = int(n_total * self.validation_split)
        n_train = n_total - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self._dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        self._train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self._val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self._current_batch_size = batch_size
        print(f"[Worker] 数据加载完成: 训练集 {n_train}, 验证集 {n_val}")
    
    def _create_model(self, config: Dict) -> nn.Module:
        """创建模型"""
        try:
            import sys
            sys.path.append('..')  # 或者用绝对路径
            sys.path.append('../DiT-main')
            sys.path.append('../GraphVAE-master')
            from population_DiT_cluster11_memberbundle import PopulationDiT
        except ImportError:
            raise ImportError("无法导入PopulationDiT模型")
        
        model = PopulationDiT(
            max_family_size=8,
            proj_dim=24,
            hidden_size=config.get('hidden_dim', 320),
            depth=config.get('num_layers', 30),
            num_heads=config.get('num_heads', 16)
        )
        
        return model.to(self.device)
    
    def _create_scheduler(self, num_timesteps: int = 200):
        """创建扩散调度器"""
        
        class DiffusionScheduler:
            def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
                self.num_timesteps = num_timesteps
                self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
                self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
                self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            
            def add_noise(self, x_start, noise, timesteps):
                sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
                
                if len(x_start.shape) == 3:
                    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
                
                return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
            
            def to(self, device):
                self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
                self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
                return self
        
        return DiffusionScheduler(num_timesteps=num_timesteps).to(self.device)
    
    def _config_to_loss_weights(self, config: Dict) -> Dict:
        """将配置转换为损失权重"""
        family_scale = config.get('family_weight_scale', 1.0)
        person_scale = config.get('person_weight_scale', 2.0)
        graph_scale = config.get('graph_weight_scale', 0.5)
        constraint_scale = config.get('constraint_weight_scale', 1.0)
        
        return {
            'family_continuous': family_scale * 1.0,
            'family_student': family_scale * 1.0,
            'person_age': person_scale * 1.0,
            'person_gender': person_scale * 1.0,
            'person_license': person_scale * 1.0,
            'person_relation': person_scale * 1.0,
            'person_education': person_scale * 1.0,
            'person_occupation': person_scale * 1.0,
            'mask_loss': constraint_scale * 1.0,
            'total_member_loss': constraint_scale * 1.0,
            'graph_adj': graph_scale * 1.0,
            'graph_node': graph_scale * 1.0,
            'graph_edge': graph_scale * 1.0
        }
    
    def _compute_gradient_conflict(
        self,
        model: nn.Module,
        loss_dict: Dict[str, torch.Tensor],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算梯度冲突暴露度
        
        Args:
            model: 模型
            loss_dict: 各Loss分项
            weights: 当前权重
        
        Returns:
            {loss_name: exposure}
        """
        # 简化的梯度计算（只取部分参数以加速）
        params = list(model.parameters())
        
        gradients = {}
        for name, loss in loss_dict.items():
            if loss.requires_grad and name in self.loss_names:
                try:
                    grads = torch.autograd.grad(
                        loss.mean(),
                        params,
                        retain_graph=True,
                        allow_unused=True
                    )
                    flat_grad = torch.cat([
                        g.view(-1) if g is not None else torch.zeros_like(params[id], device=self.device).view(-1)
                        for id, g in enumerate(grads)
                    ])
                    gradients[name] = flat_grad
                except Exception:
                    pass
        
        if len(gradients) < 2:
            return {name: 0.0 for name in self.loss_names}
        
        # 计算冲突矩阵
        grad_names = list(gradients.keys())
        n = len(grad_names)
        conflict_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                g_i = gradients[grad_names[i]]
                g_j = gradients[grad_names[j]]
                
                norm_i = torch.norm(g_i)
                norm_j = torch.norm(g_j)
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = torch.dot(g_i, g_j) / (norm_i * norm_j)
                    conflict = max(0, -cos_sim.item())
                else:
                    conflict = 0.0
                
                conflict_matrix[i, j] = conflict
                conflict_matrix[j, i] = conflict
        
        # 计算暴露度
        exposure = {}
        name_to_idx = {name: i for i, name in enumerate(grad_names)}
        
        for name in self.loss_names:
            if name in name_to_idx:
                i = name_to_idx[name]
                exp = 0.0
                for j, other_name in enumerate(grad_names):
                    if i != j:
                        w = weights.get(other_name, 1.0)
                        exp += w * conflict_matrix[i, j]
                exposure[name] = exp
            else:
                exposure[name] = 0.0
        
        return exposure
    
    def evaluate(
        self,
        config: Dict,
        budget: int,
        return_model: bool = False
    ) -> Tuple[float, Dict]:
        """
        评估配置
        
        Args:
            config: 超参数配置
            budget: 评估预算（epoch数）
            return_model: 是否返回模型
        
        Returns:
            (loss, info) 其中info包含loss_components和conflict_exposure
        """
        try:
            from losses_personmask_memberbundle6 import compute_total_loss
        except ImportError:
            # 使用简化的损失计算
            compute_total_loss = None
        
        # 提取超参数
        batch_size = int(config.get('batch_size', 128))
        lr = float(config.get('lr', 1e-4))
        weight_decay = float(config.get('weight_decay', 1e-4))
        grad_clip = float(config.get('grad_clip', 1.0))
        rho = float(config.get('rho', 0.85))
        num_timesteps = int(config.get('num_timesteps', 200))
        
        # 加载数据
        self._load_data(batch_size)
        
        # 创建模型
        model = self._create_model(config)
        scheduler = self._create_scheduler(num_timesteps)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 损失权重
        loss_weights = self._config_to_loss_weights(config)
        
        # 训练循环
        model.train()
        all_exposures = []
        train_losses = []
        step = 0
        
        start_time = time.time()
        
        for epoch in range(budget):
            epoch_loss = 0
            n_batches = 0
            
            for batch in self._train_loader:
                # 移动数据到设备
                family_data = batch['family'].to(self.device)
                member_data = batch['member'].to(self.device)
                adj_data = batch['adj'].to(self.device)
                edge_data = batch['edge'].to(self.device)
                node_data = batch['node'].to(self.device)
                family_cluster = batch['cluster'].to(torch.int).to(self.device)
                cluster_profile = batch['cluster_profile'].to(self.device)
                person_mask = torch.sum(member_data, dim=-1) != 0
                
                # 时间步
                t = torch.randint(0, num_timesteps, (family_data.shape[0],), device=self.device)
                
                # 噪声
                noise_family = torch.randn_like(family_data)
                noise_member = torch.randn_like(member_data)
                
                noise_to_member = noise_family.repeat(8, 5).view(
                    noise_member.shape[0], noise_member.shape[1], -1
                )
                noise_to_member = torch.cat([
                    noise_to_member,
                    torch.zeros_like(noise_member[:, :, 0]).view(
                        noise_member.shape[0], noise_member.shape[1], 1
                    )
                ], dim=-1)
                noise_member = noise_to_member * rho + math.sqrt(1 - rho ** 2) * noise_member
                
                # 添加噪声
                x_family_noisy = scheduler.add_noise(family_data, noise_family, t)
                x_member_noisy = scheduler.add_noise(member_data, noise_member, t)
                
                # 前向传播
                pred_family, pred_member, pred_graph = model(
                    x_family_noisy, x_member_noisy,
                    family_cluster, cluster_profile, person_mask, t, t
                )
                
                # 计算损失
                if compute_total_loss is not None:
                    loss_dict = compute_total_loss(
                        pred_family, family_data,
                        pred_member, member_data, person_mask,
                        pred_graph, adj_data, edge_data, node_data,
                        weights=loss_weights
                    )
                    total_loss = loss_dict['total_loss'].mean()
                else:
                    # 简化损失
                    total_loss = nn.functional.mse_loss(pred_family, family_data)
                    loss_dict = {'total_loss': total_loss}
                
                # 梯度冲突检测（周期性）
                if step % self.conflict_detection_freq == 0:
                    exposure = self._compute_gradient_conflict(model, loss_dict, loss_weights)
                    all_exposures.append(exposure)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                n_batches += 1
                step += 1
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            
            if (epoch + 1) % self.log_every == 0:
                print(f"  Epoch {epoch + 1}/{budget}: loss={avg_loss:.4f}")
        
        # 验证
        val_loss = self._validate(model, scheduler, loss_weights, rho, num_timesteps)
        
        # 聚合冲突暴露度
        if all_exposures:
            aggregated_exposure = {}
            for name in self.loss_names:
                values = [exp.get(name, 0.0) for exp in all_exposures]
                aggregated_exposure[name] = np.mean(values)
        else:
            aggregated_exposure = {name: 0.0 for name in self.loss_names}
        
        # 提取最后一次的loss分项
        loss_components = {}
        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = value.mean().item()
        
        info = {
            'train_losses': train_losses,
            'val_loss': val_loss,
            'loss_components': loss_components,
            'conflict_exposure': aggregated_exposure,
            'training_time': time.time() - start_time
        }


        
        if return_model:
            info['model'] = model
        
        return val_loss, info
    
    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        scheduler,
        loss_weights: Dict,
        rho: float,
        num_timesteps: int
    ) -> float:
        """验证模型"""
        try:
            from losses_personmask_memberbundle6 import compute_total_loss
        except ImportError:
            raise ImportError("无法导入损失函数模块")

        model.eval()
        total_loss = 0
        n_batches = 0
        results_family_list = []
        results_member_list = []

        raw_data_family_list = []
        raw_data_member_list = []

        for batch in self._val_loader:
            family_data = batch['family'].to(self.device)
            member_data = batch['member'].to(self.device)
            adj_data = batch['adj'].to(self.device)
            edge_data = batch['edge'].to(self.device)
            node_data = batch['node'].to(self.device)
            family_cluster = batch['cluster'].to(torch.int).to(self.device)
            cluster_profile = batch['cluster_profile'].to(self.device)
            person_mask = torch.sum(member_data, dim=-1) != 0

            t = torch.randint(0, num_timesteps, (family_data.shape[0],), device=self.device)

            noise_family = torch.randn_like(family_data)
            noise_member = torch.randn_like(member_data)

            noise_to_member = noise_family.repeat(8, 5).view(
                noise_member.shape[0], noise_member.shape[1], -1
            )
            noise_to_member = torch.cat([
                noise_to_member,
                torch.zeros_like(noise_member[:, :, 0]).view(
                    noise_member.shape[0], noise_member.shape[1], 1
                )
            ], dim=-1)
            noise_member = noise_to_member * rho + math.sqrt(1 - rho ** 2) * noise_member

            x_family_noisy = scheduler.add_noise(family_data, noise_family, t)
            x_member_noisy = scheduler.add_noise(member_data, noise_member, t)

            pred_family, pred_member, pred_graph = model(
                x_family_noisy, x_member_noisy,
                family_cluster, cluster_profile, person_mask, t, t
            )

            results_family_list.append(pred_family.detach().cpu().numpy())
            results_member_list.append(pred_member.detach().cpu().numpy())

            raw_data_family_list.append(family_data.detach().cpu().numpy())
            raw_data_member_list.append(member_data.detach().cpu().numpy())

            loss_dict = compute_total_loss(
                pred_family, family_data,
                pred_member, member_data, person_mask,
                pred_graph, adj_data, edge_data, node_data,
                weights=loss_weights
            )

            total_loss += loss_dict['total_loss'].mean().item()
            n_batches += 1

        results_family_df = pd.DataFrame(np.concatenate(results_family_list, axis=0))
        raw_family_df = pd.DataFrame(np.concatenate(raw_data_family_list, axis=0))
        results_person_df = pd.DataFrame(np.concatenate(results_member_list, axis=0).reshape(-1, 51))
        raw_person_df = pd.DataFrame(np.concatenate(raw_data_member_list, axis=0).reshape(-1, 51))

        results_family_df['have_student'] = results_family_df[[8, 9]].values.argmax(axis=1)
        raw_family_df['have_student'] = raw_family_df[[8, 9]].values.argmax(axis=1)
        results_family_df = results_family_df[[0, 1, 2, 3, 4, 5, 6, 7, 'have_student']]
        results_family_df.columns = ['family_家庭成员数量', 'family_家庭工作人口数', 'family_机动车数量',
                                     'family_脚踏自行车数量', 'family_电动自行车数量', 'family_摩托车数量',
                                     'family_老年代步车数量', 'income', 'have_student']
        raw_family_df = raw_family_df[[0, 1, 2, 3, 4, 5, 6, 7, 'have_student']]
        raw_family_df.columns = ['family_家庭成员数量', 'family_家庭工作人口数', 'family_机动车数量',
                                 'family_脚踏自行车数量', 'family_电动自行车数量', 'family_摩托车数量',
                                 'family_老年代步车数量', 'income', 'have_student']
        raw_person_df['is_real'] = (raw_person_df[50] != 0)
        results_person_df['is_real'] = raw_person_df['is_real']
        raw_person_df = raw_person_df[raw_person_df['is_real'] == True]
        results_person_df = results_person_df[results_person_df['is_real'] == True]
        results_person_df['gender'] = results_person_df[[1, 2]].values.argmax(axis=1)
        results_person_df['license'] = results_person_df[[3, 4]].values.argmax(axis=1)
        results_person_df['relation'] = results_person_df.iloc[:, 5:21].values.argmax(axis=1)
        results_person_df['education'] = results_person_df.iloc[:, 21:30].values.argmax(axis=1)
        results_person_df['occupation'] = results_person_df.iloc[:, 30:50].values.argmax(axis=1)

        raw_person_df['gender'] = raw_person_df[[1, 2]].values.argmax(axis=1)
        raw_person_df['license'] = raw_person_df[[3, 4]].values.argmax(axis=1)
        raw_person_df['relation'] = raw_person_df.iloc[:, 5:21].values.argmax(axis=1)
        raw_person_df['education'] = raw_person_df.iloc[:, 21:30].values.argmax(axis=1)
        raw_person_df['occupation'] = raw_person_df.iloc[:, 30:50].values.argmax(axis=1)
        results_person_df = results_person_df[[0, 'gender', 'license', 'relation', 'education', 'occupation', 50]]
        results_person_df.columns = ['age', 'gender', 'license', 'relation', 'education', 'occupation', 'label']
        raw_person_df = raw_person_df[[0, 'gender', 'license', 'relation', 'education', 'occupation', 50]]
        raw_person_df.columns = ['age', 'gender', 'license', 'relation', 'education', 'occupation', 'label']
        results_family_df.rename(columns={'income': 'family_家庭年收入'}, inplace=True)
        raw_family_df.rename(columns={'income': 'family_家庭年收入'}, inplace=True)

        decode_results_family = test.decode_family_continuous(results_family_df)
        for col in decode_results_family.keys():
            results_family_df[f'family_{col}'] = decode_results_family[col]
        decode_raw_family = test.decode_family_continuous(raw_family_df)
        for col in decode_raw_family.keys():
            raw_family_df[f'family_{col}'] = decode_raw_family[col]
        results_person_df['age'] = test.decode_person_continuous(results_person_df['age'])['age_actual']
        raw_person_df['age'] = test.decode_person_continuous(raw_person_df['age'])['age_actual']

        ## 有老人、有学生、有驾照的比例、性别比例、本科以上比例、雇佣比例, 集成到家庭中
        total_ls_raw = []
        total_ls_generate = []
        person_number = raw_family_df['family_家庭成员数量'].values.cumsum()
        for i in range(len(person_number)):
            if i == 0:
                start_idx = 0
            else:
                start_idx = person_number[i - 1]
            end_idx = person_number[i]
            person_in_family = raw_person_df.iloc[start_idx:end_idx, :]
            person_in_family_generate = results_person_df.iloc[start_idx:end_idx, :]

            ls = []
            ls.append(int((person_in_family['age'] > 11).sum() >= 1))
            ls.append((person_in_family['license']).mean())
            ls.append((person_in_family['gender']).mean())
            ls.append((person_in_family['education'].isin([6, 7])).mean())
            ls.append((-person_in_family['occupation'].isin([14, 13, 16])).mean())
            total_ls_raw.append(ls)

            ls_generate = []
            ls_generate.append(int((person_in_family_generate['age'] > 11).sum() >= 1))
            ls_generate.append((person_in_family_generate['license']).mean())
            ls_generate.append((person_in_family_generate['gender']).mean())
            ls_generate.append((person_in_family_generate['education'].isin([6, 7])).mean())
            ls_generate.append((-person_in_family_generate['occupation'].isin([14, 13, 16])).mean())
            total_ls_generate.append(ls_generate)

        extended_raw_family_df = pd.concat([raw_family_df.reset_index(drop=True), pd.DataFrame(total_ls_raw, columns=[
            'have_elder_ext', 'license_ext', 'gender_ext', 'education_ext', 'employed_ext'])], axis=1)
        extended_results_family_df = pd.concat([results_family_df.reset_index(drop=True),
                                                pd.DataFrame(total_ls_generate,
                                                             columns=['have_elder_ext', 'license_ext', 'gender_ext',
                                                                      'education_ext', 'employed_ext'])], axis=1)
        extended_raw_family_df['license_ext'] = pd.cut(
            extended_raw_family_df['license_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_results_family_df['license_ext'] = pd.cut(
            extended_results_family_df['license_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_raw_family_df['gender_ext'] = pd.cut(
            extended_raw_family_df['gender_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_results_family_df['gender_ext'] = pd.cut(
            extended_results_family_df['gender_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_raw_family_df['education_ext'] = pd.cut(
            extended_raw_family_df['education_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_results_family_df['education_ext'] = pd.cut(
            extended_results_family_df['education_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_raw_family_df['employed_ext'] = pd.cut(
            extended_raw_family_df['employed_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )

        extended_results_family_df['employed_ext'] = pd.cut(
            extended_results_family_df['employed_ext'],
            bins=np.linspace(0, 1, 6),  # 生成 [0,0.2,0.4,0.6,0.8,1.0]
            labels=False,  # 输出整数标签 0,1,2,3,4
            include_lowest=True  # 包含左边界0
        )
        max_value_list_extendfamily = []
        for i in range(len(extended_raw_family_df.columns)):
            raw_max = extended_raw_family_df.iloc[:, i].max()
            results_max = extended_results_family_df.iloc[:, i].max()
            max_value = max(raw_max, results_max)
            max_value_list_extendfamily.append(max_value)
        srmse_results = calc_srmse(extended_raw_family_df, extended_results_family_df, max_value_list_extendfamily)

        model.train()
        return srmse_results

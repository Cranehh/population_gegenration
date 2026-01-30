"""
PopulationDiT训练Worker模块

提供BOHB优化所需的评估函数封装
"""

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
import time


class PopulationDiTWorker:
    """
    PopulationDiT模型训练Worker

    封装模型训练逻辑，提供BOHB优化所需的评估接口

    Usage:
        worker = PopulationDiTWorker(data_dir='数据')
        loss, info = worker.evaluate(config, budget=50)
    """

    def __init__(
        self,
        data_dir: str = '数据',
        device: Optional[str] = None,
        num_workers: int = 4,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        log_every: int = 50
    ):
        """
        初始化Worker

        Args:
            data_dir: 数据目录
            device: 计算设备 ('cuda' 或 'cpu')
            num_workers: 数据加载线程数
            validation_split: 验证集比例
            early_stopping_patience: 早停耐心值
            log_every: 日志频率
        """
        self.data_dir = data_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.log_every = log_every

        # 延迟加载数据和模型
        self._dataset = None
        self._train_loader = None
        self._val_loader = None

        print(f"[PopulationDiTWorker] 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  数据目录: {self.data_dir}")

    def _load_data(self, batch_size: int):
        """
        加载数据集

        Args:
            batch_size: 批次大小
        """
        if self._dataset is not None and self._current_batch_size == batch_size:
            return

        # 导入数据加载函数
        try:
            from dataset import load_population_data, create_dataloader
        except ImportError:
            raise ImportError("无法导入dataset模块，请确保dataset.py在正确的路径")

        # 加载数据
        print(f"[Worker] 加载数据集...")
        self._dataset = load_population_data(self.data_dir)

        # 划分训练/验证集
        n_total = len(self._dataset)
        n_val = int(n_total * self.validation_split)
        n_train = n_total - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            self._dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建DataLoader
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
        """
        创建模型

        Args:
            config: 超参数配置

        Returns:
            PopulationDiT模型
        """
        try:
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
            'unique_loss': constraint_scale * 1.0,
            'graph_adj': graph_scale * 1.0,
            'graph_node': graph_scale * 1.0,
            'graph_edge': graph_scale * 1.0
        }

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
            return_model: 是否返回训练后的模型

        Returns:
            (loss, info) 验证损失和额外信息
        """
        try:
            from losses_personmask_memberbundle8 import compute_total_loss
        except ImportError:
            raise ImportError("无法导入损失函数模块")

        # 提取超参数
        batch_size = int(config.get('batch_size', 1024))
        lr = config.get('lr', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        grad_clip = config.get('grad_clip', 1.0)
        rho = config.get('rho', 0.85)
        num_timesteps = config.get('num_timesteps', 200)

        # 加载数据
        self._load_data(batch_size)

        # 创建模型
        model = self._create_model(config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Worker] 模型参数量: {n_params:,}")

        # 创建调度器
        scheduler = self._create_scheduler(num_timesteps)

        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=budget, eta_min=lr * 0.01
        )

        # 损失权重
        loss_weights = self._config_to_loss_weights(config)

        # 训练循环
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(budget):
            # 训练一个epoch
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

                # 随机时间步
                t = torch.randint(0, num_timesteps, (family_data.shape[0],), device=self.device)

                # 创建噪声
                noise_family = torch.randn_like(family_data)
                noise_member = torch.randn_like(member_data)

                # 相关噪声
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
                loss_dict = compute_total_loss(
                    pred_family, family_data,
                    pred_member, member_data, person_mask,
                    pred_graph, adj_data, edge_data, node_data,
                    weights=loss_weights
                )

                total_loss = loss_dict['total_loss'].mean()

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            # 更新学习率
            lr_scheduler.step()

            # 计算平均训练损失
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # 验证
            val_loss = self._validate(model, scheduler, loss_weights, rho, num_timesteps)
            val_losses.append(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # 日志
            if (epoch + 1) % self.log_every == 0 or epoch == budget - 1:
                elapsed = time.time() - start_time
                print(f"[Worker] Epoch {epoch + 1}/{budget}: "
                      f"train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"best={best_val_loss:.4f}, "
                      f"time={elapsed:.1f}s")

            # 早停
            if patience_counter >= self.early_stopping_patience:
                print(f"[Worker] 早停 at epoch {epoch + 1}")
                break

        # 返回结果
        info = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'n_params': n_params,
            'training_time': time.time() - start_time
        }

        if return_model:
            info['model'] = model

        return best_val_loss, info

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
            from losses_personmask_memberbundle8 import compute_total_loss
        except ImportError:
            raise ImportError("无法导入损失函数模块")

        model.eval()
        total_loss = 0
        n_batches = 0

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

            loss_dict = compute_total_loss(
                pred_family, family_data,
                pred_member, member_data, person_mask,
                pred_graph, adj_data, edge_data, node_data,
                weights=loss_weights
            )

            total_loss += loss_dict['total_loss'].mean().item()
            n_batches += 1

        model.train()
        return total_loss / n_batches


def create_evaluate_function(
    data_dir: str = '数据',
    device: Optional[str] = None,
    **worker_kwargs
) -> Callable[[Dict, int], Tuple[float, Dict]]:
    """
    创建评估函数

    用于传递给EnhancedBOHB优化器

    Args:
        data_dir: 数据目录
        device: 计算设备
        **worker_kwargs: Worker的其他参数

    Returns:
        评估函数 (config, budget) -> (loss, info)
    """
    worker = PopulationDiTWorker(data_dir=data_dir, device=device, **worker_kwargs)

    def evaluate_fn(config: Dict, budget: int) -> Tuple[float, Dict]:
        return worker.evaluate(config, budget)

    return evaluate_fn

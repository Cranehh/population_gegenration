import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = 1 
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import logging
import math
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from tqdm import tqdm
from discreteNoisy import DiscreteVariableDiffusion

from population_DiT_cluster13_memberbundle import PopulationDiT
from losses_personmask_memberbundle6 import compute_total_loss
from dataset import load_population_data, create_dataloader


def create_logger(logging_dir):
    """创建日志器"""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """更新EMA模型"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """设置模型参数的梯度开关"""
    for p in model.parameters():
        p.requires_grad = flag


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


def main(args):
    """主训练函数"""
    print(f"Training Population DiT with {args.model_config}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f"Using device: {device}")
    
    # 创建实验文件夹
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-PopulationDiT"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    # 加载数据
    logger.info("Loading dataset...")
    dataset = load_population_data(args.data_dir)
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # 创建模型
    logger.info("Creating model...")
    model = PopulationDiT(
        max_family_size=8,  # 最大家庭人数
        proj_dim=24,
        hidden_size=args.hidden_dim,
        depth=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    # 创建EMA模型
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # 创建扩散调度器
    scheduler = DiffusionScheduler(num_timesteps=args.num_timesteps).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # 初始化EMA
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    
    # 训练变量
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        epoch_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # 将数据移至GPU
            family_data = batch['family'].to(device)
            member_data = batch['member'].to(device)
            adj_data = batch['adj'].to(device)
            edge_data = batch['edge'].to(device)
            node_data = batch['node'].to(device)
            family_cluster = batch['cluster'].to(torch.int).to(device)
            cluster_profile = batch['cluster_profile'].to(device)
            person_mask = torch.sum(member_data, dim=-1) != 0  # 掩码，标记有效成员
            
            # 随机时间步
            t = torch.randint(0, 200, (family_data.shape[0],), device=device)
            t_person = t  
            
            # 创建噪声
            noise_family = torch.randn_like(family_data)
            noise_member = torch.randn_like(member_data)

            noise_to_member = noise_family.repeat(8, 5).view(noise_member.shape[0], noise_member.shape[1], -1)
            noise_to_member = torch.cat([noise_to_member, torch.zeros_like(noise_member[:, :, 0]).view(noise_member.shape[0], noise_member.shape[1], 1)], dim=-1)
            rho = 0.85
            noise_member =  noise_to_member * rho + math.sqrt(1 - rho ** 2) * noise_member

            
            # 添加噪声
            x_family_noisy = scheduler.add_noise(family_data, noise_family, t)
            x_member_noisy = noise_member
            
            # 前向传播
            pred_family, pred_member, pred_graph = model(x_family_noisy, x_member_noisy, family_cluster, cluster_profile, person_mask, t, t_person)
            
            # 计算损失
            loss_dict = compute_total_loss(
                pred_family, family_data,
                pred_member, member_data, person_mask,
                pred_graph, adj_data, edge_data, node_data,
                weights=args.loss_weights
            )
            
            total_loss = loss_dict['total_loss'].mean()
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            # 更新EMA
            update_ema(ema, model)
            
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
            
            # 日志记录
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                
                # 重置监控变量
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # 保存检查点
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "train_steps": train_steps,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # 更新学习率
        lr_scheduler.step()
        
        # 每个epoch结束后的日志
        avg_epoch_loss = epoch_loss / epoch_steps
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
    # 保存最终模型
    final_checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": args.epochs,
        "train_steps": train_steps,
        "args": args
    }
    final_checkpoint_path = f"{checkpoint_dir}/final.pt"
    torch.save(final_checkpoint, final_checkpoint_path)
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Population DiT")
    
    # 数据相关参数
    parser.add_argument("--data-dir", type=str, default="数据", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    
    # 模型相关参数
    parser.add_argument("--model-config", type=str, default="base", help="Model configuration")
    parser.add_argument("--hidden-dim", type=int, default=320, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=30, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # 扩散相关参数
    parser.add_argument("--num-timesteps", type=int, default=200, help="Number of diffusion timesteps")
    
    # 日志和保存相关参数
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--ckpt-every", type=int, default=1600, help="Save checkpoint every N steps")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    
    # 损失权重
    parser.add_argument("--family-continuous-weight", type=float, default=1.0)
    parser.add_argument("--family-student-weight", type=float, default=1.0)
    parser.add_argument("--family-income-weight", type=float, default=1.0)
    parser.add_argument("--person-age-weight", type=float, default=2.0)
    parser.add_argument("--person-gender-weight", type=float, default=2.0)
    parser.add_argument("--person-license-weight", type=float, default=2.0)
    parser.add_argument("--person-relation-weight", type=float, default=2.0)
    parser.add_argument("--person-education-weight", type=float, default=2.0)
    parser.add_argument("--person-occupation-weight", type=float, default=2.0)
    parser.add_argument("--invalid-person-weight", type=float, default=0.1)
    parser.add_argument("--mask-loss-weight", type=float, default=1)
    parser.add_argument("--total-member-loss-weight", type=float, default=1)
    parser.add_argument("--graph-adj-weight", type=float, default=0.5)
    parser.add_argument("--graph-node-weight", type=float, default=0.5)
    parser.add_argument("--graph-edge-weight", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # 构建损失权重字典
    args.loss_weights = {
        'family_continuous': args.family_continuous_weight,
        'family_student': args.family_student_weight,
        # 'family_income': args.family_income_weight,
        'person_age': args.person_age_weight,
        'person_gender': args.person_gender_weight,
        'person_license': args.person_license_weight,
        'person_relation': args.person_relation_weight,
        'person_education': args.person_education_weight,
        'person_occupation': args.person_occupation_weight,
        # 'invalid_person': args.invalid_person_weight,
        'mask_loss': args.mask_loss_weight,
        'total_member_loss': args.total_member_loss_weight,
        'graph_adj': args.graph_adj_weight,
        'graph_node': args.graph_node_weight,
        'graph_edge': args.graph_edge_weight
    }
    
    main(args)
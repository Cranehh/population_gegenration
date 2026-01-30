import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PopulationDiffusionProcess:
    """适配人口数据的扩散过程"""
    
    def __init__(self, 
                 family_continuous_dim=7,
                 family_categorical_dims=[2, 8],
                 person_continuous_dim=1, 
                 person_categorical_dims=[2, 3, 10, 8, 16],
                 max_family_size=8,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        
        self.family_continuous_dim = family_continuous_dim
        self.family_categorical_dims = family_categorical_dims
        self.person_continuous_dim = person_continuous_dim
        self.person_categorical_dims = person_categorical_dims
        self.max_family_size = max_family_size
        self.num_timesteps = num_timesteps
        
        # 创建噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 创建变量类型掩码
        self._create_variable_masks()
    
    def _create_variable_masks(self):
        """创建区分不同变量类型的掩码"""
        # 计算总的变量维度
        family_total = self.family_continuous_dim + len(self.family_categorical_dims)
        person_total = (self.person_continuous_dim + len(self.person_categorical_dims)) * self.max_family_size
        total_dim = family_total + person_total
        
        # 创建掩码 [batch_size会在使用时广播]
        self.continuous_mask = torch.zeros(total_dim)
        self.categorical_mask = torch.zeros(total_dim)
        
        idx = 0
        
        # 家庭连续变量
        self.continuous_mask[idx:idx + self.family_continuous_dim] = 1.0
        idx += self.family_continuous_dim
        
        # 家庭离散变量
        self.categorical_mask[idx:idx + len(self.family_categorical_dims)] = 1.0
        idx += len(self.family_categorical_dims)
        
        # 个人变量 (重复max_family_size次)
        for _ in range(self.max_family_size):
            # 个人连续变量 (年龄)
            self.continuous_mask[idx:idx + self.person_continuous_dim] = 1.0
            idx += self.person_continuous_dim
            
            # 个人离散变量
            self.categorical_mask[idx:idx + len(self.person_categorical_dims)] = 1.0
            idx += len(self.person_categorical_dims)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：向原始数据添加噪声"""
        if noise is None:
            noise = self._generate_structured_noise(x_start)
        
        # 获取时间步对应的参数
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # 应用扩散公式
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _generate_structured_noise(self, x):
        """生成结构化噪声，区分连续变量和离散变量"""
        batch_size = x.shape[0]
        device = x.device
        
        # 获取掩码并移动到正确设备
        continuous_mask = self.continuous_mask.to(device)
        categorical_mask = self.categorical_mask.to(device)
        
        # 为连续变量生成高斯噪声
        continuous_noise = torch.randn_like(x) * continuous_mask.unsqueeze(0)
        
        # 为离散变量生成结构化噪声
        categorical_noise = self._generate_categorical_noise(x) * categorical_mask.unsqueeze(0)
        
        return continuous_noise + categorical_noise
    
    def _generate_categorical_noise(self, x):
        """为离散变量生成特殊噪声"""
        batch_size, seq_len = x.shape
        device = x.device
        
        # 方案1: 高斯噪声 (简单有效)
        # 离散变量embedding后在连续空间，可以直接用高斯噪声
        noise = torch.randn_like(x)
        
        # 方案2: 类别混淆噪声 (可选，更符合离散变量特性)
        # categorical_noise = torch.zeros_like(x)
        # 
        # idx = 0
        # # 家庭离散变量噪声
        # idx += self.family_continuous_dim
        # for i, dim in enumerate(self.family_categorical_dims):
        #     # 生成类别混淆噪声：随机选择其他类别
        #     random_categories = torch.randint(0, dim, (batch_size,), device=device)
        #     categorical_noise[:, idx + i] = random_categories.float()
        # idx += len(self.family_categorical_dims)
        # 
        # # 个人离散变量噪声
        # for person_idx in range(self.max_family_size):
        #     idx += self.person_continuous_dim  # 跳过个人连续变量
        #     for i, dim in enumerate(self.person_categorical_dims):
        #         random_categories = torch.randint(0, dim, (batch_size,), device=device)
        #         categorical_noise[:, idx + i] = random_categories.float()
        #     idx += len(self.person_categorical_dims)
        
        return noise
    
    def _extract(self, a, t, x_shape):
        """提取时间步对应的参数值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu()).float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def p_losses(self, model, x_start, t, family_data, person_data, noise=None):
        """计算训练损失"""
        if noise is None:
            noise = self._generate_structured_noise(x_start)
        
        # 前向扩散
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 模型预测
        predicted_noise = model(x_noisy, t, family_data, person_data)
        
        # 计算损失 (区分连续和离散变量)
        loss = self._compute_hybrid_loss(predicted_noise, noise, x_start.device)
        
        return loss
    
    def _compute_hybrid_loss(self, predicted, target, device):
        """混合损失函数：对连续和离散变量使用不同权重"""
        # 获取掩码
        continuous_mask = self.continuous_mask.to(device).unsqueeze(0)
        categorical_mask = self.categorical_mask.to(device).unsqueeze(0)
        
        # 连续变量MSE损失
        continuous_loss = F.mse_loss(
            predicted * continuous_mask, 
            target * continuous_mask, 
            reduction='none'
        ).mean(dim=1)
        
        # 离散变量MSE损失 (embedding空间中)
        categorical_loss = F.mse_loss(
            predicted * categorical_mask, 
            target * categorical_mask, 
            reduction='none'
        ).mean(dim=1)
        
        # 加权组合 (可以调整权重)
        total_loss = continuous_loss + 0.5 * categorical_loss
        
        return total_loss.mean()


class PopulationDiffusionTrainer:
    """人口数据扩散模型训练器"""
    
    def __init__(self, model, diffusion_process, device='cuda'):
        self.model = model
        self.diffusion = diffusion_process
        self.device = device
        self.model.to(device)
    
    def train_step(self, family_data, person_data, optimizer):
        """单步训练"""
        batch_size = family_data.shape[0]
        
        # 将数据移动到设备
        family_data = family_data.to(self.device)
        person_data = person_data.to(self.device)
        
        # 拼接成完整序列
        x_start = self._combine_data(family_data, person_data)
        
        # 随机采样时间步
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()
        
        # 计算损失
        loss = self.diffusion.p_losses(self.model, x_start, t, family_data, person_data)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _combine_data(self, family_data, person_data):
        """组合家庭和个人数据成完整序列"""
        batch_size = family_data.shape[0]
        
        # 展平个人数据
        person_flat = person_data.reshape(batch_size, -1)
        
        # 拼接
        combined = torch.cat([family_data, person_flat], dim=1)
        
        return combined
    
    @torch.no_grad()
    def sample(self, num_samples, guidance_scale=1.0):
        """从噪声生成新的人口数据"""
        self.model.eval()
        
        # 计算完整序列长度
        family_dim = self.diffusion.family_continuous_dim + len(self.diffusion.family_categorical_dims)
        person_dim = (self.diffusion.person_continuous_dim + len(self.diffusion.person_categorical_dims)) * self.diffusion.max_family_size
        total_dim = family_dim + person_dim
        
        # 初始化为纯噪声
        x = torch.randn(num_samples, total_dim, device=self.device)
        
        # 逆向扩散采样
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            
            # 分离家庭和个人数据
            family_data = x[:, :family_dim]
            person_data = x[:, family_dim:].reshape(num_samples, self.diffusion.max_family_size, -1)
            
            # 模型预测噪声
            predicted_noise = self.model(x, t, family_data, person_data)
            
            # 去噪步骤 (简化的DDPM采样)
            beta_t = self.diffusion.betas[i]
            alpha_t = self.diffusion.alphas[i]
            alpha_cumprod_t = self.diffusion.alphas_cumprod[i]
            
            # 计算均值
            mean = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            if i > 0:
                # 添加噪声
                noise = self.diffusion._generate_structured_noise(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        
        self.model.train()
        
        # 分离并返回结果
        family_samples = x[:, :family_dim]
        person_samples = x[:, family_dim:].reshape(num_samples, self.diffusion.max_family_size, -1)
        
        return family_samples, person_samples


# 使用示例
def create_training_pipeline(family_data, person_data, encoder):
    """创建完整的训练管道"""
    
    # 1. 初始化扩散过程
    diffusion = PopulationDiffusionProcess(
        family_continuous_dim=encoder.family_continuous_dim,
        family_categorical_dims=encoder.family_categorical_dims,
        person_continuous_dim=encoder.person_continuous_dim,
        person_categorical_dims=encoder.person_categorical_dims
    )
    
    # 2. 初始化模型
    from population_DiT import PopulationDiT
    model = PopulationDiT(
        family_categorical_dims=encoder.family_categorical_dims,
        person_categorical_dims=encoder.person_categorical_dims
    )
    
    # 3. 创建训练器
    trainer = PopulationDiffusionTrainer(model, diffusion)
    
    # 4. 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    return trainer, optimizer


if __name__ == "__main__":
    # 测试噪声生成
    diffusion = PopulationDiffusionProcess()
    
    # 创建测试数据
    batch_size = 4
    total_dim = 7 + 2 + 8 * (1 + 5)  # 家庭连续 + 家庭离散 + 个人变量*8人
    x_test = torch.randn(batch_size, total_dim)
    
    print(f"原始数据形状: {x_test.shape}")
    
    # 生成结构化噪声
    noise = diffusion._generate_structured_noise(x_test)
    print(f"噪声形状: {noise.shape}")
    
    # 测试前向扩散
    t = torch.randint(0, 1000, (batch_size,))
    x_noisy = diffusion.q_sample(x_test, t, noise)
    print(f"加噪后数据形状: {x_noisy.shape}")
    
    print("人口扩散过程测试完成！")
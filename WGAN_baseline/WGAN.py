import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# 将上级目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import load_population_data, create_dataloader
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=256, output_dim=784, depth=4):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        layers = []

        # 1️⃣ 第一层：z_dim → h
        layers.append(nn.Linear(z_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 2️⃣ 第二层：h → 2h
        layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
        layers.append(nn.ReLU())

        # 3️⃣ 中间 depth - 3 层：保持 2h → 2h
        # （前面已经用了两层 Linear，这里开始重复 2h 层）
        for _ in range(depth - 3):
            layers.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
            layers.append(nn.ReLU())

        # 4️⃣ 输出层：2h → output_dim
        layers.append(nn.Linear(hidden_dim * 2, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, depth=4):
        super(Critic, self).__init__()
        
        layers = []

        # 第一层：input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))

        # 中间 depth-2 层：hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))

        # 最后一层：hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class WGAN:
    def __init__(self, 
                 z_dim=100,
                 hidden_dim=256,
                 data_dim=784,
                 depth=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.device = device
        
        # 初始化模型
        self.generator = Generator(z_dim, hidden_dim, data_dim, depth).to(device)
        self.critic = Critic(data_dim, hidden_dim, depth).to(device)
        
        # 训练记录
        self.train_history = {
            'critic_loss': [],
            'generator_loss': [],
            'wasserstein_distance': [],
            'epoch': [],
            'iteration': []
        }
        
        self.config = {
            'z_dim': z_dim,
            'hidden_dim': hidden_dim,
            'data_dim': data_dim,
            'device': str(device)
        }
        
    def train(self, 
              dataloader,
              n_epochs=100,
              n_critic=5,
              clip_value=0.01,
              lr=0.00005,
              save_interval=10,
              save_dir='./wgan_checkpoints'):
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器
        opt_g = optim.RMSprop(self.generator.parameters(), lr=lr)
        opt_c = optim.RMSprop(self.critic.parameters(), lr=lr)
        
        # 训练配置保存
        self.config.update({
            'n_epochs': n_epochs,
            'n_critic': n_critic,
            'clip_value': clip_value,
            'lr': lr,
            'train_start': datetime.now().isoformat()
        })
        
        iteration = 0
        
        for epoch in range(n_epochs):
            epoch_c_loss = []
            epoch_g_loss = []
            epoch_w_dist = []

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
            for batch in progress_bar:
                family_data = batch['family'].to(self.device)
                person_data = batch['member'].to(self.device)
                real_data = torch.cat((family_data, person_data.view(person_data.shape[0], -1)), dim=1)
                batch_size = real_data.size(0)
                
                # ========== 训练判别器 ==========
                for _ in range(n_critic):
                    opt_c.zero_grad()
                    
                    # 真实数据得分
                    real_score = self.critic(real_data)
                    
                    # 生成假数据
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_data = self.generator(z).detach()
                    fake_score = self.critic(fake_data)
                    
                    # Wasserstein距离（负值，因为要最大化）
                    critic_loss = -torch.mean(real_score) + torch.mean(fake_score)
                    
                    critic_loss.backward()
                    opt_c.step()
                    
                    # 权重裁剪
                    for p in self.critic.parameters():
                        p.data.clamp_(-clip_value, clip_value)
                
                # ========== 训练生成器 ==========
                opt_g.zero_grad()
                
                z = torch.randn(batch_size, self.z_dim).to(self.device)
                fake_data = self.generator(z)
                fake_score = self.critic(fake_data)
                
                # 生成器损失（最大化假数据得分）
                generator_loss = -torch.mean(fake_score)
                
                generator_loss.backward()
                opt_g.step()
                
                # 记录损失
                epoch_c_loss.append(critic_loss.item())
                epoch_g_loss.append(generator_loss.item())
                epoch_w_dist.append(-critic_loss.item())  # Wasserstein距离估计
                
                iteration += 1
                
                # 打印进度
                progress_bar.set_postfix({
                'C_loss': f'{np.mean(epoch_c_loss):.4f}',
                'G_loss': f'{np.mean(epoch_g_loss):.4f}',
                'w_dist': f'{np.mean(epoch_w_dist):.4f}'
            })
                # if batch_idx % 50 == 0:
                #     print(f'Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                #           f'C_loss: {critic_loss.item():.4f} G_loss: {generator_loss.item():.4f}')
            
            # 记录epoch级别统计
            self.train_history['critic_loss'].append(np.mean(epoch_c_loss))
            self.train_history['generator_loss'].append(np.mean(epoch_g_loss))
            self.train_history['wasserstein_distance'].append(np.mean(epoch_w_dist))
            self.train_history['epoch'].append(epoch)
            self.train_history['iteration'].append(iteration)
            
            # 定期保存
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                    epoch=epoch,
                    iteration=iteration
                )
                print(f'Checkpoint saved at epoch {epoch+1}')
        
        # 训练结束，保存最终模型
        self.save_checkpoint(
            os.path.join(save_dir, 'final_model.pt'),
            epoch=n_epochs-1,
            iteration=iteration
        )
        
    def save_checkpoint(self, filepath, epoch=None, iteration=None):
        """保存模型检查点"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'epoch': epoch,
            'iteration': iteration
        }
        torch.save(checkpoint, filepath)
        
        # 同时保存配置为JSON（便于查看）
        json_path = filepath.replace('.pt', '_config.json')
        with open(json_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复模型权重
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 恢复配置和历史
        self.config = checkpoint.get('config', self.config)
        self.train_history = checkpoint.get('train_history', self.train_history)
        
        print(f"Model loaded from {filepath}")
        if checkpoint.get('epoch') is not None:
            print(f"Checkpoint from epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']}")
        
        return checkpoint.get('epoch'), checkpoint.get('iteration')
    
    def generate(self, n_samples=64, z=None):
        """生成样本"""
        self.generator.eval()
        with torch.no_grad():
            if z is None:
                z = torch.randn(n_samples, self.z_dim).to(self.device)
            fake_data = self.generator(z)
        self.generator.train()
        return fake_data
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Critic损失
        axes[0].plot(self.train_history['epoch'], self.train_history['critic_loss'])
        axes[0].set_title('Critic Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Generator损失
        axes[1].plot(self.train_history['epoch'], self.train_history['generator_loss'])
        axes[1].set_title('Generator Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        
        # Wasserstein距离
        axes[2].plot(self.train_history['epoch'], self.train_history['wasserstein_distance'])
        axes[2].set_title('Wasserstein Distance')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Distance')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_training_log(self, filepath):
        """保存训练日志为JSON"""
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'history': self.train_history
            }, f, indent=4)


# 使用示例
if __name__ == "__main__":
    # 创建示例数据（这里用随机数据演示）

    data_dim = 10 + 51*8
    
    # 生成示例数据（实际使用时替换为真实数据）
    dataset = load_population_data('数据')
    dataloader = create_dataloader(
        dataset, 
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建WGAN模型
    wgan = WGAN(z_dim=512, hidden_dim=512, data_dim=data_dim, depth=3)
    
    # 训练模型
    wgan.train(
        dataloader,
        n_epochs=10000,
        n_critic=5,
        clip_value=0.01,
        lr=0.00005,
        save_interval=100
    )
    
    # 绘制训练历史
    wgan.plot_training_history('training_history.png')
    
    # 保存训练日志
    wgan.save_training_log('training_log.json')
    
    # 生成一些样本
    # generated_samples = wgan.generate(n_samples=16)
    # print(f"Generated samples shape: {generated_samples.shape}")
    
    # 加载模型示例
    # wgan_loaded = WGAN(z_dim=100, hidden_dim=256, data_dim=data_dim)
    # wgan_loaded.load_checkpoint('./wgan_checkpoints/final_model.pt')
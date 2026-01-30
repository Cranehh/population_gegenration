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


class ConditionalGenerator(nn.Module):
    def __init__(self, 
                 z_dim=100, 
                 hidden_dim=256, 
                 output_dim=784, 
                 depth=4,
                 num_classes=49,
                 class_dropout_prob=0.1):
        super(ConditionalGenerator, self).__init__()
        self.z_dim = z_dim
        
        # 条件嵌入层（与CVAE相同）
        self.y_embedder = LabelEmbedder(num_classes, hidden_dim, class_dropout_prob)
        
        self.profile_proj = nn.Sequential(
            nn.Linear(69, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cluster_norm = nn.LayerNorm(hidden_dim)
        
        layers = []

        # 1️⃣ 第一层：z_dim + hidden_dim (条件) → hidden_dim
        layers.append(nn.Linear(z_dim + hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 2️⃣ 第二层：hidden_dim → 2*hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
        layers.append(nn.ReLU())

        # 3️⃣ 中间 depth - 3 层：保持 2*hidden_dim → 2*hidden_dim
        for _ in range(depth - 3):
            layers.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
            layers.append(nn.ReLU())

        # 4️⃣ 输出层：2*hidden_dim → output_dim
        layers.append(nn.Linear(hidden_dim * 2, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
    
    def forward(self, z, cluster, c_profile):
        """
        z: [B, z_dim] - 随机噪声
        cluster: [B] - cluster标签
        c_profile: [B, 69] - cluster profile特征
        """
        # 获取条件嵌入（与CVAE相同的方式）
        cluster_embed = self.y_embedder(cluster, self.training)
        profile_embed = self.profile_proj(c_profile)
        cluster_embed = cluster_embed + profile_embed
        cluster_embed = self.cluster_norm(cluster_embed)
        
        # 拼接噪声和条件
        z_c = torch.cat([z, cluster_embed], dim=1)
        return self.net(z_c)


class ConditionalCritic(nn.Module):
    def __init__(self, 
                 input_dim=784, 
                 hidden_dim=256, 
                 depth=4,
                 num_classes=49,
                 class_dropout_prob=0.1):
        super(ConditionalCritic, self).__init__()
        
        # 条件嵌入层（与CVAE相同）
        self.y_embedder = LabelEmbedder(num_classes, hidden_dim, class_dropout_prob)
        
        self.profile_proj = nn.Sequential(
            nn.Linear(69, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cluster_norm = nn.LayerNorm(hidden_dim)
        
        layers = []

        # 第一层：input_dim + hidden_dim (条件) -> hidden_dim
        layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))

        # 中间 depth-2 层：hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))

        # 最后一层：hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, cluster, c_profile):
        """
        x: [B, input_dim] - 输入数据
        cluster: [B] - cluster标签
        c_profile: [B, 69] - cluster profile特征
        """
        # 获取条件嵌入（与CVAE相同的方式）
        cluster_embed = self.y_embedder(cluster, self.training)
        profile_embed = self.profile_proj(c_profile)
        cluster_embed = cluster_embed + profile_embed
        cluster_embed = self.cluster_norm(cluster_embed)
        
        # 拼接数据和条件
        x_c = torch.cat([x, cluster_embed], dim=1)
        return self.net(x_c)


class CWGAN:
    def __init__(self, 
                 z_dim=100,
                 hidden_dim=256,
                 data_dim=784,
                 depth=4,
                 num_classes=49,
                 class_dropout_prob=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.device = device
        
        # 初始化条件生成器和判别器
        self.generator = ConditionalGenerator(
            z_dim, hidden_dim, data_dim, depth, num_classes, class_dropout_prob
        ).to(device)
        
        self.critic = ConditionalCritic(
            data_dim, hidden_dim, depth, num_classes, class_dropout_prob
        ).to(device)
        
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
            'num_classes': num_classes,
            'class_dropout_prob': class_dropout_prob,
            'device': str(device)
        }
        
    def train(self, 
              dataloader,
              n_epochs=100,
              n_critic=5,
              clip_value=0.01,
              lr=0.00005,
              save_interval=10,
              save_dir='./cwgan_checkpoints'):
        
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

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
            for batch in progress_bar:
                family_data = batch['family'].to(self.device)
                person_data = batch['member'].to(self.device)
                cluster = batch['cluster'].to(torch.int).to(self.device)
                c_profile = batch['cluster_profile'].to(self.device)
                
                real_data = torch.cat((family_data, person_data.view(person_data.shape[0], -1)), dim=1)
                batch_size = real_data.size(0)
                
                # ========== 训练判别器 ==========
                for _ in range(n_critic):
                    opt_c.zero_grad()
                    
                    # 真实数据得分（条件判别）
                    real_score = self.critic(real_data, cluster, c_profile)
                    
                    # 生成假数据（条件生成）
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_data = self.generator(z, cluster, c_profile).detach()
                    fake_score = self.critic(fake_data, cluster, c_profile)
                    
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
                fake_data = self.generator(z, cluster, c_profile)
                fake_score = self.critic(fake_data, cluster, c_profile)
                
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
                    'W_dist': f'{np.mean(epoch_w_dist):.4f}'
                })
            
            # 记录epoch级别统计
            self.train_history['critic_loss'].append(np.mean(epoch_c_loss))
            self.train_history['generator_loss'].append(np.mean(epoch_g_loss))
            self.train_history['wasserstein_distance'].append(np.mean(epoch_w_dist))
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['iteration'].append(iteration)
            
            print(f'Epoch [{epoch+1}/{n_epochs}] '
                  f'C_loss: {np.mean(epoch_c_loss):.4f} '
                  f'G_loss: {np.mean(epoch_g_loss):.4f} '
                  f'W_dist: {np.mean(epoch_w_dist):.4f}')
            
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
    
    def generate(self, cluster, c_profile, n_samples=None, z=None):
        """
        条件生成样本
        cluster: [B] - cluster标签
        c_profile: [B, 69] - cluster profile特征
        n_samples: 如果z为None，则生成的样本数
        z: [B, z_dim] - 可选的预定义噪声
        """
        self.generator.eval()
        with torch.no_grad():
            if z is None:
                if n_samples is None:
                    n_samples = cluster.size(0)
                z = torch.randn(n_samples, self.z_dim).to(self.device)
            
            # 确保cluster和c_profile在正确的设备上
            cluster = cluster.to(self.device)
            c_profile = c_profile.to(self.device)
            
            fake_data = self.generator(z, cluster, c_profile)
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
    data_dim = 10 + 51*8
    
    # 加载数据
    dataset = load_population_data('数据')
    dataloader = create_dataloader(
        dataset, 
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建CWGAN模型
    cwgan = CWGAN(
        z_dim=512, 
        hidden_dim=512, 
        data_dim=data_dim, 
        depth=3,
        num_classes=49,
        class_dropout_prob=0.1
    )
    
    # 训练模型
    cwgan.train(
        dataloader,
        n_epochs=10000,
        n_critic=5,
        clip_value=0.01,
        lr=0.00005,
        save_interval=100
    )
    
    # 绘制训练历史
    cwgan.plot_training_history('cwgan_training_history.png')
    
    # 保存训练日志
    cwgan.save_training_log('cwgan_training_log.json')
    
    # 条件生成样本示例
    # 假设我们有特定的cluster和profile
    # sample_cluster = torch.tensor([0, 1, 2]).to(cwgan.device)
    # sample_profile = torch.randn(3, 69).to(cwgan.device)
    # generated_samples = cwgan.generate(sample_cluster, sample_profile)
    # print(f"Generated samples shape: {generated_samples.shape}")
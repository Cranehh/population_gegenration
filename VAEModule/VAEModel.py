import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from .VAEDataset import VAETrainingData

g = torch.Generator()

class VAE(nn.Module):
    """
    变分自编码器 (Variational Autoencoder)
    
    参数:
        input_dim: 输入维度（例如28*28=784 for MNIST）
        hidden_dim: 隐藏层维度
        latent_dim: 潜在空间维度
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # 编码器 (Encoder)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值 μ
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差 log(σ²)
        
        # 解码器 (Decoder)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """
        编码器：将输入映射到潜在空间的均值和方差
        """
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar, random=False):
        """
        重参数化技巧：z = μ + σ * ε，其中 ε ~ N(0,1)
        这样可以通过反向传播训练
        """
        std = torch.exp(0.5 * logvar)
        if random:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu + 3.25 * std
        return z
    
    def decode(self, z):
        """
        解码器：将潜在变量映射回输入空间
        """
        h3 = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h3))
        return x_recon
    
    def forward(self, x):
        """
        前向传播：编码 -> 重参数化 -> 解码
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

all_idx = list(range(10+51*8))   # 如果 x 的 shape 是 [batch, feature_dim]
continus_list = [0,1,2,3,4,5,6,7,10,61,112,163,214,265,316,367]

# 取补集（即非连续特征索引）
binary_list = [i for i in all_idx if i not in continus_list]

def loss_function(recon_x, x, mu, logvar):
    """
    VAE损失函数 = 重构损失 + KL散度
    
    重构损失: 衡量重构质量（使用BCE）
    KL散度: 正则化项，使潜在分布接近标准正态分布
    KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    """
    # 重构损失（二元交叉熵）
    # try:
    BCE = F.binary_cross_entropy(torch.nan_to_num(recon_x[:, binary_list], nan=0.0), torch.nan_to_num(x[:, binary_list], nan=0.0), reduction='sum')
    # except:
    #     # 找出小于0或大于1的值
    #     mask = (recon_x[:, binary_list] < 0) | (recon_x[:, binary_list] > 1)

    #     # 获取满足条件的值
    #     invalid_values = recon_x[:, binary_list][mask]

    #     # 获取位置（返回索引的元组）
    #     invalid_positions = torch.where(mask)

    #     print(f"异常值数量: {invalid_values.numel()}")
    #     print(f"异常值: {invalid_values}")
    #     print(f"位置 (行, 列): {list(zip(invalid_positions[0].tolist(), invalid_positions[1].tolist()))}")

    #     # 找出小于0或大于1的值
    #     mask2 = (x[:, binary_list] < 0) | (x[:, binary_list] > 1)

    #     # 获取满足条件的值
    #     invalid_values2 = x[:, binary_list][mask2]

    #     # 获取位置（返回索引的元组）
    #     invalid_positions2 = torch.where(mask2)

    #     print(f"异常值数量: {invalid_values2.numel()}")
    #     print(f"异常值: {invalid_values2}")
    #     print(f"位置 (行, 列): {list(zip(invalid_positions2[0].tolist(), invalid_positions2[1].tolist()))}")
    #     BCE = F.binary_cross_entropy(recon_x[:, binary_list], x[:, binary_list], reduction='sum')

    # 重构损失（MSE损失）
    MSE = F.mse_loss(torch.nan_to_num(recon_x[:, continus_list], nan=0.0), torch.nan_to_num(x[:, continus_list], nan=0.0), reduction='sum')
    # MSE = F.mse_loss(torch.nan_to_num(recon_x, nan=0.0), x, reduction='sum')
    
    # KL散度损失
    KLD = -0.5 * torch.sum(1 + torch.nan_to_num(logvar, nan=0.0) - torch.nan_to_num(mu, nan=0.0).pow(2) - torch.nan_to_num(logvar, nan=0.0).exp())
    
    return BCE + MSE + KLD


def train(model, train_loader, optimizer, epoch, device):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    epoch_step = 0
    data_len = 0
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', 
                ncols=100, leave=False)
    
    for batch in pbar:
        family_data = batch['family'].to(device)
        person_data = batch['member'].to(device)
        # print(family_data.shape)
        # print(person_data.shape)
        data = torch.cat((family_data[0], person_data.view(person_data[0].size(0), -1)), dim=1)
        optimizer.zero_grad()
        
        # 前向传播
        recon_batch, mu, logvar = model(data)
        # 计算损失
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # 反向传播
        loss.backward()
        train_loss += loss.item()
        data_len += data.size(0)
        optimizer.step()
        
        # 更新进度条显示的损失
        avg_loss_so_far = train_loss / data_len
        pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})
        epoch_step += 1
    
    avg_loss = train_loss / train_loader.dataset.data_len_list.sum()
    return avg_loss


def load_vae_model(model_path, input_dim=784, hidden_dim=400, latent_dim=20, device='cpu'):
    """
    加载训练好的VAE模型
    
    参数:
        model_path: 模型文件路径
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        latent_dim: 潜在空间维度
        device: 设备 ('cpu' 或 'cuda')
    
    返回:
        model: 加载好的VAE模型
    """
    # 创建模型实例
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 将模型移到指定设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    return model

def test(model, test_loader, device):
    """测试模型"""
    model.eval()
    test_loss = 0
    
    # 创建进度条
    pbar = tqdm(test_loader, desc='Testing', ncols=100, leave=False)
    
    with torch.no_grad():
        for data, _ in pbar:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{test_loss / len(test_loader.dataset):.4f}'})
    
    avg_loss = test_loss / len(test_loader.dataset)
    return avg_loss


def visualize_reconstruction(model, test_loader, device, n=10):
    """可视化重构结果"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)
        
        # 对比原始图像和重构图像
        fig, axes = plt.subplots(2, n, figsize=(15, 3))
        for i in range(n):
            # 原始图像
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('原始图像', fontproperties='SimHei')
            
            # 重构图像
            axes[1, i].imshow(recon_batch[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('重构图像', fontproperties='SimHei')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/reconstruction.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_latent_space(model, test_loader, device):
    """可视化2D潜在空间（仅当latent_dim=2时）"""
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            latent_vectors.append(mu.cpu())
            labels.append(label)
    
    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    if latent_vectors.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter)
        plt.xlabel('潜在维度 1', fontproperties='SimHei')
        plt.ylabel('潜在维度 2', fontproperties='SimHei')
        plt.title('潜在空间可视化', fontproperties='SimHei')
        plt.savefig('/mnt/user-data/outputs/latent_space.png', dpi=150, bbox_inches='tight')
        plt.close()


def generate_samples(model, device, n=10):
    """从潜在空间随机采样生成新图像"""
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(n * n, model.fc3.in_features).to(device)
        samples = model.decode(z).cpu()
        
        fig, axes = plt.subplots(n, n, figsize=(10, 10))
        for i in range(n):
            for j in range(n):
                axes[i, j].imshow(samples[i * n + j].view(28, 28), cmap='gray')
                axes[i, j].axis('off')
        
        plt.suptitle('生成的样本', fontproperties='SimHei', fontsize=16)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/generated_samples.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    # 超参数设置
    batch_size = 1
    epochs = 500
    learning_rate = 1e-3
    latent_dim = 24  # 潜在空间维度
    family_feature = 10
    person_feature = 51
    max_nodes = 8
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'使用设备: {device}')
    
    # 数据加载
    import glob

    family_files = np.load('../VAE训练数据/family_files.npy')
    person_files = np.load('../VAE训练数据/person_files.npy')

    # family_files.append(f'../数据/family_sample_improved_cluster.npy')
    # person_files.append(f'../数据/family_member_sample_improved_cluster.npy')
    # print(family_files[0])
    # print(person_files[0])

    data_length_list = np.load('../VAE训练数据/data_length_list.npy')

    data_length_list = np.append(data_length_list, 33169)

    train_dataset = VAETrainingData(family_files, person_files, data_length_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)

    # 创建模型
    model = VAE(input_dim=family_feature + person_feature * max_nodes, hidden_dim=1024, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'\n模型结构:')
    print(model)
    print(f'\n参数总数: {sum(p.numel() for p in model.parameters())}')
    
    # 训练模型
    print('\n开始训练...')
    print('='*60)
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 使用tqdm创建epoch进度条
    epoch_pbar = tqdm(range(1, epochs + 1), desc='Total Progress', ncols=100)
    
    for epoch in epoch_pbar:
        # 训练阶段
        train_loss = train(model, train_loader, optimizer, epoch, device)
        train_losses.append(train_loss)
        
        
        # 记录最佳模型
        if train_loss < best_test_loss:
            best_test_loss = train_loss
            torch.save(model.state_dict(), 'vae_best_model.pth')

        else:
            best_marker = ''
        
        # 更新epoch进度条显示
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'best': f'{best_test_loss:.4f}'
        })
        
        # 每个epoch结束后打印详细信息
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (epochs - epoch)
        tqdm.write(f'Epoch {epoch:3d}/{epochs} | '
                   f'Train Loss: {train_loss:.4f} | '
                   f'Time: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m')
    
    # 训练结束统计
    total_time = time.time() - start_time
    print('\n' + '='*60)
    print('训练完成!')
    print(f'总耗时: {total_time/60:.2f}分钟')
    print(f'最终训练损失: {train_losses[-1]:.4f}')
    # print(f'最终测试损失: {test_losses[-1]:.4f}')
    print(f'最佳测试损失: {best_test_loss:.4f}')
    print('='*60)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    # plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch', fontproperties='SimHei')
    plt.ylabel('损失', fontproperties='SimHei')
    plt.title('训练过程', fontproperties='SimHei')
    plt.legend(prop={'family': 'SimHei'})
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # 保存模型
    torch.save(model.state_dict(), 'vae_model.pth')
    print('\n模型已保存到 vae_model.pth')
    print('所有可视化结果已保存！')


if __name__ == '__main__':
    main()

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
import sys
import os

# 将上级目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import load_population_data, create_dataloader

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


class CVAE(nn.Module):
    """
    条件变分自编码器 (Conditional VAE)
    Encoder: q(z|person_flat, c)
    Decoder: p(x|z, c)  -> x 是 family + person_flat
    """
    def __init__(self, 
                 family_dim=10, 
                 person_flat_dim=51*8, 
                 hidden_dim=1024, 
                 latent_dim=20,
                 num_classes=49,
                 class_dropout_prob = 0.1):
        super(CVAE, self).__init__()
        self.person_flat_dim = person_flat_dim + family_dim
        self.latent_dim = latent_dim

        self.y_embedder = LabelEmbedder(num_classes, hidden_dim, class_dropout_prob)

        self.profile_proj_family = nn.Sequential(
            nn.Linear(69, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cluster_norm_family = nn.LayerNorm(hidden_dim)

        # 编码器输入维度：person_flat + family（作为条件）
        enc_input_dim = self.person_flat_dim + hidden_dim
        self.fc1 = nn.Linear(enc_input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar

        # 解码器：输入是 z + family（条件）
        dec_input_dim = latent_dim + hidden_dim
        self.fc3 = nn.Linear(dec_input_dim, hidden_dim)
        # 输出维度是完整数据维度 = family_dim + person_flat_dim
        self.fc4 = nn.Linear(hidden_dim, self.person_flat_dim)
    
    def encode(self, person_flat, c):
        """
        person_flat: [B, person_flat_dim]
        c: [B, family_dim]
        """
        h = torch.cat((person_flat, c), dim=1)
        h1 = F.relu(self.fc1(h))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """
        z: [B, latent_dim]
        c: [B, family_dim]
        returns recon_x of shape [B, family_dim + person_flat_dim]
        """
        h = torch.cat((z, c), dim=1)
        h3 = F.relu(self.fc3(h))
        x_recon = torch.sigmoid(self.fc4(h3))
        return x_recon

    def forward(self, person_flat, cluster, c_profile):
        cluster_embed = self.y_embedder(cluster, self.training)
        profile_embed = self.profile_proj_family(c_profile)
        cluster_embed = cluster_embed + profile_embed
        cluster_embed = self.cluster_norm_family(cluster_embed)
        mu, logvar = self.encode(person_flat, cluster_embed)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cluster_embed)
        return recon, mu, logvar


all_idx = list(range(10+51*8))   # 如果 x 的 shape 是 [batch, feature_dim]
continus_list = [0,1,2,3,4,5,6,7,10,61,112,163,214,265,316,367]

# 取补集（即非连续特征索引）
binary_list = [i for i in all_idx if i not in continus_list]

def loss_function(recon_x, x, mu, logvar):
    """
    与原先一致：BCE（binary_list） + MSE（continus_list） + KL
    recon_x, x: [B, full_dim]
    """
    # 防止索引为空时出错，加入保护
    loss = 0.0
    if len(binary_list) > 0:
        BCE = F.binary_cross_entropy(recon_x[:, binary_list], x[:, binary_list], reduction='sum')
        loss += BCE
    if len(continus_list) > 0:
        MSE = F.mse_loss(recon_x[:, continus_list], x[:, continus_list], reduction='sum')
        loss += MSE

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss += KLD
    return loss


def train(model, train_loader, optimizer, epoch, device):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    epoch_step = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', ncols=100, leave=False)
    
    for batch in pbar:
        family_data = batch['family'].to(device)            # [B, family_dim]
        person_data = batch['member'].to(device)            # [B, nodes, person_feature]
        person_flat = person_data.view(person_data.shape[0], -1)  # [B, person_flat_dim]
        cluster = batch['cluster'].to(torch.int).to(device)              # [B]
        c_profile = batch['cluster_profile'].to(device)     # [B, 69
        # 用于重构对比的完整数据（family + person_flat）
        data = torch.cat((family_data, person_flat), dim=1)  # [B, full_dim]
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, cluster, c_profile)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        avg_loss_so_far = train_loss / ((epoch_step + 1) * data.shape[0])
        pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})
        epoch_step += 1
    
    avg_loss = train_loss / train_loader.dataset.__len__()
    return avg_loss


def load_cvae_model(model_path, family_dim=10, person_flat_dim=51*8, hidden_dim=1024, latent_dim=20, device='cpu'):
    model = CVAE(family_dim=family_dim, person_flat_dim=person_flat_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def generate_conditional_samples(model, c, device, n=10):
    """
    根据给定的条件 c (family features) 生成样本。
    c: tensor shape [k, family_dim]；如果你想生成 n*n 样本，传入 k=n*n 或传入单个 c 重复。
    返回：samples tensor [k, full_dim]
    """
    model.eval()
    with torch.no_grad():
        # 如果 c 是单条，扩展成 n*n
        if c.dim() == 1:
            c = c.unsqueeze(0)
        k = c.shape[0]
        z = torch.randn(k, model.latent_dim).to(device)
        samples = model.decode(z, c.to(device))
    return samples.cpu()


# ---------- main 函数（只改关键部分以示例如何实例化 CVAE 并训练） ----------
def main():
    # 超参数设置（示例）
    batch_size = 1024
    epochs = 10000
    learning_rate = 1e-3
    latent_dim = 512  # 潜在空间维度
    family_feature = 10
    person_feature = 51
    max_nodes = 8
    person_flat_dim = person_feature * max_nodes
    full_dim = family_feature + person_flat_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    dataset = load_population_data('数据')
    dataloader = create_dataloader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 创建 CVAE 模型：注意传入 family_dim 和 person_flat_dim
    model = CVAE(family_dim=family_feature, 
                 person_flat_dim=person_flat_dim,
                 hidden_dim=1024, 
                 latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'\n模型结构:')
    print(model)
    print(f'\n参数总数: {sum(p.numel() for p in model.parameters())}')

    print('\n开始训练...')
    print('='*60)
    train_losses = []
    best_train_loss = float('inf')

    start_time = time.time()
    epoch_pbar = tqdm(range(1, epochs + 1), desc='Total Progress', ncols=100)

    for epoch in epoch_pbar:
        train_loss = train(model, dataloader, optimizer, epoch, device)
        train_losses.append(train_loss)

        # 保存最优模型
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'CVAE_baseline/cvae_best_model.pth')

        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'best': f'{best_train_loss:.4f}'
        })

        elapsed = time.time() - start_time
        # 注意：不要用未来时间估计（见系统提示），这里仅展示一个简短信息
        tqdm.write(f'Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Time: {elapsed/60:.1f}m')

    total_time = time.time() - start_time
    print('\n' + '='*60)
    print('训练完成!')
    print(f'总耗时: {total_time/60:.2f}分钟')
    print(f'最终训练损失: {train_losses[-1]:.4f}')
    print(f'最佳训练损失: {best_train_loss:.4f}')
    print('='*60)

    torch.save(model.state_dict(), 'CVAE_baseline/cvae_model.pth')
    print('\n模型已保存到 cvae_model.pth')


if __name__ == '__main__':
    main()

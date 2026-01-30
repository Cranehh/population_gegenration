import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteVariableDiffusion:
    """
    离散分类变量的扩散模型
    支持有序(教育程度)和无序(职业)分类变量
    """
    
    def __init__(self, num_classes, num_timesteps=1000, ordered=False, device='cpu'):
        """
        参数:
            num_classes: 类别数量
            num_timesteps: 扩散步数
            ordered: 是否为有序分类(True=教育程度, False=职业)
            device: 'cpu' or 'cuda'
        """
        self.num_classes = num_classes
        self.T = num_timesteps
        self.ordered = ordered
        self.device = device
        
        # 噪声调度
        betas = self._cosine_beta_schedule(num_timesteps)
        self.betas = torch.from_numpy(betas).float().to(device)
        
        # 预计算所有转移矩阵
        self.Q_matrices = []  # Q_t: q(x_t|x_{t-1})
        self.Q_bar_matrices = []  # Q_bar_t: q(x_t|x_0)
        self.Q_bar_prev_matrices = []  # Q_bar_{t-1}: q(x_{t-1}|x_0)
        
        self._precompute_matrices()
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦噪声调度"""
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
    
    def _create_transition_matrix(self, beta_t):
        """
        创建单步转移矩阵 Q_t
        Q_t[i,j] = p(x_t=j | x_{t-1}=i)
        """
        num_classes = self.num_classes
        
        if self.ordered:
            # 有序分类: 更倾向于转移到相邻类别
            Q = torch.zeros((num_classes, num_classes))
            for i in range(num_classes):
                Q[i, i] = 1 - beta_t
                if i > 0:
                    Q[i, i-1] = beta_t / 2
                if i < num_classes - 1:
                    Q[i, i+1] = beta_t / 2
                # 归一化
                Q[i, :] /= Q[i, :].sum()
        else:
            # 无序分类: 均匀转移到所有类别
            Q = (1 - beta_t) * torch.eye(num_classes, device=self.device) + \
                beta_t / num_classes * torch.ones((num_classes, num_classes), device=self.device)
        
        return Q
    
    def _precompute_matrices(self):
        """预计算所有时间步的转移矩阵"""
        Q_bar = torch.eye(self.num_classes, device=self.device)
        
        for t in range(self.T):
            # 单步转移矩阵
            Q_t = self._create_transition_matrix(self.betas[t])
            self.Q_matrices.append(Q_t.to(self.device))
            
            # 保存上一步的累积矩阵 (用于后验计算)
            self.Q_bar_prev_matrices.append(Q_bar.clone().to(self.device))
            
            # 更新累积转移矩阵: Q_bar_t = Q_bar_{t-1} @ Q_t
            Q_bar = Q_bar @ Q_t
            self.Q_bar_matrices.append(Q_bar.to(self.device))
    
    def forward_diffusion(self, x_0_onehot, t):
        """
        前向加噪过程: x_0 -> x_t
        
        参数:
            x_0_onehot: 原始数据的one-hot编码 (batch_size, num_classes)
            t: 时间步 (batch_size,) - 每个样本可以有不同的时间步
        
        返回:
            x_t_onehot: t时刻加噪后的one-hot编码 (batch_size, num_classes)
            q_xt_given_x0: 转移概率分布 p(x_t|x_0) (batch_size, num_classes)
        """
        batch_size = x_0_onehot.shape[0]
        device = x_0_onehot.device
        
        # 初始化输出
        x_t_onehot = torch.zeros_like(x_0_onehot)
        q_xt_given_x0 = torch.zeros_like(x_0_onehot)
        
        # 按时间步分组处理(提高效率)
        unique_t = torch.unique(t)
        
        for t_val in unique_t:
            # 找到时间步为t_val的所有样本
            mask = (t == t_val)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # 获取累积转移矩阵 Q_bar_t
            Q_bar_t = self.Q_bar_matrices[t_val.item()]  # (num_classes, num_classes)
            
            # 计算转移概率: p(x_t|x_0) = x_0 @ Q_bar_t
            x0_batch = x_0_onehot[indices]  # (n, num_classes)
            probs_batch = x0_batch @ Q_bar_t  # (n, num_classes)
            
            # 保存概率分布
            q_xt_given_x0[indices] = probs_batch
            
            # 从概率分布中采样
            sampled_classes = torch.multinomial(
                probs_batch, 
                num_samples=1
            ).squeeze(-1)  # (n,)
            
            # 转换为one-hot
            x_t_onehot[indices] = F.one_hot(
                sampled_classes, 
                self.num_classes
            ).float()
        
        return x_t_onehot, q_xt_given_x0
    
    def posterior_sample(self, x_t_onehot, x0_pred_probs, t):
        """
        反向去噪的单步采样: x_t -> x_{t-1}
        使用后验分布 q(x_{t-1}|x_t, x_0)
        
        参数:
            x_t_onehot: 当前时刻的one-hot编码 (batch_size, num_classes)
            x0_pred_probs: 模型预测的x_0概率分布 (batch_size, num_classes)
            t: 当前时间步 (batch_size,)
        
        返回:
            x_tm1_onehot: 上一时刻的one-hot编码 (batch_size, num_classes)
        """
        batch_size = x_t_onehot.shape[0]
        device = x_t_onehot.device
        
        # 初始化输出
        x_tm1_onehot = torch.zeros_like(x_t_onehot)
        
        # 按时间步分组处理
        unique_t = torch.unique(t)
        
        for t_val in unique_t:
            mask = (t == t_val)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # t=0时,直接返回预测的最可能类别
            if t_val == 0:
                sampled_classes = torch.argmax(x0_pred_probs[indices], dim=-1)
                x_tm1_onehot[indices] = F.one_hot(
                    sampled_classes, 
                    self.num_classes
                ).float()
                continue
            
            # 获取转移矩阵
            Q_t = self.Q_matrices[t_val.item()]  # (C, C)
            Q_bar_tm1 = self.Q_bar_prev_matrices[t_val.item()]  # (C, C)
            
            # 提取当前批次的数据
            x_t_batch = x_t_onehot[indices]  # (n, C)
            x0_pred_batch = x0_pred_probs[indices]  # (n, C)
            n = len(indices)
            C = self.num_classes
            
            # 计算后验分布 q(x_{t-1}|x_t, x_0)
            # 
            # 贝叶斯公式:
            # q(x_{t-1}|x_t, x_0) ∝ q(x_t|x_{t-1}) * q(x_{t-1}|x_0)
            #
            # 1. q(x_{t-1}|x_0) = x_0 @ Q_bar_{t-1}
            #    形状: (n, C) @ (C, C) -> (n, C)
            q_xtm1_given_x0 = x0_pred_batch @ Q_bar_tm1  # (n, C)
            
            # 2. q(x_t|x_{t-1}) = Q_t[x_{t-1}, x_t]
            #    我们需要对每个可能的x_{t-1},计算其转移到当前x_t的概率
            #    x_t是one-hot编码,先转换为类别索引
            x_t_classes = torch.argmax(x_t_batch, dim=-1)  # (n,)
            
            # 取出Q_t中对应x_t的列: Q_t[:, x_t]
            # 对于batch中的每个样本,取对应的列
            q_xt_given_xtm1 = Q_t[:, x_t_classes.view(-1)].T  # (n, C)
            # 每行代表: 对于该样本,从每个可能的x_{t-1}转移到x_t的概率
            
            # 3. 后验概率 (逐元素相乘)
            posterior_probs = q_xt_given_xtm1 * q_xtm1_given_x0.view(-1, self.num_classes)  # (n, C)
            
            # 4. 归一化
            posterior_probs = posterior_probs / (
                posterior_probs.sum(dim=-1, keepdim=True) + 1e-10
            )

            mask = posterior_probs.sum(dim=-1) > 0
            posterior_probs_valid = posterior_probs[mask]
            # 5. 从后验分布采样
            sampled_classes = torch.multinomial(
                posterior_probs_valid,
                num_samples=1
            ).squeeze(-1)  # (n,)
            
            # 转换为one-hot

            valid_one_hot = F.one_hot(
                sampled_classes,
                self.num_classes
            ).float()

            all_data = torch.zeros_like(posterior_probs)
            all_data[mask] = valid_one_hot
            all_data = all_data.view(n, -1, self.num_classes)
            x_tm1_onehot[indices] = all_data[indices]

        return x_tm1_onehot
    
    def obtain_noisy_sample(self, batch_size):
        device = self.device
        
        # 从纯噪声开始 (均匀分布)
        x_t = torch.randn(batch_size*8, self.num_classes).to(device)
        x_t = F.softmax(x_t, dim=-1)  # 转为概率分布
        # 采样初始类别
        init_classes = torch.multinomial(x_t, num_samples=1).squeeze(-1)
        x_t = F.one_hot(init_classes, self.num_classes).float()
        return x_t

    def sample(self, model, batch_size, cond_features=None):
        """
        完整的反向去噪采样过程: 从噪声生成数据
        
        参数:
            model: 训练好的去噪模型 (输入x_t和t,输出x_0的预测)
            batch_size: 生成样本数量
            cond_features: 条件特征 (可选),如年龄、性别等
        
        返回:
            x_0_onehot: 生成的样本 (batch_size, num_classes)
        """
        device = self.device
        
        # 从纯噪声开始 (均匀分布)
        x_t = torch.randn(batch_size, self.num_classes).to(device)
        x_t = F.softmax(x_t, dim=-1)  # 转为概率分布
        # 采样初始类别
        init_classes = torch.multinomial(x_t, num_samples=1).squeeze(-1)
        x_t = F.one_hot(init_classes, self.num_classes).float()
        
        # 从t=T-1反向迭代到t=0
        for t_val in reversed(range(self.T)):
            # 当前时间步
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # 模型预测x_0的概率分布
            with torch.no_grad():
                x0_pred_logits = model(x_t, t, cond_features)
                x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
            
            # 从后验分布采样x_{t-1}
            x_t = self.posterior_sample(x_t, x0_pred_probs, t)
            
            if (t_val + 1) % 200 == 0 or t_val == 0:
                print(f"去噪进度: {self.T - t_val}/{self.T}")
        
        return x_t


class JointDiscreteVariableDiffusion:
    """
    联合扩散模型: 同时处理多个离散变量(如教育+职业)
    """
    
    def __init__(self, variable_configs, num_timesteps=1000, device='cpu'):
        """
        参数:
            variable_configs: 变量配置列表
                例如: [
                    {'name': 'education', 'num_classes': 5, 'ordered': True},
                    {'name': 'occupation', 'num_classes': 10, 'ordered': False}
                ]
            num_timesteps: 扩散步数
            device: 'cpu' or 'cuda'
        """
        self.variable_configs = variable_configs
        self.num_vars = len(variable_configs)
        self.T = num_timesteps
        self.device = device
        
        # 为每个变量创建独立的扩散过程
        self.diffusions = {}
        for config in variable_configs:
            name = config['name']
            self.diffusions[name] = DiscreteVariableDiffusion(
                num_classes=config['num_classes'],
                num_timesteps=num_timesteps,
                ordered=config.get('ordered', False),
                device=device
            )
    
    def forward_diffusion(self, x_0_dict, t):
        """
        联合前向加噪
        
        参数:
            x_0_dict: 字典,键为变量名,值为one-hot编码
                例如: {
                    'education': tensor(batch_size, 5),
                    'occupation': tensor(batch_size, 10)
                }
            t: 时间步 (batch_size,)
        
        返回:
            x_t_dict: 加噪后的字典
            probs_dict: 转移概率字典
        """
        x_t_dict = {}
        probs_dict = {}
        
        for name, diffusion in self.diffusions.items():
            x_t, probs = diffusion.forward_diffusion(x_0_dict[name], t)
            x_t_dict[name] = x_t
            probs_dict[name] = probs
        
        return x_t_dict, probs_dict
    
    def sample(self, model, batch_size, cond_features=None):
        """
        联合采样
        
        参数:
            model: 联合去噪模型 (输入所有变量的x_t,输出所有变量的x_0预测)
            batch_size: 生成样本数量
            cond_features: 条件特征
        
        返回:
            x_0_dict: 生成的样本字典
        """
        device = self.device
        
        # 初始化所有变量为噪声
        x_t_dict = {}
        for name, diffusion in self.diffusions.items():
            num_classes = diffusion.num_classes
            init_probs = torch.randn(batch_size, num_classes).to(device)
            init_probs = F.softmax(init_probs, dim=-1)
            init_classes = torch.multinomial(init_probs, num_samples=1).squeeze(-1)
            x_t_dict[name] = F.one_hot(init_classes, num_classes).float()
        
        # 反向去噪
        for t_val in reversed(range(self.T)):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # 模型预测所有变量的x_0
            with torch.no_grad():
                x0_pred_logits_dict = model(x_t_dict, t, cond_features)
                x0_pred_probs_dict = {
                    name: F.softmax(logits, dim=-1) 
                    for name, logits in x0_pred_logits_dict.items()
                }
            
            # 对每个变量分别采样
            for name, diffusion in self.diffusions.items():
                x_t_dict[name] = diffusion.posterior_sample(
                    x_t_dict[name],
                    x0_pred_probs_dict[name],
                    t
                )
            
            if (t_val + 1) % 200 == 0 or t_val == 0:
                print(f"去噪进度: {self.T - t_val}/{self.T}")
        
        return x_t_dict
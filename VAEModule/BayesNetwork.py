import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, AICScore
from pgmpy.inference import VariableElimination
import pandas as pd
from tqdm import tqdm
import pickle
import os
from VAEModule.VAEModel import *


class BayesianNetworkForLatent:
    """
    基于VAE潜空间的贝叶斯网络建模

    参数:
        latent_dim: VAE潜空间维度
        n_bins: 离散化的箱数(将连续潜变量离散化)
        structure: 网络结构,默认为None(自动学习)
    """

    def __init__(self, latent_dim, n_bins=10, structure=None):
        self.latent_dim = latent_dim
        self.n_bins = n_bins
        self.structure = structure
        self.model = None
        self.bin_edges = {}  # 存储每个潜变量的离散化边界
        self.node_names = [f'z{i}' for i in range(latent_dim)]

    def discretize_latent_variables(self, latent_data):
        """
        将连续的潜变量离散化

        参数:
            latent_data: (n_samples, latent_dim) 的潜空间数据

        返回:
            discretized_data: 离散化后的数据
        """
        n_samples = latent_data.shape[0]
        discretized_data = np.zeros_like(latent_data, dtype=int)

        for i in range(self.latent_dim):
            # 使用等频率分箱
            _, self.bin_edges[i] = pd.qcut(latent_data[:, i],
                                           q=self.n_bins,
                                           retbins=True,
                                           duplicates='drop')
            # 离散化
            discretized_data[:, i] = np.digitize(latent_data[:, i],
                                                 self.bin_edges[i][1:-1])

        return discretized_data

    def learn_structure_from_data(self, latent_data, method='hillclimb'):
        """
        从潜空间数据学习贝叶斯网络结构

        参数:
            latent_data: (n_samples, latent_dim) 的潜空间数据
            method: 结构学习方法 ('hillclimb', 'pc', 'mmhc')

        返回:
            edges: 学习到的边列表
        """
        from pgmpy.estimators import HillClimbSearch, BicScore, AICScore
        from pgmpy.estimators import PC, MmhcEstimator

        # 离散化数据
        discretized_data = self.discretize_latent_variables(latent_data)

        # 转换为DataFrame
        df = pd.DataFrame(discretized_data,
                          columns=self.node_names)

        print(f"学习贝叶斯网络结构 (方法: {method})...")

        if method == 'hillclimb':
            # 使用爬山算法 + BIC评分
            scoring_method = AICScore(df)
            est = HillClimbSearch(df)
            self.structure = est.estimate(scoring_method=scoring_method,
                                          max_iter=1000)
        elif method == 'pc':
            # 使用PC算法
            est = PC(df)
            self.structure = est.estimate()
        elif method == 'mmhc':
            # 使用MMHC算法
            est = MmhcEstimator(df)
            self.structure = est.estimate()
        else:
            raise ValueError(f"不支持的方法: {method}")

        edges = list(self.structure.edges())
        print(f"学习到 {len(edges)} 条边")

        return edges, df

    def fit(self, latent_data, method='hillclimb', estimator='mle'):
        """
        训练贝叶斯网络

        参数:
            latent_data: (n_samples, latent_dim) 的潜空间数据
            method: 结构学习方法
            estimator: 参数估计方法 ('mle' 或 'bayes')
        """
        # 学习结构
        edges, df = self.learn_structure_from_data(latent_data, method)

        # 创建贝叶斯网络模型
        self.model = BayesianNetwork(edges)

        # 参数估计
        print(f"估计条件概率表 (方法: {estimator})...")
        if estimator == 'mle':
            self.model.fit(df, estimator=MaximumLikelihoodEstimator)
        elif estimator == 'bayes':
            self.model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')
        else:
            raise ValueError(f"不支持的估计器: {estimator}")

        print("贝叶斯网络训练完成!")
        return self

    def save_model(self, save_path):
        """
        保存贝叶斯网络模型

        参数:
            save_path: 保存路径 (例如: 'bn_model.pkl')
        """
        if self.model is None:
            raise ValueError("模型未训练,请先调用fit方法")

        # 创建保存目录
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存所有必要的信息
        model_data = {
            'model': self.model,
            'latent_dim': self.latent_dim,
            'n_bins': self.n_bins,
            'structure': self.structure,
            'bin_edges': self.bin_edges,
            'node_names': self.node_names
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {save_path}")

    @classmethod
    def load_model(cls, load_path):
        """
        加载贝叶斯网络模型

        参数:
            load_path: 模型文件路径

        返回:
            bn_model: 加载的贝叶斯网络模型实例
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)

        # 创建新实例
        bn_model = cls(
            latent_dim=model_data['latent_dim'],
            n_bins=model_data['n_bins'],
            structure=model_data['structure']
        )

        # 恢复模型状态
        bn_model.model = model_data['model']
        bn_model.bin_edges = model_data['bin_edges']
        bn_model.node_names = model_data['node_names']

        print(f"模型已从 {load_path} 加载")
        return bn_model

    def inference(self, evidence=None, variables=None):
        """
        贝叶斯推断

        参数:
            evidence: 观测证据,字典形式 {'z0': 5, 'z1': 3}
            variables: 要查询的变量列表

        返回:
            推断结果
        """
        if self.model is None:
            raise ValueError("模型未训练,请先调用fit方法")

        inference_engine = VariableElimination(self.model)

        if variables is None:
            variables = self.node_names

        result = inference_engine.query(variables=variables, evidence=evidence)

        return result

    def get_structure_info(self):
        """获取网络结构信息"""
        if self.model is None:
            raise ValueError("模型未训练")

        info = {
            'nodes': list(self.model.nodes()),
            'edges': list(self.model.edges()),
            'n_edges': len(self.model.edges()),
            'parents': {node: list(self.model.get_parents(node))
                        for node in self.model.nodes()},
            'children': {node: list(self.model.get_children(node))
                         for node in self.model.nodes()}
        }

        return info

    def sample_from_network(self, n_samples=100, evidence=None):
        """
        从贝叶斯网络中采样

        参数:
            n_samples: 采样数量
            evidence: 观测证据

        返回:
            samples: 采样结果
        """
        if self.model is None:
            raise ValueError("模型未训练")

        from pgmpy.sampling import BayesianModelSampling

        sampler = BayesianModelSampling(self.model)

        if evidence is None:
            samples = sampler.forward_sample(size=n_samples)
        else:
            samples = sampler.rejection_sample(evidence=evidence,
                                               size=n_samples)

        return samples

    def visualize_structure(self, save_path=None):
        """可视化贝叶斯网络结构"""
        if self.model is None:
            raise ValueError("模型未训练")

        import networkx as nx
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # 创建图
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())

        # 使用层次化布局
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.circular_layout(G)

        # 绘制网络
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=1500,
                with_labels=True,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                edge_color='gray',
                width=2)

        plt.title('Bayesian Network Structure for VAE Latent Space',
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"网络结构已保存到 {save_path}")

        plt.close()


def extract_latent_representations(vae_model, data_loader, device):
    """
    从VAE模型中提取潜空间表示

    参数:
        vae_model: 训练好的VAE模型
        data_loader: 数据加载器
        device: 设备

    返回:
        latent_data: (n_samples, latent_dim) 的潜空间数据
        labels: 对应的标签(如果有)
    """
    vae_model.eval()
    latent_list = []
    label_list = []

    with torch.no_grad():
        for data, label in tqdm(data_loader, desc="提取潜空间表示"):
            data = data.to(device)
            mu, logvar = vae_model.encode(data.view(-1, 784))
            # 使用均值作为潜空间表示
            latent_list.append(mu.cpu().numpy())
            label_list.append(label.numpy())

    latent_data = np.concatenate(latent_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    return latent_data, labels


def analyze_latent_dependencies(bn_model):
    """
    分析潜变量之间的依赖关系

    参数:
        bn_model: 训练好的贝叶斯网络模型

    返回:
        dependency_matrix: 依赖关系矩阵
    """
    info = bn_model.get_structure_info()
    n = bn_model.latent_dim

    # 创建依赖矩阵 (有向图的邻接矩阵)
    dependency_matrix = np.zeros((n, n), dtype=int)

    for parent, child in info['edges']:
        i = int(parent[1:])  # 从 'z0' 提取 0
        j = int(child[1:])  # 从 'z1' 提取 1
        dependency_matrix[i, j] = 1

    print("\n潜变量依赖关系分析:")
    print("=" * 50)

    for node in info['nodes']:
        idx = int(node[1:])
        parents = info['parents'][node]
        children = info['children'][node]

        print(f"\n{node}:")
        if parents:
            print(f"  父节点: {parents}")
        if children:
            print(f"  子节点: {children}")

    return dependency_matrix


def conditional_generation(vae_model, bn_model, evidence, n_samples=10, device='cpu'):
    """
    基于贝叶斯网络的条件生成

    参数:
        vae_model: VAE模型
        bn_model: 贝叶斯网络模型
        evidence: 证据,例如 {'z0': 5, 'z1': 3}
        n_samples: 生成样本数
        device: 设备

    返回:
        generated_images: 生成的图像
    """
    # 从贝叶斯网络采样
    samples = bn_model.sample_from_network(n_samples=n_samples, evidence=evidence)

    # 将离散值转换回连续潜变量
    latent_samples = np.zeros((n_samples, bn_model.latent_dim))

    for i in range(bn_model.latent_dim):
        node_name = f'z{i}'
        discrete_values = samples[node_name].values

        # 使用箱的中心值
        bins = bn_model.bin_edges[i]
        for j, val in enumerate(discrete_values):
            if val < len(bins) - 1:
                latent_samples[j, i] = (bins[val] + bins[val + 1]) / 2
            else:
                latent_samples[j, i] = bins[-1]

    # 使用VAE解码器生成图像
    latent_tensor = torch.FloatTensor(latent_samples).to(device)

    vae_model.eval()
    with torch.no_grad():
        generated_images = vae_model.decode(latent_tensor)

    return generated_images.cpu().numpy()


# 使用示例
if __name__ == '__main__':
    """
    使用示例:

    1. 训练VAE模型
    2. 提取潜空间表示
    3. 训练贝叶斯网络
    4. 保存和加载模型
    5. 进行推断和条件生成
    """
    latent_dim = 20
    # 假设已经有训练好的VAE模型和数据加载器
    # vae_model = load_vae_model('vae_model.pth', latent_dim=20)
    # train_loader = ...
    device = 'cuda'
    # 提取潜空间表示
    # latent_data, labels = extract_latent_representations(vae_model, train_loader, device)
    family_data = torch.FloatTensor(np.load(f'../数据/family_sample_improved_cluster.npy')[:, :10]).to(device)
    member_data = torch.FloatTensor(np.load(f'../数据/family_member_sample_improved_cluster.npy')).to(device)
    data = torch.cat((family_data, member_data.view(member_data.size(0), -1)), dim=1)

    model_path = 'vae_best_model.pth'
    model = VAE(input_dim=10 + 51 * 8, hidden_dim=1024, latent_dim=latent_dim)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 将模型移到指定设备
    model = model.to(device)

    # 设置为评估模式
    model.eval()
    mu, logvar = model.encode(data)
    z = model.reparameterize(mu, logvar).cpu().detach().numpy()

    # 创建并训练贝叶斯网络
    bn_model = BayesianNetworkForLatent(latent_dim=latent_dim, n_bins=10)

    # 训练
    bn_model.fit(z, method='hillclimb', estimator='mle')

    # 保存模型
    bn_model.save_model('bayesian_network_model.pkl')

    # 获取结构信息
    info = bn_model.get_structure_info()
    print("\n网络结构信息:")
    print(f"节点数: {len(info['nodes'])}")
    print(f"边数: {info['n_edges']}")
    print(f"边: {info['edges']}")

    # 分析依赖关系
    dependency_matrix = analyze_latent_dependencies(bn_model)

    # 可视化网络结构
    bn_model.visualize_structure('bayesian_network_structure.png')

    # ========== 加载模型示例 ==========
    print("\n" + "=" * 50)
    print("测试模型加载功能:")
    print("=" * 50)

    # 加载之前保存的模型
    loaded_bn_model = BayesianNetworkForLatent.load_model('bayesian_network_model.pkl')

    # 验证加载的模型
    loaded_info = loaded_bn_model.get_structure_info()
    print(f"\n加载的模型 - 节点数: {len(loaded_info['nodes'])}")
    print(f"加载的模型 - 边数: {loaded_info['n_edges']}")

    # 推断示例
    # evidence = {'z0': 5, 'z1': 3}
    # result = bn_model.inference(evidence=evidence, variables=['z2', 'z3'])
    # print("\n推断结果:")
    # print(result)

    # 从网络采样
    samples = bn_model.sample_from_network(n_samples=10)
    print("\n从贝叶斯网络采样:")
    print(samples.head())

    print("\n完成!")
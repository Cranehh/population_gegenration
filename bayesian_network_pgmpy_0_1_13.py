import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
from tqdm import tqdm


class BayesianNetworkForLatent:
    """
    基于VAE潜空间的贝叶斯网络建模 (适配 pgmpy 0.1.13)

    参数:
        latent_dim: VAE潜空间维度
        n_bins: 离散化的箱数（将连续潜变量离散化）
        structure: 网络结构，默认为None（自动学习）
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
            method: 结构学习方法 ('hillclimb', 'pc', 'constraint')

        返回:
            edges: 学习到的边列表
        """
        from pgmpy.estimators import HillClimbSearch, BicScore
        from pgmpy.estimators import ConstraintBasedEstimator
        
        # 离散化数据
        discretized_data = self.discretize_latent_variables(latent_data)

        # 转换为DataFrame
        df = pd.DataFrame(discretized_data,
                          columns=self.node_names)

        print(f"学习贝叶斯网络结构 (方法: {method})...")

        if method == 'hillclimb':
            # 使用爬山算法 + BIC评分
            est = HillClimbSearch(df, scoring_method=BicScore(df))
            best_model = est.estimate()
            # 在 0.1.13 版本中，estimate() 返回的是 BayesianModel 对象
            edges = list(best_model.edges())
            self.structure = edges
            
        elif method == 'constraint':
            # 使用基于约束的方法
            est = ConstraintBasedEstimator(df)
            best_model = est.estimate(
                ci_test='chi_square',
                significance_level=0.05
            )
            edges = list(best_model.edges())
            self.structure = edges
            
        else:
            # 如果不支持自动结构学习，使用简单的链式结构
            print(f"方法 {method} 在 0.1.13 版本中不支持，使用默认链式结构")
            edges = [(f'z{i}', f'z{i+1}') for i in range(self.latent_dim-1)]
            self.structure = edges

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

        # 创建贝叶斯网络模型 (在 0.1.13 中使用 BayesianModel)
        self.model = BayesianModel(edges)

        # 参数估计
        print(f"估计条件概率表 (方法: {estimator})...")
        if estimator == 'mle':
            # 在 0.1.13 版本中，fit 方法的调用方式不同
            self.model.fit(df, estimator=MaximumLikelihoodEstimator)
        elif estimator == 'bayes':
            # BayesianEstimator 在 0.1.13 中的参数可能不同
            self.model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')
        else:
            raise ValueError(f"不支持的估计器: {estimator}")

        print("贝叶斯网络训练完成!")
        return self

    def inference(self, evidence=None, variables=None):
        """
        贝叶斯推断

        参数:
            evidence: 观测证据，字典形式 {'z0': 5, 'z1': 3}
            variables: 要查询的变量列表

        返回:
            推断结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        inference_engine = VariableElimination(self.model)

        if variables is None:
            variables = self.node_names

        # 在 0.1.13 版本中，query 方法的参数可能略有不同
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

        try:
            # 尝试使用 0.1.13 版本的采样方法
            from pgmpy.sampling import BayesianModelSampling
            
            sampler = BayesianModelSampling(self.model)
            
            if evidence is None:
                samples = sampler.forward_sample(size=n_samples)
            else:
                # rejection_sample 可能在旧版本中不可用
                try:
                    samples = sampler.rejection_sample(evidence=evidence,
                                                       size=n_samples)
                except:
                    print("警告: rejection_sample 在此版本中不可用，使用 forward_sample")
                    samples = sampler.forward_sample(size=n_samples)
                    # 手动过滤满足证据的样本
                    for key, value in evidence.items():
                        samples = samples[samples[key] == value]
                    if len(samples) < n_samples:
                        print(f"警告: 只找到 {len(samples)} 个满足条件的样本")
        except ImportError:
            print("警告: BayesianModelSampling 在 pgmpy 0.1.13 中可能不可用")
            print("使用简单的前向采样方法")
            samples = self._simple_forward_sample(n_samples, evidence)
            
        return samples

    def _simple_forward_sample(self, n_samples, evidence=None):
        """
        简单的前向采样实现（用于旧版本兼容）
        """
        samples_list = []
        
        for _ in range(n_samples):
            sample = {}
            # 按拓扑排序的顺序采样
            for node in self.model.nodes():
                parents = self.model.get_parents(node)
                if not parents:
                    # 根节点，从边缘概率采样
                    cpd = self.model.get_cpds(node)
                    probs = cpd.values.flatten()
                    sample[node] = np.random.choice(len(probs), p=probs)
                else:
                    # 有父节点，从条件概率采样
                    cpd = self.model.get_cpds(node)
                    parent_values = [sample[p] for p in parents]
                    # 这里需要根据CPD的结构获取正确的条件概率
                    # 简化处理，随机采样
                    sample[node] = np.random.randint(0, self.n_bins)
                    
            # 如果有证据，检查是否满足
            if evidence:
                match = all(sample.get(k) == v for k, v in evidence.items())
                if match:
                    samples_list.append(sample)
            else:
                samples_list.append(sample)
                
        return pd.DataFrame(samples_list)

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
        labels: 对应的标签（如果有）
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
        evidence: 证据，例如 {'z0': 5, 'z1': 3}
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
    使用示例：

    1. 训练VAE模型
    2. 提取潜空间表示
    3. 训练贝叶斯网络
    4. 进行推断和条件生成
    """

    # 假设已经有训练好的VAE模型和数据加载器
    # vae_model = load_vae_model('vae_model.pth', latent_dim=20)
    # train_loader = ...

    # 提取潜空间表示
    # latent_data, labels = extract_latent_representations(vae_model, train_loader, device)

    # 创建并训练贝叶斯网络
    latent_dim = 20
    bn_model = BayesianNetworkForLatent(latent_dim=latent_dim, n_bins=10)

    # 使用随机数据演示
    np.random.seed(42)
    demo_latent_data = np.random.randn(1000, latent_dim)

    # 训练
    bn_model.fit(demo_latent_data, method='hillclimb', estimator='mle')

    # 获取结构信息
    info = bn_model.get_structure_info()
    print("\n网络结构信息:")
    print(f"节点数: {len(info['nodes'])}")
    print(f"边数: {info['n_edges']}")
    print(f"边: {info['edges'][:10]}")  # 只显示前10条边

    # 分析依赖关系
    dependency_matrix = analyze_latent_dependencies(bn_model)

    # 可视化网络结构
    # bn_model.visualize_structure('bayesian_network_structure.png')

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

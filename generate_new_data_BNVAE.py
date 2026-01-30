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
from VAEModule.BayesNetwork import *
import random


def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    # Python内置随机数
    random.seed(seed)

    # NumPy随机数
    np.random.seed(seed)

    # PyTorch CPU随机数
    torch.manual_seed(seed)

    # PyTorch GPU随机数（所有GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ 可选：更严格的确定性模式
    # torch.use_deterministic_algorithms(True)


# 在程序开始时调用
set_seed(42)
class DifferentiableBayesianNetwork:
    """
    可微分的贝叶斯网络概率计算
    用于DiT的classifier guidance
    """
    
    def __init__(self, bn_model, device='cuda'):
        """
        参数:
            bn_model: 训练好的BayesianNetworkForLatent模型
            device: 计算设备
        """
        self.bn_model = bn_model
        self.device = device
        self.latent_dim = bn_model.latent_dim
        self.n_bins = bn_model.n_bins
        
        # 将离散化边界转换为torch tensor
        self.bin_edges_tensor = {}
        for i in range(self.latent_dim):
            edges = torch.FloatTensor(bn_model.bin_edges[i]).to(device)
            self.bin_edges_tensor[i] = edges
        
        # 提取CPD (条件概率分布) 并转换为torch tensor
        self.cpds = self._extract_cpds()
        
        # 获取网络结构信息
        self.structure_info = bn_model.get_structure_info()
        self.parents_dict = self.structure_info['parents']
        
    def _extract_cpds(self):
        """从pgmpy模型中提取CPD并转换为torch tensor"""
        cpds = {}
        
        for cpd in self.bn_model.model.get_cpds():
            var_name = cpd.variable
            # 获取CPD的值 (numpy array)
            values = cpd.values
            
            # 转换为torch tensor
            cpds[var_name] = torch.FloatTensor(values).to(self.device)
            
        return cpds
    
    def soft_discretize(self, z, temperature=0.1):
        """
        可微分的软离散化
        使用softmax将连续值映射到离散bin的概率分布
        
        参数:
            z: (batch_size, latent_dim) 连续潜变量
            temperature: softmax温度参数，越小越接近hard discretization
            
        返回:
            soft_bins: (batch_size, latent_dim, n_bins) 每个变量在各bin的概率
        """
        batch_size = z.shape[0]
        soft_bins_list = []
        
        for i in range(self.latent_dim):
            # 获取该维度的bin边界
            edges = self.bin_edges_tensor[i]
            n_bins = len(edges) - 1
            
            # 计算到每个bin中心的距离
            # bin中心
            bin_centers = (edges[:-1] + edges[1:]) / 2  # (n_bins,)
            
            # 计算距离: (batch_size, n_bins)
            z_expanded = z[:, i:i+1]  # (batch_size, 1)
            distances = -torch.abs(z_expanded - bin_centers.unsqueeze(0))  # (batch_size, n_bins)
            
            # 使用softmax转换为概率分布
            soft_probs = F.softmax(distances / temperature, dim=1)  # (batch_size, n_bins)
            
            soft_bins_list.append(soft_probs)
        
        # (batch_size, latent_dim, n_bins)
        soft_bins = torch.stack(soft_bins_list, dim=1)
        
        return soft_bins
    
    def compute_log_prob(self, z, temperature=0.1):
        """
        计算潜变量z在贝叶斯网络下的对数概率
        保持梯度可追踪
        
        参数:
            z: (batch_size, latent_dim) 连续潜变量
            temperature: 软离散化的温度参数
            
        返回:
            log_prob: (batch_size,) 每个样本的对数概率
        """
        batch_size = z.shape[0]
        
        # 软离散化
        soft_bins = self.soft_discretize(z, temperature)  # (batch_size, latent_dim, n_bins)
        
        # 计算每个变量的对数概率
        log_probs = []
        
        for node in self.parents_dict.keys():
            node_idx = int(node[1:])  # 'z0' -> 0
            parents = self.parents_dict[node]
            
            # 获取该节点的CPD
            cpd = self.cpds[node]  # CPD tensor
            
            if len(parents) == 0:
                # 无父节点：P(node)
                # cpd shape: (n_bins,)
                node_soft_bins = soft_bins[:, node_idx, :]  # (batch_size, n_bins)
                
                # 计算加权对数概率
                # log P(node) = sum_i p(bin_i) * log(CPD[i])
                cpd_log = torch.log(cpd + 1e-10)  # (n_bins,)
                log_prob = torch.sum(node_soft_bins * cpd_log.unsqueeze(0), dim=1)  # (batch_size,)
                
            else:
                # 有父节点：P(node | parents)
                # 需要根据父节点的状态选择对应的CPD
                
                # 为简化，这里假设最多2个父节点
                # 如果有更多父节点，需要递归处理
                
                if len(parents) == 1:
                    parent_idx = int(parents[0][1:])
                    parent_soft_bins = soft_bins[:, parent_idx, :]  # (batch_size, n_bins)
                    node_soft_bins = soft_bins[:, node_idx, :]  # (batch_size, n_bins)
                    
                    # cpd shape: (n_bins_node, n_bins_parent)
                    # 计算 P(node | parent) = sum_j p(parent=j) * P(node | parent=j)
                    cpd_log = torch.log(cpd + 1e-10)  # (n_bins_node, n_bins_parent)
                    
                    # 边际化父节点
                    # (batch_size, n_bins_node, n_bins_parent)
                    weighted_cpd = cpd_log.unsqueeze(0) * parent_soft_bins.unsqueeze(1)
                    
                    # 对父节点求和
                    marginalized_cpd = torch.sum(weighted_cpd, dim=2)  # (batch_size, n_bins_node)
                    
                    # 加权node的软bins
                    log_prob = torch.sum(node_soft_bins * marginalized_cpd, dim=1)  # (batch_size,)
                    
                elif len(parents) == 2:
                    parent1_idx = int(parents[0][1:])
                    parent2_idx = int(parents[1][1:])
                    parent1_soft_bins = soft_bins[:, parent1_idx, :]  # (batch_size, n_bins)
                    parent2_soft_bins = soft_bins[:, parent2_idx, :]  # (batch_size, n_bins)
                    node_soft_bins = soft_bins[:, node_idx, :]  # (batch_size, n_bins)
                    
                    # cpd shape: (n_bins_node, n_bins_parent1, n_bins_parent2)
                    cpd_log = torch.log(cpd + 1e-10)
                    
                    # 边际化两个父节点
                    # (batch_size, n_bins_node, n_bins_parent1, n_bins_parent2)
                    weighted_cpd = cpd_log.unsqueeze(0) * \
                                   parent1_soft_bins.unsqueeze(1).unsqueeze(3) * \
                                   parent2_soft_bins.unsqueeze(1).unsqueeze(2)
                    
                    # 对父节点求和
                    marginalized_cpd = torch.sum(weighted_cpd, dim=(2, 3))  # (batch_size, n_bins_node)
                    
                    log_prob = torch.sum(node_soft_bins * marginalized_cpd, dim=1)
                
                else:
                    # 超过2个父节点的情况
                    raise NotImplementedError(f"节点 {node} 有 {len(parents)} 个父节点，暂不支持")
            
            log_probs.append(log_prob)
        
        # 总对数概率 = 各节点对数概率之和
        total_log_prob = torch.stack(log_probs, dim=0).sum(dim=0)  # (batch_size,)
        
        return total_log_prob
    
    def compute_prob(self, z, temperature=0.1):
        """
        计算潜变量z的概率
        
        参数:
            z: (batch_size, latent_dim) 连续潜变量
            temperature: 软离散化的温度参数
            
        返回:
            prob: (batch_size,) 每个样本的概率
        """
        log_prob = self.compute_log_prob(z, temperature)
        return torch.exp(log_prob)

def vectorized_distance_score(new_z, all_z):
    """
    new_z: Tensor or ndarray, shape (M, D)
    all_z: Tensor or ndarray, shape (N, D)
    return: shape (M,) vector
    """

    # --- convert to numpy if tensor ---
    is_torch = False
    if 'torch' in str(type(new_z)):
        is_torch = True
        new_z_np = new_z.detach().cpu().numpy()
        all_z_np = all_z.detach().cpu().numpy()
    else:
        new_z_np = new_z
        all_z_np = all_z

    # ||x||^2 term
    new_norm = (new_z_np ** 2).sum(axis=1)           # (M,)
    all_norm = (all_z_np ** 2).sum(axis=1)           # (N,)

    # pairwise squared distance: (M, N)
    dist_sq = new_norm[:, None] + all_norm[None, :] - 2 * new_z_np @ all_z_np.T
    dist_sq = np.maximum(dist_sq, 0)

    # 每一行的平均，再 sqrt，再 sqrt
    out = np.sqrt(np.sqrt(dist_sq.mean(axis=1)))

    if is_torch:
        return torch.from_numpy(out).to(new_z.device)
    return out




# 数据读取
import pandas as pd
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
family2014 = pd.read_csv('数据/居民出行数据/2014/family_2014.csv',dtype=str)
travel2014 = pd.read_csv('数据/居民出行数据/2014/midtable_2014.csv',dtype=str)
familymember_2014 = pd.read_csv('数据/居民出行数据/2014/family_member_2014.csv',dtype=str)
family2023 = pd.read_csv('数据/居民出行数据/2023/family_total_33169.csv',dtype=str)
travel2023 = pd.read_csv('数据/居民出行数据/2023/midtable_total_33169.csv',dtype=str)
familymember_2023 = pd.read_csv('数据/居民出行数据/2023/familymember_total_33169.csv',dtype=str)
family_cluster = pd.read_csv('数据/family_cluster_new.csv',dtype=str)
## 家庭变量筛选
valid_member_number = familymember_2023.groupby('家庭编号').size().rename('家庭成员数量_real').reset_index()
family2023 = pd.merge(family2023, valid_member_number, on='家庭编号', how='left')
family2023 = family2023[family2023['家庭成员数量'].astype(int) == family2023['家庭成员数量_real']]
valid_family = family2023[['家庭编号']]
familymember_2023 = pd.merge(familymember_2023, valid_family, on='家庭编号', how='inner')
family2023[['家庭成员数量']].value_counts()
## 家庭连续型变量
family2023[['家庭成员数量','家庭工作人口数','机动车数量','脚踏自行车数量','电动自行车数量','摩托车数量','老年代步车数量']]
have_student_family = familymember_2023[familymember_2023['职业'] == '14'].drop_duplicates(['家庭编号'])[['家庭编号']]
have_student_family['have_student'] = 1
family2023 = pd.merge(family2023, have_student_family, on='家庭编号', how='left').fillna({'have_student':0})
## 家庭离散型变量
family2023[['have_student','家庭年收入']]
family2023['家庭年收入'].isna().sum()
## 个人变量筛选
familymember_2023['age'] = 2023 - familymember_2023['出生年份'].astype(int)
familymember_2023['age_group'] = pd.cut(familymember_2023['age'], bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], right=False, labels=['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96-100'])
familymember_2023['age'].max() , familymember_2023['age'].min()
# familymember_2023['age'] = (familymember_2023['age'] - familymember_2023['age'].min()) / (familymember_2023['age'].max() - familymember_2023['age'].min())
## 连续型变量
familymember_2023[['age']]
(familymember_2023[familymember_2023['关系']=='0']['age']).describe()
familymember_2023.loc[familymember_2023['最高学历'].isna(),'最高学历'] = familymember_2023.loc[familymember_2023['最高学历'].isna(),'教育阶段']
## 离散型变量,这里的关系有点不太对，有的户主很小
familymember_2023[['性别','是否有驾照','关系','最高学历','职业']]
familymember_2023['是否有驾照'] = familymember_2023['是否有驾照'].fillna('0')


income_map = {'A':1, 'B':1, 'C':2, 'D':2, 'E':3, 'F':3, 'G':4, 'I':5, 'J':5, 'K':5}
family2023['家庭年收入'] = family2023['家庭年收入'].map(income_map)
familymember_2023['age_group'] = pd.cut(
    familymember_2023['age'],
    bins=range(0, familymember_2023['age'].max() + 6, 5),
    labels=False
)


familymember_2023['age_group'] = familymember_2023['age_group'].fillna(0)
familymember_2023['age'] = familymember_2023['age_group']
# relation_map = {'0':0, '17':1, '1':2, '2':2, '5':2, '6':2, '13':3, '14':3, '15':3, '16':3, '9':3, '10':3, '7':4, '8':4, '11':5, '12':5}
# education_map = {'1':1, '2':1, '3':2, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7}
# occupation_map = {'1':1, '2':1, '3':1, '4':2, '5':2, '6':3, '7':2, '8':3, '9':1, '10':4, '11':4, '12':4, '13':5, '14':6, '15':7, '16':8, '17':8, '18':1, '19':1, '20':8}

# familymember_2023['关系'] = familymember_2023['关系'].map(relation_map)
# familymember_2023['最高学历'] = familymember_2023['最高学历'].map(education_map)
# familymember_2023['职业'] = familymember_2023['职业'].map(occupation_map)
familymember_2023['关系'].value_counts().shape, familymember_2023['最高学历'].value_counts().shape, familymember_2023['职业'].value_counts().shape
from population_data_process_nonclip_reclass import *
## 家庭的变量编码
test = PopulationDataEncoder()
family2023 = pd.merge(family2023,family_cluster[['家庭编号','cluster']], on='家庭编号', how='left')
# 2. 拟合数据 (需要你的实际数据)
test.fit_family_data(family2023)
test.fit_person_data(familymember_2023)


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


latent_dim = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载VAE模型
model_path = 'VAEModule/vae_best_model.pth'
model_vae = VAE(input_dim=10 + 51 * 8, hidden_dim=1024, latent_dim=latent_dim)
model_vae.load_state_dict(torch.load(model_path, map_location=device))
model_vae = model_vae.to(device)
model_vae.eval()

# 加载贝叶斯网络模型
loaded_bn_model = BayesianNetworkForLatent.load_model('VAEModule/bayesian_network_model.pkl')

# 创建可微分的贝叶斯网络
diff_bn = DifferentiableBayesianNetwork(loaded_bn_model, device=device)

# 测试数据
family_data = torch.FloatTensor(np.load(f'数据/family_sample_improved_cluster.npy')[:, :10]).to(device)
member_data = torch.FloatTensor(np.load(f'数据/family_member_sample_improved_cluster.npy')).to(device)
data = torch.cat((family_data, member_data.view(member_data.size(0), -1)), dim=1)
# 编码到潜空间
with torch.no_grad():
    mu, logvar = model_vae.encode(data)
    all_z = model_vae.reparameterize(mu, logvar)


### 引导函数代码
grid_population_profile = pd.read_csv('数据/beijing_grid_population.csv')
## 要对齐的特征有什么
##栅格层面，'性别:男'， ['年龄阶段:18以下', '年龄阶段:18-24']， '年龄阶段:25-34', '年龄阶段:35-44', '年龄阶段:45-54', '年龄阶段:55-64', '年龄阶段:65以上','教育水平:高中及以下', '教育水平:大专', '教育水平:本科及以上','资产:有车',
## 行政区层面：家庭：成员数量 个人：年龄、性别
## 北京市层面：家庭：成员数量，工作人口数量 个人：年龄、性别、教育
grid_population_profile[
    ['pixel_value', '性别:男', '年龄阶段:18以下', '年龄阶段:18-24', '年龄阶段:25-34', '年龄阶段:35-44',
     '年龄阶段:45-54', '年龄阶段:55-64', '年龄阶段:65以上', '教育水平:高中及以下', '教育水平:大专',
     '教育水平:本科及以上', '资产:有车', 'local_name']]
grid_population_profile['age_from_0_to_24'] = grid_population_profile['年龄阶段:18以下'] + grid_population_profile[
    '年龄阶段:18-24']
grid_control = grid_population_profile[
    ['pixel_value', '性别:男', 'age_from_0_to_24', '年龄阶段:25-34', '年龄阶段:35-44', '年龄阶段:45-54',
     '年龄阶段:55-64', '年龄阶段:65以上', '教育水平:高中及以下', '教育水平:大专', '教育水平:本科及以上', '资产:有车',
     'local_name']]
grid_control.rename(
    columns={'性别:男': 'male_ratio', '年龄阶段:25-34': 'age_from_25_to_34', '年龄阶段:35-44': 'age_from_35_to_44',
             '年龄阶段:45-54': 'age_from_45_to_54', '年龄阶段:55-64': 'age_from_55_to_64',
             '年龄阶段:65以上': 'age_65_above', '教育水平:高中及以下': 'education_high_school_below',
             '教育水平:大专': 'education_poly', '教育水平:本科及以上': 'education_above_colledge',
             '资产:有车': 'have_car_ratio'}, inplace=True)
local_name_mapping = {"顺义区 / Shunyi": "顺义区", "大兴区 / Daxing": "大兴区", "通州区 / Tongzhou": "通州区",
                      "延庆县 / Yanqing": "延庆区", "怀柔区 / Huairou": "怀柔区", "平谷区 / Pinggu": "平谷区",
                      "朝阳区 / Chaoyang": "朝阳区", "门头沟区 / Mentougou": "门头沟区", "丰台区 / Fengtai": "丰台区"
                      }
grid_control['local_name'] = grid_control['local_name'].replace(local_name_mapping)
district_control = {
    '东城区': {'mean_family_size': 2.87, 'age_from_0_to_14': 0.138, 'age_from_15_to_64': 0.656, 'age_65_above': 0.206,
               'male_ratio': 0.486},
    '西城区': {'mean_family_size': 3.08, 'age_from_0_to_14': 0.148, 'age_from_15_to_64': 0.634, 'age_65_above': 0.217,
               'male_ratio': 0.491},
    '朝阳区': {'mean_family_size': 2.54, 'age_from_0_to_14': 0.113, 'age_from_15_to_64': 0.719, 'age_65_above': 0.168,
               'male_ratio': 0.492},
    '丰台区': {'mean_family_size': 2.39, 'age_from_0_to_14': 0.112, 'age_from_15_to_64': 0.705, 'age_65_above': 0.183,
               'male_ratio': 0.497},
    '石景山区': {'mean_family_size': 2.52, 'age_from_0_to_14': 0.115, 'age_from_15_to_64': 0.715, 'age_65_above': 0.170,
                 'male_ratio': 0.501},
    '海淀区': {'mean_family_size': 3.21, 'age_from_0_to_14': 0.119, 'age_from_15_to_64': 0.726, 'age_65_above': 0.155,
               'male_ratio': 0.498},
    '门头沟区': {'mean_family_size': 2.05, 'age_from_0_to_14': 0.113, 'age_from_15_to_64': 0.695, 'age_65_above': 0.191,
                 'male_ratio': 0.5},
    '房山区': {'mean_family_size': 2.12, 'age_from_0_to_14': 0.133, 'age_from_15_to_64': 0.729, 'age_65_above': 0.139,
               'male_ratio': 0.498},
    '通州区': {'mean_family_size': 2.06, 'age_from_0_to_14': 0.125, 'age_from_15_to_64': 0.738, 'age_65_above': 0.137,
               'male_ratio': 0.493},
    '顺义区': {'mean_family_size': 2.34, 'age_from_0_to_14': 0.117, 'age_from_15_to_64': 0.743, 'age_65_above': 0.140,
               'male_ratio': 0.493},
    '昌平区': {'mean_family_size': 2.26, 'age_from_0_to_14': 0.110, 'age_from_15_to_64': 0.786, 'age_65_above': 0.104,
               'male_ratio': 0.498},
    '大兴区': {'mean_family_size': 2.52, 'age_from_0_to_14': 0.123, 'age_from_15_to_64': 0.729, 'age_65_above': 0.148,
               'male_ratio': 0.495},
    '怀柔区': {'mean_family_size': 2.08, 'age_from_0_to_14': 0.118, 'age_from_15_to_64': 0.693, 'age_65_above': 0.189,
               'male_ratio': 0.495},
    '平谷区': {'mean_family_size': 2.31, 'age_from_0_to_14': 0.129, 'age_from_15_to_64': 0.671, 'age_65_above': 0.2,
               'male_ratio': 0.502},
    '密云区': {'mean_family_size': 2.13, 'age_from_0_to_14': 0.124, 'age_from_15_to_64': 0.698, 'age_65_above': 0.177,
               'male_ratio': 0.495},
    '延庆区': {'mean_family_size': 1.95, 'age_from_0_to_14': 0.117, 'age_from_15_to_64': 0.708, 'age_65_above': 0.175,
               'male_ratio': 0.502}}
beijing_control = {'one_member': 0.268, 'two_member': 0.342, 'three_member': 0.223, 'four_member': 0.097,
                   'above_five_member': 0.070, 'mean_worker_number': 1.1, 'male_ratio': 0.509, 'age_from_0_to_4': 0.034,
                   'age_from_5_to_9': 0.049, 'age_from_10_to_14': 0.037, 'age_from_15_to_19': 0.026,
                   'age_from_20_to_24': 0.048, 'age_from_25_to_29': 0.075, 'age_from_30_to_34': 0.101,
                   'age_from_35_to_39': 0.104, 'age_from_40_to_44': 0.086, 'age_from_45_to_49': 0.065,
                   'age_from_50_to_54': 0.078, 'age_from_55_to_59': 0.071, 'age_from_60_to_64': 0.068,
                   'age_from_65_to_69': 0.063, 'age_from_70_to_74': 0.042, 'age_from_75_to_79': 0.022,
                   'age_from_80_to_84': 0.015, 'age_above_85': 0.016, 'education_primary_and_below': 0.053,
                   'education_junior_high': 0.193, 'education_high_school': 0.172, 'education_poly': 0.208,
                   'education_college': 0.276, 'education_bachelor_and_above': 0.081}
grid_control_test = grid_control[['male_ratio', 'age_from_0_to_24', 'age_from_25_to_34',
                                  'age_from_35_to_44', 'age_from_45_to_54', 'age_from_55_to_64',
                                  'age_65_above', 'education_high_school_below', 'education_poly',
                                  'education_above_colledge', 'have_car_ratio']].values[0]
grid_control_test = torch.tensor(grid_control_test)
grid_family_num = 0
grid_person_num = 0
current_grid_distribution_test = torch.zeros_like(grid_control_test)
district_control_test = torch.tensor(list(district_control['延庆区'].values()))
district_family_num = 0
district_person_num = 0
current_district_distribution_test = torch.zeros_like(district_control_test)
beijing_control_test = torch.tensor(list(beijing_control.values()))
current_beijing_distribution_test = torch.zeros_like(beijing_control_test)
beijing_family_num = 0
beijing_person_num = 0
x_family = torch.randn(10, 10)
x_person = torch.randn(10, 8, 51)
## 输入的参数应该有一个噪声的列表[x_family, x_person],对应的t，输入的grid_person_num, grid_family_num, current_grid_distribution, grid_control, district_person_num, district_family_num, current_district_distribution, district_control, beijing_person_num, beijing_family_num, current_beijing_distribution,beijing_control
with torch.enable_grad():
    ## 先对应栅格的

    x_family = x_family.detach().requires_grad_(True)
    x_person = x_person.detach().requires_grad_(True)

    valid_person = x_family[:, 0]
    valid_person = torch.round(valid_person * 0.88397094 + 2.38862088)
    person_mask = torch.zeros(x_family.shape[0], 8, device=x_family.device).to(torch.bool)
    for i in range(valid_person.shape[0]):
        valid_member = int(valid_person[i].item())
        person_mask[i, :valid_member] = True

    valid_person = x_person[person_mask]
    # 栅格损失
    ## 栅格的男性比例损失
    male_num = valid_person[:, 2].sum()
    new_male_ratio_grid = (current_grid_distribution_test[0] * grid_person_num + male_num) / (
                grid_person_num + valid_person.shape[0])
    grid_male_loss = -F.mse_loss(new_male_ratio_grid, grid_control_test[0])

    # grid_male_grad = torch.autograd.grad(outputs=grid_male_loss, inputs=x_person)[0]
    ## 栅格的年龄损失
    tau = 0.1  # 越小越接近硬阈值，越大越平滑
    value = valid_person[:, 0] * 3.49238421 + 8.40812411
    mask_soft = torch.sigmoid(-(value - 4.5) / tau)
    age_from_0_to_24_num = mask_soft.sum()
    new_age_from_0_to_24_ratio_grid = (current_grid_distribution_test[1] * grid_person_num + age_from_0_to_24_num) / (
                grid_person_num + valid_person.shape[0])
    grid_age_from_0_to_24_loss = -F.mse_loss(new_age_from_0_to_24_ratio_grid, grid_control_test[1])
    ## 栅格的年龄损失25-34
    lower = 4.5
    upper = 6.5

    mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

    age_from_25_to_34_num = mask_soft.sum()
    new_age_from_25_to_34_ratio_grid = (current_grid_distribution_test[2] * grid_person_num + age_from_25_to_34_num) / (
                grid_person_num + valid_person.shape[0])
    grid_age_from_25_to_34_loss = -F.mse_loss(new_age_from_25_to_34_ratio_grid, grid_control_test[2])

    ## 栅格的年龄损失35-44
    lower = 6.5
    upper = 8.5

    mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

    age_from_35_to_44_num = mask_soft.sum()
    new_age_from_35_to_44_ratio_grid = (current_grid_distribution_test[3] * grid_person_num + age_from_35_to_44_num) / (
                grid_person_num + valid_person.shape[0])
    grid_age_from_35_to_44_loss = -F.mse_loss(new_age_from_35_to_44_ratio_grid, grid_control_test[3])

    ## 栅格的年龄损失45-54
    lower = 8.5
    upper = 10.5

    mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

    age_from_45_to_54_num = mask_soft.sum()
    new_age_from_45_to_54_ratio_grid = (current_grid_distribution_test[4] * grid_person_num + age_from_45_to_54_num) / (
                grid_person_num + valid_person.shape[0])
    grid_age_from_45_to_54_loss = -F.mse_loss(new_age_from_45_to_54_ratio_grid, grid_control_test[4])

    ## 栅格的年龄损失55-64
    lower = 10.5
    upper = 12.5

    mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

    age_from_55_to_64_num = mask_soft.sum()
    new_age_from_55_to_64_ratio_grid = (current_grid_distribution_test[5] * grid_person_num + age_from_55_to_64_num) / (
                grid_person_num + valid_person.shape[0])
    grid_age_from_55_to_64_loss = -F.mse_loss(new_age_from_55_to_64_ratio_grid, grid_control_test[5])

    ## 栅格的年龄损失大于65
    lower = 12.5

    mask_soft = torch.sigmoid((value - lower) / tau)
    age_above_65_num = mask_soft.sum()
    new_age_above_65_ratio_grid = (current_grid_distribution_test[6] * grid_person_num + age_above_65_num) / (
                grid_person_num + valid_person.shape[0])
    grid_age_above_65_loss = -F.mse_loss(new_age_above_65_ratio_grid, grid_control_test[6])

    ## 栅格的教育
    education_value = valid_person[:, 21:30]
    education_below_high_school = education_value[:, :4].sum()
    new_education_below_high_school_ratio_grid = (current_grid_distribution_test[
                                                      7] * grid_person_num + education_below_high_school) / (
                                                             grid_person_num + valid_person.shape[0])
    grid_education_below_high_school_loss = -F.mse_loss(new_education_below_high_school_ratio_grid,
                                                        grid_control_test[7])

    education_poly = education_value[:, 4:6].sum()
    new_education_poly_ratio_grid = (current_grid_distribution_test[8] * grid_person_num + education_poly) / (
                grid_person_num + valid_person.shape[0])
    grid_education_poly_loss = -F.mse_loss(new_education_poly_ratio_grid, grid_control_test[8])

    education_above_colledge = education_value[:, 6:].sum()
    new_education_above_colledge_ratio_grid = (current_grid_distribution_test[
                                                   9] * grid_person_num + education_above_colledge) / (
                                                          grid_person_num + valid_person.shape[0])
    grid_education_above_colledge_loss = -F.mse_loss(new_education_above_colledge_ratio_grid, grid_control_test[9])

    ## 栅格的有车比例损失
    car_number = x_family[2] * 0.53264196 + 0.50132666
    lower = 0.5

    mask_soft = torch.sigmoid((car_number - lower) / tau)
    have_car_person = mask_soft * (x_family[:, 0] * 0.88397094 + 2.38862088)
    have_car_person = have_car_person.sum()
    new_have_car_person_ratio_grid = (current_grid_distribution_test[10] * grid_person_num + have_car_person) / (
                grid_person_num + valid_person.shape[0])
    grid_have_car_person_loss = -F.mse_loss(new_have_car_person_ratio_grid, grid_control_test[10])

# 行政区损失
## 家庭规模的损失
family_size = x_family[:, 0] * 0.88397094 + 2.38862088
new_family_size_district = (current_district_distribution_test[0] * district_family_num + family_size.sum()) / (
            district_family_num + x_family.shape[0])
district_family_size_loss = -F.mse_loss(new_family_size_district, district_control_test[0])

## 男性比例损失
male_num = valid_person[:, 2].sum()
new_male_ratio_district = (current_district_distribution_test[4] * district_person_num + male_num) / (
            district_person_num + valid_person.shape[0])
district_male_loss = -F.mse_loss(new_male_ratio_district, district_control_test[4])
## 行政区的年龄损失

## 年龄损失0-14
tau = 0.1  # 越小越接近硬阈值，越大越平滑
value = valid_person[:, 0] * 3.49238421 + 8.40812411
mask_soft = torch.sigmoid(-(value - 2.5) / tau)
age_from_0_to_14_num = mask_soft.sum()
new_age_from_0_to_14_ratio_district = (current_district_distribution_test[
                                           1] * district_person_num + age_from_0_to_14_num) / (
                                                  district_person_num + valid_person.shape[0])
district_age_from_0_to_14_loss = -F.mse_loss(new_age_from_0_to_14_ratio_district, district_control_test[1])
## 行政区的年龄损失14-64
lower = 2.5
upper = 12.5

mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

age_from_15_to_64_num = mask_soft.sum()
new_age_from_15_to_64_ratio_district = (current_district_distribution_test[
                                            2] * district_person_num + age_from_15_to_64_num) / (
                                                   district_person_num + valid_person.shape[0])
district_age_from_15_to_64_loss = -F.mse_loss(new_age_from_15_to_64_ratio_district, district_control_test[2])

## 行政区的年龄损失大于65
lower = 12.5

mask_soft = torch.sigmoid((value - lower) / tau)
age_above_65_num = mask_soft.sum()
new_age_above_65_ratio_district = (current_district_distribution_test[3] * district_person_num + age_above_65_num) / (
            district_person_num + valid_person.shape[0])
district_age_above_65_loss = -F.mse_loss(new_age_above_65_ratio_district, district_control_test[3])

## 北京市损失
## 家庭规模的损失
tau = 0.1  # 越小越接近硬阈值，越大越平滑
value = x_family[:, 0] * 0.88397094 + 2.38862088
# 家庭规模为1人户
mask_soft = torch.sigmoid(-(value - 1.5) / tau)
new_family_size_beijing = (current_beijing_distribution_test[0] * beijing_family_num + mask_soft.sum()) / (
            beijing_family_num + x_family.shape[0])
beijing_family_size1_loss = -F.mse_loss(new_family_size_beijing, beijing_control_test[0])

## 家庭规模为2人户
lower = 1.5
upper = 2.5

mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

familysize2_num = mask_soft.sum()
new_family_size2_beijing = (current_beijing_distribution_test[1] * beijing_family_num + familysize2_num) / (
            beijing_family_num + x_family.shape[0])
beijing_family_size2_loss = -F.mse_loss(new_family_size2_beijing, beijing_control_test[1])

## 家庭规模为3人户
lower = 2.5
upper = 3.5

mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

familysize3_num = mask_soft.sum()
new_family_size3_beijing = (current_beijing_distribution_test[2] * beijing_family_num + familysize3_num) / (
            beijing_family_num + x_family.shape[0])
beijing_family_size3_loss = -F.mse_loss(new_family_size3_beijing, beijing_control_test[2])

## 家庭规模为4人户
lower = 3.5
upper = 4.5

mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

familysize4_num = mask_soft.sum()
new_family_size4_beijing = (current_beijing_distribution_test[3] * beijing_family_num + familysize4_num) / (
            beijing_family_num + x_family.shape[0])
beijing_family_size4_loss = -F.mse_loss(new_family_size4_beijing, beijing_control_test[3])

## 家庭规模为5人户
lower = 4.5

mask_soft = torch.sigmoid((value - lower) / tau)

familysize5_num = mask_soft.sum()
new_family_size5_beijing = (current_beijing_distribution_test[4] * beijing_family_num + familysize5_num) / (
            beijing_family_num + x_family.shape[0])
beijing_family_size5_loss = -F.mse_loss(new_family_size5_beijing, beijing_control_test[4])

## 家庭平均工作人口损失
family_worker = x_family[:, 1] * 0.77530306 + 1.36621842
new_family_worker_beijing = (current_beijing_distribution_test[5] * beijing_family_num + family_worker.sum()) / (
            beijing_family_num + x_family.shape[0])
beijing_family_worker_loss = -F.mse_loss(new_family_worker_beijing, beijing_control_test[5])
## 北京市年龄相关损失
beijing_age_thresholds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5,
                          20]
tau = 0.1  # 越小越接近硬阈值，越大越平滑
value = valid_person[:, 0] * 3.49238421 + 8.40812411
beijing_age_loss = 0
for index, age in enumerate(beijing_age_thresholds):
    if (index + 1) == len(beijing_age_thresholds):
        break
    lower = age
    upper = beijing_age_thresholds[index + 1]

    mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

    age_num = mask_soft.sum()
    new_age_ratio_beijing = (current_beijing_distribution_test[index + 7] * beijing_person_num + age_num) / (
                beijing_person_num + valid_person.shape[0])
    beijing_age_loss += -F.mse_loss(new_age_ratio_beijing, beijing_control_test[index + 7])
## 北京市性别损失
male_num = valid_person[:, 2].sum()
new_male_ratio_beijing = (current_beijing_distribution_test[6] * beijing_person_num + male_num) / (
            beijing_person_num + valid_person.shape[0])
beijing_male_loss = -F.mse_loss(new_male_ratio_beijing, beijing_control_test[6])
## 北京市教育相关损失
beijing_education_thresholds = [0, 2, 3, 4, 6, 7, 9]

tau = 0.1  # 越小越接近硬阈值，越大越平滑
education_value = valid_person[:, 21:30]
beijing_education_loss = 0
for index, education in enumerate(beijing_education_thresholds):
    if (index + 1) == len(beijing_education_thresholds):
        break
    education_beijing = education_value[:, education:beijing_education_thresholds[index + 1]].sum()

    education_num = education_beijing.sum()
    new_edu_ratio_beijing = (current_beijing_distribution_test[index + 25] * beijing_person_num + education_num) / (
                beijing_person_num + valid_person.shape[0])
    beijing_education_loss += -F.mse_loss(new_edu_ratio_beijing, beijing_control_test[index + 25])

beijing_family_size_loss = beijing_family_size1_loss + beijing_family_size2_loss + beijing_family_size3_loss + beijing_family_size4_loss + beijing_family_size5_loss

## 共同项权重计算
family_size_weights_scales = F.softmax(torch.tensor([-district_family_size_loss, -(
            beijing_family_size1_loss + beijing_family_size2_loss + beijing_family_size3_loss + beijing_family_size4_loss + beijing_family_size5_loss)]),
                                       dim=0)

person_age_weights_scales = F.softmax(torch.tensor([-(
            grid_age_from_0_to_24_loss + grid_age_from_25_to_34_loss + grid_age_from_35_to_44_loss + grid_age_from_45_to_54_loss + grid_age_from_55_to_64_loss + grid_age_above_65_loss),
                                                    -(
                                                                district_age_from_0_to_14_loss + district_age_from_15_to_64_loss + district_age_above_65_loss),
                                                    -beijing_age_loss]), dim=0)

person_gender_weights_scales = F.softmax(torch.tensor([-grid_male_loss, -district_male_loss, -beijing_male_loss]),
                                         dim=0)

person_education_weights_scales = F.softmax(torch.tensor(
    [-(grid_education_below_high_school_loss + grid_education_poly_loss + grid_education_above_colledge_loss),
     -beijing_education_loss]), dim=0)

##尺度内权重计算

within_grid_weights = F.softmax(torch.tensor(
    [-grid_male_loss, -grid_age_from_0_to_24_loss, -grid_age_from_25_to_34_loss, -grid_age_from_35_to_44_loss,
     -grid_age_from_45_to_54_loss, -grid_age_from_55_to_64_loss, -grid_age_above_65_loss,
     -grid_education_below_high_school_loss, -grid_education_poly_loss, -grid_education_above_colledge_loss,
     -grid_have_car_person_loss]), dim=0)

within_district_weights = F.softmax(torch.tensor(
    [-district_male_loss, -district_age_from_0_to_14_loss, -district_age_from_15_to_64_loss,
     -district_age_above_65_loss]), dim=0)

within_beijing_family_weights = F.softmax(torch.tensor(
    [-beijing_family_size1_loss, -beijing_family_size2_loss, -beijing_family_size3_loss, -beijing_family_size4_loss,
     -beijing_family_size5_loss, -beijing_family_worker_loss]), dim=0)

within_beijing_person_weights = F.softmax(
    torch.tensor([-beijing_age_loss, -beijing_male_loss, -beijing_education_loss]), dim=0)

## 栅格的损失加权计算

grid_male_loss = person_gender_weights_scales[0] * within_grid_weights[0] * grid_male_loss

grid_age_from_0_to_24_loss = person_age_weights_scales[0] * within_grid_weights[1] * grid_age_from_0_to_24_loss
grid_age_from_25_to_34_loss = person_age_weights_scales[0] * within_grid_weights[2] * grid_age_from_25_to_34_loss
grid_age_from_35_to_44_loss = person_age_weights_scales[0] * within_grid_weights[3] * grid_age_from_35_to_44_loss
grid_age_from_45_to_54_loss = person_age_weights_scales[0] * within_grid_weights[4] * grid_age_from_45_to_54_loss
grid_age_from_55_to_64_loss = person_age_weights_scales[0] * within_grid_weights[5] * grid_age_from_55_to_64_loss
grid_age_above_65_loss = person_age_weights_scales[0] * within_grid_weights[6] * grid_age_above_65_loss

grid_education_below_high_school_loss = person_education_weights_scales[0] * within_grid_weights[
    7] * grid_education_below_high_school_loss
grid_education_poly_loss = person_education_weights_scales[0] * within_grid_weights[8] * grid_education_poly_loss
grid_education_above_colledge_loss = person_education_weights_scales[0] * within_grid_weights[
    9] * grid_education_above_colledge_loss

grid_have_car_person_loss = within_grid_weights[10] * grid_have_car_person_loss

## 行政区的损失加权计算

district_family_size_loss = family_size_weights_scales[0] * district_family_size_loss

district_male_loss = person_gender_weights_scales[1] * within_district_weights[0] * district_male_loss
district_age_from_0_to_14_loss = person_age_weights_scales[1] * within_district_weights[
    1] * district_age_from_0_to_14_loss
district_age_from_15_to_64_loss = person_age_weights_scales[1] * within_district_weights[
    2] * district_age_from_15_to_64_loss
district_age_above_65_loss = person_age_weights_scales[1] * within_district_weights[3] * district_age_above_65_loss

## 北京市的损失加权计算
beijing_family_size1_loss = family_size_weights_scales[1] * within_beijing_family_weights[0] * beijing_family_size1_loss
beijing_family_size2_loss = family_size_weights_scales[1] * within_beijing_family_weights[1] * beijing_family_size2_loss
beijing_family_size3_loss = family_size_weights_scales[1] * within_beijing_family_weights[2] * beijing_family_size3_loss
beijing_family_size4_loss = family_size_weights_scales[1] * within_beijing_family_weights[3] * beijing_family_size4_loss
beijing_family_size5_loss = family_size_weights_scales[1] * within_beijing_family_weights[4] * beijing_family_size5_loss
beijing_family_worker_loss = within_beijing_family_weights[5] * beijing_family_worker_loss

beijing_age_loss = person_age_weights_scales[2] * within_beijing_person_weights[0] * beijing_age_loss
beijing_male_loss = person_gender_weights_scales[2] * within_beijing_person_weights[1] * beijing_male_loss
beijing_education_loss = person_education_weights_scales[1] * within_beijing_person_weights[2] * beijing_education_loss

## 汇总栅格的损失
grid_loss_total = grid_male_loss + grid_age_from_0_to_24_loss + grid_age_from_25_to_34_loss + grid_age_from_35_to_44_loss + grid_age_from_45_to_54_loss + grid_age_from_55_to_64_loss + grid_age_above_65_loss + grid_education_below_high_school_loss + grid_education_poly_loss + grid_education_above_colledge_loss + grid_have_car_person_loss

## 汇总行政区的损失
district_loss_total_family = district_family_size_loss
district_loss_total_person = district_male_loss + district_age_from_0_to_14_loss + district_age_from_15_to_64_loss + district_age_above_65_loss

## 北京市损失汇总
beijing_loss_total_family = beijing_family_size1_loss + beijing_family_size2_loss + beijing_family_size3_loss + beijing_family_size4_loss + beijing_family_size5_loss + beijing_family_worker_loss
beijing_loss_total_person = beijing_age_loss + beijing_male_loss + beijing_education_loss

grid_gradient = torch.autograd.grad(outputs=grid_loss_total, inputs=x_person, retain_graph=True)[0]

district_gradient_family = torch.autograd.grad(outputs=district_loss_total_family, inputs=x_family, retain_graph=True)[
    0]
district_gradient_person = torch.autograd.grad(outputs=district_loss_total_person, inputs=x_person, retain_graph=True)[
    0]

beijing_gradient_family = torch.autograd.grad(outputs=beijing_loss_total_family, inputs=x_family, retain_graph=True)[0]
beijing_gradient_person = torch.autograd.grad(outputs=beijing_loss_total_person, inputs=x_person, retain_graph=True)[0]

family_gradient = district_gradient_family + beijing_gradient_family
person_gradient = grid_gradient + district_gradient_person + beijing_gradient_person


## 引导函数整理
## 输入的参数应该有一个噪声的列表[x_family, x_person],对应的t，输入的grid_person_num, grid_family_num, current_grid_distribution, grid_control, district_person_num, district_family_num, current_district_distribution, district_control, beijing_person_num, beijing_family_num, current_beijing_distribution,beijing_control

def multi_scale_guide_function(x, t, grid_person_num, current_grid_distribution_test, grid_control_test,
                               district_person_num, district_family_num, current_district_distribution_test,
                               district_control_test, beijing_person_num, beijing_family_num,
                               current_beijing_distribution_test, beijing_control_test, **kwargs):
    lambda_grid = 2
    lambda_district = 1
    lambda_beijing = 0.5
    alpha = 0.05
    x_family = x[0]
    x_person = x[1]
    with torch.enable_grad():
        ## 先对应栅格的

        x_family = x_family.detach().requires_grad_(True)
        x_person = x_person.detach().requires_grad_(True)

        valid_person = x_family[:, 0]
        valid_person = torch.round(valid_person * 0.88397094 + 2.38862088)
        person_mask = torch.zeros(x_family.shape[0], 8, device=x_family.device).to(torch.bool)
        for i in range(valid_person.shape[0]):
            valid_member = int(valid_person[i].item())
            person_mask[i, :valid_member] = True

        valid_person = x_person[person_mask]
        # 栅格损失
        ## 栅格的男性比例损失
        male_num = valid_person[:, 2].sum()
        new_male_ratio_grid = (current_grid_distribution_test[0] * grid_person_num + male_num) / (
                    grid_person_num + valid_person.shape[0])
        grid_male_loss = (-F.mse_loss(new_male_ratio_grid, grid_control_test[0])) * (lambda_grid + alpha * (new_male_ratio_grid - grid_control_test[0]).abs().item())

        # grid_male_grad = torch.autograd.grad(outputs=grid_male_loss, inputs=x_person)[0]
        ## 栅格的年龄损失
        tau = 0.1  # 越小越接近硬阈值，越大越平滑
        value = valid_person[:, 0] * 3.49238421 + 8.40812411
        mask_soft = torch.sigmoid(-(value - 4.5) / tau)
        age_from_0_to_24_num = mask_soft.sum()
        new_age_from_0_to_24_ratio_grid = (current_grid_distribution_test[
                                               1] * grid_person_num + age_from_0_to_24_num) / (
                                                      grid_person_num + valid_person.shape[0])
        grid_age_from_0_to_24_loss = -F.mse_loss(new_age_from_0_to_24_ratio_grid, grid_control_test[1]) * (lambda_grid + alpha * (new_age_from_0_to_24_ratio_grid - grid_control_test[1]).abs().item())

        ## 栅格的年龄损失25-34
        lower = 4.5
        upper = 6.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        age_from_25_to_34_num = mask_soft.sum()
        new_age_from_25_to_34_ratio_grid = (current_grid_distribution_test[
                                                2] * grid_person_num + age_from_25_to_34_num) / (
                                                       grid_person_num + valid_person.shape[0])
        grid_age_from_25_to_34_loss = -F.mse_loss(new_age_from_25_to_34_ratio_grid, grid_control_test[2]) * (lambda_grid + alpha * (new_age_from_25_to_34_ratio_grid - grid_control_test[2]).abs().item())

        ## 栅格的年龄损失35-44
        lower = 6.5
        upper = 8.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        age_from_35_to_44_num = mask_soft.sum()
        new_age_from_35_to_44_ratio_grid = (current_grid_distribution_test[
                                                3] * grid_person_num + age_from_35_to_44_num) / (
                                                       grid_person_num + valid_person.shape[0])
        grid_age_from_35_to_44_loss = -F.mse_loss(new_age_from_35_to_44_ratio_grid, grid_control_test[3]) * (lambda_grid + alpha * (new_age_from_35_to_44_ratio_grid - grid_control_test[3]).abs().item())

        ## 栅格的年龄损失45-54
        lower = 8.5
        upper = 10.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        age_from_45_to_54_num = mask_soft.sum()
        new_age_from_45_to_54_ratio_grid = (current_grid_distribution_test[
                                                4] * grid_person_num + age_from_45_to_54_num) / (
                                                       grid_person_num + valid_person.shape[0])
        grid_age_from_45_to_54_loss = -F.mse_loss(new_age_from_45_to_54_ratio_grid, grid_control_test[4]) * (lambda_grid + alpha * (new_age_from_45_to_54_ratio_grid - grid_control_test[4]).abs().item())

        ## 栅格的年龄损失55-64
        lower = 10.5
        upper = 12.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        age_from_55_to_64_num = mask_soft.sum()
        new_age_from_55_to_64_ratio_grid = (current_grid_distribution_test[
                                                5] * grid_person_num + age_from_55_to_64_num) / (
                                                       grid_person_num + valid_person.shape[0])
        grid_age_from_55_to_64_loss = -F.mse_loss(new_age_from_55_to_64_ratio_grid, grid_control_test[5]) * (lambda_grid + alpha * (new_age_from_55_to_64_ratio_grid - grid_control_test[5]).abs().item())

        ## 栅格的年龄损失大于65
        lower = 12.5

        mask_soft = torch.sigmoid((value - lower) / tau)
        age_above_65_num = mask_soft.sum()
        new_age_above_65_ratio_grid = (current_grid_distribution_test[6] * grid_person_num + age_above_65_num) / (
                    grid_person_num + valid_person.shape[0])
        grid_age_above_65_loss = -F.mse_loss(new_age_above_65_ratio_grid, grid_control_test[6]) * (lambda_grid + alpha * (new_age_above_65_ratio_grid - grid_control_test[6]).abs().item())

        ## 栅格的教育
        education_value = valid_person[:, 21:30]
        education_below_high_school = education_value[:, :4].sum()
        new_education_below_high_school_ratio_grid = (current_grid_distribution_test[
                                                          7] * grid_person_num + education_below_high_school) / (
                                                                 grid_person_num + valid_person.shape[0])
        grid_education_below_high_school_loss = -F.mse_loss(new_education_below_high_school_ratio_grid,
                                                            grid_control_test[7]) * (lambda_grid + alpha * (new_education_below_high_school_ratio_grid - grid_control_test[7]).abs().item())

        education_poly = education_value[:, 4:6].sum()
        new_education_poly_ratio_grid = (current_grid_distribution_test[8] * grid_person_num + education_poly) / (
                    grid_person_num + valid_person.shape[0])
        grid_education_poly_loss = -F.mse_loss(new_education_poly_ratio_grid, grid_control_test[8]) * (lambda_grid + alpha * (new_education_poly_ratio_grid - grid_control_test[8]).abs().item())

        education_above_colledge = education_value[:, 6:].sum()
        new_education_above_colledge_ratio_grid = (current_grid_distribution_test[
                                                       9] * grid_person_num + education_above_colledge) / (
                                                              grid_person_num + valid_person.shape[0])
        grid_education_above_colledge_loss = -F.mse_loss(new_education_above_colledge_ratio_grid, grid_control_test[9]) * (lambda_grid + alpha * (new_education_above_colledge_ratio_grid - grid_control_test[9]).abs().item())

        ## 栅格的有车比例损失
        car_number = x_family[:, 2] * 0.53264196 + 0.50132666
        lower = 0.5

        mask_soft = torch.sigmoid((car_number - lower) / tau)
        have_car_person = mask_soft * (x_family[:, 0] * 0.88397094 + 2.38862088)
        have_car_person = have_car_person.sum()
        new_have_car_person_ratio_grid = (current_grid_distribution_test[10] * grid_person_num + have_car_person) / (
                    grid_person_num + valid_person.shape[0])
        grid_have_car_person_loss = -F.mse_loss(new_have_car_person_ratio_grid, grid_control_test[10]) * (lambda_grid + alpha * (new_have_car_person_ratio_grid - grid_control_test[10]).abs().item())

        # 行政区损失
        ## 家庭规模的损失
        family_size = x_family[:, 0] * 0.88397094 + 2.38862088
        new_family_size_district = (current_district_distribution_test[0] * district_family_num + family_size.sum()) / (
                    district_family_num + x_family.shape[0])
        district_family_size_loss = -F.mse_loss(new_family_size_district, district_control_test[0]) * (lambda_district + alpha * (new_family_size_district - district_control_test[0]).abs().item())

        ## 男性比例损失
        male_num = valid_person[:, 2].sum()
        new_male_ratio_district = (current_district_distribution_test[4] * district_person_num + male_num) / (
                    district_person_num + valid_person.shape[0])
        district_male_loss = -F.mse_loss(new_male_ratio_district, district_control_test[4]) * (lambda_district + alpha * (new_male_ratio_district - district_control_test[4]).abs().item())
        ## 行政区的年龄损失

        ## 年龄损失0-14
        tau = 0.1  # 越小越接近硬阈值，越大越平滑
        value = valid_person[:, 0] * 3.49238421 + 8.40812411
        mask_soft = torch.sigmoid(-(value - 2.5) / tau)
        age_from_0_to_14_num = mask_soft.sum()
        new_age_from_0_to_14_ratio_district = (current_district_distribution_test[
                                                   1] * district_person_num + age_from_0_to_14_num) / (
                                                          district_person_num + valid_person.shape[0])
        district_age_from_0_to_14_loss = -F.mse_loss(new_age_from_0_to_14_ratio_district, district_control_test[1]) * (lambda_district + alpha * (new_age_from_0_to_14_ratio_district - district_control_test[1]).abs().item())
        ## 行政区的年龄损失14-64
        lower = 2.5
        upper = 12.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        age_from_15_to_64_num = mask_soft.sum()
        new_age_from_15_to_64_ratio_district = (current_district_distribution_test[
                                                    2] * district_person_num + age_from_15_to_64_num) / (
                                                           district_person_num + valid_person.shape[0])
        district_age_from_15_to_64_loss = -F.mse_loss(new_age_from_15_to_64_ratio_district, district_control_test[2]) * (lambda_district + alpha * (new_age_from_15_to_64_ratio_district - district_control_test[2]).abs().item())

        ## 行政区的年龄损失大于65
        lower = 12.5

        mask_soft = torch.sigmoid((value - lower) / tau)
        age_above_65_num = mask_soft.sum()
        new_age_above_65_ratio_district = (current_district_distribution_test[
                                               3] * district_person_num + age_above_65_num) / (
                                                      district_person_num + valid_person.shape[0])
        district_age_above_65_loss = -F.mse_loss(new_age_above_65_ratio_district, district_control_test[3]) * (lambda_district + alpha * (new_age_above_65_ratio_district - district_control_test[3]).abs().item())

        ## 北京市损失
        ## 家庭规模的损失
        tau = 0.1  # 越小越接近硬阈值，越大越平滑
        value = x_family[:, 0] * 0.88397094 + 2.38862088
        # 家庭规模为1人户
        mask_soft = torch.sigmoid(-(value - 1.5) / tau)
        new_family_size_beijing = (current_beijing_distribution_test[0] * beijing_family_num + mask_soft.sum()) / (
                    beijing_family_num + x_family.shape[0])
        beijing_family_size1_loss = -F.mse_loss(new_family_size_beijing, beijing_control_test[0]) * (lambda_beijing + alpha * (new_family_size_beijing - beijing_control_test[0]).abs().item())

        ## 家庭规模为2人户
        lower = 1.5
        upper = 2.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        familysize2_num = mask_soft.sum()
        new_family_size2_beijing = (current_beijing_distribution_test[1] * beijing_family_num + familysize2_num) / (
                    beijing_family_num + x_family.shape[0])
        beijing_family_size2_loss = -F.mse_loss(new_family_size2_beijing, beijing_control_test[1]) * (lambda_beijing + alpha * (new_family_size2_beijing - beijing_control_test[1]).abs().item())

        ## 家庭规模为3人户
        lower = 2.5
        upper = 3.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        familysize3_num = mask_soft.sum()
        new_family_size3_beijing = (current_beijing_distribution_test[2] * beijing_family_num + familysize3_num) / (
                    beijing_family_num + x_family.shape[0])
        beijing_family_size3_loss = -F.mse_loss(new_family_size3_beijing, beijing_control_test[2]) * (lambda_beijing + alpha * (new_family_size3_beijing - beijing_control_test[2]).abs().item())

        ## 家庭规模为4人户
        lower = 3.5
        upper = 4.5

        mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

        familysize4_num = mask_soft.sum()
        new_family_size4_beijing = (current_beijing_distribution_test[3] * beijing_family_num + familysize4_num) / (
                    beijing_family_num + x_family.shape[0])
        beijing_family_size4_loss = -F.mse_loss(new_family_size4_beijing, beijing_control_test[3]) * (lambda_beijing + alpha * (new_family_size4_beijing - beijing_control_test[3]).abs().item())

        ## 家庭规模为5人户
        lower = 4.5

        mask_soft = torch.sigmoid((value - lower) / tau)

        familysize5_num = mask_soft.sum()
        new_family_size5_beijing = (current_beijing_distribution_test[4] * beijing_family_num + familysize5_num) / (
                    beijing_family_num + x_family.shape[0])
        beijing_family_size5_loss = -F.mse_loss(new_family_size5_beijing, beijing_control_test[4]) * (lambda_beijing + alpha * (new_family_size5_beijing - beijing_control_test[4]).abs().item())

        ## 家庭平均工作人口损失
        family_worker = x_family[:, 1] * 0.77530306 + 1.36621842
        new_family_worker_beijing = (current_beijing_distribution_test[
                                         5] * beijing_family_num + family_worker.sum()) / (
                                                beijing_family_num + x_family.shape[0])
        beijing_family_worker_loss = -F.mse_loss(new_family_worker_beijing, beijing_control_test[5]) * (lambda_beijing + alpha * (new_family_worker_beijing - beijing_control_test[5]).abs().item())
        ## 北京市年龄相关损失
        beijing_age_thresholds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5,
                                  15.5, 16.5, 20]
        tau = 0.1  # 越小越接近硬阈值，越大越平滑
        value = valid_person[:, 0] * 3.49238421 + 8.40812411
        beijing_age_loss = 0
        for index, age in enumerate(beijing_age_thresholds):
            if (index + 1) == len(beijing_age_thresholds):
                break
            lower = age
            upper = beijing_age_thresholds[index + 1]

            mask_soft = torch.sigmoid((value - lower) / tau) * torch.sigmoid((upper - value) / tau)

            age_num = mask_soft.sum()
            new_age_ratio_beijing = (current_beijing_distribution_test[index + 7] * beijing_person_num + age_num) / (
                        beijing_person_num + valid_person.shape[0])
            beijing_age_loss += -F.mse_loss(new_age_ratio_beijing, beijing_control_test[index + 7]) * (lambda_beijing + alpha * (new_age_ratio_beijing - beijing_control_test[index + 7]).abs().item())
        ## 北京市性别损失
        male_num = valid_person[:, 2].sum()
        new_male_ratio_beijing = (current_beijing_distribution_test[6] * beijing_person_num + male_num) / (
                    beijing_person_num + valid_person.shape[0])
        beijing_male_loss = -F.mse_loss(new_male_ratio_beijing, beijing_control_test[6]) * (lambda_beijing + alpha * (new_male_ratio_beijing - beijing_control_test[6]).abs().item())
        ## 北京市教育相关损失
        beijing_education_thresholds = [0, 2, 3, 4, 6, 7, 9]

        tau = 0.1  # 越小越接近硬阈值，越大越平滑
        education_value = valid_person[:, 21:30]
        beijing_education_loss = 0
        for index, education in enumerate(beijing_education_thresholds):
            if (index + 1) == len(beijing_education_thresholds):
                break
            education_beijing = education_value[:, education:beijing_education_thresholds[index + 1]].sum()

            education_num = education_beijing.sum()
            new_edu_ratio_beijing = (current_beijing_distribution_test[
                                         index + 25] * beijing_person_num + education_num) / (
                                                beijing_person_num + valid_person.shape[0])
            beijing_education_loss += -F.mse_loss(new_edu_ratio_beijing, beijing_control_test[index + 25]) * (lambda_beijing + alpha * (new_edu_ratio_beijing - beijing_control_test[index + 25]).abs().item())

       
        ## 汇总栅格的损失
        grid_loss_total = grid_male_loss + grid_age_from_0_to_24_loss + grid_age_from_25_to_34_loss + grid_age_from_35_to_44_loss + grid_age_from_45_to_54_loss + grid_age_from_55_to_64_loss + grid_age_above_65_loss + grid_education_below_high_school_loss + grid_education_poly_loss + grid_education_above_colledge_loss + grid_have_car_person_loss

        ## 汇总行政区的损失
        district_loss_total_family = district_family_size_loss
        district_loss_total_person = district_male_loss + district_age_from_0_to_14_loss + district_age_from_15_to_64_loss + district_age_above_65_loss

        ## 北京市损失汇总
        beijing_loss_total_family = beijing_family_size1_loss + beijing_family_size2_loss + beijing_family_size3_loss + beijing_family_size4_loss + beijing_family_size5_loss + beijing_family_worker_loss
        beijing_loss_total_person = beijing_age_loss + beijing_male_loss + beijing_education_loss

        grid_gradient = torch.autograd.grad(outputs=grid_loss_total, inputs=x_person, retain_graph=True)[0]

        district_gradient_family = \
        torch.autograd.grad(outputs=district_loss_total_family, inputs=x_family, retain_graph=True)[0]
        district_gradient_person = \
        torch.autograd.grad(outputs=district_loss_total_person, inputs=x_person, retain_graph=True)[0]

        beijing_gradient_family = \
        torch.autograd.grad(outputs=beijing_loss_total_family, inputs=x_family, retain_graph=True)[0]
        beijing_gradient_person = \
        torch.autograd.grad(outputs=beijing_loss_total_person, inputs=x_person, retain_graph=True)[0]

        family_gradient = district_gradient_family + beijing_gradient_family
        person_gradient = grid_gradient + district_gradient_person + beijing_gradient_person
        nan_number = torch.isnan(person_gradient).sum()
        if nan_number > 0:
            print('nan occurred')
        if t[0] > 0:
            person_gradient = torch.zeros_like(person_gradient)
    return 100*family_gradient, 100*person_gradient


## 多尺度数据生成
cluster_profile = pd.read_csv('数据/cluster_profile_improved.csv')
## 数据生成
from population_DiT_cluster11_memberbundle import *

torch.manual_seed(0)  # CPU 随机数种子
torch.cuda.manual_seed(0)  # 当前 GPU 随机数种子
torch.cuda.manual_seed_all(0)  # 所有 GPU 随机数种子
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import logging
import os
import numpy as np
import sys
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from tqdm import tqdm

sys.path.append('DiT-main')
# Now import DiTBlock from models
from models import DiTBlock, DiTBlockPerson
from diffusion import create_diffusion, create_diffusion_family, create_diffusion_all

from losses_personmask_memberbundle6 import compute_total_loss
from dataset import load_population_data, create_dataloader

cluster_info = torch.from_numpy(family2023['cluster'].values.astype(int))

# 加载模型
model_result = load_population_dit_checkpoint("results/019-PopulationDiT-1030-增加了更多条件信息/checkpoints/final.pt")
args = model_result['args']
model = model_result['model']
model.eval()
diffusion = create_diffusion_all('2', diffusion_steps=200, predict_xstart=True, learn_sigma=False)
grid_control['id'] = range(len(grid_control))
grid_control[grid_control['local_name'] == '东城区']['pixel_value'] / 2.3886
district_control.keys()
all_cluster = torch.from_numpy(family2023['cluster'].values.astype(int))
beijing_control_test = torch.tensor(list(beijing_control.values()))
current_beijing_distribution_test = torch.zeros_like(beijing_control_test)
beijing_family_num = 0
beijing_person_num = 0

for district in district_control.keys():

    district_control_test = torch.tensor(list(district_control[district].values()))
    district_family_num = 0
    district_person_num = 0
    current_district_distribution_test = torch.zeros_like(district_control_test)

    grid_control_district = grid_control[grid_control['local_name'] == district]
    start = 0
    if district == '大兴区':
        start = 0

    for grid in range(start, len(grid_control_district)):
        family_ls = []
        person_ls = []
        total_redidents = grid_control_district.iloc[grid]['pixel_value']
        grid_control_test = grid_control_district.iloc[grid][['male_ratio', 'age_from_0_to_24', 'age_from_25_to_34',
                                                              'age_from_35_to_44', 'age_from_45_to_54',
                                                              'age_from_55_to_64',
                                                              'age_65_above', 'education_high_school_below',
                                                              'education_poly',
                                                              'education_above_colledge', 'have_car_ratio']].values

        grid_control_test = torch.tensor(grid_control_test.astype(float))
        grid_person_num = 0
        current_grid_distribution_test = torch.zeros_like(grid_control_test)

        total_family = int(total_redidents / 2.3886)
        if total_family == 0:
            total_family += 1

        for _ in range(10000):
            if total_family >= 10000:
                generate_family_num = 10000
                total_family -= generate_family_num
            else:
                generate_family_num = total_family
                total_family = 0
            class_labels = all_cluster[torch.randint(0, len(all_cluster), (generate_family_num,))]

            cluster_profile_generate = []
            for i in class_labels.numpy():
                tmp = cluster_profile[cluster_profile['improved_cluster'] == i]
                profile = tmp.values[0][1:]
                cluster_profile_generate.append(profile)
            cluster_profile_generate = torch.tensor(cluster_profile_generate).to(torch.float32)

            family_features = 10  # 假设每个家庭成员有10个特征
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            max_nodes = 8  # 假设每个家庭最多有8个成员
            person_features = 51

            n = len(class_labels)
            z = torch.randn(n, 10, device=device)
            y = class_labels.clone().detach().to(device)

            # Setup classifier-free guidance:
            z_family = z

            z_person = torch.randn(n, max_nodes, person_features, device=device)

            noise_to_member = z.repeat(8, 5).view(z_person.shape[0], z_person.shape[1], -1)
            noise_to_member = torch.cat(
                [noise_to_member, torch.zeros_like(z_person[:, :, 0]).view(z_person.shape[0], z_person.shape[1], 1)],
                dim=-1)
            rho = 0.85
            z_person = noise_to_member * rho + math.sqrt(1 - rho ** 2) * z_person

            t = torch.randint(0, 200, (n,), device=device).long()

            model_kwargs = dict(cluster=y, cluster_profile=cluster_profile_generate.to(device),
                                grid_person_num=grid_person_num,
                                current_grid_distribution_test=current_grid_distribution_test.to(device),
                                grid_control_test=grid_control_test.to(device), district_person_num=district_person_num,
                                district_family_num=district_family_num,
                                current_district_distribution_test=current_district_distribution_test.to(device),
                                district_control_test=district_control_test.to(device),
                                beijing_person_num=beijing_person_num, beijing_family_num=beijing_family_num,
                                current_beijing_distribution_test=current_beijing_distribution_test.to(device),
                                beijing_control_test=beijing_control_test.to(device))

            # with torch.no_grad():
            #     samples, samples_person, hgt_data = model.reference_all(
            #         [z_family, z_person], t, y
            #     )

            # Sample images:
            sample_family, sample_person, pred_person = diffusion.p_sample_loop(
                model.reference_all, z_family.shape, [z_family, z_person], clip_denoised=False,
                 model_kwargs=model_kwargs, progress=True, device=device
            )

            samples = sample_family
            samples_person = sample_person

            valid_person = samples[:, 0]
            valid_person = torch.round(valid_person * 0.88397094 + 2.38862088)
            person_mask = torch.zeros(samples.shape[0], 8, device=samples.device).to(torch.bool)
            for i in range(valid_person.shape[0]):


                valid_member = int(valid_person[i].item())
                if int(valid_person[i].item()) < 1:
                    valid_member = 1
                person_mask[i, :valid_member] = True

            person_mask = person_mask.view(-1)
            samples_person_all = torch.zeros(samples_person.shape[0] * 8, 51).to(samples_person.device)
            samples_person = samples_person.view(-1, 51)
            samples_person_all[person_mask] = samples_person[person_mask]
            samples_person = samples_person_all
            samples_person = samples_person.view(-1, 8, 51)

            data = torch.cat((samples, samples_person.view(samples_person.size(0), -1)), dim=1)

            # 编码到潜空间
            with torch.no_grad():
                mu, logvar = model_vae.encode(data)
                z = model_vae.reparameterize(mu, logvar)

            # 计算概率（需要梯度）
            z_grad = z.clone().requires_grad_(True)
            # 计算对数概率
            log_prob = diff_bn.compute_log_prob(z_grad, temperature=0.1)
            log_prob = torch.nan_to_num(log_prob, nan=-100)
            threshold_bn = torch.tensor(-44.0534).to(device)
            judge_bn = log_prob < threshold_bn

            threshold_vae = torch.tensor(2.733688659593493).to(device)
            distance_vae = vectorized_distance_score(z_grad,all_z)
            distance_vae = torch.nan_to_num(distance_vae, nan=100)
            judge_vae = distance_vae > threshold_vae

            final_judge = ~(judge_bn & judge_vae)

            samples = samples[final_judge]
            samples_person = samples_person[final_judge]

            keep_num = samples.shape[0]
            generate_gap = generate_family_num - keep_num
            total_family += generate_gap
            if keep_num == 0:
                continue

            ## 数据复原与保存
            results_family_df = pd.DataFrame(samples.cpu().numpy())
            results_person_df = pd.DataFrame(samples_person.cpu().numpy().reshape(-1, 51))

            real_person_ls = []
            for i in range(len(results_family_df[0])):
                member_num = round(results_family_df.loc[i, 0] * 0.88397094 + 2.38862088)
                real_person_ls.append(results_person_df.iloc[i * 8: i * 8 + member_num, :])

            results_person_df = pd.concat(real_person_ls, ignore_index=True)

            ## 数据存储
            family_ls.append(results_family_df)
            person_ls.append(results_person_df)

            ## 栅格的信息更新
            ## 栅格的男性比例
            male_num = results_person_df.iloc[:, 2].sum()
            current_grid_distribution_test[0] = (current_grid_distribution_test[0] * grid_person_num + male_num) / (
                        grid_person_num + len(results_person_df))

            ## 栅格的年龄
            tau = 0.1  # 越小越接近硬阈值，越大越平滑
            value = results_person_df.iloc[:, 0] * 3.49238421 + 8.40812411
            mask_soft = torch.sigmoid(torch.from_numpy((-(value - 4.5) / tau).values).to(device))
            age_from_0_to_24_num = mask_soft.sum()
            current_grid_distribution_test[1] = (current_grid_distribution_test[
                                                     1] * grid_person_num + age_from_0_to_24_num) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的年龄25-34
            lower = 4.5
            upper = 6.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            age_from_25_to_34_num = mask_soft.sum()
            current_grid_distribution_test[2] = (current_grid_distribution_test[
                                                     2] * grid_person_num + age_from_25_to_34_num) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的年龄35-44
            lower = 6.5
            upper = 8.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            age_from_35_to_44_num = mask_soft.sum()
            current_grid_distribution_test[3] = (current_grid_distribution_test[
                                                     3] * grid_person_num + age_from_35_to_44_num) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的年龄45-54
            lower = 8.5
            upper = 10.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            age_from_45_to_54_num = mask_soft.sum()
            current_grid_distribution_test[4] = (current_grid_distribution_test[
                                                     4] * grid_person_num + age_from_45_to_54_num) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的年龄损失55-64
            lower = 10.5
            upper = 12.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            age_from_55_to_64_num = mask_soft.sum()
            current_grid_distribution_test[5] = (current_grid_distribution_test[
                                                     5] * grid_person_num + age_from_55_to_64_num) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的年龄损失大于65
            lower = 12.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values))
            age_above_65_num = mask_soft.sum()
            current_grid_distribution_test[6] = (current_grid_distribution_test[
                                                     6] * grid_person_num + age_above_65_num) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的教育
            education_value = results_person_df.iloc[:, 21:30].values
            education_below_high_school = education_value[:, :4].sum()
            current_grid_distribution_test[7] = (current_grid_distribution_test[
                                                     7] * grid_person_num + education_below_high_school) / (
                                                            grid_person_num + len(results_person_df))

            education_poly = education_value[:, 4:6].sum()
            current_grid_distribution_test[8] = (current_grid_distribution_test[
                                                     8] * grid_person_num + education_poly) / (
                                                            grid_person_num + len(results_person_df))

            education_above_colledge = education_value[:, 6:].sum()
            current_grid_distribution_test[9] = (current_grid_distribution_test[
                                                     9] * grid_person_num + education_above_colledge) / (
                                                            grid_person_num + len(results_person_df))

            ## 栅格的有车比例损失
            car_number = results_family_df[2] * 0.53264196 + 0.50132666
            lower = 0.5

            mask_soft = torch.sigmoid(torch.from_numpy(((car_number - lower) / tau).values))
            have_car_person = mask_soft * (results_family_df.iloc[:, 0].values * 0.88397094 + 2.38862088)
            have_car_person = have_car_person.sum()
            current_grid_distribution_test[10] = (current_grid_distribution_test[
                                                      10] * grid_person_num + have_car_person) / (
                                                             grid_person_num + len(results_person_df))
            grid_person_num += len(results_person_df)

            # 行政区损失
            ## 家庭规模的损失
            family_size = results_family_df.iloc[:, 0] * 0.88397094 + 2.38862088
            current_district_distribution_test[0] = (current_district_distribution_test[
                                                         0] * district_family_num + family_size.sum()) / (
                                                                district_family_num + len(results_family_df))
            district_family_num += len(results_family_df)

            ## 男性比例损失
            male_num = results_person_df.iloc[:, 2].sum()
            current_district_distribution_test[4] = (current_district_distribution_test[
                                                         4] * district_person_num + male_num) / (
                                                                district_person_num + len(results_person_df))

            ## 行政区的年龄损失

            ## 年龄损失0-14
            tau = 0.1  # 越小越接近硬阈值，越大越平滑
            value = results_person_df.iloc[:, 0] * 3.49238421 + 8.40812411
            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))
            age_from_0_to_14_num = mask_soft.sum()
            current_district_distribution_test[1] = (current_district_distribution_test[
                                                         1] * district_person_num + age_from_0_to_14_num) / (
                                                                district_person_num + len(results_person_df))

            ## 行政区的年龄损失14-64
            lower = 2.5
            upper = 12.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            age_from_15_to_64_num = mask_soft.sum()
            current_district_distribution_test[2] = (current_district_distribution_test[
                                                         2] * district_person_num + age_from_15_to_64_num) / (
                                                                district_person_num + len(results_person_df))

            ## 行政区的年龄损失大于65
            lower = 12.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values))
            age_above_65_num = mask_soft.sum()
            current_district_distribution_test[3] = (current_district_distribution_test[
                                                         3] * district_person_num + age_above_65_num) / (
                                                                district_person_num + len(results_person_df))
            district_person_num += len(results_person_df)

            ## 北京市损失
            ## 家庭规模的损失
            tau = 0.1  # 越小越接近硬阈值，越大越平滑
            value = results_family_df.iloc[:, 0] * 0.88397094 + 2.38862088
            # 家庭规模为1人户
            mask_soft = torch.sigmoid(torch.from_numpy((-(value - 1.5) / tau).values))
            current_beijing_distribution_test[0] = (current_beijing_distribution_test[
                                                        0] * beijing_family_num + mask_soft.sum()) / (
                                                               beijing_family_num + len(results_family_df))

            ## 家庭规模为2人户
            lower = 1.5
            upper = 2.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            familysize2_num = mask_soft.sum()
            current_beijing_distribution_test[1] = (current_beijing_distribution_test[
                                                        1] * beijing_family_num + familysize2_num) / (
                                                               beijing_family_num + len(results_family_df))

            ## 家庭规模为3人户
            lower = 2.5
            upper = 3.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            familysize3_num = mask_soft.sum()
            current_beijing_distribution_test[2] = (current_beijing_distribution_test[
                                                        2] * beijing_family_num + familysize3_num) / (
                                                               beijing_family_num + len(results_family_df))

            ## 家庭规模为4人户
            lower = 3.5
            upper = 4.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                torch.from_numpy(((upper - value) / tau).values))

            familysize4_num = mask_soft.sum()
            current_beijing_distribution_test[3] = (current_beijing_distribution_test[
                                                        3] * beijing_family_num + familysize4_num) / (
                                                               beijing_family_num + len(results_family_df))

            ## 家庭规模为5人户
            lower = 4.5

            mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values))

            familysize5_num = mask_soft.sum()
            current_beijing_distribution_test[4] = (current_beijing_distribution_test[
                                                        4] * beijing_family_num + familysize5_num) / (
                                                               beijing_family_num + len(results_family_df))

            ## 家庭平均工作人口损失
            family_worker = results_family_df.iloc[:, 1] * 0.77530306 + 1.36621842
            current_beijing_distribution_test[5] = (current_beijing_distribution_test[
                                                        5] * beijing_family_num + family_worker.sum()) / (
                                                               beijing_family_num + len(results_family_df))

            ## 北京市年龄相关损失
            beijing_age_thresholds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5,
                                      15.5, 16.5, 20]
            tau = 0.1  # 越小越接近硬阈值，越大越平滑
            value = results_person_df.iloc[:, 0] * 3.49238421 + 8.40812411
            beijing_age_loss = 0
            for index, age in enumerate(beijing_age_thresholds):
                if (index + 1) == len(beijing_age_thresholds):
                    break
                lower = age
                upper = beijing_age_thresholds[index + 1]

                mask_soft = torch.sigmoid(torch.from_numpy(((value - lower) / tau).values)) * torch.sigmoid(
                    torch.from_numpy(((upper - value) / tau).values))

                age_num = mask_soft.sum()
                current_beijing_distribution_test[index + 7] = (current_beijing_distribution_test[
                                                                    index + 7] * beijing_person_num + age_num) / (
                                                                           beijing_person_num + len(results_person_df))

            ## 北京市性别损失
            male_num = results_person_df.iloc[:, 2].sum()
            current_beijing_distribution_test[6] = (current_beijing_distribution_test[
                                                        6] * beijing_person_num + male_num) / (
                                                               beijing_person_num + len(results_person_df))

            ## 北京市教育相关损失
            beijing_education_thresholds = [0, 2, 3, 4, 6, 7, 9]

            tau = 0.1  # 越小越接近硬阈值，越大越平滑
            education_value = results_person_df.iloc[:, 21:30].values
            beijing_education_loss = 0
            for index, education in enumerate(beijing_education_thresholds):
                if (index + 1) == len(beijing_education_thresholds):
                    break
                education_beijing = education_value[:, education:beijing_education_thresholds[index + 1]].sum()

                education_num = education_beijing.sum()
                current_beijing_distribution_test[index + 25] = (current_beijing_distribution_test[
                                                                     index + 25] * beijing_person_num + education_num) / (
                                                                            beijing_person_num + len(results_person_df))

            beijing_person_num += len(results_person_df)
            beijing_family_num += len(results_family_df)

            if total_family == 0:  # 结束条件
                break

        grid_family = pd.concat(family_ls, ignore_index=True)
        grid_person = pd.concat(person_ls, ignore_index=True)

        number = np.round(grid_family[0].values * 0.88397094 + 2.38862088).astype(int).sum()

        grid_family['gird_id'] = grid_control_district.iloc[grid]['id']
        grid_person['gird_id'] = grid_control_district.iloc[grid]['id']

        grid_family['district'] = district
        grid_person['district'] = district

        grid_family.to_csv(f'生成数据_BN/人口栅格_{district}_栅格{grid}_家庭数据.csv', index=False)
        grid_person.to_csv(f'生成数据_BN/人口栅格_{district}_栅格{grid}_个人数据.csv', index=False)


## 未引导的判断
import glob

family_files = glob.glob("生成数据/人口栅格_*家庭*")
person_files = [f.replace("家庭","个人") for f in family_files]

from tqdm import tqdm
family_data = pd.concat([pd.read_csv(f) for f in tqdm(family_files)], ignore_index=True)
person_data = pd.concat([pd.read_csv(f) for f in tqdm(person_files)], ignore_index=True)
# ---- Step 1: 计算每个家庭人数 & 起止 index ----
person_number = np.round(
    family_data["0"].values * 0.88397094 + 2.38862088
).astype(int)

cum_end = np.cumsum(person_number)
cum_start = cum_end - person_number

max_members = 8
num_families = len(person_number)
feature_dim = person_data.shape[1] - 2

# ---- Step 2: 构造一个 (num_families, max_members) 的 index matrix ----
# 对每个家庭制作成员 index：start_i + [0,1,2,...7]
idx = cum_start[:, None] + np.arange(max_members)

# 对越界（成员不足的家庭）的位置设为 -1
mask_valid = (idx < cum_end[:, None])
idx = np.where(mask_valid, idx, -1)

# ---- Step 3: 用高级索引一次性取数据 ----
# person_data 的 shape: (N, feature_dim+2)，我们只取到倒数第2列
person_array = person_data.iloc[:, :-2].values

# pad 一行全 0，放到最后，用于处理 idx = -1 的情况
person_array_padded = np.vstack([person_array, np.zeros((1, feature_dim))])

# 将 -1 的 index 映射到 padded 最后一行
idx_safe = np.where(idx == -1, len(person_array), idx)

# ---- Step 4: 一步取出所有家庭成员数据 ----
total_ls_raw = person_array_padded[idx_safe]

data = np.concatenate([family_data.iloc[:,:-2], total_ls_raw.reshape(num_families, -1)], axis=1)
bn_judge_ls = []
vae_judge_ls = []
distance_vae_ls = []
for i in tqdm(range(int(data.shape[0] / 10000) +1 )):
    batch_data = torch.FloatTensor(data[i*10000 : (i+1)*10000]).to(device)

    # 编码到潜空间
    with torch.no_grad():
        mu, logvar = model_vae.encode(batch_data)
        z = model_vae.reparameterize(mu, logvar)

    # 计算概率（需要梯度）
    z_grad = z.clone().requires_grad_(True)
    # 计算对数概率
    log_prob = diff_bn.compute_log_prob(z_grad, temperature=0.1)
    log_prob = torch.nan_to_num(log_prob, nan=-100)
    threshold_bn = torch.tensor(-44.0534).to(device)
    judge_bn = log_prob < threshold_bn

    threshold_vae = torch.tensor(2.733688659593493).to(device)
    distance_vae = vectorized_distance_score(z_grad,all_z)
    distance_vae = torch.nan_to_num(distance_vae, nan=100)
    judge_vae = distance_vae > threshold_vae
    bn_judge_ls.append(judge_bn.cpu().numpy())
    vae_judge_ls.append(judge_vae.cpu().numpy())
    distance_vae_ls.append(distance_vae.cpu().numpy())
    
bn_judge_total = np.concatenate(bn_judge_ls, axis=0)
vae_judge_total = np.concatenate(vae_judge_ls, axis=0)
distance_vae_total = np.concatenate(distance_vae_ls, axis=0)

del bn_judge_ls
del vae_judge_ls
del distance_vae_ls
## 三种类别的数量
## legal
legal = (~bn_judge_total).sum() / len(bn_judge_total)

print(f'未引导legal:{legal}')
## nonlegal-near_distance
nonlegal1 = (bn_judge_total & (~vae_judge_total)).sum() / len(bn_judge_total)

print(f'未引导nonlegal-near_distance:{nonlegal1}')
## nonlegal-far_distance
nonlegal2 = (bn_judge_total & (vae_judge_total)).sum() / len(bn_judge_total)

print(f'未引导nonlegal-far_distance:{nonlegal2}')

## 加了BN

family_files = glob.glob("生成数据_BN/人口栅格_*家庭*")
person_files = [f.replace("家庭","个人") for f in family_files]

family_data = pd.concat([pd.read_csv(f) for f in tqdm(family_files)], ignore_index=True)
person_data = pd.concat([pd.read_csv(f) for f in tqdm(person_files)], ignore_index=True)

# ---- Step 1: 计算每个家庭人数 & 起止 index ----
person_number = np.round(
    family_data["0"].values * 0.88397094 + 2.38862088
).astype(int)

cum_end = np.cumsum(person_number)
cum_start = cum_end - person_number

max_members = 8
num_families = len(person_number)
feature_dim = person_data.shape[1] - 2

# ---- Step 2: 构造一个 (num_families, max_members) 的 index matrix ----
# 对每个家庭制作成员 index：start_i + [0,1,2,...7]
idx = cum_start[:, None] + np.arange(max_members)

# 对越界（成员不足的家庭）的位置设为 -1
mask_valid = (idx < cum_end[:, None])
idx = np.where(mask_valid, idx, -1)

# ---- Step 3: 用高级索引一次性取数据 ----
# person_data 的 shape: (N, feature_dim+2)，我们只取到倒数第2列
person_array = person_data.iloc[:, :-2].values

# pad 一行全 0，放到最后，用于处理 idx = -1 的情况
person_array_padded = np.vstack([person_array, np.zeros((1, feature_dim))])

# 将 -1 的 index 映射到 padded 最后一行
idx_safe = np.where(idx == -1, len(person_array), idx)

# ---- Step 4: 一步取出所有家庭成员数据 ----
total_ls_raw = person_array_padded[idx_safe]

data = np.concatenate([family_data.iloc[:,:-2], total_ls_raw.reshape(num_families, -1)], axis=1)

bn_judge_ls = []
vae_judge_ls = []
distance_vae_ls = []
for i in tqdm(range(int(data.shape[0] / 10000) +1 )):
    batch_data = torch.FloatTensor(data[i*10000 : (i+1)*10000]).to(device)

    # 编码到潜空间
    with torch.no_grad():
        mu, logvar = model_vae.encode(batch_data)
        z = model_vae.reparameterize(mu, logvar)

    # 计算概率（需要梯度）
    z_grad = z.clone().requires_grad_(True)
    # 计算对数概率
    log_prob = diff_bn.compute_log_prob(z_grad, temperature=0.1)
    log_prob = torch.nan_to_num(log_prob, nan=-100)
    threshold_bn = torch.tensor(-44.0534).to(device)
    judge_bn = log_prob < threshold_bn

    threshold_vae = torch.tensor(2.733688659593493).to(device)
    distance_vae = vectorized_distance_score(z_grad,all_z)
    distance_vae = torch.nan_to_num(distance_vae, nan=100)
    judge_vae = distance_vae > threshold_vae
    bn_judge_ls.append(judge_bn.cpu().numpy())
    vae_judge_ls.append(judge_vae.cpu().numpy())
    distance_vae_ls.append(distance_vae.cpu().numpy())
    
bn_judge_total = np.concatenate(bn_judge_ls, axis=0)
vae_judge_total = np.concatenate(vae_judge_ls, axis=0)
distance_vae_total = np.concatenate(distance_vae_ls, axis=0)

del bn_judge_ls
del vae_judge_ls
del distance_vae_ls
## 三种类别的数量
## legal
legal = (~bn_judge_total).sum() / len(bn_judge_total)

print(f'加了BNlegal:{legal}')
## nonlegal-near_distance
nonlegal1 = (bn_judge_total & (~vae_judge_total)).sum() / len(bn_judge_total)

print(f'加了BNnonlegal-near_distance:{nonlegal1}')
## nonlegal-far_distance
nonlegal2 = (bn_judge_total & (vae_judge_total)).sum() / len(bn_judge_total)

print(f'加了BNnonlegal-far_distance:{nonlegal2}')
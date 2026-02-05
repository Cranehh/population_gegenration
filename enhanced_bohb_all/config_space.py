"""
配置空间定义模块

为PopulationDiT模型定义超参数搜索空间
"""

from typing import Dict, List, Optional, Tuple

try:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
        CategoricalHyperparameter,
        OrdinalHyperparameter,
        Constant
    )
    from ConfigSpace.conditions import (
        InCondition,
        EqualsCondition,
        AndConjunction
    )
    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


def create_population_dit_configspace(
    include_loss_weights: bool = True,
    include_diffusion_params: bool = True,
    simplified: bool = False
) -> 'ConfigurationSpace':
    """
    创建PopulationDiT模型的配置空间

    Args:
        include_loss_weights: 是否包含损失权重参数
        include_diffusion_params: 是否包含扩散参数
        simplified: 是否使用简化的搜索空间

    Returns:
        ConfigSpace配置空间
    """
    if not CONFIGSPACE_AVAILABLE:
        raise ImportError("请安装ConfigSpace: pip install ConfigSpace")

    cs = ConfigurationSpace(seed=42)

    # ==================== 模型架构参数 ====================

    if simplified:
        # 简化版：使用预定义的有效组合
        hidden_dim = CategoricalHyperparameter(
            'hidden_dim',
            choices=[128, 256, 320, 384, 512],
            default_value=320
        )
        num_heads = CategoricalHyperparameter(
            'num_heads',
            choices=[8, 16],
            default_value=16
        )
    else:
        # 完整版：更多选择
        hidden_dim = CategoricalHyperparameter(
            'hidden_dim',
            choices=[128, 192, 256, 320, 384, 448, 512],
            default_value=320
        )
        num_heads = CategoricalHyperparameter(
            'num_heads',
            choices=[8, 16, 32],
            default_value=16
        )

    num_layers = UniformIntegerHyperparameter(
        'num_layers',
        lower=12,
        upper=36,
        default_value=30
    )

    cs.add_hyperparameters([hidden_dim, num_layers, num_heads])

    # ==================== 训练参数 ====================

    lr = UniformFloatHyperparameter(
        'lr',
        lower=1e-5,
        upper=1e-3,
        default_value=1e-4,
        log=True  # 对数尺度
    )

    weight_decay = UniformFloatHyperparameter(
        'weight_decay',
        lower=1e-6,
        upper=1e-2,
        default_value=1e-4,
        log=True
    )

    batch_size = CategoricalHyperparameter(
        'batch_size',
        choices=[128],
        default_value=128
    )

    grad_clip = UniformFloatHyperparameter(
        'grad_clip',
        lower=0.5,
        upper=2.0,
        default_value=1.0
    )

    cs.add_hyperparameters([lr, weight_decay, batch_size, grad_clip])

    # ==================== 扩散参数 ====================

    if include_diffusion_params:
        rho = UniformFloatHyperparameter(
            'rho',
            lower=0.7,
            upper=0.95,
            default_value=0.85
        )

        num_timesteps = CategoricalHyperparameter(
            'num_timesteps',
            choices=[200],
            default_value=200
        )

        cs.add_hyperparameters([rho, num_timesteps])

    # ==================== 损失权重参数 ====================

    if include_loss_weights:
        # 使用分组缩放策略，减少参数数量

        # 家庭级别权重缩放
        family_weight_scale = UniformFloatHyperparameter(
            'family_weight_scale',
            lower=0.5,
            upper=2.0,
            default_value=1.0
        )

        # 人员级别权重缩放
        person_weight_scale = UniformFloatHyperparameter(
            'person_weight_scale',
            lower=0.5,
            upper=3.0,
            default_value=2.0
        )

        # 图结构权重缩放
        graph_weight_scale = UniformFloatHyperparameter(
            'graph_weight_scale',
            lower=0.2,
            upper=1.5,
            default_value=0.5
        )

        # 约束损失权重缩放
        constraint_weight_scale = UniformFloatHyperparameter(
            'constraint_weight_scale',
            lower=0.5,
            upper=2.0,
            default_value=1.0
        )

        cs.add_hyperparameters([
            family_weight_scale,
            person_weight_scale,
            graph_weight_scale,
            constraint_weight_scale
        ])

    return cs


def create_architecture_only_configspace() -> 'ConfigurationSpace':
    """
    创建仅包含架构参数的配置空间

    用于第一阶段的架构搜索
    """
    if not CONFIGSPACE_AVAILABLE:
        raise ImportError("请安装ConfigSpace: pip install ConfigSpace")

    cs = ConfigurationSpace(seed=42)

    hidden_dim = CategoricalHyperparameter(
        'hidden_dim',
        choices=[128, 192, 256, 320, 384],
        default_value=320
    )

    num_layers = UniformIntegerHyperparameter(
        'num_layers',
        lower=12,
        upper=36,
        default_value=30
    )

    num_heads = CategoricalHyperparameter(
        'num_heads',
        choices=[8, 16, 32],
        default_value=16
    )

    cs.add_hyperparameters([hidden_dim, num_layers, num_heads])

    return cs


def create_training_only_configspace() -> 'ConfigurationSpace':
    """
    创建仅包含训练参数的配置空间

    用于第二阶段的训练参数搜索
    """
    if not CONFIGSPACE_AVAILABLE:
        raise ImportError("请安装ConfigSpace: pip install ConfigSpace")

    cs = ConfigurationSpace(seed=42)

    lr = UniformFloatHyperparameter(
        'lr',
        lower=1e-5,
        upper=1e-3,
        default_value=1e-4,
        log=True
    )

    weight_decay = UniformFloatHyperparameter(
        'weight_decay',
        lower=1e-6,
        upper=1e-2,
        default_value=1e-4,
        log=True
    )

    batch_size = CategoricalHyperparameter(
        'batch_size',
        choices=[512, 1024, 2048],
        default_value=1024
    )

    grad_clip = UniformFloatHyperparameter(
        'grad_clip',
        lower=0.5,
        upper=2.0,
        default_value=1.0
    )

    rho = UniformFloatHyperparameter(
        'rho',
        lower=0.7,
        upper=0.95,
        default_value=0.85
    )

    cs.add_hyperparameters([lr, weight_decay, batch_size, grad_clip, rho])

    return cs


def get_default_config() -> Dict:
    """
    获取默认配置

    Returns:
        默认超参数配置字典
    """
    return {
        'hidden_dim': 320,
        'num_layers': 30,
        'num_heads': 16,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 128,
        'grad_clip': 1.0,
        'rho': 0.85,
        'num_timesteps': 200,
        'family_weight_scale': 1.0,
        'person_weight_scale': 2.0,
        'graph_weight_scale': 0.5,
        'constraint_weight_scale': 1.0
    }


def config_to_model_args(config: Dict) -> Dict:
    """
    将配置转换为模型参数

    Args:
        config: 超参数配置

    Returns:
        模型初始化参数
    """
    return {
        'hidden_size': config.get('hidden_dim', 320),
        'depth': config.get('num_layers', 30),
        'num_heads': config.get('num_heads', 16),
        'max_family_size': 8,
        'proj_dim': 24
    }


def config_to_training_args(config: Dict) -> Dict:
    """
    将配置转换为训练参数

    Args:
        config: 超参数配置

    Returns:
        训练参数字典
    """
    return {
        'lr': config.get('lr', 1e-4),
        'weight_decay': config.get('weight_decay', 1e-4),
        'batch_size': config.get('batch_size', 1024),
        'grad_clip': config.get('grad_clip', 1.0)
    }


def config_to_loss_weights(config: Dict) -> Dict:
    """
    将配置转换为损失权重

    Args:
        config: 超参数配置

    Returns:
        损失权重字典
    """
    family_scale = config.get('family_weight_scale', 1.0)
    person_scale = config.get('person_weight_scale', 2.0)
    graph_scale = config.get('graph_weight_scale', 0.5)
    constraint_scale = config.get('constraint_weight_scale', 1.0)

    return {
        'family_continuous': family_scale * 1.0,
        'family_student': family_scale * 1.0,
        'person_age': person_scale * 1.0,
        'person_gender': person_scale * 1.0,
        'person_license': person_scale * 1.0,
        'person_relation': person_scale * 1.0,
        'person_education': person_scale * 1.0,
        'person_occupation': person_scale * 1.0,
        'mask_loss': constraint_scale * 1.0,
        'total_member_loss': constraint_scale * 1.0,
        'unique_loss': constraint_scale * 1.0,
        'graph_adj': graph_scale * 1.0,
        'graph_node': graph_scale * 1.0,
        'graph_edge': graph_scale * 1.0
    }


# ==================== 领域约束 ====================

def hidden_dim_heads_constraint(config: Dict) -> bool:
    """
    约束：hidden_dim必须是num_heads的倍数
    """
    hidden_dim = config.get('hidden_dim', 320)
    num_heads = config.get('num_heads', 16)
    return hidden_dim % num_heads == 0


def depth_lr_constraint(config: Dict) -> bool:
    """
    约束：深网络应使用较小的学习率
    """
    num_layers = config.get('num_layers', 30)
    lr = config.get('lr', 1e-4)

    # 超过24层时，学习率不宜超过5e-4
    if num_layers > 24 and lr > 5e-4:
        return False
    return True


def batch_lr_soft_constraint(config: Dict) -> bool:
    """
    软约束：大batch应配合适当的学习率（线性缩放规则）
    """
    batch_size = config.get('batch_size', 1024)
    lr = config.get('lr', 1e-4)

    # 基准：batch_size=1024, lr=1e-4
    base_batch = 1024
    base_lr = 1e-4
    expected_lr = base_lr * (batch_size / base_batch)

    # 允许3倍范围内的偏差
    if lr < expected_lr * 0.3 or lr > expected_lr * 3:
        return False
    return True


def get_all_constraints() -> List[callable]:
    """
    获取所有领域约束

    Returns:
        约束函数列表
    """
    return [
        hidden_dim_heads_constraint,
        depth_lr_constraint,
        # batch_lr_soft_constraint  # 这是软约束，可选
    ]


def repair_config(config: Dict) -> Dict:
    """
    修复不满足约束的配置

    Args:
        config: 原始配置

    Returns:
        修复后的配置
    """
    config = config.copy()

    # 修复hidden_dim和num_heads的约束
    hidden_dim = config.get('hidden_dim', 320)
    num_heads = config.get('num_heads', 16)

    if hidden_dim % num_heads != 0:
        # 找到最近的有效hidden_dim
        valid_dims = [d for d in [128, 192, 256, 320, 384, 448, 512]
                      if d % num_heads == 0]
        if valid_dims:
            config['hidden_dim'] = min(valid_dims, key=lambda x: abs(x - hidden_dim))

    # 修复深网络学习率
    num_layers = config.get('num_layers', 30)
    lr = config.get('lr', 1e-4)

    if num_layers > 24 and lr > 5e-4:
        config['lr'] = 5e-4

    return config

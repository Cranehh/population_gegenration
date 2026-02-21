#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MI-Guided BOHB 主运行脚本

使用方法:
    # 单GPU运行
    python run_mi_guided_bohb.py --n_brackets 10 --max_budget 50

    # 多GPU并行运行
    python run_mi_guided_bohb.py --parallel --n_gpus 4 --n_brackets 10

核心流程:
    1. Phase 1: 从数据计算互信息 → 构建先验分布
    2. Hyperband: 批量采样 → Successive Halving淘汰
    3. Phase 2: 梯度冲突检测 → 更新采样分布 → 下一轮
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import torch
from mi_guided_bohb.gradient_conflict import (
    GradientConflictInfo
)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mi_guided_bohb import (
    MIGuidedBOHB,
    ParallelMIGuidedBOHB,
    MutualInformationPrior,
    ConflictAwareWorker
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='MI-Guided BOHB超参数优化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument('--data_dir', type=str, default='../数据',
                        help='数据目录路径')
    parser.add_argument('--result_dir', type=str, default='../mi_bohb_results',
                        help='结果保存目录')
    
    # 并行参数
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='启用多GPU并行')
    parser.add_argument('--n_gpus', type=int, default=4,
                        help='GPU数量')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='指定GPU ID，如 "0,1,2,3"')
    
    # Hyperband参数
    parser.add_argument('--n_brackets', type=int, default=10,
                        help='运行的bracket数量')
    parser.add_argument('--min_budget', type=int, default=5,
                        help='最小评估预算(epoch)')
    parser.add_argument('--max_budget', type=int, default=50,
                        help='最大评估预算(epoch)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Successive Halving缩减因子')
    
    # 分布更新参数
    parser.add_argument('--variance_inflation_rate', type=float, default=0.5,
                        help='冲突导致的方差膨胀率')
    parser.add_argument('--mean_shift_rate', type=float, default=0.3,
                        help='向好配置偏移的速率')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='打印详细信息')
    
    return parser.parse_args()


def prepare_data_for_mi(data_dir: str, max_samples: int = 50000):
    """
    准备用于互信息计算的数据（不对个体取平均）
    """
    print("\n准备数据用于互信息计算...")

    from dataset import load_population_data
    dataset = load_population_data(data_dir)

    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    all_rows = []  # 每个有效个体一行

    for idx in indices:
        batch = dataset[idx]
        family = batch['family'].numpy()  # [n_family_vars]
        person = batch['member'].numpy()  # [max_members, n_person_vars]
        mask = np.sum(np.abs(person), axis=-1) > 0  # [max_members]

        # 对每个有效个体，拼接家庭特征和个体特征
        for i in range(len(mask)):
            if mask[i]:
                row = np.concatenate([family, person[i]])
                all_rows.append(row)

    combined_data = np.array(all_rows)

    n_family = batch['family'].numpy().shape[0]
    n_person = batch['member'].numpy().shape[1]

    variable_groups = {
        'family': list(range(n_family)),
        'person': list(range(n_family, n_family + n_person))
    }

    print(f"  数据形状: {combined_data.shape}")
    print(f"  家庭变量: {n_family}, 个人变量: {n_person}")

    return combined_data, variable_groups



def create_evaluate_function(data_dir: str, device: str = None):
    """
    创建评估函数
    
    Returns:
        evaluate_fn: (config, budget) -> (loss, loss_components, conflict_exposure)
    """
    worker = ConflictAwareWorker(
        data_dir=data_dir,
        device=device,
        conflict_detection_freq=20,
        log_every=10
    )
    
    def evaluate_fn(config, budget):
        loss, info = worker.evaluate(config, budget)
        print(info)
        conflict_info = GradientConflictInfo(
            conflict_matrix=None,
            exposure=info.get('conflict_exposure', None),
            magnitude=None,
            imbalance=0
        )

        return (
            loss,
            info.get('loss_components', {}),
            conflict_info
        )
    
    return evaluate_fn


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(args.result_dir, f'run_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    print("=" * 60)
    print("MI-Guided BOHB 超参数优化")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"结果目录: {result_dir}")
    print(f"并行模式: {'是' if args.parallel else '否'}")
    if args.parallel:
        print(f"GPU数量: {args.n_gpus}")
    print(f"Bracket数: {args.n_brackets}")
    print(f"预算范围: [{args.min_budget}, {args.max_budget}]")
    print("=" * 60)
    
    # 保存配置
    config_path = os.path.join(result_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 准备数据
    data, variable_groups = prepare_data_for_mi(args.data_dir)
    
    # 解析GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # 创建优化器
    if args.parallel:
        print("\n使用多GPU并行优化器...")
        optimizer = ParallelMIGuidedBOHB(
            data_dir=args.data_dir,
            n_gpus=args.n_gpus,
            gpu_ids=gpu_ids,
            worker_class=ConflictAwareWorker,
            worker_kwargs={'conflict_detection_freq': 20},
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            eta=args.eta,
            variance_inflation_rate=args.variance_inflation_rate,
            mean_shift_rate=args.mean_shift_rate,
            result_dir=result_dir
        )
        
        # 运行优化
        best_config, best_loss = optimizer.optimize(
            data=data,
            variable_groups=variable_groups,
            n_brackets=args.n_brackets,
            verbose=args.verbose
        )
    else:
        print("\n使用单机优化器...")
        optimizer = MIGuidedBOHB(
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            eta=args.eta,
            variance_inflation_rate=args.variance_inflation_rate,
            mean_shift_rate=args.mean_shift_rate,
            result_dir=result_dir
        )
        
        # 创建评估函数
        evaluate_fn = create_evaluate_function(args.data_dir)
        
        # 运行优化
        best_config, best_loss = optimizer.optimize(
            evaluate_fn=evaluate_fn,
            data=data,
            variable_groups=variable_groups,
            n_brackets=args.n_brackets,
            verbose=args.verbose
        )
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"\n最佳损失: {best_loss:.4f}")
    print(f"\n最佳配置:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\n结果已保存至: {result_dir}")
    print("=" * 60)
    
    return best_config, best_loss


if __name__ == '__main__':
    main()

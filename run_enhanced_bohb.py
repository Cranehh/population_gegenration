#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版BOHB超参数优化主运行脚本

使用方法:
    python run_enhanced_bohb.py --n_iterations 100 --min_budget 10 --max_budget 100

特点:
    1. 参数重要性估计：自动识别关键超参数
    2. 自适应带宽KDE：重要参数精细搜索
    3. 领域约束：自动修复不满足约束的配置
    4. Successive Halving：高效利用计算资源
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_bohb_all import (
    EnhancedBOHBOptimizer,
    SimplifiedBOHB,
    create_population_dit_configspace,
    PopulationDiTWorker,
    get_default_config
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='增强版BOHB超参数优化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基本参数
    parser.add_argument('--data_dir', type=str, default='数据',
                        help='数据目录路径')
    parser.add_argument('--result_dir', type=str, default='bohb_results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (cuda/cpu)')

    # BOHB参数
    parser.add_argument('--n_iterations', type=int, default=100,
                        help='总迭代次数')
    parser.add_argument('--min_budget', type=int, default=10,
                        help='最小评估预算(epoch)')
    parser.add_argument('--max_budget', type=int, default=100,
                        help='最大评估预算(epoch)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Successive Halving缩减因子')

    # TPE参数
    parser.add_argument('--min_points', type=int, default=20,
                        help='TPE模型最小样本数')
    parser.add_argument('--gamma', type=float, default=0.15,
                        help='TPE好配置比例')
    parser.add_argument('--n_candidates', type=int, default=64,
                        help='候选采样数')
    parser.add_argument('--random_fraction', type=float, default=0.1,
                        help='随机采样比例')

    # 重要性估计参数
    parser.add_argument('--importance_update_freq', type=int, default=10,
                        help='参数重要性更新频率')

    # 其他
    parser.add_argument('--simplified', action='store_true',
                        help='使用简化版BOHB（不使用Successive Halving）')
    parser.add_argument('--include_loss_weights', type=bool, default=True,
                        help='包含损失权重参数')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(args.result_dir, f'run_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("增强版BOHB超参数优化")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"结果目录: {result_dir}")
    print(f"设备: {args.device or 'auto'}")
    print(f"迭代次数: {args.n_iterations}")
    print(f"预算范围: [{args.min_budget}, {args.max_budget}]")
    print("=" * 60)

    # 保存配置
    config_path = os.path.join(result_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 创建配置空间
    print("\n创建配置空间...")
    configspace = create_population_dit_configspace(
        include_loss_weights=args.include_loss_weights,
        include_diffusion_params=True,
        simplified=False
    )

    print(f"配置空间包含 {len(configspace.get_hyperparameters())} 个超参数:")
    for hp in configspace.get_hyperparameters():
        print(f"  - {hp.name}: {type(hp).__name__}")

    # 创建Worker
    print("\n创建训练Worker...")
    worker = PopulationDiTWorker(
        data_dir=args.data_dir,
        device=args.device,
        num_workers=4,
        validation_split=0.1,
        early_stopping_patience=10
    )

    # 创建评估函数
    def evaluate_fn(config, budget):
        return worker.evaluate(config, budget)

    # 创建优化器
    if args.simplified:
        print("\n使用简化版BOHB...")
        optimizer = SimplifiedBOHB(
            configspace=configspace,
            budget=args.max_budget,
            min_points_in_model=args.min_points,
            importance_update_frequency=args.importance_update_freq
        )

        # 简化版优化循环
        print(f"\n开始优化 ({args.n_iterations} 次迭代)...")
        for i in range(args.n_iterations):
            config = optimizer.suggest()
            loss, info = evaluate_fn(config, args.max_budget)
            optimizer.observe(config, loss)

            print(f"\n[{i+1}/{args.n_iterations}] Loss: {loss:.4f}")

            # 定期打印最佳结果
            if (i + 1) % 10 == 0:
                best_config, best_loss = optimizer.get_best()
                print(f"  当前最佳损失: {best_loss:.4f}")
                print(f"  参数重要性: {optimizer.get_importance()}")

        best_config, best_loss = optimizer.get_best()

    else:
        print("\n使用完整版EnhancedBOHB...")
        optimizer = EnhancedBOHBOptimizer(
            configspace=configspace,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            eta=args.eta,
            min_points_in_model=args.min_points,
            importance_update_frequency=args.importance_update_freq,
            gamma=args.gamma,
            n_candidates=args.n_candidates,
            random_fraction=args.random_fraction,
            result_dir=result_dir
        )

        # 从检查点恢复
        if args.resume:
            print(f"\n从检查点恢复: {args.resume}")
            optimizer.load_checkpoint(args.resume)

        # 运行优化
        best_config, best_loss = optimizer.optimize(
            evaluate_fn=evaluate_fn,
            n_iterations=args.n_iterations,
            verbose=True
        )

    # 保存最终结果
    final_results = {
        'best_config': best_config,
        'best_loss': best_loss,
        'importance_ranking': optimizer.get_importance_ranking() if hasattr(optimizer, 'get_importance_ranking') else None,
        'args': vars(args)
    }

    results_path = os.path.join(result_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

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

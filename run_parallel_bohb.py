#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多GPU并行BOHB超参数优化

使用方法:
    python run_parallel_bohb.py --n_gpus 4 --n_iterations 100
"""

import argparse
import os
import sys
import torch.multiprocessing as mp

# 设置多进程启动方式（必须在main之前）
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

from enhanced_bohb_all import (
    ParallelBOHBOptimizer,
    create_population_dit_configspace,
)


def parse_args():
    parser = argparse.ArgumentParser(description='多GPU并行BOHB优化')

    parser.add_argument('--data_dir', type=str, default='数据')
    parser.add_argument('--result_dir', type=str, default='bohb_results_parallel')

    # GPU设置
    parser.add_argument('--n_gpus', type=int, default=4, help='使用的GPU数量')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='指定GPU ID，如 "0,1,2,3"')

    # BOHB参数
    parser.add_argument('--n_iterations', type=int, default=100)
    parser.add_argument('--min_budget', type=int, default=5)
    parser.add_argument('--max_budget', type=int, default=50)
    parser.add_argument('--eta', type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.n_gpus))

    print(f"使用GPU: {gpu_ids}")

    # 创建配置空间
    configspace = create_population_dit_configspace(
        include_loss_weights=True,
        include_diffusion_params=True,
        simplified=False
    )

    # 创建并行优化器
    optimizer = ParallelBOHBOptimizer(
        configspace=configspace,
        data_dir=args.data_dir,
        n_gpus=len(gpu_ids),
        gpu_ids=gpu_ids,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta,
        result_dir=args.result_dir,
        worker_kwargs={
            'num_workers': 2,  # 每个GPU的dataloader workers
            'validation_split': 0.1,
            'early_stopping_patience': 10
        }
    )

    # 运行优化
    best_config, best_loss = optimizer.optimize(
        n_iterations=args.n_iterations,
        verbose=True
    )

    print(f"\n最佳配置: {best_config}")
    print(f"最佳损失: {best_loss:.4f}")


if __name__ == '__main__':
    main()
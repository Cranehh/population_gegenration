"""
多GPU并行BOHB优化器
"""

import os
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
import time
import numpy as np

from .enhanced_bohb import EnhancedBOHBOptimizer, Observation
from .worker import PopulationDiTWorker
from .config_space import repair_config


def _worker_evaluate(args):
    """
    单个Worker的评估函数（在子进程中运行）

    Args:
        args: (config, budget, gpu_id, data_dir, worker_kwargs)

    Returns:
        (config, loss, info, gpu_id)
    """
    config, budget, gpu_id, data_dir, worker_kwargs = args

    # 【关键修复】确保所有类型都是Python原生类型
    budget = int(budget)
    gpu_id = int(gpu_id)
    
    # 转换config中的所有numpy类型为Python原生类型
    config = _convert_config_types(config)
    
    # 设置当前进程只能看到指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 创建Worker
    worker = PopulationDiTWorker(
        data_dir=data_dir,
        device='cuda:0',
        **worker_kwargs
    )

    
    loss, info = worker.evaluate(config, budget)
    return (config, loss, info, gpu_id)
    # try:
    #     loss, info = worker.evaluate(config, budget)
    #     return (config, loss, info, gpu_id)
    # except Exception as e:
    #     print(f"[GPU {gpu_id}] 评估失败: {e}")
    #     return (config, float('inf'), {'error': str(e)}, gpu_id)


def _convert_config_types(config: Dict) -> Dict:
    """
    将config中的numpy类型转换为Python原生类型
    
    这对于跨进程传递参数是必要的
    """
    import numpy as np
    
    converted = {}
    for key, value in config.items():
        if isinstance(value, (np.integer,)):
            converted[key] = int(value)
        elif isinstance(value, (np.floating,)):
            converted[key] = float(value)
        elif isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, np.bool_):
            converted[key] = bool(value)
        else:
            converted[key] = value
    return converted

class ParallelBOHBOptimizer:
    """
    多GPU并行BOHB优化器

    使用多个GPU同时评估不同的超参数配置
    """

    def __init__(
            self,
            configspace,
            data_dir: str = '数据',
            n_gpus: int = 4,
            gpu_ids: Optional[List[int]] = None,
            min_budget: int = 10,
            max_budget: int = 200,
            eta: int = 3,
            min_points_in_model: int = 10,
            importance_update_frequency: int = 5,
            gamma: float = 0.15,
            n_candidates: int = 64,
            random_fraction: float = 0.1,
            result_dir: Optional[str] = None,
            worker_kwargs: Optional[Dict] = None
    ):
        """
        初始化并行BOHB优化器

        Args:
            configspace: 配置空间
            data_dir: 数据目录
            n_gpus: 使用的GPU数量
            gpu_ids: 具体使用的GPU ID列表，如 [0,1,2,3]
            其余参数与EnhancedBOHBOptimizer相同
        """
        self.data_dir = data_dir
        self.n_gpus = n_gpus
        self.gpu_ids = gpu_ids if gpu_ids else list(range(n_gpus))
        self.worker_kwargs = worker_kwargs or {}

        # 创建基础BOHB优化器（用于采样和记录）
        self.bohb = EnhancedBOHBOptimizer(
            configspace=configspace,
            min_budget=min_budget,
            max_budget=max_budget,
            eta=eta,
            n_gpus=n_gpus,  # 【新增】传递GPU数量
            min_points_in_model=min_points_in_model,
            importance_update_frequency=importance_update_frequency,
            gamma=gamma,
            n_candidates=n_candidates,
            random_fraction=random_fraction,
            result_dir=result_dir
        )

        print(f"[ParallelBOHB] 初始化完成")
        print(f"  使用GPU: {self.gpu_ids}")
        print(f"  并行度: {len(self.gpu_ids)}")

    def _sample_configs(self, n: int, budget: int) -> List[Tuple[Dict, int]]:
        """采样n个配置"""
        configs = []
        for _ in range(n):
            config, _ = self.bohb.get_next_config(budget)
            # 【修复】转换类型
            clean_config = _convert_config_types(config)
            configs.append((clean_config, int(budget)))
        return configs

    def _parallel_evaluate(
            self,
            configs_with_budget: List[Tuple[Dict, int]]
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        并行评估多个配置

        Args:
            configs_with_budget: [(config, budget), ...]

        Returns:
            [(config, loss, info), ...]
        """
        results = []

        tasks = []
        for i, (config, budget) in enumerate(configs_with_budget):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            
            # 转换所有numpy类型为Python原生类型
            clean_config = _convert_config_types(config)
            clean_budget = int(budget)
            clean_gpu_id = int(gpu_id)
            
            tasks.append((clean_config, clean_budget, clean_gpu_id, self.data_dir, self.worker_kwargs))

        # 使用进程池并行执行
        # 注意：需要使用'spawn'方式启动进程以避免CUDA问题
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(self.gpu_ids), mp_context=ctx) as executor:
            futures = [executor.submit(_worker_evaluate, task) for task in tasks]

            for future in as_completed(futures):
                # try:
                config, loss, info, gpu_id = future.result()
                results.append((config, loss, info))
                print(f"  [GPU {gpu_id}] 完成: loss={loss:.4f}")
                # except Exception as e:
                #     print(f"  评估异常: {e}")

        return results

    def run_parallel_successive_halving(
            self,
            bracket: Optional[int] = None,
            verbose: bool = True
    ) -> List[Observation]:
        """
        并行运行Successive Halving

        Args:
            bracket: bracket索引
            verbose: 是否打印详细信息

        Returns:
            该bracket的所有观测
        """
        if bracket is None:
            bracket = self.bohb.current_bracket

        bracket_info = self.bohb.bracket_budgets[bracket]
        n_configs = bracket_info['n_configs']
        budgets = bracket_info['budgets']

        if verbose:
            print(f"\n[ParallelBOHB] 开始Bracket {bracket}")
            print(f"  初始配置数: {n_configs}")
            print(f"  预算序列: {budgets}")
            print(f"  并行GPU数: {len(self.gpu_ids)}")

        # 【修复】初始配置采样，确保类型正确
        configs = []
        for _ in range(n_configs):
            config, _ = self.bohb.get_next_config(int(budgets[0]))
            configs.append(_convert_config_types(config))

        observations = []

        # Successive Halving循环
        for round_idx, budget in enumerate(budgets):
            if verbose:
                print(f"\n  Round {round_idx + 1}/{len(budgets)}: "
                      f"{len(configs)} configs, budget={budget}")

            # 并行评估当前轮的所有配置
            configs_with_budget = [(c, budget) for c in configs]

            # 分批并行评估
            all_results = []
            batch_size = len(self.gpu_ids)

            for batch_start in range(0, len(configs_with_budget), batch_size):
                batch = configs_with_budget[batch_start:batch_start + batch_size]
                if verbose:
                    print(f"    批次 {batch_start // batch_size + 1}: 评估 {len(batch)} 个配置...")

                batch_results = self._parallel_evaluate(batch)
                all_results.extend(batch_results)

            # 记录观测结果
            losses = []
            for config, loss, info in all_results:
                self.bohb.observe(config, loss, budget, info)
                losses.append(loss)
                observations.append(Observation(config=config, loss=loss, budget=budget, info=info))

            # 选择top 1/eta进入下一轮
            if round_idx < len(budgets) - 1:
                n_keep = max(1, int(len(configs) / self.bohb.eta))
                indices = np.argsort(losses)[:n_keep]
                configs = [configs[i] for i in indices]

                if verbose:
                    print(f"  保留 {len(configs)} 个配置进入下一轮")

        # 更新bracket
        self.bohb.current_bracket = (self.bohb.current_bracket + 1) % (self.bohb.s_max + 1)

        return observations

    def optimize(
            self,
            n_iterations: int = 100,
            verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        运行并行BOHB优化

        Args:
            n_iterations: 总迭代次数
            verbose: 是否打印详细信息

        Returns:
            (best_config, best_loss)
        """
        print(f"\n{'=' * 60}")
        print(f"开始并行BOHB优化")
        print(f"  GPU数量: {len(self.gpu_ids)}")
        print(f"  总迭代数: {n_iterations}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        while self.bohb.iteration < n_iterations:
            self.run_parallel_successive_halving(verbose=verbose)

            best_config, best_loss = self.bohb.get_best_config()
            elapsed = time.time() - start_time

            if verbose:
                print(f"\n[进度] Iteration {self.bohb.iteration}/{n_iterations}, "
                      f"Best loss: {best_loss:.4f}, "
                      f"Time: {elapsed:.1f}s")

        best_config, best_loss = self.bohb.get_best_config()

        print(f"\n{'=' * 60}")
        print(f"优化完成!")
        print(f"  总耗时: {time.time() - start_time:.1f}s")
        print(f"  最佳损失: {best_loss:.4f}")
        print(f"{'=' * 60}")

        return best_config, best_loss

    def get_best_config(self) -> Tuple[Dict, float]:
        """获取最佳配置"""
        return self.bohb.get_best_config()

    def get_importance_ranking(self):
        """获取参数重要性排名"""
        return self.bohb.get_importance_ranking()
"""
多GPU并行BOHB执行模块

支持在多个GPU上并行评估超参数配置：
- 每个GPU运行独立的Worker
- 使用ProcessPoolExecutor管理并行
- 收集梯度冲突信息用于分布更新
"""

import os
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from .mi_guided_optimizer import MIGuidedBOHB, ConfigEvaluation, BracketResult
from .gradient_conflict import GradientConflictDetector, GradientConflictInfo


def _convert_config_types(config: Dict) -> Dict:
    """将numpy类型转换为Python原生类型"""
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


def _worker_evaluate(args: Tuple) -> Tuple[Dict, float, Dict, Optional[Dict], int]:
    """
    单GPU Worker的评估函数
    
    Args:
        args: (config, budget, gpu_id, data_dir, worker_class, worker_kwargs, loss_names)
    
    Returns:
        (config, loss, loss_components, conflict_exposure, gpu_id)
    """
    config, budget, gpu_id, data_dir, worker_class, worker_kwargs, loss_names = args
    
    # 确保类型正确
    budget = int(budget)
    gpu_id = int(gpu_id)
    config = _convert_config_types(config)
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # 创建Worker
        worker = worker_class(
            data_dir=data_dir,
            device='cuda:0',
            loss_names=loss_names,
            **worker_kwargs
        )
        
        # 评估（返回loss和各分项）
        loss, info = worker.evaluate(config, budget)
        
        # 提取loss分项和冲突信息
        loss_components = info.get('loss_components', {})
        conflict_exposure = info.get('conflict_exposure', None)
        
        return (config, loss, loss_components, conflict_exposure, gpu_id)
    
    except Exception as e:
        print(f"[GPU {gpu_id}] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return (config, float('inf'), {}, None, gpu_id)


class ParallelMIGuidedBOHB(MIGuidedBOHB):
    """
    多GPU并行的MI-Guided BOHB优化器
    
    在多个GPU上同时评估不同的超参数配置
    """
    
    def __init__(
        self,
        data_dir: str = '数据',
        n_gpus: int = 4,
        gpu_ids: Optional[List[int]] = None,
        worker_class: Optional[Any] = None,
        worker_kwargs: Optional[Dict] = None,
        min_budget: int = 10,
        max_budget: int = 100,
        eta: int = 3,
        loss_names: Optional[List[str]] = None,
        param_names: Optional[List[str]] = None,
        loss_to_param_map: Optional[Dict[str, str]] = None,
        variance_inflation_rate: float = 0.5,
        mean_shift_rate: float = 0.3,
        result_dir: Optional[str] = None
    ):
        """
        初始化并行优化器
        
        Args:
            data_dir: 数据目录
            n_gpus: GPU数量
            gpu_ids: 具体GPU ID列表
            worker_class: Worker类
            worker_kwargs: Worker初始化参数
            其余参数与MIGuidedBOHB相同
        """
        super().__init__(
            min_budget=min_budget,
            max_budget=max_budget,
            eta=eta,
            loss_names=loss_names,
            param_names=param_names,
            loss_to_param_map=loss_to_param_map,
            variance_inflation_rate=variance_inflation_rate,
            mean_shift_rate=mean_shift_rate,
            result_dir=result_dir
        )
        
        self.data_dir = data_dir
        self.n_gpus = n_gpus
        self.gpu_ids = gpu_ids if gpu_ids else list(range(n_gpus))
        self.worker_class = worker_class
        self.worker_kwargs = worker_kwargs or {}
        
        # 调整配置数量为GPU数量的倍数
        self._adjust_bracket_configs()
        
        print(f"[ParallelMIGuidedBOHB] 初始化完成")
        print(f"  使用GPU: {self.gpu_ids}")
        print(f"  并行度: {len(self.gpu_ids)}")
    
    def _adjust_bracket_configs(self):
        """调整配置数量为GPU数量的倍数"""
        for s in self.bracket_configs:
            n_configs = self.bracket_configs[s]['n_configs']
            adjusted = int(np.ceil(n_configs / len(self.gpu_ids)) * len(self.gpu_ids))
            self.bracket_configs[s]['n_configs'] = adjusted
    
    def _parallel_evaluate(
        self,
        configs: List[Dict],
        budget: int
    ) -> List[Tuple[Dict, float, Dict, Optional[Dict]]]:
        """
        并行评估多个配置
        
        Args:
            configs: 配置列表
            budget: 评估预算
        
        Returns:
            [(config, loss, loss_components, conflict_exposure), ...]
        """
        results = []
        
        # 准备任务
        tasks = []
        for i, config in enumerate(configs):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            clean_config = _convert_config_types(config)
            
            tasks.append((
                clean_config,
                int(budget),
                int(gpu_id),
                self.data_dir,
                self.worker_class,
                self.worker_kwargs,
                self.loss_names
            ))
        
        # 使用spawn方式启动进程（避免CUDA问题）
        ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(
            max_workers=len(self.gpu_ids),
            mp_context=ctx
        ) as executor:
            futures = [executor.submit(_worker_evaluate, task) for task in tasks]
            
            for future in as_completed(futures):
                try:
                    config, loss, loss_components, conflict_exposure, gpu_id = future.result()
                    results.append((config, loss, loss_components, conflict_exposure))
                    print(f"  [GPU {gpu_id}] 完成: loss={loss:.4f}")
                except Exception as e:
                    print(f"  评估异常: {e}")
        
        return results
    
    def run_successive_halving(
        self,
        evaluate_fn: Optional[Callable] = None,
        bracket: int = 0,
        conflict_callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> BracketResult:
        """
        并行运行Successive Halving
        
        Args:
            evaluate_fn: 忽略（使用并行评估）
            bracket: bracket索引
            conflict_callback: 冲突信息回调
            verbose: 是否打印详细信息
        
        Returns:
            BracketResult
        """
        bracket_info = self.bracket_configs[bracket]
        n_configs = bracket_info['n_configs']
        budgets = bracket_info['budgets']
        
        if verbose:
            print(f"\n[Bracket {bracket}] 采样 {n_configs} 个配置")
            print(f"  预算序列: {budgets}")
            print(f"  并行GPU数: {len(self.gpu_ids)}")
        
        # 采样初始配置
        configs = self.sample_configs(n_configs)
        evaluations = []
        all_exposures = []
        
        # Successive Halving循环
        for round_idx, budget in enumerate(budgets):
            if verbose:
                print(f"\n  Round {round_idx + 1}/{len(budgets)}: "
                      f"{len(configs)} configs, budget={budget}")
            
            # 分批并行评估
            batch_size = len(self.gpu_ids)
            round_results = []
            
            for batch_start in range(0, len(configs), batch_size):
                batch_configs = configs[batch_start:batch_start + batch_size]
                
                if verbose:
                    print(f"    批次 {batch_start // batch_size + 1}: "
                          f"评估 {len(batch_configs)} 个配置...")
                
                batch_results = self._parallel_evaluate(batch_configs, budget)
                round_results.extend(batch_results)
            
            # 记录评估结果
            round_evals = []
            for config, loss, loss_components, conflict_exposure in round_results:
                eval_result = ConfigEvaluation(
                    config=config,
                    loss=loss,
                    budget=budget,
                    loss_components=loss_components
                )
                round_evals.append(eval_result)
                evaluations.append(eval_result)
                
                if conflict_exposure is not None:
                    all_exposures.append(conflict_exposure)
            
            # 淘汰：保留top 1/eta
            if round_idx < len(budgets) - 1:
                n_keep = max(1, int(len(configs) / self.eta))
                # 对齐到GPU数量
                n_keep = max(len(self.gpu_ids), 
                            int(np.ceil(n_keep / len(self.gpu_ids)) * len(self.gpu_ids)))
                n_keep = min(n_keep, len(round_evals))
                
                sorted_evals = sorted(round_evals, key=lambda x: x.loss)
                configs = [e.config for e in sorted_evals[:n_keep]]
                
                if verbose:
                    print(f"  保留 {len(configs)} 个配置进入下一轮")
        
        # 聚合冲突暴露度
        if all_exposures:
            aggregated_exposure = {}
            for name in self.loss_names:
                values = [exp.get(name, 0.0) for exp in all_exposures if isinstance(exp, dict)]
                aggregated_exposure[name] = np.mean(values) if values else 0.0
        else:
            aggregated_exposure = {name: 0.0 for name in self.loss_names}
        
        # 找最佳配置
        best_eval = min(evaluations, key=lambda x: x.loss)
        
        result = BracketResult(
            bracket_id=bracket,
            evaluations=evaluations,
            best_config=best_eval.config,
            best_loss=best_eval.loss,
            aggregated_exposure=aggregated_exposure
        )
        
        self.bracket_results.append(result)
        self.all_evaluations.extend(evaluations)
        
        return result
    
    def optimize(
        self,
        data: np.ndarray,
        variable_groups: Dict[str, List[int]],
        n_brackets: int = 5,
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        运行并行MI-Guided BOHB优化
        
        Args:
            data: 用于构建先验的数据
            variable_groups: 变量分组
            n_brackets: 运行的bracket数量
            verbose: 是否打印详细信息
        
        Returns:
            (best_config, best_loss)
        """
        start_time = time.time()
        
        print(f"\n{'=' * 60}")
        print(f"开始并行MI-Guided BOHB优化")
        print(f"  GPU数量: {len(self.gpu_ids)}")
        print(f"  总bracket数: {n_brackets}")
        print(f"{'=' * 60}")
        
        # Phase 1: 构建先验
        self.build_prior_from_data(data, variable_groups)
        
        # Hyperband循环
        for bracket_idx in range(n_brackets):
            bracket = bracket_idx % (self.s_max + 1)
            
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Iteration {bracket_idx + 1}/{n_brackets}")
                print(f"{'=' * 60}")
            
            # 运行并行Successive Halving
            result = self.run_successive_halving(
                bracket=bracket,
                verbose=verbose
            )
            
            # Phase 2: 更新分布
            self.update_distribution(result, verbose=verbose)
            
            self.iteration += 1
            
            # 打印当前最佳
            best_config, best_loss = self.get_best_config()
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"\n[进度] Best loss: {best_loss:.4f}, Time: {elapsed:.1f}s")
            
            # 保存检查点
            if self.result_dir:
                self._save_checkpoint()
        
        # 获取最佳结果
        best_config, best_loss = self.get_best_config()
        
        elapsed = time.time() - start_time
        
        print(f"\n{'=' * 60}")
        print(f"优化完成!")
        print(f"  总耗时: {elapsed:.1f}s")
        print(f"  总评估数: {len(self.all_evaluations)}")
        print(f"  最佳损失: {best_loss:.4f}")
        print(f"{'=' * 60}")
        
        # 保存最终结果
        if self.result_dir:
            self._save_final_results(best_config, best_loss)
        
        return best_config, best_loss

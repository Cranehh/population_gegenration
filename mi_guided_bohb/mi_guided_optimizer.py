"""
MI-Guided BOHB优化器

整合互信息先验和梯度冲突驱动的Hyperband优化：
- Phase 1: 从数据互信息构建采样分布
- Hyperband: 批量采样 + Successive Halving
- Phase 2: 基于梯度冲突更新采样分布
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .mutual_information_prior import MutualInformationPrior
from .gradient_conflict import (
    GradientConflictDetector,
    ConflictAwareDistributionUpdater,
    GradientConflictInfo
)


@dataclass
class ConfigEvaluation:
    """配置评估结果"""
    config: Dict
    loss: float
    budget: int
    loss_components: Dict[str, float] = field(default_factory=dict)
    conflict_info: Optional[GradientConflictInfo] = None
    training_time: float = 0.0


@dataclass 
class BracketResult:
    """Bracket执行结果"""
    bracket_id: int
    evaluations: List[ConfigEvaluation]
    best_config: Dict
    best_loss: float
    aggregated_exposure: Dict[str, float]


class MIGuidedBOHB:
    """
    互信息引导的BOHB优化器
    
    核心流程：
    1. Phase 1: 互信息 → 先验分布
    2. Hyperband循环: 采样 → 训练 → 淘汰
    3. Phase 2: 梯度冲突 → 更新分布
    """
    
    def __init__(
        self,
        configspace: Optional[Any] = None,
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
        初始化优化器
        
        Args:
            configspace: ConfigSpace配置空间（可选）
            min_budget: 最小预算（epoch）
            max_budget: 最大预算（epoch）
            eta: Successive Halving缩减因子
            loss_names: Loss分项名称
            param_names: 超参数名称
            loss_to_param_map: Loss到参数的映射
            variance_inflation_rate: 冲突导致的方差膨胀率
            mean_shift_rate: 向好配置偏移的速率
            result_dir: 结果保存目录
        """
        self.configspace = configspace
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.result_dir = result_dir
        
        # 默认Loss和参数配置
        self.loss_names = loss_names or [
            'family_loss', 'person_loss', 'graph_loss', 'constraint_loss'
        ]
        
        self.param_names = param_names or [
            'family_weight_scale', 'person_weight_scale', 
            'graph_weight_scale', 'constraint_weight_scale',
            'lr', 'rho'
        ]
        
        self.loss_to_param_map = loss_to_param_map or {
            'family_loss': 'family_weight_scale',
            'person_loss': 'person_weight_scale',
            'graph_loss': 'graph_weight_scale',
            'constraint_loss': 'constraint_weight_scale'
        }
        
        # 初始化组件
        self.mi_prior = MutualInformationPrior()
        self.conflict_detector = GradientConflictDetector(self.loss_names)
        self.distribution_updater = ConflictAwareDistributionUpdater(
            param_names=self.param_names,
            loss_to_param_map=self.loss_to_param_map,
            variance_inflation_rate=variance_inflation_rate,
            mean_shift_rate=mean_shift_rate
        )
        
        # Hyperband参数
        self._init_hyperband_params()
        
        # 状态追踪
        self.all_evaluations: List[ConfigEvaluation] = []
        self.bracket_results: List[BracketResult] = []
        self.current_distribution: Dict[str, Dict[str, float]] = {}
        self.iteration = 0
        
        # 创建结果目录
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
    
    def _init_hyperband_params(self):
        """初始化Hyperband参数"""
        self.s_max = int(np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ))
        
        self.bracket_configs = {}
        for s in range(self.s_max + 1):
            n_configs = int(np.ceil(
                (self.s_max + 1) / (s + 1) * (self.eta ** s)
            ))
            min_budget_s = self.max_budget * (self.eta ** (-s))
            budgets = [int(min_budget_s * (self.eta ** i)) for i in range(s + 1)]
            
            self.bracket_configs[s] = {
                'n_configs': n_configs,
                'budgets': budgets
            }
    
    def build_prior_from_data(
        self,
        data: np.ndarray,
        variable_groups: Dict[str, List[int]]
    ):
        """
        Phase 1: 从数据构建互信息先验
        
        Args:
            data: 数据矩阵 [n_samples, n_variables]
            variable_groups: 变量分组
        """
        print("\n" + "=" * 60)
        print("Phase 1: 构建互信息先验")
        print("=" * 60)
        
        # 计算互信息矩阵
        self.mi_prior.compute_mi_matrix(data)
        self.mi_prior.set_variable_groups(variable_groups)
        
        # 构建先验分布
        prior = self.mi_prior.build_prior_distribution()
        
        # 补充其他参数的默认先验
        if 'lr' not in prior:
            prior['lr'] = {'mean': 1e-4, 'std': 5e-5}
        if 'rho' not in prior:
            prior['rho'] = {'mean': 0.85, 'std': 0.05}
        
        # 初始化分布更新器
        self.distribution_updater.initialize(prior)
        self.current_distribution = prior
        
        print(self.mi_prior.get_prior_summary())
    
    def sample_configs(self, n: int) -> List[Dict]:
        """从当前分布采样配置"""
        return self.distribution_updater.sample(n)
    
    def run_successive_halving(
        self,
        evaluate_fn: Callable,
        bracket: int,
        conflict_callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> BracketResult:
        """
        运行一次Successive Halving
        
        Args:
            evaluate_fn: 评估函数 (config, budget) -> (loss, loss_components, conflict_info)
            bracket: bracket索引
            conflict_callback: 冲突信息回调函数
            verbose: 是否打印详细信息
        
        Returns:
            BracketResult
        """
        bracket_info = self.bracket_configs[bracket]
        n_configs = bracket_info['n_configs']
        budgets = bracket_info['budgets']
        
        if verbose:
            print(f"\n[Bracket {bracket}] 采样 {n_configs} 个配置, 预算序列: {budgets}")
        
        # 采样初始配置
        configs = self.sample_configs(n_configs)
        evaluations = []
        all_conflict_infos = []
        
        # Successive Halving循环
        for round_idx, budget in enumerate(budgets):
            if verbose:
                print(f"\n  Round {round_idx + 1}/{len(budgets)}: "
                      f"{len(configs)} configs, budget={budget}")
            
            # 评估所有配置
            round_evals = []
            for i, config in enumerate(configs):
                loss, loss_components, conflict_info = evaluate_fn(config, budget)
                print(conflict_info)
                eval_result = ConfigEvaluation(
                    config=config,
                    loss=loss,
                    budget=budget,
                    loss_components=loss_components,
                    conflict_info=conflict_info,
                    training_time=loss_components.get('training_time', 0.0)
                )
                round_evals.append(eval_result)
                evaluations.append(eval_result)
                
                if conflict_info is not None:
                    all_conflict_infos.append(conflict_info)
                
                if verbose:
                    print(f"    Config {i+1}: loss={loss:.4f}")
            
            # 淘汰：保留top 1/eta
            if round_idx < len(budgets) - 1:
                n_keep = max(1, int(len(configs) / self.eta))
                sorted_evals = sorted(round_evals, key=lambda x: x.loss)
                configs = [e.config for e in sorted_evals[:n_keep]]
                
                if verbose:
                    print(f"  保留 {len(configs)} 个配置")
        
        # 聚合冲突暴露度
        if all_conflict_infos:
            aggregated_exposure = defaultdict(float)
            for info in all_conflict_infos:
                for name, exp in info.exposure.items():
                    aggregated_exposure[name] += exp
            aggregated_exposure = {
                k: v / len(all_conflict_infos) 
                for k, v in aggregated_exposure.items()
            }
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
    
    def update_distribution(
        self,
        bracket_result: BracketResult,
        verbose: bool = True
    ):
        """
        Phase 2: 基于梯度冲突更新采样分布
        
        Args:
            bracket_result: Bracket执行结果
            verbose: 是否打印详细信息
        """
        if verbose:
            print(f"\n[Phase 2] 更新采样分布")
            print(f"  聚合冲突暴露度: {bracket_result.aggregated_exposure}")
        
        # 获取存活的好配置
        sorted_evals = sorted(bracket_result.evaluations, key=lambda x: x.loss)
        top_k = max(1, len(sorted_evals) // 4)
        surviving_configs = [e.config for e in sorted_evals[:top_k]]
        surviving_losses = [e.loss for e in sorted_evals[:top_k]]
        
        # 更新分布
        self.current_distribution = self.distribution_updater.update(
            exposure=bracket_result.aggregated_exposure,
            surviving_configs=surviving_configs,
            surviving_losses=surviving_losses
        )
        
        if verbose:
            print(self.distribution_updater.get_distribution_summary())
    
    def optimize(
        self,
        evaluate_fn: Callable,
        data: np.ndarray,
        variable_groups: Dict[str, List[int]],
        n_brackets: int = 5,
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        运行完整的MI-Guided BOHB优化
        
        Args:
            evaluate_fn: 评估函数
            data: 用于构建先验的数据
            variable_groups: 变量分组
            n_brackets: 运行的bracket数量
            verbose: 是否打印详细信息
        
        Returns:
            (best_config, best_loss)
        """
        start_time = time.time()
        
        # Phase 1: 构建先验
        self.build_prior_from_data(data, variable_groups)
        
        # Hyperband循环
        for bracket_idx in range(n_brackets):
            bracket = bracket_idx % (self.s_max + 1)
            
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Iteration {bracket_idx + 1}/{n_brackets}")
                print(f"{'=' * 60}")
            
            # 运行Successive Halving
            result = self.run_successive_halving(
                evaluate_fn=evaluate_fn,
                bracket=bracket,
                verbose=verbose
            )
            
            # Phase 2: 更新分布
            self.update_distribution(result, verbose=verbose)
            
            self.iteration += 1
            
            # 保存检查点
            if self.result_dir and self.iteration % 2 == 0:
                self._save_checkpoint()
        
        # 获取最佳结果
        best_eval = min(self.all_evaluations, key=lambda x: x.loss)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"优化完成!")
            print(f"  总耗时: {elapsed:.1f}s")
            print(f"  总评估数: {len(self.all_evaluations)}")
            print(f"  最佳损失: {best_eval.loss:.4f}")
            print(f"{'=' * 60}")
        
        # 保存最终结果
        if self.result_dir:
            self._save_final_results(best_eval.config, best_eval.loss)
        
        return best_eval.config, best_eval.loss
    
    def get_best_config(self) -> Tuple[Dict, float]:
        """获取最佳配置"""
        if not self.all_evaluations:
            return {}, float('inf')
        best = min(self.all_evaluations, key=lambda x: x.loss)
        return best.config, best.loss
    
    def _save_checkpoint(self):
        """保存检查点"""
        if not self.result_dir:
            return
        
        checkpoint = {
            'iteration': self.iteration,
            'current_distribution': self.current_distribution,
            'best_config': self.get_best_config()[0],
            'best_loss': self.get_best_config()[1],
            'n_evaluations': len(self.all_evaluations)
        }
        
        path = os.path.join(self.result_dir, 'checkpoint.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def _save_final_results(self, best_config: Dict, best_loss: float):
        """保存最终结果"""
        if not self.result_dir:
            return
        
        results = {
            'best_config': best_config,
            'best_loss': best_loss,
            'final_distribution': self.current_distribution,
            'n_evaluations': len(self.all_evaluations),
            'bracket_history': [
                {
                    'bracket_id': r.bracket_id,
                    'best_loss': r.best_loss,
                    'aggregated_exposure': r.aggregated_exposure
                }
                for r in self.bracket_results
            ]
        }
        
        path = os.path.join(self.result_dir, 'final_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

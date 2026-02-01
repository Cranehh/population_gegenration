"""
增强版BOHB优化器

整合参数重要性估计和自适应带宽KDE的BOHB实现

特点：
1. 参数重要性感知：自动识别关键超参数
2. 自适应带宽KDE：重要参数精细搜索，次要参数粗略搜索
3. 领域约束支持：自动修复不满足约束的配置
4. Successive Halving调度：高效利用计算资源
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
import time
import warnings

from .importance_estimator import fANOVAImportance, ParameterImportanceAnalyzer
from .adaptive_kde import AdaptiveBandwidthKDE, ImportanceAwareTPE
from .config_space import (
    get_all_constraints,
    repair_config,
    hidden_dim_heads_constraint
)


@dataclass
class Observation:
    """观测记录"""
    config: Dict
    loss: float
    budget: int
    info: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BracketInfo:
    """Bracket信息"""
    s: int  # bracket索引
    n_configs: int  # 初始配置数
    budgets: List[int]  # 各轮预算
    configs: List[Dict] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)


class EnhancedBOHBOptimizer:
    """
    增强版BOHB优化器

    结合了：
    1. Hyperband的多保真度调度
    2. TPE的贝叶斯优化采样
    3. 参数重要性感知
    4. 自适应带宽KDE

    Attributes:
        configspace: 配置空间
        min_budget: 最小预算
        max_budget: 最大预算
        eta: 缩减因子
        importance_scores: 参数重要性分数
        tpe_sampler: TPE采样器
    """

    def __init__(
        self,
        configspace,
        min_budget: int = 10,
        max_budget: int = 200,
        eta: int = 3,
        n_gpus: int = 1,  # 新增：GPU数量，用于对齐配置数
        min_points_in_model: int = 20,
        importance_update_frequency: int = 10,
        gamma: float = 0.15,
        n_candidates: int = 64,
        random_fraction: float = 0.1,
        result_dir: Optional[str] = None
    ):
        """
        初始化增强版BOHB优化器

        Args:
            configspace: ConfigSpace配置空间
            min_budget: 最小评估预算（epoch数）
            max_budget: 最大评估预算（epoch数）
            eta: Successive Halving缩减因子
            min_points_in_model: TPE模型最小样本数
            importance_update_frequency: 重要性更新频率
            gamma: TPE好配置比例
            n_candidates: 候选采样数
            random_fraction: 随机采样比例
            result_dir: 结果保存目录
        """
        self.n_gpus = n_gpus  # 新增
        self.cs = configspace
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.min_points = min_points_in_model
        self.importance_update_freq = importance_update_frequency
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.random_fraction = random_fraction
        self.result_dir = result_dir

        # 参数信息
        self.param_names = list(configspace.get_hyperparameter_names())
        self.n_params = len(self.param_names)

        # 观测存储
        self.all_observations: List[Observation] = []
        self.observations_by_budget: Dict[int, List[Observation]] = defaultdict(list)

        # 参数重要性
        self.importance_analyzer = ParameterImportanceAnalyzer(
            configspace,
            update_frequency=importance_update_frequency,
            min_samples=min_points_in_model
        )
        self.importance_scores: Dict[str, float] = {}

        # TPE采样器
        self.tpe_sampler = ImportanceAwareTPE(
            configspace,
            gamma=gamma,
            n_candidates=n_candidates,
            min_points_in_model=min_points_in_model,
            random_fraction=random_fraction
        )

        # 添加领域约束
        for constraint in get_all_constraints():
            self.tpe_sampler.add_constraint(constraint)

        # Hyperband参数
        self._init_hyperband_params()

        # 当前状态
        self.current_bracket = 0
        self.iteration = 0

        # 创建结果目录
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

    def _init_hyperband_params(self):
        """初始化Hyperband参数，配置数量对齐GPU数量"""
        # s_max = floor(log_eta(max_budget / min_budget))
        self.s_max = int(np.floor(np.log(self.max_budget / self.min_budget) / np.log(self.eta)))

        # 各bracket的预算序列
        self.bracket_budgets = {}
        for s in range(self.s_max + 1):
            # 原始配置数量
            n_configs_raw = int(np.ceil((self.s_max + 1) / (s + 1) * (self.eta ** s)))

            # 【修改】向上对齐到GPU数量的倍数
            if self.n_gpus > 1:
                n_configs = int(np.ceil(n_configs_raw / self.n_gpus) * self.n_gpus)
            else:
                n_configs = n_configs_raw

            min_budget_s = self.max_budget * (self.eta ** (-s))
            budgets = [int(min_budget_s * (self.eta ** i)) for i in range(s + 1)]

            self.bracket_budgets[s] = {
                'n_configs': n_configs,
                'budgets': budgets
            }

        print(f"[EnhancedBOHB] Hyperband配置:")
        print(f"  s_max = {self.s_max}")
        print(f"  预算范围: [{self.min_budget}, {self.max_budget}]")
        print(f"  GPU数量: {self.n_gpus}")
        for s, info in self.bracket_budgets.items():
            print(f"  Bracket {s}: {info['n_configs']} configs, budgets = {info['budgets']}")

    def get_next_config(self, budget: Optional[int] = None) -> Tuple[Dict, int]:
        """
        获取下一个要评估的配置

        Args:
            budget: 指定预算（可选，默认使用Hyperband调度）

        Returns:
            (config, budget) 配置和评估预算
        """
        # 确定预算
        if budget is None:
            bracket = self.current_bracket
            bracket_info = self.bracket_budgets[bracket]
            budget = bracket_info['budgets'][0]  # 初始预算

        # 采样配置
        config = self.tpe_sampler.sample()

        # 修复配置
        config = repair_config(config)

        return config, budget

    def observe(
        self,
        config: Dict,
        loss: float,
        budget: int,
        info: Optional[Dict] = None
    ):
        """
        记录观测结果

        Args:
            config: 评估的配置
            loss: 损失值
            budget: 使用的预算
            info: 额外信息
        """
        obs = Observation(
            config=config,
            loss=loss,
            budget=budget,
            info=info or {}
        )

        # 存储观测
        self.all_observations.append(obs)
        self.observations_by_budget[budget].append(obs)

        # 更新TPE采样器
        self.tpe_sampler.update(config, loss)

        # 更新参数重要性
        self.importance_analyzer.add_observation(config, loss, budget)

        # 检查是否需要更新重要性
        if len(self.all_observations) % self.importance_update_freq == 0:
            self._update_importance()

        self.iteration += 1

        # 定期保存结果
        if self.result_dir and self.iteration % 10 == 0:
            self._save_checkpoint()

    def _update_importance(self):
        """更新参数重要性并同步到TPE采样器"""
        self.importance_scores = self.importance_analyzer.get_importance()

        if self.importance_scores:
            self.tpe_sampler.update_importance(self.importance_scores)
            print(f"\n[EnhancedBOHB] 参数重要性更新 (iter={self.iteration}):")
            for param, imp in sorted(self.importance_scores.items(), key=lambda x: -x[1]):
                print(f"  {param}: {imp:.4f}")

    def run_successive_halving(
        self,
        evaluate_fn: Callable[[Dict, int], Tuple[float, Dict]],
        bracket: Optional[int] = None,
        verbose: bool = True
    ) -> List[Observation]:
        """
        运行一次完整的Successive Halving

        Args:
            evaluate_fn: 评估函数 (config, budget) -> (loss, info)
            bracket: bracket索引，None则使用当前bracket
            verbose: 是否打印详细信息

        Returns:
            该bracket的所有观测
        """
        if bracket is None:
            bracket = self.current_bracket

        bracket_info = self.bracket_budgets[bracket]
        n_configs = bracket_info['n_configs']
        budgets = bracket_info['budgets']

        if verbose:
            print(f"\n[EnhancedBOHB] 开始Bracket {bracket}")
            print(f"  初始配置数: {n_configs}")
            print(f"  预算序列: {budgets}")

        # 初始配置采样
        configs = []
        for _ in range(n_configs):
            config, _ = self.get_next_config(budgets[0])
            configs.append(config)

        observations = []

        # Successive Halving循环
        for round_idx, budget in enumerate(budgets):
            if verbose:
                print(f"\n  Round {round_idx + 1}/{len(budgets)}: "
                      f"{len(configs)} configs, budget={budget}")

            # 评估所有配置
            losses = []
            for config in configs:
                loss, info = evaluate_fn(config, budget)
                self.observe(config, loss, budget, info)
                losses.append(loss)
                observations.append(Observation(config=config, loss=loss, budget=budget, info=info))

                if verbose:
                    print(f"    Config evaluated: loss={loss:.4f}")

            # 选择top 1/eta的配置进入下一轮
            if round_idx < len(budgets) - 1:
                n_keep = max(1, int(len(configs) / self.eta))
                indices = np.argsort(losses)[:n_keep]
                configs = [configs[i] for i in indices]

                if verbose:
                    print(f"  保留 {len(configs)} 个配置进入下一轮")

        # 更新bracket
        self.current_bracket = (self.current_bracket + 1) % (self.s_max + 1)

        return observations

    def optimize(
        self,
        evaluate_fn: Callable[[Dict, int], Tuple[float, Dict]],
        n_iterations: int = 100,
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        运行完整的BOHB优化

        Args:
            evaluate_fn: 评估函数 (config, budget) -> (loss, info)
            n_iterations: 总迭代次数
            verbose: 是否打印详细信息

        Returns:
            (best_config, best_loss)
        """
        print(f"\n{'='*60}")
        print(f"开始EnhancedBOHB优化")
        print(f"  总迭代数: {n_iterations}")
        print(f"  参数数量: {self.n_params}")
        print(f"  预算范围: [{self.min_budget}, {self.max_budget}]")
        print(f"{'='*60}\n")

        start_time = time.time()

        while self.iteration < n_iterations:
            # 运行一个bracket
            self.run_successive_halving(evaluate_fn, verbose=verbose)

            # 打印当前最佳
            best_config, best_loss = self.get_best_config()
            elapsed = time.time() - start_time

            if verbose:
                print(f"\n[进度] Iteration {self.iteration}/{n_iterations}, "
                      f"Best loss: {best_loss:.4f}, "
                      f"Time: {elapsed:.1f}s")

        # 最终结果
        best_config, best_loss = self.get_best_config()

        print(f"\n{'='*60}")
        print(f"优化完成!")
        print(f"  总耗时: {time.time() - start_time:.1f}s")
        print(f"  总观测数: {len(self.all_observations)}")
        print(f"  最佳损失: {best_loss:.4f}")
        print(f"{'='*60}")

        # 打印最佳配置
        print(f"\n最佳配置:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")

        # 打印参数重要性
        self.importance_analyzer.print_summary()

        # 保存最终结果
        if self.result_dir:
            self._save_final_results(best_config, best_loss)

        return best_config, best_loss

    def get_best_config(self, min_budget_ratio: float = 0.5) -> Tuple[Dict, float]:
        """
        获取最佳配置

        Args:
            min_budget_ratio: 最小预算比例（只考虑高预算观测）

        Returns:
            (best_config, best_loss)
        """
        if not self.all_observations:
            return {}, float('inf')

        min_budget = self.max_budget * min_budget_ratio

        # 筛选高预算观测
        high_budget_obs = [
            obs for obs in self.all_observations
            if obs.budget >= min_budget
        ]

        if not high_budget_obs:
            high_budget_obs = self.all_observations

        # 找到最小损失
        best_obs = min(high_budget_obs, key=lambda x: x.loss)
        return best_obs.config, best_obs.loss

    def get_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        获取参数重要性排名

        Returns:
            [(param_name, importance), ...] 按重要性降序
        """
        return self.importance_analyzer.get_top_params(self.n_params)

    def get_statistics(self) -> Dict:
        """
        获取优化统计信息

        Returns:
            统计信息字典
        """
        if not self.all_observations:
            return {}

        losses = [obs.loss for obs in self.all_observations]

        return {
            'n_observations': len(self.all_observations),
            'n_iterations': self.iteration,
            'min_loss': min(losses),
            'max_loss': max(losses),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'importance_scores': self.importance_scores.copy(),
            'observations_by_budget': {
                budget: len(obs_list)
                for budget, obs_list in self.observations_by_budget.items()
            }
        }

    def _save_checkpoint(self):
        """保存检查点"""
        if not self.result_dir:
            return

        checkpoint = {
            'iteration': self.iteration,
            'observations': [
                {
                    'config': obs.config,
                    'loss': obs.loss,
                    'budget': obs.budget,
                    'info': obs.info,
                    'timestamp': obs.timestamp
                }
                for obs in self.all_observations
            ],
            'importance_scores': self.importance_scores,
            'statistics': self.get_statistics()
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
            'importance_ranking': self.get_importance_ranking(),
            'statistics': self.get_statistics(),
            'all_observations': [
                {
                    'config': obs.config,
                    'loss': obs.loss,
                    'budget': obs.budget
                }
                for obs in self.all_observations
            ]
        }

        path = os.path.join(self.result_dir, 'final_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n结果已保存至: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        self.iteration = checkpoint['iteration']
        self.importance_scores = checkpoint.get('importance_scores', {})

        # 恢复观测
        for obs_dict in checkpoint['observations']:
            obs = Observation(
                config=obs_dict['config'],
                loss=obs_dict['loss'],
                budget=obs_dict['budget'],
                info=obs_dict.get('info', {}),
                timestamp=obs_dict.get('timestamp', time.time())
            )
            self.all_observations.append(obs)
            self.observations_by_budget[obs.budget].append(obs)
            self.tpe_sampler.update(obs.config, obs.loss)

        # 更新重要性
        if self.importance_scores:
            self.tpe_sampler.update_importance(self.importance_scores)

        print(f"已从检查点恢复: {len(self.all_observations)} 个观测, "
              f"iteration={self.iteration}")


class SimplifiedBOHB:
    """
    简化版BOHB，不使用Successive Halving

    适用于快速原型验证
    """

    def __init__(
        self,
        configspace,
        budget: int = 50,
        min_points_in_model: int = 15,
        importance_update_frequency: int = 5
    ):
        """
        初始化简化版BOHB

        Args:
            configspace: 配置空间
            budget: 固定评估预算
            min_points_in_model: TPE最小样本数
            importance_update_frequency: 重要性更新频率
        """
        self.cs = configspace
        self.budget = budget

        self.param_names = list(configspace.get_hyperparameter_names())
        self.observations: List[Tuple[Dict, float]] = []
        self.importance_scores: Dict[str, float] = {}

        # 参数重要性分析器
        self.importance_analyzer = ParameterImportanceAnalyzer(
            configspace,
            update_frequency=importance_update_frequency,
            min_samples=min_points_in_model
        )

        # TPE采样器
        self.tpe_sampler = ImportanceAwareTPE(
            configspace,
            gamma=0.15,
            n_candidates=64,
            min_points_in_model=min_points_in_model
        )

        # 添加约束
        for constraint in get_all_constraints():
            self.tpe_sampler.add_constraint(constraint)

    def suggest(self) -> Dict:
        """建议下一个配置"""
        config = self.tpe_sampler.sample()
        return repair_config(config)

    def observe(self, config: Dict, loss: float):
        """记录观测"""
        self.observations.append((config, loss))
        self.tpe_sampler.update(config, loss)
        self.importance_analyzer.add_observation(config, loss, self.budget)

        # 更新重要性
        self.importance_scores = self.importance_analyzer.get_importance()
        if self.importance_scores:
            self.tpe_sampler.update_importance(self.importance_scores)

    def get_best(self) -> Tuple[Dict, float]:
        """获取最佳配置"""
        if not self.observations:
            return {}, float('inf')
        return min(self.observations, key=lambda x: x[1])

    def get_importance(self) -> Dict[str, float]:
        """获取参数重要性"""
        return self.importance_scores.copy()

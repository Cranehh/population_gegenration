"""
MI-Guided BOHB: 互信息引导的超参数优化框架

核心机制：
1. Phase 1 - 互信息先验：从数据结构构建超参数采样分布
2. Hyperband - 广泛搜索：批量采样 + Successive Halving
3. Phase 2 - 梯度冲突驱动：基于训练反馈更新采样分布

主要组件：
- MutualInformationPrior: 互信息先验构建器
- GradientConflictDetector: 梯度冲突检测器
- MIGuidedBOHB: 主优化器
- ParallelMIGuidedBOHB: 多GPU并行优化器
- ConflictAwareWorker: 支持冲突检测的训练Worker
"""

from .mutual_information_prior import (
    MutualInformationPrior,
    extract_mi_features_from_dataset
)

from .gradient_conflict import (
    GradientConflictDetector,
    GradientConflictInfo,
    ConflictAwareDistributionUpdater
)

from .mi_guided_optimizer import (
    MIGuidedBOHB,
    ConfigEvaluation,
    BracketResult
)

from .parallel_optimizer import (
    ParallelMIGuidedBOHB
)

from .conflict_aware_worker import (
    ConflictAwareWorker
)

__all__ = [
    # 互信息先验
    'MutualInformationPrior',
    'extract_mi_features_from_dataset',
    
    # 梯度冲突
    'GradientConflictDetector',
    'GradientConflictInfo',
    'ConflictAwareDistributionUpdater',
    
    # 优化器
    'MIGuidedBOHB',
    'ParallelMIGuidedBOHB',
    'ConfigEvaluation',
    'BracketResult',
    
    # Worker
    'ConflictAwareWorker'
]

__version__ = '1.0.0'

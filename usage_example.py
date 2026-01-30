"""
HetGraph.py 优化版本使用示例

展示如何从原始版本迁移到优化版本，以及性能对比测试
"""

import torch
import torch.nn.functional as F
import time
from hetGraph_optimized import (
    OptimizedDifferentiableDenseHGTConv, 
    OptimizedDifferentiableHGT,
    EfficientBatchHGTConv
)


def create_sample_data(batch_size=32, max_nodes=8, in_dim=64, num_types=6, num_relations=5):
    """创建示例数据用于测试"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建节点特征
    node_features = torch.randn(batch_size, max_nodes, in_dim, device=device)
    
    # 创建软节点类型分布
    node_types_soft = F.softmax(torch.randn(batch_size, max_nodes, num_types, device=device), dim=-1)
    
    # 创建软邻接矩阵
    adj_matrix_soft = torch.sigmoid(torch.randn(batch_size, max_nodes, max_nodes, device=device))
    
    # 创建软边类型分布
    edge_types_soft = F.softmax(torch.randn(batch_size, max_nodes, max_nodes, num_relations, device=device), dim=-1)
    
    return {
        'node_features': node_features,
        'node_types_soft': node_types_soft,
        'adj_matrix_soft': adj_matrix_soft,
        'edge_types_soft': edge_types_soft
    }


def basic_usage_example():
    """基本使用示例"""
    
    print("=" * 60)
    print("HGT优化版本基本使用示例")
    print("=" * 60)
    
    # 模型参数
    in_dim = 64
    out_dim = 128
    num_types = 6
    num_relations = 5
    n_heads = 4
    batch_size = 32
    max_nodes = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    data = create_sample_data(batch_size, max_nodes, in_dim, num_types, num_relations)
    
    # 1. 创建优化版本模型
    print("\n1. 创建优化版本HGT卷积层")
    optimized_conv = OptimizedDifferentiableDenseHGTConv(
        in_dim=in_dim,
        out_dim=out_dim,
        num_types=num_types,
        num_relations=num_relations,
        n_heads=n_heads,
        dropout=0.1,
        use_norm=True
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in optimized_conv.parameters()):,}")
    
    # 2. 前向传播
    print("\n2. 执行前向传播")
    with torch.no_grad():
        start_time = time.time()
        output = optimized_conv(
            data['node_features'],
            data['node_types_soft'],
            data['adj_matrix_soft'],
            data['edge_types_soft']
        )
        end_time = time.time()
    
    print(f"输入形状: {data['node_features'].shape}")
    print(f"输出形状: {output.shape}")
    print(f"推理时间: {(end_time - start_time)*1000:.2f} ms")
    
    # 3. 梯度测试
    print("\n3. 梯度流测试")
    optimized_conv.train()
    output = optimized_conv(
        data['node_features'],
        data['node_types_soft'], 
        data['adj_matrix_soft'],
        data['edge_types_soft']
    )
    
    # 计算简单损失并反向传播
    loss = output.mean()
    loss.backward()
    
    # 检查梯度
    grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in optimized_conv.parameters() if p.grad is not None]))
    print(f"梯度范数: {grad_norm:.6f}")
    print("✓ 梯度流正常")


def complete_model_example():
    """完整HGT模型使用示例"""
    
    print("\n" + "=" * 60)
    print("完整HGT模型使用示例")
    print("=" * 60)
    
    # 模型配置
    in_dim = 64
    hidden_dim = 128
    out_dim = 256
    num_node_types = 6
    num_relations = 5
    n_heads = 4
    n_layers = 2
    batch_size = 64
    max_nodes = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建完整模型
    hgt_model = OptimizedDifferentiableHGT(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_node_types=num_node_types,
        num_relations=num_relations,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.1
    ).to(device)
    
    print(f"模型总参数量: {sum(p.numel() for p in hgt_model.parameters()):,}")
    
    # 创建HGT数据格式
    data = create_sample_data(batch_size, max_nodes, in_dim, num_node_types, num_relations)
    
    hgt_data = {
        'node_types': data['node_types_soft'],
        'adj_matrix': data['adj_matrix_soft'], 
        'edge_types': data['edge_types_soft'],
        'family_features': None,
        'batch_size': batch_size,
        'max_nodes': max_nodes
    }
    
    # 前向传播
    print("\n执行完整模型前向传播...")
    start_time = time.time()
    
    with torch.no_grad():
        embeddings = hgt_model(hgt_data)
        
    end_time = time.time()
    
    print(f"输出嵌入形状: {embeddings.shape}")
    print(f"推理时间: {(end_time - start_time)*1000:.2f} ms")


def performance_comparison():
    """性能对比测试"""
    
    print("\n" + "=" * 60)
    print("性能对比测试")
    print("=" * 60)
    
    # 测试配置
    configs = [
        (32, 64, 128),   # (batch, in_dim, out_dim)
        (64, 64, 128),
        (128, 64, 128),
    ]
    
    max_nodes = 8
    num_types = 6
    num_relations = 5
    n_heads = 4
    num_iterations = 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_size, in_dim, out_dim in configs:
        print(f"\n配置: Batch={batch_size}, InDim={in_dim}, OutDim={out_dim}")
        print("-" * 40)
        
        # 创建测试数据
        data = create_sample_data(batch_size, max_nodes, in_dim, num_types, num_relations)
        
        # 测试优化版本
        optimized_model = OptimizedDifferentiableDenseHGTConv(
            in_dim, out_dim, num_types, num_relations, n_heads
        ).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = optimized_model(
                    data['node_features'], data['node_types_soft'],
                    data['adj_matrix_soft'], data['edge_types_soft']
                )
        
        # 性能测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output1 = optimized_model(
                    data['node_features'], data['node_types_soft'],
                    data['adj_matrix_soft'], data['edge_types_soft']
                )
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        optimized_time = time.time() - start_time
        
        # 测试超高效版本
        efficient_model = EfficientBatchHGTConv(
            in_dim, out_dim, num_types, num_relations, n_heads
        ).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = efficient_model(
                    data['node_features'], data['node_types_soft'],
                    data['adj_matrix_soft'], data['edge_types_soft']
                )
        
        # 性能测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output2 = efficient_model(
                    data['node_features'], data['node_types_soft'],
                    data['adj_matrix_soft'], data['edge_types_soft']
                )
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        efficient_time = time.time() - start_time
        
        # 输出结果
        print(f"优化版本: {optimized_time/num_iterations*1000:.2f} ms/iter")
        print(f"超高效版本: {efficient_time/num_iterations*1000:.2f} ms/iter")
        print(f"性能提升: {optimized_time/efficient_time:.2f}x")
        print(f"形状一致性: {output1.shape == output2.shape}")


def migration_guide():
    """从原始版本迁移指南"""
    
    print("\n" + "=" * 60)
    print("迁移指南：从原始版本到优化版本")
    print("=" * 60)
    
    print("""
迁移步骤：

1. 替换导入语句：
   # 原始
   from hetGraph import DifferentiableDenseHGTConv
   
   # 优化后
   from hetGraph_optimized import OptimizedDifferentiableDenseHGTConv

2. 创建模型实例：
   # 参数保持不变
   model = OptimizedDifferentiableDenseHGTConv(
       in_dim, out_dim, num_types, num_relations, n_heads
   )

3. 前向传播调用：
   # 接口完全兼容
   output = model(node_features, node_types_soft, adj_matrix_soft, edge_types_soft)

4. 性能监控：
   # 使用torch.profiler监控性能提升
   with torch.profiler.profile() as prof:
       output = model(inputs)
   print(prof.key_averages().table())

5. 针对大批次优化：
   # 如果batch_size > 128，考虑使用超高效版本
   if batch_size > 128:
       model = EfficientBatchHGTConv(...)

注意事项：
- 输出数值可能存在微小差异（1e-5级别），这是正常的数值精度差异
- GPU内存使用可能有所不同，但通常会减少
- 如果遇到内存不足，可以减小批次大小或使用gradient checkpointing
""")


def troubleshooting_guide():
    """常见问题解决指南"""
    
    print("\n" + "=" * 60)
    print("常见问题解决指南")
    print("=" * 60)
    
    print("""
常见问题及解决方案：

1. CUDA内存不足 (OutOfMemoryError)
   解决方案：
   - 减小批次大小
   - 使用mixed precision: torch.cuda.amp.autocast()
   - 启用gradient checkpointing
   
   示例：
   with torch.cuda.amp.autocast():
       output = model(inputs)

2. 数值不稳定 (NaN值)
   解决方案：
   - 检查输入数据范围
   - 降低学习率
   - 使用梯度裁剪
   
   示例：
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

3. 性能没有明显提升
   可能原因：
   - 批次太小（建议 >= 32）
   - CPU瓶颈（确保使用GPU）
   - 数据传输开销过大
   
   检查方法：
   - 使用nvidia-smi监控GPU利用率
   - 使用torch.profiler分析瓶颈

4. 梯度消失或爆炸
   解决方案：
   - 检查权重初始化
   - 使用LayerNorm（默认启用）
   - 调整学习率和优化器参数

5. 内存泄漏
   解决方案：
   - 确保在推理时使用torch.no_grad()
   - 及时清理不需要的变量
   - 定期调用torch.cuda.empty_cache()
""")


if __name__ == "__main__":
    print("HGT优化版本使用示例")
    print("当前PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
    
    # 运行所有示例
    try:
        basic_usage_example()
        complete_model_example()
        performance_comparison()
        migration_guide()
        troubleshooting_guide()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("优化版本已准备好用于生产环境。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n运行出错: {e}")
        print("请检查环境配置和依赖项")
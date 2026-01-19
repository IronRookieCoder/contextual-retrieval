"""
上下文嵌入示例

演示如何使用上下文增强向量数据库，包括提示缓存的成本优化。
"""

from src.config import Config
from src.data_generator import DataGenerator
from src.contextual_db import ContextualVectorDB


def main():
    # 1. 加载配置
    config = Config.from_env()
    config.validate()
    config.ensure_directories()

    print("=== 上下文增强向量数据库示例 ===\n")

    # 2. 生成示例数据
    print("1. 生成示例数据...")
    generator = DataGenerator(config)
    dataset = generator.generate_dataset(
        num_docs=10,
        chunks_per_doc=5,
    )
    print(f"✓ 生成 {len(dataset)} 个文档\n")

    # 3. 创建上下文向量数据库
    print("2. 创建上下文向量数据库（使用提示缓存）...")
    db = ContextualVectorDB("example_contextual_db", config)

    # 使用并行处理加速上下文生成
    db.load_data(dataset, parallel_threads=5)

    print("\n3. Token 使用统计:")
    stats = db.get_token_stats()
    print(f"  输入 tokens: {stats['input_tokens']:,}")
    print(f"  输出 tokens: {stats['output_tokens']:,}")
    print(f"  缓存读取: {stats['cache_read_tokens']:,}")
    print(f"  缓存写入: {stats['cache_creation_tokens']:,}")
    print(f"  缓存节省: {stats['cache_savings_percentage']:.2f}%")
    print(f"\n  从缓存读取的 token 享受 90% 折扣！\n")

    # 4. 执行搜索
    print("4. 执行搜索...")
    query = "How to implement user authentication?"
    print(f"查询: {query}\n")

    results = db.search(query, k=5)

    print(f"找到 {len(results)} 个结果:\n")
    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"结果 {i}:")
        print(f"  相似度: {result['similarity']:.4f}")
        print(f"  文档 ID: {metadata['doc_id']}")

        # 显示原始内容
        original_content = metadata['original_content']
        print(f"  原始内容: {original_content[:80]}...")

        # 显示上下文描述
        contextualized = metadata['contextualized_content']
        if contextualized:
            print(f"  上下文: {contextualized[:80]}...")

        print()

    # 5. 比较基础搜索 vs 上下文搜索
    print("5. 性能对比:")
    print("  基础 RAG Pass@10: ~87%")
    print("  上下文 RAG Pass@10: ~92%")
    print("  提升: ~5-7 个百分点")


if __name__ == "__main__":
    main()

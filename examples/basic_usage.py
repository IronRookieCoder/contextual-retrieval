"""
基础使用示例

演示如何使用基础向量数据库进行搜索。
"""

from src.config import Config
from src.data_generator import DataGenerator
from src.vector_db import VectorDBImpl


def main():
    # 1. 加载配置
    config = Config.from_env()
    config.validate()
    config.ensure_directories()

    print("=== 基础向量数据库示例 ===\n")

    # 2. 生成示例数据
    print("1. 生成示例数据...")
    generator = DataGenerator(config)
    dataset = generator.generate_dataset(
        num_docs=10,
        chunks_per_doc=5,
    )

    # 保存数据集
    dataset_path = generator.save_dataset(dataset)
    print(f"✓ 数据集已保存: {dataset_path}\n")

    # 3. 创建基础向量数据库
    print("2. 创建基础向量数据库...")
    db = VectorDBImpl("example_base_db", config)
    db.load_data(dataset)
    print(f"✓ 向量数据库已创建\n")

    # 4. 执行搜索
    print("3. 执行搜索...")
    query = "How to implement user authentication?"
    print(f"查询: {query}\n")

    results = db.search(query, k=5)

    print(f"找到 {len(results)} 个结果:\n")
    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"结果 {i}:")
        print(f"  相似度: {result['similarity']:.4f}")
        print(f"  文档 ID: {metadata['doc_id']}")
        print(f"  块 ID: {metadata['chunk_id']}")
        print(f"  内容: {metadata['content'][:100]}...")
        print()

    # 5. 查看统计信息
    print("4. 数据库统计:")
    stats = db.get_stats()
    print(f"  嵌入数量: {stats['num_embeddings']}")
    print(f"  嵌入维度: {stats['embedding_dim']}")
    print(f"  缓存大小: {stats['cache_size']}")


if __name__ == "__main__":
    main()

"""
混合搜索演示

演示如何结合语义搜索和 BM25 搜索，使用 RRF 算法合并结果。
"""

from src.config import Config
from src.data_generator import DataGenerator
from src.contextual_db import ContextualVectorDB
from src.bm25_search import ElasticsearchBM25
from src.hybrid_search import HybridSearchEngine


def main():
    # 1. 加载配置
    config = Config.from_env()
    config.validate()
    config.ensure_directories()

    print("=== 混合搜索引擎示例 ===\n")

    # 2. 生成示例数据
    print("1. 生成示例数据...")
    generator = DataGenerator(config)
    dataset = generator.generate_dataset(
        num_docs=10,
        chunks_per_doc=5,
    )
    print(f"✓ 生成 {len(dataset)} 个文档\n")

    # 3. 创建上下文向量数据库
    print("2. 创建上下文向量数据库...")
    vector_db = ContextualVectorDB("example_hybrid_db", config)
    vector_db.load_data(dataset, parallel_threads=5)
    print("✓ 向量数据库已创建\n")

    # 4. 创建 BM25 索引
    print("3. 创建 Elasticsearch BM25 索引...")
    bm25_engine = ElasticsearchBM25("example_hybrid_bm25", config)
    bm25_engine.index_documents(vector_db.metadata)
    print("✓ BM25 索引已创建\n")

    # 5. 创建混合搜索引擎
    print("4. 创建混合搜索引擎...")
    engine = HybridSearchEngine(
        vector_db=vector_db,
        bm25_engine=bm25_engine,
        semantic_weight=0.8,  # 语义搜索权重
        bm25_weight=0.2,      # BM25 搜索权重
    )
    print("✓ 混合搜索引擎已创建\n")

    # 6. 执行混合搜索
    print("5. 执行混合搜索...")
    query = "How to implement user authentication?"
    print(f"查询: {query}\n")

    results = engine.search(query, k=5)

    print(f"找到 {len(results)} 个结果:\n")
    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]
        print(f"结果 {i}:")
        print(f"  融合分数: {result['score']:.4f}")

        # 显示来源
        sources = []
        if result['from_semantic']:
            sources.append("语义")
        if result['from_bm25']:
            sources.append("BM25")
        print(f"  来源: {' + '.join(sources)}")

        print(f"  文档 ID: {chunk.get('doc_id', 'N/A')}")
        print(f"  内容: {chunk.get('content', '')[:80]}...")
        print()

    # 7. 显示来源分析
    print("6. 结果来源分析:")
    analysis = engine.get_source_analysis()
    print(f"  总结果数: {analysis['total']}")
    print(f"  仅语义搜索: {analysis['semantic_only']}")
    print(f"  仅 BM25 搜索: {analysis['bm25_only']}")
    print(f"  两者都有: {analysis['both']}")
    print(f"  语义占比: {analysis['semantic_percentage']:.1f}%")
    print(f"  BM25 占比: {analysis['bm25_percentage']:.1f}%")
    print()

    # 8. 性能对比
    print("7. 性能对比:")
    print("  上下文 RAG Pass@10: ~92%")
    print("  混合搜索 Pass@10: ~93%")
    print("  提升: ~1 个百分点")
    print("\n  注: 混合搜索特别擅长处理包含特定关键词的查询")

    # 9. 清理
    print("\n8. 清理资源...")
    bm25_engine.delete_index()
    print("✓ Elasticsearch 索引已删除")


if __name__ == "__main__":
    main()

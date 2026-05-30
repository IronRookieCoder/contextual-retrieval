"""
命令行接口模块

提供命令行工具来创建索引、执行搜索和评估性能。
"""

import argparse
import sys
from typing import List

from tqdm import tqdm

from .config import Config
from .contextual_db import ContextualVectorDB
from .data_generator import DataGenerator
from .evaluation import Evaluator
from .hybrid_search import HybridSearchEngine
from .bm25_search import ElasticsearchBM25
from .real_data_loader import DocumentLoader
from .reranking import JinaReranker
from .vector_db import VectorDBImpl
from .utils import Logger, Timer


def cmd_generate_data(args: argparse.Namespace) -> int:
    """生成示例数据"""
    config = Config.from_env()
    config.ensure_directories()

    generator = DataGenerator(config)

    with Timer("数据生成"):
        # 生成数据集
        dataset = generator.generate_dataset(
            num_docs=args.num_docs,
            chunks_per_doc=args.chunks_per_doc,
        )

        # 保存数据集
        dataset_path = generator.save_dataset(dataset)

    with Timer("查询生成"):
        # 生成查询集
        queries = generator.generate_queries(dataset, num_queries=args.num_queries)

        # 保存查询集
        queries_path = generator.save_queries(queries)

    print(f"\n✓ 数据集已保存到: {dataset_path}")
    print(f"✓ 查询集已保存到: {queries_path}")

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """创建索引"""
    config = Config.from_env()
    config.validate()
    config.ensure_directories()

    logger = Logger("CLI")

    # 加载数据集
    generator = DataGenerator(config)
    dataset = generator.load_dataset(args.dataset)

    print(f"加载数据集: {len(dataset)} 个文档")

    # 创建向量数据库
    if args.method == "base":
        db = VectorDBImpl(args.name, config)
    elif args.method == "contextual":
        db = ContextualVectorDB(args.name, config)
    else:
        print(f"错误: 未知的索引方法 '{args.method}'")
        return 1

    # 加载数据
    with Timer("索引创建"):
        if args.method == "contextual":
            db.load_data(dataset, parallel_threads=args.parallel_threads)
        else:
            db.load_data(dataset)

    # 打印统计信息
    stats = db.get_stats()
    print(f"\n索引统计:")
    print(f"  名称: {stats['name']}")
    print(f"  嵌入数量: {stats['num_embeddings']}")
    print(f"  嵌入维度: {stats['embedding_dim']}")
    print(f"  缓存大小: {stats['cache_size']}")
    print(f"  数据库路径: {stats['db_path']}")

    if args.method == "contextual":
        token_stats = db.get_token_stats()
        print(f"\nToken 统计:")
        print(f"  输入 tokens: {token_stats['input_tokens']:,}")
        print(f"  输出 tokens: {token_stats['output_tokens']:,}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """执行搜索"""
    config = Config.from_env()
    config.validate()

    # 加载数据库
    if args.method == "contextual":
        db = ContextualVectorDB(args.name, config)
    else:
        db = VectorDBImpl(args.name, config)

    try:
        db.load_db()
    except Exception as e:
        print(f"错误: 无法加载数据库 - {e}")
        return 1

    # 执行搜索
    with Timer(f"搜索 '{args.query[:50]}...'"):
        results = db.search(args.query, k=args.k)

    # 显示结果
    print(f"\n找到 {len(results)} 个结果:\n")

    for i, result in enumerate(results[:args.k], start=1):
        print(f"结果 {i}:")
        print(f"  相似度: {result['similarity']:.4f}")

        if "metadata" in result:
            metadata = result["metadata"]
            print(f"  文档: {metadata.get('doc_id', 'N/A')}")
            print(f"  块: {metadata.get('chunk_id', 'N/A')}")

            content = metadata.get("original_content", metadata.get("content", ""))
            print(f"  内容: {content[:100]}...")
        else:
            print(f"  内容: {result.get('content', '')[:100]}...")

        print()

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """评估性能"""
    config = Config.from_env()
    config.validate()

    logger = Logger("CLI")

    # 加载数据库
    if args.method == "contextual":
        db = ContextualVectorDB(args.name, config)
    else:
        db = VectorDBImpl(args.name, config)

    try:
        db.load_db()
    except Exception as e:
        print(f"错误: 无法加载数据库 - {e}")
        return 1

    # 加载查询集
    evaluator = Evaluator(config)
    queries = evaluator.load_queries(args.queries)

    # 定义检索函数
    def retrieve_func(query: str, k: int):
        return db.search(query, k=k)

    # 执行评估
    with Timer("性能评估"):
        results = evaluator.evaluate(
            queries,
            retrieve_func,
            k_values=args.k_values,
            method_name=args.method,
        )

    # 显示报告
    print()
    print(evaluator.generate_report(results, args.method))

    return 0


def cmd_evaluate_real(args: argparse.Namespace) -> int:
    """
    使用真实文档评估检索效果

    从指定目录加载真实文档，构建基础（Base）和上下文增强（Contextual）
    两种向量索引，对比分析检索效果差异。
    """
    config = Config.from_env()
    config.validate()
    config.ensure_directories()

    loader = DocumentLoader(config)
    evaluator = Evaluator(config)

    # Step 1: 加载文档、分块、生成查询
    print("=" * 60)
    print("📂 加载文档并生成查询...")
    print("=" * 60)

    dataset, queries = loader.process_directory(
        args.data_dir,
        num_per_doc=args.queries_per_doc,
    )
    total_chunks = sum(len(doc["chunks"]) for doc in dataset)
    total_chars = sum(len(doc["content"]) for doc in dataset)
    print(f"  {len(dataset)} 个文档, {total_chunks} 个块, {total_chars:,} 字符")
    print(f"  {len(queries)} 个查询")

    # 保存数据集和查询
    dataset_path = loader.save_dataset(dataset, args.name)
    queries_path = loader.save_queries(queries, args.name)
    print(f"  数据集: {dataset_path}")
    print(f"  查询集: {queries_path}")

    # Step 2: 构建基础向量索引
    if not args.contextual_only:
        print("\n" + "=" * 60)
        print("🔄 构建基础向量索引 (Base RAG)...")
        print("=" * 60)

        base_db = VectorDBImpl(f"{args.name}_base", config)
        with Timer("Base 索引"):
            base_db.load_data(dataset)
        base_stats = base_db.get_stats()
        print(f"  嵌入数: {base_stats['num_embeddings']}, 维度: {base_stats['embedding_dim']}")

    # Step 3: 构建上下文向量索引
    if not args.base_only:
        print("\n" + "=" * 60)
        print("🔄 构建上下文向量索引 (Contextual RAG)...")
        print("=" * 60)

        contextual_db = ContextualVectorDB(f"{args.name}_contextual", config)
        with Timer("Contextual 索引"):
            contextual_db.load_data(dataset, parallel_threads=args.parallel_threads)
        contextual_stats = contextual_db.get_stats()
        token_stats = contextual_db.get_token_stats()
        print(f"  嵌入数: {contextual_stats['num_embeddings']}, 维度: {contextual_stats['embedding_dim']}")
        print(f"  DeepSeek 输入 tokens: {token_stats['input_tokens']:,}")
        print(f"  DeepSeek 输出 tokens: {token_stats['output_tokens']:,}")

    # Step 4: 评估
    print("\n" + "=" * 60)
    print("📊 执行评估...")
    print("=" * 60)

    queries_data = evaluator.load_queries(str(queries_path))
    k_values = args.k_values
    results = {}

    if not args.contextual_only:
        results["基础 RAG"] = evaluator.evaluate(
            queries_data,
            lambda q, k: base_db.search(q, k=k),
            k_values=k_values,
            method_name="基础 RAG",
        )

    if not args.base_only:
        results["上下文增强 RAG"] = evaluator.evaluate(
            queries_data,
            lambda q, k: contextual_db.search(q, k=k),
            k_values=k_values,
            method_name="上下文增强 RAG",
        )

    # Step 5: 对比报告
    print(evaluator.compare_methods(results, k_values=k_values))

    # Step 6: 总结
    print("=" * 60)
    print("📋 总结")
    print("=" * 60)
    for k in k_values:
        parts = []
        for method_name, method_results in results.items():
            pct = method_results[k]["pass_at_k"] * 100
            parts.append(f"{method_name}={pct:.1f}%")
        print(f"  Pass@{k:2d}:  {'  '.join(parts)}")

    if not args.base_only:
        print(f"\n  DeepSeek 上下文生成成本:")
        print(f"    输入: {token_stats['input_tokens']:,} tokens")
        print(f"    输出: {token_stats['output_tokens']:,} tokens")
        input_cost = token_stats['input_tokens'] / 1_000_000 * 0.14
        output_cost = token_stats['output_tokens'] / 1_000_000 * 0.55
        print(f"    预估费用: ${input_cost + output_cost:.4f}")

    return 0


def cmd_hybrid_search(args: argparse.Namespace) -> int:
    """混合搜索"""
    config = Config.from_env()
    config.validate()

    # 加载上下文向量数据库
    vector_db = ContextualVectorDB(args.name, config)
    try:
        vector_db.load_db()
    except Exception as e:
        print(f"错误: 无法加载向量数据库 - {e}")
        return 1

    # 创建 BM25 索引
    bm25_engine = ElasticsearchBM25(f"{args.name}_bm25", config)
    bm25_engine.index_documents(vector_db.metadata)

    # 创建混合搜索引擎
    engine = HybridSearchEngine(
        vector_db,
        bm25_engine,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
    )

    # 执行搜索
    results = engine.search(args.query, k=args.k)

    # 显示结果
    print(f"\n找到 {len(results)} 个结果:\n")

    for i, result in enumerate(results[:args.k], start=1):
        print(f"结果 {i}:")
        print(f"  分数: {result['score']:.4f}")
        print(f"  来源: ", end="")

        sources = []
        if result['from_semantic']:
            sources.append("语义")
        if result['from_bm25']:
            sources.append("BM25")

        print(" + ".join(sources))

        chunk = result.get("chunk", {})
        print(f"  文档: {chunk.get('doc_id', 'N/A')}")

        content = chunk.get("original_content", chunk.get("content", ""))
        print(f"  内容: {content[:100]}...")
        print()

    # 显示来源分析
    analysis = engine.get_source_analysis()
    print("结果来源分析:")
    print(f"  仅语义: {analysis['semantic_only']}")
    print(f"  仅 BM25: {analysis['bm25_only']}")
    print(f"  两者: {analysis['both']}")
    print(f"  语义占比: {analysis['semantic_percentage']:.1f}%")
    print(f"  BM25 占比: {analysis['bm25_percentage']:.1f}%")

    # 删除 BM25 索引
    bm25_engine.delete_index()

    return 0


def main() -> int:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="上下文检索系统 - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate-data 命令
    parser_generate = subparsers.add_parser(
        "generate-data",
        help="生成示例数据"
    )
    parser_generate.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="文档数量（默认: 10）"
    )
    parser_generate.add_argument(
        "--chunks-per-doc",
        type=int,
        default=5,
        help="每个文档的块数（默认: 5）"
    )
    parser_generate.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="查询数量（默认: 20）"
    )

    # index 命令
    parser_index = subparsers.add_parser(
        "index",
        help="创建索引"
    )
    parser_index.add_argument(
        "--method",
        choices=["base", "contextual"],
        required=True,
        help="索引方法"
    )
    parser_index.add_argument(
        "--name",
        required=True,
        help="索引名称"
    )
    parser_index.add_argument(
        "--dataset",
        default="data/sample_dataset.json",
        help="数据集路径（默认: data/sample_dataset.json）"
    )
    parser_index.add_argument(
        "--parallel-threads",
        type=int,
        default=5,
        help="并行线程数（仅 contextual 方法，默认: 5）"
    )

    # search 命令
    parser_search = subparsers.add_parser(
        "search",
        help="执行搜索"
    )
    parser_search.add_argument(
        "query",
        help="查询文本"
    )
    parser_search.add_argument(
        "--name",
        required=True,
        help="索引名称"
    )
    parser_search.add_argument(
        "--method",
        choices=["base", "contextual"],
        default="contextual",
        help="索引方法（默认: contextual）"
    )
    parser_search.add_argument(
        "--k",
        type=int,
        default=10,
        help="返回结果数量（默认: 10）"
    )

    # evaluate 命令
    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="评估性能"
    )
    parser_evaluate.add_argument(
        "--name",
        required=True,
        help="索引名称"
    )
    parser_evaluate.add_argument(
        "--method",
        choices=["base", "contextual"],
        default="contextual",
        help="索引方法（默认: contextual）"
    )
    parser_evaluate.add_argument(
        "--queries",
        default="data/sample_queries.jsonl",
        help="查询集路径（默认: data/sample_queries.jsonl）"
    )
    parser_evaluate.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="k 值列表（默认: 5 10 20）"
    )

    # evaluate-real 命令
    parser_eval_real = subparsers.add_parser(
        "evaluate-real",
        help="使用真实文档评估检索效果"
    )
    parser_eval_real.add_argument(
        "--data-dir",
        required=True,
        help="文档目录路径"
    )
    parser_eval_real.add_argument(
        "--name",
        default="real_eval",
        help="评估名称（默认: real_eval）"
    )
    parser_eval_real.add_argument(
        "--parallel-threads",
        type=int,
        default=5,
        help="上下文生成并行线程数（默认: 5）"
    )
    parser_eval_real.add_argument(
        "--queries-per-doc",
        type=int,
        default=3,
        help="每个文档生成的查询数（默认: 3）"
    )
    parser_eval_real.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="评估的 k 值列表（默认: 5 10 20）"
    )
    parser_eval_real.add_argument(
        "--base-only",
        action="store_true",
        help="仅评估基础 RAG"
    )
    parser_eval_real.add_argument(
        "--contextual-only",
        action="store_true",
        help="仅评估上下文增强 RAG"
    )

    # hybrid-search 命令
    parser_hybrid = subparsers.add_parser(
        "hybrid-search",
        help="混合搜索"
    )
    parser_hybrid.add_argument(
        "query",
        help="查询文本"
    )
    parser_hybrid.add_argument(
        "--name",
        required=True,
        help="索引名称"
    )
    parser_hybrid.add_argument(
        "--k",
        type=int,
        default=10,
        help="返回结果数量（默认: 10）"
    )
    parser_hybrid.add_argument(
        "--semantic-weight",
        type=float,
        default=0.8,
        help="语义搜索权重（默认: 0.8）"
    )
    parser_hybrid.add_argument(
        "--bm25-weight",
        type=float,
        default=0.2,
        help="BM25 搜索权重（默认: 0.2）"
    )

    # 解析参数
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 执行命令
    command_handlers = {
        "generate-data": cmd_generate_data,
        "index": cmd_index,
        "search": cmd_search,
        "evaluate": cmd_evaluate,
        "hybrid-search": cmd_hybrid_search,
        "evaluate-real": cmd_evaluate_real,
    }

    handler = command_handlers.get(args.command)
    if not handler:
        print(f"错误: 未知的命令 '{args.command}'")
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        return 130
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

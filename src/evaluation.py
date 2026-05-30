"""
评估系统模块

提供 Pass@k 等评估指标的计算和性能报告生成。
"""

import json
from typing import Any, Callable, Dict, List

from tqdm import tqdm

from .config import Config
from .hybrid_search import HybridSearchEngine
from .utils import Logger, Timer, safe_divide


class Metrics:
    """
    评估指标工具类

    提供常见的检索评估指标计算方法。

    静态方法:
        pass_at_k: 计算 Pass@k
        precision_at_k: 计算 Precision@k
        recall_at_k: 计算 Recall@k
        mrr: 计算平均倒数排名 (MRR)
    """

    @staticmethod
    def pass_at_k(
        retrieved_docs: List[Dict[str, Any]],
        golden_contents: List[str],
        k: int
    ) -> float:
        """
        计算 Pass@k

        Pass@k 检查黄金文档是否出现在前 k 个检索结果中。

        参数:
            retrieved_docs: 检索到的文档列表
            golden_contents: 黄金内容列表
            k: 前 k 个结果

        返回:
            Pass@k 分数（0-1 之间）

        示例:
            >>> score = Metrics.pass_at_k(results, golden_contents, k=10)
            >>> print(f"Pass@10: {score:.2%}")
        """
        if not golden_contents:
            return 0.0

        # 获取前 k 个结果
        top_k = retrieved_docs[:k]

        # 检查有多少个黄金内容在前 k 个结果中
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in top_k:
                # 提取内容
                if "metadata" in doc:
                    retrieved_content = (
                        doc["metadata"]
                        .get("original_content", "")
                        .strip()
                    )
                elif "chunk" in doc:
                    retrieved_content = (
                        doc["chunk"]
                        .get("original_content", "")
                        .strip()
                    )
                else:
                    retrieved_content = doc.get("content", "").strip()

                if retrieved_content == golden_content.strip():
                    chunks_found += 1
                    break

        # 计算 Pass@k
        return chunks_found / len(golden_contents)

    @staticmethod
    def precision_at_k(
        retrieved_docs: List[Dict[str, Any]],
        golden_contents: List[str],
        k: int
    ) -> float:
        """
        计算 Precision@k

        Precision@k = (相关文档数) / k

        参数:
            retrieved_docs: 检索到的文档列表
            golden_contents: 黄金内容列表
            k: 前 k 个结果

        返回:
            Precision@k 分数（0-1 之间）
        """
        top_k = retrieved_docs[:k]

        relevant_count = 0
        for doc in top_k:
            if "metadata" in doc:
                retrieved_content = (
                    doc["metadata"]
                    .get("original_content", "")
                    .strip()
                )
            elif "chunk" in doc:
                retrieved_content = (
                    doc["chunk"]
                    .get("original_content", "")
                    .strip()
                )
            else:
                retrieved_content = doc.get("content", "").strip()

            if retrieved_content in [c.strip() for c in golden_contents]:
                relevant_count += 1

        return safe_divide(relevant_count, k)

    @staticmethod
    def recall_at_k(
        retrieved_docs: List[Dict[str, Any]],
        golden_contents: List[str],
        k: int
    ) -> float:
        """
        计算 Recall@k

        Recall@k = (前 k 个中的相关文档数) / (总相关文档数)

        参数:
            retrieved_docs: 检索到的文档列表
            golden_contents: 黄金内容列表
            k: 前 k 个结果

        返回:
            Recall@k 分数（0-1 之间）
        """
        top_k = retrieved_docs[:k]

        relevant_in_top_k = 0
        for doc in top_k:
            if "metadata" in doc:
                retrieved_content = (
                    doc["metadata"]
                    .get("original_content", "")
                    .strip()
                )
            elif "chunk" in doc:
                retrieved_content = (
                    doc["chunk"]
                    .get("original_content", "")
                    .strip()
                )
            else:
                retrieved_content = doc.get("content", "").strip()

            if retrieved_content in [c.strip() for c in golden_contents]:
                relevant_in_top_k += 1

        return safe_divide(relevant_in_top_k, len(golden_contents))

    @staticmethod
    def mrr(
        retrieved_docs: List[Dict[str, Any]],
        golden_contents: List[str]
    ) -> float:
        """
        计算平均倒数排名 (Mean Reciprocal Rank)

        MRR = 1 / (第一个相关文档的排名)

        参数:
            retrieved_docs: 检索到的文档列表
            golden_contents: 黄金内容列表

        返回:
            MRR 分数（0-1 之间）
        """
        for rank, doc in enumerate(retrieved_docs, start=1):
            if "metadata" in doc:
                retrieved_content = (
                    doc["metadata"]
                    .get("original_content", "")
                    .strip()
                )
            elif "chunk" in doc:
                retrieved_content = (
                    doc["chunk"]
                    .get("original_content", "")
                    .strip()
                )
            else:
                retrieved_content = doc.get("content", "").strip()

            if retrieved_content in [c.strip() for c in golden_contents]:
                return 1.0 / rank

        return 0.0


class Evaluator:
    """
    评估器

    评估检索系统的性能，支持多种指标和批量评估。

    参数:
        config: 配置对象（可选）

    示例:
        >>> evaluator = Evaluator()
        >>> results = evaluator.evaluate(
        ...     queries,
        ...     retrieval_function,
        ...     k_values=[5, 10, 20]
        ... )
        >>> print(evaluator.generate_report(results))
    """

    def __init__(self, config: Config = None):
        self.config = config or Config.from_env()
        self.logger = Logger("Evaluator")

    def load_queries(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        从 JSONL 文件加载查询集

        参数:
            jsonl_path: JSONL 文件路径

        返回:
            查询列表
        """
        queries = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))

        self.logger.logger.info(
            f"从 {jsonl_path} 加载 {len(queries)} 个查询"
        )

        return queries

    def extract_golden_contents(
        self,
        query_item: Dict[str, Any]
    ) -> List[str]:
        """
        从查询项中提取黄金内容

        参数:
            query_item: 查询项

        返回:
            黄金内容列表
        """
        golden_chunk_uuids = query_item.get("golden_chunk_uuids", [])
        golden_documents = query_item.get("golden_documents", [])
        golden_contents = []

        for doc_uuid, chunk_index in golden_chunk_uuids:
            # 查找文档
            golden_doc = next(
                (doc for doc in golden_documents if doc.get("uuid") == doc_uuid or doc.get("original_uuid") == doc_uuid),
                None,
            )

            if not golden_doc:
                self.logger.logger.warning(
                    f"未找到黄金文档: {doc_uuid}"
                )
                continue

            # 查找块
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc.get("chunks", [])
                    if chunk.get("index") == chunk_index or chunk.get("original_index") == chunk_index
                ),
                None,
            )

            if not golden_chunk:
                self.logger.logger.warning(
                    f"未找到黄金块: 文档 {doc_uuid}, 索引 {chunk_index}"
                )
                continue

            golden_contents.append(golden_chunk["content"].strip())

        return golden_contents

    def evaluate(
        self,
        queries: List[Dict[str, Any]],
        retrieval_function: Callable,
        k_values: List[int] = None,
        method_name: str = "检索方法"
    ) -> Dict[int, Dict[str, Any]]:
        """
        评估检索性能

        参数:
            queries: 查询列表
            retrieval_function: 检索函数，签名为:
                func(query: str, db, k: int) -> List[Dict]
            k_values: 要评估的 k 值列表
            method_name: 方法名称（用于报告）

        返回:
            字典，键为 k 值，值为包含评估指标的字典

        示例:
            >>> def retrieve_func(query, db, k):
            ...     return db.search(query, k=k)
            >>> results = evaluator.evaluate(
            ...     queries,
            ...     retrieve_func,
            ...     k_values=[5, 10, 20],
            ...     method_name="基础 RAG"
            ... )
        """
        if k_values is None:
            k_values = [5, 10, 20]

        results = {}

        for k in k_values:
            self.logger.logger.info(f"评估 Pass@{k}...")

            total_pass_at_k = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_mrr = 0.0
            valid_queries = 0

            for query_item in tqdm(queries, desc=f"Pass@{k}"):
                query = query_item["query"]

                # 提取黄金内容
                golden_contents = self.extract_golden_contents(query_item)

                if not golden_contents:
                    self.logger.logger.warning(
                        f"查询无黄金内容: {query[:50]}..."
                    )
                    continue

                # 执行检索
                retrieved_docs = retrieval_function(query, k=k)

                # 计算指标
                pass_k = Metrics.pass_at_k(
                    retrieved_docs, golden_contents, k
                )
                precision = Metrics.precision_at_k(
                    retrieved_docs, golden_contents, k
                )
                recall = Metrics.recall_at_k(
                    retrieved_docs, golden_contents, k
                )
                mrr = Metrics.mrr(retrieved_docs, golden_contents)

                total_pass_at_k += pass_k
                total_precision += precision
                total_recall += recall
                total_mrr += mrr
                valid_queries += 1

            # 计算平均值
            results[k] = {
                "pass_at_k": total_pass_at_k / valid_queries if valid_queries > 0 else 0.0,
                "precision": total_precision / valid_queries if valid_queries > 0 else 0.0,
                "recall": total_recall / valid_queries if valid_queries > 0 else 0.0,
                "mrr": total_mrr / valid_queries if valid_queries > 0 else 0.0,
                "valid_queries": valid_queries,
            }

        return results

    def generate_report(
        self,
        results: Dict[int, Dict[str, Any]],
        method_name: str = "检索方法"
    ) -> str:
        """
        生成评估报告

        参数:
            results: 评估结果字典
            method_name: 方法名称

        返回:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"评估报告: {method_name}")
        lines.append("=" * 80)
        lines.append("")

        # 表头
        lines.append(f"{'k':<8} {'Pass@k':<12} {'Precision':<12} {'Recall':<12} {'MRR':<12}")
        lines.append("-" * 80)

        # 数据行
        for k in sorted(results.keys()):
            r = results[k]
            lines.append(
                f"{k:<8} "
                f"{r['pass_at_k']:>10.2%}     "
                f"{r['precision']:>10.2%}     "
                f"{r['recall']:>10.2%}     "
                f"{r['mrr']:>10.4f}"
            )

        lines.append("=" * 80)
        lines.append("")

        return "\n".join(lines)

    def compare_methods(
        self,
        methods_results: Dict[str, Dict[int, Dict[str, Any]]],
        k_values: List[int] = None
    ) -> str:
        """
        比较多种检索方法的性能

        参数:
            methods_results: 方法结果字典，格式为:
                {
                    "方法名": {k: {指标: 值, ...}, ...},
                    ...
                }
            k_values: 要比较的 k 值列表

        返回:
            格式化的比较报告字符串
        """
        if k_values is None:
            k_values = sorted(next(iter(methods_results.values())).keys())

        lines = []
        lines.append("=" * 100)
        lines.append("检索方法性能对比")
        lines.append("=" * 100)
        lines.append("")

        for k in k_values:
            lines.append(f"Pass@{k} 对比:")
            lines.append("-" * 100)

            # 表头
            header = f"{'方法':<30}"
            for method_name in methods_results.keys():
                pass_at_k = methods_results[method_name][k]["pass_at_k"]
                header += f"{pass_at_k:>10.2%}     "
            lines.append(header)

            # 找出最佳方法
            best_score = 0.0
            best_method = ""
            for method_name, results in methods_results.items():
                score = results[k]["pass_at_k"]
                if score > best_score:
                    best_score = score
                    best_method = method_name

            lines.append(f"最佳: {best_method} ({best_score:.2%})")
            lines.append("")

        lines.append("=" * 100)

        return "\n".join(lines)

    def create_hybrid_retrieval_func(
        self,
        hybrid_engine: HybridSearchEngine
    ) -> Callable:
        """
        创建混合搜索的检索函数，用于评估

        参数:
            hybrid_engine: 混合搜索引擎实例

        返回:
            符合 evaluate() 接口的检索函数

        示例:
            >>> engine = HybridSearchEngine(vector_db, bm25_engine)
            >>> retrieval_func = evaluator.create_hybrid_retrieval_func(engine)
            >>> results = evaluator.evaluate(queries, retrieval_func)
        """
        def retrieval_func(query: str, db, k: int) -> List[Dict[str, Any]]:
            """
            混合搜索检索函数

            注意: db 参数会被忽略，实际使用 hybrid_engine 中的数据库
            """
            return hybrid_engine.search(query, k=k)

        return retrieval_func

    def evaluate_hybrid(
        self,
        queries: List[Dict[str, Any]],
        hybrid_engine: HybridSearchEngine,
        k_values: List[int] = None,
        method_name: str = "混合搜索"
    ) -> Dict[int, Dict[str, Any]]:
        """
        评估混合搜索引擎性能

        这是 evaluate() 的便捷方法，专门用于混合搜索。

        参数:
            queries: 查询列表
            hybrid_engine: 混合搜索引擎实例
            k_values: 要评估的 k 值列表
            method_name: 方法名称（用于报告）

        返回:
            字典，键为 k 值，值为包含评估指标的字典

        示例:
            >>> engine = HybridSearchEngine(vector_db, bm25_engine)
            >>> results = evaluator.evaluate_hybrid(
            ...     queries,
            ...     engine,
            ...     k_values=[5, 10, 20]
            ... )
            >>> print(evaluator.generate_report(results, "混合搜索"))
        """
        # 创建检索函数
        retrieval_func = self.create_hybrid_retrieval_func(hybrid_engine)

        # 执行评估
        results = self.evaluate(
            queries,
            retrieval_func,
            k_values=k_values,
            method_name=method_name
        )

        return results

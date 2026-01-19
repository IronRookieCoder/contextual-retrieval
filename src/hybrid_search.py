"""
混合搜索引擎模块

结合语义搜索和 BM25 搜索，使用 RRF 算法合并结果。
"""

from typing import Any, Dict, List, Tuple

from .config import Config
from .contextual_db import ContextualVectorDB
from .bm25_search import ElasticsearchBM25
from .utils import Logger


class HybridSearchEngine:
    """
    混合搜索引擎

    结合语义搜索和 BM25 搜索，使用 RRF (Reciprocal Rank Fusion) 算法合并结果。

    参数:
        vector_db: 上下文向量数据库实例
        bm25_engine: Elasticsearch BM25 引擎实例
        semantic_weight: 语义搜索权重（默认 0.8）
        bm25_weight: BM25 搜索权重（默认 0.2）
        recall_multiplier: 过检索倍数（默认 8，即检索 k*8 个结果，约 160 个候选）
        config: 配置对象（可选）

    示例:
        >>> engine = HybridSearchEngine(vector_db, bm25_engine)
        >>> results = engine.search("查询文本", k=10)
        >>> print(engine.get_source_analysis())
    """

    def __init__(
        self,
        vector_db: ContextualVectorDB,
        bm25_engine: ElasticsearchBM25,
        semantic_weight: float = None,
        bm25_weight: float = None,
        recall_multiplier: int = 8,
        config: Config = None,
    ):
        self.vector_db = vector_db
        self.bm25_engine = bm25_engine
        self.config = config or Config.from_env()

        # 权重配置
        self.semantic_weight = semantic_weight or self.config.SEMANTIC_WEIGHT
        self.bm25_weight = bm25_weight or self.config.BM25_WEIGHT
        self.recall_multiplier = recall_multiplier

        # 结果来源统计
        self.source_stats = {
            "semantic_only": 0,
            "bm25_only": 0,
            "both": 0,
        }

        # 日志
        self.logger = Logger("HybridSearchEngine")

        # 验证权重
        if abs(self.semantic_weight + self.bm25_weight - 1.0) > 0.01:
            raise ValueError(
                f"权重和必须为 1.0，当前为 "
                f"{self.semantic_weight + self.bm25_weight}"
            )

    def search(
        self,
        query: str,
        k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        执行混合搜索

        参数:
            query: 查询文本
            k: 返回结果数量

        返回:
            结果列表，每个结果包含:
                - chunk: 块元数据
                - score: 融合后的分数
                - from_semantic: 是否来自语义搜索
                - from_bm25: 是否来自 BM25 搜索

        示例:
            >>> results = engine.search("查询文本", k=10)
            >>> for r in results[:3]:
            ...     print(f"分数: {r['score']:.4f}, "
            ...           f"语义: {r['from_semantic']}, BM25: {r['from_bm25']}")
        """
        # 过检索：检索更多候选结果
        recall_k = k * self.recall_multiplier

        # 语义搜索
        semantic_results = self.vector_db.search(query, k=recall_k)

        # BM25 搜索
        bm25_results = self.bm25_engine.search(query, k=recall_k)

        # RRF 融合
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            k
        )

        # 更新统计
        self._update_source_stats(fused_results)

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) 算法

        参数:
            semantic_results: 语义搜索结果
            bm25_results: BM25 搜索结果
            k: 返回结果数量

        返回:
            融合后的结果列表

        算法说明:
            对于每个结果，计算其 RRF 分数:
            score = semantic_weight / (rank_semantic + 1) +
                    bm25_weight / (rank_bm25 + 1)

            然后按分数降序排序，返回 top-k
        """
        # 构建块 ID 到结果的映射
        chunk_scores: Dict[Tuple[str, int], float] = {}
        chunk_data: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # 处理语义搜索结果
        for rank, result in enumerate(semantic_results):
            chunk_id = self._get_chunk_id(result)
            score = self.semantic_weight * (1 / (rank + 1))

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0.0
                chunk_data[chunk_id] = {
                    "chunk": result["metadata"],
                    "from_semantic": False,
                    "from_bm25": False,
                }

            chunk_scores[chunk_id] += score
            chunk_data[chunk_id]["from_semantic"] = True

        # 处理 BM25 搜索结果
        for rank, result in enumerate(bm25_results):
            chunk_id = self._get_chunk_id(result)
            score = self.bm25_weight * (1 / (rank + 1))

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0.0
                chunk_data[chunk_id] = {
                    "chunk": result,
                    "from_semantic": False,
                    "from_bm25": False,
                }

            chunk_scores[chunk_id] += score
            chunk_data[chunk_id]["from_bm25"] = True

        # 按分数排序
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 构建最终结果
        final_results = []
        for (doc_id, original_index), score in sorted_chunks[:k]:
            chunk_info = chunk_data[(doc_id, original_index)]

            # 如果是 BM25 结果，需要从 metadata 中提取内容
            if "metadata" not in chunk_info["chunk"]:
                # BM25 结果格式
                chunk_info["chunk"] = {
                    "doc_id": chunk_info["chunk"]["doc_id"],
                    "original_index": chunk_info["chunk"]["original_index"],
                    "content": chunk_info["chunk"]["content"],
                    "contextualized_content": chunk_info["chunk"].get(
                        "contextualized_content", ""
                    ),
                }

            final_results.append({
                "chunk": chunk_info["chunk"],
                "score": score,
                "from_semantic": chunk_info["from_semantic"],
                "from_bm25": chunk_info["from_bm25"],
            })

        return final_results

    def _get_chunk_id(self, result: Dict[str, Any]) -> Tuple[str, int]:
        """获取块的唯一标识符"""
        if "metadata" in result:
            # 语义搜索结果
            metadata = result["metadata"]
            return (metadata["doc_id"], metadata["original_index"])
        else:
            # BM25 搜索结果
            return (result["doc_id"], result["original_index"])

    def _update_source_stats(self, results: List[Dict[str, Any]]) -> None:
        """更新结果来源统计"""
        for r in results:
            if r["from_semantic"] and r["from_bm25"]:
                self.source_stats["both"] += 1
            elif r["from_semantic"]:
                self.source_stats["semantic_only"] += 1
            elif r["from_bm25"]:
                self.source_stats["bm25_only"] += 1

    def get_source_analysis(self) -> Dict[str, Any]:
        """
        获取结果来源分析

        返回:
            包含来源统计的字典
        """
        total = sum(self.source_stats.values())

        if total == 0:
            return {
                "total": 0,
                "semantic_only": 0,
                "bm25_only": 0,
                "both": 0,
                "semantic_percentage": 0.0,
                "bm25_percentage": 0.0,
            }

        return {
            "total": total,
            "semantic_only": self.source_stats["semantic_only"],
            "bm25_only": self.source_stats["bm25_only"],
            "both": self.source_stats["both"],
            "semantic_percentage": (
                (self.source_stats["semantic_only"] + self.source_stats["both"] * 0.5)
                / total * 100
            ),
            "bm25_percentage": (
                (self.source_stats["bm25_only"] + self.source_stats["both"] * 0.5)
                / total * 100
            ),
        }

    def reset_source_stats(self) -> None:
        """重置来源统计"""
        self.source_stats = {
            "semantic_only": 0,
            "bm25_only": 0,
            "both": 0,
        }

    def set_weights(
        self,
        semantic_weight: float,
        bm25_weight: float
    ) -> None:
        """
        设置搜索权重

        参数:
            semantic_weight: 语义搜索权重
            bm25_weight: BM25 搜索权重
        """
        if abs(semantic_weight + bm25_weight - 1.0) > 0.01:
            raise ValueError(
                f"权重和必须为 1.0，当前为 "
                f"{semantic_weight + bm25_weight}"
            )

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self.logger.logger.info(
            f"权重已更新: 语义={semantic_weight}, BM25={bm25_weight}"
        )

    def set_recall_multiplier(self, multiplier: int) -> None:
        """
        设置过检索倍数

        参数:
            multiplier: 过检索倍数
        """
        if multiplier < 1:
            raise ValueError("过检索倍数必须 >= 1")

        self.recall_multiplier = multiplier

        self.logger.logger.info(f"过检索倍数已更新: {multiplier}")

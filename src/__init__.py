"""
上下文检索系统

一个生产级的检索增强生成(RAG)系统，实现了从基础向量搜索到高级混合检索的完整技术栈。

主要模块:
- config: 配置管理
- vector_db: 基础向量数据库
- contextual_db: 上下文增强向量数据库
- bm25_search: BM25 关键词搜索
- hybrid_search: 混合搜索引擎
- reranking: Cohere 重排序器
- evaluation: 评估系统
- data_generator: 数据生成器
"""

__version__ = "1.0.0"
__author__ = "Contextual Retrieval Team"

from .config import Config
from .vector_db import VectorDB, VectorDBImpl
from .contextual_db import ContextualVectorDB
from .bm25_search import ElasticsearchBM25
from .hybrid_search import HybridSearchEngine
from .reranking import Reranker, CohereReranker
from .evaluation import Evaluator, Metrics
from .data_generator import DataGenerator

__all__ = [
    "Config",
    "VectorDB",
    "VectorDBImpl",
    "ContextualVectorDB",
    "ElasticsearchBM25",
    "HybridSearchEngine",
    "Reranker",
    "CohereReranker",
    "Evaluator",
    "Metrics",
    "DataGenerator",
]

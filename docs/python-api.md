---
name: python-api
description: 上下文检索系统各组件（基础/上下文/混合/重排序/评估）的 Python API 调用示例
type: reference
status: stable
version: 1.0.0
updated: 2026-10-04
---

# Python API 指南

> 本文档承接 README 的 Python API 示例。CLI 用法与核心思想见 [README](../README.md)，架构细节见[系统架构](architecture.md)。

前置条件：已在 `.env` 中配置 `DEEPSEEK_API_KEY` 与 `JINA_API_KEY`，且已生成数据集（CLI `generate-data`）。以下示例依次延续：后文直接复用前文定义的 `config`、`dataset`、`db` 等变量。

## 1. 基础向量搜索（普通 RAG 基线）

```python
from src.config import Config
from src.vector_db import VectorDBImpl
from src.data_generator import DataGenerator

# 加载配置
config = Config.from_env()
config.validate()

# 加载数据
generator = DataGenerator(config)
dataset = generator.load_dataset("data/sample_dataset.json")

# 创建向量数据库
db = VectorDBImpl("my_db", config)
db.load_data(dataset)

# 执行搜索
results = db.search("查询文本", k=10)

for result in results:
    print(f"相似度: {result['similarity']:.4f}")
    print(f"内容: {result['metadata']['content'][:100]}...")
```

## 2. 上下文增强搜索

```python
from src.contextual_db import ContextualVectorDB

# 创建上下文向量数据库（索引期调用 DeepSeek 生成块级上下文）
db = ContextualVectorDB("my_contextual_db", config)
db.load_data(dataset, parallel_threads=5)

# 查看 token 统计（含 DeepSeek 前缀缓存的节省情况）
stats = db.get_token_stats()
print(f"输入 tokens: {stats['input_tokens']:,}")
print(f"输出 tokens: {stats['output_tokens']:,}")

# 执行搜索
results = db.search("查询文本", k=10)
```

## 3. 混合搜索

```python
from src.hybrid_search import HybridSearchEngine
from src.bm25_search import ElasticsearchBM25

# 创建 BM25 索引（需 Elasticsearch）
bm25_engine = ElasticsearchBM25("my_index", config)
bm25_engine.index_documents(db.metadata)

# 创建混合搜索引擎
engine = HybridSearchEngine(
    vector_db=db,
    bm25_engine=bm25_engine,
    semantic_weight=0.8,
    bm25_weight=0.2,
)

# 执行混合搜索
results = engine.search("查询文本", k=10)

# 查看来源分析
analysis = engine.get_source_analysis()
print(f"语义占比: {analysis['semantic_percentage']:.1f}%")
print(f"BM25 占比: {analysis['bm25_percentage']:.1f}%")
```

## 4. 重排序

```python
from src.reranking import JinaReranker

# 创建重排序器
reranker = JinaReranker()

# 过检索 + 重排序
results = reranker.rerank_with_over_retrieval(
    query="查询文本",
    vector_db=db,
    k=10,
    recall_multiplier=10,
)

for result in results:
    print(f"重排序分数: {result['rerank_score']:.4f}")
```

> **向后兼容**：也可使用 `CohereReranker` 别名，指向相同的 `JinaReranker` 实现。

## 5. 性能评估

```python
from src.evaluation import Evaluator

# 创建评估器
evaluator = Evaluator(config)

# 加载查询集
queries = evaluator.load_queries("data/sample_queries.jsonl")

# 定义检索函数
def retrieve_func(query, k):
    return db.search(query, k=k)

# 执行评估
results = evaluator.evaluate(
    queries=queries,
    retrieval_function=retrieve_func,
    k_values=[5, 10, 20],
    method_name="上下文增强",
)

# 生成报告
report = evaluator.generate_report(results, "上下文增强")
print(report)
```

更多可运行示例见 `examples/`：`basic_usage.py`、`contextual_embeddings.py`、`hybrid_search_demo.py`、`evaluation_demo.py`。

---

变更记录：v1.0.0 / 2026-10-04 / 从 README 拆出 Python API 示例，沿用原文内容

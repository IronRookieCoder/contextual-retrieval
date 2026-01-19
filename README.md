# 上下文检索系统

一个生产级的检索增强生成（RAG）系统，实现了从基础向量搜索到高级混合检索的完整技术栈。

## 功能特性

### 核心功能

- **基础向量搜索**: 使用 Voyage AI 生成高质量嵌入，余弦相似度搜索
- **上下文增强**: 使用 Claude 生成块级上下文描述，通过提示缓存优化成本
- **BM25 搜索**: Elasticsearch 集成，支持多字段关键词搜索
- **混合搜索引擎**: 结合语义搜索和 BM25，使用 RRF 算法合并结果
- **Cohere 重排序**: 精细化结果排序，提升检索准确率
- **完整评估系统**: 支持 Pass@k、Precision@k、Recall@k、MRR 等指标

### 技术亮点

- 提示缓存优化：60-80% 成本节省
- 并行处理：多线程加速上下文生成
- LRU 查询缓存：避免重复嵌入计算
- 持久化存储：pickle 格式保存向量数据库
- 灵活配置：环境变量 + .env 文件
- 完善的错误处理：指数退避重试机制

## 性能指标

基于原始 NoteBook 的实验结果：

| 方法 | Pass@5 | Pass@10 | Pass@20 | 成本 |
|------|--------|---------|---------|------|
| 基础 RAG | 80.92% | 87.15% | 90.06% | 低 |
| + 上下文嵌入 | 88.12% | 92.34% | 94.29% | 中（一次性） |
| + 混合搜索 | 88.86% | 93.21% | 95.23% | 中（基础设施） |
| + 重排序 | 92.15% | 95.26% | 97.45% | 高（每次查询） |

## 安装

### 系统要求

- Python 3.8+
- Docker（可选，用于 BM25 搜索）

### 安装步骤

```bash
# 克隆项目
cd D:\contextual-retrieval

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API keys

# （可选）启动 Elasticsearch
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:9.2.0
```

### 环境变量

在 `.env` 文件中配置以下变量：

```bash
# API 密钥（必需）
ANTHROPIC_API_KEY=your_anthropic_api_key
VOYAGE_API_KEY=your_voyage_api_key
COHERE_API_KEY=your_cohere_api_key

# Elasticsearch（可选）
ELASTICSEARCH_URL=http://localhost:9200
```

## 快速开始

### 1. 生成示例数据

```bash
python -m src.cli generate-data \
  --num-docs 10 \
  --chunks-per-doc 5 \
  --num-queries 20
```

### 2. 创建索引

```bash
# 基础向量数据库
python -m src.cli index \
  --method base \
  --name base_db

# 上下文向量数据库
python -m src.cli index \
  --method contextual \
  --name contextual_db \
  --parallel-threads 5
```

### 3. 执行搜索

```bash
# 基础搜索
python -m src.cli search \
  "How to implement authentication?" \
  --name base_db \
  --method base \
  --k 10

# 上下文搜索
python -m src.cli search \
  "How to implement authentication?" \
  --name contextual_db \
  --method contextual \
  --k 10
```

### 4. 混合搜索

```bash
python -m src.cli hybrid-search \
  "How to implement authentication?" \
  --name contextual_db \
  --k 10 \
  --semantic-weight 0.8 \
  --bm25-weight 0.2
```

### 5. 评估性能

```bash
python -m src.cli evaluate \
  --name contextual_db \
  --method contextual \
  --queries data/sample_queries.jsonl \
  --k-values 5 10 20
```

## Python API 使用

### 基础向量搜索

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

### 上下文增强搜索

```python
from src.contextual_db import ContextualVectorDB

# 创建上下文向量数据库
db = ContextualVectorDB("my_contextual_db", config)
db.load_data(dataset, parallel_threads=5)

# 查看 token 统计
stats = db.get_token_stats()
print(f"缓存节省: {stats['cache_savings_percentage']:.2f}%")

# 执行搜索
results = db.search("查询文本", k=10)
```

### 混合搜索

```python
from src.hybrid_search import HybridSearchEngine
from src.bm25_search import ElasticsearchBM25

# 创建 BM25 索引
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

### 重排序

```python
from src.reranking import CohereReranker

# 创建重排序器
reranker = CohereReranker()

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

### 性能评估

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

## 项目结构

```
D:\contextual-retrieval\
├── src/                    # 源代码
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── utils.py           # 工具函数
│   ├── vector_db.py       # 基础向量数据库
│   ├── contextual_db.py   # 上下文向量数据库
│   ├── bm25_search.py     # BM25 搜索
│   ├── hybrid_search.py   # 混合搜索引擎
│   ├── reranking.py       # Cohere 重排序
│   ├── evaluation.py      # 评估系统
│   ├── data_generator.py  # 数据生成器
│   └── cli.py             # 命令行接口
├── tests/                 # 测试套件
├── examples/              # 使用示例
├── data/                  # 数据目录
├── docs/                  # 文档
├── requirements.txt       # Python 依赖
├── .env.example          # 环境变量模板
├── setup.py              # 安装脚本
└── README.md             # 项目文档
```

## 技术细节

### 提示缓存实现

上下文生成使用 Claude 的提示缓存功能，显著降低成本：

```python
response = self.anthropic_client.messages.create(
    model=self.config.ANTHROPIC_MODEL,
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"<document>\n{doc}\n</document>",
                "cache_control": {"type": "ephemeral"}  # 启用缓存
            },
            {
                "type": "text",
                "text": "请为以下块生成上下文描述..."
            }
        ]
    }],
    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)
```

**成本优化**：
- 第一个块：写入文档到缓存（小额溢价）
- 后续块：从缓存读取文档（90% 折扣）
- 预期节省：60-80% 输入 token 成本

### RRF 算法

混合搜索使用 Reciprocal Rank Fusion 算法：

```python
score = semantic_weight * (1 / (rank_semantic + 1)) +
        bm25_weight * (1 / (rank_bm25 + 1))
```

### 并行处理

上下文生成使用多线程并行处理：

```python
with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
    futures = [
        executor.submit(process_chunk, doc, chunk)
        for doc in dataset
        for chunk in doc["chunks"]
    ]

    for future in as_completed(futures):
        result = future.result()
```

## 依赖服务

### API 服务

- **Anthropic Claude**: 用于生成上下文描述
- **Voyage AI**: 用于生成向量嵌入
- **Cohere**: 用于重排序

### 基础设施

- **Elasticsearch**: 可选，用于 BM25 搜索

启动 Elasticsearch：

```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:9.2.0
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_vector_db.py -v

# 测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 常见问题

### Q: 如何选择合适的检索方法？

**A:** 根据需求选择：
- **高并发、成本敏感**: 上下文嵌入（92% Pass@10，无查询成本）
- **平衡生产系统**: 混合搜索（93% Pass@10，无查询成本）
- **追求最高精度**: 完整重排序（95% Pass@10，有查询成本）

### Q: 提示缓存如何工作？

**A:** 文档被写入缓存后，后续块可以 90% 折扣读取。缓存有效期 5 分钟，足够处理同一文档的所有块。

### Q: 如何调整并行线程数？

**A:** 在 CLI 中使用 `--parallel-threads` 参数，或在 Python API 中调用 `load_data(dataset, parallel_threads=5)`。

### Q: Elasticsearch 是必需的吗？

**A:** 不是。BM25 搜索和混合搜索需要 Elasticsearch，但基础向量搜索和上下文搜索不需要。

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 参考资源

- [Anthropic Cookbook - Contextual Retrieval](https://github.com/anthropics/anthropic-cookbook)
- [Voyage AI 文档](https://docs.voyageai.com/)
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank)
- [Elasticsearch BM25](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)

# 上下文检索系统 - 生产级实现计划

## 项目概述

将 `guide.ipynb` 中的上下文检索技术重构为完整的生产级 Python 系统，实现从基础 RAG 到高级检索的完整技术栈。

## 核心功能模块

### 1. 基础向量数据库 (VectorDB)
- 使用 Voyage AI 生成嵌入
- 余弦相似度搜索
- 持久化存储（pickle）
- LRU 查询缓存

### 2. 上下文向量数据库 (ContextualVectorDB)
- 使用 Claude 生成块级上下文描述
- 提示缓存优化（60-80% 成本节省）
- 并行处理（ThreadPoolExecutor）
- Token 使用统计

### 3. BM25 搜索 (ElasticsearchBM25)
- Elasticsearch 集成
- 批量索引
- 多字段搜索（content + contextualized_content）

### 4. 混合搜索引擎 (HybridSearchEngine)
- RRF（Reciprocal Rank Fusion）算法
- 可配置权重（语义 80% + BM25 20%）
- 过检索策略（recall_multiplier * k）

### 5. 重排序器 (CohereReranker)
- Cohere Rerank API 集成
- 过检索 + 精排
- 速率限制处理

### 6. 评估系统 (Evaluator)
- Pass@k 指标计算
- 批量评估和对比
- 性能报告生成

### 7. 工具模块
- 配置管理（环境变量 + .env）
- 日志系统
- 重试机制（指数退避）
- 性能计时器

### 8. 数据生成器
- 生成示例代码库文档
- 模拟评估查询集
- 文档分块工具

## 项目结构

```
D:\contextual-retrieval\
├── src/
│   ├── __init__.py              # 包初始化
│   ├── config.py                # 配置管理（单例模式）
│   ├── utils.py                 # 工具函数（日志、重试、计时器）
│   ├── vector_db.py             # 基础向量数据库
│   ├── contextual_db.py         # 上下文向量数据库
│   ├── bm25_search.py           # Elasticsearch BM25
│   ├── hybrid_search.py         # 混合搜索引擎
│   ├── reranking.py             # Cohere 重排序
│   ├── evaluation.py            # 评估系统
│   ├── data_generator.py        # 示例数据生成器
│   └── cli.py                   # 命令行接口
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # pytest 配置和 fixtures
│   ├── test_vector_db.py
│   ├── test_contextual_db.py
│   ├── test_bm25_search.py
│   ├── test_hybrid_search.py
│   ├── test_reranking.py
│   └── test_evaluation.py
├── examples/
│   ├── basic_usage.py           # 基础使用示例
│   ├── contextual_embeddings.py # 上下文嵌入示例
│   ├── hybrid_search_demo.py    # 混合搜索演示
│   └── evaluation_demo.py       # 评估演示
├── data/
│   └── sample_dataset.json      # 生成的示例数据
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── setup.py                     # 安装脚本
├── README.md                    # 项目文档
└── guide.ipynb                  # 原始 Notebook（参考）
```

## 关键文件清单

### 核心实现文件（按优先级）

1. **D:\contextual-retrieval\src\config.py**
   - Config 类（单例模式）
   - 环境变量加载（python-dotenv）
   - 配置验证（API keys 检查）
   - 数据类定义

2. **D:\contextual-retrieval\src\utils.py**
   - Logger 类（统一日志管理）
   - RetryHandler（指数退避重试）
   - Timer（性能计时器）
   - 自定义异常类层次结构

3. **D:\contextual-retrieval\src\vector_db.py**
   - VectorDB 抽象基类
   - VectorDBImpl 实现类
   - Voyage AI 集成
   - 批量嵌入（batch_size=128）
   - 余弦相似度搜索
   - LRU 查询缓存
   - Pickle 持久化

4. **D:\contextual-retrieval\src\contextual_db.py**
   - ContextualVectorDB 类（继承 VectorDBImpl）
   - Claude 上下文生成（situate_context 方法）
   - 提示缓存实现（cache_control: ephemeral）
   - 并行处理（ThreadPoolExecutor）
   - TokenCounters 数据类（统计节省）
   - 线程安全的计数器（threading.Lock）

5. **D:\contextual-retrieval\src\bm25_search.py**
   - ElasticsearchBM25 类
   - 索引创建和映射配置
   - 批量索引（elasticsearch.helpers.bulk）
   - multi_match 查询
   - 连接处理和错误重试

6. **D:\contextual-retrieval\src\hybrid_search.py**
   - HybridSearchEngine 类
   - RRF 算法实现
   - 权重融合（可配置）
   - 结果来源分析

7. **D:\contextual-retrieval\src\reranking.py**
   - Reranker 抽象基类
   - CohereReranker 实现类
   - 文档准备（content + context）
   - 速率限制（sleep 0.1s）

8. **D:\contextual-retrieval\src\evaluation.py**
   - Evaluator 类
   - Pass@k 计算
   - 批量评估
   - 报告生成
   - Metrics 工具类（Precision@k, Recall@k, MRR）

9. **D:\contextual-retrieval\src\data_generator.py**
   - DataGenerator 类
   - 文档生成（模拟代码库）
   - 分块策略（固定大小 + 重叠）
   - 查询生成
   - 数据集保存

10. **D:\contextual-retrieval\src\cli.py**
    - 命令行接口（argparse）
    - 子命令：index, search, evaluate
    - 进度条显示（tqdm）

### 配置和文档文件

11. **D:\contextual-retrieval\requirements.txt**
    ```
    anthropic>=0.18.0
    voyageai>=0.2.0
    cohere>=5.0.0
    elasticsearch>=8.0.0
    python-dotenv>=1.0.0
    numpy>=1.24.0
    pandas>=2.0.0
    tqdm>=4.65.0
    ```

12. **D:\contextual-retrieval\.env.example**
    ```
    ANTHROPIC_API_KEY=your_anthropic_api_key
    VOYAGE_API_KEY=your_voyage_api_key
    COHERE_API_KEY=your_cohere_api_key
    ELASTICSEARCH_URL=http://localhost:9200
    ```

13. **D:\contextual-retrieval\setup.py**
    - 包元数据
    - 依赖声明
    - entry_points（CLI 命令）

14. **D:\contextual-retrieval\README.md**
    - 项目介绍
    - 安装说明
    - 快速开始
    - API 文档
    - 示例代码

## 实现步骤

### 阶段 1: 基础设施

**任务清单：**
- [ ] 创建目录结构（src/, tests/, examples/, data/）
- [ ] 实现 `src/config.py` - Config 类
- [ ] 实现 `src/utils.py` - Logger, RetryHandler, Timer, 异常类
- [ ] 创建 `requirements.txt`
- [ ] 创建 `.env.example`
- [ ] 创建 `setup.py`

**验证方式：**
```bash
# 测试配置加载
python -c "from src.config import Config; c = Config.from_env(); print(c)"

# 测试日志系统
python -c "from src.utils import Logger; log = Logger('test'); log.logger.info('Test')"
```

### 阶段 2: 向量数据库

**任务清单：**
- [ ] 实现 `src/vector_db.py`
  - [ ] VectorDB 抽象基类
  - [ ] VectorDBImpl 实现类
  - [ ] Voyage AI 集成
  - [ ] 批量嵌入和搜索
  - [ ] 持久化存储
- [ ] 实现 `src/data_generator.py`
  - [ ] 文档生成
  - [ ] 分块工具
  - [ ] 数据集保存
- [ ] 生成测试数据 `data/sample_dataset.json`
- [ ] 创建 `tests/test_vector_db.py`

**验证方式：**
```bash
# 测试向量数据库
python examples/basic_usage.py

# 运行单元测试
pytest tests/test_vector_db.py -v
```

### 阶段 3: 上下文增强

**任务清单：**
- [ ] 实现 `src/contextual_db.py`
  - [ ] ContextualVectorDB 类
  - [ ] situate_context 方法（Claude 调用）
  - [ ] 提示缓存实现
  - [ ] 并行处理（ThreadPoolExecutor）
  - [ ] Token 统计
- [ ] 创建 `tests/test_contextual_db.py`
- [ ] 创建 `examples/contextual_embeddings.py`

**验证方式：**
```bash
# 测试上下文生成
python examples/contextual_embeddings.py

# 检查 token 节省统计
# 预期：60-80% tokens from cache

# 运行测试
pytest tests/test_contextual_db.py -v
```

### 阶段 4: 评估系统

**任务清单：**
- [ ] 实现 `src/evaluation.py`
  - [ ] Evaluator 类
  - [ ] Pass@k 计算
  - [ ] 批量评估
  - [ ] 报告生成
- [ ] 创建 `tests/test_evaluation.py`
- [ ] 创建 `examples/evaluation_demo.py`

**验证方式：**
```bash
# 评估基础向量数据库
python examples/evaluation_demo.py --method base

# 评估上下文向量数据库
python examples/evaluation_demo.py --method contextual

# 预期结果：上下文嵌入 Pass@10 应达到 ~92%
```

### 阶段 5: BM25 搜索

**任务清单：**
- [ ] 实现 `src/bm25_search.py`
  - [ ] ElasticsearchBM25 类
  - [ ] 索引创建和映射
  - [ ] 批量索引
  - [ ] multi_match 查询
- [ ] 创建 `tests/test_bm25_search.py`
- [ ] Docker Elasticsearch 启动脚本（可选）

**验证方式：**
```bash
# 启动 Elasticsearch
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:9.2.0

# 测试 BM25 搜索
pytest tests/test_bm25_search.py -v
```

### 阶段 6: 混合搜索

**任务清单：**
- [ ] 实现 `src/hybrid_search.py`
  - [ ] HybridSearchEngine 类
  - [ ] RRF 算法
  - [ ] 权重融合
  - [ ] 结果分析
- [ ] 创建 `tests/test_hybrid_search.py`
- [ ] 创建 `examples/hybrid_search_demo.py`

**验证方式：**
```bash
# 测试混合搜索
python examples/hybrid_search_demo.py

# 评估混合搜索性能
# 预期：Pass@10 应达到 ~93%
```

### 阶段 7: 重排序

**任务清单：**
- [ ] 实现 `src/reranking.py`
  - [ ] Reranker 抽象基类
  - [ ] CohereReranker 实现类
  - [ ] 文档准备
  - [ ] 速率限制
- [ ] 创建 `tests/test_reranking.py`
- [ ] 更新 `examples/evaluation_demo.py` 添加重排序测试

**验证方式：**
```bash
# 测试重排序
pytest tests/test_reranking.py -v

# 评估重排序性能
# 预期：Pass@10 应达到 ~95%
```

### 阶段 8: CLI 和文档

**任务清单：**
- [ ] 实现 `src/cli.py`
  - [ ] index 命令（创建索引）
  - [ ] search 命令（执行搜索）
  - [ ] evaluate 命令（评估性能）
- [ ] 创建 `README.md`
- [ ] 创建架构文档 `docs/architecture.md`
- [ ] 创建 API 文档 `docs/api.md`
- [ ] 完善所有模块的 docstrings

**验证方式：**
```bash
# 测试 CLI
python -m src.cli index --help
python -m src.cli search "query" --k 10
python -m src.cli evaluate --method contextual

# 检查文档
cat README.md
```

## 关键技术细节

### 1. 提示缓存实现

```python
# 在 situate_context 方法中
response = self.anthropic_client.messages.create(
    model=self.config.ANTHROPIC_MODEL,
    max_tokens=1000,
    temperature=0.0,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"<document>\n{doc}\n</document>",
                "cache_control": {"type": "ephemeral"}  # 关键：启用缓存
            },
            {
                "type": "text",
                "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
            }
        ]
    }],
    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)
```

**成本优化：**
- 第一个块：写入文档到缓存（支付小额溢价）
- 后续块：从缓存读取文档（90% 折扣）
- 预期节省：60-80% 输入 token 成本

### 2. RRF 算法实现

```python
def _reciprocal_rank_fusion(
    self,
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    k: int
) -> List[Dict]:
    """RRF 算法合并两个排序列表"""
    chunk_scores = {}

    # 语义搜索分数（权重 0.8）
    for rank, result in enumerate(semantic_results):
        chunk_id = self._get_chunk_id(result)
        score = self.semantic_weight * (1 / (rank + 1))
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + score

    # BM25 分数（权重 0.2）
    for rank, result in enumerate(bm25_results):
        chunk_id = self._get_chunk_id(result)
        score = self.bm25_weight * (1 / (rank + 1))
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + score

    # 排序并返回 top-k
    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [result for result, _ in sorted_chunks[:k]]
```

### 3. 并行处理和线程安全

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ContextualVectorDB:
    def __init__(self, ...):
        self.token_lock = threading.Lock()
        self.token_counts = TokenCounters()

    def load_data(self, dataset, parallel_threads=5):
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc["chunks"]:
                    futures.append(executor.submit(self._process_chunk, doc, chunk))

            for future in as_completed(futures):
                result = future.result()
                self._store_result(result)

    def _process_chunk(self, doc, chunk):
        contextualized_text, usage = self.situate_context(doc["content"], chunk["content"])

        # 线程安全的计数器更新
        with self.token_lock:
            self.token_counts.input += usage.input_tokens
            self.token_counts.output += usage.output_tokens
            self.token_counts.cache_read += usage.cache_read_input_tokens
            self.token_counts.cache_creation += usage.cache_creation_input_tokens

        return {
            "text_to_embed": f"{chunk['content']}\n\n{contextualized_text}",
            "metadata": {...}
        }
```

### 4. 错误处理策略

```python
class ContextualRetrievalError(Exception):
    """基础异常类"""
    pass

class APIError(ContextualRetrievalError):
    """API 错误"""
    pass

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple = (APIError, TimeoutError)
):
    """指数退避重试装饰器"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                time.sleep(delay)
    return wrapper
```

### 5. 配置管理

```python
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class Config:
    ANTHROPIC_API_KEY: str
    VOYAGE_API_KEY: str
    COHERE_API_KEY: str
    ELASTICSEARCH_URL: str = "http://localhost:9200"

    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    VOYAGE_MODEL: str = "voyage-2"
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"
    EMBEDDING_BATCH_SIZE: int = 128
    MAX_PARALLEL_THREADS: int = 5
    DEFAULT_K: int = 20
    SEMANTIC_WEIGHT: float = 0.8
    BM25_WEIGHT: float = 0.2

    DATA_DIR: Path = Path("./data")
    VECTOR_DB_DIR: Path = Path("./data/vector_dbs")

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
            VOYAGE_API_KEY=os.getenv("VOYAGE_API_KEY"),
            COHERE_API_KEY=os.getenv("COHERE_API_KEY"),
            ELASTICSEARCH_URL=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        )

    def validate(self) -> bool:
        if not self.ANTHROPIC_API_KEY:
            raise ConfigurationError("ANTHROPIC_API_KEY is required")
        if not self.VOYAGE_API_KEY:
            raise ConfigurationError("VOYAGE_API_KEY is required")
        if not self.COHERE_API_KEY:
            raise ConfigurationError("COHERE_API_KEY is required")
        return True
```

## 预期性能指标

基于原始 Notebook 的实验结果：

| 方法 | Pass@5 | Pass@10 | Pass@20 | 成本 |
|------|--------|---------|---------|------|
| 基础 RAG | 80.92% | 87.15% | 90.06% | 低 |
| + 上下文嵌入 | 88.12% | 92.34% | 94.29% | 中（一次性） |
| + 混合搜索 | 88.86% | 93.21% | 95.23% | 中（基础设施） |
| + 重排序 | 92.15% | 95.26% | 97.45% | 高（每次查询） |

**推荐配置：**
- 高并发、成本敏感：上下文嵌入（92% Pass@10）
- 平衡生产系统：混合搜索（93% Pass@10）
- 追求最高精度：完整重排序（95% Pass@10）

## 验证和测试

### 单元测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_vector_db.py -v

# 测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

### 集成测试
```bash
# 端到端测试
pytest tests/test_integration.py -v

# 性能基准测试
pytest tests/test_benchmark.py -v
```

### 手动验证
```bash
# 1. 生成示例数据
python -m src.cli generate-data --num-docs 10 --chunks-per-doc 5

# 2. 创建基础向量数据库索引
python -m src.cli index --method base --name base_db

# 3. 创建上下文向量数据库索引
python -m src.cli index --method contextual --name contextual_db

# 4. 执行搜索
python -m src.cli search "How to implement differential fuzzing?" --k 10

# 5. 评估性能
python -m src.cli evaluate --method contextual --k-values 5 10 20
```

## 依赖和安装

### Python 依赖
- Python 3.8+
- anthropic>=0.18.0
- voyageai>=0.2.0
- cohere>=5.0.0
- elasticsearch>=8.0.0
- python-dotenv>=1.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- tqdm>=4.65.0

### 外部服务
- Anthropic Claude API
- Voyage AI API
- Cohere API
- Elasticsearch（可选，用于 BM25）

### 安装步骤
```bash
# 克隆项目（如果适用）
cd D:\contextual-retrieval

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API keys

# 启动 Elasticsearch（可选）
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:9.2.0

# 运行测试
pytest tests/ -v

# 运行示例
python examples/basic_usage.py
```

## 文件创建顺序

### 第1批：基础设施
1. 创建目录结构
2. `src/__init__.py`
3. `src/config.py`
4. `src/utils.py`
5. `requirements.txt`
6. `.env.example`
7. `setup.py`

### 第2批：向量数据库
8. `src/vector_db.py`
9. `src/data_generator.py`
10. `tests/test_vector_db.py`
11. `examples/basic_usage.py`
12. 生成 `data/sample_dataset.json`

### 第3批：上下文增强
13. `src/contextual_db.py`
14. `tests/test_contextual_db.py`
15. `examples/contextual_embeddings.py`

### 第4批：评估系统
16. `src/evaluation.py`
17. `tests/test_evaluation.py`
18. `examples/evaluation_demo.py`

### 第5批：BM25搜索
19. `src/bm25_search.py`
20. `tests/test_bm25_search.py`

### 第6批：混合搜索
21. `src/hybrid_search.py`
22. `tests/test_hybrid_search.py`
23. `examples/hybrid_search_demo.py`

### 第7批：重排序
24. `src/reranking.py`
25. `tests/test_reranking.py`

### 第8批：CLI和文档
26. `src/cli.py`
27. `README.md`
28. `docs/architecture.md`
29. `docs/api.md`

## 注意事项

1. **API 密钥安全**：
   - 不要提交 .env 文件到版本控制
   - 使用 .env.example 作为模板
   - 在生产环境使用密钥管理服务

2. **成本控制**：
   - 提示缓存可节省 60-80% 成本
   - 批量处理减少 API 调用次数
   - 监控 token 使用情况

3. **性能优化**：
   - 使用查询缓存避免重复嵌入
   - 并行处理加速上下文生成
   - LRU 缓存限制内存使用

4. **错误处理**：
   - 所有 API 调用都应实现重试机制
   - 降级策略（上下文生成失败时使用原始文本）
   - 详细的日志记录便于调试

5. **测试建议**：
   - 使用 mock 对象进行单元测试
   - 小规模数据集进行集成测试
   - 保留原始 Notebook 的结果作为基准

## 总结

本计划将 `guide.ipynb` 中的原型代码重构为完整的生产级系统，包含：
- ✅ 模块化的代码结构
- ✅ 完善的错误处理和日志
- ✅ 灵活的配置管理
- ✅ 全面的单元测试
- ✅ 详细的文档和示例
- ✅ 命令行接口

实现顺序：基础设施 → 向量数据库 → 上下文增强 → 评估 → BM25 → 混合搜索 → 重排序 → CLI → 文档

预期最终性能：Pass@10 达到 95%（完整重排序）

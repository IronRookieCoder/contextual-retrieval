# 上下文检索系统

基于**上下文增强（Contextual Retrieval）**的 RAG 检索系统。核心思想：索引阶段用 LLM 为每个块生成文档级上下文说明，让切块不再脱离原文语境，从根源上解决普通 RAG 的检索失配问题；上下文同时贯穿嵌入、BM25、重排序整条管线。

## 核心思想：上下文增强 vs 普通 RAG

普通 RAG 将文档切块后**孤立嵌入**：块脱离原文，代词失去指代、代码失去所属模块、数字失去归属，嵌入向量无法与查询对齐，检索自然失败。

```
查询: "如何配置 API 密钥？"

普通 RAG 的块:   def __init__(self, api_key):
               → 块内没有"配置""密钥"的文档级语义 → 检索不到

上下文增强后的块: [本块来自 XXXClient 类的构造函数，负责初始化 API 密钥配置…]
               def __init__(self, api_key):
               → 上下文锚定了模块、类、职责 → 命中
```

生成机制：将完整文档与单个块交给 DeepSeek，产出一段简短的"该块在全文中的定位"说明，与原始块拼接后再嵌入：

```
文档全文 + 当前块 ──→ DeepSeek（temperature=0）──→ 上下文说明
                                                        ↓
                    原始块 + 上下文 ──→ Jina Embeddings ──→ 2048 维向量
```

关键差异：上下文不是检索时的补丁，而是**索引期一次性注入、贯穿整条管线**的资产——

| 环节 | 普通 RAG | 本系统（上下文增强） |
|------|----------|---------------------|
| 向量嵌入 | 仅块原文（`vector_db.py`，作为基线） | 块 + 上下文拼接后嵌入（`contextual_db.py`） |
| BM25 检索 | 仅块原文关键词 | `content` + `contextualized_content` 双字段索引（`bm25_search.py`） |
| 重排序 | 仅块原文 | 块原文 + 上下文拼接后送入重排序器（`reranking.py`） |
| 查询时 LLM 成本 | — | 零（上下文在索引期生成） |

成本控制：同一文档的所有块共享同一文档前缀，自动命中 DeepSeek 前缀缓存，token 消耗与节省可通过 `get_token_stats()` 查看；上下文生成支持多线程并行（`--parallel-threads`，默认 5）。

## 检索管线与性能

四层递进，第 1 层（上下文增强）是收益最大的一层，后三层可按精度与成本取舍：

1. **上下文嵌入**：块 + 上下文 → Jina Embeddings → 余弦相似度搜索
2. **上下文 BM25**：Elasticsearch 双字段关键词搜索
3. **混合搜索**：语义 + BM25 按 RRF（Reciprocal Rank Fusion，默认权重 0.8:0.2）融合，过检索 k×8 候选
4. **重排序**：Jina Reranker 对候选精排

> 架构图、端到端流程与算法细节（RRF、过检索、任务感知嵌入）见 [系统架构](docs/architecture.md)。

| 方法 | Pass@5 | Pass@10 | Pass@20 | 成本 |
|------|--------|---------|---------|------|
| 基础 RAG | 80.92% | 87.15% | 90.06% | 低 |
| + 上下文嵌入 | 88.12% | 92.34% | 94.29% | 中（索引期一次性） |
| + 混合搜索 | 88.86% | 93.21% | 95.23% | 中（需 ES，查询免费） |
| + 重排序 | 92.15% | 95.26% | 97.45% | 高（每次查询） |

> 数据来自原始 Anthropic Cookbook 实验；本项目将原版 Claude + Voyage + Cohere 替换为 DeepSeek + Jina，保持相同检索精度的同时大幅降低成本。选型：成本敏感选上下文嵌入；平衡选混合搜索；最高精度再叠加重排序。

## 快速开始

Python 3.8+；混合搜索需 Docker 运行 Elasticsearch。

```bash
pip install -r requirements.txt

# 创建 .env（必需）
#   DEEPSEEK_API_KEY=xxx   # 上下文生成，platform.deepseek.com
#   JINA_API_KEY=xxx       # 嵌入 + 重排序，jina.ai（免费额度 100 万 tokens/天）
#   ELASTICSEARCH_URL=http://localhost:9200   # 可选，混合搜索
# 可选模型覆盖：DEEPSEEK_MODEL / DEEPSEEK_BASE_URL / JINA_EMBEDDING_MODEL / JINA_RERANKER_MODEL
# 注：ANTHROPIC_API_KEY / VOYAGE_API_KEY / COHERE_API_KEY 仅作向后兼容别名

# （可选）启动 Elasticsearch
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  elasticsearch:9.2.0

# 1. 生成示例数据
python -m src.cli generate-data --num-docs 10 --chunks-per-doc 5 --num-queries 20

# 2. 创建上下文增强索引（索引期调用 LLM；--method base 为普通 RAG 基线，可对比复现上表差异）
python -m src.cli index --method contextual --name ctx_db

# 3. 搜索 / 混合搜索 / 评估
python -m src.cli search "How to implement authentication?" --name ctx_db --k 10
python -m src.cli hybrid-search "How to implement authentication?" --name ctx_db
python -m src.cli evaluate --name ctx_db --k-values 5 10 20

# 从真实文档一键评估（加载 → 分块 → 生成查询 → 建索引 → 评测，--hybrid 附带 BM25 对比）
python -m src.cli evaluate-real --data-dir "./my_docs" --name my_eval --queries-per-doc 3

# Web 检索控制台（FastAPI，浏览器打开 http://127.0.0.1:8000，覆盖数据准备/索引/检索/评估全流程）
python -m src.web.app
```

测试：`pytest tests/ -v`。Python API 用法见 [Python API 指南](docs/python-api.md) 与 `examples/`。

## 项目结构

```
src/
├── vector_db.py         # 基础向量库（普通 RAG 基线）
├── contextual_db.py     # 上下文增强向量库（核心：situate_context 生成块级上下文）
├── bm25_search.py       # Elasticsearch 上下文 BM25
├── hybrid_search.py     # RRF 混合搜索
├── reranking.py         # Jina 重排序
├── evaluation.py        # Pass@k / Precision@k / Recall@k / MRR 评估
├── data_generator.py    # 示例数据生成
├── real_data_loader.py  # 真实文档加载（分块、查询生成）
├── web/                 # FastAPI Web 控制台
├── config.py            # 配置管理
├── utils.py             # 日志、重试等工具
└── cli.py               # 命令行入口
examples/                # 可运行示例
tests/                   # 测试
docs/                    # 延伸文档（架构、Python API、FAQ）与评估报告
```

## 实现说明

- **自研内存向量库**（[存储机制详情](docs/architecture.md#3-向量存储机制)）：未使用 Pinecone/Chroma 等专用向量库，采用 pickle 持久化 + 全量余弦检索，定位是小规模对比评估（数千块量级），无 ANN/分片能力
- **任务感知嵌入**：建库用 `retrieval.passage`、查询用 `retrieval.query`，非对称嵌入提升检索质量
- **LLM 可替换**：上下文生成走 OpenAI 兼容协议，改 `DEEPSEEK_BASE_URL` 即可切换到其他兼容服务（如 Ollama）
- **Elasticsearch 可选**：仅混合搜索需要，基础/上下文向量搜索不依赖

## 延伸文档

- [系统架构](docs/architecture.md) — 模块依赖图、端到端流程、向量存储机制与核心算法
- [Python API 指南](docs/python-api.md) — 基础/上下文/混合/重排序/评估的编程调用示例
- [常见问题](docs/faq.md) — 方法选型、成本额度、并行参数、LLM 切换

## 参考

- [Anthropic — Contextual Retrieval](https://github.com/anthropics/anthropic-cookbook)
- [DeepSeek API](https://api-docs.deepseek.com) / [Jina AI](https://jina.ai)

MIT License

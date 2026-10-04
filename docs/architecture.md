---
name: architecture
description: 上下文检索系统的模块依赖、端到端业务流程、向量存储机制、RAG 演进路线与核心算法细节
type: reference
status: stable
version: 1.0.0
updated: 2026-10-04
---

# 系统架构

> 本文档承接 README 的架构细节：模块依赖图、端到端流程、向量存储机制、演进路线与核心算法。技术定位与快速开始见 [README](../README.md)。

## 1. 系统结构图

模块间的静态依赖关系：

```mermaid
graph TB
    subgraph 入口层["🎯 入口层"]
        CLI["CLI 命令行<br/>src/cli.py"]
        Web["Web 控制台<br/>src/web/app.py<br/>FastAPI + Jinja2"]
        API["Python API<br/>直接调用模块"]
    end

    subgraph 业务层["⚙️ 业务逻辑层"]
        direction TB

        DG["数据生成器<br/>data_generator.py<br/>示例文档 / 查询生成"]
        RDL["真实文档加载器<br/>real_data_loader.py<br/>文件系统 → 分块 → 查询"]

        VDB["基础向量数据库<br/>vector_db.py<br/>Jina Embeddings v4<br/>余弦相似度 / LRU 缓存"]
        CVDB["上下文向量数据库<br/>contextual_db.py<br/>DeepSeek 生成上下文<br/>多线程并行处理"]
        BM25["BM25 搜索引擎<br/>bm25_search.py<br/>Elasticsearch 集成<br/>多字段关键词匹配"]

        HS["混合搜索引擎<br/>hybrid_search.py<br/>RRF 算法融合<br/>语义 + BM25"]
        RR["重排序器<br/>reranking.py<br/>Jina Reranker v3<br/>过检索 + 精排"]

        EVAL["评估系统<br/>evaluation.py<br/>Pass@k / Precision<br/>Recall / MRR"]
    end

    subgraph 服务层["☁️ 外部服务"]
        DeepSeek[("DeepSeek API<br/>上下文生成")]
        JinaEmb[("Jina AI Embeddings<br/>2048 维向量")]
        JinaRerank[("Jina AI Reranker<br/>精排")]
        ES[("Elasticsearch<br/>BM25 关键词检索<br/>Docker 可选部署")]
    end

    subgraph 存储层["💾 存储层"]
        Pickle[("Pickle 文件<br/>data/vector_dbs/")]
        JSONL[("JSON/JSONL<br/>数据集 & 查询集")]
    end

    subgraph 配置层["🔧 配置层"]
        Config["Config<br/>config.py<br/>.env 环境变量"]
    end

    CLI --> DG
    CLI --> RDL
    CLI --> VDB
    CLI --> CVDB
    CLI --> HS
    Web --> DG
    Web --> RDL
    Web --> VDB
    Web --> CVDB
    Web --> HS
    Web --> RR
    Web --> EVAL
    API --> VDB
    API --> CVDB
    API --> HS
    API --> RR
    API --> EVAL

    VDB --> JinaEmb
    CVDB --> DeepSeek
    CVDB --> JinaEmb
    BM25 --> ES
    HS --> CVDB
    HS --> BM25
    RR --> JinaRerank
    EVAL --> VDB
    EVAL --> CVDB
    EVAL --> HS

    VDB --> Pickle
    CVDB --> Pickle
    EVAL --> JSONL
    DG --> JSONL
    RDL --> JSONL
    Config --> VDB
    Config --> CVDB
    Config --> BM25
    Config --> DeepSeek
    Config --> JinaEmb
    Config --> JinaRerank
    Config --> ES
```

## 2. 端到端业务流程

从数据准备到效果评估的完整流程：

```mermaid
flowchart LR
    subgraph 阶段1["📥 数据准备"]
        A1["加载文档<br/>文件系统 / 生成示例"] --> A2["文档分块<br/>chunk_size 可配置"]
        A2 --> A3["查询生成<br/>每个文档 1~N 条查询"]
    end

    subgraph 阶段2["🔨 索引构建"]
        direction TB
        B0{"选择索引方式"}
        B0 -->|基础 RAG| B1["Base 索引<br/>Jina Embeddings<br/>retrieval.passage 模式"]
        B0 -->|上下文增强| B2["Contextual 索引<br/>DeepSeek 生成块级上下文<br/>多线程并行加速"]
        B2 --> B3["上下文 + 原始块拼接<br/>送入 Jina Embeddings"]
        B0 -->|混合搜索| B4["BM25 索引<br/>Elasticsearch 倒排索引<br/>original + contextual 字段"]
        B1 --> B5["持久化<br/>Pickle 保存向量"]
        B3 --> B5
        B4 --> B6["ES 索引存储"]
    end

    subgraph 阶段3["🔍 检索查询"]
        direction TB
        C0{"选择检索方式"}
        C0 -->|基础 RAG| C1["向量检索<br/>余弦相似度<br/>LRU 查询缓存"]
        C0 -->|上下文增强| C2["上下文向量检索<br/>语义匹配增强"]
        C0 -->|混合搜索| C3["语义搜索<br/>over-retrieval<br/>k × 8 候选"]
        C3 --> C4["BM25 搜索<br/>关键词匹配<br/>k × 8 候选"]
        C4 --> C5["RRF 融合<br/>semantic_weight × 1/(r+1)<br/>+ bm25_weight × 1/(r+1)"]
        C5 --> C6["Top-K 结果"]
        C1 --> C7["候选结果集"]
        C2 --> C7
        C6 --> C7

        C7 --> C8{"启用重排序?"}
        C8 -->|是| C9["Jina Reranker<br/>精细重排序"]
        C8 -->|否| C10["返回最终结果"]
        C9 --> C10
    end

    subgraph 阶段4["📊 效果评估"]
        D1["加载查询集<br/>JSONL 格式"] --> D2["执行检索<br/>对每个查询执行搜索"]
        D2 --> D3["计算指标<br/>Pass@k / Precision@k<br/>Recall@k / MRR"]
        D3 --> D4["生成对比报告<br/>多方法横向对比"]
    end

    阶段1 --> 阶段2
    阶段2 --> 阶段3
    阶段3 --> 阶段4
```

## 3. 向量存储机制

**本项目未使用传统专用向量数据库（如 Pinecone、Weaviate、Chroma、Milvus 等）。**

向量存储和检索完全自主实现于 `src/vector_db.py`，采用 **Pickle 序列化 + 全量内存暴力检索**方案：

| 组件 | 实现方式 |
|------|----------|
| **存储格式** | `data/vector_dbs/<name>/vector_db.pkl` |
| **向量数据** | `List[np.ndarray]` — numpy 2048 维浮点数组列表 |
| **元数据** | `List[Dict]` — 文档 ID、块 ID、原文、上下文 |
| **缓存** | LRU 字典（上限 1000 条），避免重复嵌入计算 |
| **检索算法** | 全量余弦相似度：`np.dot(embeddings, query_embedding)` → `np.argsort` |
| **数据持久化** | pickle.dump → pickle.load |

> 适用场景：**小规模 Demo / 学术评估**（~数千个块）。无 ANN 索引（HNSW/IVF）、无分片、无分布式能力。选择此方案因为项目核心目标是**对比不同检索策略的精度差异**（base vs contextual vs hybrid vs rerank），而非构建高性能生产级检索系统。

## 4. RAG 演进路线

```
基础 RAG          标准嵌入 + 余弦相似度
    ↓
上下文增强 RAG    块嵌入前注入 DeepSeek 全局语境
    ↓
混合搜索 RAG      语义 + BM25 关键词互补 (RRF 融合)
    ↓
重排序 RAG        过检索后 Jina Reranker 二次精排
```

每个阶段在精度和成本之间取不同权衡点：

| 阶段 | 精度提升 | 增量成本 | 推荐场景 |
|------|----------|----------|----------|
| 基础 RAG | 基准线 | 无 | 快速验证 |
| + 上下文增强 | +7–9% Pass@k | DeepSeek 一次性生成 | 低成本提精 |
| + 混合搜索 | +1–2% Pass@k | Elasticsearch 基础设施 | 平衡生产系统 |
| + 重排序 | +3–4% Pass@k | 每次查询 Jina 费用 | 最高精度需求 |

## 5. 核心算法与实现细节

### 5.1 上下文生成

使用 DeepSeek API（OpenAI 兼容协议），将整篇文档和当前块拼接后发送给模型，由模型生成简洁的块级上下文描述：

```python
response = self.llm_client.chat.completions.create(
    model=self.config.DEEPSEEK_MODEL,     # 默认 deepseek 系列
    max_tokens=1000,
    temperature=0.0,
    messages=[{"role": "user", "content": prompt}],
)
```

### 5.2 RRF 混合与过检索

混合搜索使用 RRF（Reciprocal Rank Fusion）算法，融合语义搜索（余弦相似度）和 BM25 关键词搜索的结果：

```python
score = semantic_weight * (1 / (rank_semantic + 1)) +
        bm25_weight * (1 / (rank_bm25 + 1))
```

默认权重 0.8:0.2，语义搜索主导、BM25 补充精确关键词匹配。同时使用**过检索策略**（检索 k×8 个候选），确保融合后的候选池覆盖足够全面的结果。

### 5.3 Jina 任务感知嵌入

Jina Embeddings 支持任务感知嵌入，系统自动区分：

- `retrieval.passage` — 文档/块嵌入（建库时）
- `retrieval.query` — 查询嵌入（搜索时）

这种非对称嵌入策略显著提升检索质量。

### 5.4 并行处理

上下文生成使用多线程并行处理，可配置线程数以平衡速度与 API 速率限制：

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

---

变更记录：v1.0.0 / 2026-10-04 / 从 README 拆出架构细节（结构图、流程图、存储机制、算法），沿用原文内容

---
name: faq
description: 上下文检索系统常见问题：方法选型、成本额度、并行参数、Elasticsearch 依赖与 LLM 切换
type: reference
status: stable
version: 1.0.0
updated: 2026-10-04
---

# 常见问题

> 本文档承接 README 的 FAQ。技术定位与快速开始见 [README](../README.md)。

### Q: 如何选择合适的检索方法？

**A:** 根据需求选择：
- **高并发、成本敏感**：上下文嵌入 — DeepSeek 一次性生成上下文，无查询时 LLM 成本
- **平衡生产系统**：混合搜索 — 语义 + BM25 互补，无查询时额外 API 成本
- **追求最高精度**：完整重排序 — 过检索 + Jina Reranker 精排

### Q: Jina AI 免费额度够用吗？

**A:** Jina AI 提供 100 万 tokens/天的免费额度。对于嵌入和重排序，这足以支撑中小规模生产使用。超出免费额度后才按量计费。

### Q: 如何调整并行线程数？

**A:** 在 CLI 中使用 `--parallel-threads` 参数，或在 Python API 中调用 `load_data(dataset, parallel_threads=5)`。

### Q: Elasticsearch 是必需的吗？

**A:** 不是。BM25 搜索和混合搜索需要 Elasticsearch，但基础向量搜索和上下文搜索不需要。

### Q: 支持切换到其他 LLM 吗？

**A:** 支持。上下文生成使用 OpenAI 兼容协议，将 `DEEPSEEK_BASE_URL` 改为其他兼容服务的地址即可（如 Ollama `http://localhost:11434/v1`），也可通过 `DEEPSEEK_MODEL` 切换模型。

### Q: 与原始 Anthropic Cookbook 版本有何区别？

**A:** 本项目将原版的 Anthropic Claude + Voyage AI + Cohere 替换为 DeepSeek + Jina AI，显著降低成本（DeepSeek 输入价格极低，Jina AI 有免费额度），同时保持了相同的检索精度。代码保留了 `CohereReranker` 别名等向后兼容接口。

---

变更记录：v1.0.0 / 2026-10-04 / 从 README 拆出 FAQ，沿用原文内容

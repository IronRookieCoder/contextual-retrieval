from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    level: str
    text: str


@dataclass
class ConfigStatus:
    deepseek_configured: bool
    jina_configured: bool
    elasticsearch_url: str
    elasticsearch_available: bool
    deepseek_model: str
    deepseek_base_url: str
    jina_embedding_model: str
    jina_reranker_model: str
    data_dir: Path
    vector_db_dir: Path
    warnings: List[str] = field(default_factory=list)


@dataclass
class IndexSummary:
    name: str
    method: str
    num_embeddings: int
    embedding_dim: int
    cache_size: int
    db_path: str
    loaded_from_disk: bool
    token_stats: Optional[Dict[str, Any]] = None


@dataclass
class SearchResultView:
    rank: int
    doc_id: str
    chunk_id: str
    content: str
    contextualized_content: str = ""
    similarity: Optional[float] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    from_semantic: bool = False
    from_bm25: bool = False


@dataclass
class EvaluationTable:
    method_name: str
    rows: List[Dict[str, Any]]
    report: str


@dataclass
class DashboardState:
    messages: List[Message] = field(default_factory=list)
    config_status: Optional[ConfigStatus] = None
    index_summary: Optional[IndexSummary] = None
    search_results: List[SearchResultView] = field(default_factory=list)
    evaluation: Optional[EvaluationTable] = None

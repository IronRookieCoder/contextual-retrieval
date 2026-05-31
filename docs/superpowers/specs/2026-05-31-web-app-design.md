# Contextual Retrieval Web Console Design

## Purpose

Add a companion Web application for local developer debugging and demonstrations. The Web app covers the existing contextual retrieval workflow end to end: configuration checks, data preparation, index creation/loading, search, optional reranking, and evaluation.

This first version is a local single-process console. It does not add accounts, browser uploads, background job recovery, a new database, or production deployment features.

## Chosen Approach

Use a lightweight FastAPI application with Jinja templates, HTMX-style partial updates where useful, and small amounts of plain JavaScript for tabs and UI state.

This approach was chosen because it keeps the stack close to the current Python project, reuses the existing library code directly, and avoids adding a Node frontend build system for the first local demo version.

Alternatives considered:

- FastAPI JSON API plus a standalone JavaScript frontend. This would make future asynchronous task handling easier, but adds more client-side state than the first version needs.
- Web wrapper around CLI subprocesses. This would minimize Python service code, but makes structured errors, search result rendering, and evaluation tables harder to maintain.

## Scope

Included in version one:

- Single-page developer dashboard.
- Read-only `.env` configuration status.
- Local dataset path and query file path inputs.
- Sample data generation through `DataGenerator`.
- Real document directory processing through `DocumentLoader`.
- Base and contextual index creation/loading.
- Base, contextual, and hybrid search.
- Optional Jina reranking from the search page.
- Evaluation with Pass@k, Precision@k, Recall@k, and MRR.
- Clear same-page error reporting.
- Tests for service-layer validation, result formatting, and page rendering.

Excluded from version one:

- Browser file uploads.
- Editing or saving `.env` from the browser.
- User accounts, authentication, authorization, or multi-tenant isolation.
- Persistent server-side job history.
- Background task queues, task cancellation, or progress recovery after refresh.
- Production deployment packaging.

## Architecture

Add a new `src/web/` package:

- `src/web/app.py`: FastAPI application factory, route registration, and local development entry point.
- `src/web/services.py`: Web-facing service layer for configuration checks, data preparation, index operations, search, reranking, and evaluation.
- `src/web/schemas.py`: Lightweight dataclasses or Pydantic models for form inputs and normalized view results.
- `src/web/templates/`: Jinja templates for the single-page dashboard and partial result fragments.
- `src/web/static/`: CSS and small JavaScript helpers.

The Web layer calls the existing project modules directly:

- `Config`
- `DataGenerator`
- `DocumentLoader`
- `VectorDBImpl`
- `ContextualVectorDB`
- `ElasticsearchBM25`
- `HybridSearchEngine`
- `JinaReranker`
- `Evaluator`

No existing CLI behavior or vector database persistence format changes. Existing vector databases continue to use `data/vector_dbs/<name>/...pkl`.

## Run Command

Add a local development entry point:

```bash
python -m src.web.app
```

The app should bind to localhost by default and print the local URL.

## Page Structure

The app is a single-page dashboard with a left flow navigation and a dense work area.

Sections:

- Configuration
- Data
- Index
- Search
- Evaluation

The UI uses a developer-console aesthetic: compact forms, high-density status cards, clear result tables, and a strong visual distinction between configuration, actions, and outputs.

The approved visual direction is an industrial retrieval console:

- Hard-edged panels.
- Strong borders.
- Status indicators.
- Engineering-grid feel.
- Dense but scannable result areas.
- No marketing landing page.
- No oversized hero content.

## Configuration Panel

The configuration panel reads from `Config.from_env()` and shows:

- Whether `DEEPSEEK_API_KEY` is configured.
- Whether `JINA_API_KEY` is configured.
- DeepSeek base URL and model.
- Jina embedding and reranker models.
- Elasticsearch URL and availability status.
- Data directory.
- Vector DB directory.

API keys are never displayed or editable. They are shown only as configured or missing.

Configuration status gates actions:

- Missing `JINA_API_KEY` disables vector indexing, vector search, hybrid search, reranking, and evaluation paths that require embeddings.
- Missing `DEEPSEEK_API_KEY` disables contextual index creation and real-document query generation paths that require LLM calls.
- Elasticsearch unavailable disables hybrid search and hybrid evaluation, but does not block base or contextual vector search.

## Data Panel

Supported data flows:

- Use existing dataset JSON and queries JSONL paths.
- Generate sample data using `DataGenerator`.
- Process a local real-document directory using `DocumentLoader`.

The first version does not support browser uploads. Users provide local filesystem paths.

When real documents are processed, generated dataset and query files are saved under the existing `data/` directory using a user-provided or generated run name.

## Index Panel

Inputs:

- Index name.
- Method: `base` or `contextual`.
- Dataset JSON path.
- Parallel thread count for contextual indexing.

Behavior:

- Base indexing uses `VectorDBImpl.load_data(dataset)`.
- Contextual indexing uses `ContextualVectorDB.load_data(dataset, parallel_threads=n)`.
- If the named index already exists, the existing implementation loads it from disk.
- The page reports whether an index was loaded from disk or created during the request when that can be inferred.

Displayed index details:

- Index name.
- Method.
- Embedding count.
- Embedding dimension.
- Query cache size.
- DB path.
- Contextual token statistics when available.

## Search Panel

Inputs:

- Query text.
- Index name.
- Search method: base, contextual, or hybrid.
- `k`.
- Hybrid semantic and BM25 weights.
- Optional reranking toggle.
- Rerank recall multiplier.

Behavior:

- Base search loads `VectorDBImpl`.
- Contextual search loads `ContextualVectorDB`.
- Hybrid search loads `ContextualVectorDB`, creates a temporary `ElasticsearchBM25` index from vector metadata, runs `HybridSearchEngine`, and cleans up the temporary BM25 index after the request.
- Reranking is an explicit user choice and calls `JinaReranker` only when enabled.

Search results are normalized for display:

- Rank.
- Document ID.
- Chunk ID.
- Source method indicators.
- Content excerpt.
- Contextualized content when present.
- Similarity score when present.
- Hybrid fusion score when present.
- Rerank score when present.

## Evaluation Panel

Inputs:

- Index name.
- Method: base, contextual, or hybrid.
- Queries JSONL path.
- k values.
- Hybrid weights when relevant.

Behavior:

- Loads queries through `Evaluator.load_queries`.
- Builds a retrieval function based on selected method.
- Calls `Evaluator.evaluate()` or `Evaluator.evaluate_hybrid()`.
- Renders metrics as a structured table.
- Also exposes the textual report for copying.

Displayed metrics:

- Pass@k.
- Precision@k.
- Recall@k.
- MRR.
- Valid query count.

## Error Handling

Routes catch expected business and validation errors and render same-page error panels. FastAPI tracebacks should not be shown to users during normal validation failures.

Errors should be explicit and actionable:

- Missing config names the missing key but never prints secret values.
- Missing dataset or queries file includes the path.
- Invalid index name explains allowed characters.
- Missing index explains where the Web app looked.
- Elasticsearch connection errors explain that hybrid search is optional.
- API failures display provider, status summary, and a shortened message.

Service-layer error formatting must perform basic secret redaction for known environment variable values before rendering messages.

Long-running requests are synchronous in version one. The UI should show a submitted/waiting state and then render final results. Real-time progress, cancellation, and resumable jobs are out of scope.

## Validation Rules

Index names should be restricted to a filesystem-safe subset:

- Letters.
- Digits.
- Underscore.
- Hyphen.

Paths are local filesystem paths. The first version does not sandbox paths beyond normal file existence/type validation.

Hybrid weights must be positive and sum to 1.0 within a small tolerance.

k values must be positive integers.

Parallel thread count must be a positive integer.

## Testing

Add service-layer tests for:

- Configuration status with present and missing keys.
- Secret redaction.
- Dataset path validation.
- Query path validation.
- Index name validation.
- Hybrid weight validation.
- Search result normalization for base, contextual, hybrid, and reranked results.
- Evaluation metric table formatting.

Add FastAPI tests with `TestClient` for:

- Dashboard renders.
- Missing configuration displays non-secret warnings.
- Invalid form input returns the dashboard with an error panel.
- Search route can render mocked results.
- Evaluation route can render mocked metric results.

External provider calls should be monkeypatched in tests. Default tests must not require DeepSeek, Jina, or Elasticsearch network access.

Existing CLI tests and behavior should remain unchanged.

## Implementation Notes

Prefer keeping route handlers thin. Request parsing, validation, existing-class orchestration, and result normalization belong in `src/web/services.py`.

Templates should be split into small partials only where it improves clarity:

- `_config_panel.html`
- `_data_panel.html`
- `_index_panel.html`
- `_search_panel.html`
- `_evaluation_panel.html`
- `_messages.html`

The app should use plain CSS in `src/web/static/app.css`. No frontend build step is introduced in version one.

## Acceptance Criteria

- `python -m src.web.app` starts the local Web console.
- The dashboard renders without configured API keys and shows missing-key warnings without crashing.
- With valid `.env`, a developer can generate or select data, build or load an index, search it, optionally rerank results, and run evaluation from the browser.
- Existing vector database files remain compatible with CLI usage.
- Existing CLI commands still work.
- Tests pass without external API credentials by using mocks.

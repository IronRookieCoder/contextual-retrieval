from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import Config
from src.web.schemas import Message
from src.web.services import (
    WebServiceError,
    build_index,
    get_config_status,
    parse_k_values,
    prepare_sample_data,
    process_real_directory,
    run_evaluation,
    run_search,
    validate_positive_int,
)


WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="Contextual Retrieval Web Console")
    app.mount(
        "/static",
        StaticFiles(directory=str(WEB_DIR / "static")),
        name="static",
    )

    def render_dashboard(request: Request, messages=None, config=None, **context):
        config = config or Config.from_env()
        base_context = {
            "request": request,
            "messages": messages or [],
            "config_status": get_config_status(config),
            "data_result": None,
            "index_summary": None,
            "search_results": [],
            "evaluation": None,
        }
        base_context.update(context)
        return TEMPLATES.TemplateResponse(request, "dashboard.html", base_context)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return render_dashboard(request)

    @app.post("/data", response_class=HTMLResponse)
    async def prepare_data(
        request: Request,
        mode: str = Form("sample"),
        run_name: str = Form("real_eval"),
        data_dir: str = Form(""),
        num_docs: str = Form("10"),
        chunks_per_doc: str = Form("5"),
        num_queries: str = Form("20"),
        queries_per_doc: str = Form("3"),
    ):
        config = Config.from_env()
        try:
            if mode == "sample":
                result = prepare_sample_data(
                    config,
                    validate_positive_int(num_docs, "文档数"),
                    validate_positive_int(chunks_per_doc, "每文档块数"),
                    validate_positive_int(num_queries, "查询数"),
                )
            elif mode == "real":
                result = process_real_directory(
                    config,
                    data_dir,
                    run_name,
                    validate_positive_int(queries_per_doc, "每文档查询数"),
                )
            else:
                raise WebServiceError("数据模式必须是 sample 或 real。")
            return render_dashboard(
                request,
                config=config,
                messages=[Message("success", "数据准备完成。")],
                data_result=result,
            )
        except WebServiceError as exc:
            return render_dashboard(
                request,
                config=config,
                messages=[Message("error", str(exc))],
            )

    @app.post("/index", response_class=HTMLResponse)
    async def create_index(
        request: Request,
        name: str = Form(...),
        method: str = Form("contextual"),
        dataset_path: str = Form("data/sample_dataset.json"),
        parallel_threads: str = Form("5"),
    ):
        config = Config.from_env()
        try:
            summary = build_index(
                config=config,
                name=name,
                method=method,
                dataset_path=dataset_path,
                parallel_threads=validate_positive_int(parallel_threads, "并行线程数"),
            )
            message = "已加载现有索引。" if summary.loaded_from_disk else "索引创建完成。"
            return render_dashboard(
                request,
                messages=[Message("success", message)],
                index_summary=summary,
                config=config,
            )
        except WebServiceError as exc:
            return render_dashboard(request, messages=[Message("error", str(exc))], config=config)

    @app.post("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        query: str = Form(...),
        index_name: str = Form("demo_contextual"),
        method: str = Form("contextual"),
        k: str = Form("10"),
        semantic_weight: float = Form(0.8),
        bm25_weight: float = Form(0.2),
        rerank: str = Form("off"),
        recall_multiplier: str = Form("10"),
    ):
        config = Config.from_env()
        try:
            results = run_search(
                config=config,
                query=query,
                index_name=index_name,
                method=method,
                k=validate_positive_int(k, "返回数量"),
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                rerank=rerank == "on",
                recall_multiplier=validate_positive_int(recall_multiplier, "重排召回倍数"),
            )
            return render_dashboard(
                request,
                messages=[Message("success", f"找到 {len(results)} 条结果。")],
                search_results=results,
                config=config,
            )
        except WebServiceError as exc:
            return render_dashboard(request, messages=[Message("error", str(exc))], config=config)

    @app.post("/evaluation", response_class=HTMLResponse)
    async def evaluation(
        request: Request,
        index_name: str = Form("demo_contextual"),
        method: str = Form("contextual"),
        queries_path: str = Form("data/sample_queries.jsonl"),
        k_values: str = Form("5 10 20"),
        semantic_weight: float = Form(0.8),
        bm25_weight: float = Form(0.2),
    ):
        config = Config.from_env()
        try:
            table = run_evaluation(
                config=config,
                index_name=index_name,
                method=method,
                queries_path=queries_path,
                k_values=parse_k_values(k_values),
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
            )
            return render_dashboard(
                request,
                messages=[Message("success", "评估完成。")],
                evaluation=table,
                config=config,
            )
        except WebServiceError as exc:
            return render_dashboard(request, messages=[Message("error", str(exc))], config=config)

    return app


app = create_app()


def main() -> None:
    print("Contextual Retrieval Web Console: http://127.0.0.1:8000")
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

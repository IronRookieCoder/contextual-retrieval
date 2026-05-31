from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import Config
from src.web.services import get_config_status


WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="Contextual Retrieval Web Console")
    app.mount(
        "/static",
        StaticFiles(directory=str(WEB_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        config = Config.from_env()
        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            {
                "messages": [],
                "config_status": get_config_status(config),
            },
        )

    return app


app = create_app()


def main() -> None:
    print("Contextual Retrieval Web Console: http://127.0.0.1:8000")
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

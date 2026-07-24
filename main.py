import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.database import Base, engine
from app.exceptions import AppError
from app.routers import ingest, ui
from app.settings import settings

logger = logging.getLogger("mlops_dashboard")

_TEMPLATE_DIR = Path(__file__).parent / "app" / "templates"
_templates = Jinja2Templates(directory=_TEMPLATE_DIR)


def _is_api_request(request: Request) -> bool:
    """API リクエスト（JSON 応答）か UI リクエスト（HTML 応答）かを判定する。"""
    return request.url.path.startswith("/api")


def _error_response(request: Request, status_code: int, message: str):
    """モードに応じたエラー応答を返す（/api は JSON、UI は HTML 断片）。"""
    if _is_api_request(request):
        return JSONResponse({"detail": message}, status_code=status_code)
    return _templates.TemplateResponse(
        request, "partials/error.html", {"message": message}, status_code=status_code
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "MLOps Dashboard 起動 — モード: %s | DB: %s",
        settings.app_mode.value,
        settings.sqlite_url if settings.is_local_dev else "Dataiku (" + settings.app_mode.value + ")",
    )
    if settings.is_dataiku:
        logger.info(
            "Dataiku 設定 — host=%s, mgmt_project=%s",
            settings.dataiku_host or "(内部コンテキスト)",
            settings.dku_mgmt_project_key,
        )
# SQLite は常に初期化（設定値・閾値の保存に使用）
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="MLOps Dashboard", lifespan=lifespan)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    """業務例外を捕捉し、API/UI に応じた応答を返す。"""
    if exc.status_code >= 500:
        logger.error("AppError(%s): %s", exc.status_code, exc.log_message, exc_info=exc)
    else:
        logger.warning("AppError(%s): %s", exc.status_code, exc.log_message)
    return _error_response(request, exc.status_code, exc.user_message)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """想定外の例外を捕捉する最後の砦。内部情報は返さずログに残す。"""
    logger.error("未処理の例外: %s", exc, exc_info=exc)
    return _error_response(request, 500, "予期しないエラーが発生しました")


_STATIC_DIR = Path(__file__).parent / "app" / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(ingest.router)
app.include_router(ui.router)

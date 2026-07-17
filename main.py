import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.routers import agent_ingest, agent_ui, ingest, ui
from app.settings import settings

logger = logging.getLogger("mlops_dashboard")


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

_STATIC_DIR = Path(__file__).parent / "app" / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(ingest.router)
app.include_router(ui.router)
app.include_router(agent_ingest.router)
app.include_router(agent_ui.router)

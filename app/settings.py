from __future__ import annotations

import os
from enum import Enum


class AppMode(str, Enum):
    LOCAL_DEV = "local_dev"
    DEV = "dev"
    PRODUCTION = "production"


class Settings:
    def __init__(self) -> None:
        raw_mode = os.environ.get("APP_MODE", "local_dev")
        try:
            self.app_mode: AppMode = AppMode(raw_mode)
        except ValueError:
            raise ValueError(
                f"APP_MODE の値が不正です: '{raw_mode}'. "
                "有効な値: local_dev | dev | production"
            )

        # ── SQLite (local_dev) ────────────────────────────────────────
        self.sqlite_url: str = os.environ.get("DATABASE_URL", "sqlite:///./mlops.db")

        # ── Dataiku 接続設定 (dev / production) ──────────────────────
        # DSS webapp 内で実行する場合はホスト・APIキー不要（内部コンテキスト自動取得）
        # DSS 外部から接続する場合は DATAIKU_HOST と DATAIKU_API_KEY を設定
        self.dataiku_host: str = os.environ.get("DATAIKU_HOST", "")
        self.dataiku_api_key: str = os.environ.get("DATAIKU_API_KEY", "")

        # 管理プロジェクト: m_projects のマスタデータを格納
        self.dku_mgmt_project_key: str = os.environ.get(
            "DATAIKU_MGMT_PROJECT_KEY", "mlops_dev"
        )

        # ── Dataiku データセット名 ────────────────────────────────────
        # デフォルト値は SQLite のテーブル名と同じ（カラム名も準拠）
        self.dku_ds_projects: str = os.environ.get(
            "DKU_DATASET_PROJECTS", "m_projects"
        )
        self.dku_ds_inference_logs: str = os.environ.get(
            "DKU_DATASET_INFERENCE_LOGS", "t_inference_logs"
        )
        self.dku_ds_deployed_models: str = os.environ.get(
            "DKU_DATASET_DEPLOYED_MODELS", "m_deployed_models"
        )

    @property
    def is_local_dev(self) -> bool:
        return self.app_mode == AppMode.LOCAL_DEV

    @property
    def is_dataiku(self) -> bool:
        return self.app_mode in (AppMode.DEV, AppMode.PRODUCTION)

    def __repr__(self) -> str:
        return (
            f"Settings(app_mode={self.app_mode.value!r}, "
            f"sqlite_url={self.sqlite_url!r})"
        )


settings = Settings()

from __future__ import annotations

import os
from enum import Enum

from dotenv import load_dotenv

from .exceptions import ConfigurationError

load_dotenv()

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _env_bool(name: str, default: bool) -> bool:
    """環境変数を bool として読む（true/1/yes/on | false/0/no/off、大小文字不問）。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    val = raw.strip().lower()
    if val in _BOOL_TRUE:
        return True
    if val in _BOOL_FALSE:
        return False
    raise ConfigurationError(
        log_message=f"{name} の値が不正です: '{raw}'. 有効な値: true | false"
    )


class AppMode(str, Enum):
    LOCAL_DEV = "local_dev"
    DEV = "dev"
    PRODUCTION = "production"


class Settings:
    def __init__(self) -> None:
        raw_mode = os.environ.get("APP_MODE", "local_dev")
        try:
            self.app_mode: AppMode = AppMode(raw_mode)
        except ValueError as exc:
            raise ConfigurationError(
                log_message=f"APP_MODE の値が不正です: '{raw_mode}'. 有効な値: local_dev | dev | production"
            ) from exc

        # ── アプリ共通設定 ────────────────────────────────────────────
        # ダッシュボードの自動更新（詳細: 30 秒、サマリー: 60 秒）の有効化フラグ
        self.auto_refresh_enabled: bool = _env_bool("AUTO_REFRESH_ENABLED", True)

        # ログレベル（DEBUG のとき @log_call の呼び出しログも出力される）
        raw_level = os.environ.get("LOG_LEVEL", "DEBUG").strip().upper()
        if raw_level not in _LOG_LEVELS:
            raise ConfigurationError(
                log_message=(
                    f"LOG_LEVEL の値が不正です: '{raw_level}'. "
                    f"有効な値: {' | '.join(_LOG_LEVELS)}"
                )
            )
        self.log_level: str = raw_level

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
        # m_projects と同じ管理プロジェクトに配備される閾値設定データセット
        self.dku_ds_project_configs: str = os.environ.get(
            "DKU_DATASET_PROJECT_CONFIGS", "m_project_configs"
        )
        self.dku_ds_inference_logs: str = os.environ.get(
            "DKU_DATASET_INFERENCE_LOGS", "t_inference_logs"
        )
        self.dku_ds_deployed_models: str = os.environ.get(
            "DKU_DATASET_DEPLOYED_MODELS", "t_deployed_models"
        )
        self.dku_ds_reference_logs: str = os.environ.get(
            "DKU_DATASET_REFERENCE_LOGS", "t_reference_logs"
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

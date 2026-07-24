"""
Dataiku DSS データプロバイダー

dev / production モードで Dataiku データセットから pandas DataFrame を取得する。

プロジェクト構成:
  管理プロジェクト (DATAIKU_MGMT_PROJECT_KEY, デフォルト: mlops_dev)
    - m_projects        … MLOps 管理対象プロジェクト一覧
    - m_agent_projects  … エージェントプロジェクト一覧
    - m_agent_ref_data  … エージェントドリフト参照データ

  各 Ops 対象プロジェクト (m_projects.project_id = Dataiku プロジェクトキー)
    - t_inference_logs  … 推論ログ
    - m_deployed_models … デプロイ済みモデル
    - t_agent_tasks     … エージェントタスクログ
    - t_agent_steps     … エージェントステップログ

dataiku パッケージは DSS 環境内でのみ利用可能なため、遅延インポートを使用。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import String

from .models import InferenceLog, Project
from .settings import settings
from .logging_utils import log_call

logger = logging.getLogger(__name__)


# ── 内部: DataFrame 取得 ─────────────────────────────────────────────────────

@log_call
def _ensure_imports():
    """dataiku / pandas のインポートを確認し、両モジュールを返す。"""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas がインストールされていません。"
            "Dataiku モードでは pandas が必要です: uv add pandas"
        ) from exc

    try:
        import dataiku
    except ImportError as exc:
        raise RuntimeError(
            "dataiku パッケージが見つかりません。"
            "APP_MODE=dev/production は Dataiku DSS 環境内で実行してください。"
        ) from exc

    return dataiku, pd


@log_call
def _fetch_df(dataset_name: str, project_key: str | None = None):
    """指定プロジェクトの Dataiku データセットを DataFrame として取得する。

    project_key が None の場合は settings.dku_mgmt_project_key を使用する。
    """
    dataiku, _ = _ensure_imports()
    effective_key = project_key or settings.dku_mgmt_project_key
    if effective_key:
        ds = dataiku.Dataset(dataset_name, project_key=effective_key)
    else:
        ds = dataiku.Dataset(dataset_name)
    return ds.get_dataframe()


@log_call
def _write_df(df, dataset_name: str, project_key: str | None = None) -> None:
    """DataFrame を Dataiku データセットに上書き保存する。

    project_key が None の場合は settings.dku_mgmt_project_key を使用する。
    """
    dataiku, _ = _ensure_imports()
    effective_key = project_key or settings.dku_mgmt_project_key
    if effective_key:
        ds = dataiku.Dataset(dataset_name, project_key=effective_key)
    else:
        ds = dataiku.Dataset(dataset_name)
    ds.write_with_schema(df)


@log_call
def _to_naive_utc(series):
    """timezone-aware な datetime 列を UTC-naive に変換する。"""
    import pandas as pd
    col = pd.to_datetime(series, utc=True)
    if col.dt.tz is not None:
        col = col.dt.tz_localize(None)
    return col


# ── プロジェクト解決 ─────────────────────────────────────────────────────────

@log_call
def _get_target_project_key(project_id: str) -> str | None:
    """project_id に対応する Dataiku プロジェクトキーを返す。

    Dataiku プロジェクトキーは project_id と一致する。
    管理プロジェクト (dku_mgmt_project_key) の m_projects に登録済みか確認したうえで、
    project_id を文字列キーとして返す（未登録なら None）。
    """
    df = _fetch_df(settings.dku_ds_projects)
    row = df[df["project_id"].astype(str) == str(project_id)]
    if row.empty:
        return None
    return str(project_id)


# ── プロジェクト一覧（管理プロジェクトから取得） ──────────────────────────────

@log_call
def get_projects() -> list[Project]:
    """管理プロジェクトの m_projects から一覧を返す。

    local_dev モード（DB からの取得）と型を揃えるため、models.Project の
    インスタンスとして返す。
    """
    logger.info("get_projects: dataset=%s project=%s", settings.dku_ds_projects, settings.dku_mgmt_project_key)
    try:
        df = _fetch_df(settings.dku_ds_projects)
    except Exception:
        logger.exception("get_projects: データセット取得に失敗しました")
        raise
    df["created_at"] = _to_naive_utc(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    projects = [_row_to_model(row, Project) for _, row in df.iterrows()]
    logger.info("get_projects: %d 件取得", len(projects))
    return projects


@log_call
def get_project_by_id(project_id: str) -> Project | None:
    """project_id でプロジェクトを 1 件取得する。"""
    projects = get_projects()
    return next((p for p in projects if str(p.project_id) == str(project_id)), None)


@log_call
def delete_project(project_id: str) -> bool:
    """管理プロジェクトの m_projects から指定プロジェクトを削除する。

    m_projects の行のみを削除し、各 Ops プロジェクトの子データ
    (t_inference_logs 等) は変更しない。削除できたら True を返す。
    """
    df = _fetch_df(settings.dku_ds_projects)
    if df.empty or "project_id" not in df.columns:
        return False

    mask = df["project_id"].astype(str) == str(project_id)
    if not bool(mask.any()):
        return False

    remaining = df[~mask].reset_index(drop=True)
    _write_df(remaining, settings.dku_ds_projects)
    logger.info("delete_project: project_id=%s を m_projects から削除", project_id)
    return True


# ── ML 推論ログ（各 Ops 対象プロジェクトから取得） ───────────────────────────

@log_call
def get_inference_logs_df(project_id: str, since: datetime | None = None):
    """対象プロジェクトの t_inference_logs を取得する。

    project_id を Dataiku プロジェクトキーとして解決し、
    そのプロジェクト内の t_inference_logs データセットを取得する。
    """
    import pandas as pd
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return pd.DataFrame()

    df = _fetch_df(settings.dku_ds_inference_logs, project_key=dku_project_key)
    if df.empty:
        return df

    if "project_id" in df.columns:
        df = df[df["project_id"].astype(str) == str(project_id)].copy()

    df["request_timestamp"] = _to_naive_utc(df["request_timestamp"])
    if since is not None:
        df = df[df["request_timestamp"] >= since]
    if "is_error" in df.columns:
        df["is_error"] = df["is_error"].fillna(False).astype(bool)
    return df.reset_index(drop=True)


@log_call
def get_inference_logs_page(
    project_id: str,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    request_id_filter: str | None = None,
    is_error: bool | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[InferenceLog], int]:
    """t_inference_logs をフィルタ・ページング付きで取得する。

    ログ一覧 UI 向け。local_dev モード（DB クエリ）と型を揃えるため、
    models.InferenceLog インスタンスのリストと総件数のタプルを返す。
    """
    df = get_inference_logs_df(project_id)
    if df.empty:
        return [], 0

    if from_dt is not None:
        df = df[df["request_timestamp"] >= from_dt]
    if to_dt is not None:
        df = df[df["request_timestamp"] <= to_dt]
    if request_id_filter and "request_id" in df.columns:
        df = df[df["request_id"].fillna("").str.contains(request_id_filter, regex=False)]
    if is_error is not None and "is_error" in df.columns:
        df = df[df["is_error"] == is_error]

    total = len(df)
    df = df.sort_values("request_timestamp", ascending=False).reset_index(drop=True)
    start = (page - 1) * page_size
    df = df.iloc[start:start + page_size]
    logs = [_row_to_model(row, InferenceLog) for _, row in df.iterrows()]
    return logs, total


@log_call
def delete_inference_logs(project_id: str, log_ids: list[str]) -> int:
    """対象プロジェクトの t_inference_logs から指定ログを削除する。

    データセットを読み込み、log_id が log_ids に含まれる行を除外して
    書き戻す。削除した件数を返す。
    """
    if not log_ids:
        return 0
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return 0

    df = _fetch_df(settings.dku_ds_inference_logs, project_key=dku_project_key)
    if df.empty or "log_id" not in df.columns:
        return 0

    id_set = {str(i) for i in log_ids}
    mask = df["log_id"].astype(str).isin(id_set)
    deleted = int(mask.sum())
    if deleted == 0:
        return 0

    remaining = df[~mask].reset_index(drop=True)
    _write_df(remaining, settings.dku_ds_inference_logs, project_key=dku_project_key)
    logger.info("delete_inference_logs: project_id=%s から %d 件削除", project_id, deleted)
    return deleted


@log_call
def get_deployed_models_df(project_id: str):
    """対象プロジェクトの m_deployed_models を取得する。

    project_id を Dataiku プロジェクトキーとして解決し、
    そのプロジェクト内の m_deployed_models データセットを取得する。
    """
    import pandas as pd
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return pd.DataFrame()

    df = _fetch_df(settings.dku_ds_deployed_models, project_key=dku_project_key)
    if df.empty:
        return df

    if "project_id" in df.columns:
        df = df[df["project_id"].astype(str) == str(project_id)].copy()

    df["created_at"] = _to_naive_utc(df["created_at"])
    return df.reset_index(drop=True)


@log_call
def get_reference_logs_df(project_id: str, model_id: str | None = None):
    """対象プロジェクトの t_reference_logs を取得する。"""
    import pandas as pd
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return pd.DataFrame()

    df = _fetch_df(settings.dku_ds_reference_logs, project_key=dku_project_key)
    if df.empty:
        return df

    if "project_id" in df.columns:
        df = df[df["project_id"].astype(str) == str(project_id)].copy()
    if model_id is not None and "model_id" in df.columns:
        df = df[df["model_id"].astype(str) == str(model_id)].copy()

    return df.reset_index(drop=True)


# ── ユーティリティ ───────────────────────────────────────────────────────────

@log_call
def _row_to_dict(row) -> dict:
    """pandas Series を dict に変換（NaN → None, numpy 型 → Python 型）。"""
    import numpy as np
    result = {}
    for k, v in row.items():
        if v is None:
            result[k] = None
        elif isinstance(v, float) and np.isnan(v):
            result[k] = None
        elif hasattr(v, "item"):
            result[k] = v.item()
        else:
            result[k] = v
    return result


@log_call
def _row_to_model(row, model):
    """pandas Series を SQLAlchemy モデルインスタンスに変換する。

    対象モデルに定義されたカラムのみを取り込み、DataFrame に含まれる
    余分な列は無視する。DB セッションには紐付かない一時オブジェクトを返す。
    """
    data = _row_to_dict(row)
    columns = {c.name: c.type for c in model.__table__.columns}
    kwargs = {}
    for k, v in data.items():
        if k not in columns:
            continue
        # String 型カラム（ID 等）は文字列に統一する
        if v is not None and isinstance(columns[k], String):
            v = str(v)
        kwargs[k] = v
    return model(**kwargs)


@log_call
def parse_json_col(val) -> dict | None:
    """JSON 文字列カラムを dict に変換する（None / NaN 対応）。"""
    import numpy as np
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val

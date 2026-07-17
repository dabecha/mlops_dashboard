"""
Dataiku DSS データプロバイダー

dev / production モードで Dataiku データセットから pandas DataFrame を取得する。

プロジェクト構成:
  管理プロジェクト (DATAIKU_MGMT_PROJECT_KEY, デフォルト: mlops_dev)
    - m_projects        … MLOps 管理対象プロジェクト一覧
    - m_agent_projects  … エージェントプロジェクト一覧
    - m_agent_ref_data  … エージェントドリフト参照データ

  各 Ops 対象プロジェクト (m_projects.project_name = Dataiku プロジェクトキー)
    - t_inference_logs  … 推論ログ
    - m_deployed_models … デプロイ済みモデル
    - t_agent_tasks     … エージェントタスクログ
    - t_agent_steps     … エージェントステップログ

dataiku パッケージは DSS 環境内でのみ利用可能なため、遅延インポートを使用。
"""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from .settings import settings


# ── 内部: DataFrame 取得 ─────────────────────────────────────────────────────

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


def _to_naive_utc(series):
    """timezone-aware な datetime 列を UTC-naive に変換する。"""
    import pandas as pd
    col = pd.to_datetime(series, utc=True)
    if col.dt.tz is not None:
        col = col.dt.tz_localize(None)
    return col


# ── プロジェクト解決 ─────────────────────────────────────────────────────────

def _get_target_project_key(project_id: int) -> str | None:
    """project_id に対応する Dataiku プロジェクトキーを返す。

    m_projects.project_name を Dataiku プロジェクトキーとして扱う。
    管理プロジェクト (dku_mgmt_project_key) から m_projects を取得して解決する。
    """
    df = _fetch_df(settings.dku_ds_projects)
    row = df[df["project_id"] == project_id]
    if row.empty:
        return None
    return str(row.iloc[0]["project_name"])


# ── プロジェクト一覧（管理プロジェクトから取得） ──────────────────────────────

def get_projects() -> list[SimpleNamespace]:
    """管理プロジェクトの m_projects から一覧を返す。

    テンプレートが project.project_id, project.project_name 等にアクセスできるよう
    SimpleNamespace で返す。
    """
    df = _fetch_df(settings.dku_ds_projects)
    df["created_at"] = _to_naive_utc(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return [SimpleNamespace(**_row_to_dict(row)) for _, row in df.iterrows()]


def get_project_by_id(project_id: int) -> SimpleNamespace | None:
    """project_id でプロジェクトを 1 件取得する。"""
    projects = get_projects()
    return next((p for p in projects if int(p.project_id) == project_id), None)


# ── ML 推論ログ（各 Ops 対象プロジェクトから取得） ───────────────────────────

def get_inference_logs_df(project_id: int, since: datetime | None = None):
    """対象プロジェクトの t_inference_logs を取得する。

    m_projects.project_name を Dataiku プロジェクトキーとして解決し、
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
        df = df[df["project_id"] == project_id].copy()

    df["request_timestamp"] = _to_naive_utc(df["request_timestamp"])
    if since is not None:
        df = df[df["request_timestamp"] >= since]
    if "is_error" in df.columns:
        df["is_error"] = df["is_error"].fillna(False).astype(bool)
    return df.reset_index(drop=True)


def get_deployed_models_df(project_id: int):
    """対象プロジェクトの m_deployed_models を取得する。

    m_projects.project_name を Dataiku プロジェクトキーとして解決し、
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
        df = df[df["project_id"] == project_id].copy()

    df["created_at"] = _to_naive_utc(df["created_at"])
    return df.reset_index(drop=True)


def get_reference_logs_df(project_id: int, model_id: int | None = None):
    """対象プロジェクトの t_reference_logs を取得する。"""
    import pandas as pd
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return pd.DataFrame()

    df = _fetch_df(settings.dku_ds_reference_logs, project_key=dku_project_key)
    if df.empty:
        return df

    if "project_id" in df.columns:
        df = df[df["project_id"] == project_id].copy()
    if model_id is not None and "model_id" in df.columns:
        df = df[df["model_id"] == model_id].copy()

    return df.reset_index(drop=True)


# ── ユーティリティ ───────────────────────────────────────────────────────────

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

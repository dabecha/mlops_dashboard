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
    - t_deployed_models … デプロイ済みモデル
    - t_agent_tasks     … エージェントタスクログ
    - t_agent_steps     … エージェントステップログ

dataiku パッケージは DSS 環境内でのみ利用可能なため、遅延インポートを使用。
"""
from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime

from sqlalchemy import String

from .exceptions import (
    ConfigurationError,
    DataIntegrityError,
    DataSourceError,
    MissingColumnError,
    MissingDatasetError,
    ValidationError,
)
from .models import DeployedModel, InferenceLog, Project, ProjectConfig, ReferenceLog
from .prediction import parse_value
from .settings import settings
from .logging_utils import log_call

logger = logging.getLogger(__name__)


# ── 内部: エラーメッセージ組み立て ───────────────────────────────────────────

_LOW_LEVEL_HELPERS = {
    "_calling_function", "_source_error_message",
    "_fetch_df", "_write_df", "_list_dataset_names",
}


def _calling_function() -> str:
    """低レベルヘルパー（_fetch_df 等）に至る本モジュール内の呼び出し連鎖を返す。

    例: "get_inference_logs_df > _get_target_project_key"（外側から順、最大 3 段）。
    log_call デコレータの wrapper（logging_utils.py）は別ファイルのため除外される。
    特定できない場合は "(不明)" を返す。
    """
    names = []
    for frame_info in inspect.stack()[1:]:
        if frame_info.filename != __file__:
            continue
        if frame_info.function in _LOW_LEVEL_HELPERS:
            continue
        names.append(frame_info.function)
    if not names:
        return "(不明)"
    return " > ".join(reversed(names[:3]))


def _describe_cause(exc: Exception) -> str:
    """例外メッセージから推定される原因のヒントを返す（該当なしは空文字）。"""
    msg = str(exc).lower()
    if any(k in msg for k in ("does not exist", "not exist", "not found", "no such", "unknown dataset")):
        return "データセット未作成、またはデータセット名 / project_key の誤りの可能性"
    if any(k in msg for k in ("unauthorized", "forbidden", "permission", "access denied", "401", "403")):
        return "Dataiku API キーの権限不足、または認証失敗の可能性（DATAIKU_API_KEY を確認）"
    if any(k in msg for k in ("connection", "timed out", "timeout", "refused", "unreachable", "getaddrinfo", "name resolution")):
        return "DSS への接続失敗の可能性（DATAIKU_HOST / ネットワーク設定を確認）"
    if any(k in msg for k in ("schema", "column", "dtype", "cast")):
        return "データセットのスキーマ（カラム定義・型）不整合の可能性"
    return ""


def _source_error_message(operation: str, exc: Exception, **context) -> str:
    """DataSourceError 用の log_message を組み立てる。

    どの関数からの操作か（呼び出し元）、対象（dataset / project_key）、
    元例外の型・内容、推定原因を 1 行にまとめる。
    """
    caller = _calling_function()
    ctx = ", ".join(f"{k}={v if v not in (None, '') else '(未指定)'}" for k, v in context.items())
    msg = f"{caller}: {operation}に失敗 ({ctx}) | 例外: {type(exc).__name__}: {exc}"
    hint = _describe_cause(exc)
    if hint:
        msg += f" | 推定原因: {hint}"
    return msg


# ── 内部: DataFrame 取得 ─────────────────────────────────────────────────────

@log_call
def _ensure_imports():
    """dataiku / pandas のインポートを確認し、両モジュールを返す。"""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ConfigurationError(
            log_message="pandas がインストールされていません。Dataiku モードでは pandas が必要です: uv add pandas"
        ) from exc

    try:
        import dataiku
    except ImportError as exc:
        raise ConfigurationError(
            log_message="dataiku パッケージが見つかりません。APP_MODE=dev/production は Dataiku DSS 環境内で実行してください。"
        ) from exc

    return dataiku, pd


@log_call
def _fetch_df(dataset_name: str, project_key: str | None = None):
    """指定プロジェクトの Dataiku データセットを DataFrame として取得する。

    project_key が None の場合は settings.dku_mgmt_project_key を使用する。
    """
    dataiku, _ = _ensure_imports()
    effective_key = project_key or settings.dku_mgmt_project_key
    try:
        if effective_key:
            ds = dataiku.Dataset(dataset_name, project_key=effective_key)
        else:
            ds = dataiku.Dataset(dataset_name)
        return ds.get_dataframe()
    except Exception as exc:
        raise DataSourceError(
            log_message=_source_error_message(
                "データセット取得", exc,
                dataset=dataset_name, project_key=effective_key,
            )
        ) from exc


@log_call
def _write_df(df, dataset_name: str, project_key: str | None = None) -> None:
    """DataFrame を Dataiku データセットに上書き保存する。

    project_key が None の場合は settings.dku_mgmt_project_key を使用する。
    """
    dataiku, _ = _ensure_imports()
    effective_key = project_key or settings.dku_mgmt_project_key
    try:
        if effective_key:
            ds = dataiku.Dataset(dataset_name, project_key=effective_key)
        else:
            ds = dataiku.Dataset(dataset_name)
        ds.write_with_schema(df)
    except Exception as exc:
        raise DataSourceError(
            log_message=_source_error_message(
                "データセット書き込み", exc,
                dataset=dataset_name, project_key=effective_key,
            )
        ) from exc


@log_call
def _list_dataset_names(project_key: str) -> set[str]:
    """指定 Dataiku プロジェクト内のデータセット名の集合を取得する。"""
    dataiku, _ = _ensure_imports()
    try:
        client = dataiku.api_client()
        project = client.get_project(project_key)
        return {d["name"] for d in project.list_datasets()}
    except Exception as exc:
        raise DataSourceError(
            log_message=_source_error_message(
                "データセット一覧の取得", exc,
                project_key=project_key,
            )
        ) from exc


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
    _validate_columns(df, Project, settings.dku_ds_projects)
    row = df[df["project_id"].astype(str) == str(project_id)]
    if row.empty:
        return None
    return str(project_id)


@log_call
def check_project_datasets(project_id: str) -> None:
    """Ops プロジェクトに必須テーブルが揃っているか確認する。

    t_deployed_models / t_inference_logs / t_reference_logs のいずれかが
    存在しない場合はエラーログを出力し、MissingDatasetError を送出する
    （どのテーブルが不足しているかをユーザー画面に表示するため）。
    """
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return

    required = [
        settings.dku_ds_deployed_models,
        settings.dku_ds_inference_logs,
        settings.dku_ds_reference_logs,
    ]
    existing = _list_dataset_names(dku_project_key)
    missing = [name for name in required if name not in existing]
    if missing:
        logger.error(
            "プロジェクト '%s' に必須テーブルがありません: %s（Dataiku で作成が必要）",
            dku_project_key, missing,
        )
        raise MissingDatasetError(missing, dku_project_key)


# ── プロジェクト一覧（管理プロジェクトから取得） ──────────────────────────────

@log_call
def get_projects() -> list[Project]:
    """管理プロジェクトの m_projects から一覧を返す。

    local_dev モード（DB からの取得）と型を揃えるため、models.Project の
    インスタンスとして返す。
    """
    logger.info("get_projects: dataset=%s project=%s", settings.dku_ds_projects, settings.dku_mgmt_project_key)
    df = _fetch_df(settings.dku_ds_projects)
    _validate_columns(df, Project, settings.dku_ds_projects)
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
def create_project(
    project_id: str,
    project_name: str,
    description: str | None,
    task_type: str,
) -> Project:
    """管理プロジェクトの m_projects にプロジェクトを追加する。

    project_id は対象の Dataiku プロジェクトキーと一致させる必要がある
    （_get_target_project_key が project_id をキーとして解決するため）。
    project_id / project_name が重複する場合は ValidationError を送出する。
    """
    import pandas as pd
    df = _fetch_df(settings.dku_ds_projects)
    _validate_columns(df, Project, settings.dku_ds_projects)

    if not df.empty:
        if bool((df["project_id"].astype(str) == str(project_id)).any()):
            raise ValidationError("同じプロジェクトIDが既に存在します")
        if bool((df["project_name"].astype(str) == str(project_name)).any()):
            raise ValidationError("同名のプロジェクトが既に存在します")

    now = datetime.utcnow()
    new_row = {
        "project_id": str(project_id),
        "project_name": project_name,
        "description": description,
        "task_type": task_type,
        "created_at": _now_for_column(df, "created_at", now),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _write_df(df, settings.dku_ds_projects)
    logger.info("create_project: project_id=%s を m_projects に追加", project_id)
    return Project(
        project_id=str(project_id),
        project_name=project_name,
        description=description,
        task_type=task_type,
        created_at=now,
    )


@log_call
def delete_project(project_id: str) -> bool:
    """管理プロジェクトの m_projects から指定プロジェクトを削除する。

    m_projects の行のみを削除し、各 Ops プロジェクトの子データ
    (t_inference_logs 等) は変更しない。削除できたら True を返す。
    """
    df = _fetch_df(settings.dku_ds_projects)
    _validate_columns(df, Project, settings.dku_ds_projects)
    if df.empty or "project_id" not in df.columns:
        return False

    mask = df["project_id"].astype(str) == str(project_id)
    if not bool(mask.any()):
        return False

    remaining = df[~mask].reset_index(drop=True)
    _write_df(remaining, settings.dku_ds_projects)
    logger.info("delete_project: project_id=%s を m_projects から削除", project_id)
    return True


# ── プロジェクト閾値設定（管理プロジェクトから取得） ─────────────────────────

def _normalize_config_columns(df):
    """m_project_configs のカラム名ゆらぎを吸収する（is_higer_better → is_higher_better）。"""
    if "is_higer_better" in df.columns and "is_higher_better" not in df.columns:
        df = df.rename(columns={"is_higer_better": "is_higher_better"})
    return df


def _now_for_column(df, column: str, now: datetime):
    """既存カラムの dtype に合わせた現在時刻を返す（tz-aware カラムには tz 付きで代入する）。"""
    import pandas as pd
    if column in df.columns and isinstance(df[column].dtype, pd.DatetimeTZDtype):
        return pd.Timestamp(now).tz_localize("UTC").tz_convert(df[column].dtype.tz)
    return now


@log_call
def get_project_config(project_id: str) -> dict | None:
    """管理プロジェクトの m_project_configs から対象プロジェクトの閾値設定を取得する。

    m_projects と同じ管理プロジェクト (dku_mgmt_project_key) に配備されている
    データセットを参照する。未登録の場合は None を返す（呼び出し側でデフォルト値を適用）。
    """
    import pandas as pd
    df = _fetch_df(settings.dku_ds_project_configs)
    df = _normalize_config_columns(df)
    _validate_columns(df, ProjectConfig, settings.dku_ds_project_configs)
    if df.empty:
        return None

    rows = df[df["project_id"].astype(str) == str(project_id)]
    if rows.empty:
        return None

    data = _row_to_dict(rows.iloc[0])
    # 日時カラムは NaT を None に変換し、timezone-aware は naive に揃える
    for k in ("created_at", "updated_at"):
        v = data.get(k)
        if v is None or pd.isna(v):
            data[k] = None
        else:
            ts = pd.to_datetime(v, utc=True)
            data[k] = ts.tz_localize(None).to_pydatetime() if ts.tzinfo else ts.to_pydatetime()
    if data.get("is_higher_better") is not None:
        data["is_higher_better"] = bool(data["is_higher_better"])
    return data


@log_call
def upsert_project_config(project_id: str, values: dict) -> dict:
    """管理プロジェクトの m_project_configs に閾値設定を upsert し、保存内容を返す。"""
    import pandas as pd
    df = _fetch_df(settings.dku_ds_project_configs)
    df = _normalize_config_columns(df)
    _validate_columns(df, ProjectConfig, settings.dku_ds_project_configs)

    now = datetime.utcnow()
    mask = df["project_id"].astype(str) == str(project_id) if not df.empty else None
    if mask is None or not bool(mask.any()):
        next_id = 1 if df.empty else int(pd.to_numeric(df["id"], errors="coerce").fillna(0).max()) + 1
        new_row = {**values, "id": next_id, "project_id": str(project_id),
                   "created_at": _now_for_column(df, "created_at", now),
                   "updated_at": _now_for_column(df, "updated_at", now)}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        for k, v in values.items():
            df.loc[mask, k] = v
        df.loc[mask, "updated_at"] = _now_for_column(df, "updated_at", now)

    _write_df(df, settings.dku_ds_project_configs)
    logger.info("upsert_project_config: project_id=%s の閾値設定を保存", project_id)
    return {**values, "project_id": str(project_id), "updated_at": now}


# ── ML 推論ログ（各 Ops 対象プロジェクトから取得） ───────────────────────────

@log_call
def get_inference_logs_df(project_id: str, since: datetime | None = None, until: datetime | None = None):
    """対象プロジェクトの t_inference_logs を取得する。

    project_id を Dataiku プロジェクトキーとして解決し、
    そのプロジェクト内の t_inference_logs データセットを取得する。
    since / until を指定すると request_timestamp が since 以上 until 未満の行に絞る。
    """
    import pandas as pd
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return pd.DataFrame()

    df = _fetch_df(settings.dku_ds_inference_logs, project_key=dku_project_key)
    _validate_columns(df, InferenceLog, settings.dku_ds_inference_logs)
    if df.empty:
        return df

    if "project_id" in df.columns:
        df = df[df["project_id"].astype(str) == str(project_id)].copy()

    df["request_timestamp"] = _to_naive_utc(df["request_timestamp"])
    if "updated_at" in df.columns:
        df["updated_at"] = _to_naive_utc(df["updated_at"])
    # prediction_values / actual_values (JSON {"label": 値}) は代表スカラー値に変換
    for col in ("prediction_values", "actual_values"):
        if col in df.columns:
            df[col] = df[col].map(parse_value)
    if since is not None:
        df = df[df["request_timestamp"] >= since]
    if until is not None:
        df = df[df["request_timestamp"] < until]
    if "is_error" in df.columns:
        df["is_error"] = df["is_error"].fillna(False).astype(bool)
    return df.reset_index(drop=True)


@log_call
def get_inference_logs_page(
    project_id: str,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
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
    _validate_columns(df, InferenceLog, settings.dku_ds_inference_logs)
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
    """対象プロジェクトの t_deployed_models を取得する。

    project_id を Dataiku プロジェクトキーとして解決し、
    そのプロジェクト内の t_deployed_models データセットを取得する。
    """
    import pandas as pd
    dku_project_key = _get_target_project_key(project_id)
    if not dku_project_key:
        return pd.DataFrame()

    df = _fetch_df(settings.dku_ds_deployed_models, project_key=dku_project_key)
    _validate_columns(df, DeployedModel, settings.dku_ds_deployed_models)
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
    _validate_columns(df, ReferenceLog, settings.dku_ds_reference_logs)
    if df.empty:
        return df

    if "project_id" in df.columns:
        df = df[df["project_id"].astype(str) == str(project_id)].copy()
    if model_id is not None and "model_id" in df.columns:
        df = df[df["model_id"].astype(str) == str(model_id)].copy()

    # actual_values (JSON {"label": 値}) は代表スカラー値に変換
    if "actual_values" in df.columns:
        df["actual_values"] = df["actual_values"].map(parse_value)

    return df.reset_index(drop=True)


# ── ユーティリティ ───────────────────────────────────────────────────────────

def _validate_columns(df, model, dataset_name: str) -> None:
    """DataFrame のカラムがモデル定義と一致するか検証する。

    モデル (SQLAlchemy) に定義されたカラムのうち DataFrame に存在しないものが
    あれば、不足カラム名を添えてエラーログを出力し MissingColumnError を送出する。
    """
    expected = {c.name for c in model.__table__.columns}
    missing = sorted(expected - set(df.columns))
    if missing:
        logger.error(
            "テーブル %s のカラムが定義と異なります。不足カラム: %s",
            dataset_name, missing,
        )
        raise MissingColumnError(missing, dataset_name)


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
        try:
            return json.loads(val)
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(
                log_message=f"JSON カラムのパースに失敗: {val[:100]!r}: {exc}"
            ) from exc
    return val

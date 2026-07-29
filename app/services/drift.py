from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from ..exceptions import DataIntegrityError
from ..models import DeployedModel, InferenceLog, ReferenceLog
from ..prediction import parse_value
from ..settings import settings
from ..logging_utils import log_call


def _loads(raw: str):
    """特徴量などの JSON 文字列を dict に変換する（失敗時は DataIntegrityError）。"""
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(
            log_message=f"特徴量 JSON のパースに失敗: {str(raw)[:100]!r}: {exc}"
        ) from exc


@log_call
def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index を計算する。

    空ビンがあると log 比が発散して PSI が過大評価されるため、
    各ビンに擬似カウント 0.5 を加える（Laplace 平滑化）。
    """
    pseudo = 0.5
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = (exp_counts + pseudo) / (len(expected) + pseudo * bins)
    act_pct = (act_counts + pseudo) / (len(actual) + pseudo * bins)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


@log_call
def _two_sample_drift_result(
    reference: np.ndarray,
    current: np.ndarray,
    psi_warning: float,
    psi_alert: float,
    ks_alpha: float,
) -> dict:
    """2 標本の PSI + KS 検定によるドリフト判定結果を組み立てる。"""
    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    psi = _compute_psi(reference, current)

    psi_level = "ok" if psi < psi_warning else ("warning" if psi < psi_alert else "alert")
    drift_by_ks = bool(ks_pvalue < ks_alpha)
    drift_detected = drift_by_ks or psi_level == "alert"

    return {
        "status": "ok",
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "psi": round(psi, 4),
        "psi_level": psi_level,
        "drift_by_ks": drift_by_ks,
        "drift_detected": drift_detected,
        "reference_count": len(reference),
        "current_count": len(current),
        "psi_warning": psi_warning,
        "psi_alert": psi_alert,
        "ks_alpha": ks_alpha,
        "message": _build_message(drift_detected, psi_level, psi_warning, psi_alert, float(ks_pvalue), ks_alpha),
    }


def _target_drift_unavailable(status: str, message: str) -> dict:
    """ターゲットドリフトが判定不能な場合の結果を組み立てる。"""
    return {
        "status": status,
        "message": message,
        "ks_statistic": None,
        "ks_pvalue": None,
        "psi": None,
        "psi_level": None,
        "drift_detected": False,
    }


# ── ターゲットドリフト（実績値。ドリフト監視の最重要指標） ────────────────────

@log_call
def detect_target_drift(
    db: Session,
    project_id: str,
    window_size: int = 100,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
    ks_alpha: float = 0.05,
) -> dict:
    """ターゲットドリフト（実績値の分布変化）を検知する。

    参照 = 学習時の正解ラベル（t_reference_logs.actual_values）、
    現在 = 実績値が連携済みの直近 window_size 件（t_inference_logs.actual_values）。
    実績値の分布変化はモデル性能の悪化に直結するため、ドリフト監視の
    最重要指標として扱う。
    """
    if settings.is_dataiku:
        return _dku_detect_target_drift(project_id, window_size, psi_warning, psi_alert, ks_alpha)

    ref_rows = (
        db.query(ReferenceLog.actual_values)
        .filter(
            ReferenceLog.project_id == project_id,
            ReferenceLog.actual_values != None,  # noqa: E711
        )
        .limit(window_size)
        .all()
    )
    # actual_values は JSON({"label": 値})のため代表スカラー値に変換する
    reference = np.array(
        [v for r in ref_rows if (v := parse_value(r[0])) is not None],
        dtype=float,
    )
    if reference.size == 0:
        return _target_drift_unavailable(
            "no_reference_actuals",
            "学習データの正解ラベル（参照ログの actual_values）が登録されていません",
        )

    cur_logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.is_error == False,  # noqa: E712
            InferenceLog.actual_values != None,  # noqa: E711
        )
        .order_by(InferenceLog.request_timestamp.desc())
        .limit(window_size)
        .all()
    )
    current = np.array(
        [v for l in cur_logs if (v := l.actual_value) is not None],
        dtype=float,
    )
    if current.size < window_size:
        return _target_drift_unavailable(
            "insufficient_data",
            f"ターゲットドリフト検知には実績値連携済みのログが {window_size} 件必要です（現在: {current.size} 件）",
        )

    return _two_sample_drift_result(reference, current, psi_warning, psi_alert, ks_alpha)


@log_call
def _dku_detect_target_drift(
    project_id: str,
    window_size: int,
    psi_warning: float,
    psi_alert: float,
    ks_alpha: float,
) -> dict:
    from ..dataiku_client import get_inference_logs_df, get_reference_logs_df

    ref_df = get_reference_logs_df(project_id)
    if not ref_df.empty:
        ref_df = ref_df[ref_df["actual_values"].notna()].head(window_size)
    if ref_df.empty:
        return _target_drift_unavailable(
            "no_reference_actuals",
            "学習データの正解ラベル（参照ログの actual_values）が登録されていません",
        )
    reference = ref_df["actual_values"].to_numpy(dtype=float)

    logs_df = get_inference_logs_df(project_id)
    if not logs_df.empty:
        logs_df = (
            logs_df[~logs_df["is_error"] & logs_df["actual_values"].notna()]
            .sort_values("request_timestamp", ascending=False)
            .head(window_size)
        )
    if len(logs_df) < window_size:
        return _target_drift_unavailable(
            "insufficient_data",
            f"ターゲットドリフト検知には実績値連携済みのログが {window_size} 件必要です（現在: {len(logs_df)} 件）",
        )
    current = logs_df["actual_values"].to_numpy(dtype=float)

    return _two_sample_drift_result(reference, current, psi_warning, psi_alert, ks_alpha)


# ── 予測値ドリフト ────────────────────────────────────────────────────────────

@log_call
def detect_drift(
    db: Session,
    project_id: str,
    window_size: int = 100,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
    ks_alpha: float = 0.05,
) -> dict:
    if settings.is_dataiku:
        return _dku_detect_drift(project_id, window_size, psi_warning, psi_alert, ks_alpha)

    logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.is_error == False,  # noqa: E712
        )
        .order_by(InferenceLog.request_timestamp)
        .all()
    )

    if len(logs) < window_size * 2:
        return {
            "status": "insufficient_data",
            "message": f"ドリフト検知には最低 {window_size * 2} 件のデータが必要です（現在: {len(logs)} 件）",
            "ks_statistic": None,
            "ks_pvalue": None,
            "psi": None,
            "psi_level": None,
            "drift_detected": False,
        }

    reference = np.array([l.prediction_value for l in logs[:window_size]])
    current = np.array([l.prediction_value for l in logs[-window_size:]])

    return _two_sample_drift_result(reference, current, psi_warning, psi_alert, ks_alpha)


# ── 予測値ドリフト PSI 推移 ──────────────────────────────────────────────────

@log_call
def _psi_trend(
    values: np.ndarray,
    timestamps: list[datetime],
    hours: int,
    window_size: int,
) -> dict:
    """選択期間内の日次 PSI 推移を計算する。

    各日の値は「その日の最終ログ時点の直近 window_size 件 vs 参照（最古
    window_size 件）」の PSI。判定 (detect_drift) と同じ定義を過去に遡って
    計算したもので、期間 (hours) はグラフの表示範囲にのみ使用する。
    判定が可能になる前（累計 window_size × 2 件未満）の日は None を返す。
    """
    n = len(values)
    if n < window_size * 2:
        return {"labels": [], "data": [], "window_size": window_size}

    reference = values[:window_size]
    since = datetime.utcnow() - timedelta(hours=hours)

    # 期間内の日付ごとに、その日の最終ログの位置（時系列昇順のインデックス）を求める
    day_last_index: dict[str, int] = {}
    for i, ts in enumerate(timestamps):
        if ts >= since:
            day_last_index[ts.strftime("%Y-%m-%d")] = i

    labels = sorted(day_last_index)
    data: list[float | None] = []
    for day in labels:
        end = day_last_index[day] + 1
        if end < window_size * 2:
            data.append(None)
        else:
            data.append(round(_compute_psi(reference, values[end - window_size:end]), 4))

    return {"labels": labels, "data": data, "window_size": window_size}


@log_call
def get_drift_over_time(
    db: Session,
    project_id: str,
    hours: int = 168,
    window_size: int = 100,
) -> dict:
    """予測値ドリフト PSI の日次推移を返す（期間は表示範囲のみに適用）。"""
    if settings.is_dataiku:
        from ..dataiku_client import get_inference_logs_df
        df = get_inference_logs_df(project_id)
        if df.empty:
            return {"labels": [], "data": [], "window_size": window_size}
        df = df[~df["is_error"]].sort_values("request_timestamp")
        values = df["prediction_values"].to_numpy(dtype=float)
        timestamps = list(df["request_timestamp"])
    else:
        logs = (
            db.query(InferenceLog)
            .filter(
                InferenceLog.project_id == project_id,
                InferenceLog.is_error == False,  # noqa: E712
            )
            .order_by(InferenceLog.request_timestamp)
            .all()
        )
        values = np.array([l.prediction_value for l in logs], dtype=float)
        timestamps = [l.request_timestamp for l in logs]

    return _psi_trend(values, timestamps, hours, window_size)


# ── 特徴量ドリフト ────────────────────────────────────────────────────────────

@log_call
def detect_feature_drift(
    db: Session,
    project_id: str,
    window_size: int = 100,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
) -> dict:
    if settings.is_dataiku:
        return _dku_detect_feature_drift(project_id, window_size, psi_warning, psi_alert)

    latest_model = (
        db.query(DeployedModel)
        .filter(DeployedModel.project_id == project_id)
        .order_by(DeployedModel.created_at.desc())
        .first()
    )

    if not latest_model:
        return {
            "status": "no_model",
            "message": "デプロイ済みモデルが登録されていません",
            "features": [],
            "drift_detected": False,
            "model_version": None,
        }

    latest_version = latest_model.model_version

    feature_importance: dict[str, float] = {}
    if latest_model.feature_importance:
        feature_importance = _loads(latest_model.feature_importance)

    ref_logs = (
        db.query(ReferenceLog)
        .filter(
            ReferenceLog.project_id == project_id,
            ReferenceLog.model_id == latest_model.model_id,
            ReferenceLog.feature_values != None,  # noqa: E711
        )
        .limit(window_size)
        .all()
    )

    ref_features: list[dict] = [_loads(r.feature_values) for r in ref_logs]

    if not ref_features:
        return {
            "status": "no_reference_features",
            "message": "参照モデルに特徴量データがありません",
            "features": [],
            "drift_detected": False,
            "model_version": latest_version,
        }

    current_logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.is_error == False,  # noqa: E712
            InferenceLog.feature_values != None,  # noqa: E711
        )
        .order_by(InferenceLog.request_timestamp.desc())
        .limit(window_size)
        .all()
    )

    if not current_logs:
        return {
            "status": "insufficient_data",
            "message": "特徴量データが含まれる推論ログがありません",
            "features": [],
            "drift_detected": False,
            "model_version": latest_version,
        }

    current_features = [_loads(l.feature_values) for l in current_logs]
    return _compute_feature_drift_result(
        ref_features, current_features, feature_importance,
        psi_warning, psi_alert, latest_version,
    )


# ── Dataiku 実装 ─────────────────────────────────────────────────────────────

@log_call
def _dku_detect_drift(
    project_id: str,
    window_size: int,
    psi_warning: float,
    psi_alert: float,
    ks_alpha: float,
) -> dict:
    from ..dataiku_client import get_inference_logs_df

    df = get_inference_logs_df(project_id)
    df = df[~df["is_error"]].sort_values("request_timestamp").reset_index(drop=True)

    n = len(df)
    if n < window_size * 2:
        return {
            "status": "insufficient_data",
            "message": f"ドリフト検知には最低 {window_size * 2} 件のデータが必要です（現在: {n} 件）",
            "ks_statistic": None,
            "ks_pvalue": None,
            "psi": None,
            "psi_level": None,
            "drift_detected": False,
        }

    reference = df["prediction_values"].iloc[:window_size].to_numpy(dtype=float)
    current = df["prediction_values"].iloc[-window_size:].to_numpy(dtype=float)

    return _two_sample_drift_result(reference, current, psi_warning, psi_alert, ks_alpha)


@log_call
def _dku_detect_feature_drift(
    project_id: str,
    window_size: int,
    psi_warning: float,
    psi_alert: float,
) -> dict:
    from ..dataiku_client import get_deployed_models_df, get_inference_logs_df, get_reference_logs_df, parse_json_col

    models_df = get_deployed_models_df(project_id)
    if models_df.empty:
        return {
            "status": "no_model",
            "message": "デプロイ済みモデルが登録されていません",
            "features": [],
            "drift_detected": False,
            "model_version": None,
        }

    models_df = models_df.sort_values("created_at", ascending=False)
    latest_row = models_df.iloc[0]
    latest_version = latest_row["model_version"]
    latest_model_id = latest_row.get("model_id")

    feature_importance: dict[str, float] = {}
    imp_raw = parse_json_col(latest_row.get("feature_importance"))
    if imp_raw:
        feature_importance = imp_raw

    ref_df = get_reference_logs_df(project_id, model_id=latest_model_id)
    ref_df = ref_df[ref_df["feature_values"].notna()].head(window_size)

    ref_features = [
        fv for fv_raw in ref_df["feature_values"]
        if (fv := parse_json_col(fv_raw)) is not None
    ]

    if not ref_features:
        return {
            "status": "no_reference_features",
            "message": "参照ログに特徴量データがありません",
            "features": [],
            "drift_detected": False,
            "model_version": latest_version,
        }

    logs_df = get_inference_logs_df(project_id)
    logs_df = (
        logs_df[~logs_df["is_error"] & logs_df["feature_values"].notna()]
        .sort_values("request_timestamp", ascending=False)
        .head(window_size)
    )

    if logs_df.empty:
        return {
            "status": "insufficient_data",
            "message": "特徴量データが含まれる推論ログがありません",
            "features": [],
            "drift_detected": False,
            "model_version": latest_version,
        }

    current_features = [
        fv for fv_raw in logs_df["feature_values"]
        if (fv := parse_json_col(fv_raw)) is not None
    ]

    return _compute_feature_drift_result(
        ref_features, current_features, feature_importance,
        psi_warning, psi_alert, latest_version,
    )


# ── 共通ロジック ─────────────────────────────────────────────────────────────

@log_call
def _compute_feature_drift_result(
    ref_features: list[dict],
    current_features: list[dict],
    feature_importance: dict[str, float],
    psi_warning: float,
    psi_alert: float,
    model_version: str | None,
) -> dict:
    feature_names = list(ref_features[0].keys())
    results = []

    for feat in feature_names:
        ref_arr = np.array(
            [f[feat] for f in ref_features if feat in f and f[feat] is not None],
            dtype=float,
        )
        cur_arr = np.array(
            [f[feat] for f in current_features if feat in f and f[feat] is not None],
            dtype=float,
        )

        if len(ref_arr) < 5 or len(cur_arr) < 5:
            continue

        psi = _compute_psi(ref_arr, cur_arr)
        importance = feature_importance.get(feat, 0.0)
        psi_level = "ok" if psi < psi_warning else ("warning" if psi < psi_alert else "alert")

        results.append({
            "feature": feat,
            "psi": round(psi, 4),
            "psi_level": psi_level,
            "importance": round(float(importance), 4),
        })

    max_psi = max((r["psi"] for r in results), default=0.0)
    return {
        "status": "ok",
        "model_version": model_version,
        "reference_count": len(ref_features),
        "current_count": len(current_features),
        "features": results,
        "drift_detected": max_psi >= psi_alert,
        "psi_warning": psi_warning,
        "psi_alert": psi_alert,
    }


@log_call
def _build_message(
    drift_detected: bool,
    psi_level: str,
    psi_warning: float,
    psi_alert: float,
    ks_pvalue: float,
    ks_alpha: float,
) -> str:
    if not drift_detected:
        return "ドリフトは検出されていません"
    parts = []
    if psi_level == "alert":
        parts.append(f"PSI > {psi_alert}: 分布が大幅に変化しています")
    elif psi_level == "warning":
        parts.append(f"PSI {psi_warning}–{psi_alert}: 軽微な分布変化を検出")
    if ks_pvalue < ks_alpha:
        parts.append(f"KS 検定: p = {ks_pvalue:.4f}（有意差あり）")
    return "　".join(parts)

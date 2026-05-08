from __future__ import annotations

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from ..models import InferenceLog


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index を計算する。"""
    eps = 1e-6
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = (exp_counts + eps) / (len(expected) + eps * bins)
    act_pct = (act_counts + eps) / (len(actual) + eps * bins)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def detect_drift(
    db: Session,
    project_id: int,
    window_size: int = 100,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
    ks_alpha: float = 0.05,
) -> dict:
    """
    参照ウィンドウ（最古 window_size 件）と現在ウィンドウ（最新 window_size 件）を
    KS 検定と PSI で比較してドリフトを検知する。
    閾値はプロジェクト設定から渡す（未設定時はデフォルト値）。
    """
    logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.is_error == False,  # noqa: E712
        )
        .order_by(InferenceLog.timestamp)
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

    reference = np.array([l.prediction for l in logs[:window_size]])
    current = np.array([l.prediction for l in logs[-window_size:]])

    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    psi = _compute_psi(reference, current)

    psi_level = "ok" if psi < psi_warning else ("warning" if psi < psi_alert else "alert")
    drift_by_ks = ks_pvalue < ks_alpha
    drift_detected = drift_by_ks or psi_level == "alert"

    return {
        "status": "ok",
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pvalue, 4),
        "psi": round(psi, 4),
        "psi_level": psi_level,
        "drift_by_ks": drift_by_ks,
        "drift_detected": drift_detected,
        "reference_count": window_size,
        "current_count": window_size,
        "psi_warning": psi_warning,
        "psi_alert": psi_alert,
        "ks_alpha": ks_alpha,
        "message": _build_message(drift_detected, psi_level, psi_warning, psi_alert, ks_pvalue, ks_alpha),
    }


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

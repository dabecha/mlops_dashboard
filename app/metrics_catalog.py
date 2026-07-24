"""評価指標カタログ。

metric_name はここで定義したリストから選択する。二値分類・回帰それぞれの
代表的な指標の計算を numpy / scipy で実装する（新規依存なし）。

各計算関数は fn(y_true, y_pred, threshold) -> float | None。
threshold は二値分類でラベル化する確率閾値（Precision / Recall で使用）。
各指標のメタ情報:
  - task          : 対象タスク種別（binary=分類 / regression=回帰）
  - higher_better : 値が大きいほど良いか（閾値の色分け方向に使用）
  - fn            : 計算関数
  - round         : 表示丸め桁数
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


# ── 二値分類（y_pred=陽性確率, y_true=0/1） ──────────────────────────────────

def _roc_auc(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None  # 片方のクラスしか無い場合は未定義
    ranks = rankdata(y_pred)  # 同順位は平均ランク
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_pos = float(y_true.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_pred)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))  # Average Precision


def _precision(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    pred_pos = np.asarray(y_pred, dtype=float) > threshold
    tp = int((pred_pos & (y_true == 1)).sum())
    fp = int((pred_pos & (y_true == 0)).sum())
    if tp + fp == 0:
        return None
    return tp / (tp + fp)


def _recall(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    pred_pos = np.asarray(y_pred, dtype=float) > threshold
    tp = int((pred_pos & (y_true == 1)).sum())
    fn = int(((~pred_pos) & (y_true == 1)).sum())
    if tp + fn == 0:
        return None
    return tp / (tp + fn)


def _logloss(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


# ── 回帰 ────────────────────────────────────────────────────────────────────

def _mae(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mape(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _r2(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


CATALOG: dict = {
    "ROC-AUC":   {"task": "binary",     "higher_better": True,  "fn": _roc_auc,   "round": 4},
    "PR-AUC":    {"task": "binary",     "higher_better": True,  "fn": _pr_auc,    "round": 4},
    "Precision": {"task": "binary",     "higher_better": True,  "fn": _precision, "round": 4},
    "Recall":    {"task": "binary",     "higher_better": True,  "fn": _recall,    "round": 4},
    "logloss":   {"task": "binary",     "higher_better": False, "fn": _logloss,   "round": 4},
    "MAE":       {"task": "regression", "higher_better": False, "fn": _mae,       "round": 3},
    "RMSE":      {"task": "regression", "higher_better": False, "fn": _rmse,      "round": 3},
    "MAPE":      {"task": "regression", "higher_better": False, "fn": _mape,      "round": 2},
    "R2":        {"task": "regression", "higher_better": True,  "fn": _r2,        "round": 4},
}

_CLASSIFICATION = [n for n, m in CATALOG.items() if m["task"] == "binary"]
_REGRESSION = [n for n, m in CATALOG.items() if m["task"] == "regression"]


def metrics_for_task(task_type: str) -> list[str]:
    """task_type に応じた選択可能な指標名リストを返す。"""
    return _REGRESSION if task_type == "regression" else _CLASSIFICATION


def default_metric(task_type: str) -> str:
    """task_type の既定指標（リストの先頭）を返す。"""
    opts = metrics_for_task(task_type)
    return opts[0]


def resolve_metric(metric_name: str | None, task_type: str) -> str:
    """metric_name が task_type で有効ならそのまま、無効なら既定指標を返す。"""
    if metric_name in metrics_for_task(task_type):
        return metric_name
    return default_metric(task_type)


def is_higher_better(metric_name: str) -> bool:
    m = CATALOG.get(metric_name)
    return m["higher_better"] if m else True


def compute_metric(metric_name: str, y_true, y_pred, threshold: float = 0.5) -> float | None:
    """指定指標を計算する。threshold は二値分類のラベル化閾値。未定義・不正時は None。"""
    m = CATALOG.get(metric_name)
    if m is None or len(y_true) == 0:
        return None
    val = m["fn"](y_true, y_pred, threshold)
    if val is None:
        return None
    return round(float(val), m["round"])

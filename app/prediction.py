"""t_inference_logs の prediction_values / actual_values（JSON: {"label": 値}）の
パース・検証ユーティリティ。

二値分類・回帰は 1 組の {label: 値}、多クラス/多ラベルは複数組を想定する。
"""
from __future__ import annotations

import json
import math

from .exceptions import PredictionFormatError

_SINGLE_PAIR_TASKS = ("binary", "regression")


def parse_value(v):
    """{"label": 値} 形式（JSON 文字列 / dict / 数値）から代表スカラー値を取り出す。

    二値分類・回帰は 1 組なのでその値。多クラス/多ラベルは最大値を代表値とする。
    None / NaN / 不正な値は None を返す。
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
    if isinstance(v, dict):
        return float(max(v.values())) if v else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def validate_pairs(field_name: str, value, task_type: str) -> None:
    """二値分類・回帰で {"label": 値} が正確に 1 組であることを検証する。

    1 組でない場合は PredictionFormatError を送出する（多クラス/多ラベルは対象外）。
    value は dict または JSON 文字列。None は検証しない。
    """
    if value is None:
        return
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            raise PredictionFormatError(field_name, task_type, 0)
    if not isinstance(value, dict):
        raise PredictionFormatError(field_name, task_type, 0)
    if task_type in _SINGLE_PAIR_TASKS and len(value) != 1:
        raise PredictionFormatError(field_name, task_type, len(value))

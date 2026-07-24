"""表示用のフォーマットユーティリティ。"""
from __future__ import annotations


def period_label(hours) -> str:
    """時間数を「直近N日 / 週 / カ月」等の期間ラベルに整形する。"""
    if hours is None:
        return ""
    hours = int(hours)
    if hours >= 720 and hours % 720 == 0:      # 1カ月 = 30日 = 720時間
        return f"直近{hours // 720}カ月"
    if hours >= 168 and hours % 168 == 0:      # 1週 = 7日 = 168時間
        return f"直近{hours // 168}週"
    if hours >= 24 and hours % 24 == 0:
        return f"直近{hours // 24}日"
    return f"直近{hours}時間"


def format_duration(ms: float | None) -> str:
    """ミリ秒を days / hour / min / sec / ms の単位で読みやすく整形する。

    例:
        140.2      -> "140.2 ms"
        2500       -> "2 sec 500 ms"
        65000      -> "1 min 5 sec"
        3_661_500  -> "1 h 1 min 1 sec 500 ms"
        90_061_000 -> "1 d 1 h 1 min 1 sec"
    None は "—" を返す。
    """
    if ms is None:
        return "—"
    ms = max(0.0, float(ms))

    # 1 秒未満は ms（小数1桁）
    if ms < 1000:
        return f"{ms:.1f} ms"

    total = int(round(ms))
    days, rem = divmod(total, 86_400_000)
    hours, rem = divmod(rem, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)

    parts: list[str] = []
    if days:
        parts.append(f"{days} d")
    if hours:
        parts.append(f"{hours} h")
    if minutes:
        parts.append(f"{minutes} min")
    if seconds:
        parts.append(f"{seconds} sec")
    if millis:
        parts.append(f"{millis} ms")
    return " ".join(parts)

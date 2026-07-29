"""バックエンド関数の呼び出しログ用ユーティリティ。

各関数に @log_call を付与すると、呼び出し時にファイル名・関数名・入力データ
（位置引数 / キーワード引数）を DEBUG レベルで出力する。
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import os
from logging import StreamHandler, getLogger

from .settings import settings

# ── ロガー設定（共通・ハンドラ二重登録を防止） ────────────────────────────────
# レベルは .env の LOG_LEVEL で設定する（デフォルト: DEBUG）
logger = getLogger("mlops_dashboard.backend")
logger.setLevel(settings.log_level)
if not logger.handlers:
    _handler = StreamHandler()
    _handler.setLevel(settings.log_level)
    logger.addHandler(_handler)
logger.propagate = False


def _emit(filename: str, qualname: str, args: tuple, kwargs: dict) -> None:
    logger.debug("[%s:%s] args=%r kwargs=%r", filename, qualname, args, kwargs)


def log_call(func):
    """関数呼び出し時にファイル名・関数名・入力データを DEBUG ログ出力するデコレータ。

    通常関数・async 関数・ジェネレータ関数（get_db 等）のいずれにも対応する。
    """
    filename = os.path.basename(func.__code__.co_filename)
    qualname = func.__qualname__

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _emit(filename, qualname, args, kwargs)
            return await func(*args, **kwargs)
        return async_wrapper

    if inspect.isgeneratorfunction(func):
        @functools.wraps(func)
        def gen_wrapper(*args, **kwargs):
            _emit(filename, qualname, args, kwargs)
            yield from func(*args, **kwargs)
        return gen_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _emit(filename, qualname, args, kwargs)
        return func(*args, **kwargs)
    return wrapper

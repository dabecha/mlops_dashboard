"""アプリケーション共通の業務例外。

各例外は以下を保持する:
  - user_message : ユーザー向け（日本語）メッセージ。API では JSON の detail、
                   UI では画面表示に使われる。内部情報は含めない。
  - log_message  : 内部ログ用の詳細メッセージ（任意。無ければ user_message）。
  - status_code  : HTTP ステータスコード。

グローバル例外ハンドラ（main.py）がこれらを捕捉し、
/api/* は JSON、UI/htmx は HTML でユーザーに返す。
"""
from __future__ import annotations


class AppError(Exception):
    """業務例外の基底クラス。"""

    status_code: int = 500
    default_user_message: str = "予期しないエラーが発生しました"

    def __init__(self, user_message: str | None = None, *, log_message: str | None = None):
        self.user_message = user_message or self.default_user_message
        self.log_message = log_message or self.user_message
        super().__init__(self.log_message)


class NotFoundError(AppError):
    """対象リソースが存在しない（404）。"""

    status_code = 404
    default_user_message = "対象が見つかりません"


class ValidationError(AppError):
    """入力内容が不正、または前提条件を満たさない（400）。"""

    status_code = 400
    default_user_message = "入力内容が正しくありません"


class PredictionFormatError(ValidationError):
    """予測値/正解値の {label: 値} 形式が想定と異なる（二値分類・回帰は 1 組）。"""

    def __init__(self, field_name: str, task_type: str, count: int, *, log_message: str | None = None):
        self.field_name = field_name
        self.task_type = task_type
        self.count = count
        user_message = (
            f"{field_name} は {task_type} では単一の {{ラベル: 値}} 形式である必要があります"
            f"（検出: {count} 組）。"
        )
        super().__init__(
            user_message,
            log_message=log_message or f"{field_name} (task_type={task_type}) は {count} 組（1 組を期待）",
        )


class DataSourceError(AppError):
    """データソース（Dataiku 等）へのアクセスに失敗した（503）。"""

    status_code = 503
    default_user_message = "データソースへのアクセスに失敗しました。時間をおいて再度お試しください。"


class DataIntegrityError(AppError):
    """データの形式・整合性に問題がある（422）。

    例: JSON カラムのパース失敗、データセットに必須カラムが存在しない。
    """

    status_code = 422
    default_user_message = "データの形式が正しくありません"


class MissingDatasetError(DataSourceError):
    """Ops プロジェクトに必須データセット（テーブル）が存在しない（作成が必要）。"""

    status_code = 422

    def __init__(self, missing: list[str], project_key: str, *, log_message: str | None = None):
        self.missing = list(missing)
        self.project_key = project_key
        tables = "、".join(missing)
        user_message = (
            f"プロジェクト '{project_key}' に必要なテーブルがありません: {tables}。"
            "Dataiku プロジェクトに該当データセットを作成してください。"
        )
        super().__init__(
            user_message,
            log_message=log_message or f"プロジェクト '{project_key}' に必須テーブルが不足: {self.missing}",
        )


class MissingColumnError(DataIntegrityError):
    """テーブル（データセット）のカラムが定義と異なる（必須カラムが不足）。"""

    def __init__(self, missing: list[str], dataset_name: str, *, log_message: str | None = None):
        self.missing = list(missing)
        self.dataset_name = dataset_name
        cols = "、".join(missing)
        user_message = (
            f"テーブル '{dataset_name}' のカラムが定義と異なります。"
            f"不足しているカラム: {cols}。"
        )
        super().__init__(
            user_message,
            log_message=log_message or f"テーブル '{dataset_name}' の不足カラム: {self.missing}",
        )


class ConfigurationError(AppError):
    """サーバーの設定・実行環境に不備がある（500）。

    例: Dataiku モードで pandas / dataiku 未導入、APP_MODE 不正。
    """

    status_code = 500
    default_user_message = "サーバー設定に問題があります。管理者にお問い合わせください。"

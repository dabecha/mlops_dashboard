# MLOps Dashboard

複数の ML プロジェクトから推論リクエスト・結果を受け取り、リクエスト数・エラー数・応答時間分布・精度・ドリフト検知をリアルタイムで可視化する MLOps モニタリングダッシュボード。

**技術スタック:** Python / FastAPI + htmx + SQLite (SQLAlchemy) + Chart.js

---

## セットアップ

```bash
# 依存パッケージのインストール（uv が必要）
uv sync

# デモデータの生成（初回のみ）
uv run python seed_data.py

# サーバー起動
uv run uvicorn main:app --reload
```

ブラウザで http://localhost:8000 にアクセスしてください。

- **サマリーページ** (`/`) — 全プロジェクトを一覧比較（トップページ）
- **詳細ページ** (`/detail`) — プロジェクトを選択してメトリクスを詳細確認
- **API ドキュメント** (`/docs`) — Swagger UI

---

## データベース構造 (mlops.db)

SQLite ファイル。アプリ起動時に自動生成されます（`mlops.db`）。

### テーブル一覧

| テーブル | 種別 | 説明 |
|---|---|---|
| `m_projects` | マスタ | ML プロジェクト |
| `project_configs` | 設定 | ML プロジェクト閾値設定 |
| `t_deployed_models` | マスタ | デプロイ済みモデル |
| `t_inference_logs` | トランザクション | 推論リクエスト・結果ログ |
| `t_reference_logs` | トランザクション | 学習データ・特徴量ドリフト参照データ |

---

### `m_projects` テーブル

ML プロジェクトのマスターテーブル。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `project_id` | VARCHAR(100) | NO | auto | 主キー |
| `project_name` | VARCHAR(100) | NO | — | プロジェクト名（ユニーク）。Dataiku モードでは Dataiku プロジェクトキーとして使用 |
| `description` | VARCHAR(500) | YES | NULL | 説明文 |
| `task_type` | VARCHAR(20) | NO | `binary` | タスク種別: `binary` / `multi-class` / `multi-label` / `regression` |
| `created_at` | DATETIME | NO | 現在時刻 | 登録日時（日本時間） |

---

### `project_configs` テーブル

ML プロジェクトごとの閾値設定。未設定の場合はデフォルト値を使用。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `id` | INTEGER | NO | auto | 主キー |
| `project_id` | VARCHAR(100) | NO | — | `m_projects.project_id` への外部キー（ユニーク） |
| `drift_window_size` | INTEGER | NO | `100` | ドリフト検知に使う直近サンプル数 |
| `psi_warning` | REAL | NO | `0.10` | PSI 警告閾値 |
| `psi_alert` | REAL | NO | `0.25` | PSI 異常閾値 |
| `ks_alpha` | REAL | NO | `0.05` | KS 検定の有意水準 |
| `metric_name` | VARCHAR(50) | NO | `Accuracy` | 評価指標名（プロジェクト側で設定: ROC-AUC / logloss / MAE / RMSE 等） |
| `metric_warning` | REAL | NO | `75.0` | 評価指標の警告閾値（回帰・分類共通） |
| `metric_alert` | REAL | NO | `60.0` | 評価指標の異常閾値（回帰・分類共通） |
| `updated_at` | DATETIME | NO | 現在時刻 | 最終更新日時（日本時間） |

---

### `t_deployed_models` テーブル

デプロイ済みモデルの記録。モデルバージョンや特徴量メタ情報を管理する。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `model_id` | VARCHAR(100) | NO | auto | 主キー |
| `project_id` | VARCHAR(100) | NO | — | `m_projects.project_id` への外部キー |
| `model_version` | VARCHAR(100) | YES | NULL | モデルバージョン識別子 |
| `feature_dtypes` | TEXT | YES | NULL | 特徴量データ型 JSON（例: `{"age": "float32"}`） |
| `feature_importance` | TEXT | YES | NULL | 特徴量重要度 JSON（例: `{"age": 0.43, "amount": 0.57}`） |
| `is_activate` | BOOLEAN | NO | — | モデル有効化フラグ |
| `created_at` | DATETIME | NO | 現在時刻 | 登録日時（日本時間） |

---

### `t_reference_logs` テーブル

デプロイ済みモデル学習データ。1 サンプル = 1 行で学習データを蓄積し、特徴量ドリフトの参照分布として使用。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `log_id` | VARCHAR(100) | NO | auto | 主キー |
| `project_id` | VARCHAR(100) | NO | — | `m_projects.project_id` への外部キー |
| `model_id` | VARCHAR(100) | YES | NULL | `t_deployed_models.model_id` への外部キー |
| `feature_values` | TEXT | YES | NULL | 学習サンプルの特徴量 JSON（例: `{"age": 35.0, "amount": 5200.0}`） |
| `actual_values` | REAL | YES | NULL | 学習時の正解ラベル |

---

### `t_inference_logs` テーブル

ML モデルの推論リクエスト・結果ログ。1 リクエスト = 1 レコード。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `log_id` | VARCHAR(100) | NO | auto | 主キー |
| `project_id` | VARCHAR(100) | NO | — | `m_projects.project_id` への外部キー |
| `batch_log_id` | VARCHAR(100) | NO | — | 推論単位の識別子。バッチ推論は複数 `log_id` が共有、API 推論は 1 件 |
| `request_timestamp` | DATETIME | NO | 現在時刻 | 推論リクエスト受付日時（日本時間） |
| `model_id` | VARCHAR(100) | YES | NULL | `t_deployed_models.model_id` への外部キー |
| `prediction_values` | TEXT | NO | — | 予測値 JSON `{"label": 値}`（二値/回帰は1組、多クラス/多ラベルは複数組） |
| `actual_values` | TEXT | YES | NULL | 正解値 JSON `{"label": 値}`（遅延ラベリングで後から投入可） |
| `is_error` | BOOLEAN | NO | `false` | エラー発生フラグ |
| `feature_values` | TEXT | YES | NULL | 入力特徴量 JSON（例: `{"age": 35.0, "amount": 5200.0}`） |
| `feature_dtypes` | TEXT | YES | NULL | 入力特徴量データ型 JSON |
| `created_at` | DATETIME | NO | 現在時刻 | 登録日時（日本時間） |
| `updated_at` | DATETIME | NO | 現在時刻 | 更新（推論完了）日時。`request_timestamp` との差が応答時間 |

---

### ER 図

```
 m_projects
 ─────────────────────────────
 PK project_id    VARCHAR(100)
    project_name  VARCHAR(100)  UNIQUE
    description   VARCHAR(500)
    task_type     VARCHAR(20)   -- binary | multi-class | multi-label | regression
    created_at    DATETIME
    │
    │ 1:1            │ 1:N                     │ 1:N                    │ 1:N
    ▼                ▼                         ▼                        ▼
 project_configs   t_reference_logs       t_inference_logs         t_deployed_models
 ────────────────  ──────────────────     ──────────────────────── ────────────────────────────
    id       INT PK  PK log_id  VARCHAR(100)  PK log_id  VARCHAR(100)  PK model_id    VARCHAR(100)
 FK project_id VARCHAR(100) UNIQ FK project_id VARCHAR(100) FK project_id VARCHAR(100) FK project_id VARCHAR(100)
    drift_window_size FK model_id VARCHAR(100) → batch_log_id               model_version      VARCHAR(100)
    psi_warning       feature_values TEXT   request_timestamp          feature_dtypes     TEXT  -- JSON
    psi_alert         actual_values  REAL                              feature_importance TEXT  -- JSON
    ks_alpha                            FK model_id VARCHAR(100) →      is_activate        BOOLEAN
    metric_name                         prediction_values              created_at         DATETIME
    metric_warning/alert                actual_values
    updated_at                          is_error
                                        feature_values
                                        feature_dtypes
                                        created_at
                                        updated_at
```

---

## データ収集 API

各 ML プロジェクトは以下の REST API でデータを送信します。

### プロジェクト登録

```bash
POST /api/projects
Content-Type: application/json

{
  "project_name": "fraud-detection",
  "description": "クレジットカード不正検知モデル",
  "task_type": "binary"   # "binary" | "multi-class" | "multi-label" | "regression"
}
```

### デプロイ済みモデル登録

```bash
POST /api/projects/{project_id}/models
Content-Type: application/json

{
  "project_id": "fraud-detection",
  "model_version": "v1.2.0",
  "feature_dtypes": {"age": "float32", "amount": "float32"},  # 任意
  "feature_importance": {"age": 0.43, "amount": 0.57},        # 任意: 特徴量重要度
  "is_activate": true                                          # 任意: 有効化フラグ（デフォルト: true）
}
```

### 参照ログ登録（特徴量ドリフトの参照データ）

```bash
POST /api/projects/{project_id}/reference-logs
Content-Type: application/json

# 学習データの1サンプルを1リクエストで送信。複数回呼び出すことで参照分布を構築する。
{
  "model_id": "v1.2.0",                                   # 任意: t_deployed_models.model_id
  "feature_values": {"age": 35.0, "amount": 5200.0},     # 任意: 学習サンプル特徴量
  "actual_values": 1.0                                    # 任意: 正解ラベル
}
```

### 推論ログ送信

```bash
POST /api/infer
Content-Type: application/json

{
  "project_name": "fraud-detection",         # 必須
  "batch_log_id": "batch-00001",             # 必須: 推論単位の識別子（API 推論は 1 件）
  "prediction_values": {"label": 0.82},      # 必須: {"label": 値}（二値/回帰は1組）
  "model_id": "v1.2.0",                      # 任意: t_deployed_models.model_id
  "actual_values": {"label": 1.0},           # 任意: {"label": 値}（遅延ラベリング対応）
  "is_error": false,                         # 任意: エラーフラグ
  "request_timestamp": "2026-05-08T12:00:00", # 任意: 省略時はサーバー時刻
  "feature_values": {"age": 35.0, "amount": 5200.0},  # 任意: 入力特徴量
  "feature_dtypes": {"age": "float32", "amount": "float32"}  # 任意
}
# 応答時間は保存せず、request_timestamp と updated_at（推論完了時刻）の差から
# batch_log_id 単位で算出される。
```

### その他のエンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/api/projects` | プロジェクト一覧取得 |
| DELETE | `/api/projects/{project_id}` | プロジェクト削除（ログも連鎖削除） |
| GET | `/api/projects/{project_id}/config` | 閾値設定取得 |
| PUT | `/api/projects/{project_id}/config` | 閾値設定更新 |

---

## ダッシュボード機能

| 機能 | 詳細 |
|---|---|
| リクエスト数・エラー数 | 選択期間（1h / 6h / 24h / 7日）で集計 |
| 応答時間 | 平均・P50 / P95 / P99・ヒストグラム分布 |
| 精度推移 | 日次グラフ（分類: Accuracy %、回帰: MAE） |
| ドリフト検知 | PSI + KS 検定（参照ウィンドウ vs 最新ウィンドウ 各 100 件） |
| サマリーページ | 全プロジェクトを一覧比較、アラートを強調表示 |
| 自動更新 | 詳細: 30 秒、サマリー: 60 秒 |

### ドリフト検知の判定基準

| 指標 | 正常 | 警告 | 異常 |
|---|---|---|---|
| PSI | < 0.1 | 0.1 – 0.25 | > 0.25 |
| KS 検定 p 値 | ≥ 0.05 | — | < 0.05 |

どちらか一方でも異常を示した場合、「ドリフト検出」と判定します。

---

## ディレクトリ構成

```
mlops_dashboard/
├── main.py                              # FastAPI エントリポイント・起動設定
├── seed_data.py                         # デモデータ生成スクリプト
├── pyproject.toml                       # 依存パッケージ (uv)
├── .env.example                         # 環境変数設定例
├── app/
│   ├── settings.py                      # APP_MODE 等の環境変数設定
│   ├── database.py                      # SQLite + SQLAlchemy セットアップ
│   ├── dataiku_client.py                # Dataiku DSS データプロバイダー
│   ├── models.py                        # ORM モデル定義（全テーブル）
│   ├── schemas.py                       # Pydantic スキーマ
│   ├── routers/
│   │   ├── ingest.py                    # データ収集 API
│   │   └── ui.py                        # UI ページ + htmx パーシャル
│   ├── services/
│   │   ├── metrics.py                   # 集計ロジック
│   │   ├── drift.py                     # ドリフト検知（PSI / KS 検定）
│   │   └── config.py                    # 閾値設定管理
│   ├── templates/
│   │   ├── base.html                    # 共通レイアウト
│   │   ├── index.html                   # 詳細ページ
│   │   ├── summary.html                 # サマリーページ
│   │   ├── manage.html                  # プロジェクト管理ページ
│   │   └── partials/
│   │       ├── all_panels.html          # 詳細パネル群（htmx）
│   │       └── summary_panels.html      # サマリーテーブル（htmx）
│   └── static/js/
│       ├── htmx.min.js                  # htmx 1.9.12
│       └── chart.umd.min.js             # Chart.js 4.4.2
└── mlops.db                             # SQLite DB（.gitignore 対象）
```

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

- **詳細ページ** (`/`) — プロジェクトを選択してメトリクスを詳細確認
- **サマリーページ** (`/summary`) — 全プロジェクトを一覧比較
- **API ドキュメント** (`/docs`) — Swagger UI

---

## データベース構造 (mlops.db)

SQLite ファイル。アプリ起動時に自動生成されます（`mlops.db`）。

### テーブル一覧

```
projects
inference_logs
```

---

### `projects` テーブル

ML プロジェクトのマスターテーブル。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `id` | INTEGER | NO | auto | 主キー |
| `name` | VARCHAR(100) | NO | — | プロジェクト名（ユニーク） |
| `description` | VARCHAR(500) | YES | NULL | 説明文 |
| `task_type` | VARCHAR(20) | NO | `classification` | タスク種別。`classification` または `regression` |
| `created_at` | DATETIME | NO | 現在時刻 | 登録日時（UTC） |

**インデックス:** `id`（主キー）、`name`（ユニーク）

---

### `inference_logs` テーブル

ML モデルの推論リクエスト・結果ログ。1 リクエストにつき 1 レコードを記録します。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `id` | INTEGER | NO | auto | 主キー |
| `project_id` | INTEGER | NO | — | `projects.id` への外部キー |
| `timestamp` | DATETIME | NO | 現在時刻 | 推論実行日時（UTC） |
| `request_id` | VARCHAR(100) | YES | NULL | 呼び出し元が付与するリクエスト識別子 |
| `prediction` | REAL | NO | — | モデルの予測値（分類: 確率 0–1、回帰: 数値） |
| `actual_label` | REAL | YES | NULL | 正解ラベル。遅延ラベリングで後から投入可 |
| `confidence` | REAL | YES | NULL | モデルの予測信頼度（0–1） |
| `response_time_ms` | REAL | NO | — | 推論にかかった応答時間（ミリ秒） |
| `is_error` | BOOLEAN | NO | `false` | エラー発生フラグ |
| `error_message` | TEXT | YES | NULL | エラー内容（`is_error=true` のときのみ使用） |
| `feature_values` | TEXT | YES | NULL | 入力特徴量の JSON 文字列（例: `{"age": 35.0, "amount": 5200.0}`） |

**インデックス:** `id`（主キー）、`project_id`、`timestamp`

**リレーション:** `project_id` → `projects.id`（カスケード削除）

---

### ER 図

```
projects
─────────────────────────────
PK  id          INTEGER
    name        VARCHAR(100)  UNIQUE
    description VARCHAR(500)
    task_type   VARCHAR(20)   -- 'classification' | 'regression'
    created_at  DATETIME
         │
         │ 1 : N
         ▼
inference_logs
─────────────────────────────
PK  id               INTEGER
FK  project_id       INTEGER   → projects.id
    timestamp        DATETIME
    request_id       VARCHAR(100)
    prediction       REAL        -- 必須
    actual_label     REAL        -- 任意（遅延ラベリング対応）
    confidence       REAL        -- 任意
    response_time_ms REAL        -- 必須
    is_error         BOOLEAN
    error_message    TEXT
    feature_values   TEXT        -- JSON
```

---

## データ収集 API

各 ML プロジェクトは以下の REST API でデータを送信します。

### プロジェクト登録

```bash
POST /api/projects
Content-Type: application/json

{
  "name": "fraud-detection",
  "description": "クレジットカード不正検知モデル",
  "task_type": "classification"   # "classification" | "regression"
}
```

### 推論ログ送信

```bash
POST /api/infer
Content-Type: application/json

{
  "project_name": "fraud-detection",    # 必須
  "prediction": 0.82,                   # 必須: モデル出力値
  "response_time_ms": 130.5,            # 必須: 応答時間 (ms)
  "request_id": "req-00001",            # 任意
  "actual_label": 1.0,                  # 任意: 正解ラベル
  "confidence": 0.91,                   # 任意: 予測信頼度
  "is_error": false,                    # 任意: エラーフラグ
  "error_message": null,                # 任意: エラー内容
  "timestamp": "2026-05-08T12:00:00",  # 任意: 省略時はサーバー時刻
  "feature_values": {                   # 任意: 入力特徴量
    "age": 35.0,
    "amount": 5200.0
  }
}
```

### その他のエンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/api/projects` | プロジェクト一覧取得 |
| DELETE | `/api/projects/{id}` | プロジェクト削除（ログも連鎖削除） |

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
├── main.py                              # FastAPI エントリポイント
├── seed_data.py                         # デモデータ生成スクリプト
├── pyproject.toml                       # 依存パッケージ (uv)
├── app/
│   ├── database.py                      # SQLite + SQLAlchemy セットアップ
│   ├── models.py                        # ORM モデル定義
│   ├── schemas.py                       # Pydantic スキーマ
│   ├── routers/
│   │   ├── ingest.py                    # データ収集 API
│   │   └── ui.py                        # UI ページ + htmx パーシャル
│   ├── services/
│   │   ├── metrics.py                   # 集計ロジック
│   │   └── drift.py                     # ドリフト検知（PSI / KS 検定）
│   ├── templates/
│   │   ├── base.html                    # 共通レイアウト
│   │   ├── index.html                   # 詳細ページ
│   │   ├── summary.html                 # サマリーページ
│   │   └── partials/
│   │       ├── all_panels.html          # 詳細パネル群（htmx）
│   │       └── summary_panels.html      # サマリーテーブル（htmx）
│   └── static/js/
│       ├── htmx.min.js                  # htmx 1.9.12
│       └── chart.umd.min.js             # Chart.js 4.4.2
└── mlops.db                             # SQLite DB（.gitignore 対象）
```

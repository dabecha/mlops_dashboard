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
| `m_project_configs` | 設定 | ML プロジェクト閾値設定 |
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

### `m_project_configs` テーブル

ML プロジェクトごとの閾値設定。未設定の場合はデフォルト値を使用。
dev / production モードでは、`m_projects` と同じ Dataiku 管理プロジェクトに配備された `m_project_configs` データセットから取得・保存する。

| カラム | 型 | NULL | デフォルト | 説明 |
|---|---|---|---|---|
| `id` | INTEGER | NO | auto | 主キー |
| `project_id` | VARCHAR(100) | NO | — | `m_projects.project_id` への外部キー（ユニーク） |
| `model_id` | VARCHAR(100) | YES | NULL | `t_deployed_models.model_id` への外部キー |
| `metric_name` | VARCHAR(50) | NO | `ROC-AUC` | 評価指標名（種別ごとの定義リストから選択。分類: ROC-AUC / PR-AUC / Precision / Recall / logloss、回帰: MAE / RMSE / MAPE / R2） |
| `metric_warning` | REAL | NO | `75.0` | 評価指標の警告閾値（回帰・分類共通） |
| `metric_alert` | REAL | NO | `60.0` | 評価指標の異常閾値（回帰・分類共通） |
| `is_higher_better` | BOOLEAN | NO | `true` | 高いほど良い指標を表すフラグ。True の場合は指標が `metric_warning` / `metric_alert` を下回ると警告 / 異常、False の場合は上回ると警告 / 異常 |
| `drift_window_size` | INTEGER | NO | `100` | ドリフト検知に使う直近サンプル数 |
| `psi_warning` | REAL | NO | `0.10` | PSI 警告閾値 |
| `psi_alert` | REAL | NO | `0.25` | PSI 異常閾値 |
| `ks_alpha` | REAL | NO | `0.05` | KS 検定の有意水準 |
| `classification_threshold` | REAL | NO | `0.5` | 二値分類でラベル化する確率閾値（Precision / Recall で使用） |
| `created_at` | DATETIME | NO | 現在時刻 | 登録日時（日本時間） |
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
| `actual_values` | TEXT | YES | NULL | 学習時の正解ラベル JSON（例: 二値分類 `{"value": 0.8}`、多値分類 `{"cat": 0.4, "dog": 0.5, "pig": 0.1}`）。ターゲットドリフトの参照分布として使用 |

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
 m_project_configs t_reference_logs       t_inference_logs         t_deployed_models
 ────────────────  ──────────────────     ──────────────────────── ────────────────────────────
    id       INT PK  PK log_id  VARCHAR(100)  PK log_id  VARCHAR(100)  PK model_id    VARCHAR(100)
 FK project_id VARCHAR(100) UNIQ FK project_id VARCHAR(100) FK project_id VARCHAR(100) FK project_id VARCHAR(100)
 FK model_id VARCHAR(100) FK model_id VARCHAR(100) → batch_log_id           model_version      VARCHAR(100)
    drift_window_size feature_values TEXT   request_timestamp          feature_dtypes     TEXT  -- JSON
    psi_warning       actual_values  TEXT -- JSON                      feature_importance TEXT  -- JSON
    psi_alert                           FK model_id VARCHAR(100) →      is_activate        BOOLEAN
    ks_alpha                            prediction_values              created_at         DATETIME
    metric_name                         actual_values
    metric_warning/alert                is_error
    is_higher_better
    created_at/updated_at
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
  "project_id": "FRAUD_DETECTION",   # dev/production では必須（Dataiku プロジェクトキーと一致させる）。local_dev では省略可（自動採番）
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
  "actual_values": {"label": 1.0}                         # 任意: 正解ラベル {"label": 値}（二値/回帰は1組）
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
| 予測件数・エラー数 | 選択期間（From / To の日付で指定。デフォルト: To = 今日、From = 1カ月前）で集計。実績値（actual_values）の連携件数・連携率も併記 |
| 応答時間 | 平均・P50 / P95 / P99・ヒストグラム分布 |
| 精度推移 | 日次グラフ（設定した評価指標で計算） |
| ドリフト検知 | ターゲット（最重要）/ 予測値 / 特徴量の3種を監視（詳細は「[ドリフト検知](#ドリフト検知)」章を参照） |
| サマリーページ | 全プロジェクトを一覧比較、アラートを強調表示 |
| 自動更新 | 初期値は詳細: 30 秒、サマリー: 60 秒。間隔は管理ページの「表示設定」からページ別に変更可（オフ / 10秒 / 30秒 / 1分 / 5分 / 10分）。設定はブラウザに保存される。`.env` の `AUTO_REFRESH_ENABLED=false` で全体を無効化、`AUTO_REFRESH_SUMMARY_SECONDS` / `AUTO_REFRESH_DETAIL_SECONDS` で初期値を変更 |

---

## ドリフト検知

モデルの入出力・実績の分布変化を 3 種類のドリフトで監視する。**ターゲットドリフトが最重要指標**であり、実績値の分布変化はモデル性能の悪化に直結するため、ダッシュボードでも最優先で表示される。

### 3 種類のドリフトと使用する値

| 種類 | 学習側（参照分布） | 予測運用側（現在分布） | 検定 | 位置づけ |
|---|---|---|---|---|
| **ターゲットドリフト** | 学習時の正解ラベル<br>`t_reference_logs.actual_values`（最大 `drift_window_size` 件） | 実績値が連携済みの直近 `drift_window_size` 件<br>`t_inference_logs.actual_values` | PSI + KS | **最重要** |
| 予測値ドリフト | 全履歴の**最古** `drift_window_size` 件の予測値<br>`t_inference_logs.prediction_values` | 直近 `drift_window_size` 件の予測値<br>`t_inference_logs.prediction_values` | PSI + KS | 補助指標 |
| 特徴量ドリフト | 最新デプロイモデルの学習データ特徴量<br>`t_reference_logs.feature_values`（最大 `drift_window_size` 件） | 直近 `drift_window_size` 件の入力特徴量<br>`t_inference_logs.feature_values` | 特徴量ごとの PSI | 補助指標 |

- エラーログ（`is_error = true`）は全ドリフトの集計から除外される。
- 予測値ドリフトのみ参照が「学習側」ではなく「ログ収集開始直後の運用データ」である点に注意（後述の注意点参照）。

### 判定基準

| 指標 | 正常 | 警告 | 異常 |
|---|---|---|---|
| PSI | < `psi_warning` (0.1) | `psi_warning` – `psi_alert` (0.1 – 0.25) | > `psi_alert` (0.25) |
| KS 検定 p 値 | ≥ `ks_alpha` (0.05) | — | < `ks_alpha` (0.05) |

- ターゲット / 予測値: PSI が異常、または KS 検定で有意差ありのどちらか一方でも該当すれば「ドリフト検出」。
- 特徴量: 特徴量ごとに PSI を計算し、最大 PSI が `psi_alert` 以上で「ドリフト検出」（KS 検定は行わない）。特徴量重要度と組み合わせた散布図で影響度を確認できる。
- PSI は等幅 10 ビンのヒストグラム比較。空ビンによる発散を防ぐため各ビンに擬似カウント 0.5 を加える（Laplace 平滑化）。

### 関連パラメタ（`m_project_configs`）

すべてプロジェクトごとに設定でき、画面の ⚙（閾値設定）から変更できる。

| パラメタ | デフォルト | 適用範囲 |
|---|---|---|
| `drift_window_size` | 100 | 全ドリフトのウィンドウ件数（参照・現在の各サンプル数） |
| `psi_warning` | 0.10 | 全ドリフトの PSI 警告閾値 |
| `psi_alert` | 0.25 | 全ドリフトの PSI 異常閾値 |
| `ks_alpha` | 0.05 | ターゲット / 予測値の KS 検定有意水準 |

### 期間選択との関係

- **ドリフト判定はすべて画面の期間選択（From / To の日付指定）と独立**。判定は常に「直近 `drift_window_size` 件」ベースのスナップショットであり、期間を切り替えても判定結果は変わらない（リクエスト集計・精度が期間連動なのと対照的）。
- 例外は詳細ページの「PSI 推移」グラフ（予測値ドリフトのみ）。各日時点の判定 PSI を日次で遡って表示するもので、期間選択は**表示範囲**にのみ適用される（各日の値自体は期間に依存しない）。

### 注意点・制約

**ターゲットドリフト**

- 参照ログに学習時の正解ラベル（`actual_values`）が登録されていないと判定できない（「参照なし」表示）。
- 実績値の連携が `drift_window_size` 件たまるまで判定できない（「データ不足」表示）。連携の進捗はサマリー / 詳細の「実績値連携」で確認できる。
- 遅延ラベリングのため「直近 N 件の実績値」は古いリクエストに対するものになりうる。実績の連携遅延が大きいほど、検知は実世界の変化より遅れる。
- 実績値が連携されるサンプルに偏りがある場合（例: 疑わしい取引だけ調査してラベル付けする運用）、分布が歪んで誤検知・見逃しの原因になる。

**予測値ドリフト**

- 累計ログが `drift_window_size × 2` 件たまるまで判定できない（参照と現在のウィンドウが重ならないようにするため）。PSI 推移グラフでも判定可能になる前の日は欠落として表示される。
- 参照は「ログ収集開始直後の最古 N 件」であり、**学習時の分布ではない**。収集開始時点で既にドリフトしていた場合、それが基準になってしまう。
- モデルを再デプロイして予測分布が正当に変わった場合も参照は旧モデル時代のままのため、誤検知となりうる。再デプロイ後はログの整理（または参照の見直し）を検討すること。

**特徴量ドリフト**

- 参照は `created_at` が最新のデプロイモデルに紐づく参照ログ（`model_id` が一致するもの）のみを使用する。`model_id` 未設定の参照ログは対象外。
- 参照・現在の各サンプルが 5 件未満の特徴量はスキップされる。
- **カテゴリ特徴量（数値に変換できない値）は PSI 算出の対象外**。対象外となった特徴量は画面に明記される。NaN / inf の値は計算から除外される。
- `feature_importance` 未登録の特徴量は重要度 0 として表示される。

**共通**

- ウィンドウ 100 件程度では PSI にサンプリング揺らぎがあり、警告閾値（0.1）近傍は上下しうる。閾値を厳しくしすぎると誤検知が増える。
- KS 検定は連続分布向けの検定であり、二値ラベルのような離散値では近似的な扱いになる。またウィンドウを大きくすると実務上僅かな差でも有意になりやすい。
- 多クラス / 多ラベルでは `{label: 値}` の**最大値を代表スカラーに変換**して比較するため、分布監視としては限定的（二値分類・回帰が主対象の設計）。
- ドリフト検出は「分布が変化した」ことを示すもので、性能悪化を直接証明するものではない。精度推移と合わせて判断すること。

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

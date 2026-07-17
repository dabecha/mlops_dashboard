"""
デモデータ生成スクリプト

実行方法:
    uv run python seed_data.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from app.database import Base, SessionLocal, engine
from app.models import DeployedModel, InferenceLog, Project, ReferenceLog

random.seed(42)

try:
    import numpy as np
    np.random.seed(42)
except ImportError:
    pass


def make_reference_log_samples(
    project_id: int,
    model_id: int,
    feature_names: list[str],
    n_samples: int,
    feature_distributions: dict[str, tuple],
) -> list[ReferenceLog]:
    """学習データのサンプルを t_reference_logs に登録する。"""
    records = []
    for _ in range(n_samples):
        fv = {}
        for feat in feature_names:
            dist = feature_distributions[feat]
            if dist[0] == "normal":
                fv[feat] = round(random.gauss(dist[1], dist[2]), 4)
            elif dist[0] == "uniform":
                fv[feat] = round(random.uniform(dist[1], dist[2]), 4)
        records.append(
            ReferenceLog(
                project_id=project_id,
                model_id=model_id,
                feature_values=json.dumps(fv),
            )
        )
    return records


def make_logs(
    project_id: int,
    model_id: int | None,
    n: int,
    days: int,
    base_latency: float,
    error_rate: float,
    task_type: str,
    feature_names: list[str],
    feature_distributions: dict[str, tuple],
    drift_from: int | None = None,
) -> list[InferenceLog]:
    """推論ログを生成する。drift_from 以降は特徴量・予測値分布をシフトさせる。"""
    logs = []
    now = datetime.utcnow()
    start = now - timedelta(days=days)

    for i in range(n):
        ts = start + (now - start) * (i / n)
        is_error = random.random() < error_rate
        latency = max(1.0, random.gauss(base_latency, base_latency * 0.3))

        drifted = drift_from is not None and i >= drift_from

        # 特徴量（ドリフト時は平均をシフト）
        fv = {}
        for feat in feature_names:
            dist = feature_distributions[feat]
            shift = 1.5 if drifted else 0.0
            if dist[0] == "normal":
                fv[feat] = round(random.gauss(dist[1] + shift, dist[2]), 4)
            elif dist[0] == "uniform":
                fv[feat] = round(random.uniform(dist[1] + shift, dist[2] + shift), 4)

        if task_type != "regression":
            if drifted:
                pred = min(1.0, max(0.0, random.gauss(0.75, 0.15)))
            else:
                pred = min(1.0, max(0.0, random.gauss(0.45, 0.2)))
            actual = float(random.random() < 0.5) if random.random() < 0.8 else None
        else:
            if drifted:
                pred = random.gauss(120, 15)
            else:
                pred = random.gauss(100, 10)
            actual = pred + random.gauss(0, 8) if random.random() < 0.8 else None

        logs.append(
            InferenceLog(
                project_id=project_id,
                model_id=model_id,
                request_timestamp=ts,
                request_id=f"req-{i:06d}",
                prediction_values=pred,
                actual_values=actual,
                response_time_ms=latency if not is_error else latency * 3,
                is_error=is_error,
                feature_values=json.dumps(fv),
            )
        )
    return logs


def main() -> None:
    print("データベースを初期化しています…")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        db.query(InferenceLog).delete()
        db.query(ReferenceLog).delete()
        db.query(DeployedModel).delete()
        db.query(Project).delete()
        db.commit()

        projects_def = [
            {
                "project_name": "fraud-detection",
                "description": "クレジットカード不正検知モデル（二値分類）",
                "task_type": "binary",
                "n": 600,
                "days": 10,
                "base_latency": 120.0,
                "error_rate": 0.02,
                "drift_from": 500,
                "model_version": "v1.2.0",
                "feature_names": ["amount", "age", "transaction_count", "distance_from_home"],
                "feature_importance": {"amount": 0.42, "age": 0.25, "transaction_count": 0.20, "distance_from_home": 0.13},
                "feature_dtypes": {"amount": "float32", "age": "float32", "transaction_count": "int32", "distance_from_home": "float32"},
                "feature_distributions": {
                    "amount": ("normal", 5000.0, 2000.0),
                    "age": ("normal", 38.0, 10.0),
                    "transaction_count": ("normal", 5.0, 2.0),
                    "distance_from_home": ("uniform", 0.0, 50.0),
                },
                "n_ref_samples": 200,
            },
            {
                "project_name": "price-prediction",
                "description": "住宅価格予測モデル（回帰）",
                "task_type": "regression",
                "n": 400,
                "days": 7,
                "base_latency": 85.0,
                "error_rate": 0.01,
                "drift_from": None,
                "model_version": "v2.0.1",
                "feature_names": ["area_sqm", "rooms", "age_years", "distance_station"],
                "feature_importance": {"area_sqm": 0.50, "rooms": 0.20, "age_years": 0.18, "distance_station": 0.12},
                "feature_dtypes": {"area_sqm": "float32", "rooms": "int32", "age_years": "float32", "distance_station": "float32"},
                "feature_distributions": {
                    "area_sqm": ("normal", 75.0, 20.0),
                    "rooms": ("uniform", 1.0, 6.0),
                    "age_years": ("normal", 20.0, 15.0),
                    "distance_station": ("normal", 10.0, 5.0),
                },
                "n_ref_samples": 150,
            },
            {
                "project_name": "churn-model",
                "description": "顧客離脱予測モデル（二値分類）",
                "task_type": "binary",
                "n": 300,
                "days": 5,
                "base_latency": 200.0,
                "error_rate": 0.05,
                "drift_from": None,
                "model_version": "v0.9.3",
                "feature_names": ["tenure_months", "monthly_charges", "num_products", "support_calls"],
                "feature_importance": {"tenure_months": 0.35, "monthly_charges": 0.30, "num_products": 0.20, "support_calls": 0.15},
                "feature_dtypes": {"tenure_months": "int32", "monthly_charges": "float32", "num_products": "int32", "support_calls": "int32"},
                "feature_distributions": {
                    "tenure_months": ("normal", 30.0, 18.0),
                    "monthly_charges": ("normal", 65.0, 20.0),
                    "num_products": ("uniform", 1.0, 5.0),
                    "support_calls": ("normal", 2.0, 1.5),
                },
                "n_ref_samples": 120,
            },
        ]

        for pdef in projects_def:
            n = pdef.pop("n")
            days = pdef.pop("days")
            base_latency = pdef.pop("base_latency")
            error_rate = pdef.pop("error_rate")
            drift_from = pdef.pop("drift_from")
            model_version = pdef.pop("model_version")
            feature_names = pdef.pop("feature_names")
            feature_importance = pdef.pop("feature_importance")
            feature_dtypes = pdef.pop("feature_dtypes")
            feature_distributions = pdef.pop("feature_distributions")
            n_ref_samples = pdef.pop("n_ref_samples")

            project = Project(**pdef)
            db.add(project)
            db.flush()

            # デプロイ済みモデルを 1 レコード登録
            deployed_model = DeployedModel(
                project_id=project.project_id,
                model_version=model_version,
                feature_dtypes=json.dumps(feature_dtypes),
                feature_importance=json.dumps(feature_importance),
                is_activate=True,
                created_at=datetime.utcnow() - timedelta(days=days + 1),
            )
            db.add(deployed_model)
            db.flush()

            # 参照ログ（学習サンプル）を t_reference_logs に登録
            ref_records = make_reference_log_samples(
                project_id=project.project_id,
                model_id=deployed_model.model_id,
                feature_names=feature_names,
                n_samples=n_ref_samples,
                feature_distributions=feature_distributions,
            )
            db.bulk_save_objects(ref_records)

            logs = make_logs(
                project_id=project.project_id,
                model_id=deployed_model.model_id,
                n=n,
                days=days,
                base_latency=base_latency,
                error_rate=error_rate,
                task_type=pdef["task_type"],
                feature_names=feature_names,
                feature_distributions=feature_distributions,
                drift_from=drift_from,
            )
            db.bulk_save_objects(logs)
            print(f"  [{project.project_name}] 参照サンプル {n_ref_samples} 件 + 推論ログ {len(logs)} 件を生成しました")

        db.commit()
        print("\n完了！ `uv run uvicorn main:app --reload` でサーバーを起動してください。")
        print("ブラウザで http://localhost:8000 にアクセスしてください。")
    finally:
        db.close()


if __name__ == "__main__":
    main()

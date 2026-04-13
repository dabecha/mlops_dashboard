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
from app.models import InferenceLog, Project

# 再現性のためシードを固定
random.seed(42)

try:
    import numpy as np
    np.random.seed(42)
except ImportError:
    pass


def make_logs(
    project_id: int,
    n: int,
    days: int,
    base_latency: float,
    error_rate: float,
    task_type: str,
    drift_from: int | None = None,
) -> list[InferenceLog]:
    """推論ログを生成する。drift_from 以降は予測値分布をシフトさせる。"""
    logs = []
    now = datetime.utcnow()
    start = now - timedelta(days=days)

    for i in range(n):
        ts = start + (now - start) * (i / n)
        is_error = random.random() < error_rate
        latency = max(1.0, random.gauss(base_latency, base_latency * 0.3))

        # ドリフト模擬: drift_from 以降は予測値の平均をシフト
        drifted = drift_from is not None and i >= drift_from
        if task_type == "classification":
            if drifted:
                pred = min(1.0, max(0.0, random.gauss(0.75, 0.15)))  # 正常時より高め
            else:
                pred = min(1.0, max(0.0, random.gauss(0.45, 0.2)))
            actual = float(random.random() < 0.5) if random.random() < 0.8 else None
        else:  # regression
            if drifted:
                pred = random.gauss(120, 15)  # 正常時より高め
            else:
                pred = random.gauss(100, 10)
            actual = pred + random.gauss(0, 8) if random.random() < 0.8 else None

        features = {
            "feature_1": random.gauss(0, 1),
            "feature_2": random.gauss(0, 1),
            "feature_3": random.uniform(0, 10),
        }

        logs.append(
            InferenceLog(
                project_id=project_id,
                timestamp=ts,
                request_id=f"req-{i:06d}",
                prediction=pred,
                actual_label=actual,
                confidence=random.uniform(0.6, 0.99) if not is_error else None,
                response_time_ms=latency if not is_error else latency * 3,
                is_error=is_error,
                error_message="timeout" if is_error else None,
                feature_values=json.dumps(features),
            )
        )
    return logs


def main() -> None:
    print("データベースを初期化しています…")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 既存データをクリア
        db.query(InferenceLog).delete()
        db.query(Project).delete()
        db.commit()

        projects_def = [
            {
                "name": "fraud-detection",
                "description": "クレジットカード不正検知モデル（分類）",
                "task_type": "classification",
                "n": 600,
                "days": 10,
                "base_latency": 120.0,
                "error_rate": 0.02,
                "drift_from": 500,  # 直近100件でドリフト
            },
            {
                "name": "price-prediction",
                "description": "住宅価格予測モデル（回帰）",
                "task_type": "regression",
                "n": 400,
                "days": 7,
                "base_latency": 85.0,
                "error_rate": 0.01,
                "drift_from": None,
            },
            {
                "name": "churn-model",
                "description": "顧客離脱予測モデル（分類）",
                "task_type": "classification",
                "n": 300,
                "days": 5,
                "base_latency": 200.0,
                "error_rate": 0.05,
                "drift_from": None,
            },
        ]

        for pdef in projects_def:
            drift_from = pdef.pop("drift_from")
            n = pdef.pop("n")
            days = pdef.pop("days")
            base_latency = pdef.pop("base_latency")
            error_rate = pdef.pop("error_rate")

            project = Project(**pdef)
            db.add(project)
            db.flush()

            logs = make_logs(
                project_id=project.id,
                n=n,
                days=days,
                base_latency=base_latency,
                error_rate=error_rate,
                task_type=pdef["task_type"],
                drift_from=drift_from,
            )
            db.bulk_save_objects(logs)
            print(f"  [{project.name}] {len(logs)} 件のログを生成しました")

        db.commit()
        print("\n完了！ `uv run uvicorn main:app --reload` でサーバーを起動してください。")
        print("ブラウザで http://localhost:8000 にアクセスしてください。")
    finally:
        db.close()


if __name__ == "__main__":
    main()

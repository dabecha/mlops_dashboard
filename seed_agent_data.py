"""
エージェント監視デモデータ生成スクリプト

実行方法:
    uv run python seed_agent_data.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from app.database import Base, SessionLocal, engine
from app.models import AgentProject, AgentRefData, AgentStep, AgentTask

random.seed(99)

try:
    import numpy as np
    np.random.seed(99)
except ImportError:
    pass


def _gauss(mu: float, sigma: float) -> float:
    return round(random.gauss(mu, sigma), 4)


def _uniform(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 4)


def make_ref_samples(
    project_id: int,
    feature_defs: dict[str, tuple],
    feature_importance: dict[str, float],
    n: int,
    created_at: datetime,
) -> list[AgentRefData]:
    records = []
    for i in range(n):
        fv = {}
        for feat, (dist, *args) in feature_defs.items():
            if dist == "normal":
                fv[feat] = round(random.gauss(args[0], args[1]), 4)
            else:
                fv[feat] = round(random.uniform(args[0], args[1]), 4)
        records.append(AgentRefData(
            project_id=project_id,
            feature_values=json.dumps(fv),
            feature_importance=json.dumps(feature_importance) if i == 0 else None,
            created_at=created_at,
        ))
    return records


def make_tasks_and_steps(
    project_id: int,
    n_tasks: int,
    days: int,
    agent_type: str,
    tool_names: list[str],
    feature_defs: dict[str, tuple],
    model_name: str,
    base_cost_per_1k_input: float,
    base_cost_per_1k_output: float,
    success_rate: float,
    drift_from: int | None,
    quality_mean: float,
) -> tuple[list[AgentTask], list[AgentStep]]:
    tasks = []
    steps = []
    now = datetime.utcnow()
    start = now - timedelta(days=days)

    for i in range(n_tasks):
        ts = start + (now - start) * (i / n_tasks)
        drifted = drift_from is not None and i >= drift_from

        rnd = random.random()
        if rnd < success_rate:
            status = "success"
        elif rnd < success_rate + 0.05:
            status = "failure"
        else:
            status = "error"

        input_tokens = int(random.gauss(1200 if not drifted else 2200, 400))
        output_tokens = int(random.gauss(350 if not drifted else 600, 120))
        input_tokens = max(50, input_tokens)
        output_tokens = max(20, output_tokens)

        cost = (input_tokens / 1000 * base_cost_per_1k_input
                + output_tokens / 1000 * base_cost_per_1k_output)

        duration_ms = max(200.0, random.gauss(
            2500 if not drifted else 4500,
            800 if not drifted else 1500,
        ))

        quality = None
        if status == "success" and random.random() < 0.7:
            quality = min(1.0, max(0.0, random.gauss(
                quality_mean if not drifted else quality_mean - 0.15, 0.08
            )))

        fv = {}
        for feat, (dist, *args) in feature_defs.items():
            shift = 1.5 if drifted else 0.0
            if dist == "normal":
                fv[feat] = round(random.gauss(args[0] + shift, args[1]), 4)
            else:
                fv[feat] = round(random.uniform(args[0] + shift, args[1] + shift), 4)

        n_steps = random.randint(2, 6)
        task = AgentTask(
            project_id=project_id,
            session_id=f"sess-{i // 5:04d}",
            task_name=f"task-{i:05d}",
            status=status,
            quality_score=quality,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost_usd=round(cost, 8),
            total_steps=n_steps,
            duration_ms=round(duration_ms, 1),
            started_at=ts,
            feature_values=json.dumps(fv),
            error_message="RuntimeError: downstream timeout" if status == "error" else None,
        )
        tasks.append(task)

        step_dur_total = duration_ms
        for s in range(n_steps):
            step_dur = step_dur_total / n_steps * random.uniform(0.5, 1.5)
            if s == 0:
                stype = "llm_call"
                sname = model_name
                in_tok = int(input_tokens * random.uniform(0.4, 0.7))
                out_tok = int(output_tokens * random.uniform(0.4, 0.7))
                s_cost = (in_tok / 1000 * base_cost_per_1k_input
                          + out_tok / 1000 * base_cost_per_1k_output)
            elif s == n_steps - 1:
                stype = "llm_call"
                sname = model_name
                in_tok = int(input_tokens * random.uniform(0.2, 0.4))
                out_tok = int(output_tokens * random.uniform(0.2, 0.4))
                s_cost = (in_tok / 1000 * base_cost_per_1k_input
                          + out_tok / 1000 * base_cost_per_1k_output)
            else:
                stype = "tool_call" if agent_type != "multi" else random.choice(["tool_call", "agent_call"])
                sname = random.choice(tool_names)
                in_tok = None
                out_tok = None
                s_cost = None

            s_status = "success" if random.random() < 0.92 else "failure"
            steps.append(AgentStep(
                task_id=None,  # assigned after flush
                project_id=project_id,
                step_type=stype,
                step_name=sname,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=round(s_cost, 8) if s_cost else None,
                duration_ms=round(step_dur, 1),
                status=s_status,
                started_at=ts + timedelta(milliseconds=step_dur * s),
            ))

    return tasks, steps


def main() -> None:
    print("データベースを初期化しています…")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        db.query(AgentStep).delete()
        db.query(AgentTask).delete()
        db.query(AgentRefData).delete()
        db.query(AgentProject).delete()
        db.commit()

        projects_def = [
            {
                "project_name": "customer-support-agent",
                "description": "顧客サポート自動応答エージェント（ツール使用型）",
                "agent_type": "tool",
                "model_name": "claude-sonnet-4-6",
                "n_tasks": 400,
                "days": 10,
                "n_ref": 150,
                "success_rate": 0.91,
                "quality_mean": 0.83,
                "drift_from": 320,
                "tool_names": ["knowledge_base_search", "ticket_create", "user_lookup", "escalate"],
                "feature_defs": {
                    "prompt_len": ("normal", 800.0, 200.0),
                    "tool_calls": ("normal", 2.5, 0.8),
                    "context_turns": ("uniform", 1.0, 8.0),
                },
                "feature_importance": {"prompt_len": 0.50, "tool_calls": 0.30, "context_turns": 0.20},
                "base_cost_per_1k_input": 0.003,
                "base_cost_per_1k_output": 0.015,
            },
            {
                "project_name": "code-assistant",
                "description": "コード生成・レビューエージェント（ツール使用型）",
                "agent_type": "tool",
                "model_name": "claude-opus-4-8",
                "n_tasks": 250,
                "days": 7,
                "n_ref": 100,
                "success_rate": 0.87,
                "quality_mean": 0.79,
                "drift_from": None,
                "tool_names": ["code_exec", "file_read", "web_search", "linter"],
                "feature_defs": {
                    "code_lines": ("normal", 120.0, 60.0),
                    "context_len": ("normal", 1500.0, 500.0),
                },
                "feature_importance": {"code_lines": 0.45, "context_len": 0.55},
                "base_cost_per_1k_input": 0.015,
                "base_cost_per_1k_output": 0.075,
            },
            {
                "project_name": "research-pipeline",
                "description": "マルチエージェント調査・レポート生成パイプライン",
                "agent_type": "multi",
                "model_name": "claude-sonnet-4-6",
                "n_tasks": 150,
                "days": 14,
                "n_ref": 80,
                "success_rate": 0.82,
                "quality_mean": 0.76,
                "drift_from": None,
                "tool_names": ["search_agent", "analysis_agent", "writer_agent", "web_search"],
                "feature_defs": {
                    "query_complexity": ("normal", 3.5, 1.2),
                    "agent_count": ("uniform", 2.0, 5.0),
                    "source_count": ("normal", 8.0, 3.0),
                },
                "feature_importance": {
                    "query_complexity": 0.40,
                    "agent_count": 0.35,
                    "source_count": 0.25,
                },
                "base_cost_per_1k_input": 0.003,
                "base_cost_per_1k_output": 0.015,
            },
        ]

        for pdef in projects_def:
            n_tasks = pdef.pop("n_tasks")
            days = pdef.pop("days")
            n_ref = pdef.pop("n_ref")
            success_rate = pdef.pop("success_rate")
            quality_mean = pdef.pop("quality_mean")
            drift_from = pdef.pop("drift_from")
            tool_names = pdef.pop("tool_names")
            feature_defs = pdef.pop("feature_defs")
            feature_importance = pdef.pop("feature_importance")
            base_cost_per_1k_input = pdef.pop("base_cost_per_1k_input")
            base_cost_per_1k_output = pdef.pop("base_cost_per_1k_output")
            model_name = pdef["model_name"]

            project = AgentProject(**pdef)
            db.add(project)
            db.flush()

            # 参照データ登録
            ref_created_at = datetime.utcnow() - timedelta(days=days + 2)
            ref_records = make_ref_samples(
                project.project_id, feature_defs, feature_importance, n_ref, ref_created_at
            )
            db.bulk_save_objects(ref_records)
            db.flush()

            # タスクとステップ生成
            tasks, steps = make_tasks_and_steps(
                project_id=project.project_id,
                n_tasks=n_tasks,
                days=days,
                agent_type=project.agent_type,
                tool_names=tool_names,
                feature_defs=feature_defs,
                model_name=model_name,
                base_cost_per_1k_input=base_cost_per_1k_input,
                base_cost_per_1k_output=base_cost_per_1k_output,
                success_rate=success_rate,
                drift_from=drift_from,
                quality_mean=quality_mean,
            )
            db.bulk_save_objects(tasks)
            db.flush()

            # task_id を steps に紐付け
            all_tasks = (
                db.query(AgentTask)
                .filter(AgentTask.project_id == project.project_id)
                .order_by(AgentTask.started_at)
                .all()
            )
            step_idx = 0
            for t in all_tasks:
                for _ in range(t.total_steps):
                    if step_idx < len(steps):
                        steps[step_idx].task_id = t.task_id
                        step_idx += 1

            db.bulk_save_objects(steps)
            print(
                f"  [{project.project_name}] 参照 {n_ref} 件 + タスク {len(tasks)} 件"
                f" + ステップ {len(steps)} 件を生成しました"
            )

        db.commit()
        print("\n完了！ `uv run uvicorn main:app --reload` でサーバーを起動してください。")
        print("ブラウザで http://localhost:8000/agents にアクセスしてください。")
    finally:
        db.close()


if __name__ == "__main__":
    main()

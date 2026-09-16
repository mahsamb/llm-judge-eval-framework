"""Load evaluation datasets from CSV/JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_judge_eval.schemas import EvalRecord, TaskType


def load_records(path: Path | str, task: TaskType) -> list[EvalRecord]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json(path, task)
    if path.suffix.lower() == ".csv":
        return _load_csv(path, task)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_csv(path: Path, task: TaskType) -> list[EvalRecord]:
    df = pd.read_csv(path)
    required = {"id", "input_text", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    records: list[EvalRecord] = []
    for _, row in df.iterrows():
        records.append(
            EvalRecord(
                id=str(row["id"]),
                task=task,
                input_text=str(row["input_text"]),
                prediction=str(row["prediction"]),
                reference=str(row["reference"]) if "reference" in row and pd.notna(row["reference"]) else None,
                context=str(row["context"]) if "context" in row and pd.notna(row.get("context")) else None,
            )
        )
    return records


def _load_json(path: Path, task: TaskType) -> list[EvalRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[EvalRecord] = []
    for item in data:
        records.append(
            EvalRecord(
                id=str(item["id"]),
                task=task,
                input_text=str(item["input_text"]),
                prediction=str(item["prediction"]),
                reference=item.get("reference"),
                context=item.get("context"),
                metadata=item.get("metadata", {}),
            )
        )
    return records

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ai_lab.config import ExperimentConfig
from ai_lab.modeling import FEATURE_COLUMNS, train_and_evaluate


def _feature_frame() -> pd.DataFrame:
    rows = []
    for label_index, label in enumerate(("script", "ai_agent", "human_like")):
        for index in range(10):
            base = float(label_index * 100 + index)
            row = {
                "session_id": f"{label}-{index}",
                "label": label,
                "request_count": 6 + index % 3,
                "mean_interval_ms": 90 + base * 10,
                "std_interval_ms": 2 + base,
                "requests_per_second": 10.0 / (label_index + 1) + index / 100,
                "duration_ms": 500 + base * 30,
                "unique_path_ratio": 0.2 + label_index * 0.2 + index / 1000,
                "path_sequence_score": 0.9 - label_index * 0.2,
                "error_rate": label_index * 0.1,
                "repetition_rate": label_index * 0.2,
                "unique_payload_ratio": 0.8 - label_index * 0.1,
                "ua_anomaly_rate": 1.0 if label != "human_like" else 0.0,
                "mean_response_bytes": 1000 + base,
            }
            for column in FEATURE_COLUMNS:
                row.setdefault(column, base / 1000 + 0.01)
            rows.append(row)
    return pd.DataFrame(rows)


def test_model_serialization_and_metadata_schema(tmp_path: Path):
    csv_path = tmp_path / "features.csv"
    output = tmp_path / "results"
    _feature_frame().to_csv(csv_path, index=False)
    metrics = train_and_evaluate(csv_path, output, ExperimentConfig())

    model = joblib.load(output / "random_forest.joblib")
    metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
    assert list(model.classes_) == metrics["classes"]
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert "session_id" not in metadata["feature_columns"]
    assert "label" not in metadata["feature_columns"]
    assert metadata["seed"] == 20261025


def test_training_is_reproducible(tmp_path: Path):
    csv_path = tmp_path / "features.csv"
    _feature_frame().to_csv(csv_path, index=False)
    first = train_and_evaluate(csv_path, tmp_path / "one", ExperimentConfig())
    second = train_and_evaluate(csv_path, tmp_path / "two", ExperimentConfig())
    assert first["test_session_ids"] == second["test_session_ids"]
    assert first["random_forest"]["confusion_matrix"] == second["random_forest"]["confusion_matrix"]
    assert first["random_forest"]["macro_f1"] == second["random_forest"]["macro_f1"]


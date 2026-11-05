from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .modeling import FEATURE_COLUMNS


def validate_artifacts(artifacts_root: Path) -> dict:
    feature_path = artifacts_root / "processed" / "session_features.csv"
    metadata_path = artifacts_root / "processed" / "experiment_metadata.json"
    metrics_path = artifacts_root / "results" / "metrics.json"
    model_path = artifacts_root / "results" / "random_forest.joblib"
    prediction_path = artifacts_root / "results" / "test_predictions.csv"
    replay_path = artifacts_root / "results" / "defense_replay.csv"
    splits_path = artifacts_root / "results" / "data_splits.csv"
    required = [feature_path, metadata_path, metrics_path, model_path, prediction_path, replay_path, splits_path]
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing validation artifacts: {missing_files}")

    features = pd.read_csv(feature_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    replay = pd.read_csv(replay_path)
    splits = pd.read_csv(splits_path)
    model = joblib.load(model_path)
    counts = features["label"].value_counts().sort_index().to_dict()
    expected_per_class = int(metadata["configured_sessions_per_class"])
    expected = {label: expected_per_class for label in ("ai_agent", "human_like", "script")}

    checks = {
        "required_files_present": not missing_files,
        "balanced_classes": counts == expected,
        "metadata_counts_match": metadata["completed_sessions"] == counts,
        "metrics_counts_match": metrics["sample_counts"] == counts,
        "unique_session_ids": not features["session_id"].duplicated().any(),
        "no_missing_feature_values": not features[FEATURE_COLUMNS].isna().any().any(),
        "no_identifier_or_label_leakage": "session_id" not in FEATURE_COLUMNS and "label" not in FEATURE_COLUMNS,
        "model_feature_count_matches": getattr(model, "n_features_in_", None) == len(FEATURE_COLUMNS),
        "model_classes_match": list(model.classes_) == metrics["classes"],
        "test_ids_match_predictions": set(metrics["test_session_ids"]) == set(predictions["session_id"]),
        "defense_replay_uses_test_only": set(replay["session_id"]) == set(metrics["test_session_ids"]),
        "split_assignment_is_complete": set(splits["session_id"]) == set(features["session_id"]),
        "split_assignment_is_unique": not splits["session_id"].duplicated().any(),
        "test_split_matches_metrics": set(splits.loc[splits["split"] == "test", "session_id"]) == set(metrics["test_session_ids"]),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "class_counts": counts,
        "feature_columns": FEATURE_COLUMNS,
    }
    output = artifacts_root / "results" / "validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "pass":
        raise RuntimeError(f"Artifact validation failed: {checks}")
    return result


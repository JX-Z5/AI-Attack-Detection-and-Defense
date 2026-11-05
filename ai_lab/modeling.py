from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import ExperimentConfig
from .defense import choose_online_threshold, load_events, replay_defense, write_replay_artifacts


FEATURE_COLUMNS = [
    "request_count",
    "mean_interval_ms",
    "std_interval_ms",
    "requests_per_second",
    "duration_ms",
    "unique_path_ratio",
    "path_sequence_score",
    "error_rate",
    "repetition_rate",
    "unique_payload_ratio",
    "ua_anomaly_rate",
    "mean_response_bytes",
    "interval_cv",
    "interval_range_80",
    "payload_entropy",
    "payload_transition_entropy",
    "payload_switch_rate",
    "max_payload_share",
    "marker_entropy",
    "response_size_cv",
    "response_size_range",
    "response_unique_ratio",
    "marker_change_rate",
    "switch_after_marker_change",
    "switch_after_same_marker",
]
LABEL_ORDER = ["script", "ai_agent", "human_like"]


def rule_predict(row: pd.Series) -> str:
    if row["ua_anomaly_rate"] > 0.99 and row["mean_interval_ms"] < 150:
        return "script"
    if row["mean_interval_ms"] < 1500 and row["repetition_rate"] > 0.70:
        return "ai_agent"
    return "human_like"


def _scores(y_true, y_pred) -> dict:
    return {
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0
        ),
    }


def _choose_threshold(model, validation: pd.DataFrame, config: ExperimentConfig) -> dict:
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])
    ai_index = list(model.classes_).index("ai_agent")
    script_index = list(model.classes_).index("script")
    candidates = np.arange(
        config.defense.threshold_min,
        config.defense.threshold_max + config.defense.threshold_step / 2,
        config.defense.threshold_step,
    )
    feasible = []
    for threshold in candidates:
        decisions = []
        for i, row in validation.reset_index(drop=True).iterrows():
            if row["ua_anomaly_rate"] > 0.99 and row["mean_interval_ms"] < 130:
                decisions.append("block_script")
            elif probabilities[i, ai_index] >= threshold:
                decisions.append("block_ai")
            else:
                decisions.append("allow")
        human_mask = validation["label"].to_numpy() == "human_like"
        ai_mask = validation["label"].to_numpy() == "ai_agent"
        human_fbr = float(np.mean(np.array(decisions)[human_mask] != "allow")) if human_mask.any() else 0.0
        ai_block = float(np.mean(np.array(decisions)[ai_mask] == "block_ai")) if ai_mask.any() else 0.0
        if human_fbr <= config.defense.max_human_like_false_block_rate:
            feasible.append((ai_block, -human_fbr, float(threshold)))
    if not feasible:
        return {"mode": "report_only", "threshold": None, "validation_ai_block_rate": 0.0}
    best = max(feasible)
    return {"mode": "enforce", "threshold": best[2], "validation_ai_block_rate": best[0], "validation_human_false_block_rate": -best[1]}


def _evaluate_defense(model, test: pd.DataFrame, policy: dict) -> dict:
    if policy["mode"] != "enforce":
        return {"mode": "report_only", "ai_block_rate": 0.0, "script_block_rate": 0.0, "human_false_block_rate": 0.0}
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])
    ai_index = list(model.classes_).index("ai_agent")
    decisions = []
    for i, row in test.reset_index(drop=True).iterrows():
        if row["ua_anomaly_rate"] > 0.99 and row["mean_interval_ms"] < 130:
            decisions.append("block_script")
        elif probabilities[i, ai_index] >= policy["threshold"]:
            decisions.append("block_ai")
        else:
            decisions.append("allow")
    labels = test["label"].to_numpy()
    decisions = np.array(decisions)
    rate = lambda mask, value: float(np.mean(decisions[mask] == value)) if mask.any() else 0.0
    return {
        "mode": "enforce",
        "threshold": policy["threshold"],
        "ai_block_rate": rate(labels == "ai_agent", "block_ai"),
        "script_block_rate": rate(labels == "script", "block_script"),
        "human_false_block_rate": float(np.mean(decisions[labels == "human_like"] != "allow")) if (labels == "human_like").any() else 0.0,
    }


def train_and_evaluate(
    features_csv: Path,
    artifacts_dir: Path,
    config: ExperimentConfig,
    raw_events_path: Path | None = None,
) -> dict:
    data = pd.read_csv(features_csv)
    required = set(FEATURE_COLUMNS + ["session_id", "label"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature table contains missing numeric values")
    if data["session_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate session_id values")

    train_val, test = train_test_split(
        data, test_size=config.split.test_size, stratify=data["label"], random_state=config.seed
    )
    train, validation = train_test_split(
        train_val,
        test_size=config.split.validation_size_of_remaining,
        stratify=train_val["label"],
        random_state=config.seed + 1,
    )
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    cv_scores = cross_val_score(model, train[FEATURE_COLUMNS], train["label"], cv=cv, scoring="f1_macro")

    test_pred = model.predict(test[FEATURE_COLUMNS])
    rule_pred = test.apply(rule_predict, axis=1)
    policy = _choose_threshold(model, validation, config)
    defense = _evaluate_defense(model, test, policy)

    time_features = ["mean_interval_ms", "std_interval_ms", "requests_per_second", "duration_ms"]
    no_time_features = [column for column in FEATURE_COLUMNS if column not in time_features]
    ablation_model = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=config.seed, n_jobs=-1
    )
    ablation_model.fit(train[no_time_features], train["label"])
    no_time_pred = ablation_model.predict(test[no_time_features])

    metrics = {
        "seed": config.seed,
        "sample_counts": data["label"].value_counts().sort_index().to_dict(),
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "random_forest": _scores(test["label"], test_pred),
        "rule_baseline": _scores(test["label"], rule_pred),
        "without_timing_features": _scores(test["label"], no_time_pred),
        "cross_validation_macro_f1": {"mean": float(cv_scores.mean()), "std": float(cv_scores.std()), "folds": cv_scores.tolist()},
        "defense_policy": policy,
        "defense_test": defense,
        "feature_importance": dict(sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda pair: pair[1], reverse=True)),
        "classes": list(model.classes_),
        "feature_columns": FEATURE_COLUMNS,
        "test_session_ids": test["session_id"].tolist(),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if raw_events_path is not None:
        raw_events = load_events(raw_events_path)
        policy = choose_online_threshold(
            model=model,
            events=raw_events,
            validation_ids=set(validation["session_id"]),
            feature_columns=FEATURE_COLUMNS,
            warmup_requests=config.defense.warmup_requests,
            threshold_min=config.defense.threshold_min,
            threshold_max=config.defense.threshold_max,
            threshold_step=config.defense.threshold_step,
            max_human_false_block_rate=config.defense.max_human_like_false_block_rate,
        )
        online_defense, replay_log = replay_defense(
            model=model,
            events=raw_events,
            session_ids=set(test["session_id"]),
            policy=policy,
            feature_columns=FEATURE_COLUMNS,
            warmup_requests=config.defense.warmup_requests,
        )
        metrics["defense_policy"] = policy
        metrics["defense_test"] = online_defense
        write_replay_artifacts(replay_log, artifacts_dir)

    joblib.dump(model, artifacts_dir / "random_forest.joblib")
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_type": "RandomForestClassifier",
                "scikit_learn_version": sklearn.__version__,
                "seed": metrics["seed"],
                "classes": metrics["classes"],
                "feature_columns": metrics["feature_columns"],
                "sample_counts": metrics["sample_counts"],
                "split_counts": metrics["split_counts"],
                "defense_policy": metrics["defense_policy"],
                "training_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "feature_table_sha256": hashlib.sha256(features_csv.read_bytes()).hexdigest(),
                "raw_events_sha256": hashlib.sha256(raw_events_path.read_bytes()).hexdigest() if raw_events_path else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test.assign(prediction=test_pred).to_csv(artifacts_dir / "test_predictions.csv", index=False)
    pd.concat(
        [
            train[["session_id", "label"]].assign(split="train"),
            validation[["session_id", "label"]].assign(split="validation"),
            test[["session_id", "label"]].assign(split="test"),
        ],
        ignore_index=True,
    ).sort_values("session_id").to_csv(artifacts_dir / "data_splits.csv", index=False)
    return metrics


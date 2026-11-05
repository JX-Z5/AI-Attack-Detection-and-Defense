from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .config import GeneralizationConfig
from .defense import choose_online_threshold, load_events, replay_defense, write_replay_artifacts
from .modeling import FEATURE_COLUMNS


BINARY_ORDER = ["non_ai", "ai_agent"]
TIME_FEATURES = ["mean_interval_ms", "std_interval_ms", "requests_per_second", "duration_ms"]
GROUP_COLUMNS = ["session_id", "label", "source_id", "prompt_profile", "attack_task", "generator_profile"]


def _target(frame: pd.DataFrame) -> pd.Series:
    return frame["label"].where(frame["label"] == "ai_agent", "non_ai")


def _model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _metrics(y_true, y_pred, probabilities=None) -> dict:
    result = {
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=BINARY_ORDER).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=BINARY_ORDER, output_dict=True, zero_division=0),
        "support": {label: int(sum(np.asarray(y_true) == label)) for label in BINARY_ORDER},
    }
    if probabilities is not None and len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(np.asarray(y_true) == "ai_agent", probabilities))
    return result


def _bootstrap_macro_f1(y_true, y_pred, seed: int, iterations: int = 1000) -> dict:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    by_class = {label: np.flatnonzero(true == label) for label in BINARY_ORDER}
    values = []
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in by_class.values()])
        values.append(f1_score(true[indices], pred[indices], labels=BINARY_ORDER, average="macro", zero_division=0))
    return {
        "iterations": iterations,
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _rule_predict(frame: pd.DataFrame) -> np.ndarray:
    ai = (
        (frame["mean_interval_ms"] < 1000)
        & (frame["ua_anomaly_rate"] < 0.5)
        & ((frame["repetition_rate"] > 0.45) | (frame["unique_payload_ratio"] < 0.45))
    )
    return np.where(ai, "ai_agent", "non_ai")


def _validate_split(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, group_name: str) -> dict:
    sets = [set(part["session_id"]) for part in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError(f"Session leakage in {group_name}")
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if set(_target(part)) != set(BINARY_ORDER):
            raise ValueError(f"Both classes are required in {group_name} {name}")
    return {"train": len(train), "validation": len(validation), "test": len(test)}


def _evaluate_fold(
    frame: pd.DataFrame,
    train_mask,
    validation_mask,
    test_mask,
    name: str,
    seed: int,
) -> tuple[dict, pd.DataFrame, RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()
    test = frame.loc[test_mask].copy()
    split_counts = _validate_split(train, validation, test, name)
    model = _model(seed)
    model.fit(train[FEATURE_COLUMNS], _target(train))
    prediction = model.predict(test[FEATURE_COLUMNS])
    ai_index = list(model.classes_).index("ai_agent")
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, ai_index]
    metrics = _metrics(_target(test), prediction, probabilities)
    metrics["macro_f1_95_ci"] = _bootstrap_macro_f1(_target(test), prediction, seed)
    metrics["split_counts"] = split_counts
    metrics["held_out_ai_sources"] = sorted(test.loc[test["label"] == "ai_agent", "source_id"].unique().tolist())
    metrics["held_out_prompt_profiles"] = sorted(test.loc[test["label"] == "ai_agent", "prompt_profile"].unique().tolist())
    metrics["held_out_attack_tasks"] = sorted(test["attack_task"].unique().tolist())
    predictions = test[GROUP_COLUMNS].copy()
    predictions["binary_target"] = _target(test).to_numpy()
    predictions["prediction"] = prediction
    predictions["ai_probability"] = probabilities
    predictions["evaluation"] = name
    return metrics, predictions, model, train, validation, test


def _aggregate(folds: dict[str, dict]) -> dict:
    values = [item["macro_f1"] for item in folds.values()]
    return {
        "mean_macro_f1": float(np.mean(values)),
        "std_macro_f1": float(np.std(values)),
        "worst_macro_f1": float(np.min(values)),
        "best_macro_f1": float(np.max(values)),
        "fold_count": len(values),
    }


def _model_holdout_masks(frame: pd.DataFrame, models: list[str], script_profiles: list[str], human_profiles: list[str]):
    for index, test_model in enumerate(models):
        validation_model = models[(index + 1) % len(models)]
        test_script = script_profiles[index % len(script_profiles)]
        validation_script = script_profiles[(index + 1) % len(script_profiles)]
        test_human = human_profiles[index % len(human_profiles)]
        validation_human = human_profiles[(index + 1) % len(human_profiles)]
        test_mask = (
            ((frame["label"] == "ai_agent") & (frame["source_id"] == test_model))
            | ((frame["label"] == "script") & (frame["generator_profile"] == test_script))
            | ((frame["label"] == "human_like") & (frame["generator_profile"] == test_human))
        )
        validation_mask = (
            ((frame["label"] == "ai_agent") & (frame["source_id"] == validation_model))
            | ((frame["label"] == "script") & (frame["generator_profile"] == validation_script))
            | ((frame["label"] == "human_like") & (frame["generator_profile"] == validation_human))
        )
        train_mask = ~(test_mask | validation_mask)
        yield test_model, train_mask, validation_mask, test_mask


def _profile_holdout_masks(frame: pd.DataFrame, profiles: list[str], script_profiles: list[str], human_profiles: list[str]):
    for index, test_profile in enumerate(profiles):
        validation_profile = profiles[(index + 1) % len(profiles)]
        test_script = script_profiles[(index + 4) % len(script_profiles)]
        validation_script = script_profiles[(index + 5) % len(script_profiles)]
        test_human = human_profiles[(index + 4) % len(human_profiles)]
        validation_human = human_profiles[(index + 5) % len(human_profiles)]
        test_mask = (
            ((frame["label"] == "ai_agent") & (frame["prompt_profile"] == test_profile))
            | ((frame["label"] == "script") & (frame["generator_profile"] == test_script))
            | ((frame["label"] == "human_like") & (frame["generator_profile"] == test_human))
        )
        validation_mask = (
            ((frame["label"] == "ai_agent") & (frame["prompt_profile"] == validation_profile))
            | ((frame["label"] == "script") & (frame["generator_profile"] == validation_script))
            | ((frame["label"] == "human_like") & (frame["generator_profile"] == validation_human))
        )
        train_mask = ~(test_mask | validation_mask)
        yield test_profile, train_mask, validation_mask, test_mask


def _task_holdout_masks(frame: pd.DataFrame, tasks: list[str], models: list[str]):
    for index, test_task in enumerate(tasks):
        train_task = tasks[(index + 1) % len(tasks)]
        validation_model = models[index % len(models)]
        test_mask = frame["attack_task"] == test_task
        validation_mask = (
            (frame["attack_task"] == train_task)
            & (
                ((frame["label"] == "ai_agent") & (frame["source_id"] == validation_model))
                | ((frame["label"] != "ai_agent") & (frame.index % 4 == index))
            )
        )
        train_mask = ~(test_mask | validation_mask)
        yield test_task, train_mask, validation_mask, test_mask


def _choose_policy(model, validation: pd.DataFrame, config: GeneralizationConfig) -> dict:
    ai_index = list(model.classes_).index("ai_agent")
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, ai_index]
    target = _target(validation).to_numpy()
    candidates = np.arange(
        config.defense.threshold_min,
        config.defense.threshold_max + config.defense.threshold_step / 2,
        config.defense.threshold_step,
    )
    feasible = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        false_positive = float(predicted[target == "non_ai"].mean())
        detection = float(predicted[target == "ai_agent"].mean())
        if false_positive <= config.defense.max_human_like_false_block_rate:
            feasible.append((detection, -false_positive, float(threshold)))
    if not feasible:
        return {"mode": "report_only", "threshold": None, "validation_detection_rate": 0.0}
    best = max(feasible)
    return {
        "mode": "enforce",
        "threshold": best[2],
        "validation_detection_rate": best[0],
        "validation_non_ai_false_positive_rate": -best[1],
        "selection_method": "grouped_validation_sessions",
    }


def train_generalization(
    features_csv: Path,
    raw_events_path: Path,
    output_dir: Path,
    config: GeneralizationConfig,
) -> dict:
    frame = pd.read_csv(features_csv)
    required = set(FEATURE_COLUMNS + GROUP_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature table contains missing numeric values")
    if frame["session_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate session identifiers")
    if set(FEATURE_COLUMNS) & set(GROUP_COLUMNS):
        raise ValueError("Grouping metadata leaked into model features")

    all_predictions = []
    model_folds = {}
    locked = None
    for fold_index, (name, train_mask, validation_mask, test_mask) in enumerate(
        _model_holdout_masks(frame, config.ai_models, config.script_profiles, config.human_profiles)
    ):
        result = _evaluate_fold(frame, train_mask, validation_mask, test_mask, f"model:{name}", config.seed + fold_index)
        fold_metrics, predictions, model, train, validation, test = result
        model_folds[name] = fold_metrics
        all_predictions.append(predictions)
        if name == config.ai_models[-1]:
            locked = (model, train, validation, test, fold_metrics, predictions)

    prompt_folds = {}
    for fold_index, (name, train_mask, validation_mask, test_mask) in enumerate(
        _profile_holdout_masks(frame, config.prompt_profiles, config.script_profiles, config.human_profiles)
    ):
        fold_metrics, predictions, _, _, _, _ = _evaluate_fold(
            frame, train_mask, validation_mask, test_mask, f"prompt:{name}", config.seed + 100 + fold_index
        )
        prompt_folds[name] = fold_metrics
        all_predictions.append(predictions)

    task_folds = {}
    for fold_index, (name, train_mask, validation_mask, test_mask) in enumerate(
        _task_holdout_masks(frame, config.attack_tasks, config.ai_models)
    ):
        fold_metrics, predictions, _, _, _, _ = _evaluate_fold(
            frame, train_mask, validation_mask, test_mask, f"task:{name}", config.seed + 200 + fold_index
        )
        task_folds[name] = fold_metrics
        all_predictions.append(predictions)

    if locked is None:
        raise RuntimeError("Locked model-family test was not created")
    model, train, validation, test, locked_metrics, locked_predictions = locked
    rule_prediction = _rule_predict(test)
    no_time_columns = [column for column in FEATURE_COLUMNS if column not in TIME_FEATURES]
    no_time_model = _model(config.seed + 500)
    no_time_model.fit(train[no_time_columns], _target(train))
    no_time_prediction = no_time_model.predict(test[no_time_columns])
    policy = _choose_policy(model, validation, config)
    ai_index = list(model.classes_).index("ai_agent")
    test_probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, ai_index]
    if policy["mode"] == "enforce":
        blocked = test_probabilities >= policy["threshold"]
        target = _target(test).to_numpy()
        defense_session = {
            "ai_detection_rate": float(blocked[target == "ai_agent"].mean()),
            "non_ai_false_positive_rate": float(blocked[target == "non_ai"].mean()),
        }
    else:
        defense_session = {"ai_detection_rate": 0.0, "non_ai_false_positive_rate": 0.0}

    raw_events = load_events(raw_events_path)
    online_policy = choose_online_threshold(
        model,
        raw_events,
        set(validation["session_id"]),
        FEATURE_COLUMNS,
        config.defense.warmup_requests,
        config.defense.threshold_min,
        config.defense.threshold_max,
        config.defense.threshold_step,
        config.defense.max_human_like_false_block_rate,
    )
    online_defense, replay_log = replay_defense(
        model,
        raw_events,
        set(test["session_id"]),
        online_policy,
        FEATURE_COLUMNS,
        config.defense.warmup_requests,
    )

    metrics = {
        "study_role": "final_generalization_experiment",
        "primary_task": "ai_agent_vs_non_ai_behavior",
        "seed": config.seed,
        "sample_counts": frame["label"].value_counts().sort_index().to_dict(),
        "binary_sample_counts": _target(frame).value_counts().sort_index().to_dict(),
        "model_family_holdout": {"folds": model_folds, "summary": _aggregate(model_folds)},
        "prompt_profile_holdout": {"folds": prompt_folds, "summary": _aggregate(prompt_folds)},
        "attack_task_holdout": {"folds": task_folds, "summary": _aggregate(task_folds)},
        "locked_test": {
            "held_out_model": config.ai_models[-1],
            "random_forest": locked_metrics,
            "rule_baseline": _metrics(_target(test), rule_prediction),
            "without_timing_features": _metrics(_target(test), no_time_prediction),
        },
        "defense_policy": policy,
        "defense_session_test": defense_session,
        "online_defense_policy": online_policy,
        "online_defense_test": online_defense,
        "feature_importance": dict(
            sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
        ),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_group_columns": GROUP_COLUMNS,
        "classes": list(model.classes_),
        "claim_boundary": "The results test transfer across the included local model families, prompts, and two DVWA SQL injection tasks. They do not establish universal detection of every AI attacker.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "generalization_random_forest.joblib")
    pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "external_predictions.csv", index=False)
    locked_predictions.to_csv(output_dir / "locked_test_predictions.csv", index=False)
    pd.concat(
        [
            train[GROUP_COLUMNS].assign(split="train"),
            validation[GROUP_COLUMNS].assign(split="validation"),
            test[GROUP_COLUMNS].assign(split="locked_test"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "locked_group_splits.csv", index=False)
    write_replay_artifacts(replay_log, output_dir)
    metadata = {
        "model_type": "RandomForestClassifier",
        "scikit_learn_version": sklearn.__version__,
        "seed": config.seed,
        "classes": list(model.classes_),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_group_columns": GROUP_COLUMNS,
        "held_out_model": config.ai_models[-1],
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "defense_policy": online_policy,
        "feature_table_sha256": hashlib.sha256(features_csv.read_bytes()).hexdigest(),
        "raw_events_sha256": hashlib.sha256(raw_events_path.read_bytes()).hexdigest(),
    }
    (output_dir / "generalization_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "generalization_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

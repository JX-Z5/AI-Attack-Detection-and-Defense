from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .config import GeneralizationConfig
from .defense import load_events
from .features import extract_session_features
from .modeling import FEATURE_COLUMNS
from .schema import EventRecord


BROWSER_ORDER = ["human_like", "ai_agent"]
TIME_FEATURES = ["mean_interval_ms", "std_interval_ms", "requests_per_second", "duration_ms", "interval_cv", "interval_range_80"]


def _model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=5,
        max_features=0.7,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def _rule(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["mean_interval_ms"].to_numpy() < 500)
        & (frame["repetition_rate"].to_numpy() > 0.45)
        & (frame["unique_payload_ratio"].to_numpy() < 0.45)
    )


def _metrics(y_true, y_pred) -> dict:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    human = true == "human_like"
    ai = true == "ai_agent"
    return {
        "macro_precision": float(precision_score(true, pred, labels=BROWSER_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, labels=BROWSER_ORDER, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, labels=BROWSER_ORDER, average="macro", zero_division=0)),
        "ai_recall": float((pred[ai] == "ai_agent").mean()) if ai.any() else 0.0,
        "human_false_positive_rate": float((pred[human] == "ai_agent").mean()) if human.any() else 0.0,
        "confusion_matrix": confusion_matrix(true, pred, labels=BROWSER_ORDER).tolist(),
        "classification_report": classification_report(true, pred, labels=BROWSER_ORDER, output_dict=True, zero_division=0),
        "support": {label: int((true == label).sum()) for label in BROWSER_ORDER},
    }


def _bootstrap(y_true, y_pred, seed: int, iterations: int = 1000) -> dict:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(true == label) for label in BROWSER_ORDER]
    values = []
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values.append(f1_score(true[indices], pred[indices], labels=BROWSER_ORDER, average="macro", zero_division=0))
    return {
        "iterations": iterations,
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _wilson(successes: int, total: int) -> dict:
    if total == 0:
        return {"lower_95": None, "upper_95": None}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return {"lower_95": float(center - margin), "upper_95": float(center + margin)}


def _probability(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    ai_index = list(model.classes_).index("ai_agent")
    return model.predict_proba(frame[columns])[:, ai_index]


def _choose_threshold(predictions: pd.DataFrame, config: GeneralizationConfig) -> dict:
    candidates = np.arange(
        config.defense.threshold_min,
        config.defense.threshold_max + config.defense.threshold_step / 2,
        config.defense.threshold_step,
    )
    feasible = []
    for threshold in candidates:
        detected = (predictions["ai_probability"].to_numpy() >= threshold) | predictions["rule_hit"].to_numpy()
        labels = predictions["label"].to_numpy()
        human = labels == "human_like"
        human_rate = float(detected[human].mean())
        profile_rates = []
        for _, group in predictions[human].groupby("generator_profile"):
            group_detected = (group["ai_probability"].to_numpy() >= threshold) | group["rule_hit"].to_numpy()
            profile_rates.append(float(group_detected.mean()))
        worst_profile_rate = max(profile_rates, default=0.0)
        if human_rate <= config.defense.max_human_like_false_block_rate and worst_profile_rate <= config.defense.max_human_like_false_block_rate:
            result = np.where(detected, "ai_agent", "human_like")
            score = f1_score(labels, result, labels=BROWSER_ORDER, average="macro", zero_division=0)
            ai_recall = float(detected[labels == "ai_agent"].mean())
            feasible.append((score, ai_recall, -human_rate, float(threshold), worst_profile_rate))
    if not feasible:
        return {"mode": "report_only", "threshold": None, "selection_method": "group_out_of_fold"}
    selected = max(feasible)
    return {
        "mode": "enforce",
        "threshold": selected[3],
        "oof_macro_f1": selected[0],
        "oof_ai_recall": selected[1],
        "oof_human_false_positive_rate": -selected[2],
        "oof_worst_human_profile_false_positive_rate": selected[4],
        "selection_method": "group_out_of_fold",
    }


def _offline_oof(frame: pd.DataFrame, config: GeneralizationConfig, columns: list[str]) -> pd.DataFrame:
    browser = frame[frame["label"].isin(BROWSER_ORDER)].copy()
    rows = []
    for index, source in enumerate(config.ai_models):
        test = browser[(browser["label"] == "ai_agent") & (browser["source_id"] == source)]
        train = browser[~((browser["label"] == "ai_agent") & (browser["source_id"] == source))]
        model = _model(config.seed + index)
        model.fit(train[columns], train["label"])
        part = test[["session_id", "label", "source_id", "generator_profile"]].copy()
        part["ai_probability"] = _probability(model, test, columns)
        part["rule_hit"] = _rule(test)
        part["held_out_group"] = f"model:{source}"
        rows.append(part)
    for index, profile in enumerate(config.human_profiles):
        test = browser[(browser["label"] == "human_like") & (browser["generator_profile"] == profile)]
        train = browser[~((browser["label"] == "human_like") & (browser["generator_profile"] == profile))]
        model = _model(config.seed + 100 + index)
        model.fit(train[columns], train["label"])
        part = test[["session_id", "label", "source_id", "generator_profile"]].copy()
        part["ai_probability"] = _probability(model, test, columns)
        part["rule_hit"] = _rule(test)
        part["held_out_group"] = f"human:{profile}"
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    if result["session_id"].duplicated().any():
        raise ValueError("Offline out-of-fold predictions contain duplicate sessions")
    return result


def _prefix_frame(events: list[EventRecord], warmup_requests: int) -> pd.DataFrame:
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        if event.true_label in BROWSER_ORDER:
            grouped[event.session_id].append(event)
    rows = []
    for session_id in sorted(grouped):
        session = sorted(grouped[session_id], key=lambda item: item.sequence)
        for request_count in range(warmup_requests, len(session) + 1):
            feature = extract_session_features(session[:request_count]).model_dump()
            feature["prefix_request_count"] = request_count
            rows.append(feature)
    return pd.DataFrame(rows)


def _prefix_oof(prefixes: pd.DataFrame, config: GeneralizationConfig, columns: list[str]) -> pd.DataFrame:
    session_rows = []
    for index, source in enumerate(config.ai_models):
        test = prefixes[(prefixes["label"] == "ai_agent") & (prefixes["source_id"] == source)]
        train = prefixes[~((prefixes["label"] == "ai_agent") & (prefixes["source_id"] == source))]
        model = _model(config.seed + 300 + index)
        model.fit(train[columns], train["label"])
        scored = test[["session_id", "label", "source_id", "generator_profile", "prefix_request_count"]].copy()
        scored["ai_probability"] = _probability(model, test, columns)
        scored["rule_hit"] = False
        scored["held_out_group"] = f"model:{source}"
        session_rows.append(scored)
    for index, profile in enumerate(config.human_profiles):
        test = prefixes[(prefixes["label"] == "human_like") & (prefixes["generator_profile"] == profile)]
        train = prefixes[~((prefixes["label"] == "human_like") & (prefixes["generator_profile"] == profile))]
        model = _model(config.seed + 400 + index)
        model.fit(train[columns], train["label"])
        scored = test[["session_id", "label", "source_id", "generator_profile", "prefix_request_count"]].copy()
        scored["ai_probability"] = _probability(model, test, columns)
        scored["rule_hit"] = False
        scored["held_out_group"] = f"human:{profile}"
        session_rows.append(scored)
    prefix_predictions = pd.concat(session_rows, ignore_index=True)
    selected = prefix_predictions.sort_values(["session_id", "ai_probability", "prefix_request_count"]).groupby("session_id", as_index=False).tail(1)
    result = selected[["session_id", "label", "source_id", "generator_profile", "ai_probability", "rule_hit", "held_out_group"]].copy()
    if result["session_id"].duplicated().any():
        raise ValueError("Prefix out-of-fold predictions contain duplicate sessions")
    return result


def _group_results(predictions: pd.DataFrame, policy: dict, group_column: str, prefix: str) -> dict:
    threshold = policy.get("threshold")
    results = {}
    for name, group in predictions.groupby(group_column):
        if prefix == "model" and group["label"].iloc[0] != "ai_agent":
            continue
        if prefix == "human" and group["label"].iloc[0] != "human_like":
            continue
        detected = group["rule_hit"].to_numpy()
        if threshold is not None:
            detected = detected | (group["ai_probability"].to_numpy() >= threshold)
        predicted = np.where(detected, "ai_agent", "human_like")
        results[str(name)] = _metrics(group["label"], predicted)
    return results


def train_optimized_detector(
    features_csv: Path,
    raw_events_path: Path,
    output_dir: Path,
    config: GeneralizationConfig,
) -> dict:
    frame = pd.read_csv(features_csv)
    required = set(FEATURE_COLUMNS + ["session_id", "label", "source_id", "generator_profile"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing optimized feature columns: {sorted(missing)}")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Optimized feature table contains missing values")
    browser = frame[frame["label"].isin(BROWSER_ORDER)].copy()
    offline_oof = _offline_oof(frame, config, FEATURE_COLUMNS)
    offline_policy = _choose_threshold(offline_oof, config)
    offline_detected = offline_oof["rule_hit"].to_numpy()
    if offline_policy["threshold"] is not None:
        offline_detected |= offline_oof["ai_probability"].to_numpy() >= offline_policy["threshold"]
    offline_prediction = np.where(offline_detected, "ai_agent", "human_like")
    offline_metrics = _metrics(offline_oof["label"], offline_prediction)
    offline_metrics["macro_f1_95_ci"] = _bootstrap(offline_oof["label"], offline_prediction, config.seed)

    no_time_columns = [column for column in FEATURE_COLUMNS if column not in TIME_FEATURES]
    no_time_oof = _offline_oof(frame, config, no_time_columns)
    no_time_policy = _choose_threshold(no_time_oof, config)
    no_time_detected = no_time_oof["rule_hit"].to_numpy()
    if no_time_policy["threshold"] is not None:
        no_time_detected |= no_time_oof["ai_probability"].to_numpy() >= no_time_policy["threshold"]
    no_time_prediction = np.where(no_time_detected, "ai_agent", "human_like")

    events = load_events(raw_events_path)
    prefixes = _prefix_frame(events, config.defense.warmup_requests)
    prefix_oof = _prefix_oof(prefixes, config, FEATURE_COLUMNS)
    online_policy = _choose_threshold(prefix_oof, config)
    online_detected = prefix_oof["rule_hit"].to_numpy()
    if online_policy["threshold"] is not None:
        online_detected |= prefix_oof["ai_probability"].to_numpy() >= online_policy["threshold"]
    online_prediction = np.where(online_detected, "ai_agent", "human_like")
    online_metrics = _metrics(prefix_oof["label"], online_prediction)
    online_metrics["macro_f1_95_ci"] = _bootstrap(prefix_oof["label"], online_prediction, config.seed + 1)

    offline_model = _model(config.seed + 700)
    offline_model.fit(browser[FEATURE_COLUMNS], browser["label"])
    online_model = _model(config.seed + 701)
    online_model.fit(prefixes[FEATURE_COLUMNS], prefixes["label"])

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "optimized_detector.joblib"
    bundle = {
        "offline_model": offline_model,
        "online_model": online_model,
        "offline_policy": offline_policy,
        "online_policy": online_policy,
        "feature_columns": FEATURE_COLUMNS,
        "time_features": TIME_FEATURES,
        "warmup_requests": config.defense.warmup_requests,
        "rule": {"mean_interval_ms_lt": 500, "repetition_rate_gt": 0.45, "unique_payload_ratio_lt": 0.45},
        "online_rule": "disabled_after_grouped_false_positive_check",
        "classes": BROWSER_ORDER,
        "seed": config.seed,
    }
    joblib.dump(bundle, bundle_path)
    offline_oof.assign(prediction=offline_prediction).to_csv(output_dir / "offline_oof_predictions.csv", index=False)
    prefix_oof.assign(prediction=online_prediction).to_csv(output_dir / "online_oof_predictions.csv", index=False)
    metrics = {
        "study_role": "development_optimization_before_blind_test",
        "primary_task": "browser_like_ai_agent_vs_human_like_simulation",
        "sample_counts": browser["label"].value_counts().sort_index().to_dict(),
        "prefix_row_count": int(len(prefixes)),
        "offline_policy": offline_policy,
        "offline_group_oof": offline_metrics,
        "offline_model_folds": _group_results(offline_oof, offline_policy, "source_id", "model"),
        "offline_human_profile_folds": _group_results(offline_oof, offline_policy, "generator_profile", "human"),
        "without_timing_features": _metrics(no_time_oof["label"], no_time_prediction),
        "online_policy": online_policy,
        "online_group_oof": online_metrics,
        "online_model_folds": _group_results(prefix_oof, online_policy, "source_id", "model"),
        "online_human_profile_folds": _group_results(prefix_oof, online_policy, "generator_profile", "human"),
        "offline_feature_importance": dict(sorted(zip(FEATURE_COLUMNS, offline_model.feature_importances_), key=lambda pair: pair[1], reverse=True)),
        "online_feature_importance": dict(sorted(zip(FEATURE_COLUMNS, online_model.feature_importances_), key=lambda pair: pair[1], reverse=True)),
        "feature_columns": FEATURE_COLUMNS,
        "time_features": TIME_FEATURES,
        "excluded_signals": ["action_rationale", "model_identity", "prompt_profile", "generator_profile", "exact_interval_roundness"],
        "claim_boundary": "Development results select the detector. Only the later untouched model and behavior profiles are a blind test.",
    }
    metrics_path = output_dir / "optimized_development_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier with validation-frozen behavioral rule",
        "scikit_learn_version": sklearn.__version__,
        "seed": config.seed,
        "training_rows": int(len(browser)),
        "prefix_training_rows": int(len(prefixes)),
        "feature_table_sha256": hashlib.sha256(features_csv.read_bytes()).hexdigest(),
        "raw_events_sha256": hashlib.sha256(raw_events_path.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }
    (output_dir / "optimized_detector_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def evaluate_blind_detector(
    bundle_path: Path,
    blind_events_path: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    bundle = joblib.load(bundle_path)
    events = load_events(blind_events_path)
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)
    features = pd.DataFrame([extract_session_features(rows).model_dump() for rows in grouped.values()])
    columns = bundle["feature_columns"]
    offline_model = bundle["offline_model"]
    offline_policy = bundle["offline_policy"]
    browser = features[features["label"].isin(BROWSER_ORDER)].copy()
    probability = _probability(offline_model, browser, columns)
    detected = _rule(browser)
    if offline_policy["threshold"] is not None:
        detected |= probability >= offline_policy["threshold"]
    prediction = np.where(detected, "ai_agent", "human_like")
    offline_metrics = _metrics(browser["label"], prediction)
    offline_metrics["macro_f1_95_ci"] = _bootstrap(browser["label"], prediction, seed)

    online_model = bundle["online_model"]
    online_policy = bundle["online_policy"]
    warmup = int(bundle["warmup_requests"])
    online_rows = []
    request_rows = []
    for session_id in sorted(grouped):
        session = sorted(grouped[session_id], key=lambda item: item.sequence)
        decision = "allow"
        detection_request = None
        detection_latency = None
        first_time = session[0].virtual_timestamp_ms
        for count in range(1, len(session) + 1):
            probability_value = None
            rule_hit = False
            if session[count - 1].user_agent_id == "sqlmap":
                decision = "block_script"
            elif count >= warmup and online_policy["threshold"] is not None:
                feature = extract_session_features(session[:count]).model_dump()
                vector = pd.DataFrame([{name: feature[name] for name in columns}])
                probability_value = float(_probability(online_model, vector, columns)[0])
                if probability_value >= online_policy["threshold"]:
                    decision = "block_ai"
            request_rows.append(
                {
                    "session_id": session_id,
                    "true_label": session[0].true_label,
                    "request_number": count,
                    "ai_probability": probability_value,
                    "rule_hit": rule_hit,
                    "decision": decision,
                }
            )
            if decision != "allow":
                detection_request = count
                detection_latency = float(session[count - 1].virtual_timestamp_ms - first_time)
                break
        online_rows.append(
            {
                "session_id": session_id,
                "true_label": session[0].true_label,
                "source_id": session[0].source_id,
                "prompt_profile": session[0].prompt_profile,
                "attack_task": session[0].attack_task,
                "generator_profile": session[0].generator_profile,
                "decision": decision,
                "blocked": decision != "allow",
                "detection_request_count": detection_request,
                "detection_to_block_latency_ms": detection_latency,
            }
        )
    online = pd.DataFrame(online_rows)

    def rate(label: str) -> float:
        subset = online[online["true_label"] == label]
        return float(subset["blocked"].mean()) if len(subset) else 0.0

    def blocked_mean(label: str, column: str):
        subset = online[(online["true_label"] == label) & online["blocked"]]
        return float(subset[column].mean()) if len(subset) else None

    online_metrics = {
        "threshold": online_policy["threshold"],
        "warmup_requests": warmup,
        "ai_block_rate": rate("ai_agent"),
        "script_block_rate": rate("script"),
        "human_false_block_rate": rate("human_like"),
        "mean_detection_request_count": {label: blocked_mean(label, "detection_request_count") for label in ("script", "ai_agent", "human_like")},
        "mean_detection_to_block_latency_ms": {label: blocked_mean(label, "detection_to_block_latency_ms") for label in ("script", "ai_agent", "human_like")},
        "evaluated_sessions": int(len(online)),
    }
    online_metrics["ai_block_rate_95_ci"] = _wilson(
        int(online.loc[online["true_label"] == "ai_agent", "blocked"].sum()),
        int((online["true_label"] == "ai_agent").sum()),
    )
    online_metrics["human_false_block_rate_95_ci"] = _wilson(
        int(online.loc[online["true_label"] == "human_like", "blocked"].sum()),
        int((online["true_label"] == "human_like").sum()),
    )
    online_metrics["ai_block_rate_by_prompt_profile"] = {
        str(name): float(group["blocked"].mean())
        for name, group in online[online["true_label"] == "ai_agent"].groupby("prompt_profile")
    }
    online_metrics["ai_block_rate_by_attack_task"] = {
        str(name): float(group["blocked"].mean())
        for name, group in online[online["true_label"] == "ai_agent"].groupby("attack_task")
    }
    online_metrics["script_block_rate_by_profile"] = {
        str(name): float(group["blocked"].mean())
        for name, group in online[online["true_label"] == "script"].groupby("generator_profile")
    }
    online_metrics["human_false_block_rate_by_profile"] = {
        str(name): float(group["blocked"].mean())
        for name, group in online[online["true_label"] == "human_like"].groupby("generator_profile")
    }
    offline_breakdown = {
        "ai_recall_by_prompt_profile": {
            str(name): float((group["prediction"] == "ai_agent").mean())
            for name, group in browser.assign(prediction=prediction)[browser["label"] == "ai_agent"].groupby("prompt_profile")
        },
        "ai_recall_by_attack_task": {
            str(name): float((group["prediction"] == "ai_agent").mean())
            for name, group in browser.assign(prediction=prediction)[browser["label"] == "ai_agent"].groupby("attack_task")
        },
        "human_false_positive_rate_by_profile": {
            str(name): float((group["prediction"] == "ai_agent").mean())
            for name, group in browser.assign(prediction=prediction)[browser["label"] == "human_like"].groupby("generator_profile")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "blind_session_features.csv", index=False)
    browser.assign(ai_probability=probability, prediction=prediction).to_csv(output_dir / "blind_offline_predictions.csv", index=False)
    pd.DataFrame(request_rows).to_csv(output_dir / "blind_online_replay.csv", index=False)
    online.to_csv(output_dir / "blind_online_sessions.csv", index=False)
    metrics = {
        "study_role": "untouched_external_blind_test",
        "sample_counts": features["label"].value_counts().sort_index().to_dict(),
        "event_count": int(len(events)),
        "offline_ai_vs_human": offline_metrics,
        "offline_breakdown": offline_breakdown,
        "online_defense": online_metrics,
        "frozen_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "blind_events_sha256": hashlib.sha256(blind_events_path.read_bytes()).hexdigest(),
        "thresholds_reused_without_blind_tuning": True,
    }
    (output_dir / "blind_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

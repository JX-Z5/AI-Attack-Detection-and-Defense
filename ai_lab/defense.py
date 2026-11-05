from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_session_features
from .schema import EventRecord


def load_events(path: Path) -> list[EventRecord]:
    events: list[EventRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                events.append(EventRecord.model_validate_json(line))
    return events


def _group_events(events: list[EventRecord]) -> dict[str, list[EventRecord]]:
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)
    return {key: sorted(value, key=lambda item: item.sequence) for key, value in grouped.items()}


def replay_defense(
    model,
    events: list[EventRecord],
    session_ids: set[str],
    policy: dict,
    feature_columns: list[str],
    warmup_requests: int,
) -> tuple[dict, pd.DataFrame]:
    grouped = _group_events(events)
    ai_index = list(model.classes_).index("ai_agent")
    session_results: list[dict] = []
    request_rows: list[dict] = []

    for session_id in sorted(session_ids):
        rows = grouped.get(session_id, [])
        if not rows:
            raise ValueError(f"No raw events found for locked-test session {session_id}")
        label = rows[0].true_label
        decision = "allow"
        block_request = None
        block_latency_ms = None
        first_time = rows[0].virtual_timestamp_ms
        observed: list[EventRecord] = []

        for request_number, event in enumerate(rows, start=1):
            observed.append(event)
            ai_probability = None
            if event.user_agent_id == "sqlmap":
                decision = "block_script"
            elif policy.get("mode") == "enforce" and request_number >= warmup_requests:
                feature = extract_session_features(observed).model_dump()
                vector = pd.DataFrame([{name: feature[name] for name in feature_columns}])
                ai_probability = float(model.predict_proba(vector)[0, ai_index])
                if ai_probability >= float(policy["threshold"]):
                    decision = "block_ai"

            request_rows.append(
                {
                    "session_id": session_id,
                    "true_label": label,
                    "request_number": request_number,
                    "virtual_timestamp_ms": event.virtual_timestamp_ms,
                    "ai_probability": ai_probability,
                    "decision": decision,
                }
            )
            if decision != "allow":
                block_request = request_number
                block_latency_ms = float(event.virtual_timestamp_ms - first_time)
                break

        session_results.append(
            {
                "session_id": session_id,
                "true_label": label,
                "decision": decision,
                "blocked": decision != "allow",
                "detection_request_count": block_request,
                "detection_to_block_latency_ms": block_latency_ms,
            }
        )

    sessions = pd.DataFrame(session_results)
    request_log = pd.DataFrame(request_rows)

    def block_rate(label: str) -> float:
        subset = sessions[sessions["true_label"] == label]
        return float(subset["blocked"].mean()) if len(subset) else 0.0

    def blocked_mean(label: str, column: str):
        subset = sessions[(sessions["true_label"] == label) & sessions["blocked"]]
        return float(subset[column].mean()) if len(subset) else None

    metrics = {
        "mode": policy.get("mode", "report_only"),
        "threshold": policy.get("threshold"),
        "warmup_requests": warmup_requests,
        "script_block_rate": block_rate("script"),
        "ai_block_rate": block_rate("ai_agent"),
        "human_false_block_rate": block_rate("human_like"),
        "mean_detection_request_count": {
            label: blocked_mean(label, "detection_request_count")
            for label in ("script", "ai_agent", "human_like")
        },
        "mean_detection_to_block_latency_ms": {
            label: blocked_mean(label, "detection_to_block_latency_ms")
            for label in ("script", "ai_agent", "human_like")
        },
        "evaluated_sessions": int(len(sessions)),
        "processed_requests": int(len(request_log)),
    }
    return metrics, request_log


def choose_online_threshold(
    model,
    events: list[EventRecord],
    validation_ids: set[str],
    feature_columns: list[str],
    warmup_requests: int,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    max_human_false_block_rate: float,
) -> dict:
    grouped = _group_events(events)
    ai_index = list(model.classes_).index("ai_agent")
    validation_rows = []
    for session_id in sorted(validation_ids):
        rows = grouped.get(session_id, [])
        if not rows:
            raise ValueError(f"No raw events found for validation session {session_id}")
        probabilities = []
        for count in range(warmup_requests, len(rows) + 1):
            feature = extract_session_features(rows[:count]).model_dump()
            vector = pd.DataFrame([{name: feature[name] for name in feature_columns}])
            probabilities.append(float(model.predict_proba(vector)[0, ai_index]))
        validation_rows.append(
            {
                "label": rows[0].true_label,
                "max_ai_probability": max(probabilities, default=0.0),
            }
        )
    frame = pd.DataFrame(validation_rows)
    feasible = []
    for threshold in np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step):
        human = frame[frame["label"] == "human_like"]
        ai = frame[frame["label"] == "ai_agent"]
        human_fbr = float((human["max_ai_probability"] >= threshold).mean()) if len(human) else 0.0
        ai_block = float((ai["max_ai_probability"] >= threshold).mean()) if len(ai) else 0.0
        if human_fbr <= max_human_false_block_rate:
            feasible.append((ai_block, -human_fbr, float(threshold)))
    if not feasible:
        return {
            "mode": "report_only",
            "threshold": None,
            "validation_ai_block_rate": 0.0,
            "validation_human_false_block_rate": 0.0,
            "selection_method": "causal_validation_prefixes",
        }
    best = max(feasible)
    return {
        "mode": "enforce",
        "threshold": best[2],
        "validation_ai_block_rate": best[0],
        "validation_human_false_block_rate": -best[1],
        "selection_method": "causal_validation_prefixes",
    }


def write_replay_artifacts(request_log: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_log.to_csv(output_dir / "defense_replay.csv", index=False)


from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import httpx
import pandas as pd

from .config import GeneralizationConfig
from .experiment import _ollama_version
from .features import extract_all_features
from .generators import GatewayClient, build_ai_graph, run_ai_session, run_static_session
from .schema import EventRecord, LabAction


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _available_models(base_url: str) -> set[str]:
    response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=15)
    response.raise_for_status()
    return {str(item["name"]) for item in response.json().get("models", [])}


def _require_models(config: GeneralizationConfig) -> None:
    available = _available_models(config.ollama_base_url)
    missing = [model for model in config.ai_models if model not in available]
    if missing:
        raise RuntimeError(f"Required local models are missing: {missing}")


def _script_actions(rng: random.Random, count: int, profile: str, task: str) -> list[LabAction]:
    sequences = {
        "fixed_fast": ["normal_1", "quote_probe", "boolean_true", "order_probe", "safe_union", "boolean_false"],
        "fixed_slow": ["normal_2", "quote_probe", "order_probe", "boolean_true", "boolean_false", "safe_union"],
        "jitter_low": ["normal_1", "normal_2", "quote_probe", "boolean_true", "boolean_false", "order_probe"],
        "jitter_high": ["quote_probe", "normal_1", "malformed", "boolean_true", "order_probe", "safe_union"],
        "browser_fast": ["normal_1", "quote_probe", "boolean_false", "boolean_true", "safe_union", "order_probe"],
        "browser_slow": ["normal_2", "normal_1", "quote_probe", "order_probe", "boolean_true", "safe_union"],
        "reordered": ["safe_union", "boolean_false", "normal_1", "order_probe", "quote_probe", "boolean_true"],
        "retrying": ["normal_1", "quote_probe", "quote_probe", "boolean_true", "boolean_true", "order_probe"],
    }
    timing = {
        "fixed_fast": (80, 5),
        "fixed_slow": (900, 15),
        "jitter_low": (420, 180),
        "jitter_high": (1200, 1000),
        "browser_fast": (320, 180),
        "browser_slow": (1750, 850),
        "reordered": (650, 420),
        "retrying": (1100, 700),
    }
    payloads = sequences[profile]
    center, spread = timing[profile]
    actions = []
    for index in range(count):
        delay = max(40.0, center + rng.uniform(-spread, spread))
        actions.append(
            LabAction(
                path_id=task,
                payload_id=payloads[index % len(payloads)],
                delay_ms=delay,
                rationale=f"Automated {profile} profile",
            )
        )
    return actions


def _human_actions(rng: random.Random, count: int, profile: str, task: str) -> list[LabAction]:
    timing = {
        "novice": (900, 4200),
        "methodical": (600, 2100),
        "impatient": (180, 850),
        "cautious": (1700, 5600),
        "repetitive": (700, 3100),
        "exploratory": (350, 2600),
        "direct": (220, 1100),
        "distracted": (2200, 7600),
    }
    repeat_probability = {
        "novice": 0.28,
        "methodical": 0.10,
        "impatient": 0.22,
        "cautious": 0.12,
        "repetitive": 0.55,
        "exploratory": 0.08,
        "direct": 0.18,
        "distracted": 0.25,
    }
    candidates = [
        ("home", "none"),
        ("security", "none"),
        (task, "normal_1"),
        (task, "normal_2"),
        (task, "quote_probe"),
        (task, "malformed"),
        (task, "boolean_true"),
        (task, "boolean_false"),
        (task, "order_probe"),
    ]
    low, high = timing[profile]
    current = ("home", "none")
    actions = []
    for _ in range(count):
        if rng.random() < repeat_probability[profile]:
            choice = current
            rationale = f"Repeated a manual trial in the {profile} profile"
        else:
            choice = rng.choice(candidates)
            rationale = f"Selected a manual trial in the {profile} profile"
        current = choice
        actions.append(
            LabAction(
                path_id=choice[0],
                payload_id=choice[1],
                delay_ms=rng.uniform(low, high),
                rationale=rationale,
            )
        )
    return actions


def _read_existing(path: Path) -> tuple[list[EventRecord], set[str]]:
    if not path.exists():
        return [], set()
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                events.append(EventRecord.model_validate_json(line))
    return events, {event.session_id for event in events}


def _append(path: Path, events: list[EventRecord]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")


def run_generalization_experiment(
    config: GeneralizationConfig,
    artifacts_root: Path,
    ai_sessions_per_model: int | None = None,
    script_sessions: int | None = None,
    human_like_sessions: int | None = None,
    fresh: bool = False,
) -> dict:
    ai_count = ai_sessions_per_model or config.ai_sessions_per_model
    script_count = script_sessions or config.script_sessions
    human_count = human_like_sessions or config.human_like_sessions
    raw_dir = artifacts_root / "raw"
    processed_dir = artifacts_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "generalization-events.jsonl"
    if fresh and raw_path.exists():
        raw_path.unlink()
    _require_models(config)
    events, completed = _read_existing(raw_path)
    client = GatewayClient(config.gateway_url)
    graphs = {}
    failures = []
    started = time.time()

    for model_index, model_name in enumerate(config.ai_models):
        for index in range(ai_count):
            profile = config.prompt_profiles[index % len(config.prompt_profiles)]
            task = config.attack_tasks[(index // len(config.prompt_profiles)) % len(config.attack_tasks)]
            session_id = f"ai-{_slug(model_name)}-{index:04d}"
            if session_id in completed:
                continue
            rng = random.Random(config.seed + model_index * 100000 + index * 101)
            request_count = rng.randint(config.min_requests_per_session, config.max_requests_per_session)
            key = (model_name, profile, task)
            if key not in graphs:
                graphs[key] = build_ai_graph(config, client, model_name, profile, task)
            for attempt in range(1, 4):
                try:
                    session_events = run_ai_session(session_id, request_count, graphs[key])
                    events.extend(session_events)
                    _append(raw_path, session_events)
                    completed.add(session_id)
                    break
                except Exception as exc:
                    failures.append({"session_id": session_id, "attempt": attempt, "error": str(exc)})
                    if attempt == 3:
                        raise RuntimeError(f"AI session failed three times: {session_id}: {exc}") from exc
            if (index + 1) % 10 == 0 or index + 1 == ai_count:
                print(f"completed {model_name}: {index + 1}/{ai_count}", flush=True)

    for label, total, profiles in (
        ("script", script_count, config.script_profiles),
        ("human_like", human_count, config.human_profiles),
    ):
        for index in range(total):
            profile = profiles[index % len(profiles)]
            task = config.attack_tasks[(index // len(profiles)) % len(config.attack_tasks)]
            session_id = f"{label.replace('_', '-')}-{_slug(profile)}-{index:04d}"
            if session_id in completed:
                continue
            rng = random.Random(config.seed + (2000000 if label == "script" else 3000000) + index * 101)
            request_count = rng.randint(config.min_requests_per_session, config.max_requests_per_session)
            if label == "script":
                actions = _script_actions(rng, request_count, profile, task)
                ua = "sqlmap" if profile in {"fixed_fast", "fixed_slow", "jitter_low"} else "browser"
                source = "script_generator"
            else:
                actions = _human_actions(rng, request_count, profile, task)
                ua = "browser"
                source = "human_like_simulation"
            session_events = run_static_session(
                label,
                session_id,
                actions,
                client,
                config.time_scale,
                source_id=source,
                prompt_profile="not_applicable",
                attack_task=task,
                generator_profile=profile,
                user_agent_id=ua,
            )
            events.extend(session_events)
            _append(raw_path, session_events)
            completed.add(session_id)
            if (index + 1) % 25 == 0 or index + 1 == total:
                print(f"completed {label}: {index + 1}/{total}", flush=True)

    features = extract_all_features(events)
    frame = pd.DataFrame([row.model_dump() for row in features])
    feature_path = processed_dir / "generalization_session_features.csv"
    frame.to_csv(feature_path, index=False)
    expected = {
        "ai_agent": ai_count * len(config.ai_models),
        "script": script_count,
        "human_like": human_count,
    }
    counts = frame["label"].value_counts().to_dict()
    if counts != expected:
        raise RuntimeError(f"Class balance check failed: expected {expected}, got {counts}")
    metadata = {
        "study_role": "final_generalization_experiment",
        "seed": config.seed,
        "session_counts": counts,
        "event_count": len(events),
        "ai_models": config.ai_models,
        "prompt_profiles": config.prompt_profiles,
        "attack_tasks": config.attack_tasks,
        "script_profiles": config.script_profiles,
        "human_profiles": config.human_profiles,
        "ollama_version": _ollama_version(config.ollama_base_url),
        "ollama_container_image": os.environ.get("OLLAMA_CONTAINER_IMAGE", "not-recorded"),
        "human_class": "human-like simulation",
        "elapsed_seconds_this_run": time.time() - started,
        "failures": failures,
    }
    (processed_dir / "generalization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

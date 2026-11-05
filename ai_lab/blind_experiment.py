from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import BlindConfig
from .features import extract_all_features
from .generalization import _available_models, _read_existing
from .generators import GatewayClient, build_ai_graph, run_ai_session, run_static_session
from .schema import EventRecord, LabAction


def _append(path: Path, events: list[EventRecord]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")


def _script_actions(rng: random.Random, count: int, profile: str, task: str) -> list[LabAction]:
    payloads = ["normal_1", "quote_probe", "boolean_true", "normal_2", "boolean_false", "order_probe", "safe_union"]
    if profile == "adaptive_order":
        rng.shuffle(payloads)
    actions = []
    for index in range(count):
        if profile == "stealth_burst":
            delay = rng.uniform(180, 650) if index % 4 else rng.uniform(1800, 4200)
        else:
            delay = rng.uniform(350, 2400)
        payload = payloads[index % len(payloads)]
        if rng.random() < 0.18 and actions:
            payload = actions[-1].payload_id
        actions.append(
            LabAction(
                path_id=task,
                payload_id=payload,
                delay_ms=delay,
                rationale=f"Blind scripted {profile} action",
            )
        )
    return actions


def _human_actions(rng: random.Random, count: int, profile: str, task: str) -> list[LabAction]:
    candidates = [
        ("home", "none"),
        ("security", "none"),
        (task, "normal_1"),
        (task, "normal_2"),
        (task, "quote_probe"),
        (task, "malformed"),
        (task, "boolean_true"),
        (task, "boolean_false"),
    ]
    actions = []
    current = ("home", "none")
    for index in range(count):
        if profile == "bursty_recovery":
            delay = rng.uniform(180, 850) if index % 3 else rng.uniform(2600, 6500)
            repeat_probability = 0.22
        else:
            delay = rng.uniform(1100, 6800)
            repeat_probability = 0.16
        if rng.random() < repeat_probability:
            choice = current
        elif rng.random() < 0.22:
            choice = ("home", "none") if current[0] == task else ("security", "none")
        else:
            choice = rng.choice(candidates)
        current = choice
        actions.append(
            LabAction(
                path_id=choice[0],
                payload_id=choice[1],
                delay_ms=delay,
                rationale=f"Blind human-like {profile} action",
            )
        )
    return actions


def run_blind_experiment(config: BlindConfig, artifacts_root: Path, fresh: bool = False) -> dict:
    available = _available_models(config.ollama_base_url)
    if config.ai_model not in available:
        raise RuntimeError(f"Blind model is missing: {config.ai_model}")
    raw_dir = artifacts_root / "raw"
    processed_dir = artifacts_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "blind-events.jsonl"
    if fresh and raw_path.exists():
        raw_path.unlink()
    events, completed = _read_existing(raw_path)
    client = GatewayClient(config.gateway_url)
    graphs = {}
    failures = []
    started = time.time()

    for index in range(config.ai_sessions):
        profile = config.prompt_profiles[index % len(config.prompt_profiles)]
        task = config.attack_tasks[(index // len(config.prompt_profiles)) % len(config.attack_tasks)]
        session_id = f"blind-ai-falcon3-{index:04d}"
        if session_id in completed:
            continue
        rng = random.Random(config.seed + index * 101)
        request_count = rng.randint(config.min_requests_per_session, config.max_requests_per_session)
        key = (profile, task)
        if key not in graphs:
            graphs[key] = build_ai_graph(config, client, config.ai_model, profile, task)
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
                    raise RuntimeError(f"Blind AI session failed three times: {session_id}: {exc}") from exc
        if (index + 1) % 10 == 0 or index + 1 == config.ai_sessions:
            print(f"completed blind {config.ai_model}: {index + 1}/{config.ai_sessions}", flush=True)

    for label, total, profiles in (
        ("script", config.script_sessions, config.script_profiles),
        ("human_like", config.human_like_sessions, config.human_profiles),
    ):
        for index in range(total):
            profile = profiles[index % len(profiles)]
            task = config.attack_tasks[(index // len(profiles)) % len(config.attack_tasks)]
            session_id = f"blind-{label.replace('_', '-')}-{profile}-{index:04d}"
            if session_id in completed:
                continue
            rng = random.Random(config.seed + (1000000 if label == "script" else 2000000) + index * 101)
            request_count = rng.randint(config.min_requests_per_session, config.max_requests_per_session)
            actions = _script_actions(rng, request_count, profile, task) if label == "script" else _human_actions(rng, request_count, profile, task)
            session_events = run_static_session(
                label,
                session_id,
                actions,
                client,
                config.time_scale,
                source_id="blind_script_generator" if label == "script" else "blind_human_like_simulation",
                prompt_profile="not_applicable",
                attack_task=task,
                generator_profile=profile,
                user_agent_id="browser",
            )
            events.extend(session_events)
            _append(raw_path, session_events)
            completed.add(session_id)
            if (index + 1) % 25 == 0 or index + 1 == total:
                print(f"completed blind {label}: {index + 1}/{total}", flush=True)

    frame = pd.DataFrame([feature.model_dump() for feature in extract_all_features(events)])
    feature_path = processed_dir / "blind_session_features.csv"
    frame.to_csv(feature_path, index=False)
    expected = {"ai_agent": config.ai_sessions, "script": config.script_sessions, "human_like": config.human_like_sessions}
    counts = frame["label"].value_counts().to_dict()
    if counts != expected:
        raise RuntimeError(f"Blind class balance check failed: expected {expected}, got {counts}")
    metadata = {
        "study_role": "untouched_external_blind_test",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "session_counts": counts,
        "event_count": len(events),
        "ai_model": config.ai_model,
        "prompt_profiles": config.prompt_profiles,
        "attack_tasks": config.attack_tasks,
        "script_profiles": config.script_profiles,
        "human_profiles": config.human_profiles,
        "human_class": "human-like simulation",
        "elapsed_seconds_this_run": time.time() - started,
        "failures": failures,
    }
    (processed_dir / "blind_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

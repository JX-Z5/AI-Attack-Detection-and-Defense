from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd

from .config import ExperimentConfig
from .features import extract_all_features
from .generators import GatewayClient, build_ai_graph, generate_session
from .schema import EventRecord


def _model_available(base_url: str, model: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        names = {item["name"] for item in response.json().get("models", [])}
        return model in names or any(name.startswith(model + ":") for name in names)
    except Exception:
        return False


def select_model(config: ExperimentConfig) -> str:
    for model in (config.ollama_model, config.ollama_fallback_model):
        if _model_available(config.ollama_base_url, model):
            return model
    raise RuntimeError("Neither the primary nor fallback Ollama model is available")


def _ollama_version(base_url: str) -> str:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=10)
        response.raise_for_status()
        return str(response.json().get("version", "unknown"))
    except Exception:
        return "unknown"


def run_experiment(config: ExperimentConfig, artifacts_root: Path, sessions_per_class: int | None = None) -> dict:
    count = sessions_per_class or config.sessions_per_class
    raw_dir = artifacts_root / "raw"
    processed_dir = artifacts_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "experiment-events.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    model_name = select_model(config)
    client = GatewayClient(config.gateway_url)
    ai_graph = build_ai_graph(config, client, model_name)
    events: list[EventRecord] = []
    started = time.time()
    failures: list[dict] = []
    for label in ("script", "ai_agent", "human_like"):
        for index in range(count):
            for attempt in range(1, 3):
                try:
                    session_events = generate_session(label, index, config, client, ai_graph)
                    events.extend(session_events)
                    with raw_path.open("a", encoding="utf-8") as stream:
                        for event in session_events:
                            stream.write(event.model_dump_json() + "\n")
                    if (index + 1) % 5 == 0 or index + 1 == count:
                        print(f"completed {label}: {index + 1}/{count}", flush=True)
                    break
                except Exception as exc:
                    failures.append({"label": label, "index": index, "attempt": attempt, "error": str(exc)})
                    if attempt == 2:
                        raise RuntimeError(
                            f"Experiment aborted because {label} session {index} failed twice: {exc}"
                        ) from exc

    features = extract_all_features(events)
    frame = pd.DataFrame([row.model_dump() for row in features])
    feature_path = processed_dir / "session_features.csv"
    frame.to_csv(feature_path, index=False)
    metadata = {
        "seed": config.seed,
        "configured_sessions_per_class": count,
        "completed_sessions": frame["label"].value_counts().sort_index().to_dict(),
        "event_count": len(events),
        "model": model_name,
        "ollama_version": _ollama_version(config.ollama_base_url),
        "ollama_container_image": os.environ.get("OLLAMA_CONTAINER_IMAGE", "not-recorded"),
        "human_class": "human-like simulation",
        "elapsed_seconds": time.time() - started,
        "failures": failures,
    }
    expected_counts = {label: count for label in ("script", "ai_agent", "human_like")}
    if metadata["completed_sessions"] != expected_counts:
        raise RuntimeError(
            f"Class balance check failed: expected {expected_counts}, got {metadata['completed_sessions']}"
        )
    (processed_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


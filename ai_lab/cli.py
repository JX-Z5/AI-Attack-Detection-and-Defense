from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from .charts import generate_charts
from .blind_experiment import run_blind_experiment
from .config import load_blind_config, load_config, load_generalization_config
from .experiment import run_experiment
from .generalization import run_generalization_experiment
from .generalization_charts import generate_generalization_charts
from .generalization_modeling import train_generalization
from .modeling import train_and_evaluate
from .optimized_modeling import evaluate_blind_detector, train_optimized_detector
from .validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("config/experiment.yaml")
DEFAULT_GENERALIZATION_CONFIG = Path("config/generalization.yaml")
DEFAULT_BLIND_CONFIG = Path("config/blind.yaml")
DEFAULT_ARTIFACTS = Path("artifacts")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, check=check, text=True)


def _verify_compose_safety(docker: str) -> None:
    result = subprocess.run(
        [docker, "compose", "config", "--format", "json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    compose = json.loads(result.stdout)
    for name in ("target_net", "runner_net", "agent_net"):
        if not compose.get("networks", {}).get(name, {}).get("internal"):
            raise RuntimeError(f"Compose network is not internal: {name}")
    for service_name, service in compose.get("services", {}).items():
        for port in service.get("ports", []) or []:
            if port.get("host_ip") != "127.0.0.1":
                raise RuntimeError(f"Unsafe published port on {service_name}: {port}")


def command_setup(args) -> None:
    docker = args.docker
    config = load_generalization_config(_project_path(args.config))
    _run([docker, "version"])
    _run([docker, "compose", "version"])
    _verify_compose_safety(docker)
    _run([docker, "compose", "pull", "db", "dvwa", "ollama"])
    _run([docker, "compose", "build", "gateway"])
    attempts = []
    downloaded = []
    for model_name in config.ai_models:
        if not re.fullmatch(r"[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?", model_name):
            raise ValueError(f"Invalid Ollama model name: {model_name}")
        model_script = (
            "ollama serve >/tmp/ollama.log 2>&1 & pid=$!; sleep 3; "
            "ollama pull \"$MODEL\" && ollama run \"$MODEL\" \"Return only OK\" >/tmp/model-test.txt; "
            "code=$?; kill $pid; exit $code"
        )
        result = _run([
            docker, "run", "--rm", "--entrypoint", "/bin/sh",
            "-e", f"MODEL={model_name}",
            "-v", "ai_attack_lab_ollama:/root/.ollama", "alpine/ollama:latest", "-c", model_script,
        ], check=False)
        attempts.append({"model": model_name, "exit_code": result.returncode})
        if result.returncode == 0:
            downloaded.append(model_name)
        else:
            raise RuntimeError(f"Unable to download required local model: {model_name}")
    metadata = {"status": "ready", "models": downloaded, "attempts": attempts}
    setup_path = _project_path(DEFAULT_ARTIFACTS) / "processed" / "setup_metadata.json"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def command_run_experiment(args) -> None:
    config = load_config(_project_path(args.config))
    metadata = run_experiment(config, _project_path(args.artifacts), args.sessions_per_class)
    print(json.dumps(metadata, indent=2))


def command_train(args) -> None:
    config = load_config(_project_path(args.config))
    artifacts = _project_path(args.artifacts)
    metrics = train_and_evaluate(
        artifacts / "processed" / "session_features.csv",
        artifacts / "results",
        config,
        artifacts / "raw" / "experiment-events.jsonl",
    )
    generate_charts(
        artifacts / "results" / "metrics.json",
        artifacts / "processed" / "session_features.csv",
        artifacts / "results" / "figures",
    )
    validate_artifacts(artifacts)
    print(json.dumps(metrics, indent=2))


def command_run_generalization(args) -> None:
    config = load_generalization_config(_project_path(args.config))
    metadata = run_generalization_experiment(
        config,
        _project_path(args.artifacts),
        args.ai_sessions_per_model,
        args.script_sessions,
        args.human_like_sessions,
        args.fresh,
    )
    print(json.dumps(metadata, indent=2))


def command_train_generalization(args) -> None:
    config = load_generalization_config(_project_path(args.config))
    artifacts = _project_path(args.artifacts)
    metrics = train_generalization(
        artifacts / "processed" / "generalization_session_features.csv",
        artifacts / "raw" / "generalization-events.jsonl",
        artifacts / "results",
        config,
    )
    generate_generalization_charts(
        artifacts / "results" / "generalization_metrics.json",
        artifacts / "processed" / "generalization_session_features.csv",
        artifacts / "results" / "figures",
    )
    print(json.dumps(metrics, indent=2))


def command_train_optimized(args) -> None:
    config = load_generalization_config(_project_path(args.config))
    source = _project_path(args.source_artifacts)
    metrics = train_optimized_detector(
        source / "processed" / "generalization_session_features.csv",
        source / "raw" / "generalization-events.jsonl",
        _project_path(args.output),
        config,
    )
    print(json.dumps(metrics, indent=2))


def command_run_blind(args) -> None:
    config = load_blind_config(_project_path(args.config))
    metadata = run_blind_experiment(config, _project_path(args.artifacts), args.fresh)
    print(json.dumps(metadata, indent=2))


def command_evaluate_blind(args) -> None:
    config = load_blind_config(_project_path(args.config))
    artifacts = _project_path(args.artifacts)
    metrics = evaluate_blind_detector(
        _project_path(args.bundle),
        artifacts / "raw" / "blind-events.jsonl",
        artifacts / "results",
        config.seed,
    )
    print(json.dumps(metrics, indent=2))


def command_demo(args) -> None:
    artifacts = _project_path(args.artifacts)
    config = load_config(_project_path(args.config))
    replay_path = artifacts / "results" / "defense_replay.csv"
    metrics = json.loads((artifacts / "results" / "metrics.json").read_text(encoding="utf-8"))
    import httpx
    import pandas as pd
    try:
        status = httpx.get(f"{config.gateway_url.rstrip('/')}/defense-status", timeout=5).json()
    except Exception:
        status = {"enabled": False, "model_loaded": False}

    if status.get("enabled") and status.get("model_loaded"):
        from .experiment import select_model
        from .generators import GatewayClient, build_ai_graph, generate_session

        client = GatewayClient(config.gateway_url)
        graph = build_ai_graph(config, client, select_model(config))
        print("Live defense gateway:", json.dumps(status, indent=2))
        for label in ("script", "ai_agent", "human_like"):
            events = generate_session(label, 900 + len(label), config, client, graph)
            print(f"\n{label}")
            for event in events:
                print(
                    f"  request={event.sequence + 1} status={event.status_code} "
                    f"marker={event.response_marker} decision={event.gateway_decision}"
                )
                if event.gateway_decision != "allow":
                    break
        return

    data = pd.read_csv(replay_path)
    print("Validation-selected policy:", json.dumps(metrics["defense_policy"], indent=2))
    print("Locked-test online replay:", json.dumps(metrics["defense_test"], indent=2))
    for label in ("script", "ai_agent", "human_like"):
        session_id = data[data["true_label"] == label]["session_id"].iloc[0]
        print(f"\n{label} / {session_id}")
        for row in data[data["session_id"] == session_id].itertuples():
            probability = "n/a" if pd.isna(row.ai_probability) else f"{row.ai_probability:.3f}"
            print(
                f"  request={row.request_number} virtual_ms={row.virtual_timestamp_ms:.1f} "
                f"ai_probability={probability} decision={row.decision}"
            )


def command_build_report(args) -> None:
    from .reporting import build_report
    output = build_report(_project_path(args.artifacts), _project_path(args.output))
    print(json.dumps({"docx": str(output)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI attack detection and defense lab")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="Check Docker, pull images, build services, and download the local model")
    setup.add_argument("--docker", default=os.environ.get("DOCKER_EXE", "docker"))
    setup.add_argument("--config", default=DEFAULT_GENERALIZATION_CONFIG.as_posix())
    setup.set_defaults(func=command_setup)

    run = sub.add_parser("run-experiment", help="Run the three-class controlled experiment")
    run.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    run.add_argument("--artifacts", default=DEFAULT_ARTIFACTS.as_posix())
    run.add_argument("--sessions-per-class", type=int)
    run.set_defaults(func=command_run_experiment)

    train = sub.add_parser("train", help="Train the classifier and generate evaluation figures")
    train.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    train.add_argument("--artifacts", default=DEFAULT_ARTIFACTS.as_posix())
    train.set_defaults(func=command_train)

    generalization = sub.add_parser("run-generalization", help="Run the grouped multi-model generalization experiment")
    generalization.add_argument("--config", default=DEFAULT_GENERALIZATION_CONFIG.as_posix())
    generalization.add_argument("--artifacts", default=(DEFAULT_ARTIFACTS / "generalization").as_posix())
    generalization.add_argument("--ai-sessions-per-model", type=int)
    generalization.add_argument("--script-sessions", type=int)
    generalization.add_argument("--human-like-sessions", type=int)
    generalization.add_argument("--fresh", action="store_true")
    generalization.set_defaults(func=command_run_generalization)

    generalization_train = sub.add_parser("train-generalization", help="Evaluate unseen model, prompt, and task groups")
    generalization_train.add_argument("--config", default=DEFAULT_GENERALIZATION_CONFIG.as_posix())
    generalization_train.add_argument("--artifacts", default=(DEFAULT_ARTIFACTS / "generalization").as_posix())
    generalization_train.set_defaults(func=command_train_generalization)

    optimized_train = sub.add_parser("train-optimized", help="Freeze the group-robust detector before blind evaluation")
    optimized_train.add_argument("--config", default=DEFAULT_GENERALIZATION_CONFIG.as_posix())
    optimized_train.add_argument("--source-artifacts", default=(DEFAULT_ARTIFACTS / "generalization").as_posix())
    optimized_train.add_argument("--output", default=(DEFAULT_ARTIFACTS / "optimization" / "development").as_posix())
    optimized_train.set_defaults(func=command_train_optimized)

    blind = sub.add_parser("run-blind", help="Run the untouched model and behavior-profile evaluation")
    blind.add_argument("--config", default=DEFAULT_BLIND_CONFIG.as_posix())
    blind.add_argument("--artifacts", default=(DEFAULT_ARTIFACTS / "optimization" / "blind").as_posix())
    blind.add_argument("--fresh", action="store_true")
    blind.set_defaults(func=command_run_blind)

    blind_evaluation = sub.add_parser("evaluate-blind", help="Evaluate the frozen detector without retuning")
    blind_evaluation.add_argument("--config", default=DEFAULT_BLIND_CONFIG.as_posix())
    blind_evaluation.add_argument("--artifacts", default=(DEFAULT_ARTIFACTS / "optimization" / "blind").as_posix())
    blind_evaluation.add_argument("--bundle", default=(DEFAULT_ARTIFACTS / "optimization" / "development" / "optimized_detector.joblib").as_posix())
    blind_evaluation.set_defaults(func=command_evaluate_blind)

    demo = sub.add_parser("demo-defense", help="Show held-out defense decisions for each class")
    demo.add_argument("--artifacts", default=DEFAULT_ARTIFACTS.as_posix())
    demo.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    demo.set_defaults(func=command_demo)

    report = sub.add_parser("build-report", help="Build the English DOCX report from the recorded experiment")
    report.add_argument("--artifacts", default=DEFAULT_ARTIFACTS.as_posix())
    report.add_argument("--output", default=(DEFAULT_ARTIFACTS / "report" / "AI_Attack_Detection_and_Defense_Report.docx").as_posix())
    report.set_defaults(func=command_build_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# AI Attack Detection and Defense

This repository contains an isolated web-security experiment for detecting AI-controlled attack sessions from request behavior. Local language models interact with DVWA through an allowlisted gateway. The project records request events, extracts session features, trains random-forest detectors, evaluates transfer to unseen model families, and applies a frozen online blocking policy.

The comparison traffic consists of deterministic scripts and software-generated human-like simulations. No real human participants were used.

## Repository structure

| Path | Contents |
|---|---|
| `ai_lab/` | Python source code for traffic generation, the allowlisted gateway, feature extraction, model training, defense replay, blind evaluation, and result validation. |
| `config/` | YAML files containing the parameters for the pilot, six-model development experiment, and Falcon3 blind evaluation. |
| `tests/` | Automated tests for the safety restrictions, deterministic generators, feature calculations, model serialization, and evaluation pipeline. |
| `artifacts/raw/` | Request-level logs from the 150-session pilot experiment. |
| `artifacts/processed/` | Session features and metadata produced from the pilot logs. |
| `artifacts/results/` | Pilot models, predictions, metrics, defense replay results, and figures. |
| `artifacts/generalization/` | Raw events, features, trained model, predictions, metrics, and figures from the 1,800-session six-model experiment. |
| `artifacts/optimization/development/` | The frozen detector and out-of-fold predictions used before the blind experiment. |
| `artifacts/optimization/blind/` | Raw events, features, predictions, and metrics from the 300-session Falcon3 blind evaluation. |

The files under `artifacts/` are saved experiment outputs. JSONL files contain individual request events, CSV files contain session features or predictions, JSON files contain metadata and metrics, JOBLIB files contain trained models, and PNG files contain result figures. The pilot files are kept as a record of the first complete pipeline run. The main findings use the six-model development experiment and the separate Falcon3 blind evaluation.

## Dataset

The formal dataset contains 2,100 sessions and 14,139 requests.

| Partition | AI agent | Script | Human-like simulation | Requests |
|---|---:|---:|---:|---:|
| Development and generalization | 600 | 600 | 600 | 11,726 |
| Untouched blind evaluation | 100 | 100 | 100 | 2,413 |
| Total | 700 | 700 | 700 | 14,139 |

The development AI sessions contain 100 sessions from each of six Ollama models: `llama3.2:3b`, `gemma3:4b`, `phi4-mini:3.8b`, `mistral:7b`, `granite3.3:2b`, and `smollm2:1.7b`. The untouched blind set uses `falcon3:3b`.

Development sessions use four prompt profiles (`methodical`, `cautious`, `exploratory`, and `adaptive`) and two attack tasks (`sqli` and `sqli_blind`). Script and human-like sessions use separate behavior profiles defined in `config/generalization.yaml`. The blind set uses new script and human-like profiles defined in `config/blind.yaml`.

The repository also retains a 150-session, 1,190-request pilot under `artifacts/raw`, `artifacts/processed`, and `artifacts/results`. It records the initial pipeline run and is separate from the 2,100-session formal dataset.

## Experiment parameters

| Parameter | Development | Blind evaluation |
|---|---:|---:|
| Random seed | 20261102 | 20261103 |
| AI sessions | 600 | 100 |
| Script sessions | 600 | 100 |
| Human-like sessions | 600 | 100 |
| Requests per session | 5 to 8 | 6 to 10 |
| AI temperature | 0.65 | 0.75 |
| Maximum model retries | 2 | 2 |
| Defense warm-up | 4 requests | 4 requests |
| Maximum human-like false-block target | 10% | 10% |

The frozen detector is a `RandomForestClassifier` with 700 trees, minimum leaf size 5, `max_features=0.7`, `class_weight="balanced_subsample"`, and seed 20261102. It uses 25 session-level behavioral features. The offline and online thresholds are both 0.79 and were selected before the Falcon3 blind run.

Recorded blind results are stored in `artifacts/optimization/blind/results/blind_metrics.json`. The offline Falcon3-versus-human-like macro F1 is 0.808. The online policy blocks 85% of Falcon3 sessions and records 0 false blocks among 100 human-like simulations.

## Requirements

- Docker with Linux-container support
- Python 3.11 or later
- Ollama model storage for the seven listed models
- NVIDIA Container Toolkit only when using `compose.gpu.yaml`

All published service ports bind to `127.0.0.1`. The target, runner, gateway, and Ollama services communicate through internal Docker networks.

## Reproduction

Run the following commands from the repository root.

```console
python -m pip install -e ".[test]"
python -m pytest
python -m ai_lab.cli setup --config config/generalization.yaml
docker compose up -d db dvwa ollama gateway
docker compose run --rm runner python -m ai_lab.cli run-experiment --config config/experiment.yaml --artifacts artifacts
docker compose run --rm runner python -m ai_lab.cli train --config config/experiment.yaml --artifacts artifacts
docker compose run --rm runner python -m ai_lab.cli run-generalization --config config/generalization.yaml --artifacts artifacts/generalization --fresh
docker compose run --rm runner python -m ai_lab.cli train-generalization --config config/generalization.yaml --artifacts artifacts/generalization
docker compose run --rm runner python -m ai_lab.cli train-optimized --config config/generalization.yaml --source-artifacts artifacts/generalization --output artifacts/optimization/development
docker compose exec ollama ollama pull falcon3:3b
docker compose run --rm runner python -m ai_lab.cli run-blind --config config/blind.yaml --artifacts artifacts/optimization/blind --fresh
docker compose run --rm runner python -m ai_lab.cli evaluate-blind --config config/blind.yaml --artifacts artifacts/optimization/blind --bundle artifacts/optimization/development/optimized_detector.joblib
```

For an NVIDIA GPU environment, add the override file when starting the services:

```console
docker compose -f compose.yaml -f compose.gpu.yaml up -d db dvwa ollama gateway
```

Stop the laboratory with:

```console
docker compose down
```

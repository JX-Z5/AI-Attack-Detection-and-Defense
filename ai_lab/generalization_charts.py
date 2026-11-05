from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COLORS = {"ai_agent": "#c44e52", "script": "#4c72b0", "human_like": "#55a868"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_generalization_charts(metrics_path: Path, features_path: Path, output_dir: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(features_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    folds = metrics["model_family_holdout"]["folds"]
    names = list(folds)
    scores = [folds[name]["macro_f1"] for name in names]
    lower = [score - folds[name]["macro_f1_95_ci"]["lower_95"] for name, score in zip(names, scores)]
    upper = [folds[name]["macro_f1_95_ci"]["upper_95"] - score for name, score in zip(names, scores)]
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    axis.bar(range(len(names)), scores, color="#4c72b0", yerr=[lower, upper], capsize=4)
    axis.set_xticks(range(len(names)), [name.split(":")[0] for name in names], rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro F1")
    axis.set_title("Unseen model-family tests")
    _save(fig, output_dir / "model_family_holdout.png")

    summaries = {
        "Model family": metrics["model_family_holdout"]["summary"],
        "Prompt profile": metrics["prompt_profile_holdout"]["summary"],
        "Attack task": metrics["attack_task_holdout"]["summary"],
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = range(len(summaries))
    means = [item["mean_macro_f1"] for item in summaries.values()]
    worst = [item["worst_macro_f1"] for item in summaries.values()]
    width = 0.34
    axis.bar([position - width / 2 for position in positions], means, width, label="Mean", color="#4c72b0")
    axis.bar([position + width / 2 for position in positions], worst, width, label="Worst fold", color="#dd8452")
    axis.set_xticks(list(positions), list(summaries))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro F1")
    axis.set_title("Grouped external evaluations")
    axis.legend()
    _save(fig, output_dir / "grouped_evaluation_summary.png")

    matrix = metrics["locked_test"]["random_forest"]["confusion_matrix"]
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-AI", "AI agent"], yticklabels=["Non-AI", "AI agent"], ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Locked unseen-model test")
    _save(fig, output_dir / "locked_confusion_matrix.png")

    importance = pd.Series(metrics["feature_importance"]).sort_values().tail(10)
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    importance.plot.barh(ax=axis, color="#8172b3")
    axis.set_xlabel("Random forest importance")
    axis.set_title("Features in the locked model")
    _save(fig, output_dir / "generalization_feature_importance.png")

    selected = ["mean_interval_ms", "repetition_rate", "unique_payload_ratio", "unique_path_ratio"]
    melted = frame.melt(id_vars=["label"], value_vars=selected, var_name="feature", value_name="value")
    figure = sns.catplot(
        data=melted,
        x="label",
        y="value",
        col="feature",
        kind="box",
        col_wrap=2,
        sharey=False,
        palette=COLORS,
        height=3.0,
        aspect=1.25,
    )
    figure.set_axis_labels("", "Value")
    figure.set_titles("{col_name}")
    figure.fig.subplots_adjust(top=0.90)
    figure.fig.suptitle("Behavior features by generator class")
    figure.savefig(output_dir / "behavior_feature_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(figure.fig)

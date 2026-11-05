from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .modeling import LABEL_ORDER


PALETTE = ["#355C7D", "#C06C84", "#6C9A8B"]


def generate_charts(metrics_path: Path, features_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    features = pd.read_csv(features_path)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    created: list[Path] = []

    cm = metrics["random_forest"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax)
    ax.set_title("Random forest confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    importance = pd.Series(metrics["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    importance.plot.barh(ax=ax, color="#355C7D")
    ax.set_title("Random forest feature importance")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    sns.boxplot(data=features, x="label", y="mean_interval_ms", order=LABEL_ORDER, palette=PALETTE, hue="label", legend=False, ax=axes[0])
    axes[0].set_title("Mean request interval")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Milliseconds")
    sns.boxplot(data=features, x="label", y="repetition_rate", order=LABEL_ORDER, palette=PALETTE, hue="label", legend=False, ax=axes[1])
    axes[1].set_title("Consecutive path repetition")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Rate")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output_dir / "behavior_distributions.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    created.append(path)
    return created


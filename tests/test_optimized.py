import pandas as pd

from ai_lab.config import GeneralizationConfig
from ai_lab.optimized_modeling import _choose_threshold, _rule


def config():
    return GeneralizationConfig(
        ai_models=["a", "b", "c"],
        prompt_profiles=["p"],
        attack_tasks=["sqli"],
        script_profiles=["s"],
        human_profiles=["h1", "h2"],
    )


def test_behavior_rule_uses_only_observable_features():
    frame = pd.DataFrame(
        [
            {"mean_interval_ms": 400, "repetition_rate": 0.8, "unique_payload_ratio": 0.2},
            {"mean_interval_ms": 900, "repetition_rate": 0.8, "unique_payload_ratio": 0.2},
        ]
    )
    assert _rule(frame).tolist() == [True, False]


def test_threshold_respects_each_human_profile_limit():
    frame = pd.DataFrame(
        [
            {"label": "ai_agent", "generator_profile": "a", "ai_probability": 0.9, "rule_hit": False},
            {"label": "ai_agent", "generator_profile": "a", "ai_probability": 0.8, "rule_hit": False},
            {"label": "human_like", "generator_profile": "h1", "ai_probability": 0.4, "rule_hit": False},
            {"label": "human_like", "generator_profile": "h2", "ai_probability": 0.3, "rule_hit": False},
        ]
    )
    policy = _choose_threshold(frame, config())
    assert policy["mode"] == "enforce"
    assert policy["oof_worst_human_profile_false_positive_rate"] <= 0.10

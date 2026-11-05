import random

import pandas as pd

from ai_lab.generalization import _human_actions, _script_actions
from ai_lab.generalization_modeling import FEATURE_COLUMNS, GROUP_COLUMNS, _model_holdout_masks


def test_profiled_generators_are_deterministic_and_varied():
    first = _script_actions(random.Random(7), 6, "jitter_high", "sqli")
    second = _script_actions(random.Random(7), 6, "jitter_high", "sqli")
    human = _human_actions(random.Random(8), 8, "exploratory", "sqli_blind")
    assert first == second
    assert len({item.delay_ms for item in first}) > 1
    assert all(item.path_id in {"home", "security", "sqli_blind"} for item in human)


def test_group_metadata_is_not_a_feature():
    assert not set(FEATURE_COLUMNS) & set(GROUP_COLUMNS)


def test_model_holdout_has_no_ai_source_leakage():
    models = ["a", "b", "c", "d"]
    scripts = [f"s{i}" for i in range(8)]
    humans = [f"h{i}" for i in range(8)]
    rows = []
    for model in models:
        rows.append({"session_id": f"ai-{model}", "label": "ai_agent", "source_id": model, "generator_profile": model})
    for profile in scripts:
        rows.append({"session_id": f"script-{profile}", "label": "script", "source_id": "script", "generator_profile": profile})
    for profile in humans:
        rows.append({"session_id": f"human-{profile}", "label": "human_like", "source_id": "human", "generator_profile": profile})
    frame = pd.DataFrame(rows)
    name, train_mask, validation_mask, test_mask = next(_model_holdout_masks(frame, models, scripts, humans))
    assert name == "a"
    assert "a" not in set(frame.loc[train_mask, "source_id"])
    assert "a" not in set(frame.loc[validation_mask, "source_id"])
    assert set(frame.loc[test_mask & (frame["label"] == "ai_agent"), "source_id"]) == {"a"}

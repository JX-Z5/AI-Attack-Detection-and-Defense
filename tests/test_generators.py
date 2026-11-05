import random

from ai_lab.generators import _human_like_actions, _script_actions


def test_script_generator_is_deterministic():
    assert _script_actions(random.Random(7), 6) == _script_actions(random.Random(7), 6)


def test_human_like_generator_is_deterministic_and_slow():
    actions = _human_like_actions(random.Random(7), 8)
    assert actions == _human_like_actions(random.Random(7), 8)
    assert min(action.delay_ms for action in actions) >= 900


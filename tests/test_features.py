from ai_lab.features import extract_session_features
from ai_lab.schema import EventRecord


def event(sequence, ts, path_id="sqli", payload_id="normal_1", status=200, marker="query_result", ua="browser"):
    return EventRecord(
        session_id="human_like-001",
        true_label="human_like",
        sequence=sequence,
        virtual_timestamp_ms=ts,
        wall_timestamp="2026-10-25T00:00:00+00:00",
        path_id=path_id,
        path="/vulnerabilities/sqli/",
        payload_id=payload_id,
        status_code=status,
        response_bytes=100,
        response_marker=marker,
        user_agent_id=ua,
        user_agent="test",
        action_rationale="test",
    )


def test_feature_formulas():
    result = extract_session_features([
        event(0, 1000),
        event(1, 3000),
        event(2, 6000, path_id="missing", payload_id="none", status=404, marker="http_404"),
    ])
    assert result.request_count == 3
    assert result.mean_interval_ms == 2500
    assert round(result.requests_per_second, 3) == 0.6
    assert round(result.error_rate, 3) == 0.333
    assert round(result.unique_path_ratio, 3) == 0.667


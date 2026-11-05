from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import fmean, pstdev

from .schema import EventRecord, SessionFeatures


PATH_ORDER = {"home": 0, "security": 1, "sqli": 2, "sqli_blind": 3, "missing": 4}


def _entropy(values: list) -> float:
    if not values:
        return 0.0
    counts = list(Counter(values).values())
    total = sum(counts)
    return -sum((count / total) * math.log2(count / total) for count in counts)


def _sequence_score(events: list[EventRecord]) -> float:
    if len(events) < 2:
        return 0.0
    positions = [PATH_ORDER.get(event.path_id, 99) for event in events]
    orderly = sum(1 for left, right in zip(positions, positions[1:]) if right >= left)
    return orderly / (len(positions) - 1)


def extract_session_features(events: list[EventRecord]) -> SessionFeatures:
    if not events:
        raise ValueError("At least one event is required")
    ordered = sorted(events, key=lambda event: event.sequence)
    times = [event.virtual_timestamp_ms for event in ordered]
    intervals = [right - left for left, right in zip(times, times[1:])]
    duration = max(times[-1] - times[0], 1.0) if len(times) > 1 else 1.0
    paths = [event.path_id for event in ordered]
    payloads = [event.payload_id for event in ordered]
    markers = [event.response_marker for event in ordered]
    response_sizes = [event.response_bytes for event in ordered]
    ua_anomalies = sum(event.user_agent_id != "browser" for event in ordered)
    errors = sum(event.status_code >= 400 or event.response_marker in {"sql_error"} for event in ordered)
    repetitions = sum(left == right for left, right in zip(paths, paths[1:]))
    payload_changes = [left != right for left, right in zip(payloads, payloads[1:])]
    marker_changes = [left != right for left, right in zip(markers, markers[1:])]
    changed_payloads = [change for change, marker_change in zip(payload_changes, marker_changes) if marker_change]
    unchanged_payloads = [change for change, marker_change in zip(payload_changes, marker_changes) if not marker_change]
    mean_interval = fmean(intervals) if intervals else 0.0
    sorted_intervals = sorted(intervals)
    lower_index = int((len(sorted_intervals) - 1) * 0.1) if sorted_intervals else 0
    upper_index = int((len(sorted_intervals) - 1) * 0.9) if sorted_intervals else 0
    mean_response = fmean(response_sizes)
    return SessionFeatures(
        session_id=ordered[0].session_id,
        label=ordered[0].true_label,
        source_id=ordered[0].source_id,
        prompt_profile=ordered[0].prompt_profile,
        attack_task=ordered[0].attack_task,
        generator_profile=ordered[0].generator_profile,
        request_count=len(ordered),
        mean_interval_ms=mean_interval,
        std_interval_ms=pstdev(intervals) if len(intervals) > 1 else 0.0,
        requests_per_second=len(ordered) / (duration / 1000.0),
        duration_ms=duration,
        unique_path_ratio=len(set(paths)) / len(paths),
        path_sequence_score=_sequence_score(ordered),
        error_rate=errors / len(ordered),
        repetition_rate=repetitions / max(len(ordered) - 1, 1),
        unique_payload_ratio=len(set(payloads)) / len(payloads),
        ua_anomaly_rate=ua_anomalies / len(ordered),
        mean_response_bytes=mean_response,
        interval_cv=(pstdev(intervals) / mean_interval) if len(intervals) > 1 and mean_interval else 0.0,
        interval_range_80=(sorted_intervals[upper_index] - sorted_intervals[lower_index]) if sorted_intervals else 0.0,
        payload_entropy=_entropy(payloads),
        payload_transition_entropy=_entropy(list(zip(payloads, payloads[1:]))),
        payload_switch_rate=fmean(payload_changes) if payload_changes else 0.0,
        max_payload_share=max(Counter(payloads).values()) / len(payloads),
        marker_entropy=_entropy(markers),
        response_size_cv=pstdev(response_sizes) / mean_response if len(response_sizes) > 1 and mean_response else 0.0,
        response_size_range=max(response_sizes) - min(response_sizes),
        response_unique_ratio=len(set(response_sizes)) / len(response_sizes),
        marker_change_rate=fmean(marker_changes) if marker_changes else 0.0,
        switch_after_marker_change=fmean(changed_payloads) if changed_payloads else 0.0,
        switch_after_same_marker=fmean(unchanged_payloads) if unchanged_payloads else 0.0,
    )


def extract_all_features(events: list[EventRecord]) -> list[SessionFeatures]:
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)
    return [extract_session_features(grouped[key]) for key in sorted(grouped)]


from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AttackLabel = Literal["script", "ai_agent", "human_like"]


class LabAction(BaseModel):
    path_id: str
    payload_id: str
    delay_ms: float = Field(ge=0, le=30000)
    rationale: str = Field(default="", max_length=4000)


class GatewayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=r"^[a-z0-9_-]{3,80}$")
    true_label: AttackLabel
    sequence: int = Field(ge=0, le=100)
    path_id: str
    payload_id: str
    user_agent_id: str
    virtual_timestamp_ms: float = Field(ge=0)
    action_rationale: str = Field(default="", max_length=4000)
    source_id: str = Field(default="legacy", max_length=80)
    prompt_profile: str = Field(default="legacy", max_length=80)
    attack_task: str = Field(default="sqli", max_length=80)
    generator_profile: str = Field(default="legacy", max_length=80)


class EventRecord(BaseModel):
    session_id: str
    true_label: AttackLabel
    sequence: int
    virtual_timestamp_ms: float
    wall_timestamp: str
    path_id: str
    path: str
    payload_id: str
    status_code: int
    response_bytes: int
    response_marker: str
    user_agent_id: str
    user_agent: str
    action_rationale: str
    gateway_decision: str = "allow"
    source_id: str = "legacy"
    prompt_profile: str = "legacy"
    attack_task: str = "sqli"
    generator_profile: str = "legacy"

    @classmethod
    def now(cls, **kwargs):
        return cls(wall_timestamp=datetime.now(timezone.utc).isoformat(), **kwargs)


class SessionFeatures(BaseModel):
    session_id: str
    label: AttackLabel
    source_id: str = "legacy"
    prompt_profile: str = "legacy"
    attack_task: str = "sqli"
    generator_profile: str = "legacy"
    request_count: int
    mean_interval_ms: float
    std_interval_ms: float
    requests_per_second: float
    duration_ms: float
    unique_path_ratio: float
    path_sequence_score: float
    error_rate: float
    repetition_rate: float
    unique_payload_ratio: float
    ua_anomaly_rate: float
    mean_response_bytes: float
    interval_cv: float
    interval_range_80: float
    payload_entropy: float
    payload_transition_entropy: float
    payload_switch_rate: float
    max_payload_share: float
    marker_entropy: float
    response_size_cv: float
    response_size_range: float
    response_unique_ratio: float
    marker_change_rate: float
    switch_after_marker_change: float
    switch_after_same_marker: float


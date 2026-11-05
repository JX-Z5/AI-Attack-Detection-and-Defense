from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    test_size: float = Field(default=0.20, gt=0, lt=0.5)
    validation_size_of_remaining: float = Field(default=0.25, gt=0, lt=0.5)


class DefenseConfig(BaseModel):
    warmup_requests: int = Field(default=3, ge=2)
    max_human_like_false_block_rate: float = Field(default=0.10, ge=0, le=1)
    threshold_min: float = Field(default=0.50, ge=0, le=1)
    threshold_max: float = Field(default=0.95, ge=0, le=1)
    threshold_step: float = Field(default=0.01, gt=0, le=0.25)


class ExperimentConfig(BaseModel):
    seed: int = 20261025
    sessions_per_class: int = Field(default=50, ge=2)
    min_requests_per_session: int = Field(default=6, ge=3)
    max_requests_per_session: int = Field(default=10, ge=3)
    time_scale: float = Field(default=0.0, ge=0, le=1)
    ollama_model: str = "llama3.2:3b"
    ollama_fallback_model: str = "llama3.2:1b"
    ollama_base_url: str = "http://ollama:11434"
    gateway_url: str = "http://gateway:8080"
    ai_temperature: float = Field(default=0.35, ge=0, le=2)
    ai_max_retries: int = Field(default=2, ge=0, le=5)
    split: SplitConfig = SplitConfig()
    defense: DefenseConfig = DefenseConfig()

    def model_post_init(self, __context: Any) -> None:
        if self.min_requests_per_session > self.max_requests_per_session:
            raise ValueError("min_requests_per_session cannot exceed max_requests_per_session")
        if self.defense.threshold_min > self.defense.threshold_max:
            raise ValueError("threshold_min cannot exceed threshold_max")


class GeneralizationConfig(BaseModel):
    seed: int = 20261102
    ai_sessions_per_model: int = Field(default=100, ge=2)
    script_sessions: int = Field(default=600, ge=2)
    human_like_sessions: int = Field(default=600, ge=2)
    min_requests_per_session: int = Field(default=5, ge=3)
    max_requests_per_session: int = Field(default=8, ge=3)
    time_scale: float = Field(default=0.0, ge=0, le=1)
    ollama_base_url: str = "http://ollama:11434"
    gateway_url: str = "http://gateway:8080"
    ai_temperature: float = Field(default=0.65, ge=0, le=2)
    ai_max_retries: int = Field(default=2, ge=0, le=5)
    ai_models: list[str]
    prompt_profiles: list[str]
    attack_tasks: list[str]
    script_profiles: list[str]
    human_profiles: list[str]
    defense: DefenseConfig = DefenseConfig()

    def model_post_init(self, __context: Any) -> None:
        if self.min_requests_per_session > self.max_requests_per_session:
            raise ValueError("min_requests_per_session cannot exceed max_requests_per_session")
        fields = [self.ai_models, self.prompt_profiles, self.attack_tasks, self.script_profiles, self.human_profiles]
        if any(not values for values in fields):
            raise ValueError("Generalization group lists cannot be empty")
        if len(self.ai_models) < 3:
            raise ValueError("At least three AI model families are required")


class BlindConfig(BaseModel):
    seed: int = 20261103
    ai_model: str
    ai_sessions: int = Field(default=100, ge=2)
    script_sessions: int = Field(default=100, ge=2)
    human_like_sessions: int = Field(default=100, ge=2)
    min_requests_per_session: int = Field(default=6, ge=4)
    max_requests_per_session: int = Field(default=10, ge=4)
    time_scale: float = Field(default=0.0, ge=0, le=1)
    ollama_base_url: str = "http://ollama:11434"
    gateway_url: str = "http://gateway:8080"
    ai_temperature: float = Field(default=0.75, ge=0, le=2)
    ai_max_retries: int = Field(default=2, ge=0, le=5)
    prompt_profiles: list[str]
    attack_tasks: list[str]
    script_profiles: list[str]
    human_profiles: list[str]
    defense: DefenseConfig = DefenseConfig()

    def model_post_init(self, __context: Any) -> None:
        if self.min_requests_per_session > self.max_requests_per_session:
            raise ValueError("min_requests_per_session cannot exceed max_requests_per_session")
        fields = [self.prompt_profiles, self.attack_tasks, self.script_profiles, self.human_profiles]
        if any(not values for values in fields):
            raise ValueError("Blind evaluation group lists cannot be empty")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if os.environ.get("OLLAMA_BASE_URL"):
        values["ollama_base_url"] = os.environ["OLLAMA_BASE_URL"]
    if os.environ.get("AI_LAB_GATEWAY_URL"):
        values["gateway_url"] = os.environ["AI_LAB_GATEWAY_URL"]
    return ExperimentConfig.model_validate(values)


def load_generalization_config(path: str | Path) -> GeneralizationConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if os.environ.get("OLLAMA_BASE_URL"):
        values["ollama_base_url"] = os.environ["OLLAMA_BASE_URL"]
    if os.environ.get("AI_LAB_GATEWAY_URL"):
        values["gateway_url"] = os.environ["AI_LAB_GATEWAY_URL"]
    return GeneralizationConfig.model_validate(values)


def load_blind_config(path: str | Path) -> BlindConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if os.environ.get("OLLAMA_BASE_URL"):
        values["ollama_base_url"] = os.environ["OLLAMA_BASE_URL"]
    if os.environ.get("AI_LAB_GATEWAY_URL"):
        values["gateway_url"] = os.environ["AI_LAB_GATEWAY_URL"]
    return BlindConfig.model_validate(values)


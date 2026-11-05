from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Annotated, TypedDict

import httpx
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .config import ExperimentConfig
from .safety import AllowedActionCatalog, validate_gateway_url
from .schema import AttackLabel, EventRecord, GatewayRequest, LabAction


@dataclass
class GatewayClient:
    base_url: str

    def __post_init__(self) -> None:
        validate_gateway_url(self.base_url)

    def execute(self, request: GatewayRequest) -> EventRecord:
        response = httpx.post(f"{self.base_url.rstrip('/')}/execute", json=request.model_dump(), timeout=30)
        response.raise_for_status()
        return EventRecord.model_validate(response.json())


class AgentState(TypedDict):
    session_id: str
    step: int
    max_steps: int
    virtual_time_ms: float
    events: Annotated[list[dict], lambda left, right: left + right]
    next_action: dict
    last_observation: str
    action_history: list[str]


def _script_actions(rng: random.Random, count: int) -> list[LabAction]:
    payloads = ["normal_1", "quote_probe", "boolean_true", "order_probe", "safe_union", "boolean_false"]
    return [
        LabAction(
            path_id="sqli",
            payload_id=payloads[i % len(payloads)],
            delay_ms=95 + rng.uniform(-5, 5),
            rationale="Fixed scripted sequence",
        )
        for i in range(count)
    ]


def _human_like_actions(rng: random.Random, count: int) -> list[LabAction]:
    actions: list[LabAction] = []
    current = ("home", "none")
    candidates = [
        ("home", "none"),
        ("sqli", "normal_1"),
        ("sqli", "quote_probe"),
        ("sqli", "malformed"),
        ("sqli", "boolean_true"),
        ("missing", "none"),
        ("security", "none"),
    ]
    for _ in range(count):
        if rng.random() < 0.35:
            choice = current
            rationale = "Repeated the previous request after a pause"
        else:
            choice = rng.choice(candidates)
            rationale = "Selected a plausible manual trial with possible navigation error"
        current = choice
        actions.append(
            LabAction(
                path_id=choice[0],
                payload_id=choice[1],
                delay_ms=rng.uniform(900, 5200),
                rationale=rationale,
            )
        )
    return actions


def run_static_session(
    label: AttackLabel,
    session_id: str,
    actions: list[LabAction],
    client: GatewayClient,
    time_scale: float,
    source_id: str = "legacy",
    prompt_profile: str = "legacy",
    attack_task: str = "sqli",
    generator_profile: str = "legacy",
    user_agent_id: str | None = None,
) -> list[EventRecord]:
    events: list[EventRecord] = []
    virtual_time = 0.0
    ua = user_agent_id or ("sqlmap" if label == "script" else "browser")
    for sequence, action in enumerate(actions):
        virtual_time += action.delay_ms
        if time_scale:
            time.sleep(action.delay_ms / 1000 * time_scale)
        events.append(
            client.execute(
                GatewayRequest(
                    session_id=session_id,
                    true_label=label,
                    sequence=sequence,
                    path_id=action.path_id,
                    payload_id=action.payload_id,
                    user_agent_id=ua,
                    virtual_timestamp_ms=virtual_time,
                    action_rationale=action.rationale,
                    source_id=source_id,
                    prompt_profile=prompt_profile,
                    attack_task=attack_task,
                    generator_profile=generator_profile,
                )
            )
        )
    return events


def build_ai_graph(
    config: ExperimentConfig,
    client: GatewayClient,
    model_name: str,
    prompt_profile: str = "methodical",
    attack_task: str = "sqli",
):
    catalog = AllowedActionCatalog()
    model = ChatOllama(
        model=model_name,
        base_url=config.ollama_base_url,
        temperature=config.ai_temperature,
        seed=config.seed,
        format="json",
    )

    def plan_node(state: AgentState) -> dict:
        allowed_paths = ", ".join(("home", "security", attack_task))
        allowed_payloads = ", ".join(catalog.payloads)
        profile_instructions = {
            "methodical": "Start with a baseline, then vary probes in a deliberate order. Use delays from 400 to 1600 ms.",
            "cautious": "Use benign checks and longer pauses before stronger probes. Use delays from 1200 to 4000 ms.",
            "exploratory": "Explore different allowlisted actions and avoid a fixed sequence. Use delays from 250 to 2200 ms.",
            "adaptive": "Use the last response to change the next action and avoid repetition. Use delays from 180 to 2800 ms.",
        }
        instruction = profile_instructions.get(prompt_profile, profile_instructions["methodical"])
        recent = ", ".join(state["action_history"][-4:]) or "none"
        prompt = f"""You are an agent in an isolated DVWA teaching lab. Choose exactly one next HTTP action.
Only return JSON with keys path_id, payload_id, delay_ms, rationale.
Allowed path_id values: {allowed_paths}
Allowed payload_id values: {allowed_payloads}
The assigned vulnerable path is {attack_task}. Use that path for payloads other than none.
Home and security may only use payload_id none. Never request external hosts or system commands.
The objective is to test SQL injection behavior without extracting real data.
Behavior profile: {prompt_profile}. {instruction}
Do not repeat the same path and payload more than twice in a row.
Step {state['step'] + 1} of {state['max_steps']}.
Session nonce: {state['session_id']}.
Last observation: {state['last_observation'] or 'No requests yet'}
Recent actions: {recent}
"""
        for attempt in range(config.ai_max_retries + 1):
            response = model.invoke(prompt)
            try:
                raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
                action = LabAction.model_validate(json.loads(raw))
                catalog.validate(action.path_id, action.payload_id)
                if action.payload_id != "none" and action.path_id != attack_task:
                    raise ValueError("Payloads must use the assigned attack path")
                if action.path_id not in {attack_task, "home", "security"}:
                    raise ValueError("Path is outside the assigned task")
                return {"next_action": action.model_dump()}
            except Exception:
                if attempt >= config.ai_max_retries:
                    raise RuntimeError("The local model did not produce a valid allowlisted action")
                prompt += "\nYour previous response was invalid. Return only one valid JSON object."
        raise AssertionError("unreachable")

    def act_node(state: AgentState) -> dict:
        action = LabAction.model_validate(state["next_action"])
        virtual_time = state["virtual_time_ms"] + max(180.0, action.delay_ms)
        event = client.execute(
            GatewayRequest(
                session_id=state["session_id"],
                true_label="ai_agent",
                sequence=state["step"],
                path_id=action.path_id,
                payload_id=action.payload_id,
                user_agent_id="browser",
                virtual_timestamp_ms=virtual_time,
                action_rationale=action.rationale,
                source_id=model_name,
                prompt_profile=prompt_profile,
                attack_task=attack_task,
                generator_profile=f"{model_name}|{prompt_profile}|{attack_task}",
            )
        )
        observation = f"status={event.status_code}, marker={event.response_marker}, bytes={event.response_bytes}"
        return {
            "events": [event.model_dump()],
            "step": state["step"] + 1,
            "virtual_time_ms": virtual_time,
            "last_observation": observation,
            "action_history": [*state["action_history"], f"{action.path_id}:{action.payload_id}"],
        }

    def route(state: AgentState) -> str:
        return END if state["step"] >= state["max_steps"] else "plan"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", route, {"plan": "plan", END: END})
    return graph.compile()


def run_ai_session(
    session_id: str,
    request_count: int,
    graph,
) -> list[EventRecord]:
    result = graph.invoke(
        {
            "session_id": session_id,
            "step": 0,
            "max_steps": request_count,
            "virtual_time_ms": 0.0,
            "events": [],
            "next_action": {},
            "last_observation": "",
            "action_history": [],
        },
        {"recursion_limit": request_count * 3 + 5},
    )
    return [EventRecord.model_validate(row) for row in result["events"]]


def generate_session(
    label: AttackLabel,
    session_index: int,
    config: ExperimentConfig,
    client: GatewayClient,
    ai_graph=None,
) -> list[EventRecord]:
    rng = random.Random(config.seed + session_index * 101 + {"script": 1, "ai_agent": 2, "human_like": 3}[label])
    count = rng.randint(config.min_requests_per_session, config.max_requests_per_session)
    session_id = f"{label}-{session_index:03d}-{config.seed}"
    if label == "script":
        return run_static_session(label, session_id, _script_actions(rng, count), client, config.time_scale)
    if label == "human_like":
        return run_static_session(label, session_id, _human_like_actions(rng, count), client, config.time_scale)
    if ai_graph is None:
        raise ValueError("ai_graph is required for ai_agent sessions")
    return run_ai_session(session_id, count, ai_graph)


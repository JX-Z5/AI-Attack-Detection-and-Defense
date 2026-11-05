from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

from .features import extract_session_features
from .modeling import FEATURE_COLUMNS
from .safety import resolve_path, resolve_payload, resolve_user_agent
from .schema import EventRecord, GatewayRequest


DVWA_BASE_URL = os.environ.get("DVWA_BASE_URL", "http://dvwa").rstrip("/")
LOG_PATH = Path(os.environ.get("GATEWAY_LOG_PATH", "/app/artifacts/raw/gateway-events.jsonl"))
app = FastAPI(title="AI Attack Lab Safety Gateway", docs_url=None, redoc_url=None)
_sessions: dict[str, requests.Session] = {}
_lock = threading.Lock()
_database_ready = False
_histories: dict[str, list[EventRecord]] = {}
_blocked_sessions: dict[str, str] = {}
_detector = None
_detector_metadata = None
DEFENSE_ENABLED = os.environ.get("DEFENSE_ENABLED", "false").lower() == "true"
MODEL_PATH = Path(os.environ.get("DEFENSE_MODEL_PATH", "/app/artifacts/results/random_forest.joblib"))
MODEL_METADATA_PATH = Path(os.environ.get("DEFENSE_METADATA_PATH", "/app/artifacts/results/model_metadata.json"))


def _token(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find("input", {"name": "user_token"})
    return node.get("value", "") if node else ""


def _wait_for_dvwa(timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{DVWA_BASE_URL}/login.php", timeout=3).status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("DVWA did not become ready")


def _prepare_session() -> requests.Session:
    global _database_ready
    _wait_for_dvwa()
    session = requests.Session()
    session.headers.update({"User-Agent": "AI-Lab-Gateway/1.0"})

    if not _database_ready:
        setup = session.get(f"{DVWA_BASE_URL}/setup.php", timeout=10)
        session.post(
            f"{DVWA_BASE_URL}/setup.php",
            data={"create_db": "Create / Reset Database", "user_token": _token(setup.text)},
            timeout=20,
        )
        _database_ready = True

    login = session.get(f"{DVWA_BASE_URL}/login.php", timeout=10)
    response = session.post(
        f"{DVWA_BASE_URL}/login.php",
        data={
            "username": "admin",
            "password": "password",
            "Login": "Login",
            "user_token": _token(login.text),
        },
        timeout=15,
    )
    if "login.php" in response.url:
        raise RuntimeError("DVWA login failed")
    security = session.get(f"{DVWA_BASE_URL}/security.php", timeout=10)
    session.post(
        f"{DVWA_BASE_URL}/security.php",
        data={"security": "low", "seclev_submit": "Submit", "user_token": _token(security.text)},
        timeout=10,
    )
    return session


def _session_for(session_id: str) -> requests.Session:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = _prepare_session()
        return _sessions[session_id]


def _response_marker(text: str, status: int) -> str:
    lower = text.lower()
    if status >= 400:
        return f"http_{status}"
    if "sql syntax" in lower or "mysql" in lower and "error" in lower:
        return "sql_error"
    if "surname" in lower or "first name" in lower:
        return "query_result"
    if "login" in lower and "username" in lower:
        return "login_page"
    return "ordinary"


def _append_event(event: EventRecord) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")


def _load_detector():
    global _detector, _detector_metadata
    if not DEFENSE_ENABLED:
        return None, None
    if _detector is None and MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
        _detector = joblib.load(MODEL_PATH)
        _detector_metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return _detector, _detector_metadata


def _blocked_event(request: GatewayRequest, path: str, user_agent: str, decision: str) -> EventRecord:
    return EventRecord.now(
        session_id=request.session_id,
        true_label=request.true_label,
        sequence=request.sequence,
        virtual_timestamp_ms=request.virtual_timestamp_ms,
        path_id=request.path_id,
        path=path,
        payload_id=request.payload_id,
        status_code=429,
        response_bytes=0,
        response_marker="blocked_by_defense",
        user_agent_id=request.user_agent_id,
        user_agent=user_agent,
        action_rationale=request.action_rationale,
        gateway_decision=decision,
        source_id=request.source_id,
        prompt_profile=request.prompt_profile,
        attack_task=request.attack_task,
        generator_profile=request.generator_profile,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "target": "dvwa"}


@app.get("/defense-status")
def defense_status() -> dict:
    model, metadata = _load_detector()
    return {
        "enabled": DEFENSE_ENABLED,
        "model_loaded": model is not None,
        "mode": metadata.get("defense_policy", {}).get("mode") if metadata else None,
        "threshold": metadata.get("defense_policy", {}).get("threshold") if metadata else None,
    }


@app.post("/execute", response_model=EventRecord)
def execute(request: GatewayRequest) -> EventRecord:
    try:
        path = resolve_path(request.path_id)
        payload = resolve_payload(request.payload_id)
        user_agent = resolve_user_agent(request.user_agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    model, metadata = _load_detector()
    if model is not None:
        previous_decision = _blocked_sessions.get(request.session_id)
        if previous_decision:
            event = _blocked_event(request, path, user_agent, previous_decision)
            _append_event(event)
            return event
        if request.user_agent_id == "sqlmap":
            _blocked_sessions[request.session_id] = "block_script"
            event = _blocked_event(request, path, user_agent, "block_script")
            _append_event(event)
            return event

    session = _session_for(request.session_id)
    headers = {"User-Agent": user_agent}
    params = None
    if request.path_id in {"sqli", "sqli_blind"}:
        params = {"id": payload, "Submit": "Submit"}
    try:
        response = session.get(f"{DVWA_BASE_URL}{path}", params=params, headers=headers, timeout=15)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Lab target request failed: {exc}") from exc

    event = EventRecord.now(
        session_id=request.session_id,
        true_label=request.true_label,
        sequence=request.sequence,
        virtual_timestamp_ms=request.virtual_timestamp_ms,
        path_id=request.path_id,
        path=path,
        payload_id=request.payload_id,
        status_code=response.status_code,
        response_bytes=len(response.content),
        response_marker=_response_marker(response.text, response.status_code),
        user_agent_id=request.user_agent_id,
        user_agent=user_agent,
        action_rationale=request.action_rationale,
        source_id=request.source_id,
        prompt_profile=request.prompt_profile,
        attack_task=request.attack_task,
        generator_profile=request.generator_profile,
    )
    if model is not None and metadata is not None:
        history = _histories.setdefault(request.session_id, [])
        history.append(event.model_copy(deep=True))
        policy = metadata.get("defense_policy", {})
        warmup = int(os.environ.get("DEFENSE_WARMUP_REQUESTS", "3"))
        if policy.get("mode") == "enforce" and len(history) >= warmup:
            feature = extract_session_features(history).model_dump()
            frame = pd.DataFrame([{name: feature[name] for name in FEATURE_COLUMNS}])
            ai_index = list(model.classes_).index("ai_agent")
            ai_probability = float(model.predict_proba(frame)[0, ai_index])
            if ai_probability >= float(policy["threshold"]):
                _blocked_sessions[request.session_id] = "block_ai"
                event.status_code = 429
                event.response_bytes = 0
                event.response_marker = "blocked_by_defense"
                event.gateway_decision = "block_ai"
    _append_event(event)
    return event


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


if __name__ == "__main__":
    main()


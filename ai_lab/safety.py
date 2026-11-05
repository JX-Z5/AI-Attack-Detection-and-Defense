from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


PATHS = {
    "home": "/",
    "sqli": "/vulnerabilities/sqli/",
    "sqli_blind": "/vulnerabilities/sqli_blind/",
    "security": "/security.php",
    "missing": "/lab-intentionally-missing",
}

PAYLOADS = {
    "normal_1": "1",
    "normal_2": "2",
    "quote_probe": "1'",
    "boolean_true": "1' OR '1'='1' #",
    "boolean_false": "1' AND '1'='2' #",
    "order_probe": "1' ORDER BY 1 #",
    "safe_union": "1' UNION SELECT 'lab','safe' #",
    "malformed": "1'\"",
    "none": "",
}

USER_AGENTS = {
    "browser": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "sqlmap": "sqlmap/1.8.11#stable",
    "agent": "AI-Lab-Agent/1.0",
}


class SafetyViolation(ValueError):
    pass


def resolve_path(path_id: str) -> str:
    try:
        return PATHS[path_id]
    except KeyError as exc:
        raise SafetyViolation(f"Path is not allowlisted: {path_id}") from exc


def resolve_payload(payload_id: str) -> str:
    try:
        return PAYLOADS[payload_id]
    except KeyError as exc:
        raise SafetyViolation(f"Payload is not allowlisted: {payload_id}") from exc


def resolve_user_agent(user_agent_id: str) -> str:
    try:
        return USER_AGENTS[user_agent_id]
    except KeyError as exc:
        raise SafetyViolation(f"User-Agent is not allowlisted: {user_agent_id}") from exc


def validate_gateway_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SafetyViolation("Gateway URL must use HTTP or HTTPS")
    if parsed.hostname not in {"gateway", "127.0.0.1", "localhost"}:
        raise SafetyViolation(f"Gateway host is outside the lab allowlist: {parsed.hostname}")


@dataclass(frozen=True)
class AllowedActionCatalog:
    paths: tuple[str, ...] = tuple(PATHS)
    payloads: tuple[str, ...] = tuple(PAYLOADS)

    def validate(self, path_id: str, payload_id: str) -> None:
        resolve_path(path_id)
        resolve_payload(payload_id)


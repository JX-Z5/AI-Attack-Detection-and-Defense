import pytest
from pydantic import ValidationError

from ai_lab.safety import SafetyViolation, resolve_path, resolve_payload, validate_gateway_url
from ai_lab.schema import GatewayRequest


def test_allowlisted_values_resolve():
    assert resolve_path("sqli") == "/vulnerabilities/sqli/"
    assert resolve_payload("boolean_true").startswith("1'")


def test_unknown_path_and_payload_are_rejected():
    with pytest.raises(SafetyViolation):
        resolve_path("https://example.com")
    with pytest.raises(SafetyViolation):
        resolve_payload("arbitrary-shell-command")


def test_external_gateway_is_rejected():
    with pytest.raises(SafetyViolation):
        validate_gateway_url("https://example.com")


def test_gateway_schema_rejects_arbitrary_url_field():
    with pytest.raises(ValidationError):
        GatewayRequest.model_validate(
            {
                "session_id": "test-session",
                "true_label": "script",
                "sequence": 0,
                "path_id": "sqli",
                "payload_id": "normal_1",
                "user_agent_id": "sqlmap",
                "virtual_timestamp_ms": 95,
                "url": "https://example.com",
            }
        )


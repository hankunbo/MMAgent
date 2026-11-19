"""A safe diagnostic hint, not permission to auto-retry an upstream refusal."""

import json

from packages.llm_provider.http_recovery import VERSION, retryable
from packages.llm_provider.worker import safe_http_error


def test_observed_provider_capacity_message_does_not_echo_or_change_retry_policy():
    raw = json.dumps(
        {
            "error": {
                "type": "new_api_error",
                "message": (
                    "当前分组上游负载已饱和，请稍后再试 private-patient-value sk-test-secret"
                ),
            }
        },
        ensure_ascii=False,
    ).encode()
    value = safe_http_error(429, raw, key="sk-test-secret")
    assert value["message_tokens"] == ["upstream_capacity_saturated"]
    assert value["category"] == "unclassified_429"
    assert "private-patient-value" not in json.dumps(value)
    assert "sk-test-secret" not in json.dumps(value)
    assert not retryable(
        dict(value, schema_version=VERSION, origin="http_worker", operation="chat_completions")
    )


def test_other_errors_are_not_misclassified_as_confirmed_capacity_failure():
    assert safe_http_error(400, b'{"error":{"message":"Bad request"}}')["message_tokens"] == []
    assert safe_http_error(429, b'{"error":{"message":"Please wait"}}')["message_tokens"] == []

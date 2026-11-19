"""One classification used at retry and durable reservation boundaries.

Only HTTPTransport produces these observations from worker output. Caller Reply
flags and free-form diagnostics do not authorize retries. Persistence is owned by
RuntimeAgentJournal, not by a model or external caller.
"""

from datetime import UTC
from email.utils import parsedate_to_datetime

VERSION = "agent-http-observation/1.0.0"


def retryable(event):
    return (
        event is not None
        and event.get("schema_version") == VERSION
        and event.get("origin") == "http_worker"
        and event.get("operation") == "chat_completions"
        and event.get("category")
        not in {"insufficient_quota", "request_format_rejected", "secret_error_omitted"}
        and (
            event.get("http_status") in {500, 502, 503, 504}
            or (event.get("http_status") == 429 and event.get("category") == "rate_limit")
        )
    )


def retry_delay(event, *, now, default):
    raw = event.get("retry_after")
    if raw is None:
        return default
    try:
        seconds = float(raw)
        if seconds < 0 or not seconds < float("inf"):
            raise ValueError()
    except ValueError:
        try:
            at = parsedate_to_datetime(raw)
            seconds = (at.astimezone(UTC) - now).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return default
    return max(default, seconds, 0.0)


def observation_for(journal, request_id):
    matches = [
        e
        for e in journal.read()
        if e.get("schema_version") == VERSION and e.get("request_id") == request_id
    ]
    return matches[0] if len(matches) == 1 else None


def format_rejected(event):
    return (
        event is not None
        and event.get("schema_version") == VERSION
        and event.get("origin") == "http_worker"
        and event.get("operation") == "chat_completions"
        and event.get("http_status") in {400, 422, 429}
        and event.get("category") == "request_format_rejected"
        and (
            event.get("error_type") == "invalid_request_error"
            or event.get("error_code") == "invalid_json_schema"
        )
    )

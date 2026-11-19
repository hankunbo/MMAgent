"""Secrets resolved only here; no redirects, implicit proxies, retries or raw errors."""

import http.client
import json
import os
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

from packages.knowledge_claims.ports import LLMConfig

from .network_route import route_opener


def safe_transport_error(error, stage):
    """Closed vocabulary only: never serialize exception messages, URLs or credentials."""
    cause = error.reason if isinstance(error, urllib.error.URLError) else error
    category = (
        "socket_timeout"
        if isinstance(cause, TimeoutError)
        else "dns_error"
        if isinstance(cause, socket.gaierror)
        else "tls_error"
        if isinstance(cause, ssl.SSLError)
        # RemoteDisconnected also inherits ConnectionResetError. Test the more
        # specific response-header failure first; old saved events stay untouched.
        else "remote_disconnected"
        if isinstance(cause, http.client.RemoteDisconnected)
        else "connection_reset"
        if isinstance(cause, ConnectionResetError)
        else "connection_refused"
        if isinstance(cause, ConnectionRefusedError)
        else "incomplete_response"
        if isinstance(cause, http.client.IncompleteRead)
        else "invalid_json_response"
        if isinstance(cause, json.JSONDecodeError)
        else "connection_error"
        if isinstance(cause, ConnectionError)
        else "unknown_transport"
    )
    return dict(
        schema_version="safe-transport-failure/1.1.0",
        category=category,
        stage=stage
        if stage in {"before_dispatch", "dispatch_boundary", "response_headers", "response_body"}
        else "unknown",
        remote_execution="unknown",
        automatic_replay_authorized=False,
        errno=cause.errno
        if isinstance(cause, OSError) and type(cause.errno) is int and -1000 <= cause.errno <= 1000
        else None,
        tls_reason=getattr(cause, "reason", None)
        if isinstance(cause, ssl.SSLError)
        and getattr(cause, "reason", None)
        in {
            "CERTIFICATE_VERIFY_FAILED",
            "UNEXPECTED_EOF_WHILE_READING",
            "SSLV3_ALERT_HANDSHAKE_FAILURE",
            "TLSV1_ALERT_PROTOCOL_VERSION",
            "WRONG_VERSION_NUMBER",
            "DECRYPTION_FAILED_OR_BAD_RECORD_MAC",
        }
        else None,
        certificate_verify_code=getattr(cause, "verify_code", None)
        if isinstance(cause, ssl.SSLCertVerificationError)
        and type(getattr(cause, "verify_code", None)) is int
        and 0 <= cause.verify_code <= 100
        else None,
    )


def safe_http_error(status, raw, headers=None, key="", schema=None, body=None):
    headers = headers or {}
    """Only fixed categories and known wire fields leave this boundary, never raw text."""
    fields = (
        "max_completion_tokens",
        "response_format",
        "json_schema",
        "store",
        "image_url",
        "prefixItems",
        "items",
        "enum",
        "minItems",
        "maxItems",
        "pattern",
        "$ref",
    )
    lower = raw.decode("utf-8", errors="replace").lower()
    mentioned = [name for name in fields if name.lower() in lower]
    category = (
        "authentication"
        if status == 401
        else "model_permission"
        if status == 403
        else "model_or_route_unavailable"
        if status == 404
        else "rate_limit"
        if status == 429
        else "unsupported_parameter"
        if status in {400, 422} and mentioned
        else "request_rejected"
        if 400 <= status < 500
        else "remote_service_error"
    )
    from packages.object_store import checksum

    try:
        error = json.loads(raw).get("error", {})
        error = error if isinstance(error, dict) else {}
    except (ValueError, AttributeError):
        error = {}
    # Closed vocabularies: no arbitrary echoed user text leaves this boundary.
    safe_codes = {
        "rate_limit_exceeded",
        "insufficient_quota",
        "invalid_api_key",
        "invalid_request_error",
        "invalid_json_schema",
        "server_error",
        "internal_server_error",
        "upstream_error",
        "model_not_found",
        "unsupported_parameter",
    }
    code = error.get("code") if error.get("code") in safe_codes else None
    kind = error.get("type") if error.get("type") in safe_codes else None
    quota = any(
        t in lower for t in ("insufficient_quota", "quota exceeded", "余额不足", "额度不足")
    )
    if quota:
        category = "insufficient_quota"
    elif status == 429 and code != "rate_limit_exceeded":
        category = "unclassified_429"
    request_id = headers.get("x-request-id")
    if not (
        isinstance(request_id, str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id)
        and not (key and key in request_id)
    ):
        request_id = None
    retry_after = headers.get("Retry-After")
    if not (
        isinstance(retry_after, str)
        and len(retry_after) <= 64
        and re.fullmatch(r"[0-9A-Za-z,: .+-]+", retry_after)
        and not (key and key in retry_after)
    ):
        retry_after = None
    phrases = [
        p
        for p in (
            "invalid schema",
            "unsupported",
            "not supported",
            "required",
            "additionalproperties",
            "upstream",
            "timeout",
            "overloaded",
        )
        if p in lower
    ]
    # Keep a closed, non-echoed hint for the observed provider capacity failure.
    # It is diagnostic only: no new retry category or medical/dispatch authority.
    provider_message = error.get("message")
    if isinstance(provider_message, str) and "上游负载已饱和" in provider_message:
        phrases.append("upstream_capacity_saturated")

    from .wire_compatibility import safe_schema_error

    location = safe_schema_error(error, schema or {}, key)
    param = error.get("param")
    param = param if isinstance(param, str) and param in (body or {}) and param in fields else None
    if (
        status in {400, 422, 429}
        and not quota
        and (kind == "invalid_request_error" or code == "invalid_json_schema")
        and ("invalid_schema" in location["schema_rules"] or code == "invalid_json_schema")
        and (param == "response_format" or "response_format" in mentioned)
    ):
        category = "request_format_rejected"
    result = {
        **location,
        "error_param": param,
        "http_status": status,
        "category": category,
        "error_code": code,
        "error_type": kind,
        "request_id": request_id,
        "retry_after": retry_after,
        "message_tokens": phrases,
        "mentioned_fields": mentioned,
        "error_body_checksum": checksum(raw),
        "message": "raw_remote_error_omitted_may_contain_input_or_credentials",
    }

    if key and contains_secret(result, key):
        return {
            "http_status": status,
            "category": "secret_error_omitted",
            "error_body_checksum": checksum(raw),
        }
    return result


def contains_secret(value, key, depth=0):
    if isinstance(value, str):
        if key in value:
            return True
        if depth < 4:
            try:
                nested = json.loads(value)
            except (ValueError, TypeError):
                return False
            if nested != value:
                return contains_secret(nested, key, depth + 1)
    if isinstance(value, dict):
        return any(
            contains_secret(k, key, depth) or contains_secret(v, key, depth)
            for k, v in value.items()
        )
    if isinstance(value, list):
        return any(contains_secret(v, key, depth) for v in value)
    return False


def main(argv=()):
    started = time.monotonic()
    observe = False
    stage = "before_dispatch"

    def timing(phase):
        nonlocal stage
        if phase != "worker_end":
            stage = phase
        if observe:
            print(
                "TIMING " + json.dumps(dict(phase=phase, seconds=time.monotonic() - started)),
                file=sys.stderr,
                flush=True,
            )

    try:
        if argv:
            from .curl_exec import parent_death_kill

            if len(argv) != 2 or argv[0] != "--parent-pid":
                raise ValueError("worker_parent_arguments")
            parent_death_kill(int(argv[1]))
        raw = sys.stdin.buffer.read(16_777_217)
        if len(raw) > 16_777_216:
            raise ValueError()
        envelope = json.loads(raw)
        observe = envelope.get("timing_events") is True
        config = LLMConfig.model_validate(envelope["config"])
        http_timeout = envelope.get("effective_http_timeout", config.timeout_seconds)
        if (
            not isinstance(http_timeout, (int, float))
            or not 0 < http_timeout <= config.timeout_seconds
        ):
            raise ValueError("http_timeout_bound")
        # No alternate auth discovery, fallback provider or config-derived credential values.
        key = os.environ.get(config.api_key_env)
        if not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
            print(json.dumps({"error": "authentication"}))
            return 0
        if envelope.get("http_client") is not None:
            from .curl_client import exchange, prepare, read_client
            from .curl_client import response as curl_response

            try:
                command, data = prepare(
                    config,
                    read_client(envelope["http_client"]),
                    envelope.get("network_route"),
                    envelope["body"],
                    operation=envelope.get("operation"),
                    timeout=http_timeout,
                    max_bytes=envelope["max_bytes"],
                    key=key,
                )
            except (ValueError, OSError):
                print(json.dumps({"error": "configuration"}))
                return 0
            print("DISPATCH", flush=True)
            timing("dispatch_boundary")
            raw_reply, exit_code, outcome = exchange(
                command, data, timeout=http_timeout, max_bytes=envelope["max_bytes"], timing=timing
            )
            value = curl_response(
                raw_reply,
                exit_code,
                outcome,
                max_bytes=envelope["max_bytes"],
                key=key,
                body=envelope["body"],
                timing=timing,
            )
            if envelope.get("operation") == "models" and "body" in value:
                ids = [m.get("id") for m in value["body"].get("data", []) if isinstance(m, dict)]
                value["body"] = dict(
                    model_listed=config.model in ids, model=config.model, catalog_count=len(ids)
                )
            print(json.dumps(value))
            return 0
        suffix = "/responses" if config.endpoint == "responses" else "/chat/completions"
        catalog = envelope.get("operation") == "models"
        if catalog:
            suffix = "/models"
        req = urllib.request.Request(
            config.base_url.rstrip("/") + suffix,
            data=None if catalog else json.dumps(envelope["body"]).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        try:
            opener = route_opener(envelope.get("network_route"), config.base_url)
        except ValueError:
            print(json.dumps({"error": "configuration"}))
            return 0
        print("DISPATCH", flush=True)
        timing("dispatch_boundary")
        with opener.open(req, timeout=http_timeout) as response:
            timing("response_headers")
            body = response.read(envelope["max_bytes"] + 1)
            timing("response_body")
            request_id = response.headers.get("x-request-id")
        if len(body) > envelope["max_bytes"]:
            print(json.dumps({"error": "truncated"}))
            return 0
        decoded = json.loads(body)
        if (
            key.encode() in body
            or contains_secret(decoded, key)
            or (request_id and key in request_id)
        ):
            print(json.dumps({"error": "secret_response"}))
            return 0
        if catalog:
            ids = [m.get("id") for m in decoded.get("data", []) if isinstance(m, dict)]
            decoded = {
                "model_listed": config.model in ids,
                "model": config.model,
                "catalog_count": len(ids),
            }
        print(json.dumps({"body": decoded, "request_id": request_id, "http_status": 200}))
    except urllib.error.HTTPError as exc:
        error = {401: "authentication", 403: "authentication", 429: "rate_limit"}.get(
            exc.code, "configuration" if 400 <= exc.code < 500 else "transport"
        )
        print(
            json.dumps(
                {
                    "error": error,
                    "diagnostic": safe_http_error(
                        exc.code,
                        exc.read(8192),
                        exc.headers,
                        key,
                        envelope["body"]
                        .get("response_format", {})
                        .get("json_schema", {})
                        .get("schema"),
                        envelope["body"],
                    ),
                }
            )
        )
    except TimeoutError as exc:
        print(
            json.dumps(
                dict(
                    error="timeout",
                    **(
                        {"diagnostic": {"transport_failure": safe_transport_error(exc, stage)}}
                        if observe
                        else {}
                    ),
                )
            )
        )
    except Exception as exc:
        # Preserve legacy classification/retry behavior; the opt-in diagnostic is additive.
        print(
            json.dumps(
                dict(
                    error="transport",
                    **(
                        {"diagnostic": {"transport_failure": safe_transport_error(exc, stage)}}
                        if observe
                        else {}
                    ),
                )
            )
        )
    finally:
        timing("worker_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Explicit single-dispatch curl transport; bounded pipes, no credential files or shell."""

import json
import math
import os
import selectors
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from packages.grounding.models import Checksum, Value, validate
from packages.object_store import checksum

from .network_route import proxy_endpoint, read_route, route_opener

HEADER_LIMIT = 32768
MARKER = b"\nMMAGENT_CURL_META:"
METRIC_FIELDS = (
    "http_code",
    "http_connect",
    "ssl_verify_result",
    "time_connect",
    "time_appconnect",
    "time_pretransfer",
    "time_starttransfer",
    "time_total",
    "size_upload",
    "size_download",
)
WRITE_OUT = (
    MARKER.decode()
    + "{"
    + ",".join(
        '"' + name + '":"%{' + name + '}"'
        if name in {"http_code", "http_connect"}
        else '"' + name + '":%{' + name + "}"
        for name in METRIC_FIELDS
    )
    + "}"
)
BINARY = Path("/usr/bin/curl")


class CurlClient(Value):
    schema_version: Literal["bounded-http-client/1.0.0"] = "bounded-http-client/1.0.0"
    implementation: Literal["system-curl-http1.1"] = "system-curl-http1.1"
    binary_checksum: Checksum
    retries: Literal[0] = 0
    redirects: Literal[0] = 0
    parent_death_signal: Literal["SIGKILL"] = "SIGKILL"


def read_client(value):
    return None if value is None else validate(CurlClient, value)


def verify_binary(client):
    client = read_client(client)
    info = BINARY.stat()
    if (
        client is None
        or sys.platform != "linux"
        or not stat.S_ISREG(info.st_mode)
        or not os.access(BINARY, os.X_OK)
        or info.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError("curl_binary_policy")
    # Capabilities or set-ID binaries may clear Linux's parent-death signal on exec.
    if "security.capability" in os.listxattr(BINARY):
        raise ValueError("curl_binary_capabilities")
    if checksum(BINARY.read_bytes()) != client.binary_checksum:
        raise ValueError("curl_binary_changed")


def prepare(config, client, route, body, *, operation, timeout, max_bytes, key):
    verify_binary(client)
    route = read_route(route)
    if route is None:
        raise ValueError("curl_explicit_route_required")
    route_opener(route, config.base_url)  # Original exact route and bypass guard; no connection.
    url = urlsplit(config.base_url)
    if url.username or url.password or url.fragment or url.query:
        raise ValueError("curl_endpoint_binding")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 2_000_000:
        raise ValueError("curl_response_bound")
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 600
    ):
        raise ValueError("curl_time_bound")
    suffix = (
        "/models"
        if operation == "models"
        else ("/responses" if config.endpoint == "responses" else "/chat/completions")
    )
    settings = [
        ("url", config.base_url.rstrip("/") + suffix),
        ("proxy", proxy_endpoint(os.environ[route.proxy_env])),
        ("request", "GET" if operation == "models" else "POST"),
        ("user-agent", "Python-urllib/3.12"),
    ]
    if operation != "models":
        raw = json.dumps(body)
        if len(raw.encode()) > 16_777_216:
            raise ValueError("curl_request_bound")
        settings.append(("data-binary", raw))
    settings += [
        ("header", v)
        for v in (
            "Content-Type: application/json",
            "Authorization: Bearer " + key,
            "Accept:",
            "Accept-Encoding: identity",
            "Connection: close",
        )
    ]
    # JSON quoting uses the same supported backslash/quote/control escapes as curl config.
    data = "".join(k + " = " + json.dumps(v, ensure_ascii=False) + "\n" for k, v in settings)
    if (
        len(data.encode()) > 16_777_216
        or sum(
            len((k + " = " + json.dumps(v, ensure_ascii=False) + "\n").encode())
            for k, v in settings
            if k != "data-binary"
        )
        > 65000
    ):
        raise ValueError("curl_stdin_bound")
    command = [
        sys.executable,
        "-m",
        "packages.llm_provider.curl_exec",
        str(os.getpid()),
        client.binary_checksum,
        str(float(timeout)),
        str(max_bytes),
    ]
    return command, data.encode()


def header_end(raw):
    """Return the final HTTP header offset and selected fields; skip bounded 1xx blocks."""
    start = 0
    while start <= HEADER_LIMIT:
        end = raw.find(b"\r\n\r\n", start)
        if end < 0:
            return None
        if end + 4 > HEADER_LIMIT:
            raise ValueError("curl_header_bound")
        lines = bytes(raw[start:end]).split(b"\r\n")
        fields = lines[0].split(b" ", 2)
        if len(fields) < 2 or not fields[0].startswith(b"HTTP/") or not fields[1].isdigit():
            raise ValueError("curl_header_format")
        status = int(fields[1])
        if not 100 <= status <= 599:
            raise ValueError("curl_http_status")
        if 100 <= status < 200:
            start = end + 4
            continue
        headers = {}
        for line in lines[1:]:
            name, sep, value = line.partition(b":")
            if sep and name.lower() in {b"x-request-id", b"retry-after"}:
                headers[name.decode().lower()] = value.strip().decode("latin-1")
        return end + 4, status, headers
    raise ValueError("curl_header_bound")


def exchange(command, data, *, timeout, max_bytes, timing):
    """Drain stdout and write stdin concurrently, with an independent streaming size cap."""
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin"},
    )
    output = bytearray()
    deadline = time.monotonic() + timeout
    sent, header_seen, outcome = 0, None, None
    selector = selectors.DefaultSelector()
    try:
        for stream, event in (
            (proc.stdin, selectors.EVENT_WRITE),
            (proc.stdout, selectors.EVENT_READ),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcome = "timeout"
                break
            for item, event in selector.select(min(remaining, 0.1)):
                stream = item.fileobj
                if event == selectors.EVENT_WRITE:
                    try:
                        sent += os.write(stream.fileno(), data[sent : sent + 16384])
                    except BrokenPipeError:
                        sent = len(data)
                    if sent == len(data):
                        selector.unregister(stream)
                        stream.close()
                else:
                    chunk = os.read(stream.fileno(), 16384)
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    output.extend(chunk)
                    if len(output) > max_bytes + HEADER_LIMIT + 1024:
                        outcome = "truncated"
                        break
                    if header_seen is None:
                        header_seen = header_end(output)
                        if header_seen is not None:
                            timing("response_headers")
                        elif len(output) > HEADER_LIMIT:
                            outcome = "truncated"
                            break
                    if header_seen and len(output) - header_seen[0] > max_bytes + 1024:
                        outcome = "truncated"
                        break
            if outcome:
                break
        if outcome is None:
            try:
                proc.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                outcome = "timeout"
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            if not stream.closed:
                stream.close()
    return bytes(output), proc.returncode, outcome


def read_metrics(tail):
    """Closed numeric vocabulary only; never persist curl error text, URLs or raw metadata."""
    if len(tail) > 2048:
        return None
    try:
        value = json.loads(tail)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    result = dict(
        schema_version="safe-curl-phases/1.0.0", upload_is_not_server_execution_confirmation=True
    )
    for name in METRIC_FIELDS:
        number = value.get(name)
        if name in {"http_code", "http_connect"}:
            number = (
                int(number)
                if isinstance(number, str)
                and len(number) == 3
                and number.isascii()
                and number.isdigit()
                else number
            )
            valid = type(number) is int and (number == 0 or 100 <= number <= 599)
        else:
            limit = (
                600
                if name.startswith("time_")
                else 1000
                if name == "ssl_verify_result"
                else 16_777_216
            )
            valid = type(number) in {int, float} and math.isfinite(number) and 0 <= number <= limit
        result[name] = number if valid else None
    return result


def response(raw, exit_code, outcome, *, max_bytes, key, body, timing):
    from .worker import contains_secret, safe_http_error

    if key.encode() in raw:
        return {"error": "secret_response"}
    content, sep, tail = raw.rpartition(MARKER)
    metrics = read_metrics(tail) if sep else None
    phase_diagnostic = dict(curl_phases=metrics) if metrics else {}
    if outcome or exit_code:
        error = outcome or (
            "timeout" if exit_code == 28 else "truncated" if exit_code == 63 else "transport"
        )
        return dict(
            error=error,
            diagnostic=dict(
                **phase_diagnostic,
                transport_failure=dict(
                    schema_version="curl-transport-failure/1.0.0",
                    category={
                        28: "socket_timeout",
                        52: "remote_disconnected",
                        56: "receive_error",
                        35: "tls_error",
                        60: "tls_verification_failed",
                        7: "connection_error",
                        6: "dns_error",
                        63: "response_size_limit",
                    }.get(exit_code, outcome or "unconfirmed_transport"),
                    remote_execution="unknown",
                    automatic_replay_authorized=False,
                ),
            ),
        )
    final = header_end(content)
    if not sep or final is None:
        return {"error": "transport"}
    offset, status, headers = final
    if (metrics.get("http_code") if metrics else tail.strip()) != (
        status if metrics else str(status).encode()
    ):
        return {"error": "transport"}
    payload = content[offset:]
    if len(payload) > max_bytes:
        return {"error": "truncated"}
    timing("response_body")
    if not 200 <= status < 300:
        return dict(
            error={401: "authentication", 403: "authentication", 429: "rate_limit"}.get(
                status, "configuration" if 300 <= status < 500 else "transport"
            ),
            diagnostic={
                **phase_diagnostic,
                **safe_http_error(
                    status,
                    payload[:8192],
                    {
                        "x-request-id": headers.get("x-request-id"),
                        "Retry-After": headers.get("retry-after"),
                    },
                    key=key,
                    body=body,
                ),
            },
        )
    decoded = json.loads(payload)
    if contains_secret(decoded, key):
        return {"error": "secret_response"}
    return dict(
        body=decoded,
        request_id=headers.get("x-request-id"),
        http_status=status,
        **(dict(diagnostic=phase_diagnostic) if metrics else {}),
    )

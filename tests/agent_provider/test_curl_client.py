"""Explicit alternative transport: exact wire, bounded IO, safe credentials, no hidden retry."""

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_details.review_batches import (
    ReviewBatchResponseV132,
    ReviewBatchResponseV152,
    read_response,
)
from packages.grounding.models import canonical, digest
from packages.llm_provider import worker
from packages.llm_provider.config import read_provider_config, with_http_client, with_network_route
from packages.llm_provider.curl_client import (
    BINARY,
    MARKER,
    CurlClient,
    exchange,
    header_end,
    prepare,
    response,
    verify_binary,
)
from packages.llm_provider.network_route import prepare_route
from packages.llm_provider.provider import ProviderError, StructuredProvider
from packages.llm_provider.transport import HTTPTransport
from packages.object_store import checksum
from tests.agent_provider.test_network_route import origin


def client():
    return CurlClient(binary_checksum=checksum(BINARY.read_bytes()))


@pytest.mark.parametrize(
    "version,cls", [(7, ReviewBatchResponseV132), (9, ReviewBatchResponseV152)]
)
def test_new_version_exact_client_delta_and_response_reader(version, cls):
    old = with_network_route(
        origin(version), prepare_route(environ={"HTTPS_PROXY": "http://127.0.0.1:8"})
    )
    frozen = canonical(old)
    new = with_http_client(old, client())
    raw = new.model_dump(mode="json")
    raw.pop("http_client")
    raw["schema_version"] = old.schema_version
    assert raw == old.model_dump(mode="json") and canonical(old) == frozen
    assert read_provider_config(new) == new
    assert StructuredProvider(new, live_opt_in=True).transport.http_client == client()
    with pytest.raises(ProviderError, match="configuration"):
        StructuredProvider(
            new, live_opt_in=True, transport=HTTPTransport(network_route=old.network_route)
        )
    saved = cls(
        request_id="request:public",
        record_checksum=digest("record"),
        raw_model_text="{}",
        raw_checksum=checksum(b"{}"),
        error="schema",
        finish_reason="stop",
        provider_config=new,
        alignment=None,
    )
    assert read_response(saved.model_dump(mode="json")) == saved
    changed = saved.model_dump(mode="json")
    changed["schema_version"] = (
        "review-batch-response/1.3.1" if version == 7 else "review-batch-response/1.5.1"
    )
    with pytest.raises(ValueError):
        read_response(changed)


def test_binary_mismatch_refuses_and_old_transport_has_no_client():
    verify_binary(client())
    with pytest.raises(ValueError, match="curl_binary_changed"):
        verify_binary(client().model_copy(update=dict(binary_checksum=digest("foreign"))))
    assert HTTPTransport().http_client is None
    with pytest.raises(ValueError, match="explicit_route"):
        HTTPTransport(http_client=client())


def test_exact_body_only_in_stdin_not_args_or_files(monkeypatch, tmp_path):
    proxy = "http://127.0.0.1:8"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    body = dict(messages=[dict(content='public "quote"\nnew line; $(not-shell)')])
    command, stdin = prepare(
        origin(7).transport,
        client(),
        prepare_route(),
        body,
        operation=None,
        timeout=2,
        max_bytes=1000,
        key="synthetic-key",
    )
    assert "synthetic-key" not in " ".join(command)
    assert "public" not in " ".join(command)
    line = next(x for x in stdin.decode().splitlines() if x.startswith("data-binary = "))
    assert json.loads(line.split(" = ", 1)[1]) == json.dumps(body)
    assert list(tmp_path.iterdir()) == []
    assert b"Authorization: Bearer synthetic-key" in stdin


@pytest.mark.parametrize(
    "exit_code,category",
    [(52, "remote_disconnected"), (56, "receive_error"), (28, "socket_timeout")],
)
def test_unconfirmed_failure_never_grants_retry(exit_code, category):
    value = response(
        b"", exit_code, None, max_bytes=1000, key="secret", body={}, timing=lambda _: None
    )
    assert "http_status" not in value.get("diagnostic", {})
    failure = value["diagnostic"]["transport_failure"]
    assert failure["category"] == category and failure["remote_execution"] == "unknown"
    assert failure["automatic_replay_authorized"] is False


def test_response_headers_errors_redirects_secrets_and_bounds():
    def parse(status, content, extra=b""):
        raw = (
            b"HTTP/1.1 "
            + str(status).encode()
            + b" result\r\n"
            + extra
            + b"\r\n"
            + content
            + MARKER
            + str(status).encode()
        )
        return response(
            raw, 0, None, max_bytes=1000, key="synthetic-key", body={}, timing=lambda _: None
        )

    ok = parse(200, b'{"ok":true}', b"x-request-id: req-public\r\n")
    assert ok["body"] == {"ok": True} and ok["request_id"] == "req-public"
    assert header_end(bytearray(b"HTTP/1.1 200 OK\r\nx-request-id: r\r\n\r\n"))[2] == {
        "x-request-id": "r"
    }
    rate = parse(429, b'{"error":{"code":"rate_limit_exceeded"}}', b"Retry-After: 2\r\n")
    assert rate["error"] == "rate_limit" and rate["diagnostic"]["retry_after"] == "2"
    assert parse(302, b"{}")["error"] == "configuration"
    assert parse(200, b'{"key":"synthetic-key"}') == {"error": "secret_response"}
    assert parse(200, b"x" * 1001) == {"error": "truncated"}
    assert header_end(b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\n\r\n")[1] == 200


def test_pipe_stream_bound_and_timeout_reap_children(tmp_path):
    for kind in ("huge", "wait"):
        pid_path = tmp_path / kind
        script = (
            "import os,time; from pathlib import Path; "
            f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        )
        script += (
            'os.write(1,b"HTTP/1.1 200 OK\\r\\n\\r\\n"+b"x"*200000); time.sleep(30)'
            if kind == "huge"
            else "time.sleep(30)"
        )
        raw, code, why = exchange(
            [sys.executable, "-c", script], b"", timeout=1, max_bytes=1000, timing=lambda _: None
        )
        assert why == ("truncated" if kind == "huge" else "timeout")
        assert len(raw) <= 1000 + 32768 + 1024 + 16384
        pid = int(pid_path.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_worker_invalid_binary_zero_dispatch(monkeypatch, capsys):
    config = origin(7).transport
    monkeypatch.setenv(config.api_key_env, "synthetic-key")
    envelope = dict(
        config=config.model_dump(mode="json"),
        body={},
        max_bytes=100,
        http_client=client()
        .model_copy(update=dict(binary_checksum=digest("wrong")))
        .model_dump(mode="json"),
    )
    monkeypatch.setattr(
        worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(canonical(envelope)))
    )
    assert worker.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"error": "configuration"}
    assert "DISPATCH" not in captured.out and "synthetic-key" not in captured.out


@pytest.mark.parametrize("outer_caller", [False, True])
def test_worker_outer_death_stops_real_curl_child(outer_caller):
    """Only loopback proxy handshake; no TLS completion, external request or actual credential."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(8)
    proxy = f"http://127.0.0.1:{server.getsockname()[1]}"
    config = origin(7).transport
    envelope = dict(
        config=config.model_dump(mode="json"),
        body={},
        max_bytes=1000,
        http_client=client().model_dump(mode="json"),
        network_route=prepare_route(environ={"HTTPS_PROXY": proxy}).model_dump(mode="json"),
    )
    env = {"PATH": "/usr/bin:/bin", "HTTPS_PROXY": proxy, config.api_key_env: "synthetic-key"}
    command = [sys.executable, "-m", "packages.llm_provider.worker"]
    if outer_caller:
        command = [
            sys.executable,
            "-c",
            "import json,sys; from packages.llm_provider.transport import HTTPTransport; "
            "from packages.knowledge_claims.ports import LLMConfig; "
            "e=json.load(sys.stdin); "
            "HTTPTransport(network_route=e['network_route'],http_client=e['http_client'])"
            ".send(LLMConfig.model_validate(e['config']),e['body'],timeout=30,max_bytes=1000)",
        ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    conn = None
    try:
        proc.stdin.write(canonical(envelope))
        proc.stdin.close()
        conn, _ = server.accept()
        children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
        assert len(children) == 1
        worker_pid = int(children[0]) if outer_caller else proc.pid
        if outer_caller:
            children = Path(f"/proc/{worker_pid}/task/{worker_pid}/children").read_text().split()
            assert len(children) == 1
        pid = int(children[0])
        assert b"curl" in Path(f"/proc/{pid}/cmdline").read_bytes()
        proc.kill()
        proc.wait(timeout=3)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            path = Path(f"/proc/{pid}/stat")
            if not path.exists() or path.read_text().split(") ", 1)[1].startswith("Z "):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("curl_still_running_after_worker_death")
        if outer_caller:
            path = Path(f"/proc/{worker_pid}/stat")
            assert not path.exists() or path.read_text().split(") ", 1)[1].startswith("Z ")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not stream.closed:
                stream.close()
        if conn:
            conn.close()
        server.close()

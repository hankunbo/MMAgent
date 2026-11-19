"""Safe connection/upload telemetry and exact local curl serialization; no provider calls."""

import fcntl
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from packages.llm_provider.curl_client import BINARY, MARKER, WRITE_OUT, exchange, prepare, response
from packages.llm_provider.network_route import prepare_route
from tests.agent_provider.test_curl_client import client
from tests.agent_provider.test_network_route import origin


def metadata(**updates):
    value = dict(
        http_code="000",
        http_connect="200",
        ssl_verify_result=0,
        time_connect=0.01,
        time_appconnect=0.2,
        time_pretransfer=0.21,
        time_starttransfer=0.0,
        time_total=4.0,
        size_upload=45939,
        size_download=0,
    )
    value.update(updates)
    return json.dumps(value).encode()


def test_receive_error_retains_numeric_phases_but_never_grants_replay():
    raw = MARKER + metadata(url_effective="must-not-leak", errormsg="remote text")
    result = response(
        raw, 56, None, max_bytes=65536, key="synthetic-key", body={}, timing=lambda _: None
    )
    diagnostic = result["diagnostic"]
    assert result["error"] == "transport"
    assert diagnostic["transport_failure"]["remote_execution"] == "unknown"
    assert diagnostic["transport_failure"]["automatic_replay_authorized"] is False
    phases = diagnostic["curl_phases"]
    assert phases["size_upload"] == 45939 and phases["http_connect"] == 200
    assert phases["time_appconnect"] == 0.2 and phases["http_code"] == 0
    assert phases["upload_is_not_server_execution_confirmation"] is True
    assert "must-not-leak" not in json.dumps(result) and "remote text" not in json.dumps(result)


def test_json_footer_success_and_nonfinite_fields_suppressed():
    result = response(
        b'HTTP/1.1 200 OK\r\nx-request-id: public\r\n\r\n{"ok":true}'
        + MARKER
        + metadata(http_code="200", time_total=float("nan"), size_upload=-1, time_connect="secret"),
        0,
        None,
        max_bytes=1000,
        key="synthetic-key",
        body={},
        timing=lambda _: None,
    )
    assert result["body"] == {"ok": True}
    phases = result["diagnostic"]["curl_phases"]
    assert phases["time_total"] is None and phases["size_upload"] is None
    assert phases["time_connect"] is None


def test_fragmented_headers_through_real_pipe():
    script = (
        'import os,time; os.write(1,b"HTTP/1.1 200 OK\\r\\nx-request-id: p"); '
        'time.sleep(0.03); os.write(1,b"ublic\\r\\n\\r\\n{\\"ok\\":true}"); '
        f"os.write(1,{(MARKER + metadata(http_code='200'))!r})"
    )
    seen = []
    raw, code, outcome = exchange(
        [sys.executable, "-c", script], b"", timeout=2, max_bytes=1000, timing=seen.append
    )
    result = response(
        raw, code, outcome, max_bytes=1000, key="synthetic-key", body={}, timing=seen.append
    )
    assert result["body"] == {"ok": True} and result["request_id"] == "public"
    assert seen == ["response_headers", "response_body"]


def compare_local_wire(body, tmp_path, monkeypatch):
    """Test-only HTTP loopback receiver; production HTTPS/proxy gates are not relaxed."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    captured = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            captured.append(
                dict(
                    path=self.path,
                    headers=dict(self.headers),
                    body=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
                )
            )
            self.send_response(200)
            self.send_header("Content-Length", "11")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        _, data = prepare(
            origin(7).transport,
            client(),
            prepare_route(),
            body,
            operation=None,
            timeout=3,
            max_bytes=1000,
            key="synthetic-key",
        )
        settings = [
            (line.split(" = ", 1)[0], json.loads(line.split(" = ", 1)[1]))
            for line in data.decode().splitlines()
        ]
        settings = [
            (
                k,
                f"http://127.0.0.1:{server.server_port}/chat/completions"
                if k == "url"
                else ""
                if k == "proxy"
                else v,
            )
            for k, v in settings
        ]
        path = tmp_path / "public-wire.json"
        expected = json.dumps(body).encode()
        path.write_bytes(expected)
        results = []
        for from_file in (True, False):
            config = "".join(
                k
                + " = "
                + json.dumps(
                    "@" + str(path) if k == "data-binary" and from_file else v, ensure_ascii=False
                )
                + "\n"
                for k, v in settings
            )
            command = [
                str(BINARY),
                "--disable",
                "--config",
                "-",
                "--http1.1",
                "--silent",
                "--proto",
                "=http",
                "--retry",
                "0",
                "--max-time",
                "3",
                "--include",
                "--write-out",
                WRITE_OUT,
            ]
            if not from_file:
                command = [
                    sys.executable,
                    "-c",
                    "import os; from packages.llm_provider.curl_exec import install_input; "
                    "install_input(); " + f"os.execv({str(BINARY)!r}, {command!r})",
                ]
            raw, code, outcome = exchange(
                command,
                config.encode(),
                timeout=4,
                max_bytes=1000,
                timing=lambda _: None,
            )
            parsed = response(
                raw,
                code,
                outcome,
                max_bytes=1000,
                key="synthetic-key",
                body=body,
                timing=lambda _: None,
            )
            assert parsed.get("body") == {"ok": True}, (parsed, code)
            results.append(parsed["diagnostic"]["curl_phases"])
        assert len(captured) == 2 and captured[0] == captured[1], [
            dict(headers=c["headers"], bytes=len(c["body"])) for c in captured
        ]
        assert captured[0]["body"] == expected
        assert all(r["size_upload"] == len(expected) for r in results)
        return dict(
            equal_request_headers_and_body=True,
            wire_bytes=len(expected),
            local_http_requests=2,
            external_requests=0,
            production_tls_not_exercised=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_real_curl_file_and_inline_body_match(monkeypatch, tmp_path):
    body = dict(
        model="synthetic", messages=[dict(role="user", content='public "引用"\nline\\end ' * 3000)]
    )
    result = compare_local_wire(body, tmp_path, monkeypatch)
    assert result["wire_bytes"] > 62_000


def test_body_fd_is_anonymous_sealed_and_failure_closes_it(monkeypatch):
    from packages.llm_provider import curl_exec

    fd = curl_exec.sealed_fd("synthetic-only", b"public payload")
    try:
        assert os.fstat(fd).st_nlink == 0
        assert not os.get_inheritable(fd)
        assert fcntl.fcntl(fd, curl_exec.GET_SEALS) == curl_exec.SEAL_ALL
        with pytest.raises(PermissionError):
            os.write(fd, b"x")
        with pytest.raises(PermissionError):
            os.ftruncate(fd, 0)
        assert os.read(fd, 100) == b"public payload"
    finally:
        os.close(fd)
    before = set(Path("/proc/self/fd").iterdir())
    monkeypatch.setattr(
        curl_exec.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b'data-binary = "first"\ndata-binary = "second"\n')),
    )
    with pytest.raises(ValueError, match="curl_body_duplicate"):
        curl_exec.install_input()
    assert set(Path("/proc/self/fd").iterdir()) == before

"""Explicit network routing cannot change medical semantics or replay old requests."""

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_details.review_batches import (
    ReviewBatchResponseV131,
    ReviewBatchResponseV151,
    read_response,
)
from packages.grounding.models import canonical, digest
from packages.llm_provider import worker
from packages.llm_provider.config import (
    read_provider_config,
    review_format_upgrade,
    with_network_route,
)
from packages.llm_provider.network_route import prepare_route, proxy_endpoint, route_opener
from packages.llm_provider.provider import ProviderError, StructuredProvider
from packages.llm_provider.transport import HTTPTransport
from packages.object_store import checksum

ENV = {"HTTPS_PROXY": "http://127.0.0.1:8888"}


def origin(version):
    if version == 7:
        return read_provider_config(
            json.loads(Path("configs/agents/fairvlmed_relay_review_shape_v1.json").read_bytes())
        )
    from packages.multimodal_research.campaign import configuration

    return review_format_upgrade(configuration(directional=True, resilient_vision=True))


@pytest.mark.parametrize(
    "version,response_type", [(7, ReviewBatchResponseV131), (9, ReviewBatchResponseV151)]
)
def test_config_roundtrip_changes_only_explicit_route(version, response_type):
    old = origin(version)
    before = canonical(old)
    new = with_network_route(old, prepare_route(environ=ENV))
    assert canonical(old) == before
    raw = new.model_dump(mode="json")
    raw.pop("network_route")
    raw["schema_version"] = old.schema_version
    assert raw == old.model_dump(mode="json")
    assert read_provider_config(new) == new and digest(old) != digest(new)
    record = response_type(
        request_id="request:synthetic",
        record_checksum=digest("record"),
        raw_model_text="{}",
        raw_checksum=checksum(b"{}"),
        error="schema",
        finish_reason="stop",
        provider_config=new,
        alignment=None,
    )
    assert read_response(record.model_dump(mode="json")) == record
    provider = StructuredProvider(new, live_opt_in=True)
    assert provider.transport.network_route == new.network_route
    with pytest.raises(ProviderError, match="configuration"):
        StructuredProvider(new, live_opt_in=True, transport=HTTPTransport())


@pytest.mark.parametrize("route_enabled", [False, True])
def test_http_worker_envelope_binds_route_without_endpoint_or_key(monkeypatch, route_enabled):
    config = origin(7)
    policy = prepare_route(environ=ENV) if route_enabled else None
    captured = []

    def run(*args, **kwargs):
        captured.append(json.loads(kwargs["input"]))
        return SimpleNamespace(stdout=b'{"error":"configuration"}', stderr=b"", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    reply = HTTPTransport(network_route=policy).send(
        config.transport, {}, timeout=1, max_bytes=1024
    )
    assert reply.api_requests == 0 and len(captured) == 1
    assert ("network_route" in captured[0]) == route_enabled
    assert ENV["HTTPS_PROXY"] not in json.dumps(captured)
    if route_enabled:
        assert captured[0]["network_route"] == policy.model_dump(mode="json")


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "",
        "socks5://localhost:123",
        "http://user:secret@localhost:123",
        "http://localhost/path",
        "http://localhost?secret=x",
        "http://localhost#x",
        "http://localhost:99999",
        "http://local host:80",
    ],
)
def test_bad_route_rejects_without_disclosure(endpoint):
    with pytest.raises(ValueError, match="^network_proxy_endpoint_invalid$"):
        proxy_endpoint(endpoint)


def test_route_binding_and_no_proxy_fail_closed(monkeypatch):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    policy = prepare_route(environ=ENV)
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        route_opener(policy, "https://example.org", environ={"HTTPS_PROXY": "http://localhost:8"})
    with pytest.raises(ValueError, match="network_proxy_bypass_not_authorized"):
        route_opener(policy, "https://example.org", environ={**ENV, "NO_PROXY": "*"})
    with pytest.raises(ValueError, match="network_proxy_requires_https_target"):
        route_opener(policy, "http://example.org", environ=ENV)
    handlers = []
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *args: handlers.extend(args))
    route_opener(None, "https://example.org", environ=ENV)
    assert handlers[0].proxies == {}
    handlers.clear()
    route_opener(policy, "https://example.org", environ=ENV)
    assert handlers[0].proxies == {"https": ENV["HTTPS_PROXY"]}


def test_changed_route_worker_zero_dispatch(monkeypatch, capsys):
    config = origin(7)
    monkeypatch.setenv(config.transport.api_key_env, "synthetic-test-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8889")
    envelope = dict(
        config=config.transport.model_dump(mode="json"),
        body={},
        max_bytes=1024,
        network_route=prepare_route(environ=ENV).model_dump(mode="json"),
    )
    monkeypatch.setattr(
        worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(canonical(envelope)))
    )
    assert worker.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "configuration"}
    assert "DISPATCH" not in output.out and "synthetic-test-key" not in output.out


def test_proxy_still_keeps_tls_redirect_and_single_dispatch(monkeypatch, capsys):
    config = origin(7)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(config.transport.api_key_env, "synthetic-test-key")
    monkeypatch.setenv("HTTPS_PROXY", ENV["HTTPS_PROXY"])
    seen = []

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, maximum):
            return b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'

    class Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, timeout))
            assert request.get_header("Authorization") == "Bearer synthetic-test-key"
            return Response()

    def build(*handlers):
        assert handlers[0].proxies == {"https": ENV["HTTPS_PROXY"]}
        assert type(handlers[1]).__name__ == "NoRedirect"
        # No custom HTTPSHandler or insecure SSLContext is installed.
        assert len(handlers) == 2
        return Opener()

    monkeypatch.setattr(worker.urllib.request, "build_opener", build)
    envelope = dict(
        config=config.transport.model_dump(mode="json"),
        body={},
        max_bytes=1024,
        network_route=prepare_route(environ=ENV).model_dump(mode="json"),
    )
    monkeypatch.setattr(
        worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(canonical(envelope)))
    )
    assert worker.main() == 0
    output = capsys.readouterr()
    assert output.out.count("DISPATCH\n") == 1 and len(seen) == 1
    assert "synthetic-test-key" not in output.out

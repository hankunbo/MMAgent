"""The exact original optional route is preserved before claiming new work."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_local_review import prepared

from packages.grounding.models import digest
from packages.llm_provider.network_route import NetworkRoute
from packages.multimodal_research import campaign_renewal, execution_lease, implementation
from packages.multimodal_research import local_review_recovery as r


@pytest.mark.parametrize("route_kind", ["legacy_direct", "frozen_proxy", "wrong_proxy"])
def test_optional_route_precheck_before_claim(tmp_path, monkeypatch, route_kind):
    old, proof = prepared(tmp_path)
    values = {}
    docs = SimpleNamespace(
        read=values.get, write=lambda k, v: values.__setitem__(k, v), session=nullcontext
    )
    config = SimpleNamespace(transport=SimpleNamespace(base_url="https://example.invalid/v1"))
    if route_kind != "legacy_direct":
        config.network_route = NetworkRoute(
            mode="environment_https_proxy",
            proxy_env="HTTPS_PROXY",
            endpoint_checksum=digest("http://127.0.0.1:7897"),
        )
    monkeypatch.setenv(
        "HTTPS_PROXY",
        "http://127.0.0.1:7890" if route_kind == "wrong_proxy" else "http://127.0.0.1:7897",
    )
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    origin = SimpleNamespace(protocol=old, output=tmp_path / "failed", config=config)
    seen = []
    monkeypatch.setattr(campaign_renewal, "assemble", lambda path: seen.append(path) or origin)
    monkeypatch.setattr(execution_lease, "session", lambda _: nullcontext())
    monkeypatch.setattr(implementation, "freeze", lambda: {})
    monkeypatch.setattr(r, "failure", lambda _: (docs, proof))
    if route_kind == "wrong_proxy":
        with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
            r.prepare_and_run(
                previous=origin.output, output=tmp_path / "new", authorization="test", run=False
            )
        assert not values and not (tmp_path / "new").exists()
    else:
        result = r.prepare_and_run(
            previous=origin.output, output=tmp_path / "new", authorization="test", run=False
        )
        assert result["status"] == "prepared" and result["api_requests"] == 0
        assert result["cumulative_before"] == 27
    assert len(seen) == 1

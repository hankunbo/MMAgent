"""A frozen route must be usable before a sources-only successor is claimed."""

import copy

import pytest

from packages.multimodal_research import source_checkpoint as recovery
from packages.multimodal_research import unknown_acquisition
from tests.evaluation_preparation.test_source_checkpoint import saved


@pytest.mark.parametrize("check_only", [True, False])
def test_route_rejection_precedes_claim_and_output(tmp_path, monkeypatch, check_only):
    c, e, _, _, _ = saved(tmp_path, monkeypatch)
    before = copy.deepcopy(e.documents.saved)
    calls = []

    def reject(config):
        assert config is c.config
        calls.append("original_network_owner")
        raise ValueError("network_proxy_endpoint_changed")

    monkeypatch.setattr(unknown_acquisition, "check_network", reject)
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        recovery.prepare(
            previous=c.output, output=output, authorization="test", check_only=check_only
        )
    assert calls == ["original_network_owner"]
    assert not output.exists() and e.documents.saved == before


def test_current_route_uses_original_config_without_model_dispatch(tmp_path, monkeypatch):
    c, e, _, _, _ = saved(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(unknown_acquisition, "check_network", lambda config: calls.append(config))
    result = recovery.prepare(
        previous=c.output, output=tmp_path / "new", authorization="test", check_only=True
    )
    assert calls == [c.config]
    assert result["api_requests"] == result["database_writes"] == 0
    assert result["previous_requests"] == 32 and result["max_new_requests"] == 8
    assert e.documents.read("authorized-campaign-renewal") is None

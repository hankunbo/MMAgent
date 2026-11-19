"""7.1 route-preserving batch resize through original acceptance, no sockets."""

import pytest

from packages.grounding.models import digest
from packages.llm_provider.config import with_network_route
from packages.llm_provider.network_route import prepare_route
from tests.agent_provider import batch_support
from tests.expert_agent import test_legacy_accepted_aggregation as legacy


@pytest.fixture
def routed_configuration(monkeypatch):
    original = batch_support.load_config
    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setattr(
        batch_support, "load_config", lambda path: with_network_route(original(path), route)
    )


def test_routed_resize_keeps_original_scores_requests_and_readback(
    monkeypatch, routed_configuration
):
    legacy.test_legacy_judgments_keep_exact_values_and_original_requests(monkeypatch, True)


@pytest.mark.parametrize("change", ["model", "token_limit", "batch_count", "missing_proof"])
def test_routed_resize_rejects_unrelated_changes(monkeypatch, routed_configuration, change):
    legacy.test_resize_is_not_general_config_compatibility(monkeypatch, change)


def test_batch_resize_cannot_change_existing_network_route(monkeypatch, routed_configuration):
    f = legacy.combined(monkeypatch, resize=True)
    reader = f.source.readers[0]
    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:9999"})
    reader.current_config = reader.current_config.model_copy(update={"network_route": route})
    state = f.owner.current().model_copy(update={"config_checksum": digest(reader.current_config)})
    with pytest.raises(ValueError, match="accepted_review_batch_config_delta"):
        reader.read(state)
    assert not f.docs.saved

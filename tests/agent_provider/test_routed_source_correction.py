"""Network-only configuration must keep the original exact-quote correction.

All HTTP responses are injected. No Key, network, GPU or business DB access.
This diagnostic is outside the production freeze while Fair E is running.
"""

import pytest

from packages.llm_provider.config import with_network_route
from packages.llm_provider.network_route import prepare_route
from tests.agent_provider import test_source_selection as original
from tests.expert_agent import test_batch_assembly as assembly_module


@pytest.fixture(params=[False, True], ids=["legacy70", "routed71"])
def routed_assembly(request, monkeypatch):
    load = assembly_module.load_config
    if request.param:
        policy = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
        monkeypatch.setattr(
            assembly_module,
            "load_config",
            lambda path: with_network_route(load(path), policy),
        )


def test_exact_quote_correction_and_full_original_acceptance(
    routed_assembly, monkeypatch, tmp_path
):
    original.test_source_quote_feedback_then_original_three_stage_acceptance(monkeypatch, tmp_path)


def test_second_bad_quote_stops_without_third_attempt(routed_assembly, monkeypatch, tmp_path):
    original.test_source_quote_second_failure_stops_without_third_attempt(monkeypatch, tmp_path)

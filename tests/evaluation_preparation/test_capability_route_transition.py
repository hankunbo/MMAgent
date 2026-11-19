"""Original request/acceptance/score/reader chain, no real HTTP or patient input."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import (
    review_format_upgrade,
    with_http_client,
    with_network_route,
    with_replacement_network_route,
)
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research import capability_window as window
from tests.agent_provider import test_accepted_review_reuse as original
from tests.agent_provider import test_network_accepted_reuse as network
from tests.agent_provider.support import budget_for
from tests.evaluation_preparation.test_capability_window import plan_fixture
from tests.evaluation_preparation.test_directional_review import reply


def fixture(monkeypatch, *, curl=True):
    make = original.fixture

    def routed():
        f = make()
        f.config = with_network_route(
            review_format_upgrade(f.config),
            prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}),
        )
        if curl:
            f.config = with_http_client(
                f.config,
                json.loads(Path("configs/agents/system_curl_client_v1.json").read_bytes()),
            )
        f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(f.config)})
        return f

    monkeypatch.setattr(original, "fixture", routed)
    f = original.accepted_fixture(new_observation=True)
    new = with_replacement_network_route(
        f.old.config, prepare_route(environ={"HTTPS_PROXY": "http://localhost:9999"})
    )
    f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(new)})
    f.owner.confirmed_review_reader.network_configs = (f.old.config, new)
    f.budget = budget_for(
        f.old.binding.model_copy(update={"run_id": f.owner.current().material.scope.run_id}), new
    )
    f.agent = StructuredRoleAgent(
        "review",
        provider=DeterministicFakeProvider(new, [reply] * 12),
        materials=f.owner,
        budget=f.budget,
    )
    f.acceptance = ResearchAcceptance(owner=f.owner, documents=f.documents, requests=f.budget)
    return f, new


def test_original_provider_acceptance_score_readback_and_cold_ancestry(monkeypatch):
    monkeypatch.setattr(network, "fixture", fixture)
    network.test_original_provider_batch_acceptance_report_and_cold_nested_read(monkeypatch)


def fixture_v91(monkeypatch):
    return fixture(monkeypatch, curl=False)


def test_v91_original_acceptance_and_cold_ancestry_keep_http_client(monkeypatch):
    monkeypatch.setattr(network, "fixture", fixture_v91)
    network.test_original_provider_batch_acceptance_report_and_cold_nested_read(monkeypatch)


@pytest.mark.parametrize("fault", ["model", "prompt", "request_cap", "source", "unknown"])
def test_v91_route_does_not_authorize_other_changes(monkeypatch, fault):
    monkeypatch.setattr(network, "fixture", fixture_v91)
    network.test_route_does_not_authorize_other_config_or_fact_changes(monkeypatch, fault)


def test_v91_configuration_changes_route_only():
    old = with_network_route(
        review_format_upgrade(original.fixture().config),
        prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}),
    )
    changed = with_replacement_network_route(
        old, prepare_route(environ={"HTTPS_PROXY": "http://localhost:9999"})
    )
    before, after = old.model_dump(mode="json"), changed.model_dump(mode="json")
    assert old.schema_version == changed.schema_version == "agent-provider-config/9.1.0"
    assert before.pop("network_route") != after.pop("network_route")
    assert before == after


@pytest.mark.parametrize(
    "fault",
    [
        "not_installed",
        "model",
        "endpoint",
        "request_cap",
        "timeout",
        "prompt",
        "observation",
        "source",
        "unknown",
        "old_config",
    ],
)
def test_explicit_route_proof_does_not_admit_other_changes(monkeypatch, fault):
    monkeypatch.setattr(network, "fixture", fixture)
    network.test_route_does_not_authorize_other_config_or_fact_changes(monkeypatch, fault)


def test_return_boundary_mutation_rejected(monkeypatch):
    monkeypatch.setattr(network, "fixture", fixture)
    network.test_return_boundary_detects_config_pair_mutation(monkeypatch)


def test_plan_version_budget_and_original_reader_binding(tmp_path, monkeypatch):
    owner, tip, inv, args = plan_fixture(tmp_path)
    f, new = fixture(monkeypatch)
    owner.config = f.old.config
    owner.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    before = deepcopy((tip, inv))
    path = "configs/agents/environment_https_route_v1.json"
    binding = window.route_binding(owner, path)
    old = window.plan_for(owner, tip, inv, **args, current=True)
    plan = window.plan_for(owner, tip, inv, **args, current=True, route_transition=binding)
    assert plan["schema_version"] == window.ROUTED_VERSION
    assert old["schema_version"] == window.CURRENT_VERSION and "route_transition" not in old
    assert plan["previous_attempts"] == old["previous_attempts"]
    assert (
        plan["deadline"] == old["deadline"] and plan["max_new_requests"] == old["max_new_requests"]
    )
    assert plan["provider"] == binding["provider_checksum"] != old["provider"]
    assert (tip, inv) == before
    bound = window.routed_reader(f.owner.confirmed_review_reader, owner.config, new)
    assert bound is not f.owner.confirmed_review_reader
    assert bound.documents is f.old.documents
    assert bound.current(f.owner.current()).assessment_ids
    with pytest.raises(ValueError, match="explicit_version"):
        window.check(owner, dict(plan, schema_version=window.CURRENT_VERSION))
    with pytest.raises(ValueError, match="explicit_version"):
        window.check(owner, {k: v for k, v in plan.items() if k != "route_transition"})
    changed = dict(binding, origin_checksum=digest("other"))
    with pytest.raises(ValueError, match="original_binding"):
        window.route_configuration(owner, changed)
    with pytest.raises(ValueError, match="fields"):
        window.route_configuration(owner, dict(binding, authorized=True))


def test_new_cli_route_is_explicit(monkeypatch, capsys):
    import sys

    from apps import multimodal_case

    seen = []
    monkeypatch.setattr(
        window, "prepare_current", lambda **kw: seen.append(kw) or {"api_requests": 0}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "capability-current-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--authorization",
            "test",
            "--network-route",
            "configs/agents/environment_https_route_v1.json",
            "--check-only",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen[0]["network_route"] == "configs/agents/environment_https_route_v1.json"
    assert seen[0]["check_only"] is True
    assert json.loads(capsys.readouterr().out)["api_requests"] == 0

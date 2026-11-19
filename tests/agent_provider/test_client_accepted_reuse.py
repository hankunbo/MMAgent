"""Client-only recovery must preserve accepted judgments and their original requests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import review_format_upgrade, with_http_client, with_network_route
from packages.llm_provider.network_route import prepare_route
from packages.planner_control.models import Control
from tests.agent_provider import test_accepted_review_reuse as original
from tests.agent_provider.support import budget_for
from tests.evaluation_preparation.test_directional_review import reply


def fixture(monkeypatch):
    make = original.fixture

    def routed():
        f = make()
        f.config = with_network_route(
            review_format_upgrade(f.config),
            prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}),
        )
        f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(f.config)})
        return f

    monkeypatch.setattr(original, "fixture", routed)
    f = original.accepted_fixture(new_observation=True)
    client = json.loads(Path("configs/agents/system_curl_client_v1.json").read_bytes())
    new = with_http_client(f.old.config, client)
    assert (f.old.config.schema_version, new.schema_version) == (
        "agent-provider-config/9.1.0",
        "agent-provider-config/9.2.0",
    )
    f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(new)})
    f.owner.confirmed_review_reader.client_configs = (f.old.config, new)
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


def test_client_delta_reuses_original_acceptance_and_cold_nested_read(monkeypatch):
    from agents.expert_agent.reporting import read_research_report, render_research_html

    f, new = fixture(monkeypatch)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="client", round_id="A"
    )
    prior = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    assert sub.historical.assessment_ids
    assert "item:additional" not in sub.historical.assessment_ids
    assert len(f.budget.records()) == len(sub.candidate.items) - len(sub.historical.assessment_ids)
    assert all(i == prior[i.assessment_id] for i in sub.candidate.items if i.assessment_id in prior)
    f.acceptance.submit(sub, key="client")
    receipt = f.acceptance.accept(key="client")
    assert receipt.status == "accepted"
    report = read_research_report(f.acceptance, key="client", parent=f.owner.current())
    assert "accepted-review-origin" in render_research_html(report)
    count = len(f.budget.records())
    assert f.acceptance.accept(key="client") == receipt
    f.documents.run_id = f.owner.current().material.scope.run_id
    binding = f.old.binding.model_copy(update={"run_id": f.documents.run_id})
    control = validate(Control, dict(binding=binding, graph=binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = f.owner.current().model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:CLIENT_COLD_THIRD"
    third = original.state_from(raw)
    old_reader = AcceptedReviewReader(
        documents=f.old.documents,
        requests=f.old.budget,
        state_reader=f.old.owner.current,
        acceptance_key="old-accepted",
        client_configs=(f.old.config, new),
    )
    reader = AcceptedReviewReader(
        documents=f.documents,
        requests=f.budget,
        state_reader=f.owner.current,
        acceptance_key="client",
        ancestor_reader=lambda *_: old_reader,
    )
    assert reader.current(third) == reader.current(third)
    assert len(f.budget.records()) == count
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))


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
        "combined",
    ],
)
def test_client_change_does_not_authorize_other_changes(monkeypatch, fault):
    f, new = fixture(monkeypatch)
    reader = f.owner.confirmed_review_reader
    if fault == "not_installed":
        reader.client_configs = None
    elif fault in {"model", "endpoint", "timeout"}:
        field = {"model": "model", "endpoint": "base_url", "timeout": "timeout_seconds"}[fault]
        value = {
            "model": "other",
            "endpoint": "https://other.example/v1",
            "timeout": new.transport.timeout_seconds + 1,
        }[fault]
        new = new.model_copy(update={"transport": new.transport.model_copy(update={field: value})})
    elif fault == "request_cap":
        new = new.model_copy(update={"max_requests": new.max_requests + 1})
    elif fault == "prompt":
        new = new.model_copy(
            update={
                "roles": tuple(
                    r.model_copy(update={"prompt_checksum": digest("changed")})
                    if r.role == "review"
                    else r
                    for r in new.roles
                )
            }
        )
    elif fault in {"observation", "source"}:
        raw = f.owner.current().model_dump(mode="json")
        if fault == "observation":
            raw["material"]["observations"][0]["value"] += " changed"
        else:
            raw["material"]["sources"][0]["text"] += " changed"
        f.owner.state = original.state_from(raw)
    elif fault == "unknown":
        f.old.budget.journal.append(dict(phase="reserved", request_id="UNKNOWN"))
    elif fault == "combined":
        reader.network_configs = (f.old.config, new)
    else:
        reader.client_configs = (new, new)
    if fault in {"model", "endpoint", "request_cap", "timeout", "prompt"}:
        reader.client_configs = (f.old.config, new)
        f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(new)})
    with pytest.raises(ValueError, match="accepted_review_|provider_configuration|network_route_"):
        reader.current(f.owner.current())
    assert not f.budget.records()


def test_return_boundary_revalidates_client_pair(monkeypatch):
    f, _ = fixture(monkeypatch)
    reader = f.owner.confirmed_review_reader
    original_read = AcceptedReviewReader._validated_requests

    def changed(local, *args):
        result = original_read(local, *args)
        reader.client_configs = None
        return result

    monkeypatch.setattr(AcceptedReviewReader, "_validated_requests", changed)
    with pytest.raises(ValueError, match="accepted_review_reuse_owner_changed"):
        reader.current(f.owner.current())
    assert not f.budget.records()

"""Transport-only recovery preserves original acceptance and medical context."""

from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import review_format_upgrade, with_network_route
from packages.llm_provider.network_route import prepare_route
from packages.planner_control.models import Control
from tests.agent_provider import test_accepted_review_reuse as original
from tests.agent_provider.support import budget_for
from tests.evaluation_preparation.test_directional_review import reply


def fixture(monkeypatch):
    make = original.fixture

    def nullable():
        value = make()
        value.config = review_format_upgrade(value.config)
        value.owner.state = value.owner.state.model_copy(
            update={"config_checksum": digest(value.config)}
        )
        return value

    monkeypatch.setattr(original, "fixture", nullable)
    f = original.accepted_fixture(new_observation=True)
    new = with_network_route(
        f.old.config, prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
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


def test_original_provider_batch_acceptance_report_and_cold_nested_read(monkeypatch):
    from agents.expert_agent.reporting import read_research_report, render_research_html

    f, new = fixture(monkeypatch)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    assert "item:additional" not in sub.historical.assessment_ids
    assert len(f.budget.records()) == len(sub.candidate.items) - len(sub.historical.assessment_ids)
    prior = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    assert all(
        i == prior[i.assessment_id]
        for i in sub.candidate.items
        if i.assessment_id in sub.historical.assessment_ids
    )
    f.acceptance.submit(sub, key="next")
    receipt = f.acceptance.accept(key="next")
    assert receipt.status == "accepted"
    report = read_research_report(f.acceptance, key="next", parent=f.owner.current())
    assert "accepted-review-origin/1.0.0" in render_research_html(report)
    count = len(f.budget.records())
    assert f.acceptance.accept(key="next") == receipt
    assert len(f.budget.records()) == count
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))
    f.documents.run_id = f.owner.current().material.scope.run_id
    binding = f.old.binding.model_copy(update={"run_id": f.documents.run_id})
    control = validate(Control, dict(binding=binding, graph=binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = f.owner.current().model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:ROUTED_COLD_THIRD"
    third = original.state_from(raw)
    old_reader = AcceptedReviewReader(
        documents=f.old.documents,
        requests=f.old.budget,
        state_reader=f.old.owner.current,
        acceptance_key="old-accepted",
        network_configs=(f.old.config, new),
    )
    reader = AcceptedReviewReader(
        documents=f.documents,
        requests=f.budget,
        state_reader=f.owner.current,
        acceptance_key="next",
        ancestor_reader=lambda *_: old_reader,
    )
    cold = reader.current(third)
    assert cold.acceptance_key == "next"
    assert reader.current(third) == cold
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))
    assert len(f.budget.records()) == count


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
def test_route_does_not_authorize_other_config_or_fact_changes(monkeypatch, fault):
    f, new = fixture(monkeypatch)
    reader = f.owner.confirmed_review_reader
    if fault == "not_installed":
        reader.network_configs = None
    elif fault in {"model", "endpoint", "timeout"}:
        field = {"model": "model", "endpoint": "base_url", "timeout": "timeout_seconds"}[fault]
        value = {
            "model": "different-model",
            "endpoint": "https://different.example/v1",
            "timeout": new.transport.timeout_seconds + 1,
        }[fault]
        new = new.model_copy(update={"transport": new.transport.model_copy(update={field: value})})
    elif fault == "request_cap":
        new = new.model_copy(update={"max_requests": new.max_requests + 1})
    elif fault == "prompt":
        roles = tuple(
            r.model_copy(update={"prompt_checksum": digest("different")})
            if r.role == "review"
            else r
            for r in new.roles
        )
        new = new.model_copy(update={"roles": roles})
    elif fault in {"observation", "source"}:
        raw = f.owner.current().model_dump(mode="json")
        if fault == "observation":
            raw["material"]["observations"][0]["value"] += " changed"
        else:
            raw["material"]["sources"][0]["text"] += " changed"
        f.owner.state = original.state_from(raw)
    elif fault == "unknown":
        f.old.budget.journal.append(dict(phase="reserved", request_id="unknown"))
    else:
        reader.network_configs = (new, new)
    if fault in {"model", "endpoint", "request_cap", "timeout", "prompt"}:
        reader.network_configs = (f.old.config, new)
        f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(new)})
    with pytest.raises(ValueError, match="accepted_review_|network_route_|provider_configuration"):
        reader.current(f.owner.current())
    assert not f.budget.records()


def test_return_boundary_detects_config_pair_mutation(monkeypatch):
    f, _ = fixture(monkeypatch)
    reader = f.owner.confirmed_review_reader
    old = AcceptedReviewReader._validated_requests

    def change(local, *args):
        result = old(local, *args)
        reader.network_configs = None
        return result

    monkeypatch.setattr(AcceptedReviewReader, "_validated_requests", change)
    with pytest.raises(ValueError, match="accepted_review_reuse_owner_changed"):
        reader.current(f.owner.current())
    assert not f.budget.records()

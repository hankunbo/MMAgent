"""Capacity-only transition, original Provider/admission/score and cold read."""

from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research.review_capacity_recovery import configuration
from packages.planner_control.models import Control
from tests.agent_provider import test_accepted_review_reuse as original
from tests.agent_provider.support import budget_for
from tests.evaluation_preparation.test_directional_review import reply


def fixture(monkeypatch):
    make = original.fixture

    def bounded():
        f = make()
        n = len(project_review_tasks(f.owner.current()).assessment_ids)
        f.config = f.config.model_copy(
            update={
                "review_batching": f.config.review_batching.model_copy(
                    update={"batch_size": 1, "max_batches": n}
                )
            }
        )
        f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(f.config)})
        return f

    monkeypatch.setattr(original, "fixture", bounded)
    f = original.accepted_fixture(new_observation=True)
    new = configuration(f.old.config, f.old.config.review_batching.max_batches + 1)
    f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(new)})
    f.owner.confirmed_review_reader.capacity_configs = (f.old.config, new)
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


def test_original_acceptance_score_historical_bytes_and_cold_reuse(monkeypatch):
    from agents.expert_agent.reporting import read_research_report, render_research_html

    f, new = fixture(monkeypatch)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="capacity", round_id="A"
    )
    assert len(sub.plan.batches) > f.old.config.review_batching.max_batches
    assert len(f.budget.records()) == len(sub.candidate.items) - len(sub.historical.assessment_ids)
    prior = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    assert all(i == prior[i.assessment_id] for i in sub.candidate.items if i.assessment_id in prior)
    f.acceptance.submit(sub, key="capacity")
    receipt = f.acceptance.accept(key="capacity")
    assert receipt.status == "accepted"
    report = read_research_report(f.acceptance, key="capacity", parent=f.owner.current())
    assert "accepted-review-origin" in render_research_html(report)
    count = len(f.budget.records())
    assert f.acceptance.accept(key="capacity") == receipt
    f.documents.run_id = f.owner.current().material.scope.run_id
    binding = f.old.binding.model_copy(update={"run_id": f.documents.run_id})
    control = validate(Control, dict(binding=binding, graph=binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = f.owner.current().model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:CAPACITY_COLD"
    third = original.state_from(raw)
    old_reader = AcceptedReviewReader(
        documents=f.old.documents,
        requests=f.old.budget,
        state_reader=f.old.owner.current,
        acceptance_key="old-accepted",
        capacity_configs=(f.old.config, new),
    )
    reader = AcceptedReviewReader(
        documents=f.documents,
        requests=f.budget,
        state_reader=f.owner.current,
        acceptance_key="capacity",
        ancestor_reader=lambda *_: old_reader,
    )
    assert reader.current(third) == reader.current(third)
    assert len(f.budget.records()) == count
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))


@pytest.mark.parametrize(
    "fault", ["not_installed", "model", "batch_size", "request_cap", "unknown", "combined"]
)
def test_capacity_never_authorizes_other_changes(monkeypatch, fault):
    f, new = fixture(monkeypatch)
    reader = f.owner.confirmed_review_reader
    if fault == "not_installed":
        reader.capacity_configs = None
    elif fault == "unknown":
        f.old.budget.journal.append(dict(phase="reserved", request_id="UNKNOWN"))
    elif fault == "combined":
        reader.network_configs = (f.old.config, new)
    else:
        if fault == "model":
            new = new.model_copy(
                update={"transport": new.transport.model_copy(update={"model": "other"})}
            )
        elif fault == "batch_size":
            new = new.model_copy(
                update={"review_batching": new.review_batching.model_copy(update={"batch_size": 2})}
            )
        else:
            new = new.model_copy(update={"max_requests": new.max_requests + 1})
        reader.capacity_configs = (f.old.config, new)
        f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest(new)})
    with pytest.raises(ValueError, match="accepted_review_"):
        reader.current(f.owner.current())
    assert not f.budget.records()


@pytest.mark.parametrize("count", [0, 15, 21, True])
def test_configuration_ceiling_remains_finite(count):
    from packages.multimodal_research.campaign_renewal import confirmed_facts_configuration

    with pytest.raises(ValueError, match="review_capacity_configuration_bound"):
        configuration(confirmed_facts_configuration(), count)

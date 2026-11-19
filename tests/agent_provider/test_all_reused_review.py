"""All-reused is owner-verified local composition, never an empty fresh Review."""

from copy import deepcopy

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance_models import read_records
from packages.evidence_details.review_batches import aggregate, make_plan, read_submission
from packages.evidence_details.review_continuation import aggregate_continued, check_continued
from packages.grounding.models import digest
from tests.agent_provider.test_accepted_review_reuse import (
    accepted_fixture,
    accepted_successor,
    state_from,
)
from tests.evaluation_preparation import test_directional_review as support


def one_item(monkeypatch):
    original = support.dynamic_fixture
    monkeypatch.setattr(support, "dynamic_fixture", lambda _: original(1))


def scores_without_identity(scores):
    return {
        k: v
        for k, v in scores.items()
        if k not in {"scope_checksum", "task_projection_checksum", "checksum"}
    }


def compose(f, key="all:A"):
    return run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key=key, round_id="A"
    )


def test_zero_request_original_owner_score_html_and_idempotence(monkeypatch):
    one_item(monkeypatch)
    f = accepted_fixture()
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    sub = compose(f)
    assert sub.schema_version == "research-batched-submission/16.0.0"
    assert sub.composition == "all_current_items_reused" and not sub.batches
    assert read_submission(sub.model_dump(mode="json")) == sub
    assert not f.budget.records()
    f.acceptance.submit(sub, key="next")
    receipt = f.acceptance.accept(key="next")
    assert receipt.status == "accepted"
    records = read_records(f.acceptance.read(key="next")[0])
    assert records.schema_version == "case-observation-interpretation/2.18.0"
    report = read_research_report(f.acceptance, key="next", parent=f.owner.current())
    old = read_research_report(f.old.acceptance, key="old-accepted", parent=f.old.owner.current())
    assert scores_without_identity(report["scores"]) == scores_without_identity(old["scores"])
    assert "all_current_items_reused" in render_research_html(report)
    audit = len(f.documents.audit)
    assert f.acceptance.accept(key="next") == receipt
    assert compose(f) == sub
    assert len(f.documents.audit) == audit and not f.budget.records()
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))


def test_new_observation_after_zero_request_composition_is_still_consumable(monkeypatch):
    one_item(monkeypatch)
    f = accepted_fixture()
    sub = compose(f)
    f.acceptance.submit(sub, key="next")
    assert f.acceptance.accept(key="next").status == "accepted"
    g = accepted_successor(f)
    newer = compose(g, "following:A")
    assert newer.historical.schema_version == "accepted-review-origin/1.6.0"
    assert newer.composition == "accepted_origin_with_new_batches"
    assert len(g.budget.records()) == 1
    assert set(newer.historical.assessment_ids) == {"probe:task00"}
    g.acceptance.submit(newer, key="following")
    assert g.acceptance.accept(key="following").status == "accepted"


@pytest.mark.parametrize("fault", ["source", "observation", "origin", "claim", "unknown"])
def test_changed_binding_or_unknown_cannot_be_hidden_in_zero_request_closure(monkeypatch, fault):
    one_item(monkeypatch)
    f = accepted_fixture()
    sub = compose(f)
    if fault in {"source", "observation"}:
        raw = deepcopy(f.owner.current().model_dump(mode="json"))
        if fault == "source":
            raw["material"]["sources"][0]["text"] += " changed"
        else:
            raw["material"]["observations"][0]["value"] += " changed"
        f.owner.state = state_from(raw)
        error = "accepted_review_reuse_visible_context_changed"
    elif fault == "origin":
        f.old.documents.saved["accepted:old-accepted"][3]["records_checksum"] = digest("wrong")
        error = "accepted_review_reuse_receipt_binding"
    elif fault == "claim":
        f.documents.write("review-batches:all:A:batch:000:claim", ({"claimed": True},))
        error = "all_reused_unexpected_dispatch"
    else:
        f.budget.journal.append(dict(phase="reserved", request_id="unknown"))
        error = "all_reused_unexpected_dispatch"
    with pytest.raises(ValueError, match="^" + error + "$"):
        check_continued(
            f.owner.current(), sub, owner=f.owner, documents=f.documents, requests=f.budget
        )
    assert f.acceptance.submit(sub, key="bad").status == "rejected"
    assert f.acceptance.read(key="bad") is None


def test_empty_fresh_or_incomplete_reuse_still_rejected(monkeypatch):
    one_item(monkeypatch)
    f = accepted_fixture(new_observation=True)
    state = f.owner.current()
    plan = make_plan(state, f.agent.provider.config.review_batching, key="missing:A")
    with pytest.raises(ValueError, match="^batch_incomplete_or_reused_request$"):
        aggregate(state, plan, [])
    with pytest.raises(ValueError, match="^all_reused_current_coverage_required$"):
        aggregate_continued(state, plan, [], f.owner.confirmed_review_reader.current(state))


def failed_live_fixture(monkeypatch, pg=None, tmp_path=None):
    """Actual Provider/owner/grant contracts; injected HTTP, no external socket."""
    from types import SimpleNamespace

    from agents.provider_agents import StructuredRoleAgent
    from packages.evidence_details import review_continuation
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
    from packages.evidence_details.live_permission import ResearchLiveGrant, ResearchLivePermission
    from packages.evidence_details.recorded_finalization import RecordedBatchRequests
    from packages.llm_provider.provider import RequestBudget, StructuredProvider
    from packages.research_runtime.agent_records import RuntimeAgentJournal
    from tests.agent_provider.finalization_support import fixture
    from tests.agent_provider.support import Reader
    from tests.agent_provider.test_bounded_live import SyntheticJournal, SyntheticWire
    from tests.expert_agent.test_accepted_continued_postgres import stop, successor
    from tests.expert_agent.test_batch_assembly import LiveDocuments
    from tests.planner.support import MemoryRuntime

    one_item(monkeypatch)
    old = fixture(pg, corrected=True, reasoned=True, dynamic_policy=True, expired_clock=False)
    if pg is None:
        # The legacy in-memory Documents only recognizes offline v1 receipts.
        # This explicit live-branch double mirrors consumption of v2 receipts;
        # the PG test below uses the real Runtime implementation unchanged.
        old.docs.consumed = lambda graph: LiveDocuments.consumed(old.docs, graph)
    original = ResearchAcceptance(
        owner=old.owner, documents=old.docs, requests=old.budget, live_permission=old.old
    )
    original.submit(old.submission, key="original")
    assert original.accept(key="original").status == "accepted"
    binding = old.planner.read(old.docs.run_id).binding
    if pg is not None:
        docs, binding = successor(
            SimpleNamespace(documents=old.docs, binding=binding), "ALL_REUSED"
        )
        journal = RuntimeAgentJournal(docs)
        stop(docs, binding)
    else:
        binding = binding.model_copy(update={"run_id": "RUN:ALL_REUSED"})
        docs = LiveDocuments()
        docs.run_id, docs.root, docs.sessions = binding.run_id, old.docs.root, MemoryRuntime()
        stop(docs, binding)
        journal = SyntheticJournal()
        journal.documents = docs
    raw = old.state.model_dump(mode="json")
    raw["material"]["scope"].update(run_id=docs.run_id, graph=binding.graph.model_dump(mode="json"))
    owner = Reader(state_from(raw))
    owner.confirmed_review_reader = AcceptedReviewReader(
        documents=old.docs,
        requests=old.budget,
        state_reader=old.owner.current,
        acceptance_key="original",
    )
    budget = RequestBudget(
        run_id=docs.run_id, deadline=binding.deadline, max_requests=40, journal=journal
    )
    grant = old.old.grant.model_dump(mode="python")
    grant.update(
        run_id=docs.run_id,
        policy_checksum=owner.current().material.scope.policy_checksum,
        deadline=binding.deadline,
    )
    permission = ResearchLivePermission(
        grant=ResearchLiveGrant(**grant), config=old.config, documents=docs, budget=budget
    )
    wire = SyntheticWire([])
    agent = StructuredRoleAgent(
        "review",
        provider=StructuredProvider(old.config, live_opt_in=True, transport=wire),
        materials=owner,
        budget=budget,
    )
    docs.write("live-stage-intent:review", ({"event": {"code": "all"}},))
    docs.write("dynamic-phase-input", (owner.current(),))
    current = SimpleNamespace(owner=owner, agent=agent, documents=docs, budget=budget)

    def legacy(*args, **kwargs):
        raise ValueError("batch_incomplete_or_reused_request")

    with monkeypatch.context() as patch:
        patch.setattr(review_continuation, "aggregate_continued", legacy)
        with pytest.raises(ValueError, match="^batch_incomplete_or_reused_request$"):
            compose(current)
    phase = dict(
        run_id=docs.run_id, target=None, accepted=False, error="batch_incomplete_or_reused_request"
    )
    docs.write("dynamic-phase-result", (phase,))
    docs.write("multimodal-case-result", (dict(complete=False, phases=[phase]),))
    return SimpleNamespace(
        old=old,
        original=original,
        owner=owner,
        documents=docs,
        budget=budget,
        requests=RecordedBatchRequests(journal),
        wire=wire,
        config=old.config,
        permission=permission,
    )


def test_stopped_all_reused_local_grant_and_readback(monkeypatch):
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.recorded_finalization import (
        RecordedReviewPermission,
        read_saved_acceptance,
    )

    f = failed_live_fixture(monkeypatch)
    before = digest(f.documents.read("multimodal-case-result")), digest(f.budget.journal.read())
    permission, sub = RecordedReviewPermission.prepare(
        owner=f.owner,
        documents=f.documents,
        requests=f.requests,
        config=f.config,
    )
    assert permission.grant.schema_version == "recorded-review-finalization/1.3.0"
    service = ResearchAcceptance(
        owner=f.owner, documents=f.documents, requests=f.requests, live_permission=permission
    )
    key = permission.acceptance_key
    assert service.submit(sub, key=key)["status"] == "submitted_not_accepted"
    assert service.accept(key=key).status == "accepted"
    assert (
        read_saved_acceptance(
            owner=f.owner, documents=f.documents, requests=f.requests, config=f.config
        )["acceptance_key"]
        == key
    )
    assert before == (
        digest(f.documents.read("multimodal-case-result")),
        digest(f.budget.journal.read()),
    )
    assert not f.wire.calls and not f.requests.records()

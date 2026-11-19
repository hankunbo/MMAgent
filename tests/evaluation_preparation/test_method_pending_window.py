"""Original accepted partial owner, bounded method budget, no real network."""

import json
from types import SimpleNamespace

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.batch_local_recovery import recover
from packages.evidence_details.partial_finalization import RESULT_KEY, closure_keys
from packages.grounding.models import digest, validate
from packages.multimodal_research import confirmed_review_window as window
from tests.agent_provider.finalization_support import fixture
from tests.agent_provider.test_partial_finalization import permission


def accepted_method_origin(tmp_path, monkeypatch, pg=None, *, cancelled=False):
    from packages.multimodal_research import campaign

    f = fixture(
        pg,
        partial=True,
        reasoned=True,
        deadline_partial=True,
        raw_only_deadline=True,
        deadline_cleanup_cancel=cancelled,
    )
    recover(owner=f.owner, documents=f.docs, requests=f.requests)
    grant, submission = permission(f)
    acceptance = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=grant
    )
    key = closure_keys(f.docs)[2]
    acceptance.submit(submission, key=key)
    assert acceptance.accept(key=key).status == "accepted"
    local = dict(
        schema_version="multimodal-partial-finalization-result/1.1.0",
        complete=True,
        new_requests=0,
        original_result_checksum=digest(f.failed),
        phases=[dict(f.failed["phases"][-1], accepted=True, acceptance_key=key, error=None)],
        review_closure_scope=dict(local_confirmed_only=True, historical_answers_reaccepted=False),
    )
    f.docs.write(RESULT_KEY, (local,))
    (tmp_path / "result.json").write_text(json.dumps(f.failed))
    control = f.planner.read(f.docs.run_id)
    if pg is None:
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        # The production composition reconstructs a journal from documents;
        # preserve the injected transport's exact events in that same port.
        journal = RuntimeAgentJournal(f.docs)
        for event in f.budget.journal.read():
            journal.append(event)
        f.budget.journal = f.requests.journal = journal
        f.docs.contexts = SimpleNamespace(read_research_run=lambda _: SimpleNamespace())

        # The legacy memory double understands only offline receipt v1.0.
        # Match the original RuntimeDocuments live-receipt consumption here;
        # the PG integration below uses the unmodified persistent owner.
        def consumed(graph):
            saved = f.docs.read("accepted:" + key)
            return (
                bool(saved)
                and saved[3]["status"] == "accepted"
                and all(
                    saved[3]["parent_graph"][name] == getattr(graph, name)
                    for name in ("code", "version", "checksum")
                )
            )

        f.docs.consumed = consumed
    parent = SimpleNamespace(
        documents=lambda _: f.docs,
        sessions=f.docs.sessions,
        config=f.config,
        base=SimpleNamespace(
            contexts={},
            manifest=SimpleNamespace(checksum=digest("synthetic-input")),
            task=SimpleNamespace(checksum=digest("synthetic-task")),
        ),
        accepted_observations=lambda: {},
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.35.0",
            mode="review",
            deadline=control.binding.deadline.isoformat(),
            max_new_requests=7,
            max_cumulative_requests=47,
            previous_attempts=[dict(requests=40)],
            acquisition_targets=[],
            new_vision_requests=0,
            root_direction_review={"synthetic": True},
            text_review_method=dict(
                old_scores_reused=False,
                diagnostic_records=[dict(requests=1, checksum=digest("synthetic-diagnostic"))],
                diagnostic_requests=1,
                case_related_request_ceiling=48,
            ),
        ),
    )
    tip = dict(
        path=str(tmp_path),
        run_id=f.docs.run_id,
        previous_attempts=[dict(requests=40), dict(requests=len(f.wire.calls))],
    )

    class Projection:
        def __init__(self, *args, **kwargs):
            pass

        def project(self, **kwargs):
            return f.owner.current(), None

        def verify(self, state):
            assert f.owner.current() == state

    monkeypatch.setattr(campaign, "Phase", Projection)
    return f, parent, tip


def proof_plan(f, parent, tip, tmp_path):
    proof = window.origin_proof(parent, f.failed, tip, now=f.now)
    plan = window.plan_for(
        parent,
        f.failed,
        tip,
        proof,
        output=tmp_path / "new",
        authorization="synthetic test",
        now=f.now,
    )
    return proof, plan


def successor_state(state):
    raw = state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:METHOD_PENDING"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    return validate(type(state), raw)


@pytest.mark.parametrize("cancelled", [False, True])
def test_accepted_partial_requires_original_receipt_not_unconsumed_bypass(
    tmp_path, monkeypatch, cancelled
):
    from packages.evidence_details.review_continuation import ConfirmedReviewReader
    from packages.multimodal_research.campaign_renewal import historical_reader

    f, parent, tip = accepted_method_origin(tmp_path, monkeypatch, cancelled=cancelled)
    state = successor_state(f.state)
    ordinary = ConfirmedReviewReader(
        documents=f.docs, requests=f.requests, state_reader=f.owner.current
    )
    with pytest.raises(ValueError, match="already_accepted"):
        ordinary.current(state)
    reader = historical_reader(parent, tip["run_id"], preserve_prompt=True)
    original = reader.current(state)
    proof, plan = proof_plan(f, parent, tip, tmp_path)
    from packages.evidence_details.review_batches import read_submission

    saved = read_submission(f.docs.read("submission:" + closure_keys(f.docs)[2])[0]["submission"])
    assert original.submission.candidate == saved.candidate
    assert proof["confirmed_assessment_ids"]
    assert plan["schema_version"] == window.METHOD_VERSION
    assert window.method_cumulative_limit(plan) == 47
    assert plan["method_budget"]["case_related_request_ceiling"] == 48
    assert plan["method_budget"]["previous_owner_requests"] == 41
    assert plan["provider"] == digest(parent.config) and len(f.wire.calls) == 1
    before = digest(f.docs.saved), digest(f.budget.journal.read())
    assert reader.current(state) == original
    assert before == (digest(f.docs.saved), digest(f.budget.journal.read()))
    assert f.planner.read(f.docs.run_id).cancelled is cancelled
    if cancelled:
        parent.protocol["schema_version"] = window.VERSION
        with pytest.raises(ValueError, match="not_expired_partial"):
            proof_plan(f, parent, tip, tmp_path)


@pytest.mark.parametrize(
    "fault", ["unknown", "receipt", "local", "source", "budget", "repeat", "cancel_marker"]
)
def test_method_pending_rejects_changed_authority_without_new_dispatch(
    tmp_path, monkeypatch, fault
):
    f, parent, tip = accepted_method_origin(tmp_path, monkeypatch, cancelled=True)
    if fault == "unknown":
        f.budget.journal.append(dict(phase="reserved", request_id="unconfirmed"))
    elif fault == "receipt":
        key = "accepted:" + closure_keys(f.docs)[2]
        del f.docs.saved[key]
    elif fault == "local":
        f.docs.saved[RESULT_KEY][0]["original_result_checksum"] = digest("changed")
    elif fault == "source":
        f.owner.withdrawn = True
    elif fault == "budget":
        tip["previous_attempts"][-1]["requests"] = 2
    elif fault == "repeat":
        parent.protocol["schema_version"] = window.METHOD_VERSION
    else:
        (tmp_path / "cancel").touch()
    with pytest.raises(ValueError):
        proof_plan(f, parent, tip, tmp_path)
    assert len(f.wire.calls) == 1


def test_new_bound_is_not_a_fresh_full_budget_and_original_guard_enforces_it(tmp_path, monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    f, parent, tip = accepted_method_origin(tmp_path, monkeypatch)
    proof, plan = proof_plan(f, parent, tip, tmp_path)
    for bad in (0, 1, 7, 40):
        with pytest.raises(ValueError, match="scope_or_budget"):
            window.plan_for(
                parent,
                f.failed,
                tip,
                proof,
                output=tmp_path / "x",
                authorization="test",
                now=f.now,
                request_limit=bad,
            )
    for field, value in [
        ("max_cumulative_requests", 99),
        ("max_new_requests", 8),
        ("new_vision_requests", 1),
        ("reuse_confirmed_review", False),
    ]:
        with pytest.raises(ValueError, match="budget_binding"):
            window.method_cumulative_limit(dict(plan, **{field: value}))
    c = object.__new__(RenewedCampaign)
    c.protocol, c.output = plan, tmp_path
    c.base = SimpleNamespace(check=lambda: None)
    c.previous_attempts = lambda: plan["previous_attempts"]
    c.usage = lambda: 1
    c.guard_control()  # 42 is legal under original method bound, unlike legacy 40.
    c.usage = lambda: 6
    with pytest.raises(ProviderError, match="budget"):
        c.guard_control()

"""Explicit old-question correction shares the real new-observation Review path."""

from types import SimpleNamespace

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research.single_proposition import clarification_spec, clarify_review_state
from tests.agent_provider.test_accepted_review_reuse import accepted_fixture
from tests.evaluation_preparation.test_exact_acquisition_budget import (
    campaign,
    configuration,
    valid_plan,
)


def plan():
    return dict(
        valid_plan(),
        schema_version=renewal.CLARIFIED_ACQUISITION_VERSION,
        new_observation_execution_allowed=True,
        review_clarification=dict(assessment_ids=["corrected"]),
        reuse_assessment_ids=["unchanged"],
    )


def test_two_acquisitions_and_one_explicit_correction_use_one_review_batch():
    value = plan()
    renewal.validate_exact_acquisition_budget(value, configuration())
    assert renewal.acquisition_review_ids(value) == ("a", "b", "corrected", "unchanged")
    assert value["max_new_requests"] == 3
    assert sum(p["requests"] for p in value["previous_attempts"]) + 3 == 40


@pytest.mark.parametrize("ids", [[], ["corrected", "other"], ["a"], ["corrected", "corrected"]])
def test_no_hidden_extra_review_batch_or_acquisition_alias(ids):
    value = plan()
    value["review_clarification"]["assessment_ids"] = ids
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(value, configuration())


def test_old_protocol_cannot_acquire_new_clarification_semantics():
    value = dict(plan(), schema_version=renewal.EXACT_ACQUISITION_VERSION)
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(value, configuration())


def test_combined_plan_cannot_claim_to_be_text_only():
    value = dict(plan(), new_observation_execution_allowed=False)
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(value, configuration())


def test_combined_local_close_at_cap_keeps_current_source_gate(tmp_path):
    value, calls = campaign(tmp_path, version="1.34.0")
    value.guard()
    assert calls == ["current-source"]


@pytest.mark.parametrize("fault", [None, "answer", "reuse", "definition", "state"])
def test_clarification_must_bind_original_owner_task_and_exact_unchanged_subset(monkeypatch, fault):
    f = accepted_fixture(batch_size=3)
    state = f.owner.current()
    rows = [dict(id=r.id, definition=r.definition) for r in state.policy.requirements]
    ids = set(project_review_tasks(state).assessment_ids)
    corrected = sorted(ids)[0]
    task = SimpleNamespace(
        code="TASK:SYNTHETIC",
        target_code=state.material.goal_code,
        checksum=digest("synthetic-task"),
    )
    value = dict(
        plan(),
        accepted_state_checksum=digest(state),
        tip=dict(run_id="original"),
        reuse_assessment_ids=sorted(ids - {corrected}),
        review_clarification=clarification_spec(rows, task, [corrected], version="1.1.0"),
    )
    parent = SimpleNamespace(base=SimpleNamespace(task=task))
    monkeypatch.setattr(
        renewal, "accepted_reader", lambda *_a, **_k: SimpleNamespace(state_reader=lambda: state)
    )
    if fault == "answer":
        value["review_clarification"]["desired_direction"] = "refute"
    elif fault == "reuse":
        value["reuse_assessment_ids"].append(corrected)
    elif fault == "definition":
        value["review_clarification"]["original_definitions"][corrected] = digest("other")
    elif fault == "state":
        value["accepted_state_checksum"] = digest("other")
    if fault:
        with pytest.raises(ValueError, match="renewal_acquisition_clarification_binding"):
            renewal.validate_acquisition_clarification(parent, value)
    else:
        renewal.validate_acquisition_clarification(parent, value)


def test_combined_constructor_keeps_clarification_in_scope_but_not_old_reuse(tmp_path, monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign, review_observations

    state = dict(owner="original")
    reader = SimpleNamespace(state_reader=lambda: state)
    parent = SimpleNamespace(
        base=SimpleNamespace(review_target_ids=("unchanged",)), target=None, config=configuration()
    )
    value = dict(plan(), accepted_state_checksum=digest(state), tip=dict(run_id="original"))
    monkeypatch.setattr(renewal, "validate_acquisition_selection", lambda *_: None)
    monkeypatch.setattr(renewal, "accepted_reader", lambda *_a, **_k: reader)
    monkeypatch.setattr(
        MultimodalCampaign, "__init__", lambda self, base, *_a, **_k: setattr(self, "base", base)
    )
    following = renewal.RenewedCampaign(parent, tmp_path, value)
    assert following.initial_targets == ("a", "b")
    assert following.base.review_target_ids == ("a", "b", "corrected", "unchanged")
    assert following.base.review_clarification == value["review_clarification"]
    assert set(
        review_observations(following.base, dict(a=1, b=2, corrected=3, unchanged=4, other=5))
    ) == {"a", "b", "corrected", "unchanged"}
    assert parent.base.review_target_ids == ("unchanged",)


def test_original_batched_provider_acceptance_preserves_old_scores_and_reviews_changed_question():
    f = accepted_fixture(new_observation=True, batch_size=3)
    state = f.owner.current()
    rows = [dict(id=r.id, definition=r.definition) for r in state.policy.requirements]
    task = SimpleNamespace(
        code="TASK:SYNTHETIC",
        target_code=state.material.goal_code,
        checksum=digest("synthetic-task"),
    )
    origin = f.owner.confirmed_review_reader.current(state)
    corrected = origin.assessment_ids[0]
    old_bytes = digest(f.old.documents.saved)
    spec = clarification_spec(rows, task, [corrected], version="1.1.0")
    f.owner.state = clarify_review_state(state, task, spec)
    task_item = next(
        t for t in project_review_tasks(f.owner.current()).tasks if t.assessment_id == corrected
    )
    assert "支持非P不能标为support或正e" in task_item.definition.interpretation
    submission = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="combined", round_id="A"
    )
    reused = set(submission.historical.assessment_ids)
    required = set(project_review_tasks(f.owner.current()).assessment_ids) - reused
    assert corrected in required and "item:additional" in required
    assert corrected not in reused and submission.historical.assessment_ids
    assert len(f.budget.records()) == (len(required) + 2) // 3
    assert {i.assessment_id for b in submission.batches for i in b.candidate.items} == required
    f.acceptance.submit(submission, key="combined")
    assert f.acceptance.accept(key="combined").status == "accepted"
    count, audit = len(f.budget.records()), len(f.documents.audit)
    assert f.acceptance.accept(key="combined").status == "accepted"
    assert (count, audit) == (len(f.budget.records()), len(f.documents.audit))
    assert digest(f.old.documents.saved) == old_bytes
    assert spec["desired_direction"] is None and not spec["medical_relation_changed"]

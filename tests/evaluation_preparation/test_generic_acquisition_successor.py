"""Selected new Vision never becomes an implicit replay or a full-case score."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_acquisition_successor as owner
from packages.evaluation_preparation.generic_review import GenericReview


def campaign(tmp_path):
    c = owner.AcquisitionSuccessor.__new__(owner.AcquisitionSuccessor)
    c.output = tmp_path
    c.selected_target = "criterion:CHECK"
    c.visual_targets = (c.selected_target,)
    c.checked = SimpleNamespace(
        proof=dict(parent_key="parent", target=c.selected_target, request_id="request:old")
    )
    return c


@pytest.mark.parametrize("accepted", [True, False])
def test_failed_vision_stops_before_review_without_losing_failed_phase(
    tmp_path, monkeypatch, accepted
):
    c = campaign(tmp_path)
    c.phase_path(c.selected_target).mkdir()
    failed = dict(
        target=c.selected_target,
        accepted=accepted,
        evidence_limited=False,
        run_id="RUN:VISION",
        error=None if accepted else "transport",
    )
    path = c.phase_path(c.selected_target) / "result.json"
    path.write_text(json.dumps(failed))
    before = path.read_bytes()
    calls = []
    from tests.evaluation_preparation.test_partial_observation import proposal

    c.accepted_observations = lambda: {
        c.selected_target: SimpleNamespace(status="obtained", value=proposal().observation_value())
    }
    monkeypatch.setattr(GenericReview, "execute_phase", lambda _, t: calls.append(t) or "review")
    assert c.execute_phase() == ("review" if accepted else failed)
    assert calls == ([None] if accepted else [])
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", ["unknown", "ungradable"])
def test_accepted_but_uninterpretable_vision_is_retained_without_empty_review(
    tmp_path, monkeypatch, status
):
    from tests.evaluation_preparation.test_partial_observation import proposal

    c = campaign(tmp_path)
    c.phase_path(c.selected_target).mkdir()
    phase = dict(target=c.selected_target, accepted=True, run_id="RUN:VISION")
    (c.phase_path(c.selected_target) / "result.json").write_text(json.dumps(phase))
    c.accepted_observations = lambda: {
        c.selected_target: SimpleNamespace(status=status, value=proposal().observation_value())
    }
    calls = []
    monkeypatch.setattr(GenericReview, "execute_phase", lambda *a: calls.append(a))
    assert c.execute_phase() == phase
    assert not c.phase_completed(phase)
    assert calls == []


def test_no_review_report_preserves_original_phase_and_repeats_without_write(tmp_path):
    from contextlib import nullcontext

    c = campaign(tmp_path)
    c.checked.proof["inventory"] = dict(accepted_ids=["criterion:OLD"])
    c.accepted_observations = lambda: {}
    saved = {}
    writes = []

    def write(key, value):
        assert key not in saved
        saved[key] = value
        writes.append(key)

    docs = SimpleNamespace(read=lambda key: saved.get(key), write=write, session=nullcontext)
    c.documents = lambda _: docs
    phase = dict(target=c.selected_target, accepted=False, run_id="RUN:VISION", error="<error>")
    result = dict(phases=[phase], complete=False)
    report = c.export_result(result)
    assert report["scores"] is None
    assert report["acquisition_phase"] == phase
    assert report["previous_accepted_ids"] == ["criterion:OLD"]
    assert report["completion"]["review_dispatched"] is False
    assert "&lt;error&gt;" in saved["dynamic-case-report"][1]["html"]
    assert c.export_result(result) == report
    assert writes == ["dynamic-case-report"]


@pytest.mark.parametrize("target", ["criterion:FOREIGN", "", "../../old"])
def test_foreign_target_cannot_get_request_identity_or_storage(tmp_path, target):
    c = campaign(tmp_path)
    with pytest.raises(ValueError, match="generic_acquisition_unselected_phase"):
        c.phase_path(target)
    with pytest.raises(ValueError, match="generic_acquisition_unselected_phase"):
        c.request_key(target)
    assert list(tmp_path.iterdir()) == []


def test_changed_original_proof_prevents_new_acquisition(tmp_path, monkeypatch):
    c = campaign(tmp_path)
    calls = []
    monkeypatch.setattr(owner, "origin", lambda *_: SimpleNamespace(proof={"changed": True}))
    monkeypatch.setattr(c, "execute_phase", calls.append)
    with pytest.raises(ValueError, match="generic_acquisition_origin_changed_before_dispatch"):
        c.before_review()
    assert calls == []


@pytest.mark.parametrize("missing", ["parent", "execution"])
def test_each_original_lease_required_before_any_dispatch(tmp_path, monkeypatch, missing):
    c = campaign(tmp_path)
    calls = []
    c._parent_execution_guard = None if missing == "parent" else lambda: calls.append("parent")
    c._execution_guard = None if missing == "execution" else lambda: calls.append("execution")
    monkeypatch.setattr(GenericReview, "guard", lambda *_, **__: calls.append("dispatch"))
    with pytest.raises(ValueError, match="generic_acquisition_original_lease_required"):
        c.guard(dispatch=True)
    assert calls == []


def test_lease_loss_stops_before_original_budget_or_provider(tmp_path, monkeypatch):
    c = campaign(tmp_path)
    calls = []

    def lost():
        raise ValueError("owned_lease_lost")

    c._parent_execution_guard = lost
    c._execution_guard = lambda: calls.append("execution")
    monkeypatch.setattr(GenericReview, "guard", lambda *_, **__: calls.append("dispatch"))
    with pytest.raises(ValueError, match="owned_lease_lost"):
        c.guard(dispatch=True)
    assert calls == []


def test_history_is_not_replaced_by_new_observation_when_freezing(tmp_path):
    c = campaign(tmp_path)
    c.acquisition = SimpleNamespace(freeze_observations=lambda: {"historical": "exact"})
    before = c.freeze_observations()
    c.phase_path(c.selected_target).mkdir()
    (c.phase_path(c.selected_target) / "result.json").write_text('{"accepted": true}')
    assert c.freeze_observations() == before


def test_no_same_key_successor_even_if_old_attempt_failed():
    checked = SimpleNamespace(proof=dict(parent_key="parent"))
    with pytest.raises(ValueError, match="generic_acquisition_distinct_successor"):
        owner.selection_for(checked, "parent")


def test_overlapping_observations_are_not_silently_overwritten(tmp_path, monkeypatch):
    from packages.evaluation_preparation.comprehensive_campaign import ComprehensiveCampaign

    c = campaign(tmp_path)
    c.acquisition = SimpleNamespace(accepted_observations=lambda: {c.selected_target: "old"})
    monkeypatch.setattr(
        ComprehensiveCampaign, "accepted_observations", lambda _: {c.selected_target: "new"}
    )
    with pytest.raises(ValueError, match="generic_acquisition_duplicate_observation"):
        c.accepted_observations()


@pytest.mark.parametrize("fault", [None, "missing_result", "wrong_result", "pending", "cancelled"])
def test_recovery_requires_confirmed_original_stage_before_any_dispatch(tmp_path, fault):
    c = campaign(tmp_path)
    path = c.phase_path(c.selected_target)
    path.mkdir()
    request = dict(lock=dict(run_id="RUN:VISION"))
    phase = dict(run_id="RUN:VISION", target=c.selected_target, accepted=True)
    (path / "request.json").write_text(json.dumps(request))
    if fault != "missing_result":
        (path / "result.json").write_text(json.dumps(phase))
    c.contexts = SimpleNamespace(
        read_research_run=lambda _: SimpleNamespace(
            request=SimpleNamespace(model_dump=lambda **_: request)
        )
    )
    c.sessions = SimpleNamespace(
        read=lambda _: SimpleNamespace(
            phase="stopped", pending=fault == "pending", cancelled=fault == "cancelled"
        )
    )
    c.documents = lambda _: SimpleNamespace(
        read=lambda _: ({"different": True},) if fault == "wrong_result" else (phase,)
    )
    before = (path / "request.json").read_bytes()
    if fault:
        with pytest.raises(ValueError, match="generic_acquisition_recovery_"):
            owner.verified_stage_recovery(c)
    else:
        assert owner.verified_stage_recovery(c) == ["RUN:VISION"]
        assert owner.verified_stage_recovery(c) == ["RUN:VISION"]
    assert (path / "request.json").read_bytes() == before
    assert not c.phase_path(None).exists()

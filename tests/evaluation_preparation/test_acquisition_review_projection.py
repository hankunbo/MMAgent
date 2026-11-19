"""New acquisition Review scope must not retain a parent's text-only filter."""

from copy import copy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research.campaign import MultimodalCampaign, review_observations


@pytest.mark.parametrize(
    "version,expected",
    [("1.26.0", {"old"}), ("1.32.0", {"old", "new"}), ("1.37.0", {"old", "new"})],
)
def test_actual_renewal_constructor_replaces_only_new_version_scope(
    tmp_path, monkeypatch, version, expected
):
    # External storage/qualification was checked separately; this test targets
    # the production constructor's inherited field, not another plan function.
    old_state = dict(accepted="original-owner-state")
    reader = SimpleNamespace(state_reader=lambda: old_state)
    parent = SimpleNamespace(
        base=SimpleNamespace(review_target_ids=("old",)), target=None, config=SimpleNamespace()
    )
    plan = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="acquisition",
        review_partition="owner_bound_reused_and_new/1.0.0",
        acquisition_targets=["new"],
        reuse_assessment_ids=["old"],
        accepted_state_checksum=digest(old_state),
        tip=dict(run_id="original"),
    )
    monkeypatch.setattr(renewal, "validate_acquisition_selection", lambda *_: None)
    monkeypatch.setattr(renewal, "accepted_reader", lambda *_a, **_k: reader)
    monkeypatch.setattr(
        MultimodalCampaign, "__init__", lambda self, base, *_a, **_k: setattr(self, "base", base)
    )
    value = renewal.RenewedCampaign(parent, tmp_path, plan)
    assert set(review_observations(value.base, dict(old=1, new=2, unrelated=3))) == expected
    assert parent.base.review_target_ids == ("old",)
    assert value.base is not parent.base
    assert value.base.confirmed_review_reader is reader
    assert value.initial_targets == ("new",)
    assert getattr(value.base, "review_context_scope", None) == (
        "current-owner-prior-visible/1.0.0" if version == "1.37.0" else None
    )


def test_scope_preserves_unique_owner_items_not_manual_extra_votes():
    plan = dict(acquisition_targets=["c", "b"], reuse_assessment_ids=["a"])
    assert renewal.acquisition_review_ids(plan) == ("a", "b", "c")
    for change in (
        dict(acquisition_targets=[]),
        dict(acquisition_targets=["a"]),
        dict(acquisition_targets=["b", "b"]),
        dict(reuse_assessment_ids=["a", "a"]),
    ):
        with pytest.raises(ValueError, match="renewal_acquisition_review_scope"):
            renewal.acquisition_review_ids(dict(plan, **change))


def test_repair_cannot_be_used_as_generic_retry_or_new_acquisition(monkeypatch):
    from packages.multimodal_research.acquired_review import prepare_projection_repair

    def forbidden(*args, **kwargs):
        pytest.fail("invalid repair scope assembled actual owners")

    monkeypatch.setattr(renewal, "completed_parent", forbidden)
    args = dict(
        facts="same",
        previous="same",
        output="unused",
        mode="review",
        authorization="test",
        review_acquisition_gap=True,
    )
    for change in (
        dict(mode="acquisition"),
        dict(previous="other"),
        dict(reuse_confirmed_review=True),
        dict(exact_request_budget=True),
        dict(acquisition_target=["new"]),
        dict(authorization=""),
    ):
        with pytest.raises(ValueError, match="acquisition_review_projection_explicit_scope"):
            prepare_projection_repair(**dict(args, **change))


def test_projection_repair_requires_exact_original_failure():
    from packages.multimodal_research.acquired_review import projection_repair_proof

    parent = SimpleNamespace(protocol=dict(schema_version="old"))
    for result in (dict(complete=True, phases=[]), dict(complete=False, phases=[])):
        with pytest.raises(ValueError, match="acquisition_review_projection_origin"):
            projection_repair_proof(parent, copy(result), {})


def test_only_original_zero_request_cancellation_receipt_is_usable(tmp_path):
    from packages.multimodal_research.acquired_review import projection_terminal_control
    from packages.planner_control.models import Command, Control, Event, Outcome, Phase
    from tests.agent_provider.test_bounded_live import binding

    original = binding()
    event = Event(
        command=Command(key="live-review:review", expected_revision=0, action="review"),
        request_checksum=digest("original-request"),
        before=Phase.REVIEW,
        after=Phase.STOPPED,
        graph=original.graph,
        outcome=Outcome(reason="cancelled"),
        at=original.started_at,
    )
    control = Control(
        binding=original,
        graph=original.graph,
        phase=Phase.STOPPED,
        cancelled=True,
        events=(event,),
        steps=1,
    )
    phase = dict(planner=dict(control_checksum=digest("original-before")))
    closed = dict(
        schema_version="multimodal-review-cancellation/1.0.0",
        before=phase["planner"]["control_checksum"],
        after=digest(control),
        event=event.model_dump(mode="json"),
        cancelled=True,
        new_requests=0,
        remote_response_unknown_preserved=True,
    )
    docs = SimpleNamespace(read=lambda _: (closed,))
    assert projection_terminal_control(docs, control, phase, tmp_path) == digest((closed,))
    for field, value in (
        ("after", digest("other")),
        ("before", digest("other")),
        ("event", {}),
        ("new_requests", 1),
        ("cancelled", False),
        ("remote_response_unknown_preserved", False),
    ):
        bad = SimpleNamespace(read=lambda _, f=field, v=value: (dict(closed, **{f: v}),))
        with pytest.raises(ValueError, match="unproven_cancellation"):
            projection_terminal_control(bad, control, phase, tmp_path)
    with pytest.raises(ValueError, match="unproven_cancellation"):
        projection_terminal_control(SimpleNamespace(read=lambda _: None), control, phase, tmp_path)
    (tmp_path / "cancel").touch()
    with pytest.raises(ValueError, match="terminal_control"):
        projection_terminal_control(docs, control, phase, tmp_path)

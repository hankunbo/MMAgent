"""Explicit cancelled inference replacement; old runs and tool UNKNOWN stay closed."""

from types import SimpleNamespace

import pytest

from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.replacement import close_failed_review, review_replaceable


def test_single_item_configuration_preserves_transport_and_old_default():
    old, new = configuration(), configuration(review_replacement=True)
    assert old.review_batching.batch_size == 3
    assert new.review_batching.batch_size == 1
    assert old.transport == new.transport and old.roles == new.roles
    assert old.max_requests == new.max_requests == 40


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "active",
        "pending",
        "not_cancelled",
        "accepted",
        "vision",
        "success",
        "unconfirmed",
        "schema",
    ],
)
def test_only_cancelled_unaccepted_read_only_timeout_can_be_replaced(fault):
    phase = dict(target=None, accepted=False, acceptance_key=None)
    control = SimpleNamespace(phase="stopped", cancelled=True, pending=None)
    events = [
        dict(phase="reserved"),
        dict(phase="completed", record=dict(error="timeout", api_requests=1)),
    ]
    if fault == "active":
        control.phase = "select"
    if fault == "pending":
        control.pending = object()
    if fault == "not_cancelled":
        control.cancelled = False
    if fault == "accepted":
        phase["accepted"] = True
    if fault == "vision":
        phase["target"] = "check:oct"
    if fault == "success":
        events[1]["record"]["error"] = None
    if fault == "unconfirmed":
        events.pop()
    if fault == "schema":
        events[1]["record"]["error"] = "schema"
    assert review_replaceable(phase, control, events) == (fault is None)


def test_failed_unclaimed_review_cancels_original_command_without_io():
    from tests.planner.support import advance, command, setup

    planner, runtime, owner = setup()
    advance(planner, until="select")
    run_id = runtime.state.binding.run_id
    cmd = command(planner, "review", "live-review:review")
    planner.begin_recovery_review(run_id, cmd)
    before = list(owner.calls)
    result = dict(run_id=run_id, target=None, accepted=False)
    closed = close_failed_review(runtime, result)
    assert closed["cancelled"] and closed["new_requests"] == 0
    assert runtime.state.phase == "stopped" and runtime.state.pending is None
    assert owner.calls == before
    saved = runtime.state
    assert close_failed_review(runtime, result) is None and runtime.state == saved


def test_replacement_cannot_be_replaced_again(tmp_path):
    from apps.vectorengine_single_case import write_once
    from packages.multimodal_research.campaign import MultimodalCampaign

    write_once(tmp_path / "entry.json", dict(review_replacement=True))
    write_once(tmp_path / "result.json", dict(review_replacement="cancelled-read-only-once/1.0.0"))
    campaign = object.__new__(MultimodalCampaign)
    campaign.base = SimpleNamespace(previous=tmp_path, review_replacement=True)
    with pytest.raises(ValueError, match="multimodal_review_replacement_once"):
        campaign.previous_attempts()


def test_replacement_cannot_be_requested_without_prior_timeout():
    from packages.multimodal_research.campaign import MultimodalCampaign

    campaign = object.__new__(MultimodalCampaign)
    campaign.base = SimpleNamespace(previous=None, review_replacement=True)
    with pytest.raises(ValueError, match="requires_one_cancelled_timeout"):
        campaign.previous_attempts()

"""Boundary tests for current-owner active acquisition continuation."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.active_continuation import (
    load_parent,
    ordered_targets,
    reserve_review,
)


def parent(tmp_path):
    task = {"checksum": digest("task")}
    profile = digest("profile")
    spec = dict(
        schema_version="multimodal-live-entry/1.8.0",
        input_checksum=digest("input"),
        intent=task,
        active_acquisition={"checksum": profile},
        implementation={"fixture": "old"},
        provider={"version": "fixture"},
    )
    protocol = dict(
        input=spec["input_checksum"],
        task=task["checksum"],
        implementation=spec["implementation"],
        provider=digest(spec["provider"]),
        deadline=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
    )
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "lock").touch()
    args = dict(checksum=spec["input_checksum"], intent=task, profile_checksum=profile)
    return spec, protocol, args


def test_expired_original_checkpoint_can_be_considered_without_fake_result(tmp_path):
    spec, protocol, args = parent(tmp_path)
    assert load_parent(tmp_path, **args) == (spec, protocol)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "change", ["input", "profile", "implementation", "cancel", "active", "repeat", "complete"]
)
def test_parent_binding_and_no_repeat_are_not_softened(tmp_path, change):
    spec, protocol, args = parent(tmp_path)
    if change == "input":
        args["checksum"] = digest("other")
    elif change == "profile":
        args["profile_checksum"] = digest("other")
    elif change == "implementation":
        protocol["implementation"] = {"fixture": "changed"}
    elif change == "cancel":
        (tmp_path / "cancel").touch()
    elif change == "active":
        protocol["deadline"] = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    elif change == "repeat":
        spec["active_continuation_of"] = "earlier"
    elif change == "complete":
        (tmp_path / "result.json").write_text(json.dumps({"complete": True}))
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="active_continuation_"):
        load_parent(tmp_path, **args)


def test_schedule_visits_unseen_modality_before_repeated_fundus_checks():
    ledger = [
        dict(id=t, input_modalities=(m,))
        for t, m in (
            ("native", "color_fundus"),
            ("f1", "color_fundus"),
            ("f2", "color_fundus"),
            ("o1", "OCT"),
            ("o2", "OCT"),
        )
    ]
    assert ordered_targets(ledger, ("f1", "f2", "o1", "o2"), {"native": object()}) == (
        "o1",
        "f1",
        "o2",
        "f2",
    )
    assert ordered_targets(ledger, ("f2", "o1", "o2"), {"native": object(), "f1": object()}) == (
        "o1",
        "o2",
        "f2",
    )


def test_closing_reserve_keeps_deadline_and_request_ceiling():
    assert not reserve_review(seconds_left=5400, acquired=8, batch_size=3, requests_used=8)[
        "stop_acquisition"
    ]
    value = reserve_review(seconds_left=900, acquired=8, batch_size=3, requests_used=8)
    assert value["stop_acquisition"] and value["review_reserve_seconds"] > 900
    assert reserve_review(seconds_left=5400, acquired=8, batch_size=3, requests_used=38)[
        "stop_acquisition"
    ]


def test_active_continuation_protocol_survives_disk_roundtrip(tmp_path):
    from packages.multimodal_research.active_continuation import ActiveContinuation
    from packages.multimodal_research.campaign import MultimodalCampaign

    pool = object.__new__(ActiveContinuation)
    pool.initial_attempts, pool.origins, pool.values = [], {}, {}
    pool.attempted = ("active:rim_focal", "active:peripapillary")
    c = object.__new__(MultimodalCampaign)
    c.output, c.campaign_id, c.targets, c.config = tmp_path, "roundtrip", (), {}
    c.base = SimpleNamespace(
        active_seconds=5400,
        active_schedule_v19=True,
        manifest=SimpleNamespace(checksum=digest("input")),
        visible_value=SimpleNamespace(checksum=digest("visible")),
        task=SimpleNamespace(checksum=digest("task")),
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
    )
    c.continuation, c.previous_attempts = pool, lambda: []
    frozen = c.freeze()
    assert c.freeze() == frozen
    assert json.loads((tmp_path / "protocol.json").read_bytes()) == frozen
    assert pool.report() == json.loads(json.dumps(pool.report()))
    pool.attempted = (*pool.attempted, "active:changed")
    with pytest.raises(ValueError, match="multimodal_campaign_frozen_input_or_code_changed"):
        c.freeze()

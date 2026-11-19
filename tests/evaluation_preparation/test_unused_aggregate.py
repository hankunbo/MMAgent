"""Unused aggregate alternatives: no widening of facts, method or dispatch budget."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import alternative_acquisition as a
from packages.multimodal_research import campaign_renewal as c
from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research import unstarted_window as w
from packages.object_store import checksum


def old_plan():
    now = datetime.now(UTC)
    raw = b"synthetic-unchanged-view-pixels"
    view = dict(
        method=a.TILED_METHOD,
        source_checksum=digest("source"),
        native_checksum=digest("native"),
        checksum=checksum(raw),
    )
    baseline = NS(state={"accepted": "state"}, manifest={"accepted": "manifest"})
    spec = dict(
        target="observation",
        method=a.TILED_METHOD,
        source_checksum=view["source_checksum"],
        native_checksum=view["native_checksum"],
    )
    plan = dict(
        schema_version=a.AGGREGATE_VERSION,
        mode="acquisition",
        acquisition_increment_only=True,
        acquisition_targets=["observation"],
        review_target_ids=["observation"],
        acquisition_selection=dict(selected=["observation"]),
        previous_attempts=[dict(requests=32)],
        max_new_requests=3,
        max_cumulative_requests=40,
        seconds=5400,
        started_at=(now - timedelta(hours=3)).isoformat(),
        deadline=(now - timedelta(hours=1)).isoformat(),
        alternative_view=spec,
        alternative_view_metadata_checksum=digest(view),
        accepted_state_checksum=digest(baseline.state),
        accepted_origin_checksum=digest(baseline.manifest),
        execution_lease_profile=lease.PROFILE,
        new_public_search_allowed=False,
        original_unknown_not_replayed=True,
        provider=digest("same-provider"),
        implementation={"old.py": "before"},
        reuse_assessment_ids=["original-accepted"],
    )
    return now, plan, baseline, raw, view


def build(old, now, previous="old", output="new"):
    return w.plan_for(
        old,
        previous=previous,
        output=output,
        authorization="explicit unused-only successor",
        started=now,
        implementation={"new.py": "after"},
    )


def owner(tmp_path, monkeypatch):
    now, plan, baseline, raw, view = old_plan()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "protocol.json").write_text(json.dumps(plan))
    (tmp_path / "alternative-view").mkdir()
    (tmp_path / "alternative-view/view.json").write_text(json.dumps(view))
    (tmp_path / "alternative-view/view.png").write_bytes(raw)
    (tmp_path / "alternative-preparation").mkdir()
    parent = NS(aggregation_baseline=baseline, base=NS(check=lambda: None))
    origin = NS(
        output=tmp_path,
        protocol=plan,
        parent=parent,
        base=NS(runtime=NS(read_observation_bundle=lambda _: None)),
        initial_targets=("observation",),
        contexts=NS(read_research_run=lambda *args, **kwargs: None),
        sessions=NS(read=lambda _: None),
        request_key=lambda target: "old-unused:" + str(target),
        phase_path=lambda target: tmp_path / str(target),
    )
    monkeypatch.setattr(
        a, "proof", lambda actual, target, method: copy.deepcopy(plan["alternative_view"])
    )
    return origin, now


def test_successor_preserves_every_nonwindow_field_and_old_bytes():
    now, old, _, _, _ = old_plan()
    before = copy.deepcopy(old)
    new = build(old, now)
    changed = {k for k in new.keys() | old.keys() if old.get(k) != new.get(k)}
    assert changed == {
        "schema_version",
        "authorization",
        "started_at",
        "deadline",
        "implementation",
        "unstarted_window",
    }
    assert old == before and new["schema_version"] == w.AGGREGATE_VERSION
    assert datetime.fromisoformat(new["deadline"]) == now + timedelta(seconds=5400)
    assert new["unstarted_window"]["previous_protocol"] == digest(old)
    w.validate_plan(old, new, previous="old", output="new")
    assert w.AGGREGATE_VERSION in c.UNSTARTED_WINDOW_VERSIONS
    assert w.AGGREGATE_VERSION not in c.EXACT_ACQUISITION_VERSIONS | c.RETAINED_ACQUISITION_VERSIONS


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_new_requests", 4),
        ("previous_attempts", [{"requests": 38}]),
        ("previous_attempts", [{"requests": True}]),
        ("mode", "review"),
        ("acquisition_targets", ["different"]),
        ("review_target_ids", []),
        ("acquisition_increment_only", False),
        ("new_public_search_allowed", True),
        ("seconds", 99999),
        ("execution_lease_profile", "other"),
        ("unstarted_window", {"old": True}),
        ("unknown_acquisition", {"old": True}),
        ("exact_request_budget", "changed"),
    ],
)
def test_no_scope_or_budget_expansion(field, value):
    now, old, *_ = old_plan()
    old[field] = value
    with pytest.raises(ValueError, match="unstarted_aggregate_scope"):
        build(old, now)


def test_current_owner_and_exact_pixels_before_any_dispatch(tmp_path, monkeypatch):
    origin, now = owner(tmp_path / "old", monkeypatch)
    before = (origin.output / "protocol.json").read_bytes()
    w.verify_unused(origin, origin.protocol)
    destination = tmp_path / "new"
    destination.mkdir()
    w.copy_aggregate_material(origin, destination)
    for name in ("view.json", "view.png"):
        assert (destination / "alternative-view" / name).read_bytes() == (
            origin.output / "alternative-view" / name
        ).read_bytes()
    w.copy_aggregate_material(origin, destination)
    assert (origin.output / "protocol.json").read_bytes() == before
    new = build(origin.protocol, now)
    successor = NS(**dict(vars(origin), protocol=new))
    lease.validate_aggregate(successor)
    (destination / "alternative-view/view.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="unstarted_aggregate_material_changed"):
        w.copy_aggregate_material(origin, destination)


@pytest.mark.parametrize("fault", ["context", "control", "bundle", "source", "pixels", "cancel"])
def test_started_or_changed_original_refuses(tmp_path, monkeypatch, fault):
    origin, _ = owner(tmp_path, monkeypatch)
    if fault == "context":
        origin.contexts.read_research_run = lambda *a, **k: object()
    elif fault == "control":
        origin.sessions.read = lambda _: object()
    elif fault == "bundle":
        origin.base.runtime.read_observation_bundle = lambda _: ()
    elif fault == "source":
        monkeypatch.setattr(a, "proof", lambda *a, **k: dict(changed=True))
    elif fault == "pixels":
        (tmp_path / "alternative-view/view.png").write_bytes(b"changed")
    else:
        (tmp_path / "cancel").touch()
    with pytest.raises(
        ValueError,
        match="phase_already_registered|unstarted_window_control|view_already_registered|alternative_view_plan_binding|alternative_view_material_changed|unstarted_window_activity",
    ):
        w.verify_unused(origin, origin.protocol)


def test_cli_run_routes_to_original_aggregate_preparer_only(monkeypatch, tmp_path):
    now, old, *_ = old_plan()
    (tmp_path / "protocol.json").write_text(json.dumps(old))
    calls = []
    monkeypatch.setattr(w, "prepare_aggregate", lambda **kw: calls.append(kw) or {"complete": True})
    assert w.prepare_and_run(previous=tmp_path, output="new", authorization="explicit") == {
        "complete": True
    }
    assert calls == [dict(previous=tmp_path, output="new", authorization="explicit", run=True)]


def test_recursive_or_tampered_successor_is_not_authorized():
    now, old, *_ = old_plan()
    new = build(old, now)
    with pytest.raises(ValueError, match="unstarted_window_scope"):
        build(new, now + timedelta(hours=3))
    for field, value in [
        ("provider", "changed"),
        ("acquisition_targets", ["other"]),
        ("alternative_view_metadata_checksum", "changed"),
    ]:
        tampered = copy.deepcopy(new)
        tampered[field] = value
        with pytest.raises(ValueError, match="unstarted_window_plan_binding"):
            w.validate_plan(old, tampered, previous="old", output="new")

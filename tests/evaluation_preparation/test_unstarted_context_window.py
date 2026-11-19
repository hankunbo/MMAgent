"""Unused Review recovery cannot replay, erase history or renew itself."""

import copy
import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import unstarted_window as w


def old_plan(path):
    now = datetime.now(UTC)
    return now, dict(
        schema_version=w.REVIEW_ORIGIN_VERSION,
        mode="review",
        output=str(path.absolute()),
        started_at=(now - timedelta(hours=3)).isoformat(),
        deadline=(now - timedelta(hours=1)).isoformat(),
        seconds=5400,
        metadata_profile="grape-rnfl-device-unit/1.0.0",
        aggregate_context_review=True,
        acquisition_targets=[],
        reuse_assessment_ids=[],
        reassessment_ids=["a", "b"],
        review_target_ids=["a", "b"],
        new_vision_requests=0,
        new_public_search_allowed=False,
        original_unknown_not_replayed=True,
        previous_attempts=[dict(requests=26)],
        max_new_requests=2,
        max_model_dispatches=1,
        max_cumulative_requests=40,
        execution_lease_profile="original-runtime-case-pipeline-session/1.0.0",
        provider=digest("provider"),
        implementation={"old": digest("old")},
        tip=dict(run_id="RUN:parent"),
        material_context_review=dict(contexts={"a": [], "b": []}),
    )


def make(old, now, output):
    return w.plan_for(
        old,
        previous=old["output"],
        output=output,
        authorization="bounded",
        started=now,
        implementation={"new": digest("new")},
    )


def test_only_explicit_clock_identity_changes(tmp_path):
    now, old = old_plan(tmp_path / "old")
    before = copy.deepcopy(old)
    plan = make(old, now, tmp_path / "new")
    assert old == before and plan["schema_version"] == w.REVIEW_VERSION
    changed = {k for k in old if plan[k] != old[k]}
    assert changed == {"schema_version", "output", "started_at", "deadline", "implementation"}
    assert plan["unstarted_window"]["previous_protocol"] == digest(old)
    assert plan["previous_attempts"] == old["previous_attempts"]
    w.validate_plan(old, plan, previous=old["output"], output=tmp_path / "new")
    with pytest.raises(ValueError, match="unstarted_window_scope"):
        make(plan, now + timedelta(hours=3), tmp_path / "again")


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "acquisition"},
        {"acquisition_targets": ["extra"]},
        {"new_vision_requests": 1},
        {"new_public_search_allowed": True},
        {"review_target_ids": ["b"]},
        {"reuse_assessment_ids": ["a"]},
        {"max_new_requests": True},
        {"max_model_dispatches": 0},
        {"previous_attempts": [{"requests": 39}]},
        {"seconds": 999999},
        {"metadata_profile": "unreviewed"},
        {"aggregate_context_review": False},
        {"deadline": (datetime.now(UTC) + timedelta(hours=1)).isoformat()},
    ],
)
def test_scope_denials(tmp_path, change):
    now, old = old_plan(tmp_path / "old")
    old.update(change)
    with pytest.raises(ValueError, match="unstarted_review_scope"):
        make(old, now, tmp_path / "new")


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", digest("another")),
        ("max_new_requests", 3),
        ("previous_attempts", [{"requests": 0}]),
        ("review_target_ids", ["a"]),
    ],
)
def test_successor_cannot_change_saved_binding(tmp_path, field, value):
    now, old = old_plan(tmp_path / "old")
    plan = make(old, now, tmp_path / "new")
    plan[field] = value
    with pytest.raises(ValueError, match="unstarted_window_plan_binding"):
        w.validate_plan(old, plan, previous=old["output"], output=tmp_path / "new")


def test_registered_context_and_control_reject_even_without_local_request(tmp_path):
    path = tmp_path / "old"
    path.mkdir()
    _, old = old_plan(path)
    (path / "protocol.json").write_text(json.dumps(old))
    calls = []
    c = SimpleNamespace(
        output=path,
        initial_targets=(),
        contexts=SimpleNamespace(read_research_run=lambda run_id, **kw: calls.append(run_id)),
        sessions=SimpleNamespace(read=lambda _: None),
        request_key=lambda _: "unique-review",
    )
    w.verify_unused(c, old)
    assert len(calls) == 1
    c.contexts.read_research_run = lambda *a, **kw: object()
    with pytest.raises(ValueError, match="unstarted_review_phase_already_registered"):
        w.verify_unused(c, old)
    c.contexts.read_research_run = lambda *a, **kw: None
    c.sessions.read = lambda _: object()
    with pytest.raises(ValueError, match="unstarted_window_control"):
        w.verify_unused(c, old)
    c.sessions.read = lambda _: None
    (path / "cancel").touch()
    with pytest.raises(ValueError, match="unstarted_window_activity"):
        w.verify_unused(c, old)


def test_new_review_keeps_review_time_reserve_and_context_restoration():
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    c = object.__new__(RenewedCampaign)
    deadline = datetime.now(UTC) + timedelta(hours=1)
    c.base = SimpleNamespace(detailed=True)
    c.protocol = dict(schema_version=w.REVIEW_VERSION, deadline=deadline.isoformat())
    before = c.work_deadline()
    with c.phase_context(None):
        assert c.work_deadline() == deadline - timedelta(seconds=60)
    assert c.work_deadline() == before


def test_atomic_review_entry_assembles_once_and_keeps_lost_ack_plan(tmp_path, monkeypatch):
    from packages.multimodal_research import campaign_renewal as renewal
    from packages.multimodal_research import context_review, execution_lease, implementation

    old_path = tmp_path / "old"
    old_path.mkdir()
    _, old = old_plan(old_path)
    (old_path / "protocol.json").write_text(json.dumps(old))
    original_bytes = (old_path / "protocol.json").read_bytes()
    values = {
        context_review.claim_key(old): (dict(output=str(old_path), plan_checksum=digest(old)),)
    }
    docs = SimpleNamespace(
        read=values.get, write=lambda k, v: values.__setitem__(k, v), session=nullcontext
    )
    origin = SimpleNamespace(
        output=old_path,
        protocol=old,
        initial_targets=(),
        contexts=SimpleNamespace(read_research_run=lambda *a, **k: None),
        sessions=SimpleNamespace(read=lambda _: None),
        request_key=lambda _: "review-id",
        documents=lambda _: docs,
    )
    assembled, executed = [], []
    monkeypatch.setattr(renewal, "assemble", lambda p: assembled.append(p) or origin)
    monkeypatch.setattr(execution_lease, "session", lambda _: nullcontext())
    monkeypatch.setattr(implementation, "freeze", lambda: {"new": digest("new")})
    monkeypatch.setattr(
        w,
        "build_review",
        lambda *a: SimpleNamespace(execute=lambda: executed.append(a[2]) or dict(complete=True)),
    )
    assert w.prepare_and_run(
        previous=old_path, output=tmp_path / "new", authorization="bounded"
    ) == dict(complete=True)
    assert len(assembled) == len(executed) == 1
    assert (old_path / "protocol.json").read_bytes() == original_bytes
    plan = executed[0]
    saved = values[w.claim_key(old)]
    repeated = w.prepare(previous=old_path, output=tmp_path / "new", authorization="recover")
    assert repeated["deadline"] == plan["deadline"] and values[w.claim_key(old)] == saved
    with pytest.raises(ValueError, match="unstarted_window_already_claimed"):
        w.prepare(previous=old_path, output=tmp_path / "fork", authorization="fork")

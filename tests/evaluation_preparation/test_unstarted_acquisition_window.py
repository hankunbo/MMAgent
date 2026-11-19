"""An unused expired plan can be superseded, never replayed or edited."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import unstarted_window as w


def old_plan():
    now = datetime.now(UTC)
    return now, dict(
        schema_version="multimodal-campaign-renewal/1.37.0",
        mode="acquisition",
        acquisition_targets=["a", "b", "c"],
        acquisition_selection=dict(selected=["a", "b", "c"]),
        previous_attempts=[dict(requests=13), dict(requests=4)],
        max_new_requests=4,
        max_model_dispatches=4,
        max_cumulative_requests=40,
        seconds=5400,
        started_at=(now - timedelta(hours=3)).isoformat(),
        deadline=(now - timedelta(hours=1)).isoformat(),
        provider=digest("unchanged-provider"),
        original_unknown_not_replayed=True,
        execution_lease_profile="original-runtime-case-pipeline-session/1.0.0",
        new_public_search_allowed=False,
        expansion_policy=None,
        reuse_assessment_ids=["old-1", "old-2"],
        implementation={"old.py": "old"},
        exact_request_budget="dispatch_only_local_completion_zero_request/1.0",
    )


def build(old, now, **kwargs):
    return w.plan_for(
        old,
        previous="old",
        output="new",
        authorization="goal-bounded-recovery",
        started=now,
        implementation={"new.py": "new"},
        **kwargs,
    )


def test_only_explicit_window_identity_changes_and_no_history_reset():
    now, old = old_plan()
    before = copy.deepcopy(old)
    plan = build(old, now)
    assert old == before
    assert plan["schema_version"] == w.VERSION
    assert plan["provider"] == old["provider"]
    assert plan["previous_attempts"] == old["previous_attempts"]
    assert plan["reuse_assessment_ids"] == old["reuse_assessment_ids"]
    assert plan["acquisition_targets"] == old["acquisition_targets"]
    assert plan["max_new_requests"] == plan["max_model_dispatches"] == 4
    assert datetime.fromisoformat(plan["deadline"]) == now + timedelta(seconds=5400)
    assert plan["unstarted_window"]["previous_protocol"] == digest(old)
    assert plan["unstarted_window"]["predecessor_requests"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "multimodal-campaign-renewal/1.49.0"},
        {"mode": "review"},
        {"acquisition_targets": []},
        {"acquisition_targets": ["a", "a"]},
        {"max_new_requests": 0},
        {"max_new_requests": True},
        {"max_model_dispatches": 9},
        {"previous_attempts": [dict(requests=39)]},
        {"previous_attempts": [dict(requests=True)]},
        {"previous_attempts": [dict(requests=-1)]},
        {"seconds": 999999},
        {"seconds": True},
        {"deadline": (datetime.now(UTC) + timedelta(hours=1)).isoformat()},
        {"new_public_search_allowed": True},
        {"original_unknown_not_replayed": False},
        {"review_clarification": {"assessment_ids": ["a"]}},
        {"recovery_http_client": {"changed": True}},
        {"execution_lease_profile": "other"},
    ],
)
def test_no_implicit_scope_budget_or_recursive_extension(change):
    now, old = old_plan()
    old.update(change)
    with pytest.raises(ValueError, match="unstarted_window_scope"):
        build(old, now)


def owner(tmp_path, old):
    (tmp_path / "protocol.json").write_text(json.dumps(old))
    return SimpleNamespace(
        output=tmp_path,
        protocol=old,
        initial_targets=tuple(old["acquisition_targets"]),
        request_key=lambda target: "unused-window:" + str(target),
        phase_path=lambda target: tmp_path / str(target),
        contexts=SimpleNamespace(read_research_run=lambda *a, **k: None),
        sessions=SimpleNamespace(read=lambda _: None),
    )


@pytest.mark.parametrize("target", [None, "a", "b", "c"])
def test_every_phase_runtime_checked_even_if_mirror_missing(tmp_path, target):
    from db.runtime.repository.case_pipeline import identity

    _, old = old_plan()
    c = owner(tmp_path, old)
    rid = identity("RUN", "task-research:" + c.request_key(target))
    c.contexts.read_research_run = lambda run, **k: object() if run == rid else None
    with pytest.raises(ValueError, match="phase_already_registered"):
        w.verify_unused(c, old)


@pytest.mark.parametrize("name", ["cancel", "result.json", "stage-failure.json", "review"])
def test_local_activity_or_cancellation_blocks(tmp_path, name):
    _, old = old_plan()
    c = owner(tmp_path, old)
    (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="unstarted_window_activity"):
        w.verify_unused(c, old)


def test_unused_is_not_inferred_from_files_alone(tmp_path):
    _, old = old_plan()
    c = owner(tmp_path, old)
    w.verify_unused(c, old)
    c.sessions.read = lambda _: object()
    with pytest.raises(ValueError, match="unstarted_window_control"):
        w.verify_unused(c, old)


def test_provider_and_target_tamper_rejected_on_reconstruction():
    now, old = old_plan()
    p = build(old, now)
    for key, value in (("provider", "other"), ("acquisition_targets", ["different"])):
        edited = copy.deepcopy(p)
        edited[key] = value
        with pytest.raises(ValueError, match="unstarted_window_plan_binding"):
            w.validate_plan(old, edited, previous="old", output="new")


def test_final_review_reserve_and_exact_budget_keep_original_semantics():
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    c = object.__new__(RenewedCampaign)
    deadline = datetime.now(UTC) + timedelta(hours=1)
    c.base = SimpleNamespace(detailed=True)
    c.protocol = dict(schema_version=w.VERSION, deadline=deadline.isoformat())
    assert c.request_finalization_headroom == 0
    assert c.work_deadline() == deadline - timedelta(seconds=300)
    with c.phase_context(None):
        assert c.work_deadline() == deadline - timedelta(seconds=60)
    assert not c._final_review_active


def test_cli_selects_explicit_window_without_changing_provider(monkeypatch, capsys):
    from apps.multimodal_case import main

    seen = []
    monkeypatch.setattr(w, "prepare", lambda **kw: seen.append(kw) or dict(api_requests=0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "case",
            "renewal-window-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--authorization",
            "explicit",
        ],
    )
    assert main() == 0
    assert seen == [dict(previous="old", output="new", authorization="explicit")]
    assert '"api_requests": 0' in capsys.readouterr().out

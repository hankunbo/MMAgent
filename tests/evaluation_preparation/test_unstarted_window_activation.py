"""One explicit terminal activation of an unused successor, no silent refresh."""

import copy
from datetime import timedelta

import pytest

from packages.multimodal_research import unstarted_window as w
from tests.evaluation_preparation.test_unstarted_acquisition_window import old_plan


def expired_successor():
    now, old = old_plan()
    first = w.plan_for(
        old,
        previous="old",
        output="prepared",
        authorization="first-window",
        started=now,
        implementation={},
    )
    return now + timedelta(hours=3), first


def test_final_activation_retains_every_charge_scope_and_provider():
    now, old = expired_successor()
    original = copy.deepcopy(old)
    new = w.plan_for(
        old,
        previous="prepared",
        output="activated",
        authorization="terminal-run",
        started=now,
        implementation={},
        activate=True,
    )
    assert old == original
    assert new["schema_version"] == "multimodal-campaign-renewal/1.50.0"
    for key in (
        "provider",
        "previous_attempts",
        "reuse_assessment_ids",
        "max_new_requests",
        "max_model_dispatches",
        "acquisition_targets",
    ):
        assert new[key] == old[key]
    assert new["unstarted_window"]["predecessor_requests"] == 0
    w.validate_plan(old, new, previous="prepared", output="activated")


def test_activation_does_not_enable_unbounded_recursive_windows():
    now, old = expired_successor()
    new = w.plan_for(
        old,
        previous="prepared",
        output="activated",
        authorization="terminal-run",
        started=now,
        implementation={},
        activate=True,
    )
    with pytest.raises(ValueError, match="unstarted_window_scope"):
        w.plan_for(
            new,
            previous="activated",
            output="again",
            authorization="again",
            started=now + timedelta(hours=3),
            implementation={},
            activate=True,
        )
    with pytest.raises(ValueError, match="unstarted_window_scope"):
        w.plan_for(
            old,
            previous="prepared",
            output="ordinary",
            authorization="ordinary",
            started=now,
            implementation={},
        )


def test_atomic_cli_runs_original_owner_only_after_prepare_success(monkeypatch, capsys):
    from apps.multimodal_case import main

    seen = []
    monkeypatch.setattr(w, "prepare_and_run", lambda **kw: seen.append(kw) or dict(complete=True))
    monkeypatch.setattr(
        "sys.argv",
        [
            "case",
            "renewal-window-run",
            "--previous",
            "expired",
            "--output",
            "new",
            "--authorization",
            "explicit",
        ],
    )
    assert main() == 0
    assert seen == [dict(previous="expired", output="new", authorization="explicit")]
    assert '"complete": true' in capsys.readouterr().out


def test_activation_preserves_exact_headroom_and_scoped_review_reserve():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    c = object.__new__(RenewedCampaign)
    deadline = datetime.now(UTC) + timedelta(hours=1)
    c.base = SimpleNamespace(detailed=True)
    c.protocol = dict(
        schema_version="multimodal-campaign-renewal/1.50.0", deadline=deadline.isoformat()
    )
    assert c.request_finalization_headroom == 0
    with c.phase_context(None):
        assert c.work_deadline() == deadline - timedelta(seconds=60)
    assert c.work_deadline() == deadline - timedelta(seconds=300)


def test_atomic_entry_never_executes_if_preparation_fails(monkeypatch, tmp_path):
    from packages.multimodal_research import campaign_renewal as renewal

    calls = []

    def fail(**kwargs):
        calls.append("prepare")
        raise ValueError("unstarted_window_activity")

    monkeypatch.setattr(w, "prepare", fail)
    monkeypatch.setattr(renewal, "assemble", lambda *a, **kw: calls.append("execute"))
    with pytest.raises(ValueError, match="unstarted_window_activity"):
        w.prepare_and_run(previous=tmp_path, output=tmp_path / "new", authorization="explicit")
    assert calls == ["prepare"]


def test_atomic_entry_executes_original_owner_once(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from packages.multimodal_research import campaign_renewal as renewal

    calls = []
    monkeypatch.setattr(w, "prepare", lambda **kw: calls.append("prepare") or {})
    monkeypatch.setattr(
        renewal,
        "assemble",
        lambda *a, **kw: SimpleNamespace(
            execute=lambda: calls.append("execute") or dict(complete=False, requests=1)
        ),
    )
    result = w.prepare_and_run(previous=tmp_path, output=tmp_path / "new", authorization="explicit")
    assert result == dict(complete=False, requests=1)
    assert calls == ["prepare", "execute"]

"""No-dispatch successor preserves facts, cumulative budget and old failure."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import capability_recovery as recovery
from packages.multimodal_research import capability_window as window
from tests.evaluation_preparation.test_capability_window import plan_fixture


def fixture(tmp_path):
    parent, tip, inv, kwargs = plan_fixture(tmp_path)
    kwargs["now"] = datetime.now(UTC) - timedelta(hours=3)
    tip["run_id"] = "RUN:SYNTHETIC_PARENT"
    role = next(r for r in parent.config.roles if r.role == "vision")
    inv["scope_plan"] = dict(
        prompt_version=role.prompt_version, prompt_checksum=role.prompt_checksum
    )
    old = window.plan_for(parent, tip, inv, **kwargs)
    parent.base.task.target_code = "GOAL:GLAUCOMA"
    parent.base.binding = {"case_code": "CASE:SYNTHETIC"}
    parent.contexts = SimpleNamespace(read_research_run=lambda *a, **kw: None)
    parent.sessions = SimpleNamespace(read=lambda *a: None)
    parent.base.runtime = SimpleNamespace(read_observation_bundles=lambda keys: (None,) * len(keys))
    out = Path(old["output"])
    out.mkdir()
    (out / "protocol.json").write_text(json.dumps(old))
    failure = dict(
        schema_version="multimodal-stage-failure/1.1.0",
        campaign_id=out.name,
        protocol_checksum=digest(old),
        error_code="deadline",
        failure_kind="local_stage_failure",
        failure_stage="acquisition",
        input_checksum=old["input"],
        target=parent.base.task.target_code,
        previous_attempts=old["previous_attempts"],
        new_requests=0,
        phase_reservations=0,
        confirmed_phases=[],
        cancel_marker=False,
        score_status="not_scored",
        control={"exists": False},
    )
    (out / "stage-failure.json").write_text(json.dumps(failure))
    return parent, old, failure, inv


def read_origin(monkeypatch, parent, old):
    from packages.multimodal_research import capability_review_origin

    monkeypatch.setattr(capability_review_origin, "resolve", lambda p, o: o)
    return recovery.origin(parent, old["output"])


def test_new_window_preserves_every_old_fact_and_budget(tmp_path, monkeypatch):
    parent, old, _, _ = fixture(tmp_path)
    before = deepcopy(old)
    _, proof = read_origin(monkeypatch, parent, old)
    now = datetime.now(UTC)
    new = recovery.plan_for(
        old,
        proof,
        output=tmp_path / "recovery",
        authorization="continued bounded goal",
        now=now,
        implementation={},
    )
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {
        "schema_version",
        "output",
        "authorization",
        "started_at",
        "deadline",
        "prior_deadline",
    }
    assert new["previous_attempts"] == [{"requests": 33}]
    assert new["max_new_requests"] == 7
    assert new["max_model_dispatches"] == 6
    assert old == before
    assert new["prior_deadline"] == old["deadline"]
    assert datetime.fromisoformat(new["deadline"]) - now == timedelta(seconds=5400)


@pytest.mark.parametrize(
    "fault",
    [
        "cancel",
        "request",
        "result",
        "control",
        "journal",
        "reservation",
        "unknown",
        "wrong_input",
        "wrong_score",
    ],
)
def test_any_work_or_changed_failure_refuses_successor(tmp_path, monkeypatch, fault):
    parent, old, failure, _ = fixture(tmp_path)
    out = Path(old["output"])
    if fault in {"cancel", "request", "result"}:
        (
            out / {"cancel": "cancel", "request": "request.json", "result": "result.json"}[fault]
        ).write_text("{}")
    elif fault == "control":
        parent.sessions.read = lambda *a: object()
    elif fault == "journal":
        parent.base.runtime.read_observation_bundles = lambda keys: (object(),)
    else:
        key, value = {
            "reservation": ("phase_reservations", 1),
            "unknown": ("error_code", "UNKNOWN"),
            "wrong_input": ("input_checksum", digest("changed")),
            "wrong_score": ("score_status", "saved"),
        }[fault]
        failure[key] = value
        (out / "stage-failure.json").write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="capability_recovery"):
        read_origin(monkeypatch, parent, old)


@pytest.mark.parametrize("fault", ["negative", "budget", "future", "nested", "expired_proof"])
def test_plan_cannot_reset_budget_or_bypass_old_window(tmp_path, monkeypatch, fault):
    parent, old, _, _ = fixture(tmp_path)
    _, proof = read_origin(monkeypatch, parent, old)
    now, out = datetime.now(UTC), tmp_path / "recovery"
    if fault == "negative":
        old["previous_attempts"][0]["requests"] = -1
    elif fault == "budget":
        old["previous_attempts"][0]["requests"] = 34
    elif fault == "future":
        now = datetime.fromisoformat(old["started_at"])
    elif fault == "nested":
        out = Path(old["output"]) / "next"
    else:
        proof["predecessor_requests"] = 1
    proof["effective_protocol"] = digest(old)
    with pytest.raises(ValueError, match="capability_recovery_bounds"):
        recovery.plan_for(old, proof, output=out, authorization="goal", now=now, implementation={})


def test_registered_scope_requires_exact_original_identity(tmp_path, monkeypatch):
    from db.runtime.repository.case_pipeline import identity
    from packages.multimodal_research.input_scope import configuration

    parent, old, _, _ = fixture(tmp_path)
    target = old["input_scope"]["targets"][0]
    key = "mm-live:" + digest((Path(old["output"]).name, target))[7:55]
    run = identity("RUN", "task-research:" + key)
    scope = old["capability_transition"]["scope_plan"]
    policy = digest((scope, configuration(parent.config, scope), target))
    context = dict(
        run_id=run,
        request=dict(
            scope=parent.base.binding,
            input_checksum=old["input"],
            task=parent.base.task,
            policy_ref={"checksum": policy},
        ),
    )

    # Use an actual sealed context in PG; here test each independent binding gate.
    class Context(SimpleNamespace):
        pass

    ctx = Context(
        run_id=run,
        request=SimpleNamespace(
            scope=SimpleNamespace(**context["request"]["scope"]),
            input_checksum=old["input"],
            task=parent.base.task,
            policy_ref=SimpleNamespace(checksum=policy),
        ),
    )
    parent.contexts.read_research_run = lambda r, **kw: ctx if r == run else None
    # digest only this synthetic object, leaving plan digests untouched.
    original_digest = recovery.digest
    monkeypatch.setattr(
        recovery,
        "digest",
        lambda v: digest("synthetic context") if v is ctx else original_digest(v),
    )
    assert read_origin(monkeypatch, parent, old)[1]["phases"][1]["context"]
    ctx.request.policy_ref.checksum = digest("changed")
    with pytest.raises(ValueError, match="registered_context_binding"):
        recovery.origin(parent, old["output"])


def test_current_inventory_and_exact_claim_rechecked(tmp_path, monkeypatch):
    parent, old, _, found = fixture(tmp_path)
    _, proof = read_origin(monkeypatch, parent, old)
    plan = recovery.plan_for(
        old,
        proof,
        output=tmp_path / "recovery",
        authorization="goal",
        now=datetime.now(UTC),
        implementation={},
    )
    receipt = (window.claim(plan, plan["output"]),)
    parent.documents = lambda _: SimpleNamespace(read=lambda key: receipt)
    assert recovery.check(parent, plan, found=found, tip=old["tip"])[0] == old
    changed = deepcopy(found)
    changed["confirmed_observations_checksum"] = digest("changed")
    with pytest.raises(ValueError, match="current_inventory_changed"):
        recovery.check(parent, plan, found=changed, tip=old["tip"])
    receipt = None
    with pytest.raises(ValueError, match="claim_binding"):
        recovery.check(parent, plan)

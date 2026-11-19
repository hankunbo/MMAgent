"""Whole-case partial completion and same-window pending Review boundaries."""

from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.campaign_renewal import validate_context_window
from packages.multimodal_research.report import whole_case_rescored


def test_partial_acceptance_is_not_whole_case_rescored():
    final = dict(target=None, accepted=True)
    assert not whole_case_rescored(final, None)
    assert not whole_case_rescored(final, dict(review_completion=dict(pending_ids=["missing"])))
    assert whole_case_rescored(final, dict(review_completion=dict(pending_ids=[])))
    assert not whole_case_rescored(dict(target="vision", accepted=True), dict(root={}))


def test_pending_context_keeps_original_deadline_provider_and_no_acquisition():
    parent = SimpleNamespace(
        base=SimpleNamespace(execution_profile="relay-snapshot-context-v1"),
        protocol=dict(deadline="2030-01-01T01:00:00+00:00"),
        config={"fixed": "original"},
    )
    plan = dict(
        whole_case_execution_profile=parent.base.execution_profile,
        original_window_deadline=parent.protocol["deadline"],
        deadline=parent.protocol["deadline"],
        started_at="2030-01-01T00:40:00+00:00",
        pending_review={"selected_ids": ["a"]},
        new_vision_requests=0,
        new_public_search_allowed=False,
        provider=digest(parent.config),
    )
    validate_context_window(plan, parent)
    for key, value in (
        ("deadline", "2030-01-01T01:00:01+00:00"),
        ("provider", digest({})),
        ("new_vision_requests", 1),
        ("new_public_search_allowed", True),
        ("pending_review", None),
    ):
        with pytest.raises(ValueError, match="original_window_binding"):
            validate_context_window(dict(plan, **{key: value}), parent)


def test_new_structural_profile_preserves_checks_and_uses_pending_reserve():
    from packages.multimodal_research.active_requirements import load
    from packages.multimodal_research.definition_expansion import read_expansion_policy

    old = load("configs/research/multimodal_active_requirements_v9.json")
    new = load("configs/research/multimodal_structural_context_v1.json")
    assert old["checks"] == new["checks"] and old["nodes"] == new["nodes"]
    assert new["review_reserve_policy"]["local_seconds_per_batch"] == 60
    assert (
        read_expansion_policy(new["definition_expansion"]).prompt_version
        == "definition-draft-6.2.0"
    )


def unstarted_fixture(tmp_path):
    from datetime import UTC, datetime, timedelta

    from packages.multimodal_research.campaign_renewal import unstarted_context_proof
    from packages.planner_control.models import Control, Phase, Profile
    from tests.planner.support import binding

    now = datetime.now(UTC)
    start = now - timedelta(hours=1)
    b = binding(
        profile=Profile(seconds=60), started_at=start, deadline=start + timedelta(seconds=60)
    )
    control = Control(binding=b, graph=b.graph, phase=Phase.STOPPED)
    phase = dict(
        run_id=b.run_id, target=None, accepted=False, acceptance_key=None, error="deadline"
    )
    values = {"dynamic-phase-result": (phase,)}
    docs = SimpleNamespace(read=lambda key: values.get(key))
    parent = SimpleNamespace(
        protocol=dict(
            schema_version="multimodal-live-campaign/1.14.0", deadline=b.deadline.isoformat()
        ),
        config={"fixed_provider": "snapshot"},
        base=SimpleNamespace(
            execution_profile="relay-snapshot-context-v1",
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        documents=lambda rid: docs,
        sessions=SimpleNamespace(read=lambda rid: control),
    )
    result = dict(complete=False, phases=[phase])
    previous = [dict(requests=5, directory="prior")]
    tip = dict(path=str(tmp_path), run_id=b.run_id, previous_attempts=previous)
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.25.0",
        mode="review",
        facts=str(tmp_path),
        tip=tip,
        provider=digest(parent.config),
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        previous_attempts=previous,
        prior_deadline=parent.protocol["deadline"],
        seconds=2700,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=2700)).isoformat(),
        max_new_requests=8,
        max_model_dispatches=7,
        acquisition_targets=[],
        new_vision_requests=0,
        new_public_search_allowed=False,
        expansion_policy=None,
        unstarted_context_review=unstarted_context_proof(parent, result, tip),
    )
    return parent, result, plan, values


def test_zero_wire_deadline_allows_new_bounded_text_window_not_old_extension(tmp_path):
    from packages.multimodal_research.campaign_renewal import validate_unstarted_context

    parent, result, plan, _ = unstarted_fixture(tmp_path)
    old = dict(parent.protocol)
    validate_unstarted_context(plan, parent, result)
    assert parent.protocol == old
    for key, value in (
        ("provider", digest("different")),
        ("input", digest("different")),
        ("seconds", 9999),
        ("new_vision_requests", 1),
        ("new_public_search_allowed", True),
        ("max_new_requests", 40),
        ("acquisition_targets", ["invented"]),
    ):
        with pytest.raises(ValueError, match="renewal_unstarted_context_binding"):
            validate_unstarted_context(dict(plan, **{key: value}), parent, result)


@pytest.mark.parametrize(
    "boundary", ["reserved", "completed", "accepted", "unknown", "cancelled", "recursive"]
)
def test_zero_wire_review_window_rejects_started_or_wrong_origin(tmp_path, boundary):
    from packages.multimodal_research.campaign_renewal import unstarted_context_proof

    parent, result, plan, values = unstarted_fixture(tmp_path)
    if boundary in {"reserved", "completed"}:
        values["journal:0"] = (dict(phase=boundary),)
    elif boundary == "accepted":
        values["live-stage:review"] = ({"saved": True},)
    elif boundary == "unknown":
        result["phases"][0]["error"] = "timeout"
    elif boundary == "cancelled":
        (tmp_path / "cancel").touch()
    else:
        parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.25.0"
    with pytest.raises(ValueError, match="renewal_unstarted_context_not_zero_wire_deadline"):
        unstarted_context_proof(parent, result, plan["tip"])


def test_zero_wire_original_safe_stop_record_is_not_an_accepted_review(tmp_path):
    from packages.multimodal_research.campaign_renewal import unstarted_context_proof

    parent, result, plan, values = unstarted_fixture(tmp_path)
    phase = result["phases"][0]
    event = dict(code="review:stop", checksum=digest("stop"), role="review_control")
    phase["requests"] = dict(reservations=0, completed=0)
    stage = dict(
        schema_version="research-review-result/3.0.0",
        run_id=phase["run_id"],
        status="safe_stop",
        reason="deadline",
        domain_status="not_accepted",
        acceptance_key=None,
        new_medical_review=False,
        cancelled=False,
        recorded_agent_invocations=0,
        requests=phase["requests"],
        event=event,
    )
    phase["planner"] = dict(
        result_refs=[
            dict(
                code=event["code"],
                role="research_review",
                checksum=digest(stage),
                version=stage["schema_version"],
            )
        ]
    )
    values["live-stage:review"] = (stage,)
    proof = unstarted_context_proof(parent, result, plan["tip"])
    assert proof["stopped_review_checksum"] == digest((stage,))
    for key, value in (
        ("acceptance_key", "accepted:key"),
        ("run_id", "other"),
        ("reason", "timeout"),
        ("recorded_agent_invocations", 1),
        ("event", dict(event, checksum=digest("other"))),
        ("requests", dict(reservations=1, completed=0)),
    ):
        values["live-stage:review"] = (dict(stage, **{key: value}),)
        with pytest.raises(ValueError, match="renewal_unstarted_context_not_zero_wire_deadline"):
            unstarted_context_proof(parent, result, plan["tip"])


def test_unstarted_review_flag_reaches_original_cli_prepare(monkeypatch, capsys):
    from apps.multimodal_case import main
    from packages.multimodal_research import campaign_renewal

    calls = []
    monkeypatch.setattr(
        campaign_renewal, "prepare", lambda **kw: calls.append(kw) or {"prepared": True}
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "cli",
            "renewal-prepare",
            "--facts",
            "old",
            "--previous",
            "old",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "synthetic",
            "--unstarted-context-review",
        ],
    )
    assert main() == 0 and calls[0]["unstarted_context_review"] is True
    assert '"prepared": true' in capsys.readouterr().out

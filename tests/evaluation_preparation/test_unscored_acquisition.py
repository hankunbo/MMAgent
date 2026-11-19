"""Explicit route-only successor: original history and UNKNOWN are not retried."""

import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import unscored_acquisition as recovery

PROFILE = "configs/agents/multimodal_relay_environment_gpt6_review_curl_v1.json"
OLD = "configs/agents/multimodal_relay_local_gpt6_review_curl_v1.json"


class Documents:
    def __init__(self):
        self.values = {}
        self.writes = 0

    def session(self):
        return nullcontext()

    def read(self, key):
        return self.values.get(key)

    def write(self, key, value):
        assert key not in self.values
        self.values[key] = value
        self.writes += 1


def fixture(tmp_path, monkeypatch, documents=None):
    from packages.multimodal_research import implementation, unknown_acquisition

    old = tmp_path / "original"
    old.mkdir()
    result = dict(complete=False, phases=[dict(target=None, accepted=False, acceptance_key=None)])
    (old / "result.json").write_text(json.dumps(result))
    docs = Documents() if documents is None else documents
    parent = SimpleNamespace(
        config=read_provider_config(json.loads(Path(OLD).read_bytes())),
        protocol=dict(
            schema_version="multimodal-live/1.19.0", deadline="2026-10-20T01:00:00+00:00"
        ),
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        documents=lambda _: docs,
    )
    current = dict(
        tip=dict(path=str(old), run_id="RUN:SYNTHETIC", previous_attempts=[dict(requests=4)]),
        pending=dict(
            selected_ids=["unsent:text"],
            accepted_ids=[],
            quarantined_batches=[
                dict(
                    assessment_ids=["old:unknown"],
                    request_ids=["unknown-request"],
                    new_reassessment_authorized=False,
                )
            ],
        ),
        attempted=["old:vision"],
        unattempted=["new:a", "new:b"],
        quarantined=["old:unknown"],
        confirmed_observations_checksum=digest({}),
    )
    monkeypatch.setattr(renewal, "completed_parent", lambda *a, **kw: (parent, result))
    monkeypatch.setattr(renewal, "RenewedCampaign", lambda *a: SimpleNamespace(freeze=lambda: None))
    monkeypatch.setattr(recovery, "inventory", lambda *a, **kw: deepcopy(current))
    monkeypatch.setattr(implementation, "freeze", lambda: {"synthetic": digest("implementation")})
    monkeypatch.setattr(unknown_acquisition, "check_network", lambda c: None)
    args = dict(
        previous=old,
        output=tmp_path / "new",
        provider_config=PROFILE,
        authorization="synthetic-no-remote",
        request_limit=5,
        seconds=1200,
    )
    return parent, result, current, docs, args


def test_prepare_exact_route_and_replay(tmp_path, monkeypatch):
    parent, result, current, docs, args = fixture(tmp_path, monkeypatch)
    before = (args["previous"] / "result.json").read_bytes()
    preview = recovery.prepare(**args, check_only=True)
    assert preview["status"] == "ready_not_prepared" and preview["database_writes"] == 0
    assert not args["output"].exists() and docs.writes == 0
    first = recovery.prepare(**args)
    plan = json.loads((args["output"] / "protocol.json").read_bytes())
    assert first["previous_requests"] == 4 and first["api_requests"] == 0
    assert plan["review_target_ids"] == ["new:a", "new:b", "unsent:text"]
    assert plan["prior_deadline"] == parent.protocol["deadline"]
    assert plan["previous_attempts"] == current["tip"]["previous_attempts"]
    assert plan["parent_inventory"]["quarantined"] == ["old:unknown"]
    assert recovery.validate_plan(plan, parent=parent, result=result, current=current)
    second = recovery.prepare(**args)
    assert second["status"] == "already_recorded" and second["deadline"] == first["deadline"]
    assert docs.writes == 1
    assert (args["previous"] / "result.json").read_bytes() == before
    with pytest.raises(ValueError, match="already_claimed"):
        recovery.prepare(**dict(args, output=tmp_path / "fork"))
    assert not (tmp_path / "fork").exists()


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "attempted",
        "invented",
        "duplicate",
        "empty",
        "provider",
        "deadline",
        "budget",
        "history",
        "review",
        "lease",
        "search",
        "mode",
        "facts",
        "result",
        "input",
        "task",
        "version",
        "accepted_reader",
        "unknown_replay",
        "zero_budget",
        "model",
    ],
)
def test_narrow_authority_rejects_changes(tmp_path, monkeypatch, change):
    parent, result, current, docs, args = fixture(tmp_path, monkeypatch)
    recovery.prepare(**args)
    plan = json.loads((args["output"] / "protocol.json").read_bytes())
    if change in {"unknown", "attempted", "invented", "duplicate", "empty"}:
        values = {
            "unknown": ["old:unknown"],
            "attempted": ["old:vision"],
            "invented": ["absent"],
            "duplicate": ["new:a", "new:a"],
            "empty": [],
        }
        plan["acquisition_targets"] = values[change]
        plan["review_target_ids"] = sorted(set(values[change]) | {"unsent:text"})
    elif change == "deadline":
        plan["seconds"] += 1
    elif change == "budget":
        plan["max_cumulative_requests"] = 100
    elif change == "zero_budget":
        plan["max_new_requests"] = 0
    elif change == "history":
        plan["previous_attempts"] = []
    elif change == "review":
        plan["review_target_ids"].append("old:unknown")
    elif change == "lease":
        plan["execution_lease_profile"] = "flock-fallback"
    elif change == "search":
        plan["new_public_search_allowed"] = True
    elif change == "accepted_reader":
        plan["pending_review"] = {}
    elif change == "unknown_replay":
        plan["original_unknown_not_replayed"] = False
    elif change == "model":
        parent.config = parent.config.model_copy(
            update={"max_requests": parent.config.max_requests + 1}
        )
        plan["route_transition"]["origin_checksum"] = digest(parent.config)
    else:
        key = {"result": "original_fact_result", "version": "schema_version"}.get(change, change)
        plan[key] = "changed"
    with pytest.raises(ValueError, match="unscored_"):
        recovery.validate_plan(plan, parent=parent, result=result, current=current)
    assert docs.writes == 1


def test_no_hidden_legacy_optin_and_review_time_scope(tmp_path):
    with pytest.raises(ValueError, match="unscored_protocol_version"):
        renewal.RenewedCampaign(
            SimpleNamespace(),
            tmp_path,
            dict(
                schema_version="multimodal-campaign-renewal/1.23.0",
                unscored_acquisition=recovery.SEMANTICS,
            ),
        )
    campaign = object.__new__(renewal.RenewedCampaign)
    campaign.protocol = dict(schema_version=recovery.VERSION, deadline="2026-10-26T14:00:00+00:00")
    deadline = datetime.fromisoformat(campaign.protocol["deadline"])
    assert campaign.work_deadline() == deadline - timedelta(seconds=600)
    with pytest.raises(RuntimeError, match="synthetic"):
        with campaign.phase_context(None):
            assert campaign.work_deadline() == deadline - timedelta(seconds=60)
            raise RuntimeError("synthetic")
    assert campaign.work_deadline() == deadline - timedelta(seconds=600)


def test_cli_prepares_no_dispatch(monkeypatch):
    from apps import multimodal_case

    seen = {}
    monkeypatch.setattr(recovery, "prepare", lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "unscored-acquisition-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--provider-config",
            PROFILE,
            "--authorization",
            "synthetic",
            "--acquisition-target",
            "new:a",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen["request_limit"] == 24 and seen["acquisition_target"] == ["new:a"]


def test_actual_review_projection_filters_unknown_without_losing_ledger():
    from packages.multimodal_research.campaign import review_observations

    base = SimpleNamespace(review_target_ids=("new:a", "new:b"), ledger=["a", "b", "missing"])
    observations = {"new:a": {"value": "visible"}, "old:unknown": {"value": "native"}}
    assert review_observations(base, observations) == {"new:a": {"value": "visible"}}
    assert base.ledger == ["a", "b", "missing"] and "old:unknown" in observations


def test_dispatch_cap_and_lease_use_existing_owners(monkeypatch):
    from packages.llm_provider import ProviderError
    from packages.multimodal_research import execution_lease
    from packages.multimodal_research.campaign import MultimodalCampaign

    c = object.__new__(renewal.RenewedCampaign)
    c.protocol = dict(
        schema_version=recovery.VERSION,
        max_model_dispatches=4,
        execution_lease_profile=execution_lease.PROFILE,
    )
    calls, kwargs_seen = [], {}
    budget = SimpleNamespace(before_reserve=lambda value: calls.append(value))
    monkeypatch.setattr(
        MultimodalCampaign, "budget_factory", lambda self, **kw: kwargs_seen.update(kw) or budget
    )
    monkeypatch.setattr(execution_lease, "session", lambda owner: ("runtime-session", owner))
    c.usage = lambda: 3
    c.budget_factory().before_reserve("next")
    assert calls == ["next"] and kwargs_seen["bounded_phase"] is True
    c.usage = lambda: 4
    with pytest.raises(ProviderError):
        budget.before_reserve("must-not-dispatch")
    assert calls == ["next"]
    assert c.execution_session() == ("runtime-session", c)
    c.protocol["execution_lease_profile"] = "fallback"
    with pytest.raises(ValueError, match="execution_lease_profile"):
        c.execution_session()


def test_inventory_checks_terminal_owner_and_quarantine(tmp_path, monkeypatch):
    from packages.multimodal_research import pending_review

    controls = {"RUN:OLD": SimpleNamespace(cancelled=False, pending=None, phase="stopped")}
    parent = SimpleNamespace(
        protocol=dict(schema_version="multimodal-live-campaign/1.19.0"),
        sessions=SimpleNamespace(read=lambda run: controls[run]),
        accepted_observations=lambda: {},
    )
    result = dict(
        complete=False,
        phases=[dict(target=None, run_id="RUN:OLD", accepted=False, acceptance_key=None)],
    )
    pending = dict(
        selected_ids=[],
        accepted_ids=[],
        quarantined_batches=[
            dict(
                assessment_ids=["old:unknown"], request_ids=["r"], new_reassessment_authorized=False
            )
        ],
    )
    monkeypatch.setattr(
        renewal, "stopped_tip", lambda *a: dict(previous_attempts=[dict(requests=4)])
    )
    monkeypatch.setattr(pending_review, "_inventory", lambda *a: pending)
    monkeypatch.setattr(renewal, "attempted_acquisition_targets", lambda *a: {"old:vision"})
    calls = []

    def select(*a, **kw):
        calls.append(kw)
        return ["new:a"]

    monkeypatch.setattr(renewal, "unattempted_targets", select)
    value = recovery.inventory(parent, result, tmp_path)
    assert value["quarantined"] == ["old:unknown"] and value["attempted"] == ["old:vision"]
    assert calls == [dict(full_inventory=True, owner_methods=True)]
    controls["RUN:OLD"].cancelled = True
    with pytest.raises(ValueError, match="parent_control_changed"):
        recovery.inventory(parent, result, tmp_path)
    controls["RUN:OLD"].cancelled = False
    pending["accepted_ids"] = ["already-scored"]
    with pytest.raises(ValueError, match="use_accepted_recovery"):
        recovery.inventory(parent, result, tmp_path)
    parent.protocol["schema_version"] = recovery.VERSION
    with pytest.raises(ValueError, match="initial_stopped_parent"):
        recovery.inventory(parent, result, tmp_path)


@pytest.mark.parametrize("definition_increment", [False, True])
def test_progress_without_prior_score_is_not_fake_history(tmp_path, definition_increment):
    from packages.multimodal_research.report import export_acquisition_progress
    from tests.agent_provider.test_accepted_review_reuse import accepted_fixture

    f = accepted_fixture(new_observation=True)
    docs = Documents()
    phase = dict(run_id="RUN:PROGRESS", target="new:a", accepted=True)
    docs.write("dynamic-phase-result", (phase,))
    c = SimpleNamespace(
        output=tmp_path,
        campaign_id="synthetic-unscored",
        protocol=dict(schema_version=recovery.VERSION, reuse_assessment_ids=[]),
        documents=lambda _: docs,
        usage=lambda: 1,
        accepted_observations=lambda: {},
        base=SimpleNamespace(
            confirmed_review_reader=None,
            observations={
                p.id: next(
                    o
                    for o in f.owner.state.material.observations
                    if o.reference.code == p.observation_code
                )
                for p in f.owner.state.policy.items
            },
            ledger=[
                dict(id=r.id, definition=r.definition) for r in f.owner.state.policy.requirements
            ],
        ),
    )
    if definition_increment:
        from packages.multimodal_research.definition_alternative import SEMANTICS, VERSION

        revision = {"accepted": {"definition": "source-bound"}, "replaced": [{"id": "new:a"}]}
        c.protocol.update(
            schema_version=VERSION,
            definition_alternative=SEMANTICS,
            definition_increment_only=True,
            source_revision_checksum=digest(revision),
        )
        c.base.definition_current = lambda: revision
    report = export_acquisition_progress(c, phase)
    snapshot = json.loads(Path(report["json"]).read_bytes())
    assert snapshot["schema_version"] == (
        "multimodal-acquisition-progress/1.3.0"
        if definition_increment
        else "multimodal-acquisition-progress/1.2.0"
    )
    if definition_increment:
        assert snapshot["definition_revision_checksum"] == digest(revision)
    assert (
        snapshot["historical_assessed"] == 0 and snapshot["historical_acceptance_checksum"] is None
    )
    assert snapshot["new_accepted_score"] is None and snapshot["observations"] == 6
    assert all(r["historical_review"] is None for r in snapshot["rows"])
    before = docs.writes
    assert export_acquisition_progress(c, phase) == report and docs.writes == before


@pytest.mark.parametrize("definition_increment", [False, True])
def test_progress_failure_requires_exact_confirmed_original_observation(
    tmp_path, definition_increment
):
    from packages.multimodal_research.stage_failure import KEY

    phase = dict(
        run_id="RUN:PROGRESS",
        target="new:a",
        accepted=True,
        error=None,
        requests=dict(api_requests_unknown=0),
        transport=dict(remote_response_confirmed=1),
    )
    protocol = dict(schema_version=recovery.VERSION, acquisition_targets=["new:a", "new:b"])
    if definition_increment:
        from packages.multimodal_research.definition_alternative import SEMANTICS, VERSION

        revision = {"accepted": {"definition": "source-bound"}}
        protocol.update(
            schema_version=VERSION,
            definition_alternative=SEMANTICS,
            definition_increment_only=True,
            source_revision_checksum=digest(revision),
        )
    failure = dict(
        error_code="local_owner_error",
        failure_stage="acquisition",
        protocol_checksum=digest(protocol),
        score_status="not_scored",
        confirmed_phases=[phase],
    )
    (tmp_path / "stage-failure.json").write_text(json.dumps(failure))
    docs = Documents()
    docs.write(KEY, (failure,))
    docs.write("dynamic-phase-result", (phase,))
    control = SimpleNamespace(phase="stopped", cancelled=False, pending=None)
    c = SimpleNamespace(
        output=tmp_path,
        protocol=protocol,
        documents=lambda _: docs,
        sessions=SimpleNamespace(read=lambda _: control),
        accepted_observations=lambda: {"new:a": "confirmed"},
    )
    if definition_increment:
        c.base = SimpleNamespace(definition_current=lambda: revision)
        c.contexts = SimpleNamespace(read_research_run=lambda *a, **kw: None)
        c.request_key = lambda target: "unused-review"
    assert recovery.progress_failure(c)[:2] == (failure, phase)
    if definition_increment:
        assert recovery.progress_repair_version(c) == "unscored-progress-repair/1.1.0"
        c.contexts.read_research_run = lambda *a, **kw: object()
        with pytest.raises(ValueError, match="review_already_started"):
            recovery.progress_failure(c, unstarted_review=True)
        # A recorded amendment remains readable after its authorized Review starts.
        assert recovery.progress_failure(c)[:2] == (failure, phase)
        c.contexts.read_research_run = lambda *a, **kw: None
        c.base.definition_current = lambda: {"changed": True}
        with pytest.raises(ValueError, match="definition_changed"):
            recovery.progress_failure(c)
        c.base.definition_current = lambda: revision
    control.cancelled = True
    with pytest.raises(ValueError, match="original_owner_changed"):
        recovery.progress_failure(c)
    control.cancelled = False
    c.accepted_observations = lambda: {}
    with pytest.raises(ValueError, match="original_owner_changed"):
        recovery.progress_failure(c)

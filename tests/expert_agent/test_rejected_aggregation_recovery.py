"""Explicit local successors retain rejected receipts and original judgments."""

from types import SimpleNamespace

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    GRANT_KEY,
    AggregationPermission,
    check_recovery_content,
    rejected_aggregation,
)
from packages.grounding.models import digest, validate
from packages.multimodal_research.accepted_aggregation import (
    AggregationPlan,
    check_saved_aggregation,
    claim_recovery,
    recovery_origin,
)
from tests.expert_agent.test_accepted_aggregation import accepted_origin, aggregate_fixture


def rejected(monkeypatch, tmp_path):
    f = accepted_origin(monkeypatch, tmp_path)
    a = aggregate_fixture(f)
    permission, sub = AggregationPermission.prepare(
        owner=a.owner, documents=a.documents, requests=a.requests, config=f.config
    )
    service = ResearchAcceptance(
        owner=a.owner, documents=a.documents, requests=a.requests, live_permission=permission
    )
    assert service.submit(sub, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    permission.clock = lambda: permission.grant.deadline
    assert service.accept(key=ACCEPTANCE_KEY).status == "rejected"
    plan = validate(AggregationPlan, dict(facts="facts", origins=[dict(campaign="origin")]))
    original = {"requests": 3}
    manifest = dict(
        schema_version="accepted-review-aggregation-manifest/1.0.0",
        plan=plan.model_dump(mode="json"),
        original_result_checksum=digest(original),
        run_id=a.documents.run_id,
        state_checksum=digest(a.state),
        origins_checksum=digest(sub.origins),
        submission_checksum=digest(sub),
    )
    a.documents.write("dynamic-phase-input", (a.state,))
    a.documents.write("accepted-aggregation-manifest", (manifest,))
    return f, a, sub, plan, original, manifest


def proof(a, manifest, plan, original):
    return rejected_aggregation(
        a.documents,
        manifest=manifest,
        plan=plan.model_dump(mode="json"),
        original_result_checksum=digest(original),
    )


def successor(f):
    b = aggregate_fixture(f)
    raw = b.state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:AGGREGATED_SUCCESSOR"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    b.state = validate(DomainState, raw)
    b.owner.current = lambda: b.state
    b.documents.run_id = b.state.material.scope.run_id
    control = b.documents.sessions.read(b.documents.run_id)
    control = control.model_copy(
        update={"binding": control.binding.model_copy(update={"run_id": b.documents.run_id})}
    )
    b.documents.sessions.read = lambda _: control
    return b


def test_rejected_owner_successor_accepts_without_reasking_and_preserves_old(monkeypatch, tmp_path):
    f, a, old_sub, plan, original, manifest = rejected(monkeypatch, tmp_path)
    before = dict(a.documents.saved)
    old, state, sub = proof(a, manifest, plan, original)
    b = successor(f)
    permission, new_sub = AggregationPermission.prepare(
        owner=b.owner, documents=b.documents, requests=b.requests, config=f.config
    )
    check_recovery_content(state, sub, b.state, new_sub)
    claim_recovery(a.documents, tmp_path / "new", plan, old)
    audit = list(a.documents.audit)
    claim_recovery(a.documents, tmp_path / "new", plan, old)
    assert a.documents.audit == audit
    with pytest.raises(ValueError, match="successor_conflict"):
        claim_recovery(a.documents, tmp_path / "different", plan, old)
    service = ResearchAcceptance(
        owner=b.owner, documents=b.documents, requests=b.requests, live_permission=permission
    )
    assert service.submit(new_sub, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    assert service.accept(key=ACCEPTANCE_KEY).status == "accepted"
    check_saved_aggregation(service.read(key=ACCEPTANCE_KEY), new_sub, b.documents)
    assert all(a.documents.saved[k] == v for k, v in before.items())
    assert len(f.wire.calls) == 1 and not b.requests.journal.read()
    assert old_sub.candidate.items == new_sub.candidate.items
    with pytest.raises(ValueError, match="previous_rejection_requires_explicit_successor"):
        check_saved_aggregation(a.documents.read("accepted:" + ACCEPTANCE_KEY), sub, a.documents)


@pytest.mark.parametrize(
    "fault", ["manifest", "parent", "grant", "receipt", "submission", "unknown", "cancel", "result"]
)
def test_recovery_refuses_wrong_or_unconfirmed_original(monkeypatch, tmp_path, fault):
    _, a, _, plan, original, manifest = rejected(monkeypatch, tmp_path)
    if fault == "manifest":
        manifest = dict(manifest, run_id="RUN:FOREIGN")
    elif fault == "parent":
        a.documents.saved["dynamic-phase-input"][0]["config_checksum"] = digest("changed")
    elif fault == "grant":
        a.documents.saved[GRANT_KEY][0]["control_checksum"] = digest("changed")
    elif fault == "receipt":
        a.documents.saved["accepted:" + ACCEPTANCE_KEY][0]["submission_checksum"] = digest(
            "changed"
        )
    elif fault == "submission":
        a.documents.saved["submission:" + ACCEPTANCE_KEY][0]["status"] = "rejected"
    elif fault == "unknown":
        monkeypatch.setattr(
            "packages.research_runtime.agent_records.RuntimeAgentJournal.read",
            lambda _: ({"phase": "reserved", "request_id": "unknown"},),
        )
    elif fault == "cancel":
        control = a.documents.sessions.read(a.documents.run_id)
        a.documents.sessions.read = lambda _: control.model_copy(update={"cancelled": True})
    else:
        a.documents.saved["multimodal-case-result"] = ({"complete": True},)
    with pytest.raises(ValueError, match="accepted_aggregation_recovery"):
        proof(a, manifest, plan, original)


@pytest.mark.parametrize("fault", ["same_run", "input", "candidate", "origin"])
def test_current_successor_must_retain_exact_observations_and_judgments(
    monkeypatch, tmp_path, fault
):
    f, a, old_sub, _, _, _ = rejected(monkeypatch, tmp_path)
    b = successor(f)
    new_sub, _ = b.source.current(b.state)
    state = b.state
    if fault == "same_run":
        state = a.state
    elif fault == "input":
        state = state.model_copy(update={"config_checksum": digest("changed")})
    elif fault == "candidate":
        new_sub = new_sub.model_copy(
            update={"candidate": new_sub.candidate.model_copy(update={"items": ()})}
        )
    else:
        new_sub = new_sub.model_copy(update={"origins": ()})
    with pytest.raises(ValueError, match="recovery_current_content_changed"):
        check_recovery_content(a.state, old_sub, state, new_sub)


def test_recovery_locator_requires_original_plan_and_bounded_history(monkeypatch, tmp_path):
    import json

    f, a, _, plan, original, manifest = rejected(monkeypatch, tmp_path)
    directory = tmp_path / "failed"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "plan.json").write_text(plan.model_dump_json())
    campaign = SimpleNamespace(documents=lambda run: a.documents)
    assert recovery_origin(campaign, directory, plan, original)[1]["attempt"] == 1
    manifest.update(
        schema_version="accepted-review-aggregation-manifest/1.1.0", recovery={"attempt": 2}
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    a.documents.saved["accepted-aggregation-manifest"] = (manifest,)
    with pytest.raises(ValueError, match="recovery_attempt_limit"):
        recovery_origin(campaign, directory, plan, original)


@pytest.mark.parametrize("retry", [False, True])
def test_original_cli_passes_explicit_recovery_without_loading_provider(monkeypatch, retry):
    import sys

    from apps import multimodal_case
    from packages.multimodal_research import accepted_aggregation

    calls = []

    def original_entry(plan, output, *, progress, retry_from):
        calls.append((plan, output, retry_from))
        return dict(new_model_requests=0)

    monkeypatch.setattr(accepted_aggregation, "aggregate_entry", original_entry)
    args = ["multimodal_case", "aggregate-accepted", "--plan", "plan.json", "--output", "new"]
    if retry:
        args += ["--retry-from", "old-rejected"]
    monkeypatch.setattr(sys, "argv", args)
    assert multimodal_case.main() == 0
    assert calls == [("plan.json", "new", "old-rejected" if retry else None)]


def test_retry_cannot_overlap_previous_output_before_any_owner_or_write(monkeypatch, tmp_path):
    from packages.multimodal_research import accepted_aggregation as module

    plan = validate(AggregationPlan, dict(facts="facts", origins=[dict(campaign="origin")]))
    path = tmp_path / "plan.json"
    path.write_text(plan.model_dump_json())

    def forbidden(*args):
        raise AssertionError("overlap must fail before any owner registration")

    monkeypatch.setattr(module, "plan_campaign", forbidden)
    with pytest.raises(ValueError, match="output_overlaps_history"):
        module.aggregate_entry(path, tmp_path / "old", retry_from=tmp_path / "old")
    assert not (tmp_path / "old").exists()

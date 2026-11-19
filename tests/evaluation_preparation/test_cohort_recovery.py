"""Recovery audit distinguishes accepted history from current scoring and grants."""

import copy
import json

import pytest

from packages.evaluation_preparation.cohort_recovery import ancestral_result, retained_reviews
from packages.grounding.models import digest


def test_retained_null_requires_exact_observation_and_original_reason():
    observation = dict(status="obtained", reference={"code": "OBS:ONE"}, value=1)
    row = dict(id="one", observation=observation, sources=["SOURCE:ONE"], source_requirement={})
    old = dict(
        requirements=[row],
        scores=dict(
            reviews=[dict(id="one", judgment=dict(value=None))],
            direction_assessments={
                "one": dict(unassessable_reason="method-to-target relation missing")
            },
        ),
    )
    current = dict(
        requirements=[dict(row, observation=None, unscored_observation=observation)],
        completion=dict(retained_context_not_newly_reviewed_ids=["one"]),
    )
    proof = dict(run_id="RUN:OLD", acceptance_key="accepted:old", report_checksum=digest(old))
    result, missing = retained_reviews(current, [(old, proof)])
    assert not missing and result["one"]["origin"] == proof
    assert result["one"]["contributes_new_vote"] is False
    assert result["one"]["source_definition_unchanged"]
    assert result["one"]["interpretation"] == "historical_unassessable_not_new_current_review"
    changed = copy.deepcopy(current)
    changed["requirements"][0]["sources"] = ["SOURCE:NEW"]
    assert (
        retained_reviews(changed, [(old, proof)])[0]["one"]["source_definition_unchanged"] is False
    )
    changed["requirements"][0]["unscored_observation"]["value"] = 2
    assert retained_reviews(changed, [(old, proof)]) == ({}, ["one"])
    old["scores"]["direction_assessments"] = {}
    assert retained_reviews(current, [(old, proof)]) == ({}, ["one"])


def test_historical_nonzero_cannot_be_relabelled_unassessable():
    old = dict(
        requirements=[dict(id="one")],
        scores=dict(reviews=[dict(id="one", judgment=dict(value=0.1))]),
    )
    current = dict(
        requirements=[dict(id="one")],
        completion=dict(retained_context_not_newly_reviewed_ids=["one"]),
    )
    assert retained_reviews(current, [(old, {})]) == ({}, ["one"])


@pytest.mark.parametrize("layout", ["result.json", "evidence/result.json", "stage-failure.json"])
def test_explicit_original_layout_checksum_and_root(tmp_path, layout):
    directory = tmp_path / "case"
    path = directory / layout
    path.parent.mkdir(parents=True)
    value = dict(case_code="CASE:ONE", complete=False)
    path.write_text(json.dumps(value))
    entry = dict(directory=str(directory), requests=1, checksum=digest(value))
    if layout == "stage-failure.json":
        entry["kind"] = "routed_source_failure_deferred"
    assert ancestral_result(entry, tmp_path) == (path, value)
    with pytest.raises(ValueError, match="ancestry_binding"):
        ancestral_result(dict(entry, checksum=digest("other")), tmp_path)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="history_scope"):
        ancestral_result(entry, elsewhere)


def test_ambiguous_result_or_symlink_rejected(tmp_path):
    d = tmp_path / "case"
    (d / "evidence").mkdir(parents=True)
    value = dict(complete=False)
    for name in ["result.json", "evidence/result.json"]:
        (d / name).write_text(json.dumps(value))
    entry = dict(directory=str(d), requests=0, checksum=digest(value))
    with pytest.raises(ValueError, match="ancestry_layout"):
        ancestral_result(entry, tmp_path)


def test_cli_is_readonly_audit_not_new_dispatch(monkeypatch, capsys):
    from apps.case_cohort import main

    seen = []
    monkeypatch.setattr(
        "packages.evaluation_preparation.cohort_recovery.audit_case",
        lambda *a: seen.append(a) or dict(new_api_requests=0, expansion_authorized=False),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "case_cohort",
            "audit-recovery",
            "--input",
            "old",
            "--report",
            "r.json",
            "--history-root",
            "history",
        ],
    )
    assert main() == 0 and seen == [("old", "r.json", "history")]
    assert json.loads(capsys.readouterr().out)["expansion_authorized"] is False


def test_cli_explicit_local_closure_keeps_original_input(monkeypatch, capsys):
    from apps.case_cohort import main

    seen = []
    monkeypatch.setattr(
        "packages.evaluation_preparation.cohort_recovery.audit_case",
        lambda *a, **kw: seen.append((a, kw)) or dict(new_api_requests=0),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "case_cohort",
            "audit-recovery",
            "--input",
            "old",
            "--report",
            "r.json",
            "--history-root",
            "history",
            "--local-closure",
            "closed",
        ],
    )
    assert main() == 0
    assert seen == [(("old", "r.json", "history"), {"local_closure": "closed"})]
    assert json.loads(capsys.readouterr().out)["new_api_requests"] == 0


def test_history_root_cannot_be_escaped_with_parent_segments_or_symlinks(tmp_path):
    from packages.evaluation_preparation.cohort_recovery import read_file

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="history_scope"):
        read_file(allowed / ".." / "outside.json", root=allowed)
    link = allowed / "alias.json"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        read_file(link, root=allowed)


@pytest.mark.parametrize(
    "fault", [None, "pending", "active", "duplicate", "incomplete", "simulated", "false_acceptance"]
)
def test_phase_unknown_is_preserved_but_pending_or_incomplete_owner_refused(monkeypatch, fault):
    from types import SimpleNamespace

    from packages.evaluation_preparation.cohort_recovery import phase_owner

    phase = dict(run_id="RUN:ONE", target="one", accepted=False, acceptance_key=None)
    state = SimpleNamespace(phase="stopped", pending=None)
    record = dict(
        request_id="request:one", simulated=False, role="vision", error="transport", api_requests=1
    )
    events = [
        dict(phase="reserved", request_id="request:one"),
        dict(phase="completed", record=record),
    ]
    if fault == "pending":
        state.pending = {"unresolved": True}
    elif fault == "active":
        state.phase = "working"
    elif fault == "duplicate":
        events.append(events[0])
    elif fault == "incomplete":
        events.pop()
    elif fault == "simulated":
        record["simulated"] = True
    elif fault == "false_acceptance":
        phase["acceptance_key"] = "not-accepted"
    docs = SimpleNamespace(
        run_id=phase["run_id"], read=lambda key: (phase,) if key == "dynamic-phase-result" else None
    )
    campaign = SimpleNamespace(
        documents=lambda _: docs, sessions=SimpleNamespace(read=lambda _: state)
    )
    monkeypatch.setattr(
        "packages.evaluation_preparation.cohort_recovery.RuntimeAgentJournal",
        lambda _: SimpleNamespace(read=lambda: events),
    )
    if fault:
        with pytest.raises(ValueError, match="recovery_audit"):
            phase_owner(campaign, phase)
    else:
        _, accepted, requests = phase_owner(campaign, phase)
        assert accepted is None
        assert requests[0]["response_confirmed"] is None
        assert requests[0]["replay_authorized"] is False


@pytest.mark.parametrize(
    "fault",
    [None, "original", "result", "budget", "history", "owner", "review", "html"],
)
def test_local_closure_audit_requires_owner_and_exact_local_mirrors(tmp_path, monkeypatch, fault):
    """Boundary unit check; real owner/DB traversal is separately run via CLI."""
    from types import SimpleNamespace

    from packages.evaluation_preparation import cohort_recovery as audit
    from packages.evidence_details.recorded_finalization import REPORT_KEY, RESULT_KEY

    original = dict(phases=[dict(run_id="RUN:OLD")], new_requests=2, previous_attempts=[])
    result = dict(
        original_requests=2,
        new_requests=0,
        previous_attempts=[],
        original_result_checksum=digest(original),
    )
    accepted = dict(scores=dict(reviews=[]))
    report = dict(result=result, scores=accepted["scores"], original_review=accepted)
    proof = dict(closure_checksum=digest(result), confirmation=dict(acceptance_key="owner-key"))
    saved = {
        RESULT_KEY: (copy.deepcopy(result),),
        "dynamic-phase-input": ({},),
        REPORT_KEY: ("saved",),
    }
    closure = tmp_path / "closed"
    (closure / "review").mkdir(parents=True)
    report_path = tmp_path / "r.json"
    html_path = report_path.with_suffix(".html")
    html_path.write_text("exact saved html")
    old = copy.deepcopy(original)
    local_review = copy.deepcopy(accepted)
    if fault == "original":
        old["new_requests"] = 1
    elif fault == "result":
        result["injected"] = True
    elif fault == "budget":
        result["original_requests"] = 0
    elif fault == "history":
        result["previous_attempts"] = [{"requests": 0}]
    elif fault == "review":
        local_review["scores"] = {"invented": True}
    elif fault == "html":
        html_path.write_text("changed")
    for path, value in [
        (closure / "result.json", result),
        (closure / "original-failure.json", old),
        (closure / "review/review-report.json", local_review),
    ]:
        path.write_text(json.dumps(value))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    docs = SimpleNamespace(read=lambda key: saved[key])
    campaign = SimpleNamespace(documents=lambda rid: docs)

    def owner(parent, rid):
        assert parent is campaign and rid == "RUN:OLD"
        if fault == "owner":
            raise ValueError("original_owner_source_changed")
        return proof

    monkeypatch.setattr(
        "packages.multimodal_research.campaign_renewal.recorded_acquisition_origin", owner
    )
    monkeypatch.setattr(audit, "validate", lambda *a: "state")
    monkeypatch.setattr(audit, "unpack", lambda *a, **kw: (report, {"html": "exact saved html"}))

    def read_accepted(obj, *, key, parent):
        assert key == "owner-key" and parent == "state" and obj.documents is docs
        return accepted

    monkeypatch.setattr(audit, "read_research_report", read_accepted)
    if fault:
        with pytest.raises(ValueError, match="recovery_audit_local_|original_owner_source_changed"):
            audit.recorded_closure(campaign, original, closure, report, report_path, tmp_path)
    else:
        assert audit.recorded_closure(
            campaign, original, closure, report, report_path, tmp_path
        ) == (result, {"html": "exact saved html"}, proof)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

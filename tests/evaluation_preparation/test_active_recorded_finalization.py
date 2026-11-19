"""Active CLI recovery delegates only to the original read-only reconstruction."""

import json
from types import SimpleNamespace

import pytest

from packages.multimodal_research import entry, finalization, report_recovery


@pytest.fixture
def saved(tmp_path, monkeypatch):
    previous = tmp_path / "original"
    previous.mkdir()
    failure = {"complete": False, "phases": [{"error": "deadline"}]}
    spec = {"active_acquisition": {"path": "frozen-profile", "checksum": "exact"}}
    (previous / "result.json").write_text(json.dumps(failure))
    (previous / "entry.json").write_text(json.dumps(spec))
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("normal entry/registration/dispatch must not run during recovery")

    monkeypatch.setattr(entry, "owners", forbidden)
    monkeypatch.setattr(entry, "assemble_entry", forbidden)
    campaign = SimpleNamespace(base="original-current-owner")
    monkeypatch.setattr(report_recovery, "saved_campaign", lambda path: (campaign, failure))

    def finish(current, old, output, module, **flags):
        calls.append((current, old, output, module, flags))
        return {"scope": "delegation-only-not-real-acceptance"}

    monkeypatch.setattr(finalization, "_finish_entry", finish)
    return SimpleNamespace(
        previous=previous,
        output=tmp_path / "recovered",
        failure=failure,
        campaign=campaign,
        calls=calls,
    )


def test_active_complete_uses_original_saved_owner_and_shared_finalization(saved):
    result = finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert result["scope"] == "delegation-only-not-real-acceptance"
    current, old, output, module, flags = saved.calls[0]
    assert current is saved.campaign and old == saved.failure and output == saved.output
    assert module.__name__.endswith("recorded_finalization")
    assert flags == {"complete": True, "active": True}
    assert json.loads((saved.previous / "result.json").read_text()) == saved.failure


@pytest.mark.parametrize("version", ["1.48.0", "1.51.0"])
def test_configuration_successor_keeps_zero_wire_export_entry(saved, monkeypatch, version):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    plan = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="acquisition",
        review_partition="owner_bound_reused_and_new/1.0.0",
    )
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda _: (saved.campaign, saved.failure)
    )
    finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert saved.calls[0][0] is saved.campaign
    assert saved.calls[0][-1] == {"complete": True, "active": True}
    assert json.loads((saved.previous / "protocol.json").read_text()) == plan


def test_deadline_partial_renewal_restores_owner_without_dispatch(saved, monkeypatch):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    plan = dict(schema_version="multimodal-campaign-renewal/1.27.0", mode="review")
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda _: (saved.campaign, saved.failure)
    )
    result = finalization.finalize_entry(saved.previous, saved.output, complete=False)
    assert result["scope"] == "delegation-only-not-real-acceptance"
    assert saved.calls[0][3].__name__.endswith("partial_finalization")
    assert saved.calls[0][4] == {"complete": False, "active": True}
    assert json.loads((saved.previous / "protocol.json").read_text()) == plan


def test_citation_partial_root_renewal_requires_explicit_policy(saved, monkeypatch):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    saved.failure["phases"][-1]["error"] = "candidate_failed_or_context_changed"
    (saved.previous / "result.json").write_text(json.dumps(saved.failure))
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.29.0",
        mode="review",
        root_direction_review={"binding": "owner-must-revalidate"},
    )
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda _: (saved.campaign, saved.failure)
    )
    with pytest.raises(ValueError, match="renewal_version"):
        finalization.finalize_entry(saved.previous, saved.output)
    finalization.finalize_entry(saved.previous, saved.output, citation_rejection=True)
    assert saved.calls[0][-1] == dict(complete=False, active=True, citation_rejection=True)
    assert json.loads((saved.previous / "protocol.json").read_text()) == plan
    plan["mode"] = "acquisition"
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="citation_partial_original_version"):
        finalization.finalize_entry(saved.previous, saved.output, citation_rejection=True)


@pytest.mark.parametrize("version", ["1.15.0", "1.16.0"])
def test_pending_explanation_requires_explicit_local_policy(saved, monkeypatch, version):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    plan = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="review",
        pending_review={"schema_version": "pending-review-inventory/1.0.0"},
        review_increment_only=True,
    )
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda _: (saved.campaign, saved.failure)
    )
    with pytest.raises(ValueError, match="renewal_version"):
        finalization.finalize_entry(saved.previous, saved.output, complete=True)
    finalization.finalize_entry(
        saved.previous, saved.output, complete=True, lossless_explanations=True
    )
    assert saved.calls[0][3].__name__.endswith("recorded_explanation")
    assert json.loads((saved.previous / "protocol.json").read_text()) == plan


def test_current_original_failure_mismatch_stops_before_acceptance(saved, monkeypatch):
    monkeypatch.setattr(report_recovery, "saved_campaign", lambda path: (saved.campaign, {}))
    with pytest.raises(ValueError, match="original_result_changed"):
        finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert not saved.calls


def test_output_conflict_stops_before_reconstruction(saved, monkeypatch):
    saved.output.mkdir()
    monkeypatch.setattr(
        report_recovery, "saved_campaign", lambda path: pytest.fail("must not read")
    )
    with pytest.raises(ValueError, match="output_conflict"):
        finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert not saved.calls


def test_original_output_or_symlink_is_rejected(saved):
    with pytest.raises(ValueError, match="finalization_path"):
        finalization.finalize_entry(saved.previous, saved.previous, complete=True)
    saved.output.symlink_to(saved.previous, target_is_directory=True)
    with pytest.raises(ValueError, match="finalization_path"):
        finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert not saved.calls


@pytest.mark.parametrize("version", ["1.13.0", "1.14.0"])
def test_definition_review_uses_original_restore_without_dispatch(saved, monkeypatch, version):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    plan = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="review",
        review_definition_increment=True,
        definition_increment_only=True,
    )
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda path: (saved.campaign, saved.failure)
    )
    result = finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert result["scope"] == "delegation-only-not-real-acceptance"
    assert saved.calls[0][0] is saved.campaign
    assert saved.calls[0][-1] == {"complete": True, "active": True}
    assert json.loads((saved.previous / "protocol.json").read_text()) == plan


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "multimodal-campaign-renewal/1.15.0"},
        {"mode": "acquisition"},
        {"review_definition_increment": False},
        {"definition_increment_only": False},
        {"review_definition_increment": "true"},
    ],
)
def test_unbound_renewal_cannot_enter_recorded_closure(saved, monkeypatch, change):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.14.0",
        mode="review",
        review_definition_increment=True,
        definition_increment_only=True,
    )
    plan.update(change)
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal,
        "completed_parent",
        lambda path: pytest.fail("invalid protocol must not reach restoration"),
    )
    with pytest.raises(ValueError, match="recorded_finalization_renewal_version"):
        finalization.finalize_entry(saved.previous, saved.output, complete=True)
    assert not saved.calls


@pytest.mark.parametrize("fail_export", [False, True])
@pytest.mark.parametrize(
    "require_committed,commit_present", [(False, True), (True, True), (True, False)]
)
def test_existing_commit_is_exported_without_permission_or_second_acceptance(
    tmp_path, monkeypatch, capsys, fail_export, require_committed, commit_present
):
    """Assembly regression only; original-owner proof is verified separately."""
    from contextlib import nullcontext

    from agents.expert_agent import committed_review
    from packages.grounding.models import digest
    from packages.multimodal_research import campaign as campaign_module
    from packages.multimodal_research import memory, report

    old_phase = dict(run_id="RUN:ONE", target=None, accepted=False, error="deadline")
    old = dict(phases=[old_phase], complete=False, new_requests=3)
    parent, task, control, journal = {"fixture": "parent"}, "task", "stopped", ["unchanged"]
    proof = dict(
        schema_version="committed-review-readback/1.0.0",
        run_id="RUN:ONE",
        event=dict(
            role="review_control", code="review:one", version="1.0.0", checksum=digest("event")
        ),
        acceptance_key="original:domain",
        parent_checksum=digest(parent),
        accepted_checksum=digest("accepted"),
        report_checksum=digest("report"),
        intent_checksum=digest("intent"),
        journal_checksum=digest(journal),
    )
    writes, checks = {}, []

    class Docs:
        run_id = "RUN:ONE"

        def read(self, key):
            if key == "dynamic-phase-input":
                return (parent,)
            if key == "accepted:original:domain":
                return ("original-accepted",)
            if key == "new:report":
                return None
            raise AssertionError("unexpected read " + key)

        def session(self):
            return nullcontext()

        def write(self, key, value):
            assert not key.startswith("accepted:")
            writes[key] = value

    docs = Docs()
    base = SimpleNamespace(task=task, contexts={})
    c = SimpleNamespace(
        base=base,
        output=tmp_path / "original",
        config="config",
        documents=lambda rid: docs,
        accepted_observations=lambda: {},
        contexts=SimpleNamespace(
            read_research_run=lambda rid: SimpleNamespace(request=SimpleNamespace(task=task))
        ),
        sessions=SimpleNamespace(read=lambda rid: control),
    )

    class Phase:
        def __init__(self, *args, **kwargs):
            pass

        def project(self, **kwargs):
            return parent, {}

        def verify(self, state):
            assert state == parent
            checks.append("current")

    def forbidden(*args, **kwargs):
        pytest.fail("a confirmed commit must not be reaccepted or given a new grant")

    module = SimpleNamespace(
        ACCEPTANCE_KEY="new:domain",
        GRANT_KEY="new:grant",
        REPORT_KEY="new:report",
        RESULT_KEY="new:result",
        original_failure=lambda d: old,
        RecordedBatchRequests=lambda j: "readonly-requests",
        RecordedReviewPermission=SimpleNamespace(prepare=forbidden),
    )
    monkeypatch.setattr(campaign_module, "Phase", Phase)
    monkeypatch.setattr(finalization, "validate", lambda cls, value: value)
    monkeypatch.setattr(
        finalization, "RuntimeAgentJournal", lambda d: SimpleNamespace(read=lambda: journal)
    )
    monkeypatch.setattr(
        committed_review,
        "read_committed_review",
        lambda *a, **kw: proof if commit_present else None,
    )
    monkeypatch.setattr(
        finalization,
        "ResearchAcceptance",
        lambda **kw: SimpleNamespace(submit=forbidden, accept=forbidden),
    )
    monkeypatch.setattr(
        finalization, "read_research_report", lambda a, **kw: {"acceptance_key": kw["key"]}
    )

    def export(current, result):
        assert current.output == tmp_path / "original"
        assert current.report_name == "recovered"
        assert result["phases"][-1]["acceptance_key"] == "original:domain"
        if fail_export:
            raise OSError("fixture_export_failure")
        return {"complete": True}

    monkeypatch.setattr(report, "export", export)
    monkeypatch.setattr(memory, "archive", lambda *a: {"status": "fixture-only"})
    if not commit_present:
        with pytest.raises(ValueError, match="recorded_increment_requires_original_commit"):
            finalization._finish_entry(
                c,
                old,
                tmp_path / "recovered",
                module,
                complete=True,
                active=True,
                require_committed=True,
            )
        assert not writes
        return
    if fail_export:
        with pytest.raises(OSError, match="fixture_export_failure"):
            finalization._finish_entry(
                c,
                old,
                tmp_path / "recovered",
                module,
                complete=True,
                active=True,
                require_committed=require_committed,
            )
    else:
        value = finalization._finish_entry(
            c,
            old,
            tmp_path / "recovered",
            module,
            complete=True,
            active=True,
            require_committed=require_committed,
        )
        assert value["requests_unchanged"] and value["control_unchanged"]
    captured = capsys.readouterr()
    assert captured.out == ""  # The CLI's single final JSON remains machine readable.
    progress = [json.loads(line) for line in captured.err.splitlines()]
    expected = [
        "current_parent_verified",
        "accepted_report_readback_verified",
        "local_result_saved",
    ]
    if not fail_export:
        expected.append("export_saved_history_unchanged")
    assert [p["phase"] for p in progress] == expected
    assert all(p["new_api_requests"] == 0 and not p["authoritative_result"] for p in progress)
    assert all(
        set(p)
        == {"schema_version", "phase", "observed_at", "new_api_requests", "authoritative_result"}
        for p in progress
    )
    result = writes["new:result"][0]
    assert result["schema_version"] == "multimodal-recorded-finalization-result/1.2.0"
    assert result["finalization_grant_checksum"] is None
    assert result["local_review_closure"]["confirmation"] == proof
    assert checks == ["current"]


def test_local_progress_broken_sink_does_not_change_acceptance(monkeypatch):
    def broken(*args, **kwargs):
        raise BrokenPipeError("fixture_closed_stdout")

    monkeypatch.setattr("builtins.print", broken)
    finalization._progress("current_parent_verified")


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"acquisition_increment_only": "true"},
        {"mode": "review"},
        {"schema_version": "multimodal-campaign-renewal/1.25.0"},
    ],
)
def test_committed_increment_entry_requires_original_commit(saved, monkeypatch, change):
    from packages.multimodal_research import campaign_renewal

    (saved.previous / "entry.json").unlink()
    saved.failure["phases"][-1].update(target=None, accepted=False)
    (saved.previous / "result.json").write_text(json.dumps(saved.failure))
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.26.0",
        mode="acquisition",
        review_partition="owner_bound_reused_and_new/1.0.0",
        acquisition_increment_only=True,
    )
    plan.update(change)
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda _: (saved.campaign, saved.failure)
    )
    if change:
        with pytest.raises(ValueError, match="recorded_finalization_renewal_version"):
            finalization.finalize_entry(saved.previous, saved.output, complete=True)
        assert not saved.calls
    else:
        finalization.finalize_entry(saved.previous, saved.output, complete=True)
        assert saved.calls[0][-1] == {"complete": True, "active": True, "require_committed": True}
    assert json.loads((saved.previous / "result.json").read_text()) == saved.failure

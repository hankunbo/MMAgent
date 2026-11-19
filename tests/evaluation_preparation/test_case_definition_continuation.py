"""Admitted checks can be observed without inventing a target interpretation."""

import copy

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from tests.evaluation_preparation.test_definition_terminal_backlog import invoke, saved
from tests.evaluation_preparation.test_revision_acquisition import campaign, revision


def positive(tmp_path, monkeypatch):
    s = saved(tmp_path, monkeypatch)
    s.records.pop("definition:operation:0")
    s.records.pop("definition:wire:0:outcome")
    s.result["new_requests"] = 1
    s.revision.update(
        added=["new:check"],
        rejected=[],
        rows=[
            dict(
                id="new:check",
                definition_receipt_key="draft",
                definition_receipt_checksum=digest(s.revision["accepted"]["draft"]),
                review_allowed=False,
                acquisition_allowed=True,
            )
        ],
    )
    return s


def test_positive_terminal_increment_current_owner_readback(tmp_path, monkeypatch):
    s = positive(tmp_path, monkeypatch)
    before = copy.deepcopy(s.records)
    proof = invoke(s)
    assert proof["definition"]["completed_increment"]["changed_ids"] == ["new:check"]
    assert proof["definition"]["completed_increment"]["replay_authorized"] is False
    assert s.restored == [proof] and s.records == before
    assert invoke(s) == proof


@pytest.mark.parametrize("fault", ["row", "acceptance", "unknown", "cancelled", "source_changed"])
def test_positive_increment_does_not_bypass_origin(tmp_path, monkeypatch, fault):
    s = positive(tmp_path, monkeypatch)
    if fault == "row":
        s.revision["rows"] = []
    elif fault == "acceptance":
        s.revision["accepted"] = {}
    elif fault == "unknown":
        s.events.pop()
    elif fault == "cancelled":
        (s.path / "cancel").write_text("cancelled")
    else:

        def revoked(*args):
            raise ValueError("source_changed")

        monkeypatch.setattr(renewal, "restore_origin_materials", revoked)
    with pytest.raises(ValueError):
        invoke(s)


@pytest.mark.parametrize("pending", [False, True])
def test_new_window_separates_acquisition_and_review(tmp_path, pending):
    c = campaign(tmp_path)
    c.protocol["schema_version"] = "multimodal-campaign-renewal/1.79.0"
    r = revision()
    r["rows"][0].update(dependency_assessment={}, review_allowed=False, acquisition_allowed=True)
    if pending:
        r["replaced"] = []
        c.protocol["pending_definition_acquisition"] = {"targets": ["check:unattempted"]}
    assert renewal.gate_selection(c, r) == ()
    assert c.targets == ("check:unattempted",)


@pytest.mark.parametrize("fault", ["unknown", "confirmed", "missing_input", "not_allowed"])
def test_new_window_preserves_acquisition_gates(tmp_path, fault):
    c = campaign(
        tmp_path,
        attempted=("check:unattempted",) if fault == "unknown" else (),
        confirmed=("check:unattempted",) if fault == "confirmed" else (),
    )
    c.protocol["schema_version"] = "multimodal-campaign-renewal/1.79.0"
    r = revision()
    r["rows"][0].update(
        dependency_assessment={}, review_allowed=False, acquisition_allowed=fault != "not_allowed"
    )
    if fault == "missing_input":
        c.targets = ()
    renewal.gate_selection(c, r)
    assert c.targets == ()


def test_successor_keeps_original_definition_owner_and_source(tmp_path):

    from packages.multimodal_research.definition_expansion import DefinitionExpansion
    from tests.agent_provider.support import Documents
    from tests.evaluation_preparation.test_definition_expansion import fixture

    original = fixture(tmp_path / "prior")
    prior = original.advance()
    original.campaign.base.definition_current = original.read_revision
    c = copy.copy(original.campaign)
    c.base = copy.copy(original.campaign.base)
    c.parent = original.campaign
    c.campaign_id = "successor"
    c.protocol = dict(
        original.campaign.protocol,
        schema_version=renewal.CASE_DEFINITION_VERSION,
        tip=dict(path=str(original.campaign.output), definition=dict(revision=digest(prior))),
    )
    new = DefinitionExpansion(
        c,
        documents=Documents(),
        sources=original.sources,
        provider=None,
        budget=original.budget,
        scope=original.scope,
        policy=original.policy.model_dump(mode="json"),
    )
    result = new.commit_revision(new.original_rows, {}, {}, [], [], [], stop="no_new_definition")
    new.install(result)
    assert result["accepted"] == {}  # no copied/reissued original receipt
    assert result["rows"] == prior["rows"]
    assert result["materials"] == prior["materials"]
    assert new.read_revision() == result
    assert c.targets == tuple(prior["added"])
    original.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="source_changed"):
        new.read_revision()


def test_successor_display_keeps_inherited_check_without_current_addition(tmp_path):
    from types import SimpleNamespace

    from packages.multimodal_research.active_acquisition import structure
    from tests.evaluation_preparation.test_definition_expansion import fixture

    owner = fixture(tmp_path)
    revision = owner.advance()
    inherited = copy.deepcopy(revision)
    inherited.update(added=[], replaced=[], merged=[])
    owner.install(inherited)
    owner.campaign.base.active_owner = SimpleNamespace(
        check=lambda: None, report=lambda: {}, plan={"nodes": [], "rows": []}, native=None
    )
    report = dict(
        requirements=[dict(id=r["id"]) for r in inherited["rows"]],
        report_production={},
        scores=None,
    )
    structure(report, owner.campaign.base)
    assert report["scores"] is None
    for check in report["requirements"]:
        assert check["source_requirement"]["id"] == check["id"]
        assert any(
            e["target"] == "check:" + check["id"] for e in report["presentation_structure"]["edges"]
        )


def test_cold_schedule_restores_confirmed_current_target_not_previous_unknown(tmp_path):
    c = campaign(tmp_path)
    c.output = tmp_path / "current"
    c.protocol.update(
        schema_version=renewal.CASE_DEFINITION_VERSION,
        pending_definition_acquisition={"targets": ["check:unattempted"]},
    )
    r = revision()
    r.update(replaced=[])
    r["rows"][0].update(acquisition_allowed=True, review_allowed=False, dependency_assessment={})
    c.targets = ()
    renewal.restore_definition_schedule(c, {"path": str(c.output)}, ("check:unattempted",), r)
    assert c.targets == ("check:unattempted",)
    assert c.base.review_target_ids == ()
    renewal.restore_definition_schedule(c, {"path": "other"}, (), r)
    assert c.targets == ("check:unattempted",)


@pytest.mark.parametrize("fail", [False, True])
def test_stage_uses_current_original_lease_and_releases_on_error(monkeypatch, fail):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from packages.multimodal_research import stages

    calls = []

    @contextmanager
    def lease():
        calls.append("acquire")
        try:
            yield
        finally:
            calls.append("release")

    def unsupported(*args):
        raise OSError("NFS flock must not be used for this protocol")

    monkeypatch.setattr(stages.fcntl, "flock", unsupported)
    c = SimpleNamespace(
        protocol={"schema_version": renewal.CASE_DEFINITION_VERSION}, execution_session=lease
    )
    if fail:
        with pytest.raises(ValueError, match="body_failure"):
            with stages.stage_execution_session(c):
                raise ValueError("body_failure")
    else:
        with stages.stage_execution_session(c):
            calls.append("body")
    assert calls == (["acquire", "release"] if fail else ["acquire", "body", "release"])


@pytest.mark.parametrize("registered", [None, "context", "control"])
def test_definition_lease_amendment_requires_original_unstarted_owners(tmp_path, registered):
    from types import SimpleNamespace

    from packages.multimodal_research.execution_lease import PROFILE, unstarted

    c = SimpleNamespace(
        output=tmp_path,
        initial_targets=(),
        protocol={
            "schema_version": renewal.CASE_DEFINITION_VERSION,
            "mode": "definitions",
            "definition_increment_only": True,
            "execution_lease_profile": PROFILE,
        },
        request_key=lambda t: str(t),
        contexts=SimpleNamespace(
            read_research_run=lambda rid, missing_ok: {} if registered == "context" else None
        ),
        sessions=SimpleNamespace(read=lambda rid: {} if registered == "control" else None),
    )
    if registered:
        with pytest.raises(ValueError, match="phase_already_registered"):
            unstarted(c)
    else:
        unstarted(c)
        (tmp_path / "cancel").write_text("cancelled")
        with pytest.raises(ValueError, match="unstarted_definition_scope"):
            unstarted(c)

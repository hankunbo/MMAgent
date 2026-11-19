"""New scope CLI exclusion/dispatch boundaries; old stages remain explicit."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import entry, stages
from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research.active_requirements import load
from tests.evaluation_preparation.test_multimodal_entry_lease import bound_entry, fake_repository
from tests.evaluation_preparation.test_scope_selection import PROFILE


def scoped_spec(path):
    spec = bound_entry(path)
    spec["active_acquisition"] = dict(path=PROFILE, checksum=digest(load(PROFILE)))
    (path / "entry.json").write_text(json.dumps(spec))
    return spec


def test_scope_stage_prepare_holds_original_lease_without_dispatch(tmp_path, monkeypatch):
    fake_repository(monkeypatch)
    spec = scoped_spec(tmp_path)
    calls = []

    def assemble(path, *, preflight_only=False):
        assert lease.check_current(path).spec == spec
        calls.append("preflight" if preflight_only else "assemble")
        return SimpleNamespace(
            freeze=lambda: calls.append("freeze"),
            protocol={"schema_version": "multimodal-live-campaign/1.21.0"},
        )

    monkeypatch.setattr(entry, "assemble_entry", assemble)
    monkeypatch.setattr(stages, "prepare", lambda _: calls.append("plan") or {"prepared": True})
    assert stages.cli(SimpleNamespace(command="stage-prepare", input=tmp_path)) == {
        "prepared": True
    }
    assert calls == ["preflight", "assemble", "freeze", "plan"]
    assert lease.check_current() is None
    assert not (tmp_path / "assembly.lock").exists()


@pytest.mark.parametrize("marker", ["cancel", "stage-failure.json", "stage-closure-error.json"])
def test_initial_stopped_entry_not_registered(tmp_path, monkeypatch, marker):
    fake_repository(monkeypatch)
    scoped_spec(tmp_path)
    (tmp_path / marker).touch()
    monkeypatch.setattr(entry, "assemble_entry", lambda *a, **kw: pytest.fail("stopped assembly"))
    with pytest.raises(ValueError, match="stage_previous_failure_read_only"):
        stages.cli(SimpleNamespace(command="stage-prepare", input=tmp_path))
    assert lease.check_current() is None


def test_original_stage_run_cold_assembly_keeps_lease_through_dispatch(tmp_path, monkeypatch):
    fake_repository(monkeypatch)
    scoped_spec(tmp_path)
    (tmp_path / "protocol.json").write_text(
        json.dumps({"schema_version": "multimodal-live-campaign/1.21.0"})
    )
    calls = []
    original = SimpleNamespace(freeze=lambda: calls.append("freeze"))

    def assemble(path):
        assert lease.check_current(path) is not None
        calls.append("assemble")
        return original

    def run(campaign, stage):
        assert campaign is original and lease.check_current(tmp_path) is not None
        calls.append(stage)
        return {"stage": stage}

    monkeypatch.setattr(entry, "assemble_entry", assemble)
    monkeypatch.setattr(stages, "run", run)
    assert stages.cli(SimpleNamespace(command="stage-run", input=tmp_path, stage="sources")) == {
        "stage": "sources"
    }
    assert calls == ["assemble", "freeze", "sources"] and lease.check_current() is None


def test_profile_drift_refused_and_old_protocol_not_promoted(tmp_path):
    spec = scoped_spec(tmp_path)
    spec["active_acquisition"]["checksum"] = digest("changed")
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="stage_scope_profile_changed"):
        stages.scoped_entry(tmp_path)
    (tmp_path / "protocol.json").write_text(
        json.dumps({"schema_version": "multimodal-live-campaign/1.20.0"})
    )
    assert stages.scoped_entry(tmp_path) is False
    with pytest.raises(ValueError, match="stage_cli_requires_prepared_renewal"):
        stages.assemble_for_stage(tmp_path)


def test_scoped_report_restores_current_native_facts_before_media(tmp_path, monkeypatch):
    from packages.multimodal_research import report, scope_selection
    from tests.evaluation_preparation.test_native_collection_report import fixture

    campaign, saved_report, artifacts, reads = fixture()
    native = campaign.base.active_owner.native
    campaign.base.active_owner.native = None
    plan = {"scope": "saved-qualified"}
    calls = []

    def ensure(current, *, allow_native):
        assert current is campaign and allow_native is False
        calls.append("restore")
        current.base.active_owner.native = native
        return plan

    # Read the first persisted phase only after exact current-owner restoration;
    # demonstrate that real media() sees all original per-component artifacts.
    def documents(run):
        assert run == "run" and calls == ["restore"]
        images = report.media(campaign, saved_report)["images"]
        assert set(images) == {a["asset_code"] for a in artifacts}
        raise LookupError("phase-read-boundary")

    campaign.documents = documents
    monkeypatch.setattr(scope_selection, "ensure", ensure)
    with pytest.raises(LookupError, match="phase-read-boundary"):
        report.export(
            campaign, {"scope_method_plan": plan, "phases": [{"run_id": "run"}]}, tmp_path
        )
    assert len(reads) == 2


def test_scoped_report_wrong_plan_refused_before_read(tmp_path, monkeypatch):
    from packages.multimodal_research import report, scope_selection

    monkeypatch.setattr(scope_selection, "ensure", lambda *a, **kw: {"different": True})
    campaign = SimpleNamespace(documents=lambda _: pytest.fail("wrong binding read"))
    with pytest.raises(ValueError, match="scope_method_report_result_binding"):
        report.export(campaign, {"scope_method_plan": {}, "phases": [{"run_id": "run"}]}, tmp_path)


def test_read_only_method_restore_cannot_create_a_runtime_plan(tmp_path, monkeypatch):
    from packages.multimodal_research import scope_selection

    def forbidden():
        pytest.fail("read-only recovery attempted a new plan write")

    plan = dict(
        schema_version="input-scope-method-plan/1.0.0",
        scope_records=[dict(phase=dict(run_id="synthetic"))],
        rows=[],
    )
    base = SimpleNamespace(active_owner=SimpleNamespace(profile={"scope_selection_policy": True}))
    campaign = SimpleNamespace(
        base=base,
        campaign_id="synthetic",
        output=tmp_path,
        documents=lambda _: SimpleNamespace(read=lambda _: None),
        guard_control=forbidden,
    )
    monkeypatch.setattr(scope_selection.SelectionOwner, "current", lambda _: plan)
    with pytest.raises(ValueError, match="scope_selection_saved_plan_missing"):
        scope_selection.ensure(campaign, allow_native=False)
    assert list(tmp_path.iterdir()) == []


def test_read_only_method_restore_repairs_only_committed_mirror(tmp_path):
    from packages.multimodal_research.scope_selection import SelectionOwner

    plan = dict(
        schema_version="input-scope-method-plan/1.0.0",
        scope_records=[dict(phase=dict(run_id="synthetic"))],
        rows=[],
    )
    docs = SimpleNamespace(read=lambda _: (plan,))
    campaign = SimpleNamespace(campaign_id="synthetic", output=tmp_path, documents=lambda _: docs)
    owner = SelectionOwner(campaign)
    owner.current = lambda: plan
    assert owner.restore() == plan
    _, _, mirror = owner.location(plan)
    first = mirror.read_bytes()
    mirror.unlink()
    assert owner.restore() == plan and mirror.read_bytes() == first
    assert owner.restore() == plan and mirror.read_bytes() == first

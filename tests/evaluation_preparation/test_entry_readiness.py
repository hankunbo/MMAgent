"""Public commands, local resource drift and explicit component/profile dispatch."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps import multimodal_case
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.multimodal_research import component_consumer, entry, entry_preflight
from packages.object_store import checksum
from tests.dataset_adapters.test_multimodal import gamma
from tests.evaluation_preparation.test_component_consumer import config


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["multimodal_case", *map(str, args)])
    return multimodal_case.main()


def prepared(tmp_path, monkeypatch, *, component=True):
    root = gamma(tmp_path / "source")
    manifest = prepare("gamma", root, "0001", selection="training").manifest
    task, _ = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")
    path, _ = config(tmp_path, manifest=manifest, task=task)
    value = json.loads(path.read_bytes())
    value["target"] = "active:cup_geometry"
    path.write_text(json.dumps(value))

    def views(command, **kwargs):
        out = Path(command[4])
        out.mkdir(parents=True)
        (out / "views.json").write_text(json.dumps(dict(views=[])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", views)
    out = tmp_path / "prepared"
    args = ["--component-consumer", path] if component else []
    assert (
        invoke(
            monkeypatch,
            "live-prepare",
            "--dataset",
            "gamma",
            "--root",
            root,
            "--sample",
            "0001",
            "--selection",
            "training",
            "--task",
            "advanced_glaucoma",
            "--registry",
            "configs/research/multimodal_propositions_v1.json",
            "--active-acquisition",
            "configs/research/multimodal_active_requirements_v5.json",
            "--resilient-vision",
            "--view-profile",
            "detail-v1",
            "--output",
            out,
            *args,
        )
        == 0
    )
    return out, path


def test_cli_prepare_preflight_component_and_run_reaches_current_owner(
    tmp_path, monkeypatch, capsys
):
    out, _ = prepared(tmp_path, monkeypatch)
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.16.0"
    from apps import fundus_tool_registry

    monkeypatch.setattr(fundus_tool_registry, "load_verified", lambda: pytest.fail("old route"))
    calls = []

    def boundary():
        calls.append("current_owner")
        raise RuntimeError("synthetic private connection detail: do not expose")

    monkeypatch.setattr(entry, "owners", boundary)
    assert invoke(monkeypatch, "preflight", "--input", out) == 0
    assert calls == []
    assert invoke(monkeypatch, "run", "--input", out) == 2
    assert calls == ["current_owner"]
    failure = json.loads(next(out.glob("entry-failure-*.json")).read_bytes())
    assert failure["phase"] == "owner_assembly" and failure["database_writes"] is None
    assert failure["api_requests"] is None
    assert "private connection" not in capsys.readouterr().out


def test_legacy_drift_stops_before_database_and_names_resource(tmp_path, monkeypatch):
    out, _ = prepared(tmp_path, monkeypatch, component=False)
    from apps import fundus_tool_registry as tool

    resource = tmp_path / "measurements.py"
    resource.write_text("new implementation")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"resources": {str(resource): checksum(b"old implementation")}}))
    monkeypatch.setattr(tool, "LOCK", lock)
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("DB before preflight"))
    assert invoke(monkeypatch, "run", "--input", out) == 2
    first = next(out.glob("entry-failure-*.json"))
    original = first.read_bytes()
    failure = json.loads(original)
    assert failure["error_code"] == "fundus_installation_resource_changed"
    assert failure["resource_mismatches"][0]["path"] == str(resource)
    assert failure["database_writes"] == failure["tool_dispatches"] == 0
    assert invoke(monkeypatch, "preflight", "--input", out) == 2
    assert len(list(out.glob("entry-failure-*.json"))) == 2 and first.read_bytes() == original


@pytest.mark.parametrize("field", ["target", "component_id", "input_checksum", "task_checksum"])
def test_consumer_binding_cannot_bypass_local_preflight(tmp_path, monkeypatch, field):
    out, path = prepared(tmp_path, monkeypatch)
    spec = json.loads((out / "entry.json").read_bytes())
    raw = json.loads(path.read_bytes())
    raw[field] = checksum(b"wrong") if field.endswith("checksum") else "wrong"
    path.write_text(json.dumps(raw))
    from packages.grounding.models import digest

    spec["component_consumer"]["checksum"] = digest(component_consumer.load(path))
    (out / "entry.json").write_text(json.dumps(spec))
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("DB before binding"))
    assert invoke(monkeypatch, "preflight", "--input", out) == 2


def test_preflight_success_does_not_cache_authorization_for_run(tmp_path, monkeypatch):
    out, path = prepared(tmp_path, monkeypatch)
    assert invoke(monkeypatch, "preflight", "--input", out) == 0
    Path(component_consumer.load(path).installation.path).write_text("changed")
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("DB after drift"))
    assert invoke(monkeypatch, "run", "--input", out) == 2


@pytest.mark.parametrize("version", ["1.14.0", "1.15.0"])
def test_component_gap_profile_cannot_masquerade_as_old_entry(tmp_path, monkeypatch, version):
    out, _ = prepared(tmp_path, monkeypatch)
    spec = json.loads((out / "entry.json").read_bytes())
    spec["schema_version"] = "multimodal-live-entry/" + version
    (out / "entry.json").write_text(json.dumps(spec))
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("wrong version"))
    assert invoke(monkeypatch, "preflight", "--input", out) == 2


def test_no_native_requirement_does_not_require_fundus_installation():
    manifest = SimpleNamespace(
        components=[
            SimpleNamespace(modality="oct", kind="volume", status="obtained", payload_ids=("oct",))
        ]
    )
    profile = dict(checks=[dict(method="vision", modality="oct")])
    assert (
        entry_preflight.check_local_route(manifest, None, profile, None)
        == "no_native_fundus_required"
    )


@pytest.mark.parametrize("status", ["missing", "withheld"])
def test_unavailable_fundus_does_not_require_unrelated_installation(status, monkeypatch):
    from apps import fundus_tool_registry

    monkeypatch.setattr(
        fundus_tool_registry,
        "LOCK",
        SimpleNamespace(read_bytes=lambda: pytest.fail("no tool preflight for unavailable input")),
    )
    manifest = SimpleNamespace(
        components=[
            SimpleNamespace(modality="color_fundus", kind="image", status=status, payload_ids=())
        ]
    )
    profile = dict(checks=[dict(method="disc_cup_measurement", modality="color_fundus")])
    assert (
        entry_preflight.check_local_route(manifest, None, profile, None)
        == "no_native_fundus_required"
    )


def test_obtained_fundus_still_requires_current_installation(tmp_path, monkeypatch):
    from apps import fundus_tool_registry

    lock = tmp_path / "installation.json"
    lock.write_text(
        json.dumps(dict(resources={str(tmp_path / "missing-model"): "sha256:" + "0" * 64}))
    )
    monkeypatch.setattr(fundus_tool_registry, "LOCK", lock)
    manifest = SimpleNamespace(
        components=[
            SimpleNamespace(
                modality="color_fundus", kind="image", status="obtained", payload_ids=("image",)
            )
        ]
    )
    profile = dict(checks=[dict(method="disc_cup_measurement", modality="color_fundus")])
    with pytest.raises(
        entry_preflight.LocalReadinessError, match="fundus_installation_resource_changed"
    ):
        entry_preflight.check_local_route(manifest, None, profile, None)


def test_cli_routes_new_profile_to_original_component_projection_not_native(tmp_path, monkeypatch):
    """Actual entry/install/projection, with storage and current component reads injected.

    This is software composition, not actual-DB qualification or a real Review.
    """
    import copy

    from apps.vectorengine_case_materials import ref
    from packages.grounding.models import digest
    from packages.multimodal_research import campaign, native_fundus
    from tests.evaluation_preparation.test_active_acquisition import source_fixture
    from tests.evaluation_preparation.test_component_observations import installed_fixture

    out, _ = prepared(tmp_path, monkeypatch)
    visible = SimpleNamespace(case=SimpleNamespace(case_code="CASE:SYNTHETIC"))
    registered = SimpleNamespace(schema_version="dataset-registration/4.0.0", visible=visible)
    inputs = SimpleNamespace(
        repository=SimpleNamespace(read_registration=lambda *args: registered),
        read_visible=lambda *args: visible,
        register_multimodal=lambda *args, **kwargs: pytest.fail("new registration"),
    )
    monkeypatch.setattr(entry, "owners", lambda: (None, None, inputs, "synthetic-target"))
    from packages.multimodal_research import material

    sources = source_fixture()
    monkeypatch.setattr(
        material,
        "MedicalSources",
        lambda: SimpleNamespace(
            initial=sources,
            read=lambda: sources,
        ),
    )
    monkeypatch.setattr(native_fundus, "acquire", lambda *a, **k: pytest.fail("old native"))
    bases = []

    def input_evidence(**kwargs):
        base = SimpleNamespace(
            **kwargs,
            manifest=kwargs["prepared"].manifest,
            detailed=True,
            resources=(ref("resource", "ASSET:SYNTHETIC_INPUT", digest("input")),),
        )
        bases.append(base)
        return base

    monkeypatch.setattr(campaign, "InputEvidence", input_evidence)

    class CommandCampaign(campaign.MultimodalCampaign):
        def __init__(self, base, target, **kwargs):
            self.base, self.target = base, target
            self.targets = ("active:cup_geometry", "active:rim_diffuse")

        def execute(self):
            return dict(complete=True, synthetic_command_routing_only=True)

    monkeypatch.setattr(campaign, "MultimodalCampaign", CommandCampaign)

    def confirmed(base, target, selection, *, guard):
        guard()
        adapter = installed_fixture(tmp_path)
        adapter.base, adapter.target = base, selection.target
        adapter.input_row = copy.deepcopy(next(r for r in base.ledger if r["id"] == adapter.target))
        adapter.input_resources = base.resources
        adapter.input_binding = digest((adapter.input_row, base.resources))
        return adapter

    monkeypatch.setattr(component_consumer, "assemble", confirmed)
    assert invoke(monkeypatch, "run", "--input", out) == 0
    base = bases[0]
    assert base.active_owner.native is None
    assert (
        base.definition_expansion_policy["schema_version"] == "campaign-definition-expansion/1.23.0"
    )
    assert base.observations["active:cup_geometry"].status == "obtained"
    assert base.component_consumer_selection
    base.component_observations.check()

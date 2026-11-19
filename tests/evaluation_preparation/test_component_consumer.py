"""Explicit consumer configuration and existing-entry dispatch boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import component_consumer as module
from packages.object_store import checksum


def config(tmp_path, *, manifest=None, task=None):
    record = tmp_path / "public.json"
    record.write_text("{}")
    file = dict(path=str(record), checksum=checksum(record.read_bytes()), byte_size=2)
    value = dict(
        schema_version="component-consumer-selection/1.0.0",
        installation=file,
        source_selection=file,
        resources=dict(allowed_roots=[str(tmp_path)]),
        producer_run="RUN:PRODUCER",
        request_key="original-request",
        execution_key="original-execution",
        registration_key="original-input",
        component_id="fundus",
        target="check:geometry",
        input_checksum=manifest.checksum if manifest else digest("input"),
        task_checksum=task.checksum if task else digest("task"),
        output_store=str(tmp_path),
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(value))
    return path, module.load(path)


@pytest.mark.parametrize("change", ["checksum", "content", "link", "unknown", "bool"])
def test_selection_rejects_drift_and_untyped_authority(tmp_path, change):
    path, value = config(tmp_path)
    raw = json.loads(path.read_bytes())
    if change == "checksum":
        with pytest.raises(ValueError, match="selection_changed"):
            module.load(path, expected=digest("wrong"))
        return
    if change == "content":
        Path(value.installation.path).write_text("[]")
    elif change == "link":
        link = tmp_path / "link.json"
        link.symlink_to(value.installation.path)
        raw["installation"]["path"] = str(link)
    elif change == "unknown":
        raw["authorized"] = True
    else:
        raw["installation"]["byte_size"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        module.load(path)


def test_registration_reads_original_identity_without_register(tmp_path):
    _, value = config(tmp_path)
    visible = SimpleNamespace(case=SimpleNamespace(case_code="CASE:ORIGINAL"))
    saved = SimpleNamespace(schema_version="dataset-registration/4.0.0", visible=visible)
    calls = []

    def read(key, check):
        calls.append((key, check))
        return saved

    inputs = SimpleNamespace(
        repository=SimpleNamespace(read_registration=read), read_visible=lambda case: visible
    )
    manifest = SimpleNamespace(
        checksum=value.input_checksum, components=[SimpleNamespace(id="fundus")]
    )
    task = SimpleNamespace(checksum=value.task_checksum)
    assert module.registration(inputs, value, manifest, task) is saved
    assert calls == [("original-input", value.input_checksum)]
    task.checksum = digest("wrong task")
    with pytest.raises(ValueError, match="input_or_task"):
        module.registration(inputs, value, manifest, task)
    assert len(calls) == 1


def test_public_entry_config_is_selected_and_old_version_rejected(tmp_path, monkeypatch):
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research import entry
    from tests.dataset_adapters.test_multimodal import gamma

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
    entry.prepare_entry(
        dataset="gamma",
        root=root,
        sample="0001",
        selection="training",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_active_requirements_v4.json",
        resilient_vision=True,
        view_profile="detail-v1",
        component_consumer=path,
    )
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.14.0"
    assert spec["component_consumer"]["checksum"] == digest(module.load(path))

    def boundary():
        raise RuntimeError("original_owner_boundary")

    monkeypatch.setattr(entry, "owners", boundary)
    with pytest.raises(RuntimeError, match="original_owner_boundary"):
        entry.assemble_entry(out)
    spec["schema_version"] = "multimodal-live-entry/1.13.0"
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="component_consumer_entry_version"):
        entry.assemble_entry(out)


@pytest.mark.parametrize("cold", [False, True])
def test_install_uses_existing_campaign_and_freezes_selection(tmp_path, monkeypatch, cold):
    path, selection = config(tmp_path)
    calls = []
    base = SimpleNamespace(output=tmp_path, observations={"check:geometry": "existing"})
    campaign = SimpleNamespace(
        base=base,
        target="db",
        targets=("check:geometry", "other"),
        guard_control=lambda: calls.append("control"),
    )
    if cold:
        (tmp_path / "protocol.json").write_text("{}")

    def assemble(base, target, selected, *, guard):
        assert selected == selection
        guard()
        calls.append("assembled")
        return SimpleNamespace(
            install=lambda: calls.append("install"), restore=lambda: calls.append("restore")
        )

    monkeypatch.setattr(module, "assemble", assemble)
    module.install_campaign(campaign, selection, dict(path=str(path), checksum=digest(selection)))
    assert calls == ["assembled", "restore" if cold else "install"]
    assert campaign.targets == campaign.initial_targets == ("other",)
    assert base.component_consumer_selection["checksum"] == digest(selection)

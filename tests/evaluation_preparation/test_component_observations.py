"""Byte projection and consumer binding probes, not a real campaign claim."""

import copy
import json
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.grounding.models import digest
from packages.multimodal_research.component_observations import (
    ComponentObservations,
    project_fundus,
)
from packages.multimodal_research.fundus_mask_adapter import measure_masks
from packages.multimodal_research.fundus_measurements import measure
from packages.object_store import checksum
from packages.research_runtime.component_steps import ComponentPlannerSteps
from tests.tool_execution.test_fundus_bundle import material as material


def artifact(raw):
    return SimpleNamespace(asset_code="ASSET:TEST_MASK", checksum=checksum(raw))


def test_bundle_projection_reuses_original_geometry(material, tmp_path):
    for name, raw in material.masks.items():
        (tmp_path / (name + ".png")).write_bytes(raw)
    value, geometry = project_fundus(
        material.bundle, input_checksum=checksum(material.image), artifact=artifact(material.bundle)
    )
    assert geometry == measure(tmp_path) == measure_masks(material.masks)
    assert value["values"] == geometry["values"]
    assert value["physical_scale"] is None and geometry["direction"] is None
    assert not geometry["quality_is_accuracy"]


@pytest.mark.parametrize("mode", ["input", "artifact"])
def test_wrong_bytes_or_input_cannot_become_observation(material, mode):
    meta = artifact(material.bundle)
    image = checksum(material.image)
    if mode == "input":
        image = digest("different image")
    else:
        meta.checksum = digest("different artifact")
    with pytest.raises(ValueError, match="component_observation_.*checksum"):
        project_fundus(material.bundle, input_checksum=image, artifact=meta)


def test_no_missing_mask_is_silently_generated(material):
    with pytest.raises(ValueError, match="fundus_mask_set"):
        measure_masks({k: v for k, v in material.masks.items() if k != "cup"})


@pytest.mark.parametrize("field", ["case_code", "input_checksum", "visible_checksum", "task"])
def test_wrong_consumer_rejected_before_original_result_read(field):
    task = SimpleNamespace(checksum=digest("task"))
    request = SimpleNamespace(
        scope=SimpleNamespace(case_code="CASE:ONE"),
        input_checksum=digest("input"),
        visible_checksum=digest("visible"),
        task=task,
    )
    owner = object.__new__(ComponentPlannerSteps)
    owner.requests = SimpleNamespace(parent=SimpleNamespace(request=request))
    base = SimpleNamespace(
        binding={"case_code": "CASE:ONE"},
        task=task,
        visible_value=SimpleNamespace(checksum=request.visible_checksum),
        read_dependency_input=lambda: SimpleNamespace(checksum=digest("input")),
    )
    if field == "case_code":
        base.binding["case_code"] = "CASE:TWO"
    elif field == "task":
        base.task = SimpleNamespace(checksum=digest("other"))
    else:
        setattr(request, field, digest("other"))
    with pytest.raises(ValueError, match="component_observation_consumer_binding"):
        ComponentObservations(
            base=base, owner=owner, execution_key="unused", target="unused", guard=lambda: None
        )


def installed_fixture(tmp_path):
    item = object.__new__(ComponentObservations)
    item.target = "check:geometry"
    item.value = {"values": {"vertical_cdr": 0.5}}
    item.record = {"quality": 0.7, "status": "obtained"}
    item.origin = {"artifact": {"asset_code": "ASSET:MASK", "checksum": digest("mask")}}
    saved = copy.deepcopy((item.value, item.record, item.origin))
    item.binding = digest(saved)
    item.read = lambda: copy.deepcopy(saved)
    item.base = SimpleNamespace(
        output=tmp_path,
        observations={},
        ledger=[{"id": item.target}],
        resources=(ref("resource", "ASSET:INPUT", digest("input")),),
    )
    item.input_row = copy.deepcopy(item.base.ledger[0])
    item.input_resources = item.base.resources
    item.input_binding = digest((item.input_row, item.input_resources))
    return item


@pytest.mark.parametrize("field", ["value", "record", "origin"])
def test_constructor_cache_change_cannot_be_installed(tmp_path, field):
    item = installed_fixture(tmp_path)
    getattr(item, field)["injected"] = True
    with pytest.raises(ValueError, match="component_observation_changed"):
        item.install()
    assert not item.base.observations


@pytest.mark.parametrize("field", ["observation", "quality", "resources", "dependency"])
def test_installed_projection_cannot_drift(tmp_path, field):
    item = installed_fixture(tmp_path)
    item.install()
    item.check()
    base = item.base
    if field == "observation":
        base.observations.clear()
    elif field == "quality":
        base.ledger[0]["quality"] = 1.0
    elif field == "resources":
        base.resources = base.resources[:1]
    else:
        base.ledger[0]["dependency_paths"] = ()
    with pytest.raises(ValueError, match="component_observation_projection_changed"):
        item.check()


def test_report_copy_and_duplicate_install(tmp_path):
    item = installed_fixture(tmp_path)
    item.install()
    item.report()["geometry"]["quality"] = 1.0
    item.check()
    with pytest.raises(ValueError, match="component_observation_already_installed"):
        item.install()
    assert len(item.base.observations) == 1 and len(item.base.resources) == 2


@pytest.mark.parametrize("changed", [False, True])
def test_cold_restore_requires_exact_frozen_observation(tmp_path, changed):
    item = installed_fixture(tmp_path)
    frozen = item.report()
    if changed:
        frozen["observation_binding"] = digest("wrong")
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"component_observations": frozen, "deadline": "unchanged"}))
    original = path.read_bytes()
    if changed:
        with pytest.raises(ValueError, match="component_observation_frozen_binding_changed"):
            item.restore()
        assert not item.base.observations
    else:
        item.restore()
        item.check()
        assert item.base.observations[item.target] == item.observation()
    assert path.read_bytes() == original


def test_cold_restore_rejects_equal_boolean_number(tmp_path):
    item = installed_fixture(tmp_path)
    item.record["quality"] = 1.0
    saved = copy.deepcopy((item.value, item.record, item.origin))
    item.binding = digest(saved)
    item.read = lambda: copy.deepcopy(saved)
    frozen = item.report()
    frozen["geometry"]["quality"] = True
    assert frozen == item.report()  # Python equality alone is not exact JSON identity.
    (tmp_path / "protocol.json").write_text(json.dumps({"component_observations": frozen}))
    with pytest.raises(ValueError, match="component_observation_frozen_binding_changed"):
        item.restore()
    assert not item.base.observations


@pytest.mark.parametrize("empty", [False, True])
def test_installed_measurement_enters_original_review_projection(tmp_path, material, empty):
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.multimodal_research.active_requirements import load, project
    from packages.multimodal_research.campaign import configuration
    from packages.multimodal_research.material import state
    from tests.dataset_adapters.test_multimodal import gamma
    from tests.evaluation_preparation.test_active_acquisition import source_fixture, task
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

    prepared = prepare("gamma", gamma(tmp_path / "dataset"), "0001")
    medical, intent = source_fixture(), task()
    ledger, _ = project(prepared.manifest, medical, intent, load())
    target = next(r for r in ledger if r["acquisition_method"] == "disc_cup_measurement")
    item = installed_fixture(tmp_path)
    item.target = target["id"]
    raw = material.bundle
    if empty:
        from packages.tool_execution.fundus_bundle import encode_bundle
        from tests.tool_execution.test_geometry_qc import mask_png

        png = mask_png((512, 512), bytes(512 * 512))
        raw = encode_bundle(
            input_checksum=checksum(material.image),
            input_shape=(2, 3),
            source_checksum=digest("empty"),
            masks=dict.fromkeys(material.masks, png),
            device="cpu",
            ensemble_std_mean=0.0,
        )
    item.value, item.record = project_fundus(
        raw, input_checksum=checksum(material.image), artifact=artifact(raw)
    )
    original = copy.deepcopy((item.value, item.record, item.origin))
    item.binding, item.read = digest(original), lambda: copy.deepcopy(original)
    seed = dynamic_fixture(1).owner.current()
    item.base.ledger, item.base.resources = ledger, seed.resources
    item.input_row, item.input_resources = copy.deepcopy(target), seed.resources
    item.input_binding = digest((item.input_row, item.input_resources))
    item.install()
    visible = SimpleNamespace(
        case=SimpleNamespace(case_code=seed.material.scope.case_code, laterality="UNKNOWN"),
        checksum=seed.material.scope.visible_checksum,
        components=prepared.manifest,
    )
    root = SimpleNamespace(asset_code="ASSET:ROOT", checksum=digest("root"))
    projected = state(
        context=SimpleNamespace(
            run_id=seed.material.scope.run_id, request=SimpleNamespace(task=intent)
        ),
        visible=visible,
        root=root,
        config=configuration(),
        medical=medical,
        ledger=item.base.ledger,
        observations=item.base.observations,
        resources=item.base.resources,
    )
    tasks = project_review_tasks(projected).tasks
    if empty:
        assert not tasks
        assert item.base.observations[item.target].status == "missing"
        assert (
            next(r for r in item.base.ledger if r["id"] == item.target)["missing_reason"]
            == "confirmed_geometry_unavailable"
        )
        return
    assert len(tasks) == 1
    assert tasks[0].observation == item.base.observations[item.target].reference
    assert projected.policy.items[0].quality.value == item.record["quality"]


def test_control_only_guard_does_not_reenter_observation_checks(tmp_path):
    from datetime import UTC, datetime, timedelta

    from packages.multimodal_research.campaign import MultimodalCampaign

    campaign = object.__new__(MultimodalCampaign)
    calls = []
    campaign.base = SimpleNamespace(check=lambda: (calls.append("check"), campaign.guard_control()))
    campaign.output = tmp_path
    campaign.protocol = {"previous_attempts": []}
    campaign.previous_attempts = lambda: []
    campaign.work_deadline = lambda: datetime.now(UTC) + timedelta(seconds=60)
    campaign.usage = lambda: 0
    campaign.guard()
    assert calls == ["check"]

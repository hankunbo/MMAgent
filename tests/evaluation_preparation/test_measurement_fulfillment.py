"""An explicit source-component fulfillment is one review, not extra evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.multimodal_research.active_acquisition import install, structure
from packages.multimodal_research.active_requirements import load, project
from packages.multimodal_research.requirement_catalog import DEFAULT_PROFILE, fulfill_measurements
from tests.dataset_adapters.test_input_capabilities import projected
from tests.dataset_adapters.test_multimodal import gdp, grape
from tests.evaluation_preparation.test_active_acquisition import source_fixture
from tests.evaluation_preparation.test_capability_selection import task

OLD = "configs/research/multimodal_catalog_acquisition_v1.json"


def test_fulfillment_schema_and_policy_reject_unknown_or_visual_questions(tmp_path):
    from packages.multimodal_research.requirement_catalog import (
        MeasurementFulfillment,
        validate_fulfillment,
    )

    schema = Path("packages/multimodal_research/schemas/MeasurementFulfillment.json")
    assert schema.read_text() == (
        json.dumps(MeasurementFulfillment.model_json_schema(), sort_keys=True, indent=2) + "\n"
    )
    for key in ("unknown", "oct_rnfl"):
        profile = load(DEFAULT_PROFILE)
        profile["measurement_fulfillment"]["questions"] = {key: ["tds"]}
        with pytest.raises(ValueError, match="measurement_fulfillment_question_method"):
            validate_fulfillment(profile)
    profile = load(DEFAULT_PROFILE)
    del profile["requirement_catalog"]
    with pytest.raises(ValueError, match="measurement_fulfillment_requires_catalog"):
        validate_fulfillment(profile)


def selection(p, profile):
    return project(p.manifest, source_fixture(), task(), load(profile), capabilities=projected(p))


def test_same_source_measurement_gets_one_joint_review_with_original_questions(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    before, old_plan = selection(p, OLD)
    rows, plan = selection(p, DEFAULT_PROFILE)
    assert len(rows) == len(before) - 1
    assert plan["schema_version"] == "source-bound-acquisition-plan/1.3.0"
    shared = next(r for r in rows if r.get("fulfilled_requirements"))
    original = next(r for r in before if r["id"] == "active:functional_stage")
    assert shared["component"] == original["component"]
    assert shared["parent"] == "FUNCTION"
    assert shared["category"] == "tds"
    binding = shared["fulfilled_requirements"][0]
    assert binding["original_question"] == original["definition"].model_dump(mode="json")
    assert original["definition"].interpretation in shared["definition"].interpretation
    assert not binding["medical_completeness_claimed"]
    assert not any(r["id"] == original["id"] for r in rows)
    assert plan["measurement_fulfillment"]["removed_duplicate_review_count"] == 1
    assert old_plan["schema_version"] == "source-bound-acquisition-plan/1.2.0"


def test_visits_remain_separate_and_other_measurements_never_merged(tmp_path):
    p = prepare("grape", grape(tmp_path), "1:OD", cutoff_visit=1)
    before, _ = selection(p, OLD)
    rows, plan = selection(p, DEFAULT_PROFILE)
    shared = [r for r in rows if r.get("fulfilled_requirements")]
    assert len(shared) == 2 and len({r["component"] for r in shared}) == 2
    assert len(rows) == len(before) - 2
    assert all(r["category"] == "vf" for r in shared)
    assert {r["id"] for r in before if r["category"] == "iop"} == {
        r["id"] for r in rows if r["category"] == "iop"
    }
    assert plan["measurement_fulfillment"]["clinical_completeness_claimed"] is False


@pytest.mark.parametrize("change", ["source", "method", "concept", "definition_bound"])
def test_no_silent_merge_when_qualification_or_joint_definition_does_not_match(tmp_path, change):
    from packages.grounding.models import digest, validate

    p = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    before, plan = selection(p, OLD)
    rows = copy.deepcopy(before)
    broad = next(r for r in rows if r["id"] == "active:functional_stage")
    selected = next(r for r in rows if r["id"] == "check:" + broad["component"])
    profile = load(DEFAULT_PROFILE)
    if change in {"source", "definition_bound"}:
        value = selected["definition"].model_dump(mode="json")
        if change == "source":
            value["source_refs"] = [source_fixture()["aao_signs"].reference.model_dump(mode="json")]
        else:
            value["interpretation"] = "Bounded original statement. " * 8
        value["reference"]["checksum"] = digest(
            {k: v for k, v in value.items() if k != "reference"}
        )
        selected["definition"] = validate(type(selected["definition"]), value)
    elif change == "method":
        selected["acquisition_method"] = "vision"
    else:
        profile["measurement_fulfillment"]["questions"]["functional_stage"] = ["unrelated"]
    result, _, evidence = fulfill_measurements(rows, plan["rows"], p.manifest, profile)
    assert len(result) == len(before)
    assert evidence["removed_duplicate_review_count"] == 0
    assert evidence["outcomes"][0]["state"] == "retained"


def test_original_install_reads_each_component_once_and_report_preserves_relationship(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path / "data"), "data_0001.npz")
    output = tmp_path / "run"
    output.mkdir()
    sources = source_fixture()
    base = SimpleNamespace(
        prepared=p,
        manifest=p.manifest,
        output=output,
        task=task(),
        sources=SimpleNamespace(initial=sources, read=lambda: sources),
    )
    install(base, load(DEFAULT_PROFILE), defer_native=True)
    numeric = [r for r in base.ledger if r["acquisition_method"] == "source_measurement"]
    assert len({r["component"] for r in numeric}) == len(numeric)
    shared = next(r for r in numeric if r.get("fulfilled_requirements"))
    assert shared["id"] in base.observations
    assert "active:functional_stage" not in base.observations
    before = {k: v.model_dump_json() for k, v in base.observations.items()}
    install(base, load(DEFAULT_PROFILE), defer_native=True, read_only=True)
    assert {k: v.model_dump_json() for k, v in base.observations.items()} == before
    report = dict(
        requirements=[
            dict(id=r["id"], presentation_sections=dict(acquisition={}, rationale={}))
            for r in base.ledger
        ],
        report_production={},
    )
    value = structure(report, base)
    item = next(r for r in value["requirements"] if r["id"] == shared["id"])
    assert item["presentation_sections"]["acquisition"]["shared_measurement_questions"]
    assert any(
        e["target"] == "check:" + shared["id"] and e["source"] == "medical:FUNCTION"
        for e in value["presentation_structure"]["edges"]
    )


def test_original_review_task_projection_consumes_one_combined_definition(tmp_path):
    from apps.vectorengine_case_materials import ref
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.grounding.models import digest
    from packages.multimodal_research import material
    from packages.multimodal_research.campaign import configuration
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

    p = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    rows, _ = selection(p, DEFAULT_PROFILE)
    seed = dynamic_fixture(1).owner.current()
    scope = seed.material.scope
    context = SimpleNamespace(run_id=scope.run_id, request=SimpleNamespace(task=task()))
    visible = SimpleNamespace(
        case=SimpleNamespace(case_code=scope.case_code, laterality="UNKNOWN"),
        checksum=scope.visible_checksum,
        components=SimpleNamespace(checksum=scope.input_checksum),
    )
    root = SimpleNamespace(asset_code=scope.graph.code, checksum=scope.graph.checksum)
    observations = {}
    by_id = {c.id: c for c in p.manifest.components}
    for r in rows:
        if r["acquisition_method"] != "source_measurement" or not r["component"]:
            continue
        value = material.numeric(by_id[r["component"]], dict(p.payload_bytes))
        if value is not None:
            observations[r["id"]] = material.observation(r["id"], value)
    state = material.state(
        context=context,
        visible=visible,
        root=root,
        config=configuration(),
        medical=source_fixture(),
        ledger=rows,
        observations=observations,
        resources=(ref("resource", "RESOURCE:INPUT", digest("input")),),
    )
    tasks = project_review_tasks(state)
    assert not any(t.assessment_id == "active:functional_stage" for t in tasks.tasks)
    selected = next(t for t in tasks.tasks if t.assessment_id == "check:rnflt:tds")
    assert "同一原始测量" in selected.definition.interpretation

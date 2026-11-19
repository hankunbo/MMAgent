"""Current definition owners survive automatic catalog reconciliation."""

import copy
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner, requirement_rows
from packages.grounding.models import canonical, digest
from packages.multimodal_research.active_acquisition import successor_projection
from packages.multimodal_research.active_requirements import load, project
from tests.agent_provider.support import Documents
from tests.dataset_adapters.test_multimodal import gamma
from tests.evaluation_preparation.test_active_acquisition import source_fixture
from tests.evaluation_preparation.test_definition_expansion import Sources, draft

PROFILE = "configs/research/multimodal_catalog_acquisition_v6.json"


def fixture(tmp_path):
    prepared = prepare("gamma", gamma(tmp_path), "0001")
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    sources = source_fixture()
    rows, _ = project(
        prepared.manifest,
        sources,
        task,
        load("configs/research/multimodal_structural_coverage_v3.json"),
    )
    materials = Sources()
    owner = ResearchDefinitionOwner(
        documents=Documents(),
        sources=materials,
        campaign_id="synthetic-transition",
        target=task.target_code,
        input_checksum=prepared.manifest.checksum,
        guard=lambda: None,
        input_reader=lambda: prepared.manifest,
    )
    receipt = owner.accept("new", draft(task.target_code, modality="color_fundus"))
    added = requirement_rows(receipt, prepared.manifest)
    assert len(added) == 1
    added[0]["definition_receipt_key"] = "new"
    rows.extend(added)
    revision = dict(rows=rows, accepted={"new": receipt})

    def current():
        assert owner.read("new") == receipt
        return revision

    base = SimpleNamespace(
        manifest=prepared.manifest,
        prepared=prepared,
        task=task,
        ledger=rows,
        sources=SimpleNamespace(read=lambda: sources),
        definition_current=current,
        read_dependency_input=lambda: prepared.manifest,
    )
    return base, materials, revision


def test_original_acceptance_survives_catalog_selection_without_parent_mutation(tmp_path):
    base, _, _ = fixture(tmp_path)
    before = canonical(base.ledger)
    value = successor_projection(base, load(PROFILE))
    assert value["coverage"] == dict(previous=19, catalog=18, selected=19)
    assert len([r for r in value["changes"] if r["action"] == "retained_equivalent"]) == 16
    assert {
        r["requirement_id"] for r in value["changes"] if r["action"] == "retained_input_gap"
    } == {"active:cup_asymmetry", "active:functional_stage"}
    assert value["changes"][-1]["action"] == "retained_current_definition"
    assert digest(value["requirements"][-1]) == digest(base.ledger[-1])
    assert not value["execution_authorized"]
    assert all(not r["historical_review_is_current_authorization"] for r in value["changes"])
    assert canonical(base.ledger) == before
    assert successor_projection(base, load(PROFILE)) == value


def test_withdrawn_original_source_does_not_gain_catalog_authority(tmp_path):
    base, materials, _ = fixture(tmp_path)
    materials.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        successor_projection(base, load(PROFILE))


@pytest.mark.parametrize("fault", ["no_owner", "receipt", "ledger", "missing_static"])
def test_unproven_or_dropped_historical_definition_rejected(tmp_path, fault):
    base, _, revision = fixture(tmp_path)
    if fault == "no_owner":
        base.definition_current = None
    elif fault == "receipt":
        base.ledger[-1]["definition_receipt_checksum"] = digest("wrong")
    elif fault == "ledger":
        revision["rows"] = copy.deepcopy(base.ledger)
        base.ledger[0]["id"] = "changed-without-owner"
    else:
        base.ledger[0]["id"] = "no-longer-in-catalog"
    with pytest.raises(ValueError, match="capability_transition_"):
        successor_projection(base, load(PROFILE))


def test_interpretation_change_requires_reassessment_not_historical_score_copy(tmp_path):
    base, _, _ = fixture(tmp_path)
    base.ledger[0]["evidence_unit"] = "old-shared-dependency"
    value = successor_projection(base, load(PROFILE))
    assert value["changes"][0]["action"] == "requires_reassessment"
    assert not value["changes"][0]["interpretation_unchanged"]


def test_requirement_plan_cli_rejects_task_override_before_owner_read(monkeypatch):
    import sys

    from apps.multimodal_case import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "requirements-plan",
            "--previous",
            "old",
            "--task",
            "new-task",
            "--output",
            "unused",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def native_fixture(tmp_path):
    from apps.vectorengine_case_materials import ref
    from packages.multimodal_research.active_acquisition import bind_native_observation

    base, _, _ = fixture(tmp_path)
    target = next(r for r in base.ledger if r["id"] == "active:cup_geometry")
    artifact = dict(asset_code="ASSET:MEASUREMENT", checksum=digest("original-measurement"))
    native = dict(
        schema_version="native-fundus-acceptance/1.0.0",
        status="obtained",
        plan=dict(target=target["id"], component=target["component"]),
        measurement=dict(
            values={"vertical_cdr": 0.62},
            quality=0.8,
            coordinate_system="pixel_grid",
            radial_rim=None,
            radial_missing_reason="not_available",
        ),
        prediction=dict(training_overlap="unknown"),
        artifact=artifact,
        artifacts=[artifact],
    )
    base.resources = (ref("resource", "ASSET:INPUT", digest("original-input")),)
    base.observations = {}
    reads = []
    base.active_owner = SimpleNamespace(native=native, check=lambda: reads.append("current"))
    bind_native_observation(base, target, native)
    return base, native, reads


def test_current_native_fact_survives_plan_without_new_execution_or_parent_mutation(tmp_path):
    import json

    base, _, reads = native_fixture(tmp_path)
    before = canonical(dict(rows=base.ledger, facts=base.observations, resources=base.resources))
    for _ in range(2):
        value = successor_projection(base, load(PROFILE))
        row = next(r for r in value["requirements"] if r["id"] == "active:cup_geometry")
        change = next(r for r in value["changes"] if r["requirement_id"] == row["id"])
        assert change["action"] == "retained_equivalent"
        assert row["quality"] == 0.8
        assert row["dependency_paths"] == [["ASSET:INPUT", "ASSET:MEASUREMENT"]]
        proof = value["retained_native_observations"]
        assert len(proof) == 1 and proof[0]["new_tool_execution"] is False
        assert proof[0]["observation_checksum"] == digest(base.observations[row["id"]])
        assert json.loads(base.observations[row["id"]].value)["values"] == {"vertical_cdr": 0.62}
        assert (
            canonical(dict(rows=base.ledger, facts=base.observations, resources=base.resources))
            == before
        )
    assert len(reads) >= 4


@pytest.mark.parametrize("fault", ["fact", "quality", "component", "definition", "revoked"])
def test_native_history_is_not_authority_for_changed_fact_or_current_qualification(tmp_path, fault):
    base, native, _ = native_fixture(tmp_path)
    row = next(r for r in base.ledger if r["id"] == "active:cup_geometry")
    if fault == "fact":
        base.observations.clear()
    elif fault == "quality":
        row["quality"] = 1.0
    elif fault == "component":
        native["plan"]["component"] = "wrong-input"
    elif fault == "definition":
        row["definition"] = {"changed": True}
    else:

        def withdrawn():
            raise ValueError("native_fundus_current_binding")

        base.active_owner.check = withdrawn
    with pytest.raises(
        ValueError, match="capability_transition_native|native_fundus_current_binding"
    ):
        successor_projection(base, load(PROFILE))


def test_native_owner_rechecked_after_source_projection(tmp_path):
    base, _, reads = native_fixture(tmp_path)

    def changed_at_final_gate():
        reads.append("current")
        if len(reads) == 3:
            raise ValueError("native_fundus_current_binding")

    base.active_owner.check = changed_at_final_gate
    with pytest.raises(ValueError, match="native_fundus_current_binding"):
        successor_projection(base, load(PROFILE))

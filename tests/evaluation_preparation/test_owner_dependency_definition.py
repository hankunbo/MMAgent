"""Typed current-owner conditions, never model-supplied qualifications."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_acceptance import requirement_rows
from packages.evidence_reuse.definitions import CONFIRMED_METHOD_CONDITION, TECHNICAL_CONDITIONS
from packages.grounding.models import digest
from packages.multimodal_research.definition_context import owner_capabilities, requirement_context
from packages.multimodal_research.definition_expansion import (
    definition_config,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_confirmed_interpretation_definition import confirmed_fixture
from tests.evaluation_preparation.test_definition_dependency_bindings import setup


def contract(index, kind, *, method="confirmed_fundus_disc_cup_geometry", unit="ratio"):
    quality = kind if kind in TECHNICAL_CONDITIONS else None
    return dict(
        index=index,
        text=TECHNICAL_CONDITIONS[quality] if quality else CONFIRMED_METHOD_CONDITION,
        kind="quality" if quality else "method",
        blocks=["interpretation"],
        modality=None,
        concept=None,
        rationale="Exact synthetic owner condition, not medical accuracy",
        excerpt_indices=[0],
        predicate="confirmed_technical_qc" if quality else "confirmed_method",
        input_alias=None,
        requirement_alias="requirement:0",
        method=method,
        unit=unit,
        quality_check=quality,
    )


def confirmed(tmp_path):
    owner, value, item, previous, docs = confirmed_fixture(tmp_path, technical=True)
    value["schema_version"] = "source-definition-draft/7.1.0"
    value["dependency_contracts"] = [
        contract(i, kind)
        for i, kind in enumerate(("method", "artifact_integrity", "mask_measurement_available"))
    ]
    value["dependencies"] = [c["text"] for c in value["dependency_contracts"]]
    return owner, value, item, previous, docs


def raw():
    owner, manifest, sources, docs, base, value = setup()
    value.update(schema_version="source-definition-draft/6.0.0", input_aliases=["input:0"])
    value["dependency_contracts"][0].update(
        input_alias="input:0", requirement_alias=None, quality_check=None
    )
    return owner, manifest, sources, docs, base, value


def test_original_owner_resolves_method_and_narrow_qc_preserving_old_bytes(tmp_path):
    owner, value, item, previous, docs = confirmed(tmp_path)
    old = copy.deepcopy(previous)
    receipt = owner.accept("typed", value)
    assert receipt["schema_version"] == "accepted-research-definition/6.0.0"
    assert receipt["confirmed_basis"]["schema_version"] == "confirmed-interpretation-basis/2.0.0"
    assert all(i["satisfied"] for i in receipt["dependency_assessment"]["items"])
    assert receipt["definition"]["interpretation_status"] == "installed"
    assert owner.read("typed") == receipt
    current, row = owner.read_confirmed_projection("typed")
    assert current == receipt and row["id"] == item.target
    assert owner.read("previous", historical=True) == old
    assert (
        item.interpretation_basis(item.target)["schema_version"]
        == "confirmed-interpretation-basis/1.0.0"
    )
    assert "technical_checks" not in item.interpretation_basis(item.target)
    with pytest.raises(ValueError, match="definition_confirmed_projection_owner_required"):
        requirement_rows(receipt, item.base.manifest)


@pytest.mark.parametrize("kind", ["quality", "patient_input"])
def test_clinical_quality_and_missing_vf_remain_unresolved(tmp_path, kind):
    owner, value, _, _, _ = confirmed(tmp_path)
    text = "Independent medical condition still required"
    value["dependencies"].append(text)
    value["dependency_contracts"].append(
        dict(
            index=3,
            text=text,
            kind=kind,
            blocks=["interpretation"],
            modality="VF" if kind == "patient_input" else None,
            concept="visual_field" if kind == "patient_input" else None,
            rationale="No current proof of this distinct condition",
            excerpt_indices=[0],
            predicate="unresolved",
        )
    )
    receipt = owner.accept("limited", value)
    assert receipt["definition"]["interpretation_status"] == "missing"
    assert not receipt["dependency_assessment"]["items"][-1]["satisfied"]


def test_examination_only_does_not_become_directional_after_technical_checks(tmp_path):
    owner, value, _, _, _ = confirmed(tmp_path)
    value["relation_use"] = "examination_only"
    saved = owner.accept("examination", value)
    assert all(i["satisfied"] for i in saved["dependency_assessment"]["items"])
    assert not saved["dependency_assessment"]["review_allowed"]


@pytest.mark.parametrize(
    "mutation", ["alias", "method", "unit", "quality_text", "caller_flag", "old_version"]
)
def test_wrong_or_broadened_condition_cannot_write(tmp_path, mutation):
    owner, value, _, _, docs = confirmed(tmp_path)
    if mutation in {"alias", "method", "unit"}:
        value["dependency_contracts"][0][{"alias": "requirement_alias"}.get(mutation, mutation)] = (
            "requirement:1" if mutation == "alias" else "wrong"
        )
    elif mutation == "quality_text":
        value["dependencies"][1] = value["dependency_contracts"][1]["text"] = (
            "Segmentation is medically accurate"
        )
    elif mutation == "caller_flag":
        value["dependency_contracts"][0]["satisfied"] = True
    else:
        value["schema_version"] = "source-definition-draft/5.1.0"
    with pytest.raises(ValueError):
        owner.accept("invalid", value)
    assert docs.read("definition:invalid") is None


def test_current_changed_fact_refuses_historical_remains_exact(tmp_path):
    owner, value, item, _, _ = confirmed(tmp_path)
    saved = owner.accept("typed", value)
    item.value["tampered"] = True
    with pytest.raises(ValueError, match="component_observation_changed"):
        owner.read("typed")
    assert owner.read("typed", historical=True) == saved


def test_input_alias_selects_exact_component_and_new_row():
    owner, manifest, _, _, _, value = raw()
    saved = owner.accept("raw", value)
    assert saved["schema_version"] == "accepted-research-definition/5.0.0"
    row = requirement_rows(saved, manifest, fact_reader=owner.fact_reader)[0]
    assert row["component"] == "image-a" and row["review_allowed"]
    assert owner.read("raw") == saved


@pytest.mark.parametrize(
    "mutation", ["unknown", "derived_concept", "display_alias", "omitted", "withheld"]
)
def test_input_alias_errors_are_not_silently_corrected(mutation):
    owner, manifest, _, docs, _, value = raw()
    if mutation == "unknown":
        value["dependency_contracts"][0]["input_alias"] = "input:9"
    elif mutation == "derived_concept":
        value["dependency_contracts"][0]["concept"] = "derived_measurement"
    elif mutation == "display_alias":
        value["dependency_contracts"][0]["input_alias"] = "input-component:0"
    elif mutation == "omitted":
        value["input_aliases"] = []
    else:
        manifest.components = (manifest.components[0].model_copy(update={"status": "withheld"}),)
    with pytest.raises(ValueError, match="definition_input_alias|input_alias"):
        owner.accept("invalid", value)
    assert docs.read("definition:invalid") is None


def test_alias_disambiguates_same_concept_without_borrowing_other_input():
    owner, manifest, _, _, _, value = raw()
    manifest.components += (
        manifest.components[0].model_copy(
            update={"id": "image-b", "status": "missing", "payload_ids": ()}
        ),
    )
    value["input_aliases"] = ["input:1"]
    value["dependency_contracts"][0]["input_alias"] = "input:1"
    saved = owner.accept("missing", value)
    assert not saved["dependency_assessment"]["review_allowed"]
    assert (
        requirement_rows(saved, manifest, fact_reader=owner.fact_reader)[0]["component"]
        == "image-b"
    )


def test_raw_input_after_source_read_change_is_rejected():
    owner, manifest, sources, docs, _, value = raw()
    original = sources.read

    def changing(code):
        result = original(code)
        # Change after input has been read, during later source material reads.
        if getattr(owner, "seen_input", False):
            manifest.components = (manifest.components[0].model_copy(update={"id": "changed"}),)
        return result

    sources.read = changing

    def read_input():
        owner.seen_input = True
        return SimpleNamespace(checksum=manifest.checksum, components=manifest.components)

    owner.input_reader = read_input
    with pytest.raises(ValueError, match="definition_dependency_input_changed"):
        owner.accept("changed", value)
    assert docs.read("definition:changed") is None


def test_public_raw_aliases_separate_from_requirement_display_order(tmp_path):
    owner, value, item, _, _ = confirmed(tmp_path)
    rows = item.base.ledger
    payload = dict(
        existing_requirement_context=requirement_context(rows, [], item.base.observations)
    )
    item.base.component_observations = item
    result = owner_capabilities(payload, item.base, rows)
    assert result["input_capabilities"][0]["input_alias"] == "input:0"
    assert result["input_capabilities"][0]["concept"] == "texture"
    entry = result["existing_requirement_context"]["requirements"][0]
    assert entry["input_alias"] == "input:0" and entry["component"] == "input-component:0"
    assert result["confirmed_observation_capabilities"][0]["method"] == value["method"]
    assert "image-a" not in str(result) and "observation_binding" not in str(result)


def test_new_policy_prompt_and_freeze_explicit_old_config_unchanged():
    from packages.multimodal_research.campaign import configuration
    from packages.multimodal_research.implementation import freeze

    path = Path("configs/research/multimodal_definition_gate_v22.json")
    new = read_expansion_policy(json.loads(path.read_bytes()))
    old = read_expansion_policy(
        json.loads(
            Path("configs/research/multimodal_definition_gate_v21_seed_r1.json").read_bytes()
        )
    )
    assert (
        new.prompt_version == "definition-draft-6.0.0"
        and old.prompt_version == "definition-draft-5.0.0"
    )
    configured = definition_config(configuration(), prompt_version=new.prompt_version)
    assert (
        next(r for r in configured.roles if r.role == "source").prompt_version == new.prompt_version
    )
    assert str(path) in freeze()


def test_source_provider_to_same_id_revision_and_repeat(tmp_path):
    from tests.evaluation_preparation.test_confirmed_interpretation_definition import (
        exercise_source_proposal,
    )

    exercise_source_proposal(tmp_path, typed=True)


def test_new_trial_entry_dispatch_and_old_policy_reject(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial
    from tests.evaluation_preparation.test_trial_entry_window import prepare_entry

    _, spec = prepare_entry(tmp_path, monkeypatch)
    value = definition_trial.prepare(
        tmp_path,
        expansion_policy="configs/research/multimodal_definition_gate_v22.json",
        authorization="offline typed owner dependency contract test only",
        seconds=5400,
        entry_time_window=True,
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.18.0"
    assert definition_trial.read(tmp_path) == value
    value["schema_version"] = "multimodal-definition-trial/1.17.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        definition_trial.checked(value, spec)


@pytest.mark.parametrize("number", [0.0, -1.5])
def test_exact_raw_numeric_alias_and_unit(number):
    from packages.dataset_adapters.multimodal.models import Component

    owner, manifest, _, _, _, value = raw()
    manifest.components += (
        Component(
            id="metric",
            kind="scalar",
            concept="metric",
            modality="measurement",
            status="obtained",
            value=number,
            unit="unit-x",
            unit_basis="synthetic source",
            source_locator="synthetic",
        ),
    )
    value["dependency_contracts"][0].update(
        input_alias="input:1",
        predicate="source_numeric_available",
        modality="measurement",
        concept="metric",
        method="source_numeric_read",
        unit="unit-x",
    )
    saved = owner.accept("numeric", value)
    item = saved["dependency_assessment"]["items"][0]
    assert item["satisfied"] and item["evidence"]["numeric"]["value"] == number
    value["dependency_contracts"][0]["unit"] = "wrong"
    assert not owner.accept("wrong-unit", value)["dependency_assessment"]["review_allowed"]


def test_current_source_withdrawal_and_no_dependency_remain_distinct(tmp_path):
    owner, value, _, _, _ = confirmed(tmp_path)
    value.update(dependencies=[], dependency_contracts=[])
    saved = owner.accept("no-deps", value)
    assert saved["dependency_assessment"]["review_allowed"]
    owner.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        owner.read("no-deps")
    assert owner.read("no-deps", historical=True) == saved


@pytest.mark.parametrize("available", [False, None])
def test_unproven_quality_not_promoted_by_confirmed_method(tmp_path, available):
    owner, value, item, _, _ = confirmed(tmp_path)
    # Lower-level synthetic owner fixture only; production proof is covered by PG.
    item.record["qc"]["measurement_available"] = available
    saved = copy.deepcopy((item.value, item.record, item.origin))
    item.binding = digest(saved)
    item.read = lambda: copy.deepcopy(saved)
    receipt = owner.accept("qc", value)
    checks = receipt["dependency_assessment"]["items"]
    assert checks[0]["satisfied"] and checks[1]["satisfied"]
    assert not checks[2]["satisfied"] and not receipt["dependency_assessment"]["review_allowed"]

"""One registered question, explicit modality-specific research operationalization."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.multimodal_research.active_acquisition import install
from packages.multimodal_research.active_requirements import load, project
from packages.multimodal_research.campaign import MultimodalCampaign
from tests.dataset_adapters.support import fair
from tests.dataset_adapters.test_input_capabilities import projected, reseal
from tests.dataset_adapters.test_multimodal import gamma
from tests.evaluation_preparation.test_active_acquisition import source_fixture

PROFILE = "configs/research/multimodal_catalog_acquisition_v6.json"
CATALOG = "configs/research/requirement_catalog_v2.json"
REGISTRY = "configs/research/multimodal_propositions_v2.json"
SLO_QUESTIONS = {
    "rim_diffuse",
    "rim_focal",
    "cup_size",
    "vessel_nasalization",
    "circumlinear_vessel",
    "rnfl_focal",
    "rnfl_diffuse",
    "peripapillary",
    "tilt",
}


def selection(prepared, profile=PROFILE, medical=None):
    task = load_intent("advanced_glaucoma", REGISTRY)[0]
    return project(
        prepared.manifest,
        medical or source_fixture(),
        task,
        load(profile),
        capabilities=projected(prepared),
    )


def test_slo_matches_original_questions_without_color_grants_or_duplicate_slots(tmp_path):
    fair(tmp_path)
    prepared = prepare("fairvlmed", tmp_path, "data_00000.npz")
    rows, plan = selection(prepared)
    actual = [r for r in rows if r.get("component") == "slo_fundus"]
    assert {r["id"] for r in actual} == {"active:" + q for q in SLO_QUESTIONS}
    assert len(rows) == 19  # Original 18 medical questions plus original age; no SLO clone.
    for row in actual:
        assert row["input_modalities"] == ("SLO",)
        assert row["acquisition_method"] == "vision"
        assert "灰度" in row["definition"].interpretation
        assert row["capability_selection"]["execution_authorized"] is False
        assert row["input_variant"]["selected_modality"] == "SLO"
    absent = {r["id"] for r in plan["rows"] if r["state"] == "input_gap"}
    assert {"active:cup_geometry", "active:hemorrhage", "active:pallor"} <= absent
    base = SimpleNamespace(active_owner=True, ledger=rows, observations={})
    scheduled = MultimodalCampaign.select_targets(base, {"slo_fundus"})
    assert set(scheduled) == {"active:" + q for q in SLO_QUESTIONS}


def test_color_projection_keeps_old_questions_and_no_slo_aliases(tmp_path):
    prepared = prepare("gamma", gamma(tmp_path), "0001")
    old, old_plan = selection(prepared, "configs/research/multimodal_catalog_acquisition_v5.json")
    new, new_plan = selection(prepared)
    assert old == new
    assert old_plan["rows"] == new_plan["rows"]
    assert len(new) == 18


def test_variants_keep_input_identity_and_match_modality_not_dataset(tmp_path):
    prepared = prepare("gamma", gamma(tmp_path), "0001")
    components = tuple(
        c.model_copy(update={"modality": "SLO"}) if c.modality == "color_fundus" else c
        for c in prepared.manifest.components
    )
    slo = replace(
        prepared, manifest=reseal(CaseComponents, prepared.manifest, components=components)
    )
    rows, _ = selection(slo)
    assert any(r.get("input_variant") for r in rows)
    alias = replace(
        slo,
        manifest=reseal(
            CaseComponents,
            slo.manifest,
            dataset="harvard_gdp",
            task="glaucoma_detection",
            selection="development_preparation",
            sample="synthetic_source_alias",
        ),
    )
    other, trace = selection(alias)
    assert rows == other
    assert trace["input_checksum"] != slo.manifest.checksum


def test_source_withdrawal_and_cold_plan_recheck_preserve_original_owner(tmp_path):
    fair(tmp_path / "data")
    prepared = prepare("fairvlmed", tmp_path / "data", "data_00000.npz")
    out = tmp_path / "run"
    out.mkdir()
    medical = source_fixture()
    base = SimpleNamespace(
        prepared=prepared,
        manifest=prepared.manifest,
        task=load_intent("advanced_glaucoma", REGISTRY)[0],
        output=out,
        sources=SimpleNamespace(initial=medical, read=lambda: medical),
    )
    # This scope tests original active owner persistence; Memory/survey execute later.
    profile = load(PROFILE)
    install(base, profile, defer_native=True)
    frozen = (out / "active-plan.json").read_bytes()
    install(base, profile, defer_native=True, read_only=True)
    assert (out / "active-plan.json").read_bytes() == frozen
    del medical["aao_signs"]
    with pytest.raises(ValueError, match="active_source_or_requirement_changed"):
        base.active_owner.check()


@pytest.mark.parametrize("bad", ["duplicate", "native", "old_version", "unknown_version"])
def test_invalid_variant_or_catalog_version_rejected(tmp_path, bad):
    from packages.multimodal_research.requirement_catalog import read

    value = json.loads(Path(CATALOG).read_bytes())
    check = next(c for c in value["checks"] if c["id"] == "rim_focal")
    if bad == "duplicate":
        check["input_variants"].append(check["input_variants"][0])
    elif bad == "native":
        check["method"] = "disc_cup_measurement"
    elif bad == "old_version":
        value["schema_version"] = "research-requirement-catalog/1.0.0"
    else:
        value["schema_version"] = "research-requirement-catalog/99.0.0"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read(dict(path=str(path), checksum=digest(value)))


def test_new_default_is_generic_and_old_explicit_profile_unchanged():
    from packages.multimodal_research.reuse import CAPABILITY_PROFILE

    assert CAPABILITY_PROFILE == PROFILE
    assert (
        len(load(PROFILE)["checks"])
        == len(load("configs/research/multimodal_catalog_acquisition_v5.json")["checks"])
        == 18
    )
    assert all(
        "input_variants" not in c
        for c in load("configs/research/multimodal_catalog_acquisition_v5.json")["checks"]
    )

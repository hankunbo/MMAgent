"""Current public profile: semantics depend on capabilities, never dataset labels.

The metamorphic inputs are synthetic; their altered provenance is not used to
register any real case or authorize execution.
"""

from dataclasses import replace

import pytest

from apps.vectorengine_case_materials import ref
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.llm_provider.models import Source
from packages.multimodal_research.active_requirements import load, project
from tests.dataset_adapters.test_input_capabilities import projected, reseal
from tests.dataset_adapters.test_multimodal import gamma, gdp
from tests.evaluation_preparation.test_active_acquisition import source_fixture

PROFILE = "configs/research/multimodal_catalog_acquisition_v9.json"


def selection(prepared, *, sources=None):
    profile = load(PROFILE)
    if sources is None:
        sources = source_fixture()
        for key in {c["source"] for c in profile["checks"]}:
            text = "\n".join(c["quote"] for c in profile["checks"] if c["source"] == key)
            sources[key] = Source(
                reference=ref("source", "SOURCE:" + key.upper(), digest(text)),
                locator="Synthetic exact current-catalog passage",
                purpose="research_material",
                state="research_only",
                text=text,
            )
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v2.json")[0]
    return project(prepared.manifest, sources, task, profile, capabilities=projected(prepared))


@pytest.mark.parametrize("kind", ["color_oct", "thickness_map", "slo"])
def test_current_catalog_ignores_synthetic_dataset_alias_but_retains_identity(tmp_path, kind):
    if kind == "color_oct":
        prepared = prepare("gamma", gamma(tmp_path), "0001")
    elif kind == "thickness_map":
        prepared = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    else:
        # A synthetic JPEG SLO component tests semantics without falsifying the
        # FairVLMed adapter's real dataset/version/PNG identity contract.
        prepared = prepare("gamma", gamma(tmp_path), "0001")
        components = tuple(
            c.model_copy(update={"modality": "SLO"}) if c.kind == "image" else c
            for c in prepared.manifest.components
        )
        prepared = replace(
            prepared, manifest=reseal(CaseComponents, prepared.manifest, components=components)
        )
    dataset, split, task = (
        ("gamma", "training", "glaucoma_grading")
        if prepared.manifest.dataset == "harvard_gdp"
        else ("harvard_gdp", "development_preparation", "glaucoma_detection")
    )
    other = replace(
        prepared,
        manifest=reseal(
            CaseComponents,
            prepared.manifest,
            dataset=dataset,
            sample="synthetic_identity_only",
            selection=split,
            task=task,
        ),
    )
    rows, trace = selection(prepared)
    alias_rows, alias_trace = selection(other)
    assert trace["input_checksum"] != alias_trace["input_checksum"]
    assert trace["capability_checksum"] != alias_trace["capability_checksum"]
    assert rows == alias_rows
    assert trace["rows"] == alias_trace["rows"]
    assert rows and all(
        r["capability_selection"]["execution_authorized"] is False
        for r in rows
        if "capability_selection" in r
    )


def test_current_missing_photo_changes_dispatch_without_erasing_questions(tmp_path):
    full = prepare("gamma", gamma(tmp_path / "full"), "0001")
    missing = prepare("gamma", gamma(tmp_path / "missing", photo=False), "0001")
    _, first = selection(full)
    _, second = selection(missing)
    a = {r["id"]: r for r in first["rows"]}
    b = {r["id"]: r for r in second["rows"]}
    assert set(a) == set(b)
    assert a["active:rim_diffuse"]["state"] == "scheduled"
    assert b["active:rim_diffuse"]["state"] == "input_gap"
    assert b["active:oct_rnfl"]["state"] == a["active:oct_rnfl"]["state"] == "scheduled"


def test_current_map_is_not_raw_oct_even_with_same_modality_label(tmp_path):
    prepared = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    components = tuple(
        c.model_copy(update={"modality": "OCT"}) if c.kind == "map" else c
        for c in prepared.manifest.components
    )
    prepared = replace(
        prepared, manifest=reseal(CaseComponents, prepared.manifest, components=components)
    )
    rows, trace = selection(prepared)
    raw = [r for r in trace["rows"] if r["id"].startswith("active:oct_")]
    assert raw and all(r["state"] == "input_gap" for r in raw)
    assert any(r["acquisition_method"] == "source_measurement" for r in rows)

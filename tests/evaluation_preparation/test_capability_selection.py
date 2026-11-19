"""Actual input -> original active owner -> scheduler; no external inference."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.multimodal_research.active_acquisition import install, structure
from packages.multimodal_research.active_requirements import load, project
from packages.multimodal_research.campaign import MultimodalCampaign
from tests.dataset_adapters.test_input_capabilities import projected, reseal
from tests.dataset_adapters.test_multimodal import gamma, gdp
from tests.evaluation_preparation.test_active_acquisition import source_fixture

# Keep this suite bound to the historical representation-only profile.
DEFAULT_PROFILE = "configs/research/multimodal_catalog_acquisition_v1.json"


def task():
    return load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]


def planned(p, profile=None, medical=None):
    return project(
        p.manifest,
        source_fixture() if medical is None else medical,
        task(),
        profile or load(DEFAULT_PROFILE),
        capabilities=projected(p),
    )


def test_original_questions_recalled_without_dataset_selector_or_assumed_visibility(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    rows, trace = planned(p)
    assert len(rows) == 18
    assert len({r["id"] for r in rows}) == 18
    assert len({r["evidence_unit"] for r in rows}) < len(rows)
    assert sum(r["state"] == "scheduled" for r in trace["rows"]) == 16
    assert all(
        r["capability_selection"]["state"] == "conditional"
        for r in trace["rows"]
        if r["state"] == "scheduled"
    )
    assert all(r["capability_selection"]["execution_authorized"] is False for r in rows)
    assert trace["schema_version"] == "source-bound-acquisition-plan/1.2.0"
    assert trace["capability_checksum"] == projected(p).checksum


def test_synthetic_source_name_change_keeps_selection_but_not_identity(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    other = replace(
        p,
        manifest=reseal(
            CaseComponents,
            p.manifest,
            dataset="harvard_gdp",
            task="glaucoma_detection",
            selection="development_preparation",
            sample="synthetic_alias",
        ),
    )
    first, a = planned(p)
    second, b = planned(other)
    assert a["input_checksum"] != b["input_checksum"]
    assert a["capability_checksum"] != b["capability_checksum"]
    assert first == second and a["rows"] == b["rows"]


def test_same_dataset_missing_photo_keeps_medical_demands_and_only_schedules_oct(tmp_path):
    p = prepare("gamma", gamma(tmp_path, photo=False), "0001")
    rows, trace = planned(p)
    assert len(rows) == 18
    scheduled = {r["id"] for r in trace["rows"] if r["state"] == "scheduled"}
    assert scheduled == {
        "active:oct_" + n for n in ("rnfl", "ganglion", "continuity", "alternative")
    }
    base = SimpleNamespace(active_owner=True, ledger=rows, observations={})
    assert (
        set(MultimodalCampaign.select_targets(base, {c.id for c in p.manifest.components}))
        == scheduled
    )


def test_map_named_oct_is_not_raw_oct_and_numeric_data_are_retained(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    components = tuple(
        c.model_copy(update={"modality": "OCT"}) if c.kind == "map" else c
        for c in p.manifest.components
    )
    p = replace(p, manifest=reseal(CaseComponents, p.manifest, components=components))
    rows, trace = planned(p)
    oct_rows = [r for r in trace["rows"] if r["id"].startswith("active:oct_")]
    assert oct_rows and all(r["state"] == "input_gap" for r in oct_rows)
    assert all(
        r["capability_selection"]["excluded_components"][0]["reason"] == "representation_mismatch"
        for r in oct_rows
    )
    assert any(r["acquisition_method"] == "source_measurement" for r in rows)


def test_current_source_unavailable_is_definition_backlog_not_eligible(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    medical = source_fixture()
    del medical["aao_signs"]
    rows, trace = planned(p, medical=medical)
    assert any(r["state"] == "definition_backlog" for r in trace["rows"])
    assert not any(r["id"] == "active:cup_geometry" for r in rows)


def test_original_install_scheduler_readback_and_report_consume_selected_ledger(tmp_path):
    p = prepare("gamma", gamma(tmp_path / "source"), "0001")
    directory = tmp_path / "run"
    directory.mkdir()
    sources = source_fixture()
    base = SimpleNamespace(
        prepared=p,
        manifest=p.manifest,
        task=task(),
        output=directory,
        sources=SimpleNamespace(initial=sources, read=lambda: sources),
    )
    install(base, load(DEFAULT_PROFILE), defer_native=True)
    assert len(MultimodalCampaign.select_targets(base, {c.id for c in p.manifest.components})) == 16
    initial = (directory / "active-plan.json").read_bytes()
    cap = (directory / "input-capabilities.json").read_bytes()
    # Cold assembly reads immutable plans without discovery or native execution.
    install(base, load(DEFAULT_PROFILE), defer_native=True, read_only=True)
    assert (directory / "active-plan.json").read_bytes() == initial
    assert (directory / "input-capabilities.json").read_bytes() == cap
    report = dict(
        requirements=[dict(id=r["id"]) for r in base.ledger], report_production={}, scores={}
    )
    tree = structure(report, base)
    assert all("capability_selection" in r["source_requirement"] for r in tree["requirements"])
    del sources["aao_signs"]
    with pytest.raises(ValueError, match="active_source_or_requirement_changed"):
        base.active_owner.check()


@pytest.mark.parametrize("change", ["inline", "catalog", "projected", "capabilities"])
def test_frozen_catalog_and_original_input_binding_cannot_be_bypassed(tmp_path, change):
    p = prepare("gamma", gamma(tmp_path / "data"), "0001")
    profile = load(DEFAULT_PROFILE)
    if change == "inline":
        raw = json.loads(Path(DEFAULT_PROFILE).read_bytes())
        raw["checks"] = profile["checks"]
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="must_not_override"):
            load(path)
    elif change == "catalog":
        path = tmp_path / "catalog.json"
        path.write_text(Path(profile["requirement_catalog"]["path"]).read_text() + " ")
        value = json.loads(path.read_bytes())
        value["checks"][0]["instruction"] += "changed"
        path.write_text(json.dumps(value))
        profile["requirement_catalog"]["path"] = str(path)
        with pytest.raises(ValueError, match="catalog_changed"):
            planned(p, profile)
    elif change == "projected":
        profile["checks"][0]["input_capabilities"]["representations"] = ["map"]
        with pytest.raises(ValueError, match="catalog_projection_changed"):
            planned(p, profile)
    else:
        with pytest.raises(ValueError, match="original_input_capabilities_required"):
            project(p.manifest, source_fixture(), task(), profile)


def test_equivalent_questions_merge_but_related_questions_keep_shared_unit(tmp_path):
    profile = json.loads(Path(DEFAULT_PROFILE).read_bytes())
    catalog = json.loads(Path(profile["requirement_catalog"]["path"]).read_bytes())
    duplicate = dict(catalog["checks"][0], id="geometry_alias", equivalent_to="cup_geometry")
    catalog["checks"].append(duplicate)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    profile["requirement_catalog"] = dict(path=str(path), checksum=digest(catalog))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    p = prepare("gamma", gamma(tmp_path / "data"), "0001")
    rows, trace = planned(p, load(profile_path))
    assert len(rows) == 18
    assert (
        next(r for r in trace["rows"] if r["id"] == "geometry_alias")["state"]
        == "equivalent_merged"
    )
    assert {r["id"] for r in rows if r["evidence_unit"] == "cup_geometry"} == {
        "active:cup_geometry",
        "active:cup_size",
    }
    catalog["checks"][-1]["instruction"] += "different medical question"
    path.write_text(json.dumps(catalog))
    profile["requirement_catalog"]["checksum"] = digest(catalog)
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="not_same_question"):
        load(profile_path)


def test_legacy_profile_does_not_silently_enable_capability_selection(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    profile = load("configs/research/multimodal_active_requirements_v1.json")
    rows, trace = project(p.manifest, source_fixture(), task(), profile)
    assert len(rows) == 18
    assert trace["schema_version"] == "source-bound-acquisition-plan/1.0.0"
    assert "capability_matching" not in trace
    assert all("capability_selection" not in r for r in rows)


def test_automatic_entry_rejects_manual_override_before_any_owner_or_api(tmp_path):
    from packages.multimodal_research.entry import prepare_entry

    with pytest.raises(ValueError, match="conflicts_with_manual_profile"):
        prepare_entry(
            dataset="gamma",
            root=tmp_path,
            sample="0001",
            task="advanced_glaucoma",
            output=tmp_path / "out",
            automatic_requirements=True,
            active_acquisition=DEFAULT_PROFILE,
        )
    assert not (tmp_path / "out").exists()


def test_catalog_schema_snapshot_and_execution_policy_target():
    from packages.multimodal_research.requirement_catalog import RequirementCatalog

    expected = json.dumps(RequirementCatalog.model_json_schema(), sort_keys=True, indent=2) + "\n"
    assert (
        Path("packages/multimodal_research/schemas/RequirementCatalog.json").read_text() == expected
    )
    # Catalog target membership does not silently expand a frozen search/run policy.
    profile = load(DEFAULT_PROFILE)
    assert profile["goals"] == [profile["definition_expansion"]["public_target"]]


@pytest.mark.parametrize("changed", [False, True])
def test_public_preview_uses_current_sources_and_never_overwrites(tmp_path, monkeypatch, changed):
    import sys

    from apps.multimodal_case import main
    from packages.dataset_adapters.multimodal.preparation import persist
    from packages.multimodal_research import material

    p = prepare("gamma", gamma(tmp_path / "data"), "0001")
    saved = persist(p, tmp_path / "input")
    sources = source_fixture()
    reads = []

    def current():
        reads.append(True)
        return {} if changed else sources

    monkeypatch.setattr(
        material, "MedicalSources", lambda: SimpleNamespace(initial=sources, read=current)
    )
    out = tmp_path / "selected.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "requirements-plan",
            "--input",
            str(saved),
            "--task",
            "advanced_glaucoma",
            "--output",
            str(out),
        ],
    )
    if changed:
        with pytest.raises(ValueError, match="preview_source_changed"):
            main()
        assert not out.exists()
    else:
        assert main() == 0
        value = json.loads(out.read_bytes())
        assert len(value["requirements"]) == 18
        assert value["execution_authorized"] is False
        assert value["api_requests"] == value["database_writes"] == 0
        checksum = value.pop("checksum")
        assert checksum == digest(value)
        with pytest.raises(ValueError, match="new_output_required"):
            main()
    assert len(reads) == 1

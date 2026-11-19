"""Actual fresh preparation/preflight with versioned root method reuse."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from packages.multimodal_research import entry
from tests.dataset_adapters.test_multimodal import gdp, grape

PROFILE = "configs/research/multimodal_catalog_acquisition_v6.json"


def prepare(tmp_path, *, reuse=True, automatic=True, profile=None, dataset="harvard_gdp"):
    root = (gdp if dataset == "harvard_gdp" else grape)(tmp_path / "data")
    out = tmp_path / "run"
    entry.prepare_entry(
        dataset=dataset,
        root=root,
        sample="data_0001.npz" if dataset == "harvard_gdp" else "1:OD",
        cutoff_visit=1 if dataset == "grape" else None,
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        automatic_requirements=automatic,
        active_acquisition=profile,
        reuse_procedures=reuse,
        resilient_vision=True,
        view_profile="detail-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )
    return out, json.loads((out / "entry.json").read_bytes())


@pytest.mark.parametrize("dataset", ["harvard_gdp", "grape"])
def test_original_auto_prepare_selects_capability_profile_and_preflight(
    tmp_path, monkeypatch, dataset
):
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("preflight accessed database"))
    out, spec = prepare(tmp_path, dataset=dataset)
    assert Path(spec["active_acquisition"]["path"]).name == "automatic-requirements.json"
    assert spec["automatic_requirement_selection"]["target"] == "GOAL:ADVANCED_GLAUCOMA"
    assert (
        Path(spec["automatic_requirement_selection"]["template"]).name
        == "multimodal_catalog_acquisition_v10.json"
    )
    proof = spec["procedure_reuse_contract"]
    assert proof["schema_version"] == "root-input-method-reuse/1.0.0"
    assert proof["task_checksum"] == spec["intent"]["checksum"]
    assert proof["input_checksum"] == spec["input_checksum"]
    assert proof["policy"] == "proposition-input-method-reuse/1.0.0"
    assert spec["schema_version"] == "multimodal-live-entry/1.26.0"
    assert spec["max_requests"] == 40 and spec["seconds"] == 5400
    assert "dataset" not in proof
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    before = (out / "entry.json").read_bytes()
    assert entry.assemble_entry(out, preflight_only=True)["api_requests"] == 0
    assert (out / "entry.json").read_bytes() == before
    for change in ("missing", "version", "task", "profile", "disabled"):
        bad = deepcopy(spec)
        if change == "missing":
            bad.pop("procedure_reuse_contract")
        elif change == "disabled":
            bad["reuse_procedures"] = False
        else:
            key = {
                "version": "schema_version",
                "task": "task_checksum",
                "profile": "profile_checksum",
            }[change]
            bad["procedure_reuse_contract"][key] = "wrong"
        (out / "entry.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="root_question_entry_binding|root_method_reuse"):
            entry.assemble_entry(out, preflight_only=True)
    (out / "entry.json").write_bytes(before)


def test_default_root_without_reuse_still_uses_current_input_survey(tmp_path):
    out, spec = prepare(tmp_path, reuse=False)
    assert Path(spec["active_acquisition"]["path"]).name == "automatic-requirements.json"
    assert (
        Path(spec["automatic_requirement_selection"]["template"]).name
        == "multimodal_catalog_acquisition_v10.json"
    )
    assert "procedure_reuse_contract" not in spec
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"


def test_explicit_previous_capability_profile_remains_exact(tmp_path):
    out, spec = prepare(
        tmp_path,
        automatic=False,
        profile="configs/research/multimodal_catalog_acquisition_v5.json",
    )
    assert Path(spec["active_acquisition"]["path"]).name == "multimodal_catalog_acquisition_v5.json"
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"


def test_root_reuse_requires_actual_capability_profile_before_persistence(tmp_path):
    with pytest.raises(ValueError, match="root_method_reuse_capability_profile"):
        prepare(
            tmp_path,
            automatic=False,
            profile="configs/research/multimodal_catalog_acquisition_v2.json",
        )
    assert not (tmp_path / "run").exists()


def test_old_explicit_v6_still_has_exact_profile_and_no_new_policy(tmp_path):
    out, spec = prepare(tmp_path, automatic=False, profile=PROFILE)
    assert "automatic_requirement_selection" not in spec
    assert Path(spec["active_acquisition"]["path"]).name == Path(PROFILE).name
    assert entry.assemble_entry(out, preflight_only=True)["api_requests"] == 0


def test_explicit_historical_v8_remains_exact(tmp_path):
    out, spec = prepare(
        tmp_path,
        automatic=False,
        profile="configs/research/multimodal_catalog_acquisition_v8.json",
    )
    assert "automatic_requirement_selection" not in spec
    assert Path(spec["active_acquisition"]["path"]).name == "multimodal_catalog_acquisition_v8.json"
    original = (out / "entry.json").read_bytes()
    assert entry.assemble_entry(out, preflight_only=True)["api_requests"] == 0
    assert (out / "entry.json").read_bytes() == original


def test_saved_automatic_v8_template_stays_readable_after_default_changes(tmp_path, monkeypatch):
    from packages.multimodal_research import capability_window

    with monkeypatch.context() as previous_default:
        previous_default.setattr(
            capability_window,
            "CORRECTED_SCOPE_PROFILE",
            "configs/research/multimodal_catalog_acquisition_v8.json",
        )
        out, spec = prepare(tmp_path)
    raw = (out / "entry.json").read_bytes()
    assert (
        Path(spec["automatic_requirement_selection"]["template"]).name
        == "multimodal_catalog_acquisition_v8.json"
    )
    assert entry.assemble_entry(out, preflight_only=True)["api_requests"] == 0
    assert (out / "entry.json").read_bytes() == raw

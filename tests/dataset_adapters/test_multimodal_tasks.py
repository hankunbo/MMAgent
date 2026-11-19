"""Task selection is explicit, not a hidden dataset-to-classifier dispatch."""

import json

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.preparation import persist
from packages.dataset_adapters.multimodal.tasks import (
    Registry,
    TaskProjection,
    load_intent,
    project,
    write_projection,
)
from packages.disease_packs.models import seal
from tests.dataset_adapters.test_multimodal import gamma, gdp, grape


def test_same_input_can_select_different_task_without_mutation(tmp_path):
    source = prepare("gamma", gamma(tmp_path), "0001").manifest
    before = source.model_dump_json()
    binary = project(source, "glaucoma_detection")
    ordinal = project(source, "glaucoma_grading")
    assert binary.task.output_kind == "binary_evidence"
    assert ordinal.task.output_kind == "ordinal_evidence"
    assert len(ordinal.task.ordered_labels) == 3
    assert binary.selected_components == ordinal.selected_components == ("fundus", "oct")
    assert source.model_dump_json() == before
    assert binary.required_classifier is None and binary.score is None
    assert binary.execution_authorized is False


def test_missing_components_retained_zero_and_negative_untouched(tmp_path):
    source = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz").manifest
    result = project(source, "glaucoma_detection")
    assert all(c.selected for c in result.components)
    missing = next(c for c in result.components if c.input_status == "missing")
    assert missing.disposition == "input_missing" and missing.next_step == "retain_gap"
    assert next(c for c in source.components if c.concept == "age").value == 0
    assert next(c for c in source.components if c.concept == "md").value == -2.5
    assert "VF_label_derivation_must_be_audited_before_evaluation" in result.task_gaps


def test_observed_change_uses_visible_visits_and_cutoff(tmp_path):
    root = grape(tmp_path)
    baseline = prepare("grape", root, "1:OD", cutoff_visit=0).manifest
    later = prepare("grape", root, "1:OD", cutoff_visit=1).manifest
    first, second = project(baseline, "observed_change"), project(later, "observed_change")
    assert "fewer_than_two_visible_visits" in first.task_gaps
    assert "fewer_than_two_visible_visits" not in second.task_gaps
    assert second.cutoff_visit == 1
    assert "98765" not in second.model_dump_json() and "SECRET" not in second.model_dump_json()


def test_unknown_and_duplicate_profiles_rejected(tmp_path):
    from packages.dataset_adapters.multimodal.tasks import DEFAULT_TASKS

    with pytest.raises(ValueError, match="not_registered"):
        load_intent("unknown")
    values = json.loads(DEFAULT_TASKS.read_bytes())
    values["profiles"].append(values["profiles"][0])
    with pytest.raises(ValueError, match="duplicate"):
        Registry.model_validate_json(json.dumps(values))


def test_no_matching_modality_is_explicit_gap_not_case_failure(tmp_path):
    from packages.dataset_adapters.multimodal.tasks import DEFAULT_TASKS

    values = json.loads(DEFAULT_TASKS.read_bytes())
    for profile in values["profiles"]:
        profile["allowed_modalities"] = ["another_modality"]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(values))
    source = prepare("harvard_gdp", gdp(tmp_path / "input"), "data_0001.npz").manifest
    result = project(source, "glaucoma_detection", path)
    assert result.selected_components == ()
    assert "no_input_component_for_task" in result.task_gaps
    assert all(c.disposition == "excluded" for c in result.components)


def test_projection_checks_selection_and_does_not_overwrite_export(tmp_path):
    source = prepare("harvard_gdp", gdp(tmp_path / "input"), "data_0001.npz")
    path = persist(source, tmp_path / "prepared")
    out = tmp_path / "task.json"
    value = write_projection(path, "glaucoma_detection", out)
    assert TaskProjection.model_validate_json(out.read_bytes()).checksum == value["checksum"]
    with pytest.raises(FileExistsError):
        write_projection(path, "glaucoma_grading", out)
    value.pop("checksum")
    value["selected_components"] = []
    with pytest.raises(ValueError, match="task_projection_components"):
        seal(TaskProjection, **value)

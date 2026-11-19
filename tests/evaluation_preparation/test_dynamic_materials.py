from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.dynamic_materials import DynamicCaseEvidence


def test_no_segmentation_files_are_required(tmp_path):
    owner = DynamicCaseEvidence.__new__(DynamicCaseEvidence)
    owner.load_measurements(tmp_path, object())
    assert owner.measurement is None and owner.tool_assets == ()
    assert not owner.quality["bbox_cdr_usable"]


def test_partial_saved_receipt_is_technical_not_invented_missing(tmp_path):
    (tmp_path / "measurements.json").touch()
    owner = DynamicCaseEvidence.__new__(DynamicCaseEvidence)
    with pytest.raises(ValueError, match="dynamic_partial_tool_receipt"):
        owner.load_measurements(tmp_path, object())


def test_structure_is_source_index_not_old_case_template(monkeypatch):
    import packages.evaluation_preparation.case_materials as legacy
    import packages.evaluation_preparation.dynamic_materials as dynamic

    monkeypatch.setattr(legacy, "template", lambda: pytest.fail("old template must not be opened"))
    monkeypatch.setattr(
        dynamic,
        "load_catalog",
        lambda: dict(
            tasks={
                "association:NEW": dict(selected_source_ids=["source"]),
                "criterion:UNBOUND": dict(selected_source_ids=[]),
            }
        ),
    )
    result = DynamicCaseEvidence.build_structure(SimpleNamespace())
    assert "association:NEW" in result["requirement_ids"]
    assert "criterion:UNBOUND" not in result["requirement_ids"]
    assert all("model" not in key for key in result["requirement_ids"])

import json

import pytest

from packages.dataset_adapters.reading import AdapterError
from packages.evaluation_preparation.models import EvaluationPreparation
from packages.evaluation_preparation.service import (
    ROOT,
    build,
    inventory,
    probe_adapter,
    split_overlap,
)
from tests.dataset_adapters.support import fair, odir


def test_preparation_never_unlocks_or_asserts_independence():
    value = json.loads((ROOT / "configs/evaluation/preparation_v1.json").read_text())
    for key in ("final_test_unlocked", "selection_uses_labels", "patient_independence_proven"):
        with pytest.raises(ValueError):
            EvaluationPreparation.model_validate_json(json.dumps({**value, key: True}))
    assert EvaluationPreparation.model_validate_json(json.dumps(value)).decisions_pending


def test_fair_original_owner_probe_no_hidden_values(tmp_path):
    adapter, _ = fair(tmp_path)
    result = probe_adapter(adapter)
    assert result["patient_group"] is None
    assert result["images"][0]["modality"] == "SLO"
    assert result["prepared_fingerprint"]
    assert "SECRET" not in json.dumps(result)
    assert "slo_fundus" in result["observed_field_names"]
    with pytest.raises(AdapterError, match="validation_test_locked"):
        adapter.prepare("data_00000.npz", split="test")


def test_odir_original_owner_preserves_two_eyes(tmp_path):
    adapter, _ = odir(tmp_path)
    result = probe_adapter(adapter)
    assert result["identity_basis"] == "dataset_local_explicit_id"
    assert [i["laterality"] for i in result["images"]] == ["LEFT", "RIGHT"]
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    "groups",
    [
        {"development": ["x"], "test": ["x"]},
        {"development": [None]},
        {"test": ["unknown"]},
    ],
)
def test_invalid_patient_split(groups):
    with pytest.raises(ValueError):
        split_overlap(groups)


def test_valid_explicit_group_split():
    split_overlap({"development": ["patient_a", "patient_a"], "test": ["patient_b"]})


def test_inventory_no_content_read_and_rejects_links(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"not decoded in metadata inventory")
    result = inventory(tmp_path, {"train": ("*.jpg",)})
    assert len(result["train"]) == 1 and "checksum" not in result["train"][0]
    (tmp_path / "b.jpg").symlink_to(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="member_unsafe"):
        inventory(tmp_path, {"train": ("*.jpg",)})


def test_missing_roots_quarantines_not_fake_ready(tmp_path):
    registry, layout = build({"MMAGENT_GAMMA_ROOT": str(tmp_path)})
    assert not layout
    assert next(e for e in registry.entries if e["dataset"] == "gamma")["status"] == "quarantine"
    assert not registry.preparation.final_test_unlocked
    assert registry.runtime_writes == 0


def test_empty_file_rejected(tmp_path):
    (tmp_path / "a.jpg").touch()
    with pytest.raises(ValueError, match="member_empty"):
        inventory(tmp_path, {"train": ("*.jpg",)})

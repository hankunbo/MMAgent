"""Isolated proposal: original registered bytes + byte adapter parity and cleanup."""

import json
import os
from pathlib import Path

import pytest

from packages.object_store import checksum
from tests.tool_execution.test_fundus_bundle import material as material
from tests.tool_execution.test_geometry_qc import mask_png


@pytest.fixture
def adapter(monkeypatch):
    from packages.multimodal_research import fundus_mask_adapter as value
    from packages.multimodal_research import fundus_measurements as old

    path = Path(old.__file__).resolve()
    lock = json.loads(Path("configs/tools/automorph-fundus-v1.json").read_bytes())
    assert checksum(path.read_bytes()) == lock["resources"][str(path)]
    monkeypatch.setattr(value, "measure", old.measure)
    return value, old


@pytest.mark.parametrize("kind", ["normal", "empty"])
def test_original_outputs_preserved_and_private_temporaries_removed(
    adapter, material, tmp_path, kind
):
    value, old = adapter
    raw = (
        material.masks
        if kind == "normal"
        else dict.fromkeys(material.masks, mask_png((512, 512), bytes(512 * 512)))
    )
    for name, data in raw.items():
        (tmp_path / (name + ".png")).write_bytes(data)
    seen = []

    def measured(path):
        seen.append(path)
        assert os.stat(path).st_mode & 0o777 == 0o700
        assert all(os.stat(p).st_mode & 0o777 == 0o600 for p in path.iterdir())
        return old.measure(path)

    value.measure = measured
    assert value.measure_masks(raw) == old.measure(tmp_path)
    assert seen and not seen[0].exists()


def test_original_failure_kept_and_temporary_removed(adapter, material):
    value, _ = adapter
    seen = []

    def fail(path):
        seen.append(path)
        raise ValueError("original_measurement_failure")

    value.measure = fail
    with pytest.raises(ValueError, match="original_measurement_failure"):
        value.measure_masks(material.masks)
    assert seen and not seen[0].exists()


@pytest.mark.parametrize("mode", ["missing", "extra", "not_bytes", "too_large"])
def test_invalid_mask_input_is_not_repaired(adapter, material, mode):
    value, _ = adapter
    raw = dict(material.masks)
    if mode == "missing":
        raw.pop("cup")
    elif mode == "extra":
        raw["arbitrary"] = b""
    elif mode == "not_bytes":
        raw["cup"] = "not bytes"
    else:
        raw["cup"] = b"X" * (4 * 1024 * 1024 + 1)
    value.measure = lambda _: pytest.fail("invalid masks reached owner")
    with pytest.raises(ValueError, match="fundus_mask_"):
        value.measure_masks(raw)

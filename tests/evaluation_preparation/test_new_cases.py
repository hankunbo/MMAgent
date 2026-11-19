import json

import pytest

from packages.evaluation_preparation.new_cases import fair_case, prepare, select_fair
from tests.dataset_adapters.support import fair
from tests.dataset_adapters.test_papila import dataset


def test_fair_no_note_or_label_in_visible(tmp_path):
    adapter, _ = fair(tmp_path, count=3)
    names = select_fair(adapter, excluded={"data_00000.npz"})
    assert names == ["data_00001.npz", "data_00002.npz"]
    visible, hidden, image = fair_case(adapter, names[0])
    assert image.startswith(b"\x89PNG") and hidden["labels"]
    assert all(token not in json.dumps(visible) for token in ("SECRET", "glaucoma", "note", "gpt4"))
    assert visible["patient_group"] is None and not visible["runtime_admission"]


@pytest.mark.parametrize("count", [0, 11, True])
def test_count_bound(tmp_path, count):
    adapter, _ = fair(tmp_path)
    with pytest.raises(ValueError):
        select_fair(adapter, excluded={}, count=count)


def test_preparation_real_owner_separation(tmp_path):
    fair_root, papila_root = tmp_path / "fair", tmp_path / "papila"
    fair_root.mkdir()
    papila_root.mkdir()
    fair(fair_root, count=4)
    dataset(papila_root)
    output = tmp_path / "result"
    summary = prepare(fair_root, papila_root, output)
    assert summary["papila_patients"] == 2 and summary["papila_eyes"] == 4
    assert summary["api_requests"] == summary["runtime_writes"] == 0
    protocol = json.loads((output / "protocol.json").read_text())
    assert len(protocol["entries"]) == 6
    assert len(list((output / "visible").glob("papila*.json"))) == 8
    assert not any("RET002" in p.name for p in (output / "visible").iterdir())
    for path in (output / "visible").glob("*.json"):
        content = path.read_text()
        assert "diagnosis" not in content.lower() and "SECRET" not in content
    assert (output / "restricted").stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        prepare(fair_root, papila_root, output)

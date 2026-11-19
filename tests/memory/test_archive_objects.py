from types import SimpleNamespace

import pytest

from packages.memory.archive_objects import ArchiveObjectReader
from packages.object_store import LocalObjectStore, checksum


def test_visible_output_and_missing_only_no_evaluator(tmp_path):
    v, o = [LocalObjectStore(tmp_path / n) for n in ("visible", "output")]
    reader = ArchiveObjectReader(v, o)
    for store, raw in ((v, b"input"), (o, b"tool-output")):
        uri = store.promote(store.stage(raw), checksum(raw))
        assert reader.read(SimpleNamespace(uri=uri)) == raw
    with pytest.raises(ValueError):
        reader.read(SimpleNamespace(uri="file:///etc/passwd"))
    with pytest.raises(ValueError, match="separation"):
        ArchiveObjectReader(v, v)


def test_collision_is_not_silently_resolved(tmp_path):
    from unittest.mock import Mock

    v, o = Mock(root="visible"), Mock(root="output")
    v._path.return_value.exists.return_value = True
    o._path.return_value.exists.return_value = True
    with pytest.raises(ValueError, match="collision"):
        ArchiveObjectReader(v, o).read(SimpleNamespace(uri="object://x"))
    v.read_bounded.assert_not_called()
    o.read_bounded.assert_not_called()

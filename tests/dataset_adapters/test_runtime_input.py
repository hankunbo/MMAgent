"""Generic selection / optional-input contracts; synthetic data only."""

import io
import json
import struct
import zipfile

import pytest

from packages.dataset_adapters.prepared import capture
from packages.dataset_adapters.reading import AdapterError
from tests.dataset_adapters.support import csv_write, fair, npy


def runtime_adapter(old):
    from packages.dataset_adapters.runtime_input import FairVLMedRuntimeAdapter

    return FairVLMedRuntimeAdapter(
        old.root, old.catalog, old.dataset.reference(), old.pack.reference()
    )


def replace_npz(path, *, missing=(), extra=None):
    source = path.read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as old, zipfile.ZipFile(output, "w") as new:
        for name in old.namelist():
            if name[:-4] not in missing:
                new.writestr(name, old.read(name))
        if extra:
            new.writestr(*extra)
    path.write_bytes(output.getvalue())


def test_runtime_subset_does_not_require_other_images(tmp_path):
    old, _ = fair(tmp_path, count=2)
    (tmp_path / "Dataset/Training/data_00001.npz").unlink()
    with pytest.raises(AdapterError, match="missing_or_extra"):
        old.prepare("data_00000.npz")
    prepared = capture(runtime_adapter(old).prepare("data_00000.npz"))
    assert prepared.preparation_version == "prepared-input/3.0.0"
    assert prepared.images[0][1:] == ((2, 3), "UNKNOWN", "SLO")
    assert prepared.fields["selection_version"] == "training_selected_input/1.0.0"


@pytest.mark.parametrize(
    "missing",
    [
        ("age",),
        ("glaucoma",),
        ("age", "gender", "race", "ethnicity", "language", "maritalstatus", "glaucoma"),
    ],
)
def test_optional_npz_fields_survive_capture(tmp_path, missing):
    old, _ = fair(tmp_path)
    replace_npz(tmp_path / "Dataset/Training/data_00000.npz", missing=missing)
    with pytest.raises(AdapterError):
        old.prepare("data_00000.npz")
    snap = capture(runtime_adapter(old).prepare("data_00000.npz"))
    assert all(k not in snap.fields["npz"] for k in missing)


def test_minimal_source_index_no_note_or_labels(tmp_path):
    old, _ = fair(tmp_path)
    csv_write(
        tmp_path / "ReadMe/data_summary.csv",
        ("filename", "use"),
        [dict(filename="data_00000.npz", use="training")],
    )
    snap = capture(runtime_adapter(old).prepare("data_00000.npz"))
    assert snap.note is None
    assert set(snap.fields["csv"]) == {"filename", "use"}


def test_unknown_safe_member_is_quarantined(tmp_path):
    old, _ = fair(tmp_path)
    extra = npy((), "<i8", struct.pack("<q", 99123))
    replace_npz(tmp_path / "Dataset/Training/data_00000.npz", extra=("extra_value.npy", extra))
    snap = capture(runtime_adapter(old).prepare("data_00000.npz"))
    assert "extra_value" not in snap.fields["npz"]
    assert any(m["key"] == "extra_value" for m in snap.fields["npz_metadata"])
    assert "99123" not in json.dumps(snap.fields)


@pytest.mark.parametrize("kind", ["no_image", "pickle", "duplicate", "path"])
def test_unsafe_or_missing_necessary_data_rejected(tmp_path, kind):
    old, _ = fair(tmp_path)
    p = tmp_path / "Dataset/Training/data_00000.npz"
    if kind == "no_image":
        replace_npz(p, missing=("slo_fundus",))
    elif kind == "pickle":
        replace_npz(p, extra=("extra.npy", npy((), "|O8", b"0" * 8)))
    elif kind == "duplicate":
        replace_npz(p, extra=("age.npy", npy((), "<U2", "42".encode("utf-32-le"))))
    else:
        replace_npz(p, extra=("../extra.npy", npy((), "<i8", b"0" * 8)))
    with pytest.raises(AdapterError):
        runtime_adapter(old).prepare("data_00000.npz")


def test_capture_rejects_mutated_optional_values(tmp_path):
    old, _ = fair(tmp_path)
    sample = runtime_adapter(old).prepare("data_00000.npz")
    sample.fields["npz"]["age"] = "99"
    with pytest.raises(AdapterError, match="prepared_input_inconsistent"):
        capture(sample)


def test_holdout_and_path_still_locked(tmp_path):
    old, _ = fair(tmp_path)
    adapter = runtime_adapter(old)
    for name, split in [("data_00000.npz", "test"), ("../data_00000.npz", "training")]:
        with pytest.raises(AdapterError):
            adapter.prepare(name, split=split)


def test_runtime_snapshot_fingerprint_is_versioned_and_immutable(tmp_path):
    old, _ = fair(tmp_path)
    legacy = capture(old.prepare("data_00000.npz"))
    sample = runtime_adapter(old).prepare("data_00000.npz")
    snap = capture(sample)
    original = snap.fingerprint()
    sample.fields["npz"].clear()
    assert snap.fingerprint() == original != legacy.fingerprint()

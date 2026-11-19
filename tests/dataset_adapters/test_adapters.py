import json
from dataclasses import replace

import pytest

from packages.dataset_adapters.adapters import FAIR_FIELDS, ODIR_FIELDS
from packages.dataset_adapters.models import Projection
from packages.dataset_adapters.projection import project_note
from packages.dataset_adapters.reading import AdapterError, npy, read_npz, read_source
from packages.tool_execution.qc import grayscale_png
from tests.dataset_adapters.support import csv_write, fair, npz, odir
from tests.dataset_adapters.support import npy as make_npy


def test_alignment_and_lossless_pixels(tmp_path):
    a, _ = fair(tmp_path)
    s = a.prepare(a.candidates()[0])
    assert grayscale_png(s.images[0][0], 100) == ((2, 3), bytes(range(6)))
    assert s.images[0][2:] == ("UNKNOWN", "SLO")
    assert s.fields["csv"]["glaucoma"] == "1" and s.fields["npz"]["glaucoma"] == 1
    assert s.identity_basis == "unknown"
    assert project_note(s.note).visible_text == ""


@pytest.mark.parametrize("split", ["validation", "test", "Validation", "Test", "../training"])
def test_reserved_denied_before_io(tmp_path, split):
    a, _ = fair(tmp_path)
    with pytest.raises(AdapterError, match="locked"):
        a.prepare("bad", split=split)
    with pytest.raises(AdapterError, match="locked"):
        a.candidates(split=split)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing_npz", "extra_npz", "traversal", "missing_field"]
)
def test_alignment_failures(tmp_path, mutation):
    a, rows = fair(tmp_path)
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "missing_npz":
        (tmp_path / "Dataset/Training/data_00000.npz").unlink()
    elif mutation == "extra_npz":
        (tmp_path / "Dataset/Training/extra.npz").write_bytes(npz())
    elif mutation == "traversal":
        rows[0]["filename"] = "../secret.npz"
    fields = FAIR_FIELDS
    if mutation == "missing_field":
        fields = tuple(f for f in fields if f != "note")
        rows[0].pop("note")
    csv_write(tmp_path / "ReadMe/data_summary.csv", fields, rows)
    with pytest.raises(AdapterError):
        a.prepare("data_00000.npz")


@pytest.mark.parametrize(
    "payload",
    [
        b"not_zip",
        npz(replacement=make_npy((2, 3), "|O8", b"x" * 48)),
        npz(replacement=make_npy((999999, 999999), "|u1", b"")),
        npz(replacement=make_npy((2, 3), "|u1", b"1")),
        npz(extra=("../secret.npy", b"SECRET")),
        npz(extra=("unknown.npy", b"x")),
        npz(replacement=make_npy((2000, 2000), "|u1", b"\0" * 4000000)),
    ],
)
def test_npz_malicious_corrupt_bounded(payload):
    with pytest.raises(AdapterError, match="npz_corrupt_unsafe_or_schema_invalid") as e:
        read_npz(payload)
    assert "SECRET" not in str(e.value)


def test_npy_no_eval():
    with pytest.raises((ValueError, SyntaxError)):
        npy(b"\x93NUMPY\x01\x00" + b"\x06\x00" + b"open()")


def test_symlinks_and_bounds(tmp_path):
    (tmp_path / "real").write_bytes(b"1234")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    for name, limit in [("link", 10), ("real", 3), ("../outside", 10)]:
        with pytest.raises(AdapterError):
            read_source(tmp_path, name, limit)


@pytest.mark.parametrize(
    "note",
    [
        "No IOP OD current: 18 mmHg",
        "IOP OD possibly 18 mmHg",
        "IOP OD 18 / OS 22 mmHg",
        "IOP OD historical: 18 mmHg",
        "IOP OD current: 18 20 mmHg",
        "OCT scan mentioned; no pixels",
    ],
)
def test_ambiguous_observations_stay_unknown(note):
    p = project_note(note)
    assert not any(c.observation for c in p.candidates)
    assert all(c.laterality == "UNKNOWN" and c.timepoint == "unknown" for c in p.candidates)
    assert not p.visible_text


def test_mapping_and_controlled_candidate():
    note = "IOP OS current: 19.5 mmHg\r\n\nAssessment: SECRET\n"
    p = project_note(note)
    assert "".join(note[s.start : s.end] for s in p.spans) == note
    assert all(s.hidden and s.visible_start == s.visible_end == 0 for s in p.spans)
    c = p.candidates[0]
    assert c.observation.value == 19.5 and c.observation.unit == "mmHg"
    assert c.laterality == "LEFT" and c.timepoint == "current"
    data = json.loads(p.model_dump_json())
    data["spans"][0]["end"] -= 1
    with pytest.raises(ValueError):
        Projection.model_validate_json(json.dumps(data))


def test_version_checksum_changes_and_unknown_patient(tmp_path):
    a, rows = fair(tmp_path, count=2)
    one = a.prepare("data_00000.npz")
    two = a.prepare("data_00001.npz")
    assert (
        one.patient_hash == two.patient_hash
    )  # documented UNKNOWN sentinel, never a patient group
    with pytest.raises(AdapterError, match="prepared_input_inconsistent"):
        replace(one, fields=one.fields | {"changed": True}).fingerprint()
    rows[0]["note"] += "changed"
    csv_write(tmp_path / "ReadMe/data_summary.csv", FAIR_FIELDS, rows)
    assert a.prepare("data_00000.npz").fingerprint() != one.fingerprint()
    ref = a.dataset.reference().model_copy(update={"checksum": "sha256:" + "0" * 64})
    with pytest.raises(ValueError):
        a.catalog.dataset(ref)
    ref = a.dataset.reference().model_copy(update={"version": "9.0.0"})
    with pytest.raises(KeyError):
        a.catalog.dataset(ref)
    with pytest.raises(AdapterError):
        a.relate(a)


def test_odir_bilateral_scope_and_cross_dataset_rejection(tmp_path):
    a, rows = odir(tmp_path / "odir")
    s = a.prepare("11")
    assert [i[2] for i in s.images] == ["LEFT", "RIGHT"]
    assert len(s.fields["source_rows"]) == 2 and set(s.fields["patient_labels"]) == set("NDGCAHMO")
    b, _ = fair(tmp_path / "fair")
    with pytest.raises(ValueError):
        a.relate(b)
    with pytest.raises(AdapterError):
        a.prepare("11", split="test")
    rows[1]["G"] = "0"
    csv_write(tmp_path / "odir/full_df.csv", ODIR_FIELDS, rows)
    with pytest.raises(AdapterError, match="rows_conflict"):
        a.prepare("11")


@pytest.mark.parametrize("mode", ["duplicate", "missing", "path", "image_owner", "eye"])
def test_odir_negative(tmp_path, mode):
    a, rows = odir(tmp_path)
    if mode == "duplicate":
        rows.append(rows[0])
    elif mode == "missing":
        (tmp_path / "ODIR-5K/ODIR-5K/Training Images/11_left.jpg").unlink()
    elif mode == "path":
        rows[0]["Left-Fundus"] = "../secret.jpg"
    elif mode == "image_owner":
        rows.append(rows[0] | {"ID": "12"})
    else:
        rows[0]["Right-Fundus"] = rows[0]["Left-Fundus"]
    csv_write(tmp_path / "full_df.csv", ODIR_FIELDS, rows)
    with pytest.raises(AdapterError):
        a.prepare("11")


def test_schema_snapshots():
    from packages.dataset_adapters.schema_tools import schemas

    schemas()


def test_fixed_candidate_selection_and_pending_truth(tmp_path):
    from packages.dataset_adapters.golden import prepare_candidates

    a, _ = fair(tmp_path, count=20)
    candidates = prepare_candidates(a)
    assert len(candidates) == 20
    assert all(
        c.reference_status == "pending"
        and c.expert_expected_evidence is None
        and c.expected_template_version is None
        and not c.tool_dag
        for c in candidates
    )
    with pytest.raises(ValueError):
        prepare_candidates(a, count=19)


def test_repeated_measurements_and_empty_projection():
    p = project_note("IOP OD current: 18 mmHg\nIOP OD current: 23 mmHg")
    assert len(p.candidates) == 2 and all(c.observation is None for c in p.candidates)
    assert project_note("").raw_length == 0


def test_duplicate_npz_members_rejected():
    import io
    import warnings
    import zipfile

    data = io.BytesIO(npz())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(data, "a") as z:
            z.writestr("slo_fundus.npy", make_npy((2, 3), "|u1", b"123456"))
    with pytest.raises(AdapterError):
        read_npz(data.getvalue())

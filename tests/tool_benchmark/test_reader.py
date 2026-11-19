"""Offline counterexamples. Synthetic codec fixtures never count as model inference."""

import json
import struct
from pathlib import Path

import pytest

from packages.grounding.models import canonical, seal
from packages.tool_benchmark import (
    BenchmarkError,
    BenchmarkManifest,
    BenchmarkReader,
    manifest,
    papila_sample,
)
from packages.tool_benchmark.formats import contour_mask, grayscale_bmp, jpeg_dimensions, mask_png
from packages.tool_benchmark.models import EvaluationPolicy, ModelInput
from packages.tool_benchmark.policy import readiness
from packages.tool_benchmark.reader import bound_read, reference
from packages.tool_benchmark.schema_tools import schemas
from packages.tool_execution.qc import grayscale_png


def jpeg(h=5, w=5, suffix=b""):
    # A deliberately header-only fixture: reader does not claim JPEG full decode QC.
    body = bytes([8]) + struct.pack(">HH", h, w) + bytes([3])
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(body) + 2) + body + suffix + b"\xff\xd9"


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "FundusImages").mkdir()
    contours = tmp_path / "ExpertsSegmentations/Contours"
    contours.mkdir(parents=True)
    (tmp_path / "README.md").write_text("# PAPILAv1.1\n")
    for sample in ("RET001OD", "RET001OS", "RET002OD", "RET002OS"):
        (tmp_path / f"FundusImages/{sample}.jpg").write_bytes(jpeg(suffix=sample.encode()))
        for target in ("disc", "cup"):
            for expert in ("exp1", "exp2"):
                (contours / f"{sample}_{target}_{expert}.txt").write_text("1 1\n3 1\n3 3\n1 3\n")
    return tmp_path


def document(dataset):
    return manifest(
        [
            papila_sample(dataset, "RET001OD", split="development"),
            papila_sample(dataset, "RET001OS", split="development"),
            papila_sample(dataset, "RET002OD", split="reserved"),
        ]
    )


def test_reader_preserves_experts_and_only_image_crosses_boundary(dataset):
    doc = document(dataset)
    reader = BenchmarkReader({"PAPILA": dataset}, doc)
    image = reader.model_input("PAPILA", "RET001OD")
    assert image.purpose == "shadow" and not image.decisive_evidence
    assert b"exp1" not in canonical(image) and b"truth" not in canonical(image)
    truths = reader.truth_masks("PAPILA", "RET001OD", "optic_disc")
    assert set(truths) == {"exp1", "exp2"}
    assert sum(truths["exp1"]) == 9
    with pytest.raises(BenchmarkError, match="locked"):
        reader.model_input("PAPILA", "RET002OD")
    with pytest.raises(BenchmarkError, match="locked"):
        reader.truth_masks("PAPILA", "RET002OD", "optic_disc")
    with pytest.raises(BenchmarkError, match="missing"):
        reader.truth_masks("PAPILA", "RET001OD", "retinal_vessel")


def test_group_leakage_rejected_even_with_new_seal(dataset):
    samples = [
        papila_sample(dataset, "RET001OD", split="development"),
        papila_sample(dataset, "RET001OS", split="reserved"),
    ]
    with pytest.raises(ValueError, match="group_split"):
        manifest(samples)


def test_truth_input_forgery_and_wrong_scope_rejected(dataset):
    doc = document(dataset)
    values = doc.model_dump(mode="json")
    values["samples"][0]["image"] = values["samples"][0]["truths"][0]["file"]
    with pytest.raises(ValueError, match="truth_in_model_input"):
        seal(BenchmarkManifest, **values)
    with pytest.raises(ValueError):
        seal(BenchmarkManifest, **{**doc.model_dump(mode="json"), "scope": "RUN"})
    reader = BenchmarkReader({"PAPILA": dataset}, doc)
    image = reader.model_input("PAPILA", "RET001OD").model_dump(mode="json")
    image["truth"] = "forbidden"
    with pytest.raises(ValueError):
        ModelInput.model_validate_json(json.dumps(image))


@pytest.mark.parametrize(
    "field,value", [("group_id", "PAPILA:OTHER"), ("laterality", "left"), ("dimensions", [1, 25])]
)
def test_resealed_manifest_cannot_rewrite_source_identity(dataset, field, value):
    values = document(dataset).model_dump(mode="json")
    values["samples"][0][field] = value
    forged = seal(BenchmarkManifest, **values)
    with pytest.raises(ValueError, match="source_binding"):
        BenchmarkReader({"PAPILA": dataset}, forged)


def test_changed_image_and_contour_are_rechecked(dataset):
    reader = BenchmarkReader({"PAPILA": dataset}, document(dataset))
    (dataset / "FundusImages/RET001OD.jpg").write_bytes(jpeg(suffix=b"changed"))
    with pytest.raises(BenchmarkError, match="changed"):
        reader.model_input("PAPILA", "RET001OD")
    (dataset / "ExpertsSegmentations/Contours/RET001OS_disc_exp1.txt").write_text("0 0\n1 0\n1 1\n")
    with pytest.raises(BenchmarkError, match="changed"):
        reader.truth_masks("PAPILA", "RET001OS", "optic_disc")


@pytest.mark.parametrize(
    "variant", ["missing", "ambiguous", "duplicate_image", "symlink", "traversal", "oversize"]
)
def test_bad_source_files_fail_closed(dataset, variant):
    if variant == "missing":
        (dataset / "ExpertsSegmentations/Contours/RET001OD_cup_exp1.txt").unlink()
    elif variant == "ambiguous":
        (dataset / "ExpertsSegmentations/Contours/RET001OD_cup_exp1 2.txt").write_text("1 1")
    elif variant == "duplicate_image":
        (dataset / "FundusImages/RET001OS.jpg").write_bytes(
            (dataset / "FundusImages/RET001OD.jpg").read_bytes()
        )
    elif variant == "symlink":
        (dataset / "FundusImages/RET001OD.jpg").unlink()
        (dataset / "FundusImages/RET001OD.jpg").symlink_to(dataset / "FundusImages/RET001OS.jpg")
    elif variant == "traversal":
        with pytest.raises(ValueError):
            papila_sample(dataset, "../RET001OD", split="development")
        return
    else:
        with (dataset / "FundusImages/RET001OD.jpg").open("wb") as f:
            f.truncate(16777217)
    with pytest.raises((ValueError, OSError)):
        document(dataset)


@pytest.mark.parametrize(
    "data",
    [b"nan 1\n2 3\n4 1", b"-1 1\n2 3\n4 1", b"1 1\n2 2\n3 3", b"1 1 2\n2 3\n3 1", b"x 1\n2 3\n4 1"],
)
def test_bad_contour_rejects(data):
    with pytest.raises(BenchmarkError):
        contour_mask(data, (5, 5))


def test_polygon_closing_boundaries_and_fractional_coordinates():
    square = b"1 1\n3 1\n3 3\n1 3\n"
    assert contour_mask(square, (5, 5)) == contour_mask(square + b"1 1\n", (5, 5))
    assert sum(contour_mask(b"0.5 0.5\n3.5 0.5\n3.5 3.5\n0.5 3.5", (5, 5))) == 9
    shape, pixels = grayscale_png(mask_png((5, 5), contour_mask(square, (5, 5))), 25)
    assert shape == (5, 5) and sum(pixels) == 9


@pytest.mark.parametrize("data", [b"", b"\xff\xd8\xff\xd9", jpeg(0, 1), jpeg()[:-1]])
def test_jpeg_metadata_rejects_truncation_or_dimensions(data):
    with pytest.raises(BenchmarkError):
        jpeg_dimensions(data)


def test_bmp_palette_is_decoded_not_assumed():
    # A 2x1 image with palette indices 0,1 representing gray 128,0 (not index values).
    palette = bytes([128, 128, 128, 0, 0, 0, 0, 0])
    raw = b"\x00\x01\x00\x00"
    data = b"BM" + struct.pack("<IHHI", 66, 0, 0, 62)
    data += struct.pack("<IiiHHIIiiII", 40, 2, 1, 1, 8, 0, 4, 0, 0, 2, 0) + palette + raw
    assert grayscale_bmp(data) == ((1, 2), bytes([128, 0]))
    for bad in (data[:-1], data + b"x", data[:30] + b"\x01" + data[31:]):
        with pytest.raises(BenchmarkError):
            grayscale_bmp(bad)


def test_policy_pending_and_gate_cannot_be_removed(dataset):
    result = readiness(EvaluationPolicy(), document(dataset))
    assert not result["formal_evaluation_allowed"]
    assert {"policy_pending", "prior_exposure_unverified", "gates_5_6_pending"} <= set(
        result["reasons"]
    )
    with pytest.raises(ValueError):
        EvaluationPolicy(required_gates=(0, 1, 2))


def test_schema_snapshots():
    root = Path(__file__).resolve().parents[2] / "packages/tool_benchmark/schemas"
    for name, value in schemas().items():
        assert (root / name).read_text() == value


def test_bound_read_never_accepts_checksum_only_uri_replacement(dataset):
    ref = reference(dataset, "README.md")
    with pytest.raises(BenchmarkError):
        bound_read(dataset, ref.model_copy(update={"uri": "file:///etc/passwd"}))


def test_refuge_requires_local_label_source_and_keeps_png_format(tmp_path):
    from packages.tool_benchmark.formats import mask_png
    from packages.tool_benchmark.models import LabelDefinition
    from packages.tool_benchmark.reader import refuge_sample

    (tmp_path / "train/images").mkdir(parents=True)
    (tmp_path / "train/mask").mkdir()
    (tmp_path / "train/images/N0001.jpg").write_bytes(jpeg(1, 2))
    (tmp_path / "train/mask/N0001.png").write_bytes(mask_png((1, 2), b"\x00\x01"))
    (tmp_path / "labels.txt").write_text("Synthetic fixture: background 2, disc rim 1, cup 0.")
    labels = LabelDefinition(
        version="synthetic/1.0",
        source=reference(tmp_path, "labels.txt"),
        source_excerpt="background 2, disc rim 1, cup 0",
        background=2,
        disc_rim=1,
        cup=0,
    )
    sample = refuge_sample(
        tmp_path, "N0001", official_split="train", split="development", labels=labels
    )
    assert sample.truths[0].encoding == "png_gray_labels"
    doc = manifest([sample], labels=labels)
    reader = BenchmarkReader({"REFUGE2": tmp_path}, doc)
    assert reader.truth_masks("REFUGE2", "N0001", "optic_disc") == {"challenge": b"\x01\x01"}
    assert reader.truth_masks("REFUGE2", "N0001", "optic_cup") == {"challenge": b"\x01\x00"}
    with pytest.raises(ValueError, match="definition_unverified"):
        manifest([sample])
    (tmp_path / "labels.txt").write_text("changed")
    with pytest.raises(ValueError):
        BenchmarkReader({"REFUGE2": tmp_path}, doc)


def test_exact_four_mib_input_boundary(dataset):
    first = document(dataset)
    path = dataset / "FundusImages/RET001OD.jpg"
    raw = path.read_bytes()
    path.write_bytes(raw[:-2] + b"x" * (4194304 - len(raw)) + raw[-2:])
    doc = document(dataset)
    reader = BenchmarkReader({"PAPILA": dataset}, doc)
    assert reader.model_input("PAPILA", "RET001OD").image.byte_size == 4194304
    path.write_bytes(path.read_bytes()[:-2] + b"x\xff\xd9")
    reader = BenchmarkReader({"PAPILA": dataset}, document(dataset))
    with pytest.raises(BenchmarkError, match="object_limit"):
        reader.model_input("PAPILA", "RET001OD")
    assert doc.checksum != first.checksum


def test_formal_freeze_missing_bindings_cannot_be_fabricated():
    from packages.tool_benchmark.models import EvaluationFreeze

    with pytest.raises(ValueError):
        EvaluationFreeze.model_validate_json('{"manifest_checksum":"sha256:' + "0" * 64 + '"}')


def test_shared_onboarding_reader_has_no_publication_authority():
    from packages.contracts.example_data import onboarding_decision
    from packages.tool_benchmark.policy import read_onboarding_decision

    value = onboarding_decision()
    result = read_onboarding_decision(value)
    assert not result.visible_to_current_run
    value["visible_to_current_run"] = True
    with pytest.raises(ValueError):
        read_onboarding_decision(value)

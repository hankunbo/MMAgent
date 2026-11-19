import io
import json
import math
import struct
import zipfile

import pytest

from packages.dataset_adapters.reading import grayscale_image
from packages.tool_benchmark.development import summarize
from packages.tool_benchmark.fairseg import (
    KEYS,
    FairSegDevelopmentReader,
    array,
    decode_labels,
    members,
)
from packages.tool_benchmark.geometry_metrics import compare, geometry, ratios
from packages.tool_benchmark.models import BenchmarkError
from packages.tool_benchmark.reader import reference
from packages.tool_execution.qc import grayscale_png


def npy(shape, dtype, data, order=False):
    header = repr(dict(descr=dtype, fortran_order=order, shape=shape)).encode() + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + data


def archive(overrides=None):
    arrays = {key: npy((), "<i8", struct.pack("<q", 0)) for key in KEYS}
    arrays.update(
        slo_fundus=npy((2, 3), "|u1", bytes(range(6))),
        disc_cup_mask=npy((2, 3), "<f8", struct.pack("<6d", 0, -1, -2, 0, -1, -2)),
    )
    arrays.update(overrides or {})
    f = io.BytesIO()
    with zipfile.ZipFile(f, "w") as z:
        for key, value in arrays.items():
            z.writestr(key + ".npy", value)
    return f.getvalue()


def reader(tmp_path, raw=None):
    (tmp_path / "sample.npz").write_bytes(raw or archive())
    return FairSegDevelopmentReader(
        tmp_path, {"s": reference(tmp_path, "sample.npz")}, purpose="development_reanalysis"
    )


def test_reader_image_and_truth_separation(tmp_path):
    value = reader(tmp_path)
    shape, png = value.model_input("s")
    assert grayscale_png(png, 16777216) == ((2, 3), bytes(range(6)))
    assert value.truth("s") == (shape, bytes([0, 1, 2, 0, 1, 2]))
    assert value.strata("s") == {"gender": 0, "race": 0}


def test_model_projection_never_decodes_mask(tmp_path):
    value = reader(tmp_path, archive({"disc_cup_mask": b"invalid hidden mask"}))
    assert value.model_input("s")[0] == (2, 3)
    with pytest.raises(BenchmarkError, match="npy_header"):
        value.truth("s")


def test_fortran_truth_transposes_without_resizing():
    value = array(npy((2, 3), "<f8", struct.pack("<6d", 0, 0, -1, -1, -2, -2), True))
    assert decode_labels(value) == ((2, 3), bytes([0, 1, 2, 0, 1, 2]))


@pytest.mark.parametrize("invalid", [1.0, -3.0, float("nan"), float("inf")])
def test_invalid_labels(invalid):
    with pytest.raises(BenchmarkError, match="truth_labels"):
        decode_labels(array(npy((1, 1), "<f8", struct.pack("<d", invalid))))


@pytest.mark.parametrize("dtype", ["|O", ">f8", "<f4"])
def test_unsupported_array_schema(dtype):
    with pytest.raises(BenchmarkError, match="npy_schema"):
        array(npy((1, 1), dtype, b"12345678"))


def test_archive_unknown_or_duplicate_members():
    raw = archive({"diagnostic_note": b"secret"})
    with pytest.raises(BenchmarkError, match="archive_members"):
        members(raw, ["slo_fundus"])


def test_changed_source_cannot_be_reused(tmp_path):
    value = reader(tmp_path)
    (tmp_path / "sample.npz").write_bytes(archive({"age": npy((), "<i8", struct.pack("<q", 7))}))
    with pytest.raises(BenchmarkError, match="dataset_content_changed"):
        value.truth("s")


@pytest.mark.parametrize("purpose", ["test", "reserved", "clinical"])
def test_formal_split_remains_locked(tmp_path, purpose):
    with pytest.raises(BenchmarkError, match="formal_evaluation_locked"):
        FairSegDevelopmentReader(tmp_path, {}, purpose=purpose)


def test_unknown_sample_not_read(tmp_path):
    with pytest.raises(BenchmarkError, match="sample_not_registered"):
        reader(tmp_path).truth("other")


def test_known_boundary_distance_and_overlap():
    value = geometry(bytes([1, 0, 0, 0]), bytes([0, 0, 0, 1]), (2, 2))
    assert value["dice"] == 0 and value["area_fraction_error"] == 0
    assert value["assd_px"] == value["hd95_px"] == math.sqrt(2)
    assert geometry(b"\x01", b"\x01", (1, 1))["hd95_px"] == 0


def test_boundary_resource_limit_is_missing_not_zero():
    data = bytes(i % 2 for i in range(10000))
    result = geometry(data, data, (100, 100))
    assert result["boundary_status"] == "pair_budget_exceeded"
    assert result["hd95_px"] is None and result["dice"] == 1


def test_empty_mask_has_qc_not_fake_cdr():
    result = compare(b"\0" * 4, bytes([0, 1, 2, 0]), (2, 2))
    assert result["cup"]["dice"] == 0
    assert result["cup"]["hd95_px"] is None
    assert result["ratios"]["vertical_cdr"]["absolute_error"] is None
    assert "empty_predicted_disc" in result["qc_flags"]


def test_vertical_and_area_ratio_are_distinct():
    assert ratios(bytes([1, 1, 1, 0, 2, 0]), (2, 3)) == dict(vertical_cdr=0.5, area_cdr=0.25)


def test_aggregation_keeps_failed_rows_and_json_null():
    result = compare(bytes([1, 1, 1, 2]), bytes([1, 1, 1, 2]), (2, 2))
    rows = [dict(status="computed", metrics=result), dict(status="failed", reason="missing")]
    summary = summarize(rows)
    assert summary["selected"] == 2 and summary["technical_failure_rate"] == 0.5
    assert summary["metrics"]["cup"]["dice"]["mean"] == 1
    assert summary["metrics"]["cup"]["dice"]["n"] == 1
    assert summary["metrics"]["cup"]["dice"]["confidence_interval"] is None
    json.dumps(summary, allow_nan=False)


def test_png_projection_pixels_stable():
    assert grayscale_png(grayscale_image((1, 2), b"\x00\xff"), 16777216) == ((1, 2), b"\x00\xff")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    from packages.tool_benchmark import development as d

    data, predictions = tmp_path / "data", tmp_path / "predictions"
    (data / "Test").mkdir(parents=True)
    (predictions / "inputs").mkdir(parents=True)
    sources = []
    for n in range(6):
        path = tmp_path / f"source-{n}"
        path.write_text("source")
        sources.append(str(path))
    from pathlib import Path

    Path(sources[2]).write_text(
        "cond[mask == -1] = (0, 0, 255)   # disc rim = blue\n"
        "cond[mask == -2] = (255, 0, 0)   # cup = red\n"
    )
    weight = d.sha(sources[-2]).removeprefix("sha256:")
    Path(sources[-1]).write_text(
        json.dumps(dict(checkpoint_sha256=weight, source_commit="fixture"))
    )
    old, receipts = [], []
    for n in range(8):
        name = f"s{n}"
        path = data / "Test" / (name + ".npz")
        path.write_bytes(archive({"age": npy((), "<i8", struct.pack("<q", n))}))
        image = predictions / "inputs" / (name + ".png")
        image.write_bytes(grayscale_image((2, 3), bytes(range(6))))
        directory = predictions / "predictions" / name
        directory.mkdir(parents=True)
        labels = directory / "labels.png"
        labels.write_bytes(grayscale_image((2, 3), bytes([0, 1, 2, 0, 1, 2])))
        result = dict(
            status="success",
            ground_truth_used=False,
            checkpoint_sha256=weight,
            source_commit="fixture",
            input_sha256=d.sha(image)[7:],
            preprocessing=dict(original_size_hw=[2, 3]),
            outputs={"labels.png": {"sha256": d.sha(labels)[7:]}},
        )
        (directory / "result.json").write_text(json.dumps(result))
        old.append(dict(id=name, sha256=d.sha(path)[7:]))
        receipts.append(
            dict(
                id=name,
                status="success",
                labels_sha256=d.sha(labels)[7:],
                extracted_png_sha256=d.sha(image)[7:],
            )
        )
    (predictions / "manifest.json").write_text(json.dumps(dict(selected_count=8, files=old)))
    (predictions / "predictions-complete.json").write_text(
        json.dumps(
            dict(
                predicted_count=8,
                ground_truth_read=False,
                manifest_sha256=d.sha(predictions / "manifest.json")[7:],
                files=receipts,
            )
        )
    )
    monkeypatch.setattr(d, "DATA", data)
    monkeypatch.setattr(d, "PREDICTIONS", predictions)
    monkeypatch.setattr(d, "SOURCES", tuple(sources))
    return d, d.prepare()


def test_actual_prepare_read_compare_report_chain(frozen):
    d, protocol = frozen
    rows, summary = d.evaluate(protocol)
    assert len(rows) == summary["computed"] == 8
    assert summary["metrics"]["disc"]["dice"]["mean"] == 1
    assert summary["ratio_errors"]["vertical_cdr"]["mae"] == 0
    assert summary["new_model_calls"] == 0
    assert summary["strata"]["gender"]["0"]["selected"] == 8
    assert not summary["independent_test"]


def test_forged_protocol_copy_rejected(frozen):
    d, protocol = frozen
    with pytest.raises(ValueError):
        d.evaluate(protocol.model_copy(update={"new_model_calls": 1}))


def test_changed_prediction_counted_as_failure_not_dropped(frozen):
    d, protocol = frozen
    (d.PREDICTIONS / "predictions/s0/labels.png").write_bytes(b"damaged")
    rows, result = d.evaluate(protocol)
    assert result["computed"] == 7 and result["selected"] == 8
    assert result["technical_failure_rate"] == 0.125
    assert rows[0]["status"] == "failed"


def test_changed_weights_reject_before_any_result(frozen):
    from pathlib import Path

    d, protocol = frozen
    Path(d.SOURCES[-2]).write_text("different")
    with pytest.raises(BenchmarkError, match="evaluation_source_drift"):
        d.evaluate(protocol)

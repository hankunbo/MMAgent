"""Actual deterministic metric process and adversarial format/semantic QC."""

import json
import struct
import subprocess
import sys
import zlib
from types import SimpleNamespace

import pytest

from Algorithm_Tool.disc_geometry.runner import decode
from packages.contracts import Artifact
from packages.grounding.models import digest, validate
from packages.grounding.synthetic import ref
from packages.object_store import LocalObjectStore, checksum
from packages.tool_execution.geometry_qc import GeometryJSONPolicy, GeometryQCRegistry
from packages.tool_execution.models import ExecutionError
from tests.tool_admission.support import fixture


def mask_png(shape, pixels):
    h, w = shape

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    raw = b"".join(b"\x00" + pixels[y * w : (y + 1) * w] for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def prepared(tmp_path):
    f = fixture(tmp_path)
    source = next(iter(f.owners.records.values())).artifact
    data = mask_png((2, 3), bytes([0, 255, 255, 0, 255, 0]))
    store = LocalObjectStore(tmp_path / "geometry-objects")
    uri = store.promote(store.stage(data), checksum(data))
    source = validate(
        Artifact,
        {
            **source.model_dump(mode="json"),
            "uri": uri,
            "checksum": checksum(data),
            "dimensions": [2, 3],
            "qc_status": "PASS",
        },
    )
    constraint = f.mapping.candidates[0].output
    # QC must bind the actual input coordinate/laterality, independently of the JSON.
    constraint = constraint.model_copy(
        update={"laterality": source.laterality, "coordinate_space": source.coordinate_space}
    )
    policy = GeometryJSONPolicy(
        artifact_ref=constraint.artifact_ref,
        artifact_type_code="ARTIFACT:06E:GEOMETRY",
        unit_term=constraint.unit,
    )
    p = ref("policy", "POLICY:06E:GEOMETRY").model_copy(update={"checksum": digest(policy)})
    output = SimpleNamespace(
        qc_policy_ref=p, qc_port="technical_output_qc/1.0.0", constraint=constraint
    )
    rule = SimpleNamespace(
        media_types=("application/json",), dimensions_min=(1,), dimensions_max=(1,)
    )
    return SimpleNamespace(
        data=data,
        source=source,
        qc=GeometryQCRegistry({p: policy}, store),
        output=output,
        rule=rule,
        root=tmp_path,
    )


def test_real_metric_process_consumes_mask_and_qc_recomputes(prepared):
    f = prepared
    source = f.root / "pred.png"
    source.write_bytes(f.data)
    out = f.root / "metric.json"
    result = subprocess.run(
        [
            sys.executable,
            "Algorithm_Tool/disc_geometry/runner.py",
            "--disc-mask",
            str(source),
            "--output",
            str(out),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(out.read_bytes())
    assert value["area_px"] == 3 and value["bbox_xyxy"] == [1, 0, 3, 2]
    assert f.qc.evaluate(f.output, f.rule, out.read_bytes(), [f.source]).status.value == "PASS"


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate",
        "nonfinite",
        "extra",
        "wrong_checksum",
        "wrong_area",
        "bool",
        "nested",
        "size",
        "invalid_utf8",
    ],
)
def test_malicious_json_rejected(prepared, mode):
    f = prepared
    value = decode(f.data)
    if mode == "extra":
        value["approved"] = True
    elif mode == "wrong_checksum":
        value["input_checksum"] = digest("other")
    elif mode == "wrong_area":
        value["area_px"] = 5
    elif mode == "bool":
        value["area_px"] = True
    data = json.dumps(value).encode()
    if mode == "duplicate":
        data = data[:-1] + b',"area_px":3}'
    elif mode == "nonfinite":
        data = data.replace(b'"area_px": 3', b'"area_px": NaN')
    elif mode == "nested":
        data = b"[" * 2000 + b"0" + b"]" * 2000
    elif mode == "size":
        data = b" " * 65537
    elif mode == "invalid_utf8":
        data = b"\xff"
    with pytest.raises(ExecutionError):
        f.qc.evaluate(f.output, f.rule, data, [f.source])


def test_exact_json_size_boundary(prepared):
    f = prepared
    data = json.dumps(decode(f.data)).encode()
    data += b" " * (65536 - len(data))
    assert f.qc.evaluate(f.output, f.rule, data, [f.source]).status.value == "PASS"
    with pytest.raises(ExecutionError):
        f.qc.evaluate(f.output, f.rule, data + b" ", [f.source])


def test_empty_and_invalid_mask_refused():
    with pytest.raises(ValueError, match="empty"):
        decode(mask_png((2, 3), bytes(6)))
    with pytest.raises(ValueError, match="label"):
        decode(mask_png((2, 3), bytes([1] * 6)))

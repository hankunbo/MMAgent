import io
import json
import struct
import subprocess
import zipfile
from pathlib import Path

import pytest

from apps.glaucoma_case_supplement import demographics
from apps.glaucoma_case_view import project
from apps.segman_tool_registry import load_verified
from packages.dataset_adapters.reading import grayscale_image
from packages.object_store import checksum
from tests.test_glaucoma_case_view import data


def npy_bytes(shape, dtype, body):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode()
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body


def archive(age="63", gender=1):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("slo_fundus.npy", npy_bytes((2, 3), "|u1", b"123456"))
        z.writestr("age.npy", npy_bytes((), "<U2", age.encode("utf-32-le")))
        z.writestr("gender.npy", npy_bytes((), "<i8", struct.pack("<q", gender)))
        z.writestr("glaucoma.npy", b"deliberately unreadable forbidden label")
    return out.getvalue(), checksum(grayscale_image((2, 3), b"123456"))


def test_field_values_and_absence_not_deferred_and_label_never_opened(monkeypatch):
    raw, image = archive()
    original = zipfile.ZipFile.read

    def read(self, name, *args, **kwargs):
        assert getattr(name, "filename", name) != "glaucoma.npy"
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", read)
    v = demographics(raw, image)
    assert v["fields"]["age"]["case_value"] == 63
    assert v["fields"]["gender"]["display_value"] == "男性"
    assert v["fields"]["language"]["label"] == "当前输入无法取证"
    assert v["fields"]["family_history"]["label"] == "当前输入无法取证"
    assert "glaucoma" not in v["fields"]
    assert v["old_visible_manifest_modified"] is False


def test_demographic_same_image_required_and_unknown_not_zero():
    raw, image = archive(gender=-1)
    assert demographics(raw, image)["fields"]["gender"]["case_value"] is None
    with pytest.raises(ValueError, match="image_mismatch"):
        demographics(raw, "sha256:" + "0" * 64)
    bad, image = archive(gender=9)
    with pytest.raises(ValueError, match="code_unknown"):
        demographics(bad, image)


def test_resource_drift_is_rejected(tmp_path, monkeypatch):
    from apps import segman_tool_registry as r

    resource = tmp_path / "resource"
    resource.write_bytes(b"a")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "source_commit": r.COMMIT,
                "checkpoint_sha256": r.WEIGHT_SHA,
                "resources": [{"path": str(resource), "bytes": 1, "sha256": r.digest(resource)}],
            }
        )
    )
    monkeypatch.setattr(r, "LOCK", lock)
    load_verified()
    resource.write_bytes(b"b")
    with pytest.raises(ValueError, match="resource_drift"):
        load_verified()


def test_cpu_mask_union_and_existing_formula_on_synthetic_files(tmp_path):
    python = Path("/opt/data/test/envs/SLOctolyzer/bin/python")
    if not python.is_file():
        pytest.skip("optional deployed measurement environment")
    image = tmp_path / "input.png"
    image.write_bytes(grayscale_image((2, 3), b"123456"))
    outputs = {}
    for name, pixels in {
        "labels": bytes([0, 1, 0, 0, 2, 0]),
        "cup": bytes([0, 0, 0, 0, 255, 0]),
        "rim": bytes([0, 255, 0, 0, 0, 0]),
        "disc": bytes([0, 255, 0, 0, 255, 0]),
    }.items():
        path = tmp_path / (name + ".png")
        path.write_bytes(grayscale_image((2, 3), pixels))
        outputs[path.name] = {
            "sha256": checksum(path.read_bytes())[7:],
            "bytes": path.stat().st_size,
        }
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "input_sha256": checksum(image.read_bytes())[7:],
                "pixel_counts": {"cup": 1, "rim": 1, "disc": 2},
                "outputs": outputs,
            }
        )
    )
    args = [
        str(python),
        "Algorithm_Tool/segman_slo/inspect_outputs.py",
        "--directory",
        str(tmp_path),
        "--image",
        str(image),
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["metrics"]["vcdr"]["value"] == value["metrics"]["acdr"]["value"] == 0.5
    (tmp_path / "cup.png").write_bytes(grayscale_image((2, 3), bytes([255, 0, 0, 0, 0, 0])))
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "segman_binary_union_mismatch" in result.stderr


def supplement_fixture():
    s, r = data()
    r["input"]["checksum"] = "sha256:" + "1" * 64
    r["artifacts"] = [
        {
            "asset_code": "ASSET:OID_EXTRA001",
            "scope": {"case_code": r["case_code"]},
            "extensions": {"component_task": "segman_supplement"},
        }
    ]
    r["supplement"] = {
        "case_code": r["case_code"],
        "input_checksum": r["input"]["checksum"],
        "artifact_codes": ["ASSET:OID_EXTRA001"],
        "registered_tool": {"code": "synthetic_segman"},
        "mask_check": {
            "metrics": {
                **{m: {"value": 0.4} for m in ("vcdr", "acdr", "hcdr", "rdar")},
                "rim_profile": {"values": {"left_px": 3}},
            },
            "pixel_counts": {"cup": 2, "rim": 3, "disc": 5},
        },
        "demographics": {
            "fields": {
                "age": {
                    "label": "已有结果",
                    "category": "observed",
                    "case_value": 63,
                    "display_value": 63,
                }
            }
        },
        "visual_review": {
            "method": "direct_image_visual_inspection",
            "reviewer": "synthetic",
            "observations": {
                key: {
                    "status": "已有观察·判读不确定",
                    "summary": "synthetic indeterminate",
                    "next_action": "review",
                    "finding": "indeterminate",
                }
                for key in ("RNFL_REFLECTANCE", "DISC_BLEED", "IMAGE_QC")
            },
        },
    }
    return s, r


def test_observed_supplement_replaces_pending_without_diagnostic_vote():
    s, r = supplement_fixture()
    result = project(s, r)
    nodes = {n["key"]: n["patient_evidence"] for n in result["nodes"]}
    assert nodes["CUP_RIM_MEASURE"]["status"] == "已有结果"
    assert nodes["CUP_RIM_MEASURE"]["results"][0]["value"] == 0.4
    assert nodes["RISK_FIELDS"]["acquisition"]["fields"]["age"]["case_value"] == 63
    assert nodes["DISC_BLEED"]["observation_source"]["finding"] == "indeterminate"
    assert nodes["RNFL_THICKNESS"]["status"] == "当前输入无法取证"
    assert result["acquisition_summary"]["unavailable_current_input"] == 7
    assert "tool_unverified" not in result["acquisition_summary"]
    assert result["acquisition_policy"]["diagnostic_score_changed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_code", "CASE:ANON_OTHER001"),
        ("input_checksum", "bad"),
        ("artifact_codes", ["missing"]),
    ],
)
def test_supplement_wrong_scope_or_unregistered_source_rejected(field, value):
    s, r = supplement_fixture()
    r["supplement"][field] = value
    with pytest.raises(ValueError, match="supplement_projection"):
        project(s, r)

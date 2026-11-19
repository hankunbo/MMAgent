"""Main ENV-001 tests the isolated scientific runner without installing numpy here."""

import copy
import json
import subprocess

import pytest

from apps import vessel_branch_tool as tool

CHECKS = [
    "known_y_60",
    "known_y_90",
    "known_y_120",
    "parent_child_60",
    "parent_child_90",
    "parent_child_120",
    "rotation_1",
    "rotation_2",
    "rotation_3",
    "coordinate_restore_1",
    "coordinate_restore_2",
    "coordinate_restore_3",
    "scale_2",
    "short_y_scale_aware",
    "crossing_excluded",
    "unrooted_rejected",
    "loop_rejected",
    "empty_null_not_zero",
    "deterministic_repeat",
    "float_mask",
    "nonbinary",
    "rgb",
    "nan_center",
    "infinite_radius",
    "outside_center",
    "tiny_radius",
    "large_mask",
    "scale_qc_missing",
    "scale_qc_ambiguous",
    "scale_qc_angle_drift",
    "png_rgb",
    "png_nonbinary",
    "png_large",
    "png_binding",
]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("branch-synthetic") / "results"
    p = subprocess.run(
        [
            str(tool.PYTHON),
            "-m",
            "Algorithm_Tool.vessel_branch_geometry.verify",
            "--output",
            str(root),
        ],
        cwd=tool.ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            "PATH": str(tool.PYTHON.parent) + ":/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TMPDIR": str(root.parent),
        },
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads((root / "verification.json").read_bytes())


@pytest.mark.parametrize("name", CHECKS)
def test_geometry_case(measured, name):
    assert measured["checks"][name] is True


def test_result_boundary_and_tampering(measured):
    result = measured["results"]["y90"]
    result["checksum"] = "sha256:" + tool.checksum(result)
    assert tool.validate_output(result) == result
    for changes in [{"count": 3}, {"median_degrees": 0}, {"status": "no_valid_bifurcation"}]:
        bad = {**result, **changes}
        bad["checksum"] = "sha256:" + tool.checksum(
            {k: v for k, v in bad.items() if k != "checksum"}
        )
        with pytest.raises(ValueError, match="summary"):
            tool.validate_output(bad)


@pytest.mark.parametrize(
    "mutation",
    ["mean", "policy", "coordinate", "angle_source", "scale", "input_shape", "nonfinite"],
)
def test_rehashed_result_still_requires_consistency(measured, mutation):
    bad = copy.deepcopy(measured["results"]["y90"])
    row = next(j for j in bad["junctions"] if j["status"] == "measured")
    if mutation == "mean":
        bad["mean_degrees"] = 0
    elif mutation == "policy":
        bad["policy"]["scale_angle_tolerance_degrees"] = 180
    elif mutation == "coordinate":
        row["vertex_xy"] = [0]
    elif mutation == "angle_source":
        row["daughter_samples_xy"][0][1][0] += 10
    elif mutation == "scale":
        row["scale_checks"][0]["passed"] = False
    elif mutation == "input_shape":
        bad["input_bindings"] = {
            "vein_mask": {"dimensions_hw": [2, 3]},
            "disc_mask": {"dimensions_hw": [3, 2]},
        }
    else:
        bad["mean_degrees"] = float("inf")
        with pytest.raises(ValueError, match="JSON compliant"):
            tool.validate_output(bad)
        return
    bad["checksum"] = "sha256:" + tool.checksum({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ValueError, match="branch_output_"):
        tool.validate_output(bad)

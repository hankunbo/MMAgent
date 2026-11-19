"""ENV-001 exercises the bounded scientific process; no models/GPU."""

import copy
import json
import subprocess

import pytest

from Algorithm_Tool.av_vascular_metrics.contract import checksum, validate
from apps import av_vascular_tool as tool

CHECKS = [
    "deterministic",
    "overlap_excluded",
    "density_exact",
    "separate_classes",
    "annulus_explicit",
    "straight_ratio_one",
    "rotated_straight",
    "diagonal_euclidean",
    "bent_greater_one",
    "short_excluded",
    "line_fractal_one",
    "plane_fractal_two",
    "empty_fractal_null",
    "empty_width_null",
    "empty_tortuosity_null",
    "width_normalized",
    "nonbinary",
    "rgb",
    "mismatch",
    "empty_disc",
    "split_disc",
    "clipped_annulus_reported",
]


@pytest.fixture(scope="module")
def av_fixture(tmp_path_factory):
    output = tmp_path_factory.mktemp("av-metrics") / "results"
    p = subprocess.run(
        [
            str(tool.PYTHON),
            "-m",
            "Algorithm_Tool.av_vascular_metrics.verify",
            "--output",
            str(output),
        ],
        cwd=tool.ROOT,
        timeout=60,
        capture_output=True,
        text=True,
        env={
            "PATH": str(tool.PYTHON.parent) + ":/usr/bin:/bin",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(output.parent),
        },
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads((output / "checks.json").read_bytes()), json.loads(
        (output / "result.json").read_bytes()
    )


@pytest.mark.parametrize("key", CHECKS)
def test_scientific_fixture(av_fixture, key):
    assert av_fixture[0][key]


@pytest.mark.parametrize(
    "change", ["unknown", "density", "width", "ratio", "shape", "summary", "nan"]
)
def test_invalid_rehashed_output(av_fixture, change):
    r = copy.deepcopy(av_fixture[1])
    m = r["classes"]["artery"]
    if change == "unknown":
        r["clinical_probability"] = 0.9
    elif change == "density":
        m["whole_image"]["skeleton_density"] += 0.1
    elif change == "width":
        m["whole_image"]["width_proxy_disc_ratio"] += 0.1
    elif change == "ratio":
        m["tortuosity"]["segments"][0]["ratio"] += 0.1
    elif change == "shape":
        r["input_bindings"]["disc_mask"]["dimensions_hw"][0] += 1
    elif change == "summary":
        m["tortuosity"]["mean"] += 0.1
    elif change == "nan":
        m["skeleton_fractal"]["value"] = float("nan")
    with pytest.raises(ValueError):
        r["checksum"] = checksum({k: v for k, v in r.items() if k != "checksum"})
        validate(r)


def test_public_import_no_model():
    p = subprocess.run(
        [
            "/opt/data/test/envs/MMAgent1/bin/python",
            "-c",
            "import sys;import apps.av_vascular_tool;"
            "assert not {'torch','numpy','segment_anything'} & set(sys.modules)",
        ],
        cwd=tool.ROOT,
        timeout=15,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr

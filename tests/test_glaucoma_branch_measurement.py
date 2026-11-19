"""Synthetic measurements only; current case and native Tool DB are not used."""

import copy
import json
from types import SimpleNamespace

import pytest

from apps import glaucoma_branch_measurement as branch
from apps import vessel_branch_tool as tool
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_associations import apply as associate
from apps.glaucoma_evidence_plan import checksum
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from tests.test_glaucoma_associations import example, source_bundle
from tests.test_glaucoma_structure_ui import browser


def add_synthetic_measurement(value, *, store=None, repo=None):
    value = copy.deepcopy(value)
    case = value["case_evidence"]
    disc = next(
        a for a in case["artifacts"] if a["artifact_type_code"] == "ARTIFACT:SLO_SEGMAN_DISC"
    )
    prototype = json.loads(
        (tool.ROOT / "packages/contracts/examples/v1/Artifact.valid.json").read_bytes()
    )
    vein = {**prototype, **copy.deepcopy(case["input"])}
    vein["scope"] = {"scope_type": "CASE", "case_code": case["case_code"]}
    disc.setdefault("dimensions", case["input"]["dimensions"])
    vein.update(
        asset_code=identity("ASSET", case["case_code"] + ":synthetic-branch-vein"),
        artifact_type_code="ARTIFACT:SLO_VEIN_MASK",
        derived_from=[case["input"]["asset_code"]],
        checksum=branch.byte_checksum(b"synthetic vein fixture"),
    )
    vein["provenance"] = {
        **vein["provenance"],
        "source_codes": vein["derived_from"],
        "input_checksums": [case["input"]["checksum"]],
    }
    if store is not None:
        vein["uri"] = store.promote(store.stage(b"synthetic vein fixture"), vein["checksum"])
    case["artifacts"].append(vein)
    record = {
        "schema_version": "vessel-branch-geometry/1.0.0",
        "policy": copy.deepcopy(tool.POLICY),
        "policy_checksum": "sha256:" + tool.checksum(tool.POLICY),
        "unit": "degree",
        "research_only": True,
        "clinical_threshold": None,
        "disc_center_xy": [12, 22],
        "disc_radius_px": 3,
        "status": "measured",
        "count": 1,
        "median_degrees": 90.0,
        "mean_degrees": 90.0,
        "junctions": [
            {
                "junction_id": "J001",
                "vertex_xy": [12, 12],
                "arms": 3,
                "status": "measured",
                "reason": None,
                "angle_degrees": 90.0,
                "root_xy": [12, 19],
                "parent_samples_xy": [[12, 14], [12, 17]],
                "daughter_samples_xy": [[[10, 10], [8, 8]], [[14, 10], [16, 8]]],
                "scale_checks": [
                    {"scale": s, "angle_degrees": 90.0, "passed": True}
                    for s in tool.POLICY["scale_check_factors"]
                ],
            }
        ],
        "input_bindings": {
            port: {"sha256": a["checksum"], "dimensions_hw": a["dimensions"]}
            for port, a in [("vein_mask", vein), ("disc_mask", disc)]
        },
    }
    record["checksum"] = "sha256:" + tool.checksum(record)
    raw = branch.encoded(record)
    artifact = copy.deepcopy(vein)
    artifact.update(
        asset_code=identity("ASSET", case["case_code"] + ":synthetic-branch-result"),
        checksum=branch.byte_checksum(raw),
        artifact_type_code="ARTIFACT:SLO_VENULAR_BRANCH_GEOMETRY",
        media_type="application/json",
        dimensions=[],
        measurements=[],
        coordinate_space="NOT_APPLICABLE",
        unit=None,
        research_only=True,
        derived_from=sorted([vein["asset_code"], disc["asset_code"]]),
    )
    artifact["provenance"] = {
        **artifact["provenance"],
        "source_codes": artifact["derived_from"],
        "input_checksums": [vein["checksum"], disc["checksum"]],
    }
    if store is not None:
        artifact["uri"] = store.promote(store.stage(raw), artifact["checksum"])
    artifact = Artifact.model_validate(artifact).model_dump(mode="json")
    if repo is not None:
        repo.register_observation_bundle(
            (Artifact.model_validate(vein), Artifact.model_validate(artifact)),
            identity("TASK", case["case_code"] + ":synthetic-branch-save"),
        )
    return branch.attach(value, record, artifact)


def fixture(monkeypatch):
    old = associate(example(monkeypatch), source_bundle())
    return old, add_synthetic_measurement(old)


def test_registered_mask_reader_uses_existing_object_bound(tmp_path):
    from packages.object_store import LocalObjectStore

    store = LocalObjectStore(tmp_path / "owned")
    body = b"synthetic input; PNG semantics are checked by the runner"
    sha = branch.byte_checksum(body)
    ref = SimpleNamespace(uri=store.promote(store.stage(body), sha), checksum=sha)
    assert branch.read_registered_input(store, ref) == body
    ref.checksum = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="input_drift"):
        branch.read_registered_input(store, ref)


def test_real_artifact_constructor_obeys_frozen_contract(tmp_path):
    from packages.object_store import LocalObjectStore

    store = LocalObjectStore(tmp_path / "owned")
    raw = b'{"synthetic":true}'
    refs = {
        k: SimpleNamespace(
            asset_code="ASSET:OID_SYNTHETIC_" + k.upper(), checksum="sha256:" + v * 64
        )
        for k, v in [("vein", "a"), ("disc", "b")]
    }
    result = branch.measurement_artifact(
        {"case_code": "CASE:ANON_SYNTHETIC_BRANCH", "input": {"laterality": "UNKNOWN"}},
        refs,
        raw,
        store.promote(store.stage(raw), branch.byte_checksum(raw)),
    )
    assert result.provenance.generator_version == "case-branch-measurement-1.0.0"
    assert result.provenance.input_checksums == tuple(x.checksum for x in refs.values())
    assert result.scope.case_code == "CASE:ANON_SYNTHETIC_BRANCH"


def test_attachment_keeps_history_but_displays_measured_node(monkeypatch):
    old, current = fixture(monkeypatch)
    validate_snapshot(old)
    validate_snapshot(current)
    assert "branch_geometry" not in old
    assert old["nodes"] == current["nodes"]
    assert old["source_associations"] == current["source_associations"]
    browser(
        current,
        r"""
run("show('VENULAR_BRANCH_ANGLE')");
check(text(elements.detail).includes('90.00°'),'measurement missing');
check(text(elements.detail).includes('已有结果'),'new status missing');
check(!text(elements.detail).includes('算法待补'),'old pending state leaked');
check(text(elements.detail).includes('尺度复核'),'scale detail missing');
check(all(elements.detail).some(e=>e.tag==='svg'),'overlay absent');
run('completionOverview()');
check(text(elements.detail).includes('测得1个静脉分叉：90.00°'),
 'overview retained stale branch summary');
run('for(const n of D.nodes)show(n.key)');
""",
    )


@pytest.mark.parametrize("change", ["case", "port", "derived", "summary", "artifact_bytes", "mean"])
def test_rehashed_attachment_rejects_mismatch(monkeypatch, change):
    _, v = fixture(monkeypatch)
    x = v["branch_geometry"]
    if change == "case":
        x["artifact"]["scope"]["case_code"] = "CASE:ANON_OTHER"
    elif change == "port":
        r = x["record"]["input_bindings"]
        r["vein_mask"], r["disc_mask"] = r["disc_mask"], r["vein_mask"]
    elif change == "derived":
        x["artifact"]["derived_from"] = [v["case_evidence"]["input"]["asset_code"]]
    elif change == "summary":
        x["summary"] = "made up"
    elif change == "mean":
        x["record"]["mean_degrees"] = 0
    else:
        x["artifact"]["checksum"] = "sha256:" + "f" * 64
    x["record"]["checksum"] = "sha256:" + tool.checksum(
        {k: z for k, z in x["record"].items() if k != "checksum"}
    )
    x["checksum"] = checksum({k: z for k, z in x.items() if k != "checksum"})
    v["checksum"] = checksum({k: z for k, z in v.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate_snapshot(v)

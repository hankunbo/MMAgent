"""Append exact registered CPU measurements; keep the previous graph unchanged."""

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from apps import vessel_branch_tool as tool
from apps.glaucoma_evidence_plan import checksum
from packages.contracts import Artifact

VERSION = "case-branch-measurement/1.0.0"
NODE = "VENULAR_BRANCH_ANGLE"
INPUT_TYPES = {
    "vein_mask": {"ARTIFACT:SLO_VEIN_MASK"},
    "disc_mask": {"ARTIFACT:SLO_DISC_MASK", "ARTIFACT:SLO_SEGMAN_DISC"},
}


def encoded(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def byte_checksum(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read_registered_input(store, ref):
    # Runtime's existing managed-object port has a 4 MiB bound, narrower than
    # the standalone PNG runner. Do not enlarge that storage contract here.
    data = store.read_bounded(ref.uri, 4_194_304)
    if byte_checksum(data) != ref.checksum:
        raise ValueError("registered_branch_input_drift")
    return data


def summary(record):
    values = [j["angle_degrees"] for j in record["junctions"] if j["status"] == "measured"]
    if not values:
        return "算法已完成，本图没有通过拓扑、父子支和尺度检查的分叉。"
    return (
        f"测得{len(values)}个静脉分叉："
        + "、".join(f"{v:.2f}°" for v in values)
        + f"；中位数{record['median_degrees']:.2f}°。"
    )


def validate(value):
    x = value.get("branch_geometry")
    if x is None:
        return
    if x["schema"] != VERSION or x["node"] != NODE:
        raise ValueError("branch_attachment_version_or_node")
    if x["checksum"] != checksum({k: v for k, v in x.items() if k != "checksum"}):
        raise ValueError("branch_attachment_checksum")
    record = tool.validate_output(x["record"])
    artifact = Artifact.model_validate(x["artifact"])
    case = value["case_evidence"]
    by_code = {a["asset_code"]: a for a in case["artifacts"]}
    if (
        artifact.scope.case_code != case["case_code"]
        or not artifact.research_only
        or artifact.artifact_type_code != "ARTIFACT:SLO_VENULAR_BRANCH_GEOMETRY"
        or artifact.checksum != byte_checksum(encoded(record))
        or by_code.get(artifact.asset_code) != x["artifact"]
    ):
        raise ValueError("branch_attachment_artifact_scope_or_content")
    selected = []
    if set(record["input_bindings"]) != {"vein_mask", "disc_mask"}:
        raise ValueError("branch_attachment_input_ports")
    for port, binding in record["input_bindings"].items():
        matches = [a for a in case["artifacts"] if a["checksum"] == binding["sha256"]]
        if len(matches) != 1:
            raise ValueError("branch_attachment_input_binding")
        a = matches[0]
        if (
            a["scope"]["case_code"] != case["case_code"]
            or a["artifact_type_code"] not in INPUT_TYPES[port]
            or case["input"]["asset_code"] not in a["derived_from"]
            or a["dimensions"] != binding["dimensions_hw"]
        ):
            raise ValueError("branch_attachment_input_scope")
        selected.append(a["asset_code"])
    if sorted(artifact.derived_from) != sorted(selected):
        raise ValueError("branch_attachment_derivation")
    if x["summary"] != summary(record) or x["tool"] != {
        "code": tool.CODE,
        "version": tool.VERSION,
        "implementation": tool.IMPLEMENTATION,
    }:
        raise ValueError("branch_attachment_summary_or_tool")


def attach(value, record, artifact):
    result = copy.deepcopy(value)
    for key in ["display_explanations", "case_graph_artifact", "checksum"]:
        result.pop(key, None)
    if "branch_geometry" in result:
        raise ValueError("new_base_without_branch_attachment_required")
    result["case_evidence"]["artifacts"].append(artifact)
    x = {
        "schema": VERSION,
        "node": NODE,
        "record": record,
        "artifact": artifact,
        "tool": {"code": tool.CODE, "version": tool.VERSION, "implementation": tool.IMPLEMENTATION},
        "summary": summary(record),
    }
    x["checksum"] = checksum(x)
    result["branch_geometry"] = x
    result["title"] = "青光眼病例证据图 · 静脉分叉测量版"
    result["checksum"] = checksum(result)
    validate(result)
    return result


def measurement_artifact(case, refs, raw, uri):
    from db.runtime.repository.case_pipeline import identity

    sha = byte_checksum(raw)
    return Artifact.model_validate(
        {
            "schema_version": "1.0",
            "contract_type": "Artifact",
            "asset_code": identity("ASSET", case["case_code"] + sha),
            "artifact_type_code": "ARTIFACT:SLO_VENULAR_BRANCH_GEOMETRY",
            "uri": uri,
            "checksum": sha,
            "media_type": "application/json",
            "qc_status": "PASS",
            "dimensions": [],
            "measurements": [],
            "derived_from": sorted(x.asset_code for x in refs.values()),
            "research_only": True,
            "scope": {"scope_type": "CASE", "case_code": case["case_code"]},
            "laterality": case["input"]["laterality"],
            "coordinate_space": "NOT_APPLICABLE",
            "unit": None,
            "provenance": {
                "actor": "PLANNER",
                "created_at": datetime.now(UTC).isoformat(),
                "generator_version": VERSION.replace("/", "-"),
                "source_codes": sorted(x.asset_code for x in refs.values()),
                "input_checksums": [x.checksum for x in refs.values()],
                "knowledge_versions": [],
            },
        }
    )


def main():
    from apps.evidence_graph_html import validate_snapshot, write_export
    from apps.glaucoma_case_storage import read_view, store_view
    from apps.patient_service_assembly import OBJECT_ROOTS
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from db.runtime.repository.case_pipeline import identity
    from packages.object_store import LocalObjectStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--measurement", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".json").exists():
        raise ValueError("new_graph_output_required")
    base = json.loads(a.base.read_bytes())
    validate_snapshot(base)
    record = tool.validate_output(json.loads(a.measurement.read_bytes()))
    case = base["case_evidence"]
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    current = {x.asset_code: x for x in repo.get_case_bundle(case["case_code"]).artifacts}
    refs = {}
    for port, binding in record["input_bindings"].items():
        matches = [x for x in current.values() if x.checksum == binding["sha256"]]
        if (
            len(matches) != 1
            or matches[0].scope.case_code != case["case_code"]
            or matches[0].artifact_type_code not in INPUT_TYPES[port]
            or case["input"]["asset_code"] not in matches[0].derived_from
        ):
            raise ValueError("registered_branch_input_required")
        refs[port] = matches[0]
    # Recompute from registered object bytes: caller-supplied scores plus a new
    # checksum are not accepted as an actual measurement. CPU only, no model.
    with tempfile.TemporaryDirectory(prefix="branch-recheck-", dir=ROOT_TMP()) as temp:
        paths = {}
        for port, ref in refs.items():
            data = read_registered_input(store, ref)
            paths[port] = Path(temp) / (port + ".png")
            paths[port].write_bytes(data)
        output = Path(temp) / "measurement.json"
        with tool.native_connection() as c:
            plan = tool.resolve(c, paths["vein_mask"], paths["disc_mask"], output)
        subprocess.run(
            plan["argv"],
            check=True,
            timeout=60,
            cwd=tool.ROOT,
            env={
                "PATH": str(tool.PYTHON.parent) + ":/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "TMPDIR": temp,
            },
        )
        if json.loads(output.read_bytes()) != record:
            raise ValueError("registered_branch_recomputation_mismatch")
    raw = encoded(record)
    sha = byte_checksum(raw)
    code = identity("ASSET", case["case_code"] + sha)
    artifact = current.get(code)
    if artifact is None:
        artifact = measurement_artifact(case, refs, raw, store.promote(store.stage(raw), sha))
    result = attach(base, record, artifact.model_dump(mode="json"))
    validate_snapshot(result)
    repo.register_observation_bundle((artifact,), identity("TASK", "branch:" + artifact.asset_code))
    graph = store_view(result, repo=repo, store=store)
    saved = read_view(case["case_code"], graph["asset_code"], repo=repo, store=store)
    write_export(saved, a.output)
    print(
        json.dumps(
            {
                "measurement_artifact": artifact.asset_code,
                "graph_artifact": graph["asset_code"],
                "output": str(a.output),
                "summary": summary(record),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def ROOT_TMP():
    return tool.ROOT / ".local"


if __name__ == "__main__":
    main()

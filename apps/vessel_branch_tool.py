"""Register and call the fixed topology-aware CPU measurement in native Tool catalog."""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from psycopg.types.json import Jsonb

from Algorithm_Tool.vessel_branch_geometry.policy import POLICY, REASONS
from apps.slo_tool_registry import ROOT, checksum, digest, native_connection

PYTHON = Path("/opt/data/test/envs/PVBM/bin/python")
RUNNER = ROOT / "Algorithm_Tool/vessel_branch_geometry/runner.py"
VERSION = "1.0.0"
CODE = "calculate_slo_venular_branch_angles"
IMPLEMENTATION = "topology_disc_scaled_cpu_v1"
LOCK = ROOT / "configs/tools/venular-branch-geometry-v1.json"
PACKAGES = ["numpy", "scipy", "scikit-image", "networkx", "pillow"]
PORTS = [
    ("input", "vein_mask", "slo:VeinMask"),
    ("input", "disc_mask", "slo:OpticDiscMask"),
    ("output", "observation", "slo:VenularBranchGeometry"),
]
FORMULA = {
    "angle": "degrees(acos(dot(unit(child1),unit(child2))))",
    "parent": "unique disc-root path; ambiguous roots/cycles excluded",
    "directions": "TLS line fit of 13 samples over 0.10–0.25 disc diameters; native-grid angle",
    "scale": "128 px equivalent disc diameter; canonical quarter rotation",
    "topology": "junction merge 0.05D; terminal spur pruning <=0.08D; only degree 3",
    "roi": "whole image outside equivalent disc circle; not SIVA annulus",
    "baseline": "https://github.com/aim-lab/PVBM @ 5edb79a6eff5eceb94f1acc623b8d13e126bcfa7",
    "ownership": "MMAgent independent algorithm; official PVBM unchanged",
    "scale_qc": "0.75x and 1.5x perturbations of canonical 128px-disc grid; "
    "unique match within 0.08D; delta <=5 degrees",
}


def argv():
    return [
        str(PYTHON),
        "-m",
        "Algorithm_Tool.vessel_branch_geometry.runner",
        "--vein-mask",
        "{input.vein_mask}",
        "--disc-mask",
        "{input.disc_mask}",
        "--output",
        "{output.observation}",
    ]


def environment():
    result = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) "
            f"for n in {PACKAGES!r}" + "}))",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
        env={"PATH": str(PYTHON.parent) + ":/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
    )
    return json.loads(result.stdout)


def prepare(evidence):
    if evidence.stat().st_size > 2_097_152:
        raise ValueError("test_evidence_size_limit")
    xml = ElementTree.parse(evidence).getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    count = sum(int(s.get("tests", "0")) for s in suites)
    if count < 25 or any(
        int(s.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise ValueError("passing_branch_tests_required")
    resources = [RUNNER, Path(__file__), PYTHON.resolve(), PYTHON.parent.parent / "pyvenv.cfg"]
    resources += [RUNNER.with_name(name) for name in ["stability.py", "policy.py"]]
    spec = {
        "schema": "venular-branch-installation/1.0.0",
        "tool": CODE,
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "formula": FORMULA,
        "environment": environment(),
        "resources": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in resources
        ],
        "test_evidence": {
            "path": str(evidence.resolve()),
            "sha256": digest(evidence),
            "tests": count,
        },
        "clinical_validated": False,
        "runtime_06_bundle_installed": False,
    }
    if LOCK.exists():
        if json.loads(LOCK.read_bytes()) != spec:
            raise ValueError("installation_content_conflict")
    else:
        with LOCK.open("x") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec


def load_verified():
    spec = json.loads(LOCK.read_bytes())
    if (spec["tool"], spec["version"], spec["implementation"], spec["formula"]) != (
        CODE,
        VERSION,
        IMPLEMENTATION,
        FORMULA,
    ) or spec["environment"] != environment():
        raise ValueError("branch_definition_or_environment_drift")
    for r in [*spec["resources"], spec["test_evidence"]]:
        if digest(r["path"]) != r["sha256"]:
            raise ValueError("branch_resource_drift")
    return spec


def rows(c):
    return c.execute(
        "SELECT v.version,v.enabled,i.implementation_key,i.enabled,i.validation_status,"
        "i.provenance->>'installation_checksum' FROM medgraph.algorithm_tool t "
        "JOIN medgraph.tool_version v ON v.tool_id=t.id "
        "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id WHERE t.code=%s",
        (CODE,),
    ).fetchall()


def register(c, spec):
    if not c.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("native_owner_required")
    c.execute("SELECT pg_advisory_xact_lock(7150930)")
    old = rows(c)
    count = c.execute(
        "SELECT count(*) FROM medgraph.algorithm_tool WHERE code=%s", (CODE,)
    ).fetchone()[0]
    if count:
        if old != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", checksum(spec))]:
            raise ValueError("branch_existing_state_or_content_conflict")
        return {"status": "already_registered", "tool": CODE, "rows": old}
    ids = {}
    for direction, _port, code in PORTS:
        schema = (
            {"format": "image/png", "mode": "L", "values": [0, 255], "modality": "SLO"}
            if direction == "input"
            else {
                "format": "application/json",
                "schema_version": "vessel-branch-geometry/1.0.0",
                "research_only": True,
            }
        )
        found = c.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if found:
            if found[1] != schema:
                raise ValueError("branch_artifact_type_conflict")
            ids[code] = found[0]
        else:
            ids[code] = c.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, code, Jsonb(schema)),
            ).fetchone()[0]
    meta = {
        "intended_use": "research_only",
        "decisive_evidence": False,
        "input_modality": "SLO",
        "installation_checksum": checksum(spec),
        "input_requirements": "same-case aligned binary vein/disc masks, square pixels; "
        "case identity verified by Runtime/caller",
        "limitations": "topology is inferred from masks; "
        "no ground-truth angle accuracy or clinical cutoff claimed",
        "parameter_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    runtime = {
        "manager": "venv",
        "prefix": str(PYTHON.parent.parent),
        "python_executable": str(PYTHON),
        "device": "cpu",
        "python": "3.11",
    }
    tid = c.execute(
        "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
        "VALUES(%s,%s,'calculator',%s,%s) RETURNING id",
        (
            CODE,
            "静脉分叉识别与盘径归一化分支角",
            "区分交叉、父干与两子支；保留排除点及原因。",
            Jsonb(meta),
        ),
    ).fetchone()[0]
    vid = c.execute(
        "INSERT INTO medgraph.tool_version(tool_id,version,runtime,metadata) "
        "VALUES(%s,%s,%s,%s) RETURNING id",
        (tid, VERSION, Jsonb(runtime), Jsonb(meta)),
    ).fetchone()[0]
    for direction, port, code in PORTS:
        c.execute(
            "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,artifact_type_id,"
            "required,cardinality_min,cardinality_max,constraints) VALUES(%s,%s,%s,%s,true,1,1,%s)",
            (vid, direction, port, ids[code], Jsonb({"research_only": True})),
        )
    iid = c.execute(
        "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
        "implementation_kind,entrypoint_uri,working_directory,runtime_environment,invocation,"
        "formula_spec,hardware_requirements,provenance,validation_status,validation_report) "
        "VALUES(%s,%s,'formula_python',%s,%s,%s,%s,%s,%s,%s,'smoke_tested',%s) RETURNING id",
        (
            vid,
            IMPLEMENTATION,
            RUNNER.as_uri(),
            str(ROOT),
            Jsonb(runtime),
            Jsonb({"argv": argv(), "shell": False, "timeout_seconds": 60}),
            Jsonb(FORMULA),
            Jsonb({"accelerator": "cpu", "required": False}),
            Jsonb({"installation_checksum": checksum(spec)}),
            Jsonb(spec["test_evidence"]),
        ),
    ).fetchone()[0]
    for i, r in enumerate(spec["resources"]):
        c.execute(
            "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
            "resource_role,"
            "uri,sha256,byte_size,required) VALUES(%s,%s,'fixed_local_resource',%s,%s,%s,true)",
            (iid, f"resource_{i}", Path(r["path"]).as_uri(), r["sha256"], r["bytes"]),
        )
    return {"status": "registered", "tool": CODE, "rows": rows(c)}


def validate_output(result):
    if result.get("schema_version") != "vessel-branch-geometry/1.0.0" or result.get("checksum") != (
        "sha256:" + checksum({k: v for k, v in result.items() if k != "checksum"})
    ):
        # Runner JSON canonicalization is the same sort_keys json default.
        raise ValueError("branch_output_binding")
    if (
        result.get("policy") != POLICY
        or result.get("policy_checksum") != "sha256:" + checksum(POLICY)
        or result.get("unit") != "degree"
        or result.get("research_only") is not True
        or result.get("clinical_threshold") is not None
    ):
        raise ValueError("branch_output_policy")

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def coordinate(point):
        if not isinstance(point, list) or len(point) != 2 or not all(number(v) for v in point):
            raise ValueError("branch_output_coordinate")

    coordinate(result["disc_center_xy"])
    if not number(result["disc_radius_px"]) or result["disc_radius_px"] <= 2:
        raise ValueError("branch_output_disc")
    entries = result["junctions"]
    if len(entries) > 2048 or len({j["junction_id"] for j in entries}) != len(entries):
        raise ValueError("branch_output_junction_identity")
    values = []
    for j in entries:
        coordinate(j["vertex_xy"])
        if type(j["arms"]) is not int or j["arms"] < 3:
            raise ValueError("branch_output_arms")
        value = j["angle_degrees"]
        if j["status"] == "measured":
            if (
                j["reason"] is not None
                or isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 180
                or j["arms"] != 3
                or len(j["daughter_samples_xy"]) != 2
            ):
                raise ValueError("branch_output_measurement")
            values.append(value)
        elif j["status"] != "excluded" or j["reason"] not in REASONS or value is not None:
            raise ValueError("branch_output_exclusion")
        if j["status"] == "measured" or j["reason"] == "scale_topology_or_angle_unstable":
            coordinate(j["root_xy"])
            for line in [j["parent_samples_xy"], *j["daughter_samples_xy"]]:
                if len(line) != 2:
                    raise ValueError("branch_output_line")
                for point in line:
                    coordinate(point)
            vectors = [
                [line[1][d] - line[0][d] for d in range(2)] for line in j["daughter_samples_xy"]
            ]
            norms = [math.hypot(*v) for v in vectors]
            if not all(n > 0 for n in norms):
                raise ValueError("branch_output_degenerate_line")
            cosine = sum(a * b for a, b in zip(*vectors, strict=True)) / math.prod(norms)
            reconstructed = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            original = value if j["status"] == "measured" else j.get("candidate_angle_degrees")
            if not number(original) or not math.isclose(reconstructed, original, abs_tol=1e-8):
                raise ValueError("branch_output_angle_source")
            checks = j.get("scale_checks", [])
            if [s["scale"] for s in checks] != POLICY["scale_check_factors"]:
                raise ValueError("branch_output_stability")
            for s in checks:
                degree = s["angle_degrees"]
                if degree is not None and (not number(degree) or not 0 <= degree <= 180):
                    raise ValueError("branch_output_stability")
                passed = (
                    degree is not None
                    and abs(degree - original) <= POLICY["scale_angle_tolerance_degrees"]
                )
                if s["passed"] is not passed:
                    raise ValueError("branch_output_stability")
            if all(s["passed"] for s in checks) != (j["status"] == "measured"):
                raise ValueError("branch_output_stability")
    if (
        type(result["count"]) is not int
        or result["count"] != len(values)
        or result["median_degrees"] != (statistics.median(values) if values else None)
        or result["status"] != ("measured" if values else "no_valid_bifurcation")
    ):
        raise ValueError("branch_output_summary")
    if values:
        if not number(result["mean_degrees"]) or not math.isclose(
            result["mean_degrees"], statistics.mean(values), abs_tol=1e-10
        ):
            raise ValueError("branch_output_mean")
    elif result["mean_degrees"] is not None:
        raise ValueError("branch_output_mean")
    if "input_bindings" in result:
        bindings = result["input_bindings"]
        if set(bindings) != {"vein_mask", "disc_mask"}:
            raise ValueError("branch_output_inputs")
        shapes = [binding["dimensions_hw"] for binding in bindings.values()]
        if shapes[0] != shapes[1]:
            raise ValueError("branch_output_inputs")
    return result


def resolve(c, vein, disc, output):
    spec = load_verified()
    if rows(c) != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", checksum(spec))]:
        raise ValueError("branch_tool_unavailable_or_changed")
    manifest = c.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", (CODE, VERSION, IMPLEMENTATION)
    ).fetchone()[0]
    if (
        manifest["invocation"] != {"argv": argv(), "shell": False, "timeout_seconds": 60}
        or manifest["working_directory"] != str(ROOT)
        or manifest["formula_spec"] != FORMULA
    ):
        raise ValueError("branch_manifest_drift")
    paths = {
        "{input.vein_mask}": vein.resolve(strict=True),
        "{input.disc_mask}": disc.resolve(strict=True),
        "{output.observation}": output.resolve(),
    }
    if any(not p.is_relative_to("/opt/data/test") for p in paths.values()) or output.exists():
        raise ValueError("branch_path_or_existing_output")
    return {
        "tool": CODE,
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "argv": [str(paths.get(v, v)) for v in argv()],
        "research_only": True,
        "inputs": {"vein_mask": "sha256:" + digest(vein), "disc_mask": "sha256:" + digest(disc)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "register", "show", "run"])
    p.add_argument("--test-evidence", type=Path)
    p.add_argument("--vein-mask", type=Path)
    p.add_argument("--disc-mask", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.action == "prepare":
        if not a.test_evidence:
            p.error("--test-evidence required")
        result = prepare(a.test_evidence)
    else:
        with native_connection() as c:
            if a.action == "register":
                result = register(c, load_verified())
            elif a.action == "show":
                result = c.execute(
                    "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                    (CODE, VERSION, IMPLEMENTATION),
                ).fetchone()[0]
            else:
                if not all([a.vein_mask, a.disc_mask, a.output]):
                    p.error("--vein-mask --disc-mask --output required")
                result = resolve(c, a.vein_mask, a.disc_mask, a.output)
        if a.action == "run" and a.execute:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            done = subprocess.run(
                result["argv"],
                cwd=ROOT,
                timeout=60,
                check=False,
                env={
                    "PATH": str(PYTHON.parent) + ":/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "TMPDIR": str(a.output.parent),
                    "CUDA_VISIBLE_DEVICES": "",
                },
            )
            result["exit_code"] = done.returncode
            if done.returncode == 0:
                output = validate_output(json.loads(a.output.read_bytes()))
                if {k: v["sha256"] for k, v in output["input_bindings"].items()} != result[
                    "inputs"
                ]:
                    raise ValueError("branch_input_changed")
                result["measurement"] = {
                    k: output[k] for k in ["status", "count", "median_degrees", "checksum"]
                }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(done.returncode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

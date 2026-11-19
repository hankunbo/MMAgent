"""Install/call the fixed A/V CPU measurement in the existing medgraph catalog."""

import argparse
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from psycopg.types.json import Jsonb

from Algorithm_Tool.av_vascular_metrics.contract import POLICY, validate
from Algorithm_Tool.av_vascular_metrics.contract import VERSION as SCHEMA
from apps.slo_tool_registry import ROOT, checksum, digest, native_connection
from apps.vessel_branch_tool import PYTHON, environment

CODE = "calculate_slo_av_vascular_metrics"
VERSION = "1.0.0"
IMPLEMENTATION = "aligned_av_cpu_v1"
LOCK = ROOT / "configs/tools/av-vascular-metrics-v1.json"
RUNNER = ROOT / "Algorithm_Tool/av_vascular_metrics/runner.py"
PORTS = [
    ("input", "artery_mask", "slo:ArteryMask"),
    ("input", "vein_mask", "slo:VeinMask"),
    ("input", "disc_mask", "slo:OpticDiscMask"),
    ("output", "observation", "slo:AVVascularMetrics"),
]


def argv():
    return [
        str(PYTHON),
        "-m",
        "Algorithm_Tool.av_vascular_metrics.runner",
        "--artery-mask",
        "{input.artery_mask}",
        "--vein-mask",
        "{input.vein_mask}",
        "--disc-mask",
        "{input.disc_mask}",
        "--output",
        "{output.observation}",
    ]


def prepare(evidence):
    if evidence.stat().st_size > 2_097_152:
        raise ValueError("av_test_size")
    tree = ElementTree.parse(evidence).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
    n = sum(int(s.get("tests", "0")) for s in suites)
    if n < 15 or any(int(s.get(k, "0")) for s in suites for k in ["errors", "failures", "skipped"]):
        raise ValueError("av_passing_tests_required")
    paths = [
        RUNNER,
        RUNNER.with_name("contract.py"),
        RUNNER.with_name("__init__.py"),
        ROOT / "Algorithm_Tool/slo_observations/runner.py",
        Path(__file__),
        ROOT / "apps/vessel_branch_tool.py",
        ROOT / "apps/slo_tool_registry.py",
        PYTHON.resolve(),
        PYTHON.parent.parent / "pyvenv.cfg",
    ]
    spec = {
        "schema": "av-vascular-installation/1.0.0",
        "tool": CODE,
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "formula": POLICY,
        "environment": environment(),
        "resources": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths
        ],
        "test_evidence": {"path": str(evidence.resolve()), "sha256": digest(evidence), "tests": n},
        "clinical_validated": False,
        "runtime_06_bundle_installed": False,
    }
    if LOCK.exists():
        if json.loads(LOCK.read_bytes()) != spec:
            raise ValueError("av_installation_conflict")
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
        POLICY,
    ) or spec["environment"] != environment():
        raise ValueError("av_installation_drift")
    for r in [*spec["resources"], spec["test_evidence"]]:
        if digest(r["path"]) != r["sha256"]:
            raise ValueError("av_resource_drift")
    return spec


def rows(c):
    return c.execute(
        "SELECT v.version,v.enabled,i.implementation_key,i.enabled,i.validation_status,"
        "i.provenance->>'installation_checksum' FROM medgraph.algorithm_tool t "
        "JOIN medgraph.tool_version v ON v.tool_id=t.id JOIN medgraph.tool_implementation i "
        "ON i.tool_version_id=v.id WHERE t.code=%s",
        (CODE,),
    ).fetchall()


def register(c, spec):
    if not c.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("av_native_owner_required")
    c.execute("SELECT pg_advisory_xact_lock(7150931)")
    if c.execute("SELECT count(*) FROM medgraph.algorithm_tool WHERE code=%s", (CODE,)).fetchone()[
        0
    ]:
        if rows(c) != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", checksum(spec))]:
            raise ValueError("av_existing_conflict")
        return {"status": "already_registered", "tool": CODE}
    ids = {}
    for direction, _, code in PORTS:
        schema = (
            {"format": "image/png", "mode": "L", "values": [0, 255], "modality": "SLO"}
            if direction == "input"
            else {"format": "application/json", "schema_version": SCHEMA, "research_only": True}
        )
        old = c.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if old:
            if old[1] != schema:
                raise ValueError("av_artifact_type_conflict")
            ids[code] = old[0]
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
        "input_requirements": "same-case aligned binary artery/vein/disc masks, square pixels; "
        "Runtime verifies provenance",
        "limitations": "native-grid vessel metrics; not perfusion, capillaries, "
        "CRAE/CRVE or population diagnostic cutoffs",
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
            "动静脉分型血管测量",
            "分别计算骨架密度、弧弦迂曲度、分形及盘径相对宽度。",
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
            Jsonb(POLICY),
            Jsonb({"accelerator": "cpu", "required": False}),
            Jsonb({"installation_checksum": checksum(spec)}),
            Jsonb(spec["test_evidence"]),
        ),
    ).fetchone()[0]
    for index, r in enumerate(spec["resources"]):
        c.execute(
            "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
            "resource_role,uri,sha256,byte_size,required) "
            "VALUES(%s,%s,'fixed_local_resource',%s,%s,%s,true)",
            (iid, f"resource_{index}", Path(r["path"]).as_uri(), r["sha256"], r["bytes"]),
        )
    return {"status": "registered", "tool": CODE}


def resolve(c, inputs, output):
    spec = load_verified()
    if rows(c) != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", checksum(spec))]:
        raise ValueError("av_tool_unavailable")
    m = c.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", (CODE, VERSION, IMPLEMENTATION)
    ).fetchone()[0]
    if (
        m["invocation"] != {"argv": argv(), "shell": False, "timeout_seconds": 60}
        or m["formula_spec"] != POLICY
        or m["working_directory"] != str(ROOT)
    ):
        raise ValueError("av_manifest_drift")
    if set(inputs) != {"artery_mask", "vein_mask", "disc_mask"}:
        raise ValueError("av_ports")
    paths = {"{input." + k + "}": v.resolve(strict=True) for k, v in inputs.items()}
    paths["{output.observation}"] = output.resolve()
    if output.exists() or any(not p.is_relative_to("/opt/data/test") for p in paths.values()):
        raise ValueError("av_paths_or_existing_output")
    return {
        "argv": [str(paths.get(v, v)) for v in argv()],
        "tool": CODE,
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "installation_checksum": checksum(spec),
        "input_checksums": {k: "sha256:" + digest(v) for k, v in inputs.items()},
    }


def execute(plan, output):
    done = subprocess.run(
        plan["argv"],
        cwd=ROOT,
        timeout=60,
        check=True,
        env={
            "PATH": str(PYTHON.parent) + ":/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TMPDIR": str(output.parent),
        },
    )
    value = validate(json.loads(output.read_bytes()))
    if {k: v["sha256"] for k, v in value["input_bindings"].items()} != plan["input_checksums"]:
        raise ValueError("av_input_drift_during_execution")
    return value, done.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "register", "show", "run"])
    p.add_argument("--test-evidence", type=Path)
    for port in ["artery", "vein", "disc"]:
        p.add_argument("--" + port + "-mask", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.action == "prepare":
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
                result = resolve(
                    c,
                    {k: getattr(a, k) for k in ["artery_mask", "vein_mask", "disc_mask"]},
                    a.output,
                )
        if a.action == "run" and a.execute:
            _, result["exit_code"] = execute(result, a.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

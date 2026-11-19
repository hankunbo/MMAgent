"""Fixed SegMAN-B component registration in the existing native Tool database."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from psycopg.types.json import Jsonb

from apps.slo_tool_registry import ROOT, checksum, digest, native_connection

DEPLOYMENT = Path("/opt/data/test/segman-slo-deployment")
SOURCE = Path("/opt/data/test/ControlGlaucoma-SegMAN")
PYTHON = Path("/opt/data/test/envs/SegMAN-SLO/bin/python")
RUNNER = ROOT / "Algorithm_Tool/segman_slo/runner.py"
LOCK = ROOT / "configs/tools/segman-slo-resources-v1.json"
CODE = "segman_slo_disc_cup_segmenter"
VERSION = "1.0.0"
IMPLEMENTATION = "local_segman_cuda_v1"
COMMIT = "d51b0e259f87f42253fb92a746e0cbbd5c739813"
WEIGHT_SHA = "66df9d0d9785cf443e5d3c31e3650c5e02d96a13674db40ed15ea44a00c5ac0d"
PORTS = {
    "image": ("input", "slo:ImagePNG"),
    "cup_mask": ("output", "slo:OpticCupMask"),
    "rim_mask": ("output", "slo:OpticRimMask"),
    "disc_mask": ("output", "slo:OpticDiscMask"),
    "labels": ("output", "slo:DiscCupLabels"),
    "overlay": ("output", "slo:DiscCupOverlay"),
    "observation": ("output", "slo:SegMANObservation"),
}


def prepare():
    """Read-only verification, then create an immutable resource declaration."""
    handoff = json.loads((DEPLOYMENT / "handoff.json").read_bytes())
    final = json.loads((DEPLOYMENT / "artifacts/20260917/final-verification.json").read_bytes())
    for name, expected in final["delivery_file_sha256"].items():
        if digest(DEPLOYMENT / name) != expected:
            raise ValueError("segman_delivery_drift")
    weight = DEPLOYMENT / "weights/segman_b.pth"
    if weight.stat().st_size != 622904973 or digest(weight) != WEIGHT_SHA:
        raise ValueError("segman_weight_drift")
    head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True)
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"])
    if head.strip() != COMMIT or dirty:
        raise ValueError("segman_source_drift")
    for key in ("missing_keys", "unexpected_keys", "shape_mismatches", "dtype_mismatches"):
        if handoff["cpu_state_match"][key]:
            raise ValueError("segman_incomplete_state")
    files = [RUNNER, weight, DEPLOYMENT / "scripts/segman_slo.py", PYTHON.resolve()]
    files += [DEPLOYMENT / n for n in final["delivery_file_sha256"]]
    files += [DEPLOYMENT / "artifacts/20260917/final-verification.json"]
    files += list((SOURCE / "SegMAN/segmentation").rglob("*.py"))
    files += [SOURCE / "train/vcdr_loss.py"]
    files += [Path(p) for p in handoff["runtime_environment"]["native_extensions"].values()]
    files += list(
        (PYTHON.parent.parent / "lib/python3.10/site-packages/mmcv/parallel").glob("*.py")
    )
    files += [Path(handoff["single_image"]["final_result"])]
    files += [
        Path(handoff["fairseg"]["manifest"]),
        Path(handoff["fairseg"]["predictions_complete"]),
    ]
    metric_path = DEPLOYMENT / "results/fairseg-check-001/evaluation/metrics.json"
    files += [metric_path]
    receipts = {}
    for name in ("single-image", "fairseg-predict", "fairseg-evaluate"):
        path = DEPLOYMENT / ("artifacts/20260917/" + name + "-resources.json")
        receipt = json.loads(path.read_bytes())
        if receipt["exit_code"] != 0:
            raise ValueError("segman_process_evidence_failed")
        receipts[name] = {"path": str(path), "sha256": digest(path), "exit_code": 0}
        files.append(path)
    metrics = json.loads(metric_path.read_bytes())
    if (
        metrics["evaluation_count"] != 8
        or not metrics["ground_truth_loaded_only_after_all_predictions"]
    ):
        raise ValueError("segman_evaluation_evidence")
    averages = {
        name: sum(row["metrics"][name]["dice"] for row in metrics["files"]) / 8
        for name in ("cup", "rim", "disc")
    }
    spec = {
        "schema_version": "segman-component-installation/1.0.0",
        "source_commit": COMMIT,
        "checkpoint_sha256": WEIGHT_SHA,
        "environment": handoff["runtime_environment"],
        "output_contract": handoff["output_contract"],
        "resources": [
            {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(set(files))
        ],
        "validation_report": {
            "installation_handoff": str(DEPLOYMENT / "handoff.json"),
            "same_case_result": handoff["single_image"]["final_result"],
            "same_case_input_sha256": handoff["single_image"]["input_sha256"],
            "process_receipts": receipts,
            "fairseg_count": 8,
            "fairseg_mean_dice": averages,
            "fairseg_metrics": str(
                DEPLOYMENT / "results/fairseg-check-001/evaluation/metrics.json"
            ),
            "clinical_validated": False,
            "training_overlap": "unknown",
            "runtime_06_bundle_installed": False,
            "scope": "native_registered_research_component",
        },
    }
    if LOCK.exists():
        if json.loads(LOCK.read_bytes()) != spec:
            raise ValueError("segman_existing_lock_conflict")
    else:
        with LOCK.open("x") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return spec


def load_verified():
    spec = json.loads(LOCK.read_bytes())
    if spec["source_commit"] != COMMIT or spec["checkpoint_sha256"] != WEIGHT_SHA:
        raise ValueError("segman_fixed_identity_mismatch")
    for item in spec["resources"]:
        p = Path(item["path"])
        if p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
            raise ValueError("segman_resource_drift")
    return spec


def argv_template():
    return [
        str(PYTHON),
        str(RUNNER),
        "--image",
        "{input.image}",
        "--observation",
        "{output.observation}",
    ]


def register(connection, spec):
    """Transactional insertion; repeats never change or re-enable existing facts."""
    if not connection.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("native_owner_required")
    connection.execute("SELECT pg_advisory_xact_lock(7150926)")
    seal = checksum(spec)
    old = connection.execute(
        "SELECT v.version,v.enabled,i.implementation_key,i.enabled,i.validation_status,"
        "i.provenance->>'installation_checksum' FROM medgraph.algorithm_tool t "
        "JOIN medgraph.tool_version v ON v.tool_id=t.id "
        "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id WHERE t.code=%s",
        (CODE,),
    ).fetchall()
    if old:
        if old != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", seal)]:
            raise ValueError("segman_existing_state_or_content_conflict")
        return {"status": "already_registered", "tool": CODE, "installation_checksum": seal}
    types = {}
    for port, (_direction, code) in PORTS.items():
        schema = {"format": "image/png", "mode": "L", "values": [0, 255], "modality": "SLO"}
        if port == "image":
            schema = {
                "format": "image/png",
                "modality": "SLO",
                "mode": "L",
                "runtime_artifact_code": "ARTIFACT:SLO_IMAGE",
            }
        elif port == "labels":
            schema = {
                "format": "image/png",
                "mode": "L",
                "values": [0, 1, 2],
                "classes": ["background", "rim", "cup"],
            }
        elif port == "overlay":
            schema = {"format": "image/png", "mode": "RGB", "purpose": "visualization"}
        elif port == "observation":
            schema = {"format": "application/json", "producer": CODE, "version": VERSION}
        old_type = connection.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if old_type:
            if old_type[1] != schema:
                raise ValueError("segman_artifact_type_conflict")
            types[code] = old_type[0]
        else:
            types[code] = connection.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, port, Jsonb(schema)),
            ).fetchone()[0]
    metadata = {
        "intended_use": "research_only",
        "decisive_evidence": False,
        "input_modality": "SLO",
        "installation_checksum": seal,
    }
    tid = connection.execute(
        "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
        "VALUES(%s,%s,'segmenter',%s,%s) RETURNING id",
        (
            CODE,
            "SegMAN-B SLO视杯/盘沿/视盘分割",
            "固定三类SLO任务权重；原尺寸掩膜；单GPU串行研究组件",
            Jsonb(metadata),
        ),
    ).fetchone()[0]
    runtime = {
        "manager": "venv",
        "prefix": str(PYTHON.parent.parent),
        "python_executable": str(PYTHON),
        "device": "cuda:0",
        **spec["environment"],
    }
    vid = connection.execute(
        "INSERT INTO medgraph.tool_version(tool_id,version,runtime,metadata) "
        "VALUES(%s,%s,%s,%s) RETURNING id",
        (
            tid,
            VERSION,
            Jsonb(runtime),
            Jsonb(
                {
                    **metadata,
                    "parameter_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ),
        ),
    ).fetchone()[0]
    for port, (direction, code) in PORTS.items():
        connection.execute(
            "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,artifact_type_id,"
            "required,cardinality_min,cardinality_max,constraints) "
            "VALUES(%s,%s,%s,%s,true,1,1,%s)",
            (
                vid,
                direction,
                port,
                types[code],
                Jsonb({"research_only": True, "same_input_dimensions": True}),
            ),
        )
    iid = connection.execute(
        "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
        "implementation_kind,entrypoint_uri,working_directory,runtime_environment,invocation,"
        "hardware_requirements,provenance,validation_status,validation_report) "
        "VALUES(%s,%s,'python_cli',%s,%s,%s,%s,%s,%s,'smoke_tested',%s) RETURNING id",
        (
            vid,
            IMPLEMENTATION,
            RUNNER.as_uri(),
            str(ROOT),
            Jsonb(runtime),
            Jsonb({"argv": argv_template(), "shell": False, "timeout_seconds": 600}),
            Jsonb(
                {
                    "accelerator": "cuda",
                    "required": True,
                    "concurrency": 1,
                    "strong_isolation": False,
                }
            ),
            Jsonb(
                {
                    "installation_checksum": seal,
                    "source_commit": COMMIT,
                    "checkpoint_sha256": WEIGHT_SHA,
                    "output_contract": spec["output_contract"],
                }
            ),
            Jsonb(spec["validation_report"]),
        ),
    ).fetchone()[0]
    for index, item in enumerate(spec["resources"]):
        connection.execute(
            "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
            "resource_role,uri,sha256,byte_size,required) "
            "VALUES(%s,%s,'fixed_local_resource',%s,%s,%s,true)",
            (
                iid,
                f"resource_{index:04d}",
                Path(item["path"]).as_uri(),
                item["sha256"],
                item["bytes"],
            ),
        )
    return {"status": "registered", "tool": CODE, "installation_checksum": seal}


def manifest(connection):
    state = connection.execute(
        "SELECT v.enabled,i.enabled,i.validation_status FROM medgraph.tool_implementation i "
        "JOIN medgraph.tool_version v ON v.id=i.tool_version_id "
        "JOIN medgraph.algorithm_tool t ON t.id=v.tool_id "
        "WHERE t.code=%s AND v.version=%s AND i.implementation_key=%s",
        (CODE, VERSION, IMPLEMENTATION),
    ).fetchone()
    if state != (True, True, "smoke_tested"):
        raise ValueError("segman_not_registered_or_disabled")
    return connection.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", (CODE, VERSION, IMPLEMENTATION)
    ).fetchone()[0]


def resolve_component(connection, image, directory):
    from apps.glaucoma_case_evidence import png_shape

    spec = load_verified()
    m = manifest(connection)
    if (
        m["provenance"]["installation_checksum"] != checksum(spec)
        or m["invocation"].get("argv") != argv_template()
        or m["invocation"].get("shell") is not False
        or m["working_directory"] != str(ROOT)
    ):
        raise ValueError("segman_manifest_drift")
    image, directory = image.resolve(strict=True), directory.resolve()
    if not image.is_relative_to("/opt/data/test") or not directory.is_relative_to(ROOT / ".local"):
        raise ValueError("segman_component_path_scope")
    if image.stat().st_size > 4_194_304 or directory.exists():
        raise ValueError("segman_input_bound_or_existing_output")
    shape = png_shape(image.read_bytes())
    prediction = directory / "prediction"
    outputs = {p + "_mask": str(prediction / (p + ".png")) for p in ("cup", "rim", "disc")}
    outputs.update({p: str(prediction / (p + ".png")) for p in ("labels", "overlay")})
    outputs["observation"] = str(prediction / "result.json")
    return {
        "tool": CODE,
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "research_only": True,
        "scope": "native_component_not_patient_runtime",
        "argv": [
            str(PYTHON),
            str(RUNNER),
            "--image",
            str(image),
            "--observation",
            outputs["observation"],
        ],
        "outputs": outputs,
        "shape_hw": list(shape),
        "input_sha256": digest(image),
        "manifest_checksum": checksum(m),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "register", "show", "run"))
    p.add_argument("--input", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.action == "prepare":
        spec = prepare()
        print(
            json.dumps(
                {"installation_checksum": checksum(spec), "resources": len(spec["resources"])}
            )
        )
        return
    with native_connection() as c:
        if args.action == "run":
            if not args.input or not args.output_dir:
                p.error("run requires --input and --output-dir")
            result = resolve_component(c, args.input, args.output_dir)
        else:
            result = register(c, load_verified()) if args.action == "register" else manifest(c)
    if args.action == "run" and args.execute:
        environment = {
            "PATH": str(PYTHON.parent) + ":/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        child = subprocess.Popen(result["argv"], cwd=ROOT, env=environment, start_new_session=True)
        try:
            code = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)
            code = 124
        result["exit_code"] = code
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(code)
    if args.action == "run":
        result["execute"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

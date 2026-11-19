"""Register seven fixed SLO observation tools in the EXISTING local medgraph.

No service start, schema migration, credential output, or clinical qualification.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from apps.research_single_case import credential_boundary

ROOT = Path("/opt/data/test/MMAgent")
SOURCE = Path("/opt/data/test/SLOctolyzer")
PYTHON = Path("/opt/data/test/envs/SLOctolyzer/bin/python")
RUNNER = ROOT / "Algorithm_Tool/slo_observations/runner.py"
LOCK = ROOT / "configs/tools/sloctolyzer-resources-v1.json"
EVIDENCE = ROOT / "docs/experiments/slo-tools-v1/verification.json"
VERSION = "1.0.0"
IMPLEMENTATION = "local_slo_cpu_v1"
TOOLS = {
    "vessels": (
        "sloctolyzer_vessel_segmenter",
        "SLO血管分割",
        "segmenter",
        [("vessel_mask", "slo:VesselMask")],
    ),
    "avod": (
        "sloctolyzer_avod_segmenter",
        "SLO动静脉与视盘分割",
        "segmenter",
        [
            ("artery_mask", "slo:ArteryMask"),
            ("vein_mask", "slo:VeinMask"),
            ("disc_mask", "slo:OpticDiscMask"),
        ],
    ),
    "fovea": (
        "sloctolyzer_fovea_detector",
        "SLO中央凹定位",
        "detector",
        [("fovea_map", "slo:FoveaHeatmap")],
    ),
    "density": ("calculate_slo_vessel_density", "SLO血管面积占比", "calculator", []),
    "width": ("calculate_slo_global_calibre", "SLO全局像素血管宽度", "calculator", []),
    "tortuosity": ("calculate_slo_arc_chord_tortuosity", "SLO弧弦比曲折度", "calculator", []),
    "fractal": ("calculate_slo_box_dimension", "SLO盒计数分形维数", "calculator", []),
}
FORMULAS = {
    "density": {"expression": "N_vessel/(height*width)", "unit": "dimensionless"},
    "width": {"expression": "N_vessel/N_zhang_skeleton", "unit": "pixel"},
    "tortuosity": {
        "expression": "median(sum(norm(p[i+1]-p[i]))/norm(p[-1]-p[0]))",
        "unit": "dimensionless",
        "path_policy": "8-neighbour-no-diagonal-shortcut",
        "min_length_px": 10,
        "closed_or_short_segments": "excluded_and_counted",
    },
    "fractal": {
        "expression": "-slope(log(box_size),log(nonempty_nonfull_box_count))",
        "unit": "dimensionless",
        "scales": "2**n down to 4",
        "min_scales": 3,
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare():
    files = [RUNNER, SOURCE / "requirements.txt", PYTHON.resolve()]
    files += sorted((SOURCE / "sloctolyzer").rglob("*.py"))
    files += [
        SOURCE / "weights" / (n + "_weights.pth")
        for n in ("slosegmenter", "avosegmenter", "fovsegmenter")
    ]
    files += [SOURCE / "outputs/fairseg_smoke/summary.json"]
    environment = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import importlib.metadata as m,json; "
            "print(json.dumps({n:m.version(n) for n in "
            '["torch","torchvision","numpy","scikit-image","segmentation-models-pytorch","timm"]}))',
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    lock = {
        "schema_version": "slo-installation/1.0.0",
        "source_repository": "https://github.com/jaburke166/SLOctolyzer",
        "upstream_commit": subprocess.run(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip(),
        "local_changes": "Three loaders use weights_only=False for official full-model files; "
        "hash-bound trusted local CPU deployment, not arbitrary weights.",
        "environment": json.loads(environment.stdout),
        "resources": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files
        ],
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        if json.loads(LOCK.read_text()) != lock:
            raise ValueError("existing_installation_lock_conflict_use_new_version")
    else:
        LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    return {"resource_lock": str(LOCK), "resources": len(files), "database_written": False}


def specification():
    lock = json.loads(LOCK.read_text())
    for r in lock["resources"]:
        if Path(r["path"]).stat().st_size != r["bytes"] or digest(r["path"]) != r["sha256"]:
            raise ValueError("resource_drift")
    evidence = json.loads(EVIDENCE.read_text())
    if evidence.get("status") != "passed" or set(evidence.get("tasks", [])) != set(TOOLS):
        raise ValueError("seven_tool_verification_required")
    for record in evidence["commands"]:
        if record["exit_code"] != 0:
            raise ValueError("failed_verification_command")
    for record in evidence["files"]:
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("verification_output_drift")
    return {
        "lock": lock,
        "lock_sha256": digest(LOCK),
        "version": VERSION,
        "evidence": str(EVIDENCE),
        "evidence_sha256": digest(EVIDENCE),
        "tools": {k: list(v) for k, v in TOOLS.items()},
        "formulas": FORMULAS,
    }


def native_connection():
    state, values = credential_boundary().inputs()
    connection = psycopg.connect(
        host=state["host"],
        port=state["port"],
        dbname="medgraph",
        user=values["POSTGRES_USER"],
        password=values["POSTGRES_PASSWORD"],
        connect_timeout=3,
    )
    try:
        connection.execute("SET LOCAL statement_timeout='15000'")
        connection.execute("SET LOCAL lock_timeout='5000'")
        if (
            connection.execute(
                "SELECT system_identifier::text FROM pg_control_system()"
            ).fetchone()[0]
            != state["cluster_identifier"]
        ):
            raise ValueError("original_database_identity_mismatch")
        return connection
    except Exception:
        connection.close()
        raise


def rows(connection):
    return connection.execute(
        "SELECT tool_code,version,implementation_key,enabled,validation_status "
        "FROM medgraph.v_tool_execution_manifest WHERE tool_code=ANY(%s) ORDER BY tool_code",
        ([v[0] for v in TOOLS.values()],),
    ).fetchall()


def resolve_component(connection, task, image, output_dir):
    """Reuse native resource checks; resolve only this fixed venv's exact argv.

    The legacy executor resolves the Python symlink and both rejects /opt/conda
    and would lose the venv. Keep the verified venv entry path, no global allowlist change.
    """
    code, _, _, extra = TOOLS[task]
    states = connection.execute(
        "SELECT v.enabled,i.enabled,i.validation_status FROM medgraph.tool_implementation i "
        "JOIN medgraph.tool_version v ON v.id=i.tool_version_id "
        "JOIN medgraph.algorithm_tool t ON t.id=v.tool_id "
        "WHERE t.code=%s AND v.version=%s AND i.implementation_key=%s",
        (code, VERSION, IMPLEMENTATION),
    ).fetchone()
    if states != (True, True, "smoke_tested"):
        raise ValueError("component_not_registered_or_disabled")
    manifest = connection.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", (code, VERSION, IMPLEMENTATION)
    ).fetchone()[0]
    executor_path = ROOT / "algorithm-capability-db/executor/run_tool.py"
    module_spec = importlib.util.spec_from_file_location("native_tool_executor", executor_path)
    executor = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(executor)
    executor._validate_resources(manifest, verify_hashes=True)
    image = image.resolve(strict=True)
    output_dir = output_dir.resolve()
    if not image.is_relative_to("/opt/data/test") or not output_dir.is_relative_to(
        "/opt/data/test"
    ):
        raise ValueError("component_path_outside_local_root")
    inp = "vessel_mask" if task in FORMULAS else "image"
    outputs = {"observation": str(output_dir / (task + ".json"))}
    outputs.update(
        {p: str(output_dir / (p + (".npy" if p == "fovea_map" else ".png"))) for p, _ in extra}
    )
    if any(Path(p).exists() for p in outputs.values()):
        raise ValueError("output_already_exists")
    template = [
        str(PYTHON),
        str(RUNNER),
        task,
        "--input",
        "{input." + inp + "}",
        "--output",
        "{output.observation}",
    ]
    for port, _ in extra:
        template += ["--" + port.replace("_", "-"), "{output." + port + "}"]
    if (
        manifest["invocation"].get("argv") != template
        or manifest["invocation"].get("shell") is not False
    ):
        raise ValueError("fixed_component_invocation_mismatch")
    # Preserve the lexical venv executable. Only this exact template is accepted.
    expected = [
        str(PYTHON),
        str(RUNNER),
        task,
        "--input",
        str(image),
        "--output",
        outputs["observation"],
    ]
    for port, _ in extra:
        expected += ["--" + port.replace("_", "-"), outputs[port]]
    if manifest["working_directory"] != str(ROOT):
        raise ValueError("fixed_component_invocation_mismatch")
    return {
        "tool": code,
        "version": VERSION,
        "argv": expected,
        "outputs": outputs,
        "research_only": True,
        "execution_scope": "native_component_not_patient_runtime",
    }


def register(connection, spec):
    """Caller owns transaction; no upsert can re-enable withdrawn/changed tools."""
    if not connection.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("native_owner_required")
    connection.execute("SELECT pg_advisory_xact_lock(7150926)")
    seal = checksum(spec)
    existing = rows(connection)
    if existing:
        if len(existing) != 7 or any(
            r[1:] != (VERSION, IMPLEMENTATION, True, "smoke_tested") for r in existing
        ):
            raise ValueError("existing_tool_state_conflict")
        saved = connection.execute(
            "SELECT i.provenance->>'installation_checksum' "
            "FROM medgraph.tool_implementation i "
            "JOIN medgraph.tool_version v ON v.id=i.tool_version_id "
            "JOIN medgraph.algorithm_tool t ON t.id=v.tool_id WHERE t.code=ANY(%s)",
            ([v[0] for v in TOOLS.values()],),
        ).fetchall()
        if any(row[0] != seal for row in saved):
            raise ValueError("existing_installation_content_conflict")
        return {"status": "already_registered", "tools": existing}

    types = {
        "slo:ImagePNG": (
            "SLO灰度图",
            {
                "format": "image/png",
                "modality": "SLO",
                "mode": "L",
                "runtime_artifact_code": "ARTIFACT:SLO_IMAGE",
            },
        ),
        "slo:FoveaHeatmap": (
            "中央凹模型响应图",
            {
                "format": "application/x-npy",
                "dtype": "float32",
                "range": [0, 1],
                "calibration": "unvalidated",
            },
        ),
        "slo:Observation": (
            "SLO研究观察JSON",
            {
                "format": "application/json",
                "schema_version": "slo-observation/1.0.0",
                "clinical_effect": None,
            },
        ),
    }
    for code in ("VesselMask", "ArteryMask", "VeinMask", "OpticDiscMask"):
        types["slo:" + code] = (
            code,
            {"format": "image/png", "mode": "L", "values": [0, 255], "modality": "SLO"},
        )
    ids = {}
    for code, (name, schema) in types.items():
        prior = connection.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if prior:
            if prior[1] != schema:
                raise ValueError("artifact_type_conflict")
            ids[code] = prior[0]
        else:
            ids[code] = connection.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, name, Jsonb(schema)),
            ).fetchone()[0]
    for task, (code, name, kind, extra) in TOOLS.items():
        metadata = {
            "intended_use": "research_only",
            "decisive_evidence": False,
            "input_modality": "SLO",
            "installation_checksum": seal,
        }
        if task == "vessels":
            metadata["model_dependencies"] = ["slosegmenter", "avosegmenter"]
            metadata["postprocessing"] = "official_binary_union_AV_minus_disc"
        tid = connection.execute(
            "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
            "VALUES(%s,%s,%s,%s,%s) RETURNING id",
            (code, name, kind, "CPU组件工具；不生成青光眼诊断或医学支持度", Jsonb(metadata)),
        ).fetchone()[0]
        runtime = {
            "manager": "venv",
            "prefix": str(PYTHON.parent.parent),
            "python_executable": str(PYTHON),
            "device": "cpu",
            **spec["lock"]["environment"],
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
        input_type = "slo:VesselMask" if task in FORMULAS else "slo:ImagePNG"
        ports = [
            ("input", "image" if task not in FORMULAS else "vessel_mask", input_type),
            ("output", "observation", "slo:Observation"),
        ]
        ports += [("output", port, artifact) for port, artifact in extra]
        for direction, port, artifact in ports:
            connection.execute(
                "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,"
                "artifact_type_id,required,cardinality_min,cardinality_max,constraints) "
                "VALUES(%s,%s,%s,%s,true,1,1,%s)",
                (vid, direction, port, ids[artifact], Jsonb({"research_only": True})),
            )
        argv = [
            str(PYTHON),
            str(RUNNER),
            task,
            "--input",
            "{input." + ports[0][1] + "}",
            "--output",
            "{output.observation}",
        ]
        for port, _ in extra:
            argv += ["--" + port.replace("_", "-"), "{output." + port + "}"]
        iid = connection.execute(
            "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
            "implementation_kind,entrypoint_uri,working_directory,runtime_environment,invocation,formula_spec,"
            "hardware_requirements,provenance,validation_status,validation_report) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'smoke_tested',%s) RETURNING id",
            (
                vid,
                IMPLEMENTATION,
                "formula_python" if task in FORMULAS else "python_cli",
                RUNNER.as_uri(),
                str(ROOT),
                Jsonb(runtime),
                Jsonb({"argv": argv, "shell": False, "timeout_seconds": 180}),
                Jsonb(FORMULAS[task]) if task in FORMULAS else None,
                Jsonb({"accelerator": "cpu", "required": False}),
                Jsonb(
                    {
                        "installation_checksum": seal,
                        "repository": spec["lock"]["source_repository"],
                        "source_commit": spec["lock"]["upstream_commit"],
                        "resource_lock_sha256": spec["lock_sha256"],
                    }
                ),
                Jsonb(
                    {
                        "status": "component_smoke_passed",
                        "clinical_validated": False,
                        "segmentation_accuracy_evaluated": False,
                        "runtime_06_bundle_installed": False,
                        "evidence": spec["evidence"],
                        "sha256": spec["evidence_sha256"],
                        "original_model_smoke": "FairSeg Test 08001..08003; CPU; no metrics/Dice",
                        "integration_smoke": "existing exported 08001 PNG; no labels read",
                    }
                ),
            ),
        ).fetchone()[0]
        resources = spec["lock"]["resources"] + [
            {"path": str(LOCK), "sha256": spec["lock_sha256"], "bytes": LOCK.stat().st_size}
        ]
        for index, item in enumerate(resources):
            connection.execute(
                "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
                "resource_role,uri,sha256,byte_size,required) "
                "VALUES(%s,%s,'fixed_local_resource',%s,%s,%s,true)",
                (
                    iid,
                    f"resource_{index:03}",
                    Path(item["path"]).as_uri(),
                    item["sha256"],
                    item["bytes"],
                ),
            )
    return {"status": "registered", "tools": rows(connection)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "register", "list", "show", "run"])
    parser.add_argument("--task", choices=list(TOOLS))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    else:
        spec = specification() if args.action == "register" else None
        with native_connection() as connection:
            if args.action == "register":
                result = register(connection, spec)
            else:
                if args.action == "run":
                    if not args.task or not args.input or not args.output_dir:
                        raise ValueError("task_input_output_required")
                    result = resolve_component(connection, args.task, args.input, args.output_dir)
                elif args.action == "show":
                    if not args.task:
                        raise ValueError("task_required")
                    result = connection.execute(
                        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                        (TOOLS[args.task][0], VERSION, IMPLEMENTATION),
                    ).fetchone()[0]
                else:
                    result = {"tools": rows(connection)}
        # Emitted only after the transaction committed successfully.
        if args.action == "run" and args.execute:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            environment = {
                "PATH": str(PYTHON.parent) + ":/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "MPLBACKEND": "Agg",
                "TMPDIR": str(args.output_dir.resolve()),
                "XDG_CACHE_HOME": str(args.output_dir.resolve() / "cache"),
                "MPLCONFIGDIR": str(args.output_dir.resolve() / "cache/matplotlib"),
                "OMP_NUM_THREADS": "8",
            }
            completed = subprocess.run(
                result["argv"], cwd=ROOT, env=environment, timeout=180, check=False
            )
            result["exit_code"] = completed.returncode
            if completed.returncode:
                print(json.dumps(result, ensure_ascii=False))
                raise SystemExit(completed.returncode)
        elif args.action == "run":
            result["execute"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, psycopg.Error) as exc:
        print(json.dumps({"status": "not_completed", "error_type": type(exc).__name__}))
        raise SystemExit(2) from None

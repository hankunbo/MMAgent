"""Research-only native registration; no formal 06/11 publication or efficacy claim."""

import json
from pathlib import Path

from psycopg.types.json import Jsonb

from apps.automorph_resources import ROOT as SOURCE
from apps.slo_tool_registry import ROOT, checksum, native_connection
from apps.vectorengine_single_case import write_once
from packages.multimodal_research.fundus_worker import verify_resources
from packages.object_store import checksum as content_checksum

CODE = "automorph_fundus_disc_cup_ensemble"
VERSION = "1.0.0"
IMPLEMENTATION = "local_automorph_eight_wnet_v1"
LOCK = ROOT / "configs/tools/automorph-fundus-v1.json"
PYTHON = "/opt/conda/bin/python"


def argv(image, output):
    return [
        PYTHON,
        "-m",
        "packages.multimodal_research.fundus_worker",
        "--image",
        str(image),
        "--output",
        str(output),
    ]


def prepare(smoke, evaluation):
    smoke, evaluation = Path(smoke).absolute(), Path(evaluation).absolute()
    result = json.loads((smoke / "result.json").read_bytes())
    metrics = json.loads(evaluation.read_bytes())
    if (
        result["status"] != "predictions_saved"
        or metrics["status"] != "evaluated"
        or not metrics["truth_loaded_after_prediction"]
    ):
        raise ValueError("fundus_onboarding_evidence_missing")
    source = verify_resources()
    paths = [SOURCE / name for name in source["files"]]
    paths += [
        SOURCE / "ensemble-source-lock.json",
        Path(__file__).absolute(),
        ROOT / "apps/automorph_resources.py",
        ROOT / "packages/multimodal_research/fundus_worker.py",
        ROOT / "packages/multimodal_research/fundus_measurements.py",
        evaluation,
        smoke / "result.json",
    ]
    paths += list(smoke.glob("*.png"))
    spec = dict(
        schema_version="fundus-tool-installation/1.0.0",
        code=CODE,
        version=VERSION,
        source=source,
        runtime=result["runtime"],
        resources={str(p): content_checksum(p.read_bytes()) for p in sorted(paths)},
        validation_report=dict(
            smoke=str(smoke),
            evaluation=str(evaluation),
            metrics=metrics,
            intended_use="research_only",
            clinical_validated=False,
            decisive_evidence=False,
            training_overlap=result["training_overlap"],
            formal_06_bundle_installed=False,
        ),
    )
    write_once(LOCK, spec)
    return spec


def load_verified():
    spec = json.loads(LOCK.read_bytes())
    if (spec["code"], spec["version"], spec["source"]) != (CODE, VERSION, verify_resources()):
        raise ValueError("fundus_installation_identity")
    for path, expected in spec["resources"].items():
        if content_checksum(Path(path).read_bytes()) != expected:
            raise ValueError("fundus_installation_resource_changed")
    return spec


def register(connection):
    spec = load_verified()
    if not connection.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("native_owner_required")
    connection.execute("SELECT pg_advisory_xact_lock(7150926)")
    existing = connection.execute(
        "SELECT id FROM medgraph.algorithm_tool WHERE code=%s", (CODE,)
    ).fetchone()
    if existing:
        manifest(connection)
        return dict(status="already_registered", tool=CODE)
    metadata = dict(
        intended_use="research_only",
        decisive_evidence=False,
        input_modality="color_fundus",
        installation_checksum=checksum(spec),
    )
    tid = connection.execute(
        "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
        "VALUES(%s,%s,'segmenter',%s,%s) RETURNING id",
        (
            CODE,
            "AutoMorph彩色眼底八模型杯盘分割",
            "模型子流程；不是完整AutoMorph，不作分期分类；质量加权研究观察",
            Jsonb(metadata),
        ),
    ).fetchone()[0]
    runtime = dict(python_executable=PYTHON, **spec["runtime"])
    vid = connection.execute(
        "INSERT INTO medgraph.tool_version(tool_id,version,runtime,metadata) "
        "VALUES(%s,%s,%s,%s) RETURNING id",
        (tid, VERSION, Jsonb(runtime), Jsonb(metadata)),
    ).fetchone()[0]
    for port, direction, schema in (
        ("image", "input", dict(format="image/jpeg", modality="color_fundus", mode="RGB")),
        (
            "prediction_bundle",
            "output",
            dict(
                format="application/json",
                schema_version="automorph-fundus-measurement/1.0.0",
                masks="512_grid_unmodified_labels",
            ),
        ),
    ):
        code = "fundus:AutoMorph" + port
        old = connection.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if old and old[1] != schema:
            raise ValueError("fundus_artifact_contract_conflict")
        aid = (
            old[0]
            if old
            else connection.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, port, Jsonb(schema)),
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,artifact_type_id,"
            "required,cardinality_min,cardinality_max,constraints) "
            "VALUES(%s,%s,%s,%s,true,1,1,%s)",
            (vid, direction, port, aid, Jsonb(dict(research_only=True))),
        )
    iid = connection.execute(
        "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
        "implementation_kind,entrypoint_uri,working_directory,runtime_environment,invocation,"
        "hardware_requirements,provenance,validation_status,validation_report) "
        "VALUES(%s,%s,'python_cli',%s,%s,%s,%s,%s,%s,'smoke_tested',%s) RETURNING id",
        (
            vid,
            IMPLEMENTATION,
            (ROOT / "packages/multimodal_research/fundus_worker.py").as_uri(),
            str(ROOT),
            Jsonb(runtime),
            Jsonb(
                dict(
                    argv=argv("{input.image}", "{output.directory}"),
                    shell=False,
                    timeout_seconds=180,
                )
            ),
            Jsonb(dict(accelerator="cuda", required=False, concurrency=1)),
            Jsonb(dict(installation_checksum=checksum(spec))),
            Jsonb(spec["validation_report"]),
        ),
    ).fetchone()[0]
    for index, (name, sha) in enumerate(spec["resources"].items()):
        path = Path(name)
        connection.execute(
            "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
            "resource_role,uri,sha256,byte_size,required) "
            "VALUES(%s,%s,'fixed_local_resource',%s,%s,%s,true)",
            (iid, f"resource_{index:04d}", path.as_uri(), sha[7:], path.stat().st_size),
        )
    manifest(connection)
    return dict(
        status="registered", tool=CODE, validation_status="smoke_tested", clinical_validated=False
    )


def manifest(connection):
    spec = load_verified()
    state = connection.execute(
        "SELECT v.enabled,i.enabled,i.validation_status FROM medgraph.tool_implementation i "
        "JOIN medgraph.tool_version v ON v.id=i.tool_version_id "
        "JOIN medgraph.algorithm_tool t ON t.id=v.tool_id "
        "WHERE t.code=%s AND v.version=%s AND i.implementation_key=%s",
        (CODE, VERSION, IMPLEMENTATION),
    ).fetchone()
    if state != (True, True, "smoke_tested"):
        raise ValueError("fundus_tool_not_current")
    result = connection.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", (CODE, VERSION, IMPLEMENTATION)
    ).fetchone()[0]
    if result["provenance"]["installation_checksum"] != checksum(spec) or result[
        "invocation"
    ] != dict(argv=argv("{input.image}", "{output.directory}"), shell=False, timeout_seconds=180):
        raise ValueError("fundus_tool_manifest_changed")
    return result


if __name__ == "__main__":
    with native_connection() as connection:
        print(json.dumps(register(connection)))

"""Two fixed CPU algorithms in the existing native medgraph Tool catalog."""

import argparse
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from psycopg.types.json import Jsonb

from apps import disc_geometry
from apps.slo_tool_registry import ROOT, checksum, digest, native_connection

PYTHON = Path("/opt/data/test/envs/MMAgent1/bin/python")
VERSION = "1.0.0"
IMPLEMENTATION = "local_disc_geometry_cpu_v1"
LOCK = ROOT / "configs/tools/disc-geometry-v1.json"
TOOLS = {
    "rim": (
        "calculate_slo_rim_directional_width",
        "盘沿方向宽度与分布测量",
        "slo:RimDirectionalGeometry",
    ),
    "ellipse": ("calculate_slo_disc_ellipse", "视盘椭圆度与方向", "slo:DiscEllipseGeometry"),
}
SOURCES = {
    "rim": [
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/17102005/",
            "relation": "ISNT照片四方向宽度；不等式I≥S≥N≥T",
        },
        {
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5705386/",
            "relation": "正常眼亦常不满足ISNT；宽度、面积和OCT厚度不可混用",
        },
    ],
    "ellipse": [
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/15733984/",
            "relation": "年轻近视人群：照片椭圆度关联屈光度和眼轴；不是青光眼诊断阈值",
        },
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/26505465/",
            "relation": "近视青光眼随访：倾斜组进展较少",
        },
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/28114308/",
            "relation": "近视OAG双眼配对随访：更大倾斜关联更快进展；"
            "不同人群与终点不可合并成统一权重",
        },
        {
            "url": "https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops",
            "relation": "二阶矩等效椭圆轴长；不同于照片边界最短/最长径或3D倾斜角",
        },
    ],
}
FORMULAS = {
    "rim": {
        "width_px": "r_disc(theta)-r_cup(theta)",
        "normalized_width": "width_px/(r_disc(theta)+r_disc(theta+180))",
        "sampling": "360 rays; 1 degree; 0.25 px; nearest pixel; reject reentry per ray",
        "distribution": "four image-axis 90-degree sectors; linear-interpolated quantiles",
        "orientation": "image axes only in this installed component; anatomy is not inferred",
    },
    "ellipse": {
        "axes_px": "4*sqrt(eigenvalues(covariance(foreground_pixel_centres)))",
        "ovality": "minor_axis/major_axis",
        "elongation": "major_axis/minor_axis",
        "angle": "0.5*atan2(2*cov_xy,cov_xx-cov_yy), modulo 180 degrees",
        "isotropy": "relative eigenvalue gap <=1e-6 gives null angle; numerical rule",
    },
}


def argv_template(task):
    argv = [
        str(PYTHON),
        "-m",
        "apps.disc_geometry",
        task,
        "--disc-mask",
        "{input.disc_mask}",
        "--output",
        "{output.observation}",
    ]
    if task == "rim":
        argv += ["--cup-mask", "{input.cup_mask}"]
    return argv


def prepare(evidence):
    """Freeze exact local implementation only after a passing bounded test suite."""
    if evidence.stat().st_size > 2_097_152:
        raise ValueError("test_evidence_too_large")
    xml = ElementTree.parse(evidence).getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    if (
        not suites
        or sum(int(s.get("tests", "0")) for s in suites) < 20
        or any(int(s.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise ValueError("passing_geometry_test_evidence_required")
    paths = [
        ROOT / "apps/disc_geometry.py",
        Path(__file__),
        ROOT / "packages/tool_execution/qc.py",
        PYTHON.resolve(),
    ]
    spec = {
        "schema_version": "disc-geometry-installation/1.0.0",
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "tools": TOOLS,
        "formulas": FORMULAS,
        "sources": SOURCES,
        "resources": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths
        ],
        "test_evidence": {
            "path": str(evidence.resolve()),
            "sha256": digest(evidence),
            "tests": sum(int(s.get("tests", "0")) for s in suites),
        },
    }
    spec = json.loads(json.dumps(spec))
    if LOCK.exists():
        if json.loads(LOCK.read_bytes()) != spec:
            raise ValueError("geometry_lock_content_conflict")
    else:
        with LOCK.open("x", encoding="utf-8") as target:
            json.dump(spec, target, ensure_ascii=False, indent=2)
            target.write("\n")
    return spec


def load_verified():
    spec = json.loads(LOCK.read_bytes())
    if (
        spec["version"] != VERSION
        or spec["implementation"] != IMPLEMENTATION
        or spec["tools"] != {k: list(v) for k, v in TOOLS.items()}
        or spec["formulas"] != FORMULAS
        or spec["sources"] != SOURCES
    ):
        raise ValueError("geometry_definition_drift")
    for item in spec["resources"]:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("geometry_resource_drift")
    if digest(spec["test_evidence"]["path"]) != spec["test_evidence"]["sha256"]:
        raise ValueError("geometry_test_evidence_drift")
    return spec


def rows(connection):
    return connection.execute(
        "SELECT t.code,v.version,v.enabled,i.implementation_key,i.enabled,i.validation_status,"
        "i.provenance->>'installation_checksum' FROM medgraph.algorithm_tool t "
        "JOIN medgraph.tool_version v ON v.tool_id=t.id "
        "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id "
        "WHERE t.code=ANY(%s) ORDER BY t.code",
        ([v[0] for v in TOOLS.values()],),
    ).fetchall()


def register(connection, spec):
    """Operator registration; caller owns one transaction for both tools."""
    if not connection.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("native_owner_required")
    connection.execute("SELECT pg_advisory_xact_lock(7150926)")
    seal = checksum(spec)
    existing = rows(connection)
    count = connection.execute(
        "SELECT count(*) FROM medgraph.algorithm_tool WHERE code=ANY(%s)",
        ([v[0] for v in TOOLS.values()],),
    ).fetchone()[0]
    if count:
        if (
            count != 2
            or len(existing) != 2
            or any(
                r[1:] != (VERSION, True, IMPLEMENTATION, True, "smoke_tested", seal)
                for r in existing
            )
        ):
            raise ValueError("geometry_existing_state_or_content_conflict")
        return {"status": "already_registered", "tools": existing}
    types = {
        code: {"format": "image/png", "mode": "L", "values": [0, 255], "modality": "SLO"}
        for code in ("slo:OpticDiscMask", "slo:OpticCupMask")
    }
    for task, (_, _, code) in TOOLS.items():
        types[code] = {
            "format": "application/json",
            "schema_version": disc_geometry.VERSION,
            "task": task,
            "use": "research_measurement",
        }
    ids = {}
    for code, schema in types.items():
        old = connection.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if old:
            if old[1] != schema:
                raise ValueError("geometry_artifact_type_conflict")
            ids[code] = old[0]
        else:
            ids[code] = connection.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, code, Jsonb(schema)),
            ).fetchone()[0]
    for task, (code, name, output_type) in TOOLS.items():
        metadata = {
            "intended_use": "research_only",
            "decisive_evidence": False,
            "input_modality": "SLO",
            "installation_checksum": seal,
            "sources": SOURCES[task],
            "disease_weight": None,
            "weight_status": "not_assigned; association population/endpoint must be retained",
            "input_requirements": "aligned original-grid binary masks; square pixels; "
            "one component; no holes or edge clipping; cup inside disc",
            "same_case_scope": "must be verified by caller/Runtime; "
            "shape alone is not case identity",
            "parameter_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
        tid = connection.execute(
            "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
            "VALUES(%s,%s,'calculator',%s,%s) RETURNING id",
            (
                code,
                name,
                "输入已有分割掩膜，输出几何测量和精确来源；无需模型推理。",
                Jsonb(metadata),
            ),
        ).fetchone()[0]
        runtime = {
            "manager": "conda",
            "prefix": str(PYTHON.parent.parent),
            "python_executable": str(PYTHON),
            "device": "cpu",
            "python": "3.12",
        }
        vid = connection.execute(
            "INSERT INTO medgraph.tool_version(tool_id,version,runtime,metadata) "
            "VALUES(%s,%s,%s,%s) RETURNING id",
            (tid, VERSION, Jsonb(runtime), Jsonb(metadata)),
        ).fetchone()[0]
        ports = [
            ("input", "disc_mask", "slo:OpticDiscMask"),
            ("output", "observation", output_type),
        ]
        if task == "rim":
            ports.append(("input", "cup_mask", "slo:OpticCupMask"))
        for direction, port, artifact in ports:
            connection.execute(
                "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,"
                "artifact_type_id,required,cardinality_min,cardinality_max,constraints) "
                "VALUES(%s,%s,%s,%s,true,1,1,%s)",
                (vid, direction, port, ids[artifact], Jsonb({"research_only": True})),
            )
        iid = connection.execute(
            "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
            "implementation_kind,entrypoint_uri,working_directory,runtime_environment,invocation,"
            "formula_spec,hardware_requirements,provenance,validation_status,validation_report) "
            "VALUES(%s,%s,'formula_python',%s,%s,%s,%s,%s,%s,%s,'smoke_tested',%s) RETURNING id",
            (
                vid,
                IMPLEMENTATION,
                (ROOT / "apps/disc_geometry.py").as_uri(),
                str(ROOT),
                Jsonb(runtime),
                Jsonb({"argv": argv_template(task), "shell": False, "timeout_seconds": 60}),
                Jsonb(FORMULAS[task]),
                Jsonb({"accelerator": "cpu", "required": False}),
                Jsonb({"installation_checksum": seal, "sources": SOURCES[task]}),
                Jsonb(
                    {
                        "status": "algorithm_tests_passed",
                        "evidence": spec["test_evidence"],
                        "clinical_validated": False,
                        "runtime_06_bundle_installed": False,
                    }
                ),
            ),
        ).fetchone()[0]
        for index, item in enumerate(spec["resources"]):
            connection.execute(
                "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
                "resource_role,uri,sha256,byte_size,required) VALUES(%s,%s,%s,%s,%s,%s,true)",
                (
                    iid,
                    f"resource_{index:03}",
                    "fixed_local_resource",
                    Path(item["path"]).as_uri(),
                    item["sha256"],
                    item["bytes"],
                ),
            )
    return {"status": "registered", "tools": rows(connection)}


def resolve(connection, task, disc, cup, output):
    spec = load_verified()
    state = [r for r in rows(connection) if r[0] == TOOLS[task][0]]
    if len(state) != 1 or state[0][1:] != (
        VERSION,
        True,
        IMPLEMENTATION,
        True,
        "smoke_tested",
        checksum(spec),
    ):
        raise ValueError("geometry_component_unavailable_or_changed")
    manifest = connection.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
        (TOOLS[task][0], VERSION, IMPLEMENTATION),
    ).fetchone()[0]
    if manifest["invocation"] != {
        "argv": argv_template(task),
        "shell": False,
        "timeout_seconds": 60,
    } or manifest["working_directory"] != str(ROOT):
        raise ValueError("geometry_manifest_drift")
    if (task == "rim") != (cup is not None):
        raise ValueError("geometry_input_ports_mismatch")
    paths = {
        "{input.disc_mask}": disc.resolve(strict=True),
        "{output.observation}": output.resolve(),
    }
    if cup is not None:
        paths["{input.cup_mask}"] = cup.resolve(strict=True)
    if any(not p.is_relative_to("/opt/data/test") for p in paths.values()):
        raise ValueError("geometry_path_outside_project_storage")
    if output.exists():
        raise ValueError("geometry_output_already_exists")
    return {
        "tool_code": TOOLS[task][0],
        "version": VERSION,
        "implementation": IMPLEMENTATION,
        "argv": [str(paths.get(token, token)) for token in argv_template(task)],
        "research_only": True,
        "output": str(paths["{output.observation}"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "register", "list", "show", "run"))
    parser.add_argument("--task", choices=tuple(TOOLS))
    parser.add_argument("--test-evidence", type=Path)
    parser.add_argument("--disc-mask", type=Path)
    parser.add_argument("--cup-mask", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.test_evidence:
            parser.error("--test-evidence required")
        result = prepare(args.test_evidence)
    else:
        with native_connection() as connection:
            if args.action == "register":
                result = register(connection, load_verified())
            elif args.action == "list":
                result = {"tools": rows(connection)}
            else:
                if not args.task:
                    parser.error("--task required")
                if args.action == "show":
                    result = connection.execute(
                        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                        (TOOLS[args.task][0], VERSION, IMPLEMENTATION),
                    ).fetchone()[0]
                else:
                    if not args.disc_mask or not args.output:
                        parser.error("--disc-mask and --output required")
                    result = resolve(
                        connection, args.task, args.disc_mask, args.cup_mask, args.output
                    )
        if args.action == "run" and args.execute:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                result["argv"],
                cwd=ROOT,
                timeout=60,
                check=False,
                env={
                    "PATH": str(PYTHON.parent) + ":/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "TMPDIR": str(args.output.parent.resolve()),
                },
            )
            result["exit_code"] = completed.returncode
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(completed.returncode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

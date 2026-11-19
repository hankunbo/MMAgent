"""One explicit partial-geometry calculator in the original medgraph Tool catalog.

Registration qualifies software measurement only. Original Runtime must bind
patient inputs and the domain owner must accept its limited interpretation.
"""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

from psycopg.types.json import Jsonb

from apps.slo_tool_registry import ROOT, checksum, digest, native_connection

CODE = "calculate_slo_partial_rim_geometry"
VERSION = "1.0.1"
IMPLEMENTATION = "local_partial_disc_rim_cpu_v1"
SCHEMA = "partial-disc-rim-geometry/1.0.0"
OUTPUT_TYPE = "slo:PartialRimDirectionalGeometry"
PYTHON = Path("/opt/data/test/envs/MMAgent1/bin/python")
RUNNER = Path(__file__).with_name("disc_geometry_partial.py")
LOCK = ROOT / "configs/tools/partial-rim-geometry-v1.0.1.json"
FORMULA = {
    "method": "disc_centroid_single_interval_partial_rays/1.0.0",
    "width": "last_disc_exit(theta)-last_single_cup_exit(theta)",
    "denominator": "disc_exit(theta)+disc_exit(theta+180)",
    "sampling": "360 rays; 1 degree; 0.25px; nearest-pixel; no mask edit",
    "partial": "only one contiguous disc interval on both axis directions and one cup interval",
    "missing": "no cup intersection or multiple intervals gives null, never width zero",
    "global_minimum": "null unless all 360 rays are available",
    "coverage": "geometric fraction, not automatic quality weight or probability",
}


def read(path):
    return json.loads(Path(path).read_bytes())


def argv():
    return [
        str(PYTHON),
        "-c",
        "import runpy; runpy.run_path(" + repr(str(RUNNER.resolve())) + ", run_name='__main__')",
        "--disc-mask",
        "{input.disc_mask}",
        "--cup-mask",
        "{input.cup_mask}",
        "--output",
        "{output.observation}",
    ]


def validation(evidence):
    """Bind the executed CPU tests to exact algorithm and test bytes."""
    evidence = Path(evidence).resolve(strict=True)
    sealed = read(evidence / "candidate-freeze.json")
    for name in (
        "apps/disc_geometry_partial.py",
        "tests/test_partial_rim.py",
        "cpu.xml",
        "cpu.json",
        "cpu.log",
    ):
        path = evidence / name
        if "sha256:" + digest(path) != sealed[name]:
            raise ValueError("partial_rim_validation_bytes_changed")
    if digest(RUNNER) != digest(evidence / "apps/disc_geometry_partial.py"):
        raise ValueError("partial_rim_tested_implementation_changed")
    receipt = read(evidence / "cpu.json")
    if receipt["direct_exit"] or receipt["log_checksum"] != "sha256:" + digest(
        evidence / "cpu.log"
    ):
        raise ValueError("partial_rim_validation_failed")
    xml = ElementTree.parse(evidence / "cpu.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    required = {
        "test_complete_masks_retain_original_numerical_rays",
        "test_offset_cup_has_partial_measured_width_not_zero_for_unseen_rays",
        "test_disc_hole_limits_both_axis_directions_without_filling",
        "test_cup_hole_is_not_first_boundary_or_zero_rim",
        "test_missing_center_geometry_is_recorded_without_invented_center_or_minimum",
        "test_unrelated_original_validations_remain",
        "test_saved_file_bindings_and_cli_no_overwrite",
        "test_changed_image_shape_rejects",
    }
    names = {
        case.get("name").split("[")[0] for suite in suites for case in suite.findall("testcase")
    }
    if not required <= names or any(
        int(suite.get(k, "0")) for suite in suites for k in ("failures", "errors", "skipped")
    ):
        raise ValueError("partial_rim_required_validation_missing")
    return {
        "path": str(evidence / "candidate-freeze.json"),
        "sha256": digest(evidence / "candidate-freeze.json"),
        "tests": sum(int(s.get("tests", "0")) for s in suites),
        "clinical_validated": False,
    }


def prepare(evidence):
    proof = validation(evidence)
    paths = [
        RUNNER,
        Path(__file__),
        ROOT / "apps/disc_geometry.py",
        ROOT / "packages/tool_execution/qc.py",
        PYTHON.resolve(),
    ]
    spec = dict(
        schema="partial-rim-installation/1.0.0",
        tool=CODE,
        version=VERSION,
        implementation=IMPLEMENTATION,
        formula=FORMULA,
        argv=argv(),
        test_evidence=proof,
        resources=[
            dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p)) for p in paths
        ],
        clinical_validated=False,
        runtime_case_binding_required=True,
    )
    if LOCK.exists():
        if read(LOCK) != spec:
            raise ValueError("partial_rim_installation_conflict")
    else:
        with LOCK.open("x") as stream:
            json.dump(spec, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return spec


def load_verified():
    spec = read(LOCK)
    if (spec["tool"], spec["version"], spec["implementation"], spec["formula"], spec["argv"]) != (
        CODE,
        VERSION,
        IMPLEMENTATION,
        FORMULA,
        argv(),
    ):
        raise ValueError("partial_rim_definition_drift")
    for item in [*spec["resources"], spec["test_evidence"]]:
        p = Path(item["path"])
        if digest(p) != item["sha256"] or ("bytes" in item and p.stat().st_size != item["bytes"]):
            raise ValueError("partial_rim_resource_drift")
    if (
        spec["test_evidence"] != validation(Path(spec["test_evidence"]["path"]).parent)
        or spec.get("clinical_validated") is not False
        or spec.get("runtime_case_binding_required") is not True
        or {Path(item["path"]).resolve() for item in spec["resources"]}
        != {
            RUNNER.resolve(),
            Path(__file__).resolve(),
            ROOT / "apps/disc_geometry.py",
            ROOT / "packages/tool_execution/qc.py",
            PYTHON.resolve(),
        }
    ):
        raise ValueError("partial_rim_installation_scope_drift")
    return spec


def rows(connection):
    return connection.execute(
        "SELECT v.version,v.enabled,i.implementation_key,i.enabled,i.validation_status,"
        "i.provenance->>'installation_checksum' FROM medgraph.algorithm_tool t "
        "JOIN medgraph.tool_version v ON v.tool_id=t.id "
        "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id "
        "WHERE t.code=%s AND v.version=%s",
        (CODE, VERSION),
    ).fetchall()


def register(connection, spec):
    if spec != load_verified():
        raise ValueError("partial_rim_unverified_installation")
    if not connection.execute(
        "SELECT pg_get_userbyid(nspowner)=current_user FROM pg_namespace WHERE nspname='medgraph'"
    ).fetchone()[0]:
        raise ValueError("native_owner_required")
    connection.execute("SELECT pg_advisory_xact_lock(7150926)")
    seal = checksum(spec)
    prior = rows(connection)
    count = connection.execute(
        "SELECT count(*) FROM medgraph.algorithm_tool WHERE code=%s", (CODE,)
    ).fetchone()[0]
    if prior:
        if count != 1 or prior != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", seal)]:
            raise ValueError("partial_rim_existing_state_or_content_conflict")
        return dict(status="already_registered", tools=prior)
    if count not in (0, 1):
        raise ValueError("partial_rim_existing_tool_conflict")
    types = {
        k: dict(format="image/png", mode="L", values=[0, 255], modality="SLO")
        for k in ("slo:OpticDiscMask", "slo:OpticCupMask")
    }
    types[OUTPUT_TYPE] = dict(
        format="application/json",
        schema_version=SCHEMA,
        task="partial_rim",
        use="research_measurement",
    )
    ids = {}
    for code, schema in types.items():
        old = connection.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s", (code,)
        ).fetchone()
        if old:
            if old[1] != schema:
                raise ValueError("partial_rim_artifact_type_conflict")
            ids[code] = old[0]
        else:
            ids[code] = connection.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, code, Jsonb(schema)),
            ).fetchone()[0]
    metadata = dict(
        intended_use="research_only",
        decisive_evidence=False,
        installation_checksum=seal,
        input_modality="SLO",
        disease_weight=None,
        input_requirements="aligned unchanged binary single-component masks; "
        "cup contained in disc; "
        "no clipping; holes/offset cup restrict available rays",
        same_case_scope="original Runtime must verify case and current input binding",
        limitations="local minimum not global; segmentation contact not proof of missing tissue",
        parameter_schema=dict(type="object", properties={}, additionalProperties=False),
    )
    runtime = dict(
        manager="conda",
        prefix=str(PYTHON.parent.parent),
        python_executable=str(PYTHON),
        device="cpu",
        python="3.12",
    )
    tid = (
        connection.execute(
            "SELECT id FROM medgraph.algorithm_tool WHERE code=%s", (CODE,)
        ).fetchone()[0]
        if count
        else connection.execute(
            "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
            "VALUES(%s,%s,'calculator',%s,%s) RETURNING id",
            (CODE, "部分可测方向盘沿几何", "保留未测方向；不修补掩膜或推断疾病。", Jsonb(metadata)),
        ).fetchone()[0]
    )
    vid = connection.execute(
        "INSERT INTO medgraph.tool_version(tool_id,version,runtime,metadata) "
        "VALUES(%s,%s,%s,%s) RETURNING id",
        (tid, VERSION, Jsonb(runtime), Jsonb(metadata)),
    ).fetchone()[0]
    for direction, port, artifact in (
        ("input", "disc_mask", "slo:OpticDiscMask"),
        ("input", "cup_mask", "slo:OpticCupMask"),
        ("output", "observation", OUTPUT_TYPE),
    ):
        connection.execute(
            "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,artifact_type_id,"
            "required,cardinality_min,cardinality_max,constraints) VALUES(%s,%s,%s,%s,true,1,1,%s)",
            (vid, direction, port, ids[artifact], Jsonb(dict(research_only=True))),
        )
    iid = connection.execute(
        "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
        "implementation_kind,"
        "entrypoint_uri,working_directory,runtime_environment,invocation,formula_spec,hardware_requirements,"
        "provenance,validation_status,validation_report) "
        "VALUES(%s,%s,'formula_python',%s,%s,%s,%s,%s,%s,%s,'smoke_tested',%s) RETURNING id",
        (
            vid,
            IMPLEMENTATION,
            RUNNER.resolve().as_uri(),
            str(ROOT),
            Jsonb(runtime),
            Jsonb(dict(argv=argv(), shell=False, timeout_seconds=60)),
            Jsonb(FORMULA),
            Jsonb(dict(accelerator="cpu", required=False)),
            Jsonb(dict(installation_checksum=seal)),
            Jsonb(
                dict(
                    status="algorithm_tests_passed",
                    evidence=spec["test_evidence"],
                    clinical_validated=False,
                    runtime_06_bundle_installed=False,
                )
            ),
        ),
    ).fetchone()[0]
    for i, item in enumerate(spec["resources"]):
        connection.execute(
            "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
            "resource_role,"
            "uri,sha256,byte_size,required) VALUES(%s,%s,%s,%s,%s,%s,true)",
            (
                iid,
                f"resource_{i:03}",
                "fixed_local_resource",
                Path(item["path"]).as_uri(),
                item["sha256"],
                item["bytes"],
            ),
        )
    return dict(status="registered", tools=rows(connection))


def resolve(connection, disc, cup, output):
    spec = load_verified()
    if rows(connection) != [(VERSION, True, IMPLEMENTATION, True, "smoke_tested", checksum(spec))]:
        raise ValueError("partial_rim_unavailable_or_changed")
    manifest = connection.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", (CODE, VERSION, IMPLEMENTATION)
    ).fetchone()[0]
    if (
        not manifest
        or manifest["invocation"] != dict(argv=argv(), shell=False, timeout_seconds=60)
        or manifest["working_directory"] != str(ROOT)
        or manifest["formula_spec"] != FORMULA
    ):
        raise ValueError("partial_rim_manifest_drift")
    paths = {
        "{input.disc_mask}": Path(disc).resolve(strict=True),
        "{input.cup_mask}": Path(cup).resolve(strict=True),
        "{output.observation}": Path(output).resolve(),
    }
    if any(not p.is_relative_to("/opt/data/test") for p in paths.values()) or Path(output).exists():
        raise ValueError("partial_rim_output_or_path_scope")
    return dict(
        tool_code=CODE,
        version=VERSION,
        implementation=IMPLEMENTATION,
        argv=[str(paths.get(token, token)) for token in argv()],
        manifest_checksum=checksum(manifest),
        input_checksums={k: digest(v) for k, v in paths.items() if k.startswith("{input.")},
        research_only=True,
        output=str(paths["{output.observation}"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "register", "show", "resolve"))
    parser.add_argument("--test-evidence", type=Path)
    parser.add_argument("--disc-mask", type=Path)
    parser.add_argument("--cup-mask", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.test_evidence:
            parser.error("--test-evidence required")
        result = prepare(args.test_evidence)
    else:
        with native_connection() as connection:
            if args.action == "register":
                result = register(connection, load_verified())
            elif args.action == "show":
                result = dict(tools=rows(connection))
            else:
                if not args.disc_mask or not args.cup_mask or not args.output:
                    parser.error("--disc-mask --cup-mask --output required")
                result = resolve(connection, args.disc_mask, args.cup_mask, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

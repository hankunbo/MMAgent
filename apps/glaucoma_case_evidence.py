"""Execute registered observations on the existing first case and bind real artifacts."""

import argparse
import base64
import fcntl
import json
import os
import signal
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from apps import slo_tool_registry as registry
from apps.glaucoma_evidence_bindings import ROOT, dump, existing_case_binding, write_new
from apps.research_single_case import credential_boundary
from db.runtime.repository.case_evidence import CaseEvidenceRepository
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.object_store import LocalObjectStore, checksum

DIRECTORY = ROOT / ".local/glaucoma-case-evidence-v1"
TASKS = ("vessels", "avod", "fovea", "density", "width", "tortuosity", "fractal")


def png_shape(data):
    if len(data) > 4_194_304 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise ValueError("case_image_format")
    w, h = struct.unpack(">II", data[16:24])
    if data[24:29] != bytes((8, 0, 0, 0, 0)) or not 0 < w * h <= 4_000_000:
        raise ValueError("case_image_shape")
    return h, w


def read_json(path):
    if path.stat().st_size > 4_194_304:
        raise ValueError("observation_size")
    return json.loads(path.read_bytes())


def execute(task, source, directory):
    from packages.evaluation_preparation.generic_execution_lock import current, wait_process
    from packages.evaluation_preparation.terminal_cohort import stop_child

    job = directory / task
    receipt = job / "execution.json"
    if receipt.exists():
        result = read_json(receipt)
        if result["exit_code"] != 0:
            raise ValueError("previous_execution_failed:" + task)
        return result
    if (job / "started.json").exists():
        raise ValueError("execution_unconfirmed_no_automatic_retry:" + task)
    with registry.native_connection() as connection:
        call = registry.resolve_component(connection, task, source, job)
        manifest = connection.execute(
            "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
            (registry.TOOLS[task][0], registry.VERSION, registry.IMPLEMENTATION),
        ).fetchone()[0]
    job.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    write_new(
        job / "started.json",
        {
            "started_at": started,
            "input_checksum": checksum(source.read_bytes()),
            "manifest_checksum": "sha256:" + registry.checksum(manifest),
        },
    )
    environment = {
        "PATH": str(registry.PYTHON.parent) + ":/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "MPLBACKEND": "Agg",
        "TMPDIR": str(job),
        "XDG_CACHE_HOME": str(job / "cache"),
        "MPLCONFIGDIR": str(job / "cache/matplotlib"),
        "OMP_NUM_THREADS": "8",
    }
    current()
    with (job / "process.log").open("xb") as log:
        process = subprocess.Popen(
            call["argv"],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = wait_process(process, timeout=180, stop=stop_child)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
            code = 124
    result = {
        "task": task,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "exit_code": code,
        "call": call,
        "manifest": manifest,
        "manifest_checksum": "sha256:" + registry.checksum(manifest),
        "input_checksum": checksum(source.read_bytes()),
    }
    write_new(receipt, result)
    if code:
        raise ValueError("component_failed:" + task)
    return result


def validate_output(task, record, result, source, shape):
    if (
        record["task"] != task
        or record["schema_version"] != "slo-observation/1.0.0"
        or result["exit_code"] != 0
    ):
        raise ValueError("observation_task_mismatch")
    if "sha256:" + record["input"]["sha256"] != checksum(source.read_bytes()) or result[
        "input_checksum"
    ] != checksum(source.read_bytes()):
        raise ValueError("observation_input_mismatch")
    if (
        tuple(record["input"]["shape_hw"]) != shape
        or record["clinical_effect"] is not None
        or record["research_only"] is not True
    ):
        raise ValueError("observation_scope_mismatch")
    if record["installation_sha256"] != registry.digest(registry.LOCK):
        raise ValueError("observation_installation_mismatch")
    if set(record["outputs"]) != set(result["call"]["outputs"]) - {"observation"}:
        raise ValueError("observation_ports_mismatch")
    for port, ref in record["outputs"].items():
        path = Path(result["call"]["outputs"][port])
        if ref["uri"] != path.as_uri() or ref["sha256"] != registry.digest(path):
            raise ValueError("observation_output_mismatch")
        if port.endswith("_mask") and png_shape(path.read_bytes()) != shape:
            raise ValueError("observation_dimensions_mismatch")
    if task in registry.FORMULAS:
        import math

        m = record["measurement"]
        if (m["value"] is None) == (m["unavailable_reason"] is None):
            raise ValueError("measurement_missing_reason")
        if m["value"] is not None and (
            type(m["value"]) not in (int, float) or not math.isfinite(m["value"])
        ):
            raise ValueError("measurement_nonfinite")


def run(directory=DIRECTORY):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(directory)


def _run(directory):
    historical = existing_case_binding()
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    case_code = historical["outputs"][0]["source"]["scope"]["case_code"]
    case = repo.get_case_bundle(case_code)
    expected = historical["outputs"][0]["output"]["input_checksum"]
    images = [
        a
        for a in case.artifacts
        if a.artifact_type_code == "ARTIFACT:SLO_IMAGE" and a.checksum == expected
    ]
    if len(images) != 1:
        raise ValueError("exact_case_image_required")
    image = images[0]
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root

    check_object_root(OBJECT_ROOTS["visible"], "visible")
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    raw = store.read_bounded(image.uri, 4_194_304)
    shape = png_shape(raw)
    if checksum(raw) != expected or tuple(image.dimensions) != shape:
        raise ValueError("case_image_binding_mismatch")
    input_path = directory / "input.png"
    if input_path.exists():
        if input_path.read_bytes() != raw:
            raise ValueError("existing_input_mismatch")
    else:
        with input_path.open("xb") as f:
            f.write(raw)
    write_new(
        directory / "input.json",
        {
            "case_code": case_code,
            "sample": historical["sample"],
            "artifact": dump(image),
            "shape_hw": list(shape),
        },
    )
    results = {}
    for task in TASKS:
        source = directory / "vessels/vessel_mask.png" if task in registry.FORMULAS else input_path
        result = execute(task, source, directory)
        record = read_json(Path(result["call"]["outputs"]["observation"]))
        validate_output(task, record, result, source, shape)
        results[task] = {"execution": result, "observation": record}
        print(
            json.dumps({"task": task, "exit_code": result["exit_code"], "state": "observed"}),
            flush=True,
        )
    # Durable materialization also makes SQL-ack loss a readback-only retry.
    pending = directory / "artifacts.json"
    if not pending.exists():
        artifacts = []
        ports = {}
        for task, result in results.items():
            src = ports["vessels:vessel_mask"] if task in registry.FORMULAS else image
            for port, filename in result["execution"]["call"]["outputs"].items():
                body = Path(filename).read_bytes()
                uri = store.promote(store.stage(body), checksum(body))
                kind = "SLO_OBSERVATION" if port == "observation" else "SLO_" + port.upper()
                artifact = Artifact.model_validate(
                    {
                        "schema_version": "1.0",
                        "contract_type": "Artifact",
                        "asset_code": identity("ASSET", case_code + task + port + checksum(body)),
                        "artifact_type_code": "ARTIFACT:" + kind,
                        "uri": uri,
                        "checksum": checksum(body),
                        "media_type": "application/json"
                        if port == "observation"
                        else ("image/png" if port.endswith("mask") else "application/octet-stream"),
                        "qc_status": "PASS",
                        "dimensions": list(shape) if port != "observation" else [],
                        "measurements": [],
                        "derived_from": [src.asset_code],
                        "research_only": True,
                        "scope": {"scope_type": "CASE", "case_code": case_code},
                        "laterality": image.laterality,
                        "coordinate_space": "PIXEL_2D",
                        "unit": None,
                        "extensions": {"component_task": task, "output_port": port},
                        "provenance": {
                            "actor": "TOOL_AGENT",
                            "created_at": result["execution"]["finished_at"],
                            "generator_version": "case-observations-1.0.0",
                            "source_codes": [src.asset_code],
                            "input_checksums": [src.checksum],
                            "tool_versions": [
                                {
                                    "code": "TOOL:SLO:" + task.upper(),
                                    "version": registry.VERSION,
                                    "checksum": result["execution"]["manifest_checksum"],
                                }
                            ],
                        },
                    }
                )
                artifacts.append(artifact)
                ports[task + ":" + port] = artifact
        write_new(pending, {"artifacts": [dump(a) for a in artifacts]})
    artifacts = tuple(Artifact.model_validate(a) for a in read_json(pending)["artifacts"])
    for a in artifacts:
        if checksum(store.read_bounded(a.uri, 4_194_304)) != a.checksum:
            raise ValueError("staged_artifact_drift")
    repo.register_observation_bundle(
        artifacts, identity("TASK", "case-observations-v1:" + case_code)
    )
    actual = {a.asset_code: a for a in repo.get_case_bundle(case_code).artifacts}
    if any(actual[a.asset_code] != a for a in artifacts):
        raise ValueError("case_artifact_readback_mismatch")
    report = {
        "schema": "glaucoma-case-evidence/1.0.0",
        "case_code": case_code,
        "sample": historical["sample"],
        "input": dump(image),
        "results": results,
        "artifacts": [dump(a) for a in artifacts],
        "fairclip": historical,
        "object_root": str(OBJECT_ROOTS["visible"]),
        "execution_scope": "registered_components_case_artifacts",
        "accepted_validations": [],
    }
    write_new(directory / "case-evidence.json", report)
    return report


def load_verified(directory=DIRECTORY):
    report = read_json(directory / "case-evidence.json")
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    artifacts = {a.asset_code: a for a in repo.get_case_bundle(report["case_code"]).artifacts}
    from apps.patient_service_assembly import OBJECT_ROOTS

    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    for raw in [report["input"], *report["artifacts"]]:
        a = Artifact.model_validate(raw)
        if (
            artifacts.get(a.asset_code) != a
            or checksum(store.read_bounded(a.uri, 4_194_304)) != a.checksum
        ):
            raise ValueError("case_evidence_readback_mismatch")
    for task, item in report["results"].items():
        matching = [
            a
            for a in report["artifacts"]
            if a["extensions"] == {"component_task": task, "output_port": "observation"}
        ]
        if len(matching) != 1:
            raise ValueError("case_observation_identity")
        a = Artifact.model_validate(matching[0])
        actual = json.loads(store.read_bounded(a.uri, 4_194_304))
        if actual != item["observation"] or item["execution"]["exit_code"] != 0:
            raise ValueError("case_observation_value_mismatch")
        if (
            a.provenance.tool_versions[0].checksum != item["execution"]["manifest_checksum"]
            or item["execution"]["manifest_checksum"]
            != "sha256:" + registry.checksum(item["execution"]["manifest"])
            or item["execution"]["input_checksum"] not in a.provenance.input_checksums
        ):
            raise ValueError("case_observation_provenance_mismatch")
    historical = existing_case_binding()
    if (
        report["fairclip"] != historical
        or historical["outputs"][0]["source"]["scope"]["case_code"] != report["case_code"]
    ):
        raise ValueError("historical_case_mismatch")
    report["images"] = {}
    for raw in [report["input"], *report["artifacts"]]:
        a = Artifact.model_validate(raw)
        if a.media_type == "image/png":
            name = a.extensions.get("output_port", "input")
            report["images"][name] = (
                "data:image/png;base64,"
                + base64.b64encode(store.read_bounded(a.uri, 4_194_304)).decode()
            )
    from apps.glaucoma_case_supplement import attach_verified

    attach_verified(report, repo, store)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    args = parser.parse_args()
    report = run() if args.action == "run" else load_verified()
    print(
        json.dumps(
            {
                "case_code": report["case_code"],
                "sample": report["sample"],
                "tasks": list(report["results"]),
                "artifacts": len(report["artifacts"]),
            }
        )
    )


if __name__ == "__main__":
    main()

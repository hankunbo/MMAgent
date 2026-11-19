"""One bounded partial-rim CPU successor using the original case Artifact owner.

This stage persists measurement facts, not a clinical or Review acceptance.
New attempts cannot be created by changing an output path; the case/method key
is fixed. UNKNOWN process claims and failed attempts are never replayed.
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps import partial_rim_case
from apps import partial_rim_tool_registry as tool
from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import CasePipelineRepository, identity
from packages.contracts import Artifact
from packages.evaluation_preparation.case_execution import process
from packages.grounding.models import canonical, digest
from packages.object_store import checksum

bound_inputs = partial_rim_case.bound_inputs

VERSION = "partial-rim-case-measurement/1.0.0"
FOLDER = "partial-rim-case-measurement-v1"


def read(path):
    return json.loads(Path(path).read_bytes())


def current_manifest():
    spec = tool.load_verified()
    with tool.native_connection() as connection:
        if tool.rows(connection) != [
            (tool.VERSION, True, tool.IMPLEMENTATION, True, "smoke_tested", tool.checksum(spec))
        ]:
            raise ValueError("partial_rim_current_tool_unavailable")
        manifest = connection.execute(
            "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
            (tool.CODE, tool.VERSION, tool.IMPLEMENTATION),
        ).fetchone()[0]
    if not manifest or manifest["provenance"]["installation_checksum"] != tool.checksum(spec):
        raise ValueError("partial_rim_current_manifest_changed")
    return manifest


def bundle_key(proof):
    return identity("TASK", VERSION + ":" + proof["case_code"] + ":" + proof["key"])


def prepare(key):
    registered, proof = bound_inputs(key)
    root = registered[0]
    folder = root / FOLDER
    folder.mkdir(mode=0o700, exist_ok=True)
    # The original lease serializes prepare/execute/recover for this one method.
    from packages.evaluation_preparation.generic_execution_lock import target

    owner = CasePipelineRepository(target())
    with owner.guarded_session(proof["case_code"], bundle_key(proof)) as lease:
        lease()
        path = folder / "request.json"
        if path.exists():
            request = read(path)
            check_request(key, registered, proof, request)
            return request
        if (folder / "process/process-claim.json").exists() or (
            folder / "observation.json"
        ).exists():
            raise ValueError("partial_rim_unbound_previous_execution")
        if (root / "cancel-request.json").exists():
            raise ValueError("partial_rim_origin_cancelled")
        with tool.native_connection() as connection:
            plan = tool.resolve(
                connection,
                proof["masks"]["disc"]["path"],
                proof["masks"]["cup"]["path"],
                folder / "observation.json",
            )
        manifest = current_manifest()
        if plan["manifest_checksum"] != tool.checksum(manifest):
            raise ValueError("partial_rim_preparation_manifest_changed")
        lease()
        now = datetime.now(UTC)
        request = dict(
            schema_version=VERSION,
            key=key,
            original_input_proof=proof,
            plan=plan,
            manifest=manifest,
            bundle_key=bundle_key(proof),
            created_at=now.isoformat(),
            deadline=(now + timedelta(seconds=300)).isoformat(),
            max_tool_starts=1,
            max_api_requests=0,
            clinical_accepted=False,
        )
        write_once(path, request)
        return request


def check_request(key, registered, proof, request):
    root = registered[0]
    values = {
        "{input.disc_mask}": proof["masks"]["disc"]["path"],
        "{input.cup_mask}": proof["masks"]["cup"]["path"],
        "{output.observation}": str((root / FOLDER / "observation.json").resolve()),
    }
    expected = [values.get(token, token) for token in tool.argv()]
    if Path.cwd().resolve() != tool.ROOT.resolve():
        raise ValueError("partial_rim_original_working_directory_required")
    if (
        request.get("schema_version") != VERSION
        or request.get("key") != key
        or request.get("original_input_proof") != proof
        or request.get("bundle_key") != bundle_key(proof)
        or request.get("max_tool_starts") != 1
        or request.get("max_api_requests") != 0
        or request.get("clinical_accepted") is not False
        or request["plan"]["argv"] != expected
        or request["plan"]["output"] != expected[-1]
        or request["plan"]["manifest_checksum"] != tool.checksum(request["manifest"])
        or datetime.fromisoformat(request["deadline"])
        - datetime.fromisoformat(request["created_at"])
        != timedelta(seconds=300)
    ):
        raise ValueError("partial_rim_request_or_original_binding_changed")
    if request["manifest"] != current_manifest():
        raise ValueError("partial_rim_current_qualification_changed")


def verify_measurement(result, proof):
    if (
        result.get("schema_version") != tool.SCHEMA
        or result.get("disease_probability") is not None
        or result["inputs"]["disc_mask"]["sha256"]
        != proof["masks"]["disc"]["checksum"].removeprefix("sha256:")
        or result["inputs"]["cup_mask"]["sha256"]
        != proof["masks"]["cup"]["checksum"].removeprefix("sha256:")
    ):
        raise ValueError("partial_rim_measurement_input_or_version")


def save_measurement(registered, proof, request, result, receipt):
    """Original Runtime bundle and exact object readback, including lost commit reply."""
    _, _, runtime, objects = registered
    verify_measurement(result, proof)
    raw = canonical(
        dict(
            schema_version=VERSION,
            request=request,
            measurement=result,
            process=receipt,
            medical_direction=None,
            clinical_accepted=False,
        )
    )
    key = request["bundle_key"]
    saved = runtime.read_observation_bundle(key)
    if saved is not None and len(saved) != 1:
        raise ValueError("partial_rim_saved_inventory")
    uri = saved[0].uri if saved else objects.promote(objects.stage(raw), checksum(raw))
    if objects.read_bounded(uri, 4194304) != raw:
        raise ValueError("partial_rim_saved_object_changed")
    # The component derivation is in the immutable body; the original masks and
    # failure bundle remain explicit same-case parents, not copied patient facts.
    sources = [
        Artifact.model_validate(proof["masks"][n]["original_asset"]) for n in ("disc", "cup")
    ]
    sources.append(Artifact.model_validate(proof["original_tool_artifact"]))
    image = Artifact.model_validate(proof["image"])
    artifact = Artifact.model_validate(
        dict(
            schema_version="1.0",
            contract_type="Artifact",
            asset_code=identity("ASSET", key + checksum(raw)),
            artifact_type_code="ARTIFACT:RESEARCH_PARTIAL_RIM_MEASUREMENT",
            uri=uri,
            checksum=checksum(raw),
            media_type="application/json",
            qc_status="UNASSESSED",
            dimensions=[],
            measurements=[],
            derived_from=[a.asset_code for a in sources],
            research_only=True,
            scope=image.scope,
            laterality=image.laterality,
            coordinate_space="PIXEL_2D",
            unit=None,
            extensions=dict(
                generator=VERSION,
                tool_code=tool.CODE,
                method=tool.FORMULA["method"],
                interpretation_pending=True,
            ),
            provenance=dict(
                actor="TOOL_AGENT",
                created_at=receipt["finished_at"],
                generator_version="partial-rim-case-measurement-1.0.0",
                source_codes=[a.asset_code for a in sources],
                input_checksums=[a.checksum for a in sources],
            ),
        )
    )
    runtime.register_observation_bundle((artifact,), key)
    if runtime.read_observation_bundle(key) != (artifact,) or runtime.read_case_artifact(
        proof["case_code"], artifact.asset_code
    ) != artifact.model_copy(update={"derived_from": tuple(sorted(artifact.derived_from))}):
        raise ValueError("partial_rim_runtime_readback_changed")
    return dict(
        schema_version=VERSION,
        key=request["key"],
        bundle_key=key,
        request_checksum=digest(request),
        artifact=artifact.model_dump(mode="json"),
        measurement=result,
        process=receipt,
        clinical_accepted=False,
        review_accepted=False,
        new_api_requests=0,
    )


def execute(key, *, recover_only=False):
    from packages.evaluation_preparation.generic_execution_lock import target

    registered, proof = bound_inputs(key)
    root = registered[0]
    folder = root / FOLDER
    path = folder / "request.json"
    request = read(path)
    owner = CasePipelineRepository(target())
    with owner.guarded_session(proof["case_code"], bundle_key(proof)) as lease:
        check_request(key, registered, proof, request)
        lease()
        process_folder = folder / "process"
        process_receipt = process_folder / "process-result.json"
        if recover_only and not process_receipt.is_file():
            raise ValueError("partial_rim_no_confirmed_process_for_recovery")

        def guard():
            lease()
            if read(path) != request:
                raise ValueError("partial_rim_request_changed_after_claim")
            if (root / "cancel-request.json").exists() or (folder / "cancel-request.json").exists():
                raise ValueError("partial_rim_successor_cancelled")
            if datetime.now(UTC) >= datetime.fromisoformat(request["deadline"]):
                raise ValueError("partial_rim_successor_deadline")

        if not process_receipt.is_file():
            if recover_only:
                raise ValueError("partial_rim_recovery_never_dispatches")
            guard()
            receipt = process(
                request["plan"]["argv"], directory=process_folder, seconds=60, lease_guard=guard
            )
        else:
            # process() validates exact successful receipt before returning. A
            # confirmed local result can finish without renewing the old window.
            receipt = process(request["plan"]["argv"], directory=process_folder, seconds=60)
        again, current_proof = bound_inputs(key)
        check_request(key, again, current_proof, request)
        lease()
        result = read(folder / "observation.json")
        record = save_measurement(again, current_proof, request, result, receipt)
        write_once(folder / "bound-observation.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "recover"))
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    result = (
        prepare(args.key)
        if args.action == "prepare"
        else execute(args.key, recover_only=args.action == "recover")
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

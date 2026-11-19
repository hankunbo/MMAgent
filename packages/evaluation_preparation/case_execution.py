"""New-case input registration and bounded native-tool execution through existing owners.

This preparation port never invokes an LLM or treats an old case's observations as
new evidence. Unknown process outcomes are not automatically relaunched.
"""

import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from packages.dataset_adapters.prepared import capture
from packages.grounding.models import digest
from packages.object_store import checksum

from .new_cases import save
from .service import adapter_for

INPUT_ROOT = Path(
    "docs/acceptance/13c-papila-preparation-v1/prepare-20261002T035116Z-7fe07235/cases"
)
OUTPUT_ROOT = Path(".local/13-new-slo-v1")
SAMPLES = ("data_00002.npz", "data_00003.npz")


def selected_input(sample_id, *, input_root=INPUT_ROOT, fair_root=None):
    if sample_id not in SAMPLES:
        raise ValueError("new_case_outside_two_image_protocol")
    root = Path(input_root)
    protocol = json.loads((root / "protocol.json").read_bytes())
    if protocol["input_set_checksum"] != digest(protocol["input_files"]):
        raise ValueError("new_case_protocol_changed")
    entries = [e for e in protocol["entries"] if e["dataset"] == "FairVLMed"]
    if tuple(e["sample_id"] for e in entries) != SAMPLES:
        raise ValueError("new_case_selection_changed")
    relative = "visible/fairvlmed_" + Path(sample_id).stem + ".json"
    data = (root / relative).read_bytes()
    if checksum(data) != protocol["input_files"][relative]:
        raise ValueError("new_case_visible_changed")
    value = json.loads(data)
    image_path = root / "visible" / value["image_file"]
    if image_path.resolve().parent != (root / "visible").resolve():
        raise ValueError("new_case_image_path")
    raw = image_path.read_bytes()
    if checksum(raw) != value["image_checksum"] or checksum(raw) != next(
        e["image_checksum"] for e in entries if e["sample_id"] == sample_id
    ):
        raise ValueError("new_case_image_changed")
    adapter = adapter_for("fairvlmed", fair_root or "/opt/data/test/hf_cache/FairVLMed")
    sample = adapter.prepare(sample_id, split="training")
    snap = capture(sample)
    if snap.images[0][0] != raw:
        raise ValueError("new_case_original_image_changed")
    return sample, raw, value


def owners():
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from db.runtime.repository.dataset_input import DatasetInputRepository
    from packages.dataset_adapters.service import DatasetPreparationService
    from packages.research_runtime.deployment import (
        existing_store,
        load_configuration,
        read_installed_bundle,
    )

    config = load_configuration(Path("configs/research/fairclip_single_case.json"))
    installed = read_installed_bundle(config["installed_bundle"])
    runtime = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    visible = existing_store(installed["visible_root"], "visible")
    restricted = existing_store(installed["evaluator_root"], "evaluation")
    preparation = DatasetPreparationService(DatasetInputRepository(runtime), visible, restricted)
    return runtime, visible, preparation


def register(sample_id, *, output_root=OUTPUT_ROOT):
    from apps.vectorengine_single_case import write_once

    sample, raw, value = selected_input(sample_id)
    runtime, objects, preparation = owners()
    result = preparation.register(sample, key="13-new-slo-v1:" + sample_id)
    visible = preparation.read_visible(result.visible.case.case_code)
    if visible != result.visible or len(visible.case.artifacts) != 1:
        raise ValueError("new_case_registration_readback")
    image = visible.case.artifacts[0]
    if objects.read_bounded(image.uri, 4194304) != raw:
        raise ValueError("new_case_registered_image_mismatch")
    directory = Path(output_root) / Path(sample_id).stem
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(directory / "visible.json", visible)
    binding = dict(
        schema_version="new-case-binding/1.0.0",
        sample_id=sample_id,
        case_code=visible.case.case_code,
        image=image.model_dump(mode="json"),
        visible_checksum=visible.checksum,
        approved_preparation_checksum=digest(value),
        demographics=value["demographics"],
        demographic_encoding=value["demographics_encoding"],
        llm_requests=0,
    )
    write_once(directory / "binding.json", binding)
    path = directory / "input.png"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("new_case_output_input_conflict")
    else:
        with path.open("xb") as stream:
            path.chmod(0o600)
            stream.write(raw)
    return directory, binding, runtime, objects


def process(argv, *, directory, seconds=600, runner=subprocess.Popen, lease_guard=None):
    """Claim before launch; no replay of failed/unconfirmed work under a new key."""
    directory = Path(directory)
    receipt = directory / "process-result.json"
    if receipt.exists():
        value = json.loads(receipt.read_bytes())
        if value["argv"] != argv or value["exit_code"] != 0:
            raise ValueError("new_case_prior_failure_or_command_conflict")
        return value
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    save(
        directory / "process-claim.json", dict(argv=argv, started_at=datetime.now(UTC).isoformat())
    )
    environment = {
        "PATH": str(Path(argv[0]).parent) + ":/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "MPLBACKEND": "Agg",
    }
    from .generic_execution_lock import current, wait_process
    from .terminal_cohort import stop_child

    current()
    if lease_guard is not None:
        lease_guard()
    started = datetime.now(UTC)
    with (directory / "process.log").open("xb") as stream:
        child = runner(
            argv,
            cwd=Path.cwd(),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = wait_process(child, timeout=seconds, stop=stop_child, extra_guard=lease_guard)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)
            code = 124
    result = dict(
        argv=argv,
        exit_code=code,
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=(datetime.now(UTC) - started).total_seconds(),
    )
    save(receipt, result)
    if code:
        raise ValueError("new_case_model_process_failed")
    return result


def segman(
    sample_id,
    *,
    directory_adapter=False,
    isolated=False,
    system_path=False,
    registered=None,
    before_dispatch=None,
):
    from apps import segman_tool_registry as tool
    from apps.vectorengine_single_case import write_once

    from .generic_execution_lock import scoped_lock

    if registered is not None:
        from .generic_input import validate_registered

        registered = validate_registered(sample_id, registered)
    directory, binding, runtime, objects = registered or register(sample_id)
    with scoped_lock(directory / "lock", resource="segman-component"):
        task = (
            "segman-r03"
            if system_path
            else "segman-r02"
            if isolated
            else "segman-r01"
            if directory_adapter
            else "segman"
        )
        plan_path = directory / (task + "-plan.json")
        if not plan_path.exists():
            with tool.native_connection() as connection:
                plan = tool.resolve_component(connection, directory / "input.png", directory / task)
            write_once(plan_path, plan)
        plan = json.loads(plan_path.read_bytes())
        prediction = (directory / task / "prediction").resolve()
        outputs = {p + "_mask": str(prediction / (p + ".png")) for p in ("cup", "rim", "disc")}
        outputs.update({p: str(prediction / (p + ".png")) for p in ("labels", "overlay")})
        outputs["observation"] = str(prediction / "result.json")
        if plan["outputs"] != outputs or plan["argv"] != [
            str(tool.PYTHON),
            str(tool.RUNNER),
            "--image",
            str((directory / "input.png").resolve()),
            "--observation",
            outputs["observation"],
        ]:
            raise ValueError("new_case_native_command_binding")
        # Revalidate registry/resources even on recovery; no installation or qualification upgrade.
        with tool.native_connection() as connection:
            spec = tool.load_verified()
            manifest = tool.manifest(connection)
        if plan["manifest_checksum"] != tool.checksum(manifest) or manifest["provenance"][
            "installation_checksum"
        ] != tool.checksum(spec):
            raise ValueError("new_case_tool_manifest_changed")
        if plan["input_sha256"] != checksum((directory / "input.png").read_bytes())[7:]:
            raise ValueError("new_case_plan_image_changed")
        command = list(plan["argv"])
        if directory_adapter:
            prior = directory / "segman-process/process-claim.json"
            if prior.exists():
                confirmed = directory / "segman-process/process-result.json"
                if not confirmed.exists() or json.loads(confirmed.read_bytes())["exit_code"] != 1:
                    raise ValueError("segman_repair_requires_confirmed_prior_failure")
                if (
                    "artifacts/weight-integrity.json"
                    not in (directory / "segman-process/process.log").read_text()
                ):
                    raise ValueError("segman_repair_failure_outside_directory_defect")
            adapter_path = Path(
                "apps/segman_generic_runner.py"
                if registered is not None
                else "apps/segman_case_runner.py"
            )
            command[1] = str(adapter_path.resolve())
            if isolated:
                previous_claim = directory / "segman-r01-process/process-claim.json"
                if previous_claim.exists():
                    previous_result = directory / "segman-r01-process/process-result.json"
                    if (
                        not previous_result.exists()
                        or json.loads(previous_result.read_bytes())["exit_code"] != 1
                    ):
                        raise ValueError("segman_isolation_requires_confirmed_failure")
                    if (
                        "module 'platform' has no attribute 'system'"
                        not in (directory / "segman-r01-process/process.log").read_text()
                    ):
                        raise ValueError("segman_isolation_failure_outside_import_defect")
                command.insert(1, "-I")
            if system_path:
                previous_claim = directory / "segman-r02-process/process-claim.json"
                if previous_claim.exists():
                    previous_result = directory / "segman-r02-process/process-result.json"
                    if (
                        not previous_result.exists()
                        or json.loads(previous_result.read_bytes())["exit_code"] != 1
                    ):
                        raise ValueError("segman_path_requires_confirmed_failure")
                    if (
                        "No such file or directory: 'ldconfig'"
                        not in (directory / "segman-r02-process/process.log").read_text()
                    ):
                        raise ValueError("segman_path_failure_outside_environment_defect")
            write_once(
                directory
                / (
                    "directory-adapter-system-path.json"
                    if system_path
                    else "directory-adapter-isolated.json"
                    if isolated
                    else "directory-adapter.json"
                ),
                dict(
                    version="new-case-directory-adapter/1.0.0",
                    original_plan_checksum=digest(plan),
                    adapter_checksum=checksum(adapter_path.read_bytes()),
                    argv=command,
                    isolated_imports=isolated,
                    native_registry_unchanged=True,
                    fix="create_relocated_artifacts_directory_before_build",
                ),
            )
        if before_dispatch is not None:
            before_dispatch()
        result = process(command, directory=directory / (task + "-process"))
        measurements, assets = save_measurements(binding, plan, result, runtime, objects)
        write_once(directory / "measurements.json", measurements)
        write_once(directory / "tool-artifacts.json", [a.model_dump(mode="json") for a in assets])
        return dict(directory=str(directory), binding=binding, plan=plan, result=result)


def measurements_from_outputs(binding, plan):
    from apps.glaucoma_rim_radial import radial_profile
    from apps.segman_tool_registry import COMMIT, WEIGHT_SHA
    from packages.tool_benchmark.geometry_metrics import ratios
    from packages.tool_execution.qc import grayscale_png

    report = json.loads(Path(plan["outputs"]["observation"]).read_bytes())
    if (
        report["input_sha256"] != binding["image"]["checksum"][7:]
        or report["source_commit"] != COMMIT
        or report["checkpoint_sha256"] != WEIGHT_SHA
        or report["status"] != "success"
        or report["ground_truth_used"] is not False
    ):
        raise ValueError("new_case_model_output_binding")
    masks = {}
    shape = tuple(binding["image"]["dimensions"])
    for name in ("cup", "disc", "rim", "labels"):
        path = Path(plan["outputs"].get(name + "_mask", plan["outputs"].get(name, "")))
        raw = path.read_bytes()
        if checksum(raw)[7:] != report["outputs"][name + ".png"]["sha256"]:
            raise ValueError("new_case_mask_checksum")
        dimensions, pixels = grayscale_png(raw, 1048576)
        if dimensions != shape:
            raise ValueError("new_case_mask_shape")
        masks[name] = pixels
    labels = masks["labels"]
    for name, codes in (("cup", {2}), ("rim", {1}), ("disc", {1, 2})):
        if masks[name] != bytes(255 if v in codes else 0 for v in labels):
            raise ValueError("new_case_mask_semantics")
    values = ratios(labels, shape)
    try:
        radial = radial_profile(shape, masks["disc"], masks["cup"])
        values["radial_rim_minimum_ratio"] = radial["minimum"]["ratio"]
        radial_missing = None
    except ValueError as error:
        values["radial_rim_minimum_ratio"] = None
        radial_missing = str(error)
    return dict(
        schema_version="new-case-structure-measurements/1.0.0",
        case_code=binding["case_code"],
        input_checksum=binding["image"]["checksum"],
        measurements=values,
        radial_missing_reason=radial_missing,
        method="original_geometry_and_radial_owners",
        diagnosis=None,
        actual_model="SegMAN-B",
        research_only=True,
        observation_checksum=checksum(Path(plan["outputs"]["observation"]).read_bytes()),
    )


def save_measurements(binding, plan, result, runtime, objects):
    from db.runtime.repository.case_pipeline import identity
    from packages.contracts import Artifact
    from packages.grounding.models import canonical

    measurement = measurements_from_outputs(binding, plan)
    image = Artifact.model_validate_json(json.dumps(binding["image"]))
    artifacts = []
    payloads = {name: Path(path).read_bytes() for name, path in plan["outputs"].items()}
    payloads["measurements"] = canonical(measurement)
    namespace = ":" + binding["execution_namespace"] if binding.get("execution_namespace") else ""
    key = identity("TASK", "13-new-case-segman:" + binding["case_code"] + namespace)
    saved = runtime.read_observation_bundle(key)
    by_port = {a.extensions["output_port"]: a for a in saved} if saved else {}
    if saved and (len(saved) != len(payloads) or set(by_port) != set(payloads)):
        raise ValueError("new_case_tool_saved_inventory")
    for name, raw in payloads.items():
        uri = by_port[name].uri if saved else objects.promote(objects.stage(raw), checksum(raw))
        if saved and objects.read_bounded(uri, 4194304) != raw:
            raise ValueError("new_case_tool_saved_content")
        data = dict(
            schema_version="1.0",
            contract_type="Artifact",
            asset_code=identity("ASSET", binding["case_code"] + namespace + name + checksum(raw)),
            artifact_type_code="ARTIFACT:SLO_SEGMAN_" + name.upper(),
            uri=uri,
            checksum=checksum(raw),
            media_type="application/json"
            if name in ("observation", "measurements")
            else "image/png",
            qc_status="UNASSESSED",
            dimensions=list(image.dimensions)
            if name not in ("observation", "measurements")
            else [],
            measurements=[],
            derived_from=[image.asset_code],
            research_only=True,
            scope=dict(scope_type="CASE", case_code=binding["case_code"]),
            laterality=image.laterality,
            coordinate_space="PIXEL_2D",
            unit=None,
            extensions=dict(component_task="segman", output_port=name),
            provenance=dict(
                actor="TOOL_AGENT",
                created_at=result["finished_at"],
                generator_version="new-case-tools-1.0.0",
                source_codes=[image.asset_code],
                input_checksums=[image.checksum],
            ),
        )
        artifacts.append(Artifact.model_validate(data))
    runtime.register_observation_bundle(tuple(artifacts), key)
    if runtime.read_observation_bundle(key) != tuple(artifacts):
        raise ValueError("new_case_tool_artifact_readback")
    return measurement, artifacts

"""Native registered tool -> original Runtime Artifact owner, with exact recovery."""

import json
from datetime import UTC, datetime

from apps import fundus_tool_registry as tool
from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evaluation_preparation.case_execution import process
from packages.grounding.models import canonical, digest
from packages.object_store import checksum

from .fundus_measurements import measure


def current_manifest():
    with tool.native_connection() as connection:
        return tool.manifest(connection)


def result_directory(base, requirement, *, component_scoped=False):
    directory = base.output / "native-fundus"
    if component_scoped:
        directory /= digest((requirement["id"], requirement["component"]))[7:39]
    return directory


def read_saved(base, requirement, *, component_scoped=False, legacy_record=None):
    """Read one exact persisted component result; never execute a tool for recovery."""
    record = legacy_record
    if component_scoped:
        path = result_directory(base, requirement, component_scoped=True) / "accepted.json"
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("saved_report_native_symlink")
        if not path.is_file():
            raise ValueError("saved_report_native_fact_missing")
        record = json.loads(path.read_bytes())
        if record.get("schema_version") != "native-fundus-acceptance/1.1.0":
            raise ValueError("saved_report_native_component_version")
    if record is None:
        raise ValueError("saved_report_native_fact_missing")
    if (requirement["id"], requirement["component"]) != (
        record["plan"]["target"],
        record["plan"]["component"],
    ):
        raise ValueError("saved_report_native_binding")
    recheck(base, record)
    return record


def acquire(base, requirement, *, before_dispatch, component_scoped=False):
    component = next(c for c in base.manifest.components if c.id == requirement["component"])
    if (
        component.modality != "color_fundus"
        or component.kind != "image"
        or len(component.payload_ids) != 1
    ):
        raise ValueError("native_fundus_requires_single_color_fundus")
    base.prepared.recheck()
    raw = dict(base.prepared.payload_bytes)[component.payload_ids[0]]
    directory = result_directory(base, requirement, component_scoped=component_scoped)
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "accepted.json"
    if done.exists():
        if component_scoped:
            return read_saved(base, requirement, component_scoped=True)
        record = json.loads(done.read_bytes())
        if (record["plan"]["target"], record["plan"]["component"]) != (
            requirement["id"],
            component.id,
        ):
            raise ValueError("native_fundus_saved_component_binding")
        recheck(base, record)
        return record
    manifest = current_manifest()
    image = directory / "input.jpg"
    if image.exists():
        if image.read_bytes() != raw:
            raise ValueError("native_fundus_input_conflict")
    else:
        with image.open("xb") as stream:
            stream.write(raw)
    prediction = directory / "prediction"
    argv = tool.argv(image, prediction)
    plan = dict(
        tool=tool.CODE,
        version=tool.VERSION,
        implementation=tool.IMPLEMENTATION,
        manifest_checksum=digest(manifest),
        input_checksum=checksum(raw),
        component=component.id,
        argv=argv,
        target=requirement["id"],
    )
    write_once(directory / "plan.json", plan)
    before_dispatch()
    base.check()
    if current_manifest() != manifest:
        raise ValueError("native_fundus_manifest_changed_before_dispatch")
    from .execution_lease import native_resource_session

    with native_resource_session() as lease_guard:
        # Existing immutable tool command and installed worker remain unchanged.
        options = {"lease_guard": lease_guard} if lease_guard is not None else {}
        receipt = process(argv, directory=directory / "process", seconds=180, **options)
    base.check()
    if current_manifest() != manifest:
        raise ValueError("native_fundus_manifest_changed_after_return")
    predicted = json.loads((prediction / "result.json").read_bytes())
    if predicted["input_checksum"] != checksum(raw) or predicted["source_checksum"] != checksum(
        (tool.SOURCE / "ensemble-source-lock.json").read_bytes()
    ):
        raise ValueError("native_fundus_prediction_binding")
    for name, expected in predicted["artifacts"].items():
        if (
            name not in {"cup.png", "disc.png", "rim.png", "labels.png", "overlay.png"}
            or checksum((prediction / name).read_bytes()) != expected
        ):
            raise ValueError("native_fundus_output_changed")
    if set(predicted["artifacts"]) != {
        "cup.png",
        "disc.png",
        "rim.png",
        "labels.png",
        "overlay.png",
    }:
        raise ValueError("native_fundus_output_inventory")
    measurement = measure(prediction)
    write_once(directory / "measurements.json", measurement)
    payloads = {name: (prediction / name).read_bytes() for name in predicted["artifacts"]}
    payloads.update(
        prediction=canonical(predicted), measurements=canonical(measurement), plan=canonical(plan)
    )
    key = identity("TASK", "active-fundus:" + str(directory if component_scoped else base.output))
    saved = base.runtime.read_observation_bundle(key)
    by_port = {a.extensions["output_port"]: a for a in saved} if saved else {}
    if saved and set(by_port) != set(payloads):
        raise ValueError("native_fundus_saved_inventory")
    artifacts = []
    for name, data in payloads.items():
        uri = (
            by_port[name].uri
            if saved
            else base.objects.promote(base.objects.stage(data), checksum(data))
        )
        if base.objects.read_bounded(uri, 4194304) != data:
            raise ValueError("native_fundus_object_binding")
        artifact = Artifact.model_validate(
            dict(
                schema_version="1.0",
                contract_type="Artifact",
                asset_code=identity("ASSET", key + name + checksum(data)),
                artifact_type_code="ARTIFACT:FUNDUS_AUTOMORPH:" + name.upper().replace(".", "_"),
                uri=uri,
                checksum=checksum(data),
                media_type="image/png" if name.endswith(".png") else "application/json",
                qc_status="UNASSESSED",
                dimensions=[512, 512] if name.endswith(".png") else [],
                measurements=[],
                derived_from=[base.root.asset_code],
                research_only=True,
                scope=dict(scope_type="CASE", case_code=base.binding["case_code"]),
                laterality=component.laterality,
                coordinate_space="PIXEL_2D",
                unit=None,
                extensions=dict(
                    component_task=tool.CODE,
                    output_port=name,
                    input_checksum=checksum(raw),
                    component=component.id,
                ),
                provenance=dict(
                    actor="TOOL_AGENT",
                    created_at=receipt["finished_at"],
                    generator_version="active-fundus-1.0.0",
                    source_codes=[base.root.asset_code],
                    input_checksums=[base.root.checksum],
                ),
            )
        )
        artifacts.append(artifact)
    base.runtime.register_observation_bundle(tuple(artifacts), key)
    if base.runtime.read_observation_bundle(key) != tuple(artifacts):
        raise ValueError("native_fundus_readback")
    record = dict(
        schema_version="native-fundus-acceptance/1.1.0"
        if component_scoped
        else "native-fundus-acceptance/1.0.0",
        case_code=base.binding["case_code"],
        input_checksum=base.manifest.checksum,
        status=measurement["status"],
        measurement=measurement,
        prediction=predicted,
        plan=plan,
        bundle_key=key,
        artifacts=[a.model_dump(mode="json") for a in artifacts],
        artifact=next(
            a.model_dump(mode="json")
            for a in artifacts
            if a.extensions["output_port"] == "measurements"
        ),
        scope="native_registered_research_not_formal_06",
        process=receipt,
    )
    write_once(done, record)
    return record


def records(native):
    """Explicit collection envelope; old single-result bytes keep their meaning."""
    if native is None:
        return ()
    if native.get("schema_version") != "native-fundus-collection/1.0.0":
        return (native,)
    items = tuple(native["records"])
    keys = [(r["plan"]["target"], r["plan"]["component"]) for r in items]
    if (
        not 1 <= len(items) <= 4
        or len(keys) != len(set(keys))
        or any(r["schema_version"] != "native-fundus-acceptance/1.1.0" for r in items)
    ):
        raise ValueError("native_fundus_collection_binding")
    return items


def recheck(base, record):
    if (
        record["case_code"] != base.binding["case_code"]
        or record["input_checksum"] != base.manifest.checksum
        or record["plan"]["manifest_checksum"] != digest(current_manifest())
    ):
        raise ValueError("native_fundus_current_binding")
    saved = base.runtime.read_observation_bundle(record["bundle_key"])
    if [a.model_dump(mode="json") for a in saved or ()] != record["artifacts"]:
        raise ValueError("native_fundus_saved_binding")
    by_port = {a.extensions["output_port"]: a for a in saved}
    if set(by_port) != {
        "cup.png",
        "disc.png",
        "rim.png",
        "labels.png",
        "overlay.png",
        "prediction",
        "measurements",
        "plan",
    }:
        raise ValueError("native_fundus_saved_inventory")
    content = {}
    for artifact in saved:
        raw = base.objects.read_bounded(artifact.uri, 4194304)
        if checksum(raw) != artifact.checksum:
            raise ValueError("native_fundus_saved_object")
        if artifact.scope.case_code != base.binding["case_code"] or tuple(
            artifact.derived_from
        ) != (base.root.asset_code,):
            raise ValueError("native_fundus_saved_scope")
        content[artifact.extensions["output_port"]] = raw
    if (
        json.loads(content["measurements"]) != record["measurement"]
        or json.loads(content["prediction"]) != record["prediction"]
        or json.loads(content["plan"]) != record["plan"]
        or by_port["measurements"].model_dump(mode="json") != record["artifact"]
        or record["status"] != record["measurement"]["status"]
    ):
        raise ValueError("native_fundus_record_content_binding")


def preparation_guard(base):
    """One bounded pre-acquisition lease, inherited by the case campaign deadline."""
    path = base.output / "active-start.json"
    from .execution_lease import check_current

    check_current(base.output)
    if not path.exists():
        from datetime import timedelta

        now = datetime.now(UTC)
        write_once(
            path,
            dict(
                started_at=now.isoformat(),
                deadline=(
                    now + timedelta(seconds=getattr(base, "active_seconds", 2700))
                ).isoformat(),
                input_checksum=base.manifest.checksum,
            ),
        )
    lease = json.loads(path.read_bytes())
    if lease["input_checksum"] != base.manifest.checksum:
        raise ValueError("active_preparation_input_changed")
    base.active_start = lease

    def guard():
        check_current(base.output)
        if (
            datetime.now(UTC) >= datetime.fromisoformat(lease["deadline"])
            or (base.output / "cancel").exists()
            or (base.output / "result.json").exists()
            or (base.output / "protocol.json").exists()
        ):
            raise ValueError("active_preparation_stopped_or_already_started")

    return guard

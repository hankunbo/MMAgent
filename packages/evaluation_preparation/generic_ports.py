"""Output roles resolve to exact Runtime artifacts, not caller-selected historical paths."""

import json

from apps.vectorengine_single_case import write_once
from packages.contracts import Artifact
from packages.object_store import checksum


def install(registered):
    directory, binding, runtime, objects = registered
    path = directory / "tool-artifacts.json"
    if not path.exists():
        return
    artifacts = [Artifact.model_validate_json(json.dumps(a)) for a in json.loads(path.read_bytes())]
    ports = {}
    for a in artifacts:
        role = a.extensions["output_port"]
        if role in ports:
            raise ValueError("generic_duplicate_output_port")
        if (
            runtime.read_case_artifact(binding["case_code"], a.asset_code) != a
            or a.derived_from != (binding["image"]["asset_code"],)
            or checksum(objects.read_bounded(a.uri, 4194304)) != a.checksum
        ):
            raise ValueError("generic_output_owner_binding")
        ports[role] = a.model_dump(mode="json")
    write_once(
        directory / "output-ports.json",
        dict(
            schema_version="case-output-ports/1.0.0",
            case_code=binding["case_code"],
            input_checksum=binding["image"]["checksum"],
            execution_namespace=binding["execution_namespace"],
            ports=ports,
        ),
    )


def read(directory, binding, runtime, objects):
    path = directory / "output-ports.json"
    if not path.exists():
        return None
    value = json.loads(path.read_bytes())
    if (
        value.get("schema_version") != "case-output-ports/1.0.0"
        or value["case_code"] != binding["case_code"]
        or value["input_checksum"] != binding["image"]["checksum"]
        or value["execution_namespace"] != binding["execution_namespace"]
    ):
        raise ValueError("generic_output_scope")
    artifacts = {}
    raw = {}
    for role, body in value["ports"].items():
        a = Artifact.model_validate_json(json.dumps(body))
        if (
            a.extensions["output_port"] != role
            or a.derived_from != (binding["image"]["asset_code"],)
            or runtime.read_case_artifact(binding["case_code"], a.asset_code) != a
        ):
            raise ValueError("generic_output_owner_binding")
        data = objects.read_bounded(a.uri, 4194304)
        if checksum(data) != a.checksum:
            raise ValueError("generic_output_changed")
        artifacts[role], raw[role] = a, data
    if not {"cup_mask", "disc_mask", "measurements"} <= artifacts.keys():
        raise ValueError("generic_output_missing_port")
    measurement = json.loads(raw["measurements"])
    if (
        measurement["input_checksum"] != binding["image"]["checksum"]
        or measurement["case_code"] != binding["case_code"]
    ):
        raise ValueError("generic_measurement_scope")
    return artifacts, raw, measurement

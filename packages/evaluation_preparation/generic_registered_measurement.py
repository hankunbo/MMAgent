"""Current original Runtime measurements for an explicit new interpretation phase.

This port reads a confirmed tool successor. It does not execute a tool, accept a
medical interpretation, change a historical phase, or grant model requests.
"""

import json
from pathlib import Path

from apps.vectorengine_case_materials import ref
from packages.contracts import Artifact
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ObservationInput, TaskDefinition
from packages.object_store import checksum

TARGET = "criterion:RIM_RADIAL_LT_01"
SCOPE = "registered-partial-rim-observation/1.0.0"
METHOD = "registered_partial_rim_single_interval"
UNIT = "same_axis_disc_diameter_ratio"


def read(path):
    return json.loads(Path(path).read_bytes())


def current_record(key):
    """Re-read original input, manifest, committed object and successful process."""
    from apps import partial_rim_execution as execution

    registered, proof = execution.bound_inputs(key)
    root, _, runtime, objects = registered
    folder = root / execution.FOLDER
    request = read(folder / "request.json")
    execution.check_request(key, registered, proof, request)
    receipt = read(folder / "process/process-result.json")
    if receipt["exit_code"] != 0 or receipt["argv"] != request["plan"]["argv"]:
        raise ValueError("registered_measurement_confirmed_process_required")
    saved = runtime.read_observation_bundle(request["bundle_key"])
    if saved is None or len(saved) != 1:
        raise ValueError("registered_measurement_original_commit_required")
    artifact = saved[0]
    current = runtime.read_case_artifact(proof["case_code"], artifact.asset_code)
    if current != artifact.model_copy(
        update={"derived_from": tuple(sorted(artifact.derived_from))}
    ):
        raise ValueError("registered_measurement_current_artifact_changed")
    payload = objects.read_bounded(artifact.uri, 4194304)
    if checksum(payload) != artifact.checksum:
        raise ValueError("registered_measurement_original_object_changed")
    body = json.loads(payload)
    measurement = read(folder / "observation.json")
    execution.verify_measurement(measurement, proof)
    expected_body = dict(
        schema_version=execution.VERSION,
        request=request,
        measurement=measurement,
        process=receipt,
        medical_direction=None,
        clinical_accepted=False,
    )
    if canonical(body) != canonical(expected_body) or canonical(body) != payload:
        raise ValueError("registered_measurement_original_object_binding")
    expected_record = dict(
        schema_version=execution.VERSION,
        key=key,
        bundle_key=request["bundle_key"],
        request_checksum=digest(request),
        artifact=artifact.model_dump(mode="json"),
        measurement=measurement,
        process=receipt,
        clinical_accepted=False,
        review_accepted=False,
        new_api_requests=0,
    )
    if read(folder / "bound-observation.json") != expected_record:
        raise ValueError("registered_measurement_local_receipt_binding")
    sources = [
        Artifact.model_validate(proof["masks"][n]["original_asset"]) for n in ("disc", "cup")
    ]
    sources.append(Artifact.model_validate(proof["original_tool_artifact"]))
    if (
        artifact.scope.case_code != proof["case_code"]
        or artifact.artifact_type_code != "ARTIFACT:RESEARCH_PARTIAL_RIM_MEASUREMENT"
        or artifact.derived_from != tuple(a.asset_code for a in sources)
        or artifact.provenance.source_codes != tuple(a.asset_code for a in sources)
        or artifact.provenance.input_checksums != tuple(a.checksum for a in sources)
        or not artifact.research_only
    ):
        raise ValueError("registered_measurement_parent_or_case_binding")
    execution.check_request(key, registered, proof, request)
    return dict(
        schema_version=SCOPE,
        key=key,
        case_code=proof["case_code"],
        input_checksum=proof["input_checksum"],
        image=proof["image"],
        artifact=artifact.model_dump(mode="json"),
        bundle_key=request["bundle_key"],
        request_checksum=digest(request),
        process_checksum=digest(receipt),
        original_input_proof_checksum=digest(proof),
        measurement=measurement,
        manifest=request["manifest"],
        component_retention={
            n: proof["masks"][n]["component_view"]["retained_fraction"] for n in ("disc", "cup")
        },
        clinical_accepted=False,
        definition_accepted=False,
        dispatch_authorized=False,
    )


def measured_observation(record):
    """Bounded lossless summary references the complete original Runtime object."""
    if record.get("schema_version") != SCOPE or record.get("clinical_accepted") is not False:
        raise ValueError("registered_measurement_scope")
    measurement = record["measurement"]["measurement"]
    available, unavailable = measurement["valid_rays"], measurement["invalid_rays"]
    minimum = measurement["minimum_available_normalized_width"]
    if (
        type(available) is not int
        or type(unavailable) is not int
        or not 0 < available <= 360
        or available + unavailable != 360
        or minimum is None
    ):
        raise ValueError("registered_measurement_no_interpretable_rays")
    # A local minimum is never silently treated as the complete circumference.
    if available < 360 and measurement["full_circumference_minimum"] is not None:
        raise ValueError("registered_measurement_partial_global_conflict")
    compact = dict(
        minimum_available_ratio=minimum["width_over_same_axis_disc_diameter"],
        available_directions=available,
        total_directions=360,
        full_circumference_minimum=measurement["full_circumference_minimum"],
        image_angle_degrees=minimum["angle_degrees"],
        anatomical_sector="unknown",
        method="original_disc_centroid_single_interval_no_mask_edit",
        missing_directions="unobserved_geometry_not_normal_or_zero",
        limitations=(
            "mask geometry is not anatomical accuracy; sampled contact is not proven tissue loss"
        ),
        full_record=record["artifact"]["asset_code"],
    )
    raw = dict(
        status="obtained", method=METHOD, unit=UNIT, value=canonical(compact).decode(), image=None
    )
    raw["reference"] = ref("observation", "OBS:REGISTERED:" + TARGET, digest(raw))
    return validate(ObservationInput, raw)


def pending_definition(previous):
    """Describe measurement scope, explicitly withholding a new medical relation."""
    raw = previous.model_dump(mode="json", exclude={"reference"})
    raw.update(
        examination="在可测方向计算同轴盘径归一化的局部盘沿宽度；保留缺失方向与完整测量产物。",
        applicability_conditions=(
            "使用同病例原杯盘掩膜、原盘重心和单区间边界；"
            "只描述可测方向，不将局部最小值当作全周最小值，不推定未测方向或解剖扇区。"
            "掩膜边界接触不自动证明组织缺失。"
        ),
        interpretation=None,
        interpretation_status="missing",
    )
    raw["reference"] = ref("definition", "POLICY:REGISTERED:PARTIAL_RIM_PENDING", digest(raw))
    return validate(TaskDefinition, raw)


def project_registered(seed, record):
    """New phase only; same requirement/dependence group, explicitly new observation."""
    from .dynamic_stages import rebuild

    if (
        seed.material.scope.input_checksum != record["input_checksum"]
        or seed.material.scope.case_code != record["case_code"]
    ):
        raise ValueError("registered_measurement_phase_input_mismatch")
    observation = measured_observation(record)
    raw = seed.policy.model_dump(mode="json")
    rows = [r for r in raw["requirements"] if r["id"] == TARGET]
    items = [i for i in raw["items"] if i["id"] == TARGET]
    if len(rows) != 1 or len(items) != 1 or rows[0]["input_modalities"] != ["SLO"]:
        raise ValueError("registered_measurement_existing_requirement_required")
    row, item = rows[0], items[0]
    prior = next(
        (o for o in seed.material.observations if o.reference.code == item["observation_code"]),
        None,
    )
    if prior is None:
        raise ValueError("registered_measurement_original_observation_missing")
    previous_definition = validate(TaskDefinition, row["definition"])
    if canonical(item["definition"]) != canonical(previous_definition):
        raise ValueError("registered_measurement_previous_definition_conflict")
    definition = pending_definition(previous_definition)
    row.update(definition=definition.model_dump(mode="json"), missing_reason=None)
    resource = ref("resource", record["artifact"]["asset_code"], record["artifact"]["checksum"])
    retention = record["component_retention"]
    if any(
        type(retention[n]) not in (int, float) or not 0 <= retention[n] <= 1
        for n in ("disc", "cup")
    ):
        raise ValueError("registered_measurement_component_retention")
    item.update(
        observation_code=observation.reference.code,
        method=METHOD,
        unit=UNIT,
        definition=definition.model_dump(mode="json"),
        quality=dict(
            value=retention["disc"] * retention["cup"],
            method="component_view_retention_v1",
            reason=(
                "Q is original component retention, not accuracy; "
                "assess partial coverage and remaining uncertainty in O/A with reasons"
            ),
            source=resource.model_dump(mode="json"),
        ),
        dependency_paths=[
            list(dict.fromkeys([*path, resource.code])) for path in item["dependency_paths"]
        ],
    )
    # Earlier accepted local judgments saw the original context. Preserve those
    # facts byte-for-byte; only the target item points to the new measurement.
    # This is one replaced judgment, not two independent votes or a joint Review.
    observations = list(seed.material.observations)
    existing = [o for o in observations if o.reference.code == observation.reference.code]
    if existing:
        raise ValueError("registered_measurement_new_observation_identity_required")
    observations.append(observation)
    resources = list(seed.resources)
    if any(r.code == resource.code and r != resource for r in resources):
        raise ValueError("registered_measurement_resource_collision")
    if resource not in resources:
        resources.append(resource)
    state = rebuild(seed, validate(type(seed.policy), raw), observations, resources=resources)
    transition = dict(
        schema_version="registered-observation-transition/1.0.0",
        assessment_id=TARGET,
        previous_definition=previous_definition.model_dump(mode="json"),
        previous_observation=prior.model_dump(mode="json"),
        previous_observation_retained_as_context=True,
        observation=observation.model_dump(mode="json"),
        requirement_count_before=len(seed.policy.requirements),
        requirement_count_after=len(state.policy.requirements),
        original_group=item["correlation_group"],
        original_dependence_block=item["dependence_block"],
        confirmed_record_checksum=digest(record),
        bundle_key=record["bundle_key"],
        interpretation_pending=True,
        clinical_accepted=False,
    )
    return state, transition


def interpretation_basis(state, record, transition):
    """Definition owner input over the new observation, not an adopted definition."""
    requirement = next(r for r in state.policy.requirements if r.id == TARGET)
    item = next(i for i in state.policy.items if i.id == TARGET)
    observation = next(
        o for o in state.material.observations if o.reference.code == item.observation_code
    )
    row = requirement.model_dump(mode="json")
    row.update(
        component=record["image"]["asset_code"],
        acquisition_method="confirmed_tool_observation",
        expected_method=METHOD,
        expected_unit=UNIT,
    )
    origin = dict(
        schema_version=SCOPE,
        input_checksum=record["input_checksum"],
        target=state.material.goal_code,
        tool_artifact=record["artifact"],
        image_artifact=record["image"],
        tool_result_checksum=digest(record["measurement"]),
        projection_component="partial_rim",
        projection_checksum=digest(observation),
        registered_record_checksum=digest(record),
        method_semantics=dict(
            method=METHOD,
            unit=UNIT,
            definition="local rim width divided by the complete same-axis disc diameter",
            reference_match="partial_geometry_not_global_minimum_or_anatomical_sector",
            use="same_rim_requirement_with_explicit_method_revision_not_an_independent_vote",
        ),
        qualification="original_runtime_registered_measurement_not_clinical_or_definition_acceptance",
    )
    return json.loads(
        canonical(
            dict(
                schema_version="confirmed-generic-interpretation-basis/1.1.0",
                requirement=row,
                observation=observation.reference,
                method=METHOD,
                unit=UNIT,
                input_projection_checksum=digest((row, origin)),
                observation_binding=digest((observation, origin)),
                origin=origin,
                observation_value=observation,
                observation_transition=transition,
            )
        )
    )

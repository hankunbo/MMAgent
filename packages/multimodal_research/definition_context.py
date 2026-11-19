"""Public requirement semantics for drafting, never current-use authorization."""

from packages.grounding.models import validate
from packages.llm_provider.models import TaskDefinition


def public_examination(row):
    """Remove only the exact material-owner component suffix, not medical text.

    Original definitions and checksums are never changed. Equivalence still
    requires the original exact component binding in the expansion owner.
    """
    definition = validate(TaskDefinition, row["definition"])
    name = definition.examination
    component = row.get("component")
    suffix = " · " + component if isinstance(component, str) and component else None
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def same_examination(left, right):
    return left["component"] == right["component"] and (
        public_examination(left).strip().casefold() == public_examination(right).strip().casefold()
    )


def requirement_context(rows, query_plan, observations):
    """Allowlist original owner-built context; exclude patient values and answers."""
    if len(rows) > 64:
        raise ValueError("definition_context_capacity")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("definition_context_duplicate_requirement")
    targets = {item for plan in query_plan for item in plan.get("target_relation_gap_ids", ())}
    unbound = {item for plan in query_plan for item in plan.get("unbound_interpretation_ids", ())}
    if not (targets | unbound) <= set(ids):
        raise ValueError("definition_context_unknown_requirement")
    entries = []
    components = {}
    for row in rows:
        definition = validate(TaskDefinition, row["definition"])
        observation = observations.get(row["id"])
        method, unit = row.get("expected_method"), row.get("expected_unit")
        if observation is not None:
            if method is not None and method != observation.method:
                raise ValueError("definition_context_method_binding")
            if unit is not None and unit != observation.unit:
                raise ValueError("definition_context_unit_binding")
            method, unit = observation.method, observation.unit
        elif row["acquisition_method"] == "vision":
            method, unit = method or "live_image_observation", unit or "qualitative"
        component = row["component"]
        if component is not None:
            if not isinstance(component, str) or not component:
                raise ValueError("definition_context_component")
            # Dataset IDs may contain identifiers. Only a per-request alias leaves
            # the owner; original rows still bind the actual component at admission.
            component = components.setdefault(component, "input-component:" + str(len(components)))
        entries.append(
            dict(
                requirement_id="requirement:" + str(len(entries)),
                component=component,
                examination=public_examination(row),
                applicability_conditions=definition.applicability_conditions,
                existing_interpretation=definition.interpretation,
                interpretation_status=definition.interpretation_status,
                input_modalities=list(row["input_modalities"]),
                acquisition_method=row["acquisition_method"],
                method=method,
                unit=unit,
                requested_revision=(
                    "target_relation"
                    if row["id"] in targets
                    else "interpretation"
                    if row["id"] in unbound
                    else "not_requested_by_current_gap_plan"
                ),
            )
        )
    return dict(
        schema_version="definition-requirement-context/1.0.0",
        requirements=entries,
        semantics="proposal_context_not_authorization",
        same_examination_revision="original_owner_checks_input_method_and_source",
        duplicate_vote_allowed=False,
        empty_proposal_allowed=True,
    )


def public_gap_context(rows, query_plan):
    """Keep the same per-request aliases in all Source gap references.

    Original plans stay intact in owner receipts. Unknown fields are not copied
    into the public prompt, and absent-definition slots have a separate domain.
    """
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("definition_context_duplicate_requirement")
    aliases = {key: "requirement:" + str(index) for index, key in enumerate(ids)}
    absent = {}
    result = []
    for plan in query_plan:
        public = {
            key: plan[key]
            for key in ("query", "selected", "uncovered_modalities", "reason", "retrieval_mode")
            if key in plan
        }
        for key in ("unbound_interpretation_ids", "target_relation_gap_ids"):
            if key in plan:
                if any(item not in aliases for item in plan[key]):
                    raise ValueError("definition_context_unknown_requirement")
                public[key] = [aliases[item] for item in plan[key]]
        if "missing_definition_ids" in plan:
            public["missing_definition_ids"] = [
                absent.setdefault(item, "missing-definition:" + str(len(absent)))
                for item in plan["missing_definition_ids"]
            ]
        result.append(public)
    return result


def owner_capabilities(payload, base, rows, *, confirmed_owner=None):
    """New request version: raw inputs and confirmed methods are separate domains.

    Metadata advertises resolvable selectors, not satisfied conditions. The
    acceptance owner reloads inputs and original facts for each new use.
    """
    from packages.evidence_reuse.definition_dependencies import input_aliases

    from .component_observations import ComponentObservations
    from .confirmed_numeric_observations import ConfirmedNumericObservations

    mapping = input_aliases(base.manifest)
    by_id = {c.id: alias for alias, c in mapping.items()}
    capabilities = []
    for alias, c in mapping.items():
        methods = []
        if c.status in {"obtained", "partial"}:
            methods = (
                ["live_image_observation"]
                if c.kind in {"image", "volume", "image_series"}
                else ["source_numeric_read"]
                if c.kind in {"scalar", "vector", "map"}
                else []
            )
        capabilities.append(
            dict(
                input_alias=alias,
                kind=c.kind,
                modality=c.modality,
                concept=c.concept,
                status=c.status,
                installed_methods=methods,
                unit=(c.unit or "source_units_unqualified")
                if "source_numeric_read" in methods
                else None,
                current_values_and_method_still_require_owner_validation=True,
            )
        )
    confirmed = []
    owner = getattr(base, "component_observations", None)
    if confirmed_owner is not None:
        if (
            type(confirmed_owner) is not ConfirmedNumericObservations
            or confirmed_owner.base is not base
        ):
            raise ValueError("definition_numeric_capability_owner_binding")
        owner = confirmed_owner
    entries = payload["existing_requirement_context"]["requirements"]
    for row, entry in zip(rows, entries, strict=True):
        entry["input_alias"] = by_id.get(row["component"])
        if type(owner) is ConfirmedNumericObservations and row["id"] in owner.input_rows:
            owner.check_projection(row["id"])
            confirmed.append(
                dict(
                    requirement_alias=entry["requirement_id"],
                    input_alias=entry["input_alias"],
                    method=entry["method"],
                    unit=entry["unit"],
                    resolvable_conditions=["confirmed_method"],
                    semantics="current_original_input_projection_reconstruction_required",
                    exclusions=[
                        "clinical_quality",
                        "test_reliability",
                        "stage_relationship",
                        "mask_measurement_available",
                    ],
                )
            )
            metadata = owner.method_metadata(row["id"])
            if metadata is not None:
                # Only public author prose/locator; no numeric value, patient ID,
                # payload, actual observation or case component identifier leaves.
                confirmed[-1]["author_method_metadata"] = metadata
                confirmed[-1]["metadata_scope"] = (
                    "owner_revalidated_release_field_meaning_not_individual_quality_or_direction"
                )
        if type(owner) is ComponentObservations and owner.target == row["id"]:
            owner.check_projection()  # no dispatch or source authorization implied
            confirmed.append(
                dict(
                    requirement_alias=entry["requirement_id"],
                    input_alias=entry["input_alias"],
                    method=entry["method"],
                    unit=entry["unit"],
                    resolvable_conditions=[
                        "confirmed_method",
                        "artifact_integrity",
                        "mask_measurement_available",
                    ],
                    semantics="current_owner_resolution_required_not_asserted_satisfied",
                    exclusions=[
                        "segmentation_accuracy",
                        "clinical_quality",
                        "visual_field_availability",
                        "stage_relationship",
                    ],
                )
            )
    payload["input_capabilities"] = capabilities
    payload["confirmed_observation_capabilities"] = confirmed
    payload["dependency_capability_version"] = (
        "definition-owner-capabilities/1.1.0"
        if type(owner) is ConfirmedNumericObservations
        else "definition-owner-capabilities/1.0.0"
    )
    if type(owner) is ConfirmedNumericObservations and owner.method_context_profile is not None:
        payload["dependency_capability_version"] = "definition-owner-capabilities/1.2.0"
        payload["author_method_contract"] = {
            "meaning": "Public author field semantics attached to this exact confirmed method.",
            "use": (
                "Use explicitly documented measurement region and method; "
                "do not infer from abbreviations alone."
            ),
            "qualification": (
                "The original owner revalidates these source bytes and field bindings "
                "with confirmed_method."
            ),
            "limits": (
                "No new patient value, verified field unit, individual reliability, norm or "
                "disease relation. Keep other unverified conditions unresolved."
            ),
        }
    return payload


def candidate_contract_choices(payload):
    """Copy owner metadata, never infer an observation or medical relationship.

    This opt-in request projection is not a response repair or authorization.
    Source candidates still pass the original schema and current acceptance.
    """
    from copy import deepcopy

    from packages.evidence_reuse.definitions import CONFIRMED_METHOD_CONDITION

    result = deepcopy(payload)
    raw = []
    for item in payload["input_capabilities"]:
        for method in item["installed_methods"]:
            if method not in {"live_image_observation", "source_numeric_read"}:
                continue
            vision = method == "live_image_observation"
            unit = "qualitative" if vision else item["unit"]
            if not isinstance(unit, str) or not unit:
                raise ValueError("definition_contract_choice_unit")
            raw.append(
                dict(
                    copy_fields=dict(
                        schema_version="source-definition-draft/6.1.0",
                        input_aliases=[item["input_alias"]],
                        modalities=[item["modality"]],
                        acquisition="vision" if vision else "source_measurement",
                        method=method,
                        unit=unit,
                    ),
                    input_condition_selectors=dict(
                        kind="patient_input",
                        predicate="input_available" if vision else "source_numeric_available",
                        input_alias=item["input_alias"],
                        modality=item["modality"],
                        concept=item["concept"],
                        requirement_alias=None,
                        quality_check=None,
                        method=None if vision else method,
                        unit=None if vision else unit,
                    ),
                )
            )
    entries = {
        e["requirement_id"]: e for e in payload["existing_requirement_context"]["requirements"]
    }
    confirmed = []
    for item in payload["confirmed_observation_capabilities"]:
        entry = entries.get(item["requirement_alias"])
        if (
            entry is None
            or any(not isinstance(item.get(k), str) or not item[k] for k in ("method", "unit"))
            or any(item[k] != entry[k] for k in ("method", "unit"))
        ):
            raise ValueError("definition_contract_choice_confirmed_binding")
        confirmed.append(
            dict(
                copy_fields=dict(
                    schema_version="source-definition-draft/7.1.0",
                    requirement_alias=item["requirement_alias"],
                    examination=entry["examination"],
                    modalities=entry["input_modalities"],
                    acquisition="confirmed_observation",
                    method=item["method"],
                    unit=item["unit"],
                ),
                method_condition_selectors=dict(
                    text=CONFIRMED_METHOD_CONDITION,
                    kind="method",
                    predicate="confirmed_method",
                    requirement_alias=item["requirement_alias"],
                    method=item["method"],
                    unit=item["unit"],
                    input_alias=None,
                    modality=None,
                    concept=None,
                    quality_check=None,
                ),
            )
        )
    result["candidate_contract_choices"] = dict(
        schema_version="definition-candidate-contract-choices/1.0.0",
        raw_input=raw,
        confirmed_observation=confirmed,
        semantics="format_choices_not_definitions_or_authorization",
        empty_confirmed_disallows_confirmed_observation_candidates=True,
        medical_fields_and_required_conditions_must_be_source_grounded=True,
        missing_input_definitions_remain_allowed_with_explicit_unresolved_conditions=True,
    )
    return result

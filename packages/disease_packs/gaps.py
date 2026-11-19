"""Emit existing v1 gap contracts without installing, downloading or invoking any tool."""

from packages.contracts import CapabilityGap, ToolRequirementDocument
from packages.graph_engine import canonical_checksum

from .models import parse_pack


def missing_capability(pack, graph, slot, *, available_inputs=()):
    pack = parse_pack(pack)
    if slot not in pack.definition.evidence_slots or slot.capability_code is None:
        raise ValueError("a declared tool capability slot is required")
    if slot.node_code not in {n.node_code for n in graph.nodes} or graph.scope.run_id is None:
        raise ValueError("capability gap requires an existing node in a run")
    token = canonical_checksum((pack.checksum, canonical_checksum(graph), slot.node_code))[
        7:31
    ].upper()
    requirement_id, job_id = "REQUIREMENT:OID_PACK" + token, "TASK:OID_PACK" + token
    provenance = graph.provenance.model_dump(mode="json") | {
        "actor": "TOOL_AGENT",
        "generator_version": "04C.1.0.0",
        "input_checksums": [pack.checksum, canonical_checksum(graph)],
    }
    common = dict(
        schema_version="1.0",
        laterality=graph.laterality.value,
        coordinate_space="NOT_APPLICABLE",
        unit=None,
        provenance=provenance,
        target_node_code=slot.node_code,
        origin_run_id=graph.scope.run_id,
        capability_code=slot.capability_code,
    )
    scope = dict(
        scope_type="RUN",
        case_code=graph.scope.case_code,
        graph_code=graph.graph_code,
        run_id=graph.scope.run_id,
    )
    gap = CapabilityGap.model_validate(
        common
        | dict(
            contract_type="CapabilityGap",
            scope=scope,
            gap_id=requirement_id,
            gap_codes=["NO_IMPLEMENTATION"],
            observation="UNOBSERVED_TOOL_GAP",
            capability="UNAVAILABLE_CAPABILITY",
            validity="UNASSESSED",
            clinical_effect="NONE",
            alternative_paths=[],
            blocks_required_path=True,
        )
    )
    artifact = dict(required=True, laterality=graph.laterality.value, unit=None)
    requirement = ToolRequirementDocument.model_validate(
        common
        | dict(
            contract_type="ToolRequirementDocument",
            scope=scope | dict(scope_type="TOOL_EXTENSION", task_id=job_id, graph_code=None),
            requirement_id=requirement_id,
            extension_job_id=job_id,
            clinical_purpose=pack.definition.target_proposition,
            available_inputs=list(available_inputs),
            required_input_contract=[
                artifact
                | dict(
                    artifact_type_code=slot.input_artifact_code,
                    coordinate_space=slot.input_coordinate_space,
                )
            ],
            required_output_contract=[
                artifact
                | dict(
                    artifact_type_code=slot.output_artifact_code,
                    coordinate_space=slot.output_coordinate_space,
                )
            ],
            attempted_candidates=[],
            # DatasetReferences are declarations, NOT a frozen 06E benchmark manifest.
            validation_plan=dict(frozen_datasets=[], metrics=[], threshold_policy=None),
            blocked_nodes=[slot.node_code],
            conclusion_effect="INSUFFICIENT_EVIDENCE",
            available_alternatives=[],
            replay_when_available=True,
        )
    )
    return gap, requirement

"""Age rule application from an owner-verified current report, without patient values."""

import json

from packages.grounding.models import digest
from packages.memory.archive_models import ArchiveRecord


def prepare_age(owner, *, key, case_code, run_id, origin, parent, refs, selected):
    definition = selected["definition"]
    if (
        selected["method"] != "dataset_public_age"
        or selected["unit"] != "year"
        or definition["interpretation_status"] != "installed"
        or not definition["source_refs"]
    ):
        raise ValueError("memory_age_rule_binding")
    observation = next(
        (
            o
            for o in parent["material"]["observations"]
            if o["reference"]["code"] == selected["observation_code"]
        ),
        None,
    )
    contexts = [
        a for a in refs.values() if a.artifact_type_code == "ARTIFACT:RESEARCH_INPUT_CONTEXT"
    ]
    if observation is None or observation["status"] != "obtained" or len(contexts) != 1:
        raise ValueError("memory_age_observation_missing")
    context = contexts[0]
    context_value = json.loads(owner.read_bytes(context))
    observation_value = json.loads(observation["value"])
    if (
        context_value.get("schema_version") != "dataset-public-context/1.0.0"
        or context_value.get("case_code") != case_code
        or context_value.get("projection_policy") != "fairvlmed-public-age/1.0.0"
        or context_value["age"]["status"] != "obtained"
        or context_value["age"]["value"] is None
        or observation_value.get("age") != context_value["age"]["value"]
    ):
        raise ValueError("memory_age_observation_binding")
    if context.asset_code not in {k for path in selected["dependency_paths"] for k in path}:
        raise ValueError("memory_age_context_dependency")
    return ArchiveRecord(
        key=key,
        kind="tool_experience",
        case_code=case_code,
        run_id=run_id,
        applicability=("USE:RESEARCH_WORKFLOW",),
        nature="source_fact",
        sources=tuple(
            owner.reference(case_code=case_code, run_id=run_id, code=k) for k in sorted(refs)
        ),
        detail=dict(
            type="branch_workflow",
            target="criterion:AGE",
            modality="age",
            modality_source="original_bound_runtime_context",
            origin_code=origin.asset_code,
            outcome="recorded",
            steps=(
                dict(
                    operation="read_input",
                    tool="dataset_public_age",
                    version="1.0.0",
                    implementation="original_public_context_owner",
                    source_checksum=context.checksum,
                    input_roles=("current_case_age",),
                    output_roles=("age_in_years",),
                    artifact_codes=(context.asset_code,),
                ),
                dict(
                    operation="review",
                    tool="source_bound_review",
                    version="1.0.0",
                    implementation="original_research_acceptance",
                    source_checksum=digest(definition),
                    input_roles=("age_in_years", "current_source_bound_definition"),
                    output_roles=("new_case_review",),
                    artifact_codes=(origin.asset_code,),
                ),
            ),
        ),
    )

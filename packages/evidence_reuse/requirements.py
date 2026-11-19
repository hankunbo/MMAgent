"""Explain catalog gaps using the actual projected owner definitions, not old flags."""

import json
from pathlib import Path

from packages.dataset_adapters.components import available_concepts
from packages.grounding.models import digest


def audit(catalog, plan, components):
    registry = json.loads(Path("configs/research/evidence_context_roles_v1.json").read_bytes())
    if registry["schema_version"] != "evidence-context-roles/1.0.0":
        raise ValueError("reuse_context_roles_version")
    selected = {r["id"]: r for r in plan["requirements"]}
    if len(selected) != len(plan["requirements"]):
        raise ValueError("reuse_duplicate_requirement")
    if set(selected) - set(catalog["tasks"]):
        raise ValueError("reuse_unknown_requirement")
    source_refs = {s["code"]: s for s in plan["sources"]}
    available = set(available_concepts(components))
    rows = []
    for key, task in sorted(catalog["tasks"].items()):
        requirement = selected.get(key)
        definition = requirement["definition"] if requirement else task.get("definition")
        role = "disease_requirement"
        reasons, action = [], "review_current_observation"
        if requirement:
            refs = definition.get("source_refs", [])
            if not refs or any(source_refs.get(r["code"]) != r for r in refs):
                raise ValueError("reuse_requirement_source_binding")
            missing = sorted(set(requirement["input_modalities"]) - available)
            if definition["interpretation_status"] != "installed":
                reasons.append("definition_not_installed")
                action = "extract_definition_from_current_sources"
            if missing:
                reasons.append("patient_input_missing")
                if not reasons[:-1]:
                    action = "record_gap_no_patient_value_invention"
            disposition = "scheduled"
            target = key
        elif key in catalog.get("merges", {}):
            mapping = catalog["merges"][key]
            target = mapping["target"]
            if target not in selected:
                raise ValueError("reuse_merge_target")
            disposition, role, missing = "merged", "shared_question", []
            reasons.append("same_question_no_extra_vote")
            action = "follow_bound_requirement"
        elif task.get("use") in {"measurement_quality_or_input_context", "observation_only"}:
            disposition, role, missing, target = "context_or_dependency", "context", [], None
            reasons.append("no_independent_disease_claim")
            if task.get("related_definition"):
                role = "observation_alias"
                reasons.append("related_definition_not_extra_evidence")
                target = task["related_definition"]
            action = "retain_context_then_bind_actual_consumer"
            entry = registry["roles"].get(key)
            if entry:
                role = entry["role"]
                missing = sorted(set(entry["inputs"]) - available)
                target = entry["consumer"]
                if missing:
                    reasons.append("patient_input_missing")
                    action = "record_context_gap_no_disease_direction"
        else:
            disposition, missing, target = "definition_backlog", [], None
            if task.get("missing_exact_source_ids"):
                reasons.append("exact_source_missing")
                action = "bounded_source_search"
            elif task.get("selected_source_ids"):
                reasons.append("definition_not_installed")
                action = "extract_definition_from_current_sources"
            else:
                reasons.append("medical_relationship_not_bound")
                action = "bounded_source_search"
        rows.append(
            dict(
                id=key,
                role=role,
                disposition=disposition,
                reasons=reasons,
                target=target,
                missing_components=missing,
                next_action=action,
                definition_checksum=digest(definition),
                counts_in_coverage=key in selected,
                independent_disease_vote=False,
            )
        )
    return dict(
        schema_version="requirement-resolution/1.0.0",
        catalog_checksum=digest(catalog),
        context_registry_checksum=digest(registry),
        components_checksum=digest(components),
        requirement_checksum=digest(plan["requirements"]),
        frozen_denominator=len(selected),
        rows=rows,
        qualitative_relations_allowed=True,
        search_policy=dict(
            max_queries=3,
            max_candidates=5,
            no_progress_limit=1,
            execution="not_dispatched_by_audit",
        ),
        medical_thresholds_added=False,
    )


def align_acquisition_requirements(frozen, current):
    """Compare immutable medical requirements while allowing original method variants."""
    from packages.evaluation_preparation.quality_materials import compatible_acquisition_definition

    if len(frozen) != len(current):
        raise ValueError("upstream_definition_changed")
    aligned = []
    for old, new in zip(frozen, current, strict=True):
        if old["id"] != new["id"] or not compatible_acquisition_definition(
            old["id"], old["definition"], new["definition"]
        ):
            raise ValueError("upstream_definition_changed")
        row = dict(new, definition=old["definition"], basis_kind=old["basis_kind"])
        if row != old:
            raise ValueError("upstream_definition_changed")
        aligned.append(row)
    return aligned

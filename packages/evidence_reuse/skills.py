"""Compile research procedure hints to existing narrow ports, never arbitrary code.

Only the two tested recipes are executable; unknown skills are explicit gaps.
Current 05/06 and research owners still authorize every dispatched operation.
"""

import json
from pathlib import Path

from packages.dataset_adapters.components import available_concepts
from packages.evidence_reuse.models import SkillPlan, check
from packages.grounding.models import digest

REGISTRY = Path("configs/research/evidence_skill_ports_v1.json")
INVALID_MEMORY = {"memory_parent_invalid", "memory_source_withdrawn", "memory_source_missing"}


def recipes():
    value = json.loads(REGISTRY.read_bytes())
    if (
        set(value) != {"schema_version", "recipes"}
        or value["schema_version"] != "evidence-skill-port-registry/1.0.0"
    ):
        raise ValueError("skill_registry_version")
    result = value["recipes"]
    for recipe in result.values():
        if set(recipe) != {
            "targets",
            "modality",
            "inputs",
            "tools",
            "operations",
            "steps",
            "fallback",
        }:
            raise ValueError("skill_recipe_fields")
        for field in ("targets", "inputs", "tools", "operations"):
            if not recipe[field] or len(set(recipe[field])) != len(recipe[field]):
                raise ValueError("skill_recipe_members")
            recipe[field] = tuple(recipe[field])
    return result


def select(memory, recipe, query_case):
    """One bounded original-owner read per candidate. Not published planning eligibility."""
    rejected = []
    if memory is None:
        return None, [dict(reason="memory_not_installed_baseline_available")]
    for key in memory.repository.procedure_keys(
        targets=recipe["targets"], modality=recipe["modality"], limit=20
    ):
        try:
            result = memory.repository.read(key)
        except ValueError as error:
            if str(error) not in INVALID_MEMORY:
                raise
            rejected.append(dict(key=key, reason=str(error)))
            continue
        record = result.record
        if record.kind != "tool_experience" or record.detail.target not in recipe["targets"]:
            continue
        detail = record.detail
        if record.case_code == query_case:
            rejected.append(dict(key=key, reason="same_case_not_external_experience"))
        elif detail.modality != recipe["modality"]:
            rejected.append(dict(key=key, reason="modality_mismatch"))
        elif (
            tuple(s.operation for s in detail.steps) != recipe["operations"]
            or tuple(s.tool for s in detail.steps) != recipe["tools"]
        ):
            rejected.append(dict(key=key, reason="procedure_not_supported_by_installed_ports"))
        else:
            # Only method references leave this owner. Never return artifacts or values.
            return dict(key=key, checksum=result.checksum), rejected
    return None, rejected


def compile_plans(plan, components, *, memory=None):
    installed = recipes()
    available = set(available_concepts(components))
    results, decisions = [], []
    for requirement in plan["requirements"]:
        key = requirement["id"]
        if key not in installed:
            decisions.append(dict(id=key, reason="original_baseline_no_compiled_skill"))
            continue
        recipe = installed[key]
        if requirement["definition"]["interpretation_status"] != "installed":
            decisions.append(dict(id=key, reason="definition_not_installed"))
            continue
        missing = tuple(sorted(set(recipe["inputs"]) - available))
        chosen, rejected = select(memory, recipe, components.scope) if not missing else (None, [])
        row = check(
            SkillPlan,
            dict(
                schema_version="research-skill-plan/1.0.0",
                scope=components.scope,
                input_checksum=plan["input_checksum"],
                requirement_id=key,
                definition_checksum=digest(requirement["definition"]),
                components_checksum=digest(components),
                origin="research_memory_procedure" if chosen else "current_owner_baseline",
                memory_key=chosen["key"] if chosen else None,
                memory_checksum=chosen["checksum"] if chosen else None,
                steps=recipe["steps"],
                missing_components=missing,
                fallback=recipe["fallback"],
            ),
        )
        results.append(row.model_dump(mode="json"))
        decisions.append(dict(id=key, rejected=rejected, selected=chosen))
    return dict(
        plans=results,
        decisions=decisions,
        search_limit_per_recipe=20,
        reuse_policy="research_procedure_hints_not_published_memory",
        clinical_evidence=False,
    )


def recheck_plans(value, plan, components, *, memory):
    installed = recipes()
    current = {r["id"]: r for r in plan["requirements"]}
    for raw in value["plans"]:
        row = check(SkillPlan, raw)
        if (
            row.scope != components.scope
            or row.input_checksum != plan["input_checksum"]
            or row.components_checksum != digest(components)
            or row.requirement_id not in current
            or row.definition_checksum != digest(current[row.requirement_id]["definition"])
        ):
            raise ValueError("skill_current_binding")
        if row.requirement_id not in installed:
            raise ValueError("skill_recipe_not_installed")
        recipe = installed[row.requirement_id]
        expected = [
            dict(
                s,
                depends_on=list(s["depends_on"]),
                input_roles=list(s["input_roles"]),
                output_roles=list(s["output_roles"]),
            )
            for s in recipe["steps"]
        ]
        if [s.model_dump(mode="json") for s in row.steps] != expected:
            raise ValueError("skill_port_plan_changed")
        missing = tuple(sorted(set(recipe["inputs"]) - set(available_concepts(components))))
        if row.missing_components != missing:
            raise ValueError("skill_input_availability_changed")
        if row.memory_key:
            if memory is None:
                raise ValueError("skill_memory_owner_missing")
            record = memory.recheck(row.memory_key, row.memory_checksum).record
            if (
                record.case_code == components.scope
                or record.detail.target not in recipe["targets"]
                or record.detail.modality != recipe["modality"]
                or tuple(s.tool for s in record.detail.steps) != recipe["tools"]
                or tuple(s.operation for s in record.detail.steps) != recipe["operations"]
            ):
                raise ValueError("skill_memory_recipe_changed")
    return value

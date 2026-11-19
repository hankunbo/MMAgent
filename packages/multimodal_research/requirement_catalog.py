"""Source-bound examination catalog and input-representation matching.

This is a view over registered research definitions, not another Knowledge
database or an execution grant. Current source qualification remains with the
original active-requirement/Knowledge owners. Dataset IDs are never selectors.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.dataset_adapters.models import Value
from packages.dataset_adapters.multimodal.capabilities import InputCapabilities
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.grounding.models import canonical, digest, validate

POLICY = "proposition-input-representation/1.0.0"
DEFAULT_PROFILE = "configs/research/multimodal_catalog_acquisition_v2.json"
FULFILLMENT = "same-component-measurement-fulfillment/1.0.0"


class Prerequisites(Value):
    representations: tuple[
        Literal["image", "image_series", "volume", "map", "vector", "scalar"], ...
    ] = Field(min_length=1)
    # Empty means any concept within the explicitly declared modality/kind.
    concepts: tuple[str, ...] = ()
    scope: Literal["visible_region_pending", "source_measurement", "paired_eyes"]
    required_unit: str | None = None


class Examination(Value):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.:-]{1,100}$")
    parent: str
    modality: str
    method: Literal["vision", "disc_cup_measurement", "paired_observation", "source_measurement"]
    source: str
    quote: str = Field(min_length=12)
    title: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    evidence_unit: str
    goals: tuple[str, ...] = Field(min_length=1)
    input_capabilities: Prerequisites
    equivalent_to: str | None = None

    @model_validator(mode="after")
    def method_input(self):
        if len(set(self.goals)) != len(self.goals):
            raise ValueError("catalog_duplicate_goal")
        if self.method in {"vision", "disc_cup_measurement"} and self.input_capabilities.scope != (
            "visible_region_pending"
        ):
            raise ValueError("catalog_visual_scope")
        if self.method == "paired_observation" and self.input_capabilities.scope != "paired_eyes":
            raise ValueError("catalog_pair_scope")
        return self


class Group(Value):
    id: str
    title: str
    parent: str


class RequirementCatalog(Value):
    schema_version: Literal["research-requirement-catalog/1.0.0"]
    id: str
    provenance: str
    nodes: tuple[Group, ...]
    checks: tuple[Examination, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def references(self):
        by_id = {c.id: c for c in self.checks}
        nodes = {n.id: n for n in self.nodes}
        if len(by_id) != len(self.checks) or len(nodes) != len(self.nodes) or "root" in nodes:
            raise ValueError("catalog_duplicate_identity")
        for node in nodes:
            seen, current = set(), node
            while current != "root":
                if current in seen or current not in nodes:
                    raise ValueError("catalog_hierarchy")
                seen.add(current)
                current = nodes[current].parent
        for check in self.checks:
            if check.parent not in nodes:
                raise ValueError("catalog_check_parent")
            if check.equivalent_to is not None:
                original = by_id.get(check.equivalent_to)
                if original is None or original.equivalent_to is not None or original == check:
                    raise ValueError("catalog_equivalence_target")
                exclude = {"id", "parent", "equivalent_to"}
                if check.model_dump(exclude=exclude) != original.model_dump(exclude=exclude):
                    raise ValueError("catalog_equivalence_not_same_question")
        return self


class InputVariant(Value):
    """Same source-bound question, explicit observation scope for another modality."""

    modality: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=240)


class ExaminationV2(Examination):
    input_variants: tuple[InputVariant, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def variants(self):
        modalities = [v.modality for v in self.input_variants]
        if len(set(modalities)) != len(modalities) or self.modality in modalities:
            raise ValueError("catalog_duplicate_input_variant")
        if self.input_variants and (
            self.method != "vision" or self.input_capabilities.scope != "visible_region_pending"
        ):
            raise ValueError("catalog_variants_require_qualitative_visual_method")
        return self


class RequirementCatalogV2(RequirementCatalog):
    schema_version: Literal["research-requirement-catalog/1.1.0"]
    checks: tuple[ExaminationV2, ...] = Field(min_length=1, max_length=128)


def examination(value):
    return validate(ExaminationV2 if "input_variants" in value else Examination, value)


def select_input_variant(check, component):
    """Select a frozen operationalization, never infer modality equivalence."""
    if component is None or component.modality == check["modality"]:
        return check, None
    verified = examination(check)
    selected = next(
        (v for v in getattr(verified, "input_variants", ()) if v.modality == component.modality),
        None,
    )
    if selected is None:
        raise ValueError("catalog_input_variant_not_registered")
    return dict(check, modality=selected.modality, instruction=selected.instruction), dict(
        schema_version="requirement-input-variant/1.0.0",
        question_id=check["id"],
        question_checksum=digest(check),
        primary_modality=check["modality"],
        selected_modality=selected.modality,
        instruction=selected.instruction,
        authority="registered_research_scope_requires_original_source_and_method_recheck",
    )


class MeasurementFulfillment(Value):
    schema_version: Literal["same-component-measurement-fulfillment/1.0.0"]
    # Explicit clinical question -> source concepts, never dataset or sample IDs.
    questions: dict[str, tuple[str, ...]] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def nonempty(self):
        if any(not k or not v or len(set(v)) != len(v) for k, v in self.questions.items()):
            raise ValueError("measurement_fulfillment_inventory")
        return self


def validate_fulfillment(profile):
    value = profile.get("measurement_fulfillment")
    if value is None:
        return None
    if "requirement_catalog" not in profile:
        raise ValueError("measurement_fulfillment_requires_catalog")
    policy = validate(MeasurementFulfillment, value)
    checks = {c["id"]: c for c in profile["checks"]}
    for key in policy.questions:
        if key not in checks or checks[key]["method"] != "source_measurement":
            raise ValueError("measurement_fulfillment_question_method")
    return policy


def fulfill_measurements(rows, dispositions, manifest, profile):
    """One source component gets one joint review, retaining all declared questions.

    This consolidates only an explicitly mapped catalog question and the original
    owner's reading of that exact component. Different concepts, eyes, visits,
    source qualifications or modalities are never inferred to be equivalent.
    A provided value does not prove reliability, repeatability or clinical stage.
    """
    policy = validate_fulfillment(profile)
    if policy is None:
        return rows, dispositions, None
    components = {c.id: c for c in manifest.components}
    numeric = {
        r["component"]: r
        for r in rows
        if r["id"] == "check:" + str(r["component"])
        and r.get("acquisition_method") == "source_measurement"
    }
    trace = {r["id"]: r for r in dispositions}
    removed, outcomes = set(), []
    updated = {r["id"]: dict(r) for r in rows}
    for row in rows:
        question = next(
            (
                q
                for q in policy.questions
                if row["id"]
                in {
                    "active:" + q,
                    "active:" + q + ":" + str(row.get("component")),
                }
            ),
            None,
        )
        if question is None:
            continue
        component = components.get(row.get("component"))
        canonical_row = numeric.get(row.get("component"))
        if (
            component is None
            or canonical_row is None
            or component.concept not in policy.questions[question]
        ):
            outcomes.append(
                dict(
                    requirement=row["id"],
                    state="retained",
                    reason="no_matching_original_measurement",
                )
            )
            continue
        if (
            component.status not in {"obtained", "partial"}
            or row.get("capability_selection", {}).get("state") == "missing_input"
            or row.get("acquisition_method") != "source_measurement"
            or row["input_modalities"] != canonical_row["input_modalities"]
            or set(row["definition"].source_refs) != set(canonical_row["definition"].source_refs)
            or row["definition"].interpretation_status != "installed"
            or canonical_row["definition"].interpretation_status != "installed"
        ):
            outcomes.append(
                dict(
                    requirement=row["id"],
                    state="retained",
                    reason="source_method_or_interpretation_not_same",
                )
            )
            continue
        selected = updated[canonical_row["id"]]
        before = selected["definition"]
        raw = before.model_dump(mode="json")
        raw["interpretation"] += (
            " 同一原始测量同时回应登记检查「"
            + row["definition"].examination
            + "」："
            + row["definition"].interpretation
            + " 这里只合并同一输入的取证与评阅，不表示可靠性、重复检查或分期依据已齐全；"
            "缺项与适用限制仍须逐项说明，不额外产生一票。"
        )
        if len(raw["interpretation"]) > 240:
            outcomes.append(
                dict(
                    requirement=row["id"],
                    state="retained",
                    reason="joint_definition_exceeds_original_wire_bound",
                )
            )
            continue
        raw["reference"]["checksum"] = digest({k: v for k, v in raw.items() if k != "reference"})
        selected["definition"] = validate(type(before), raw)
        binding = dict(
            schema_version=FULFILLMENT,
            requirement=row["id"],
            component=component.id,
            component_checksum=digest(component),
            original_question=row["definition"].model_dump(mode="json"),
            original_measurement_definition=canonical_row["definition"].model_dump(mode="json"),
            original_parent=row["parent"],
            original_measurement_parent=canonical_row["parent"],
            fulfillment="shared_observation_and_joint_review_planned",
            medical_completeness_claimed=False,
        )
        selected.setdefault("fulfilled_requirements", []).append(binding)
        selected["parent"] = row["parent"]
        trace[selected["id"]]["parent"] = row["parent"]
        trace[selected["id"]].setdefault("fulfilled_requirements", []).append(binding)
        trace[row["id"]].update(
            state="fulfilled_by_source_measurement",
            fulfilled_by=selected["id"],
            reason="same_component_one_observation_and_joint_review",
            original_question=row["definition"].model_dump(mode="json"),
            medical_completeness_claimed=False,
        )
        removed.add(row["id"])
        outcomes.append(dict(requirement=row["id"], state="shared", fulfilled_by=selected["id"]))
    result = [updated[r["id"]] for r in rows if r["id"] not in removed]
    return (
        result,
        [trace[r["id"]] for r in dispositions],
        dict(
            policy=profile["measurement_fulfillment"],
            outcomes=outcomes,
            original_requirement_count=len(rows),
            executable_requirement_count=len(result),
            removed_duplicate_review_count=len(removed),
            coverage_basis="joint_review_requirements_after_explicit_source_component_fulfillment",
            clinical_completeness_claimed=False,
        ),
    )


def read(reference):
    if set(reference) != {"path", "checksum"}:
        raise ValueError("catalog_reference")
    value = json.loads(Path(reference["path"]).read_bytes())
    if digest(value) != reference["checksum"]:
        raise ValueError("catalog_changed")
    models = {
        "research-requirement-catalog/1.0.0": RequirementCatalog,
        "research-requirement-catalog/1.1.0": RequirementCatalogV2,
    }
    if value.get("schema_version") not in models:
        raise ValueError("catalog_schema_version")
    return validate(models[value["schema_version"]], value)


def projection(catalog, target=None):
    goals = sorted({g for c in catalog.checks for g in c.goals})
    if target is not None and target not in goals:
        raise ValueError("catalog_runtime_target_missing")
    return dict(
        checks=[c.model_dump(mode="json") for c in catalog.checks],
        nodes=[n.model_dump(mode="json") for n in catalog.nodes],
        goals=[target] if target is not None else goals,
        capability_matching=POLICY,
    )


def expand_profile(profile):
    """Resolve one frozen catalog; a run policy cannot override its definitions."""
    if profile.get("schema_version") != "source-bound-acquisition-profile/5.0.0" or any(
        key in profile
        for key in ("checks", "nodes", "goals", "capability_matching", "input_semantics")
    ):
        raise ValueError("catalog_profile_must_not_override_definitions")
    return dict(
        profile,
        **projection(
            read(profile["requirement_catalog"]),
            profile.get("definition_expansion", {}).get("public_target"),
        ),
    )


def verify_projection(profile):
    if profile.get("capability_matching") != POLICY or "input_semantics" in profile:
        raise ValueError("catalog_matching_policy")
    current = projection(
        read(profile["requirement_catalog"]),
        profile.get("definition_expansion", {}).get("public_target"),
    )
    if any(profile.get(k) != v for k, v in current.items()):
        raise ValueError("catalog_projection_changed")


def verify_capabilities(manifest, capabilities):
    manifest = CaseComponents.model_validate_json(manifest.model_dump_json())
    if capabilities is None:
        raise ValueError("catalog_original_input_capabilities_required")
    capabilities = InputCapabilities.model_validate_json(capabilities.model_dump_json())
    if capabilities.input_checksum != manifest.checksum:
        raise ValueError("catalog_capability_input_binding")
    components = {c.id: c for c in manifest.components}
    if set(components) != {c.component_id for c in capabilities.components}:
        raise ValueError("catalog_capability_inventory")
    for c in capabilities.components:
        original = components[c.component_id]
        if c.component_checksum != digest(original) or (
            c.representation,
            c.modality,
            c.concept,
            c.input_status,
            c.unit,
            c.unit_basis,
        ) != (
            original.kind,
            original.modality,
            original.concept,
            original.status,
            original.unit,
            original.unit_basis,
        ):
            raise ValueError("catalog_capability_component_binding")
    return capabilities


def match(check, capabilities):
    """Return real compatible components and explain excluded/conditional inputs.

    No source, model, tool inventory, case name or dataset name enters selection.
    Scope-pending visual checks may inspect their scope but cannot assert that
    anatomy is visible or authorize a measurement before its method checks.
    """
    check = examination(check)
    needs = check.input_capabilities
    modalities = {check.modality, *(v.modality for v in getattr(check, "input_variants", ()))}
    selected, excluded = [], []
    for c in capabilities.components:
        if c.modality not in modalities or c.input_status == "withheld":
            continue
        reason = None
        if c.representation not in needs.representations:
            reason = "representation_mismatch"
        elif needs.concepts and c.concept not in needs.concepts:
            reason = "measurement_concept_mismatch"
        if reason:
            excluded.append(dict(component=c.component_id, reason=reason))
        else:
            selected.append(c)
    return selected, excluded


def disposition(check, component, excluded):
    needs = validate(Prerequisites, check["input_capabilities"])
    if component is None:
        state, reason = "missing_input", "no_compatible_input_representation"
    elif (
        component.input_status not in {"obtained", "partial"} or component.acquisition_fraction == 0
    ):
        state, reason = "missing_input", "input_unavailable"
    elif needs.scope == "paired_eyes":
        state, reason = "conditional", "original_owner_pair_identity_required"
    elif needs.scope == "visible_region_pending":
        state, reason = "conditional", "qualitative_scope_and_current_method_required"
    elif component.unit is None or component.unit_basis is None:
        state, reason = "conditional", "source_unit_and_meaning_required"
    elif needs.required_unit and needs.required_unit != component.unit:
        state, reason = "conditional", "qualified_unit_conversion_required"
    else:
        state, reason = "input_available", "original_source_measurement_representation"
    return dict(
        state=state,
        reason=reason,
        excluded_components=excluded,
        component_checksum=component.component_checksum if component else None,
        execution_authorized=False,
        scope_qualification="not_inferred_from_modality_or_dataset",
    )


def prepare_selection(input_path, task_name, output, *, registry):
    """Diagnostic stage over the exact production projection and current owners.

    Reads original Knowledge and input owners; no registration, dispatch, model,
    mutation, secret inspection or execution authorization. The run will rebuild
    the same selection rather than trusting this exported preview.
    """
    from packages.dataset_adapters.multimodal.capabilities import from_preparation
    from packages.dataset_adapters.multimodal.preparation import verify
    from packages.dataset_adapters.multimodal.tasks import load_intent

    from .active_requirements import load, project
    from .material import MedicalSources

    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("catalog_preview_new_output_required")
    manifest = verify(input_path)
    capabilities = from_preparation(input_path)
    task, _ = load_intent(task_name, registry)
    profile = load(DEFAULT_PROFILE)
    sources = MedicalSources()
    rows, plan = project(manifest, sources.initial, task, profile, capabilities=capabilities)
    if sources.read() != sources.initial:
        raise ValueError("catalog_preview_source_changed")
    result = dict(
        schema_version="requirement-selection-preview/1.0.0",
        input_checksum=manifest.checksum,
        task=task.model_dump(mode="json"),
        capabilities=capabilities.model_dump(mode="json"),
        plan=plan,
        requirements=rows,
        execution_authorized=False,
        qualification="current_source_input_representation_only_visual_scope_pending",
        api_requests=0,
        model_runs=0,
        database_writes=0,
    )
    result["checksum"] = digest(result)
    with out.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    return dict(
        output=str(out),
        checksum=result["checksum"],
        requirements=len(rows),
        scheduled=sum(r["state"] == "scheduled" for r in plan["rows"]),
        input_gaps=sum(r["state"] == "input_gap" for r in plan["rows"]),
        definition_backlog=sum(r["state"] == "definition_backlog" for r in plan["rows"]),
        qualification=result["qualification"],
        execution_authorized=False,
        api_requests=0,
        model_runs=0,
        database_writes=0,
    )


def prepare_transition(previous, output):
    """Original-owner historical reconciliation, without claiming a new run.

    Deliberately consumes stopped_tip rather than adding phase usage: the latter
    omits Source/search operations of completed definition-only windows.
    """
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .active_acquisition import successor_projection
    from .active_requirements import load
    from .campaign_renewal import completed_parent, stopped_tip
    from .implementation import freeze
    from .reuse import CAPABILITY_PROFILE

    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("catalog_preview_new_output_required")
    path = Path(previous).absolute()
    if out.is_relative_to(path):
        raise ValueError("capability_transition_output_must_not_change_history")
    # Exit the read scopes (and their current-source checks) before writing output.
    with historical_fact_read_scope(), _projection_visible_reads():
        parent, result = completed_parent(path)
        definition_only = parent.protocol.get("mode") == "definitions"
        tip = stopped_tip(parent, path, completed_source=definition_only)
        if tip["result"] != digest(result):
            raise ValueError("capability_transition_terminal_result_changed")
        transition = successor_projection(parent.base, load(CAPABILITY_PROFILE))
        facts = parent.accepted_observations()
        bound = parent.base.observations
        if any(digest(facts[k]) != digest(bound[k]) for k in facts.keys() & bound.keys()):
            raise ValueError("capability_transition_observation_owner_conflict")
        attempts = tip["previous_attempts"]
        if any(type(r["requests"]) is not int or r["requests"] < 0 for r in attempts):
            raise ValueError("capability_transition_invalid_accounting")
        used = sum(r["requests"] for r in attempts)
        limit = parent.cumulative_request_limit()
        if used > limit:
            raise ValueError("capability_transition_cumulative_budget_exceeded")
        value = dict(
            schema_version="capability-transition-preview/1.1.0",
            previous=str(path),
            parent_protocol_checksum=digest(parent.protocol),
            terminal_owner_proof=tip,
            transition=transition,
            confirmed_observations={k: digest(v) for k, v in sorted(facts.items())},
            bound_input_tool_observations={k: digest(v) for k, v in sorted(bound.items())},
            observation_counts=dict(
                confirmed_phases=len(facts),
                bound_input_tools=len(bound),
                unique_requirements=len(facts.keys() | bound.keys()),
                accepted_direction_reviews="not_counted_by_this_projection",
            ),
            budget=dict(used=used, cumulative_limit=limit, remaining=limit - used),
            implementation=freeze(),
            execution_authorized=False,
            api_requests=0,
            model_runs=0,
            database_writes=0,
            new_scores=0,
        )
    value["checksum"] = digest(value)
    with out.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
    return dict(
        output=str(out),
        checksum=value["checksum"],
        coverage=transition["coverage"],
        confirmed_observations=len(facts),
        observation_counts=value["observation_counts"],
        budget=value["budget"],
        execution_authorized=False,
        api_requests=0,
        database_writes=0,
        new_scores=0,
    )

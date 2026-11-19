"""Read-only capability selection over an original completed Generic campaign.

The original case, input and goal stay canonical. A registered public task is a
selection vocabulary binding, never a replacement for the historical goal or
an authorization to copy judgments to differently defined questions.
"""

import json
from pathlib import Path

from packages.grounding.models import canonical, digest

VERSION = "generic-capability-selection/1.0.0"
TASK_REGISTRY = "configs/research/multimodal_tasks_v2.json"
TEMPLATE = "configs/research/multimodal_catalog_acquisition_v8.json"


def selection_profile(intent):
    from packages.multimodal_research.automatic_profile import derive
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from packages.multimodal_research.requirement_catalog import expand_profile, verify_projection

    raw, derivation = derive(TEMPLATE, intent)
    policy = read_expansion_policy(raw["definition_expansion"])
    if policy.public_task_checksum != intent.checksum or policy.public_target != intent.target_code:
        raise ValueError("generic_capability_derived_task_changed")
    profile = expand_profile(raw)
    verify_projection(profile)
    return raw, derivation, profile


def task_alias(public, intent):
    """One explicitly supported legacy root contract, not fuzzy title matching."""
    if (
        public.get("schema_version") != "generic-original-public-proposition/1.0.0"
        or public.get("original_root_version") != "generic-source-case-root/2.0.0"
        or public.get("original_root_key") != "glaucoma"
        or public.get("title") != "glaucoma"
        or public.get("scope") != "declared_research_question_not_diagnosis_or_dataset_label"
        or (intent.code, intent.target_code, intent.output_kind)
        != ("TASK:GLAUCOMA_DETECTION", "GOAL:GLAUCOMA_DETECTION", "binary_evidence")
        or not public.get("target", "").startswith("GOAL:OID_")
        or not public.get("original_catalog_checksum")
        or not public.get("root_builder_checksum")
    ):
        raise ValueError("generic_capability_original_question_not_supported")
    return dict(
        schema_version="generic-original-task-vocabulary/1.0.0",
        canonical_target=public["target"],
        selection_target=intent.target_code,
        original_proposition=public,
        registered_task=intent.model_dump(mode="json"),
        task_checksum=intent.checksum,
        scope="selection_only_original_patient_goal_unchanged",
        old_scores_relabelled=False,
    )


def observation_inventory(state):
    """Preserve all requirements; absence is not fabricated as a neutral value."""
    items = {i.id: i for i in state.policy.items}
    observations = {o.reference.code: o for o in state.material.observations}
    rows = []
    for requirement in state.policy.requirements:
        item = items.get(requirement.id)
        observed = observations.get(item.observation_code) if item else None
        rows.append(
            dict(
                id=requirement.id,
                requirement=requirement.model_dump(mode="json"),
                item_checksum=digest(item) if item else None,
                observation=observed.reference.model_dump(mode="json") if observed else None,
                status=observed.status if observed else "not_obtained",
                method=observed.method if observed else None,
                unit=observed.unit if observed else None,
                score_reuse_authorized=False,
            )
        )
    return rows


def confirmed_methods(state, base):
    """Current original tool facts remain selectable beside raw input capabilities.

    These bindings preserve each original question and method. They do not grant
    equivalence to a differently defined catalog question or authorize new work.
    """
    from .comprehensive_materials import measured_projection
    from .generic_definition_facts import measurement_basis

    base.check_tools()
    available = measured_projection(base.tools)
    observations = {o.reference.code: o for o in state.material.observations}
    selected = []
    for item in state.policy.items:
        observed = observations.get(item.observation_code)
        if observed is None or observed.status != "obtained":
            continue
        projection = available.get(item.id)
        if item.method == "registered_segman_geometry" or (
            projection is not None
            and (item.method, item.unit) == (projection["method"], projection["unit"])
        ):
            selected.append(item.id)
    facts = {target: measurement_basis(state, base, target) for target in selected}
    base.check_tools()
    return dict(
        schema_version="generic-current-method-candidates/1.0.0",
        input_checksum=state.material.scope.input_checksum,
        target=state.material.goal_code,
        facts=facts,
        scope="original_confirmed_observations_not_new_question_equivalence",
        score_reuse_authorized=False,
        new_execution_authorized=False,
    )


def inspect(campaign, *, own_claim=None):
    """Use original task/input/source owners; no registration, claim or requests."""
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.dataset_adapters.multimodal.analysis import build
    from packages.dataset_adapters.multimodal.capabilities import project as capabilities
    from packages.dataset_adapters.multimodal.fairvlmed import prepare
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope
    from packages.multimodal_research.active_requirements import project
    from packages.multimodal_research.material import MedicalSources

    from .generic_campaign import GenericCampaign
    from .generic_definition_input import public_proposition
    from .generic_definition_successor import completed_history
    from .generic_pending_review import original_phase

    if type(campaign) is not GenericCampaign:
        raise ValueError("generic_capability_original_campaign_required")
    with historical_fact_read_scope(), _projection_visible_reads():
        history = completed_history(campaign, own_claim=own_claim)
        docs, state, phase, final = original_phase(campaign)
        public = public_proposition(campaign.base, state.material.goal_code)
        intent, registry = load_intent("glaucoma_detection", Path(TASK_REGISTRY))
        alias = task_alias(public, intent)
        # This branch is a parser for the existing Generic input owner. Neither
        # the dataset name nor sample name is used by requirement matching.
        prepared = prepare(campaign.spec["root"], campaign.spec["sample"])
        manifest = prepared.manifest
        images = [p for p in manifest.payloads if p.media_type == "image/png"]
        if len(images) != 1 or images[0].checksum != campaign.base.image_artifact.checksum:
            raise ValueError("generic_capability_original_image_changed")
        age = next(c for c in manifest.components if c.id == "age")
        old_age = campaign.base.input_context["age"]
        if (age.status, age.value, age.unit) != tuple(
            old_age[k] for k in ("status", "value", "unit")
        ):
            raise ValueError("generic_capability_original_age_changed")
        capability = capabilities(manifest, build(manifest, dict(prepared.payload_bytes)))
        raw, derivation, profile = selection_profile(intent)
        medical = MedicalSources()
        selected, selection = project(
            manifest, medical.read(), intent, profile, capabilities=capability
        )
        methods = confirmed_methods(state, campaign.base)
        phase.verify(state)
        if medical.read() != medical.initial:
            raise ValueError("generic_capability_medical_sources_changed")
        original = observation_inventory(state)
        accepted = docs.read("accepted:" + final["acceptance_key"])
        if not accepted:
            raise ValueError("generic_capability_original_acceptance_missing")
        return dict(
            schema_version=VERSION,
            original_case=state.material.scope.case_code,
            original_input_checksum=state.material.scope.input_checksum,
            normalized_input_checksum=manifest.checksum,
            task_alias=alias,
            registry=registry,
            input_capabilities=capability.model_dump(mode="json"),
            derivation=derivation,
            derived_profile=raw,
            original_state_checksum=digest(state),
            original_acceptance=dict(key=final["acceptance_key"], checksum=digest(accepted)),
            original_history=history,
            original_requirements=original,
            confirmed_method_candidates=methods,
            selected_requirements=json.loads(canonical(selected)),
            selection=selection,
            old_requirement_ids_preserved=[r["id"] for r in original],
            correspondence_status="requires_explicit_method_definition_and_observation_matching",
            clinical_acceptance=False,
            score_reuse_authorized=False,
            dispatch_authorized=False,
            api_requests=0,
            database_writes=0,
        )

"""Owner-derived visible tasks. No old reviews or fixed model answers cross this port."""

from packages.grounding.models import digest, validate
from packages.llm_provider.models import ReviewTask, ReviewTemplate

from .acceptance_models import DomainState, TaskPolicyV2


def project_review_tasks(state):
    state = validate(DomainState, state)
    if not isinstance(state.policy, TaskPolicyV2):
        raise ValueError("review_task_definition_upgrade_required")
    observations = {o.reference.code: o for o in state.material.observations}
    sources = {s.reference.code: s.reference for s in state.material.sources}
    resources = {r.code for r in state.resources}
    tasks, fixed = [], []
    for p in sorted(state.policy.items, key=lambda p: p.id):
        o = observations.get(p.observation_code)
        if o is None or (o.method, o.unit, state.laterality) != (p.method, p.unit, p.laterality):
            raise ValueError("review_observation_method_unit_eye")
        if len(set(p.source_codes)) != len(p.source_codes) or any(
            k not in sources for k in p.source_codes
        ):
            raise ValueError("review_source_missing_or_duplicate")
        refs = tuple(sources[k] for k in p.source_codes)
        if set(p.definition.source_refs) != set(refs):
            raise ValueError("review_definition_reference_mismatch")
        if any(not set(path) <= resources for path in p.dependency_paths):
            raise ValueError("review_dependency_missing")
        if p.fixed_model_row is not None:
            if (
                p.fixed_observation_ref != o.reference
                or p.fixed_model_row.source_binding_checksum
                != digest([s for s in state.material.sources if s.reference.code in p.source_codes])
            ):
                raise ValueError("fixed_model_source_or_observation_changed")
            fixed.append(p.id)
            continue
        tasks.append(
            ReviewTask(
                assessment_id=p.id,
                node=p.node,
                goal=state.material.goal_code,
                definition=p.definition,
                observation=o.reference,
                allowed_sources=refs,
                method=p.method,
                unit=p.unit,
                laterality=p.laterality,
                applicable=p.applicable,
            )
        )
    return ReviewTemplate(
        scope_checksum=digest(state.material.scope),
        policy_checksum=digest(state.policy),
        scale=state.policy.scale,
        tasks=tuple(tasks),
        fixed_assessment_ids=tuple(fixed),
    ).check_material(state.material)


def review_payload(material, template, round_id):
    template = validate(ReviewTemplate, template).check_material(material)
    return dict(
        role="review",
        material=material.model_dump(mode="json"),
        assessment_ids=list(template.assessment_ids),
        scale=template.scale.model_dump(mode="json"),
        review_tasks=template.model_dump(mode="json"),
        round=round_id,
    )

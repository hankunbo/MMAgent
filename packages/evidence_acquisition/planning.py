"""Deterministic gap routing over current owner material, never over old ratings."""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate

from .models import AcquisitionPlan, GapCard, Requirement, SearchProgress, SearchSuggestion


def plan_acquisition(state, requirements):
    state = validate(DomainState, state)
    tasks = project_review_tasks(state).tasks
    required = tuple(validate(Requirement, r) for r in requirements)
    by_id = {r.assessment_id: r for r in required}
    if len(by_id) != len(required) or set(by_id) != {t.assessment_id for t in tasks}:
        raise ValueError("acquisition_requirement_coverage")
    observations = {o.reference.code: o for o in state.material.observations}
    has_image = any(o.image is not None for o in observations.values())
    cards = []
    for task in tasks:
        req, obs = by_id[task.assessment_id], observations[task.observation.code]
        available = obs.status == "obtained" and obs.value is not None and bool(obs.value.strip())
        if obs.status == "obtained" and not available:
            raise ValueError("acquisition_empty_observation")
        independent = req.purpose == "disease_relation"
        missing_rule = independent and task.definition.interpretation_status == "missing"
        blockers, actions = [], []
        if missing_rule:
            blockers.append("interpretation_reference")
            actions.append("bind_existing_reference")
            if req.public_topics:
                actions.append("search_knowledge")
        if available and req.method_details_required:
            blockers.append("method_provenance")
            actions.append("read_method_provenance")
        if req.unavailable_inputs:
            if available:
                raise ValueError("acquisition_available_input_conflict")
            blockers.extend(req.unavailable_inputs)
            # No pointless automatic searches to invent a missing patient examination.
            disposition, actions = "current_input_unavailable", []
        elif obs.status == "ungradable":
            blockers.append("observation_quality")
            disposition, actions = "quality_limited", []
        elif not available:
            blockers.append("observation")
            if req.acquisition == "image" and has_image:
                actions.extend(("reuse_observation", "targeted_vision"))
            elif req.acquisition == "tool_gap":
                actions.append("request_tool_approval")
            else:
                actions.append("reuse_observation")
            disposition = "needs_acquisition"
        elif blockers:
            disposition = "needs_acquisition"
        elif not independent:
            disposition = "auxiliary_observed"
        else:
            disposition = "ready_for_review"
            actions.append("review")
        cards.append(
            GapCard(
                assessment_id=task.assessment_id,
                node=task.node,
                purpose=req.purpose,
                observation=obs.reference,
                observation_status=obs.status,
                interpretation_status=(
                    task.definition.interpretation_status
                    if independent
                    else "not_independent_disease_rule"
                ),
                blockers=tuple(blockers),
                missing_inputs=req.unavailable_inputs,
                missing_source_ids=req.missing_source_ids,
                next_actions=tuple(dict.fromkeys(actions)),
                public_topics=req.public_topics,
                priority=req.priority,
                disposition=disposition,
            )
        )
    return AcquisitionPlan(
        state_checksum=digest(state),
        requirements_checksum=digest(required),
        cards=tuple(sorted(cards, key=lambda c: (-c.priority, c.assessment_id))),
    )


def search_plan(card, suggestion, progress, provider_codes):
    """Translate an LLM suggestion into the existing 04 bounded public query contract."""
    from packages.knowledge_search.broad import broad_plan

    card = validate(GapCard, card)
    suggestion = validate(SearchSuggestion, suggestion)
    progress = validate(SearchProgress, progress)
    if (
        "search_knowledge" not in card.next_actions
        or suggestion.assessment_id != card.assessment_id
        or not set(suggestion.topics) <= set(card.public_topics)
        or len(set(suggestion.topics)) != len(suggestion.topics)
    ):
        raise ValueError("acquisition_search_scope")
    if progress.pending_unknown or progress.cancelled or progress.deadline_reached:
        raise ValueError("acquisition_search_stopped_or_unknown")
    if progress.rounds >= 2 or progress.no_progress >= 2:
        raise ValueError("acquisition_search_exhausted")
    if (
        not provider_codes
        or len(provider_codes) > 2
        or len(set(provider_codes)) != len(provider_codes)
    ):
        raise ValueError("acquisition_search_provider_limit")
    # Existing privacy catalog and SQL query validator remain unchanged.
    return broad_plan(tuple(provider_codes), topics=suggestion.topics, limit=5, max_pages=1)


def affected_assessments(state, *, observation_codes=(), source_codes=(), resource_codes=()):
    """All affected task IDs, not permission to reuse ratings across changed parents."""
    state = validate(DomainState, state)
    known_obs = {o.reference.code for o in state.material.observations}
    known_sources = {s.reference.code for s in state.material.sources}
    known_resources = {r.code for r in state.resources}
    if (
        not set(observation_codes) <= known_obs
        or not set(source_codes) <= known_sources
        or not set(resource_codes) <= known_resources
    ):
        raise ValueError("acquisition_unknown_dependency")
    return tuple(
        p.id
        for p in state.policy.items
        if p.observation_code in observation_codes
        or set(p.source_codes) & set(source_codes)
        or any(set(path) & set(resource_codes) for path in p.dependency_paths)
    )

"""Deterministic, read-only extraction from current original owners. No scoring formulas."""

from packages.evidence_scoring.accepted import rows_from_candidate
from packages.grounding.models import digest
from packages.memory.distillation_models import (
    AbstractData,
    Advice,
    Change,
    Context,
    DistillationRecord,
    PairData,
    RecoveryData,
)
from packages.memory.models import Dependency
from packages.memory.writer_models import EpisodeData, WriterRecord

DISPLAY = {
    "necessary": "necessary_within_snapshot_policy",
    "context_redundant": "context_redundant_under_frozen_metrics",
    "components_changed": "components_or_selected_score_changed",
    "conclusion_changed": "research_direction_changed",
    "indeterminate": "indeterminate",
}
LIMITS = (
    "frozen_system_snapshot_policy_only",
    "not_medical_causality",
    "no_single_difference_assumption",
    "hypothetical_not_executed_tool_result",
)


def dependency(record):
    return Dependency(key=record.key, checksum=digest(record))


def union_sources(records):
    values = {}
    for record in records:
        for s in record.sources:
            key = (s.owner, s.run_id, s.key, s.kind)
            if key in values and values[key] != s:
                raise ValueError("memory_source_binding")
            values[key] = s
    return tuple(values[k] for k in sorted(values))


def roles(sources):
    return tuple(dict(source=digest(s), owner=s.owner, nature=s.nature) for s in sources)


def context(episode, owner):
    d = episode.detail
    snapshot = owner.snapshot()
    original_input = (
        snapshot["control"]["binding"]["input_checksum"] if snapshot["control"] else None
    )
    return Context(
        episode=dependency(episode),
        case_code=episode.case_code,
        run_id=episode.run_id,
        input_checksum=original_input,
        input_missing_reason=None if original_input else "no_original_planner_input_binding",
        episode_input_checksum=d.input_checksum,
        runtime_context_checksum=snapshot["runtime"]["context_checksum"],
        terminal=d.terminal,
        graph=d.final_graph,
        scores=d.scores,
        versions=tuple(r.model_dump(mode="json") for r in d.versions),
        runtime_versions=d.runtime_versions,
    )


def owner_for(registry, run_id):
    owner = getattr(registry, "owners", {}).get(run_id)
    if owner is None or not hasattr(owner, "snapshot"):
        raise ValueError("memory_source_port_unavailable")
    return owner


def checked_episode(episode, registry):
    if not isinstance(episode, WriterRecord) or not isinstance(episode.detail, EpisodeData):
        raise ValueError("memory_distillation_episode_required")
    owner = owner_for(registry, episode.run_id)
    # Original owner reconstructs termination, accepted batch and every required source.
    owner.verify_record(episode)
    for source in episode.sources:
        registry.verify(source)
    return owner


def judgment(fact, target):
    policy = next(i for i in fact.state.policy.items if i.id == target)
    proposal = next((i for i in fact.candidate.items if i.assessment_id == target), None)
    value = proposal.judgment.value if proposal else policy.fixed_model_row.judgment.value
    observation = next(
        o for o in fact.state.material.observations if o.reference.code == policy.observation_code
    )
    # The original free-text measurement remains at its owner; reference it, don't copy notes.
    return value, observation.reference.model_dump(mode="json")


def changes(report, fact, alternatives):
    # Match the original 09 engine's per-target availability, even when another
    # intervention makes the whole comparison indeterminate. The scoring owner
    # validates acquisition/citations; caller availability flags are not authority.
    replacement_rows = {
        name: {row.id: row for row in rows_from_candidate(other.state, other.candidate)}
        for name, other in alternatives.items()
    }
    result = []
    for intervention in report["candidate"]["interventions"]:
        kind = intervention["kind"]
        for target in intervention["targets"]:
            resource = kind == "remove_resource"
            original, obs = (
                (
                    next(r for r in fact.state.resources if r.code == target).model_dump(
                        mode="json"
                    ),
                    None,
                )
                if resource
                else judgment(fact, target)
            )
            removed = kind in {"remove_resource", "remove_evidence"}
            after, after_obs, missing = None, None, None
            if kind == "reverse":
                after, after_obs = -original if original is not None else None, obs
            elif not removed:
                alternative = alternatives.get(intervention["alternative"])
                if alternative is None:
                    missing = "missing_alternative_result"
                else:
                    after, after_obs = judgment(alternative, target)
                    row = replacement_rows[intervention["alternative"]][target]
                    if not row.observation_available or row.judgment.value is None:
                        missing = "missing_observation_or_interpretation"
            result.append(
                Change(
                    intervention=kind,
                    target=target,
                    operation="remove"
                    if removed
                    else "reverse"
                    if kind == "reverse"
                    else "replace",
                    removed=removed,
                    value_kind="resource_reference" if resource else "interpretation_effect",
                    original_value=original,
                    simulated_value=after,
                    original_observation=obs,
                    simulated_observation=after_obs,
                    missing_reason=missing,
                )
            )
    return tuple(result)


def assessment_context(fact):
    return [
        dict(
            id=p.id,
            node=p.node,
            role=p.role,
            kind=p.kind,
            method=p.method,
            unit=p.unit,
            laterality=p.laterality,
            applicable=p.applicable,
            definition_checksum=digest(p.definition),
            correlation_group=p.correlation_group,
            dependence_block=p.dependence_block,
            dependency_paths=[list(path) for path in p.dependency_paths],
            source_codes=list(p.source_codes),
            observation_code=p.observation_code,
        )
        for p in fact.state.policy.items
    ]


def pair(episode, source, owner, registry, request_key, ctx):
    research = owner.research
    simulation = research.simulation
    report = simulation.read_evidence(candidate_id=source.key)
    if report is None or digest(report) != source.checksum:
        raise ValueError("memory_source_binding")
    fact = research.fact_reader()
    if (
        report["baseline"]["reference"] != fact.references()
        or fact.binding.input_checksum != ctx.input_checksum
    ):
        raise ValueError("memory_source_binding")
    alternatives, extra = {}, []
    for name, comparison in report["alternatives"].items():
        other = simulation.alternative_readers[name]()
        installed = owner_for(registry, other.binding.run_id)
        if (
            installed.research is None
            or installed.research.fact_reader().references() != comparison["after"]
        ):
            raise ValueError("memory_source_binding")
        if (
            comparison["before"] != fact.references()
            or comparison["after"] != other.references()
            or other.binding.case_code != episode.case_code
            or other.binding.input_checksum != ctx.input_checksum
        ):
            raise ValueError("memory_scope")
        alternatives[name] = other
        graph = installed.planner.replay(other.binding.run_id).graph.model_dump(mode="json")
        extra.extend(
            installed.research.accepted_sources(
                case_code=other.binding.case_code, run_id=other.binding.run_id, graph=graph
            )
        )
    sources = union_sources((episode,))
    sources = tuple(
        sorted(
            {digest(s): s for s in (*sources, *extra)}.values(),
            key=lambda s: (s.owner, s.run_id, s.key, s.kind),
        )
    )
    docs = research.documents
    validation = docs.read("simulation:validation:" + source.key)
    review = docs.read("simulation:review:" + source.key)
    if validation is None:
        raise ValueError("memory_source_missing")
    v = validation[0]
    if v["report_checksum"] != report["checksum"] or v["fact_checksum"] != report["fact_checksum"]:
        raise ValueError("memory_source_binding")
    return DistillationRecord(
        key="distill:"
        + digest([request_key, episode.key, source.owner, source.key, source.kind])[7:55],
        kind="counterfactual_pair",
        case_code=episode.case_code,
        run_id=episode.run_id,
        applicability=episode.applicability,
        sources=sources,
        parents=(dependency(episode),),
        nature="simulation",
        detail=PairData(
            request_key=request_key,
            context=ctx,
            report_source=source,
            simulation_id=report["simulation_id"],
            policy=simulation.policy.model_dump(mode="json"),
            assessment_contexts={
                "original": assessment_context(fact),
                **{name: assessment_context(other) for name, other in alternatives.items()},
            },
            semantics_version=source.semantics_version,
            changes=changes(report, fact, alternatives),
            affected=tuple(report["affected"]),
            remaining=tuple(report["remaining"]),
            baseline=report["baseline"],
            simulated=report["simulated"],
            comparisons=report["comparisons"],
            alternatives=report["alternatives"],
            classification=report["classification"],
            display=DISPLAY[report["classification"]],
            reasons=tuple(report["reasons"]),
            review=dict(
                required=v["required_review"],
                validation_checksum=digest(validation),
                saved_review_checksum=digest(review) if review else None,
                original_role="agent_opinion_not_fact",
                status="saved_opinion" if review else "not_recorded",
            ),
            source_roles=roles(sources),
            limitations=LIMITS,
            original_counters=report["counters"],
        ),
    )


def recovery(episode, source, owner, request_key, ctx):
    # This original adapter validates incident/policy/HTML/failure reader/current Control.
    research = owner.research
    fact = research.fact_reader()
    report, _, _, nature = research._resolve(fact, source.owner, source.key, source.kind)
    if digest(report) != source.checksum:
        raise ValueError("memory_source_binding")
    return DistillationRecord(
        key="distill:"
        + digest([request_key, episode.key, source.owner, source.key, source.kind])[7:55],
        kind="recovery_experience",
        case_code=episode.case_code,
        run_id=episode.run_id,
        applicability=episode.applicability,
        sources=union_sources((episode,)),
        parents=(dependency(episode),),
        nature=nature,
        detail=RecoveryData(
            request_key=request_key,
            context=ctx,
            report_source=source,
            failure=report.incident.failure.model_dump(mode="json"),
            policy_checksum=report.incident.policy_checksum,
            command=report.incident.command.model_dump(mode="json"),
            before_phase=report.incident.control.phase,
            observed_phase=report.control_observed.phase,
            conclusion_status=report.conclusion_status,
            stop_reason=report.stop_reason,
            unresolved=tuple(g.model_dump(mode="json") for g in report.unresolved),
            attempted_alternatives=report.attempted_alternatives,
            retained_evidence=tuple(r.model_dump(mode="json") for r in report.retained_evidence),
            expert_result_checksum=digest(report.expert_result),
        ),
    )


def build(request_key, episodes, registry):
    """Caller selects Episodes, never required source sets or classification/condition values."""
    from pydantic import TypeAdapter

    from packages.grounding.models import Token

    TypeAdapter(Token).validate_python(request_key)
    episodes = tuple(sorted({e.key: e for e in episodes}.values(), key=lambda e: e.key))
    if not 1 <= len(episodes) <= 16:
        raise ValueError("memory_distillation_bound")
    experiences, missing, counterexamples, contexts = [], [], [], {}
    for episode in episodes:
        owner = checked_episode(episode, registry)
        ctx = contexts[episode.key] = context(episode, owner)
        for source in episode.sources:
            if source.owner == "simulation":
                experiences.append(pair(episode, source, owner, registry, request_key, ctx))
            elif source.owner == "recovery":
                experiences.append(recovery(episode, source, owner, request_key, ctx))
        for role in ("simulation", "recovery"):
            if not any(s.owner == role for s in episode.sources):
                missing.append(
                    dict(episode=episode.key, field=role, reason="no_saved_source_in_episode")
                )
        terminal = episode.detail.terminal
        # All bounded stops/failures remain visible; no claim they are clinical failures.
        if terminal.runtime_status != "COMPLETED_SUPPORTED":
            counterexamples.append(
                dict(
                    episode=episode.key,
                    role="failure_or_bounded_outcome",
                    runtime_status=terminal.runtime_status,
                    reason=terminal.reason,
                )
            )
    for experience in experiences:
        d = experience.detail
        if isinstance(d, PairData) and d.classification != "context_redundant":
            counterexamples.append(
                dict(
                    experience=experience.key,
                    role="sensitivity_or_incomplete_comparison",
                    classification=d.classification,
                )
            )
        elif isinstance(d, RecoveryData):
            counterexamples.append(
                dict(
                    experience=experience.key,
                    role="original_failure",
                    category=d.failure["category"],
                )
            )
    parents = (*episodes, *experiences)
    if len(parents) > 64:
        raise ValueError("memory_distillation_bound")
    sources = union_sources(parents)
    applicability = tuple(sorted(set.intersection(*(set(e.applicability) for e in parents))))
    if not applicability:
        raise ValueError("memory_applicability")
    codes = {"review_bound_context_before_any_reuse"}
    codes.update(
        DISPLAY[e.detail.classification] for e in experiences if isinstance(e.detail, PairData)
    )
    if missing:
        codes.add("retain_missing_comparison_or_recovery")
    if counterexamples:
        codes.add("retain_failure_and_counterexample_sources")
    anchor = episodes[0]
    abstract = DistillationRecord(
        key="distill:" + digest([request_key, "abstract"])[7:55],
        kind="abstract_pattern",
        case_code=anchor.case_code,
        run_id=anchor.run_id,
        applicability=applicability,
        sources=sources,
        parents=tuple(dependency(e) for e in parents),
        nature="abstract_candidate",
        detail=AbstractData(
            request_key=request_key,
            contexts=tuple(contexts[e.key] for e in episodes),
            supporting_episodes=tuple(e.key for e in episodes),
            counterexamples=tuple(counterexamples),
            experiences=tuple(dependency(e) for e in experiences),
            missing=tuple(missing),
            advice=Advice(codes=tuple(sorted(codes))),
            case_count=len({e.case_code for e in episodes}),
            run_count=len({e.run_id for e in episodes}),
            evidence_count=len(sources),
            source_roles=roles(sources),
            not_applicable=(
                "unseen_context",
                "unknown_required_condition",
                "current_patient_fact",
                "clinical_causality",
                "execution_or_publication_without_separate_approval",
            ),
        ),
    )
    return (*experiences, abstract)


def rebuild(record, entities, registry):
    detail = record.detail
    contexts = detail.contexts if isinstance(detail, AbstractData) else (detail.context,)
    try:
        episodes = tuple(entities[c.episode.key] for c in contexts)
    except KeyError:
        raise ValueError("memory_distillation_dependencies") from None
    expected = build(detail.request_key, episodes, registry)
    if not any(record == r for r in expected):
        raise ValueError("memory_distillation_binding")
    return {r.key: r for r in expected}

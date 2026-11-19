"""Explicit v6.1 input bridge, reusing v5's numerical and coverage aggregation.

No REVIEW_BUNDLE, first-case ratings or model outputs are used as defaults.
"""

from packages.evidence_scoring.continuous import summarize
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.current import completeness
from packages.grounding.models import digest, validate


def check_fixed_model(policy, obs):
    """Validate a fixed result or an explicitly unobserved, entirely unscored slot.

    A missing model is not an observed negative result. Keep it in the coverage
    denominator without requiring an invented model output at aggregate time.
    """
    fixed = validate(ContinuousReview, policy.fixed_model_row)
    available = obs.status == "obtained" and obs.value is not None and bool(obs.value.strip())
    missing = (
        obs.status in {"unknown", "missing"}
        and obs.value is None
        and not fixed.observation_available
        and not fixed.criterion_checked
        and fixed.judgment.value is None
        and fixed.weight is None
        and all(r.value is None for r in (fixed.observation, fixed.relation, fixed.applicability))
    )
    if (
        (
            fixed.id,
            fixed.node,
            fixed.kind,
            fixed.role,
            fixed.correlation_group,
            fixed.dependence_block,
            fixed.dependency_paths,
            fixed.observation_available,
        )
        != (
            policy.id,
            policy.node,
            policy.kind,
            policy.role,
            policy.correlation_group,
            policy.dependence_block,
            [list(p) for p in policy.dependency_paths],
            available,
        )
        or not policy.applicable
        or not (available or missing)
    ):
        raise ValueError("fixed_model_policy_binding")
    return fixed


def rows_from_candidate(state, candidate, *, batch_ids=None):
    from packages.evidence_details.acceptance_models import TaskPolicyV5, TaskPolicyV6
    from packages.evidence_scoring.continuous_models import TaskContinuousReview
    from packages.evidence_scoring.review_protocol import ReasonedReviewItem

    reasoned = bool(getattr(state.policy, "direction_policy", None))
    if any(isinstance(i, ReasonedReviewItem) != reasoned for i in candidate.items):
        raise ValueError("review_direction_protocol_binding")

    observations = {o.reference.code: o for o in state.material.observations}
    sources = {s.reference.code for s in state.material.sources}
    resources = {r.code for r in state.resources}
    proposed = {i.assessment_id: i for i in candidate.items}
    from packages.evidence_details.review_tasks import project_review_tasks

    project_review_tasks(state)  # Requires the installed v2 definition and exact dependencies.
    expected = {i.id for i in state.policy.items}
    nonmodel = {i.id for i in state.policy.items if i.fixed_model_row is None}
    if batch_ids is not None:
        if (
            not batch_ids
            or len(set(batch_ids)) != len(batch_ids)
            or not set(batch_ids) <= nonmodel
            or set(proposed) != set(batch_ids)
        ):
            raise ValueError("review_batch_item_scope")
    elif set(proposed) not in (expected, nonmodel):
        raise ValueError("review_item_scope")
    rows = []
    for policy in state.policy.items:
        if batch_ids is not None and policy.id not in batch_ids:
            continue
        item = proposed.get(policy.id)
        obs = observations.get(policy.observation_code)
        if obs is None or (obs.method, obs.unit, state.laterality) != (
            policy.method,
            policy.unit,
            policy.laterality,
        ):
            raise ValueError("review_observation_method_unit_eye")
        if not set(policy.source_codes) <= sources:
            raise ValueError("review_source_missing")
        if any(not set(p) <= resources for p in policy.dependency_paths):
            raise ValueError("review_dependency_missing")
        available = obs.status == "obtained" and obs.value is not None and bool(obs.value.strip())
        if obs.status == "obtained" and not available:
            raise ValueError("review_acquisition_value_missing")
        if policy.fixed_model_row is not None:
            fixed = check_fixed_model(policy, obs)
            if item is None:
                rows.append(fixed)
                continue
        if len(item.observation_citations) != len(set(item.observation_citations)) or len(
            item.reference_citations
        ) != len(set(item.reference_citations)):
            raise ValueError("review_duplicate_citation")
        if not set(item.observation_citations) <= {obs.reference.code} or not set(
            item.reference_citations
        ) <= set(policy.source_codes):
            raise ValueError("review_citation_scope")
        if item.observation_available != available:
            raise ValueError("review_acquisition_mismatch")
        if item.status == "assessed" and (
            policy.definition.interpretation_status != "installed"
            or not policy.applicable
            or obs.status != "obtained"
            or not set(policy.source_codes) <= set(item.reference_citations)
        ):
            raise ValueError("review_applicability")
        from packages.evidence_details.acceptance_models import ObservedItemPolicy

        if isinstance(policy, ObservedItemPolicy) and policy.role == "model":
            if item.status == "assessed" and (
                item.judgment.value != policy.model_direction
                or item.judgment.method != "fixed_model_margin"
            ):
                raise ValueError("dynamic_model_direction_immutable")
        if obs.status in {"missing", "unknown"} and item.status != "input_missing":
            raise ValueError("review_missing_status")
        if (obs.status == "ungradable") != (item.status == "observation_ungradable"):
            raise ValueError("review_ungradable_status")
        if policy.fixed_model_row:
            fixed = validate(ContinuousReview, policy.fixed_model_row)
            if (item.observation, item.relation, item.applicability, item.judgment) != (
                fixed.observation,
                fixed.relation,
                fixed.applicability,
                fixed.judgment,
            ):
                raise ValueError("fixed_model_mapping_immutable")
            rows.append(fixed)
            continue
        vals = [item.observation.value, item.relation.value, item.applicability.value]
        from packages.evidence_details.acceptance_models import QualityItemPolicy

        observation_rating = item.observation
        if isinstance(policy, QualityItemPolicy) and vals[0] is not None:
            vals[0] *= policy.quality.value
            observation_rating = item.observation.model_copy(
                update=dict(
                    value=vals[0],
                    reason=f"O_raw={item.observation.value}; Q={policy.quality.value}; "
                    + item.observation.reason,
                )
            )
        weight = vals[0] * vals[1] * vals[2] if all(v is not None for v in vals) else None
        e = item.judgment.value
        task_relative = isinstance(state.policy, TaskPolicyV5)
        row = (TaskContinuousReview if task_relative else ContinuousReview)(
            id=policy.id,
            node=policy.node,
            title=policy.id,
            kind=policy.kind,
            assessment_checksum=digest(dict(observation=obs, review=item)),
            source_binding_checksum=digest(
                [s for s in state.material.sources if s.reference.code in policy.source_codes]
            ),
            observation=observation_rating,
            relation=item.relation,
            applicability=item.applicability,
            role=policy.role,
            target=state.policy.hypothesis_bindings[policy.id]
            if isinstance(state.policy, TaskPolicyV6)
            else state.policy.target_code
            if task_relative
            else "glaucoma_research_hypothesis",
            judgment=item.judgment,
            direction="unavailable"
            if e is None
            else "support"
            if e > 0
            else "refute"
            if e < 0
            else "neutral",
            observation_available=available,
            criterion_checked=policy.kind == "criterion" and item.status == "assessed",
            reason=item.judgment.reason,
            correlation_group=policy.correlation_group,
            dependence_block=policy.dependence_block,
            dependency_paths=[list(p) for p in policy.dependency_paths],
            weight=weight,
            support=weight * max(e, 0.0) if e is not None else 0.0,
            refutation=weight * max(-e, 0.0) if e is not None else 0.0,
        )
        rows.append(row)
    return rows


def build_accepted(state, candidate, *, reviewed_ids=None):
    from packages.evidence_details.acceptance_models import TaskPolicyV3, TaskPolicyV4, TaskPolicyV5
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.evidence_scoring.continuous_policy import POLICY
    from packages.evidence_scoring.dynamic_policy import POLICY as DYNAMIC_POLICY

    dynamic = isinstance(state.policy, TaskPolicyV3)
    quality_weighted = isinstance(state.policy, TaskPolicyV4)
    from packages.evidence_scoring.quality_policy import POLICY as QUALITY_POLICY
    from packages.evidence_scoring.task_policy import POLICY as TASK_POLICY
    from packages.evidence_scoring.task_policy import SCOPES

    task_relative = isinstance(state.policy, TaskPolicyV5)
    from packages.evidence_details.acceptance_models import TaskPolicyV6
    from packages.evidence_scoring.hypotheses import decision, summarize_hypotheses

    multi = isinstance(state.policy, TaskPolicyV6)

    def summary(rows, **removed):
        return (
            summarize_hypotheses(rows, state.policy, **removed)
            if multi
            else summarize(rows, **removed)
        )

    expected_policy = (
        TASK_POLICY
        if task_relative
        else QUALITY_POLICY
        if quality_weighted
        else DYNAMIC_POLICY
        if dynamic
        else POLICY
    )
    scopes = SCOPES if task_relative else ("slo_image_research", "comprehensive_research")
    if state.policy.aggregation_checksum != digest(expected_policy):
        raise ValueError("aggregation_policy_changed")
    if reviewed_ids is not None and not dynamic:
        raise ValueError("partial_review_requires_dynamic_policy")
    if candidate is None:
        if not dynamic or state.policy.items or reviewed_ids is not None:
            raise ValueError("empty_score_requires_empty_dynamic_evidence")
        rows, statuses = [], {}
    else:
        rows = rows_from_candidate(state, candidate, batch_ids=reviewed_ids)
        statuses = {i.assessment_id: i.status for i in candidate.items}
    statuses.update(
        {
            p.id: "assessed"
            if p.fixed_model_row.judgment.value is not None
            else "reference_missing"
            for p in state.policy.items
            if p.fixed_model_row is not None
        }
    )
    observed = {o.reference.code: o for o in state.material.observations}
    acquired_ids = {
        p.id for p in state.policy.items if observed[p.observation_code].status == "obtained"
    }
    obtained = {
        r.id
        for r in rows
        if r.id in acquired_ids
        and r.observation_available
        and statuses[r.id]
        not in {"method_mismatch", "observation_ungradable", "input_missing", "intermediate"}
    }
    interpreted = {r.id for r in rows if r.judgment.value is not None}
    groups = [
        dict(
            id=g.id,
            importance=dict(g.importance),
            slots=[
                dict(
                    assessment_ids=list(s),
                    acquired=bool(set(s) & obtained),
                    interpreted=bool(set(s) & interpreted),
                )
                for s in g.slots
            ],
        )
        for g in state.policy.groups
    ]
    raw = dict(
        schema_version="case-research-scoring/10.0.0"
        if multi
        else "case-research-scoring/9.0.0"
        if task_relative
        else "case-research-scoring/8.0.0"
        if quality_weighted
        else "case-research-scoring/7.0.0"
        if dynamic
        else "case-research-scoring/6.1.0",
        acquisition_policy="actual_obtained_value/1.0.0",
        task_projection_checksum=digest(project_review_tasks(state)),
        policy_checksum=digest(state.policy),
        scope_checksum=digest(state.material.scope),
        reviews=[r.model_dump(mode="json") for r in rows],
        root=dict(summary=summary(rows), coverage=completeness(groups, scopes=scopes)),
        removals=[
            dict(resource=r.code, summary=summary(rows, removed_resources={r.code}))
            for r in state.resources
        ],
        clinical_acceptance=False,
        disease_probability=None,
    )
    if task_relative:
        raw["target"] = dict(code=state.policy.target_code, summary=state.policy.target_summary)
        raw["task_policy"] = TASK_POLICY
    if multi:
        raw["hypothesis_comparison"] = decision(raw["root"]["summary"])
        raw["evaluation_mapping"] = {h.label: h.evaluation_label for h in state.policy.hypotheses}
        raw["shared_observations_are_not_independent"] = True
        for removal in raw["removals"]:
            removal["hypothesis_comparison"] = decision(removal["summary"])
    if dynamic:
        if quality_weighted:
            raw["quality_weighting"] = dict(
                policy=QUALITY_POLICY,
                factors={p.id: p.quality.model_dump(mode="json") for p in state.policy.items},
                raw_observation_ratings={
                    i.assessment_id: i.observation.model_dump(mode="json") for i in candidate.items
                }
                if candidate
                else {},
            )
            row_map = {r.id: r for r in rows}
            policy_map = {p.id: p for p in state.policy.items}
            raw["requirement_evaluations"] = [
                dict(
                    id=r.id,
                    availability_score=float(r.id in acquired_ids),
                    interpretation_score=float(r.id in interpreted),
                    quality_factor=policy_map[r.id].quality.value if r.id in policy_map else 0.0,
                    gap_score=0.0 if r.id in interpreted else 1.0,
                    support=row_map[r.id].support if r.id in row_map else 0.0,
                    refutation=row_map[r.id].refutation if r.id in row_map else 0.0,
                    missing_reason=r.missing_reason,
                    disease_direction=row_map[r.id].judgment.value if r.id in row_map else None,
                )
                for r in state.policy.requirements
            ]
        # The denominator comes from the frozen requirement ledger, not returned rows.
        # Acquisition is an owner-confirmed observation, independent of API completion.
        for group, original in zip(groups, state.policy.groups, strict=True):
            for slot, ids in zip(group["slots"], original.slots, strict=True):
                slot["acquired"] = bool(set(ids) & acquired_ids)
        raw.update(
            requirement_checksum=digest(state.policy.requirements),
            requirements=[r.model_dump(mode="json") for r in state.policy.requirements],
            groups=groups,
            review_completion=dict(
                reviewed_ids=sorted(r.id for r in rows),
                pending_ids=sorted({p.id for p in state.policy.items} - {r.id for r in rows}),
            ),
        )
        raw["root"]["coverage"] = completeness(groups, scopes=scopes)
        for removal in raw["removals"]:
            removed = removal["resource"]
            from packages.evidence_scoring.engine import path_available

            retained = {
                p.id
                for p in state.policy.items
                if path_available([list(path) for path in p.dependency_paths], {removed})
            }
            removal["coverage"] = completeness(
                [
                    dict(
                        id=g.id,
                        importance=dict(g.importance),
                        slots=[
                            dict(
                                acquired=bool(set(s) & acquired_ids & retained),
                                interpreted=bool(set(s) & interpreted & retained),
                            )
                            for s in g.slots
                        ],
                    )
                    for g in state.policy.groups
                ],
                scopes=scopes,
            )
    if getattr(state.policy, "direction_policy", None):
        raw["schema_version"] = "case-research-scoring/11.0.0"
        raw["direction_policy"] = state.policy.direction_policy
        raw["direction_assessments"] = (
            {
                i.assessment_id: i.direction_assessment.model_dump(mode="json")
                for i in candidate.items
            }
            if candidate
            else {}
        )
        raw["evidence_counts"] = dict(
            requirements=len(state.policy.requirements),
            obtained_observations=len(acquired_ids),
            directional_assessments=len(interpreted),
            nonzero_e=sum(r.judgment.value is not None and r.judgment.value != 0 for r in rows),
            nonzero_weighted_contributions=sum(r.support != 0 or r.refutation != 0 for r in rows),
            correlation_groups=len({r.correlation_group for r in rows}),
            independent_evidence_count=None,
        )
    return {**raw, "checksum": digest(raw)}


def validate_accepted(state, candidate, value):
    expected = build_accepted(state, candidate)
    if value != expected:
        raise ValueError("accepted_score_reconstruction")
    return expected

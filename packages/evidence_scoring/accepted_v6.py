"""Explicit v6 input bridge, reusing v5's numerical and coverage aggregation.

No REVIEW_BUNDLE, first-case ratings or model outputs are used as defaults.
"""

from packages.evidence_scoring.continuous import summarize
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.current import completeness
from packages.grounding.models import digest, validate


def rows_from_candidate(state, candidate):
    observations = {o.reference.code: o for o in state.material.observations}
    sources = {s.reference.code for s in state.material.sources}
    resources = {r.code for r in state.resources}
    proposed = {i.assessment_id: i for i in candidate.items}
    if set(proposed) != {i.id for i in state.policy.items}:
        raise ValueError("review_item_scope")
    rows = []
    for policy in state.policy.items:
        item = proposed[policy.id]
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
        if len(item.observation_citations) != len(set(item.observation_citations)) or len(
            item.reference_citations
        ) != len(set(item.reference_citations)):
            raise ValueError("review_duplicate_citation")
        if not set(item.observation_citations) <= {obs.reference.code} or not set(
            item.reference_citations
        ) <= set(policy.source_codes):
            raise ValueError("review_citation_scope")
        if item.observation_available != (obs.status != "missing"):
            raise ValueError("review_acquisition_mismatch")
        if item.status == "assessed" and (
            not policy.applicable
            or obs.status != "obtained"
            or not set(policy.source_codes) <= set(item.reference_citations)
        ):
            raise ValueError("review_applicability")
        if obs.status == "missing" and item.status != "input_missing":
            raise ValueError("review_missing_status")
        if obs.status == "ungradable" and item.status != "observation_ungradable":
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
        vals = [item.observation.value, item.relation.value, item.applicability.value]
        weight = vals[0] * vals[1] * vals[2] if all(v is not None for v in vals) else None
        e = item.judgment.value
        row = ContinuousReview(
            id=policy.id,
            node=policy.node,
            title=policy.id,
            kind=policy.kind,
            assessment_checksum=digest(dict(observation=obs, review=item)),
            source_binding_checksum=digest(
                [s for s in state.material.sources if s.reference.code in policy.source_codes]
            ),
            observation=item.observation,
            relation=item.relation,
            applicability=item.applicability,
            role=policy.role,
            target="glaucoma_research_hypothesis",
            judgment=item.judgment,
            direction="unavailable"
            if e is None
            else "support"
            if e > 0
            else "refute"
            if e < 0
            else "neutral",
            observation_available=item.observation_available,
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


def build_accepted(state, candidate):
    from packages.evidence_scoring.continuous_policy import POLICY

    if state.policy.aggregation_checksum != digest(POLICY):
        raise ValueError("aggregation_policy_changed")
    rows = rows_from_candidate(state, candidate)
    statuses = {i.assessment_id: i.status for i in candidate.items}
    obtained = {
        r.id
        for r in rows
        if r.observation_available
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
        schema_version="case-research-scoring/6.0.0",
        policy_checksum=digest(state.policy),
        scope_checksum=digest(state.material.scope),
        reviews=[r.model_dump(mode="json") for r in rows],
        root=dict(summary=summarize(rows), coverage=completeness(groups)),
        removals=[
            dict(resource=r.code, summary=summarize(rows, removed_resources={r.code}))
            for r in state.resources
        ],
        clinical_acceptance=False,
        disease_probability=None,
    )
    return {**raw, "checksum": digest(raw)}


def validate_accepted(state, candidate, value):
    expected = build_accepted(state, candidate)
    if value != expected:
        raise ValueError("accepted_score_reconstruction")
    return expected

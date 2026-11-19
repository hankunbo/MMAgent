"""Non-factual 6.1 projection. Accepted fact validation is deliberately unchanged."""

from packages.evidence_scoring.accepted import rows_from_candidate
from packages.evidence_scoring.api import validate as validate_score
from packages.evidence_scoring.continuous import summarize
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.current import completeness
from packages.evidence_scoring.engine import path_available
from packages.grounding.models import validate


def project(state, candidate, scores, *, removed_ids=(), removed_resources=(), overrides=None):
    """Only validated immutable facts enter; output cannot pass accepted-score dispatch."""
    if scores.get("schema_version") != "case-research-scoring/6.1.0":
        raise ValueError("simulation_exact_6_1_required")
    validate_score(dict(state=state, candidate=candidate), scores)
    rows = rows_from_candidate(state, candidate)
    ids = {r.id for r in rows}
    resources = {r.code for r in state.resources}
    if not set(removed_ids) <= ids or not set(removed_resources) <= resources:
        raise ValueError("simulation_unknown_removal")
    overrides = overrides or {}
    if not set(overrides) <= ids:
        raise ValueError("simulation_unknown_override")
    original = {r.id: r for r in rows}
    for key, row in overrides.items():
        row = validate(ContinuousReview, row)
        if row.id != key or (row.node, row.kind, row.role) != (
            original[key].node,
            original[key].kind,
            original[key].role,
        ):
            raise ValueError("simulation_override_type")
    rows = [validate(ContinuousReview, overrides.get(r.id, r)) for r in rows]
    active = {
        r.id
        for r in rows
        if r.id not in removed_ids and path_available(r.dependency_paths, set(removed_resources))
    }
    statuses = {i.assessment_id: i.status for i in candidate.items}
    acquired = {
        r.id
        for r in rows
        if r.observation_available
        and statuses.get(r.id, "assessed")
        not in {"method_mismatch", "observation_ungradable", "input_missing", "intermediate"}
    }
    interpreted = {r.id for r in rows if r.judgment.value is not None}
    # Overrides already passed their own fact validation. Missing old input can be replaced.
    for key, row in overrides.items():
        if row.observation_available:
            acquired.add(key)
    groups = [
        dict(
            id=g.id,
            importance=dict(g.importance),
            slots=[
                dict(
                    assessment_ids=list(slot),
                    acquired=bool(set(slot) & acquired & active),
                    interpreted=bool(set(slot) & interpreted & active),
                )
                for slot in g.slots
            ],
        )
        for g in state.policy.groups
    ]
    return dict(
        schema_version="hypothetical-scoring/6.1.0",
        root=dict(
            summary=summarize(rows, set(removed_resources), set(removed_ids)),
            coverage=completeness(groups),
        ),
        remaining=sorted(active),
        missing=sorted(active - interpreted),
        roles=sorted({r.role for r in rows if r.id in active and (r.support or r.refutation)}),
        clinical_acceptance=False,
        disease_probability=None,
    )


def reverse(row):
    """Hypothesis only: reverse an already available effect and its interval."""
    row = validate(ContinuousReview, row)
    if row.judgment.value is None:
        raise ValueError("simulation_reverse_missing_interpretation")
    raw = row.model_dump(mode="json")
    raw["judgment"].update(
        value=-row.judgment.value, lower=-row.judgment.upper, upper=-row.judgment.lower
    )
    raw.update(
        support=row.refutation,
        refutation=row.support,
        direction="support"
        if row.judgment.value < 0
        else "refute"
        if row.judgment.value > 0
        else "neutral",
    )
    return validate(ContinuousReview, raw)

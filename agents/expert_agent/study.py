"""Map fresh Fake candidates to original validate_round/compare_rounds, before acceptance."""

from packages.evidence_details.models import digest as protocol_digest
from packages.evidence_scoring.accepted import rows_from_candidate
from packages.evidence_scoring.review_protocol import ReviewItem, ReviewRound, validate_round
from packages.evidence_scoring.review_study import compare_rounds
from packages.grounding.models import digest


def compare_candidates(state, submissions, protocol, ports, *, live_permission=None):
    material = state.material
    ids = {p.id for p in state.policy.items}
    citations = {s.reference.code for s in material.sources} | {
        o.reference.code for o in material.observations
    }
    rounds = []
    for sub in submissions:
        candidate = sub.candidate
        if sub.scope != material.scope:
            raise ValueError("round_frozen_scope")
        records = [r for r in ports.budget.records() if r.request_id == sub.request_id]
        if len(records) != 1 or records[0].output_checksum != digest(candidate):
            raise ValueError("round_request_binding")
        if not records[0].simulated:
            from packages.evidence_details.live_permission import require_live_permission

            require_live_permission(live_permission, state, role="review", record=records[0])
        live = not records[0].simulated
        rows = {r.id: r for r in rows_from_candidate(state, candidate)}
        items = list(candidate.items)
        for p in state.policy.items:
            if p.fixed_model_row is None:
                continue
            row = rows[p.id]
            items.append(
                ReviewItem(
                    assessment_id=p.id,
                    status="assessed" if row.judgment.value is not None else "reference_missing",
                    observation_available=row.observation_available,
                    observation_citations=[p.observation_code],
                    reference_citations=list(p.source_codes),
                    comparison="Domain fixed mapping; not sent to the model.",
                    observation=row.observation,
                    relation=row.relation,
                    applicability=row.applicability,
                    judgment=row.judgment,
                    limitation_allocation="Original fixed mapping",
                    unresolved=[],
                )
            )
        value = ReviewRound(
            schema_version="case-review-round/1.0.0",
            protocol_checksum=protocol_digest(protocol),
            material_checksum=material.scope.material_checksum,
            round_id=candidate.round,
            reviewer_kind="llm" if live else "assistant_manual_simulation",
            model_identifier=records[0].model,
            session_identifier=sub.request_id,
            prompt_checksum=sub.prompt_checksum,
            generation_settings={
                "mode": "bounded_live_research" if live else "synthetic_fake",
                "config_checksum": sub.config_checksum,
            },
            prior_answers_visible=False,
            items=items,
            purpose="candidate_review_not_production_authorization",
        )
        rounds.append(
            validate_round(
                value,
                expected_ids=ids,
                material_checksum=material.scope.material_checksum,
                protocol_checksum=protocol_digest(protocol),
                citation_ids=citations,
            )
        )
    result = compare_rounds(
        dict(
            schema_version="research-review-comparison-input/1.0.0",
            state=state,
            candidate=submissions[0].candidate,
        ),
        rounds,
        material_checksum=material.scope.material_checksum,
        citation_ids=citations,
        protocol=protocol,
        prompt_checksum=submissions[0].prompt_checksum,
    )
    return dict(
        schema_version="research-frozen-rounds/1.0.0",
        rounds=[r.model_dump(mode="json") for r in rounds],
        comparison=result,
        selected_round="A",
        selection_policy="first_valid_round_A_B_C",
        correlation="same_model_correlated_not_independent_medical_truth",
        real_ABC_executed=(len(rounds) == 3 and all(r.reviewer_kind == "llm" for r in rounds)),
        acceptance_count=0,
    )

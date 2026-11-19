"""Offline single-case mechanism checks; never authorizes or replaces reviews.

The caller supplies an already reconstructed v5 case. Experimental row copies
exist only inside this module; production policies and patient facts stay intact.
"""

from packages.evidence_details.models import digest
from packages.evidence_scoring import current
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.engine import path_available
from packages.evidence_scoring.review_protocol import validate_round


def indicators(state):
    summary = state["summary"]
    return {
        "S": summary["combined"]["support"],
        "R": summary["combined"]["refutation"],
        "conflict": summary["combined"]["conflict"],
        "D": summary["confidence"]["value"],
        "direction": summary["confidence"]["direction"],
        "parameter_envelope": [
            summary["confidence"]["lower"],
            summary["confidence"]["upper"],
        ],
        "coverage": state["coverage"],
    }


def _copy(row, **changes):
    return ContinuousReview.model_validate({**row.model_dump(mode="json"), **changes})


def run(base):
    rows, resources, acquired, interpreted = current.rows_from_records(base)
    baseline = current.state(rows, acquired, interpreted)
    expected = indicators(baseline)
    trials = []

    def add(name, changed_rows=None, removed_ids=frozenset(), removed_resources=frozenset()):
        result = current.state(
            rows if changed_rows is None else changed_rows,
            acquired,
            interpreted,
            removed_ids,
            removed_resources,
        )
        metrics = indicators(result)
        trials.append(
            {
                "id": name,
                "removed_assessments": sorted(removed_ids),
                "removed_resources": sorted(removed_resources),
                "affected_assessments": sorted(
                    r.id
                    for r in rows
                    if r.id in removed_ids
                    or not path_available(r.dependency_paths, removed_resources)
                ),
                "indicators": metrics,
                "D_delta": metrics["D"] - expected["D"]
                if metrics["D"] is not None and expected["D"] is not None
                else None,
            }
        )
        return metrics

    order = add("reverse_order", list(reversed(rows)))
    exact = add("exact_duplicate", rows + rows)
    # A new display/reference ID for the same fact must not create another vote.
    copies = [_copy(r, id="study-copy:" + r.id) for r in rows]
    renamed = add("same_family_copy", rows + copies)
    checks = {
        "order_invariant": order == expected,
        "exact_duplicate_invariant": exact == expected,
        "same_family_copy_invariant": renamed == expected,
    }
    # Remove just the family-level deduplication, keeping shared-input discount.
    no_family = [_copy(r, correlation_group="study-family:" + r.id) for r in rows]
    add("ablation_no_family_deduplication", no_family)
    # A second, explicit ablation also treats the leaves as independent inputs.
    independent = [_copy(r, dependence_block="study-block:" + r.id) for r in no_family]
    add("ablation_no_family_or_shared_input_discount", independent)
    add("without_age", removed_ids={"criterion:AGE"} & {r.id for r in rows})
    add("without_risk_background", removed_ids={r.id for r in rows if r.role == "risk"})
    measurement_ids = acquired - interpreted
    measurement = add("without_acquired_uninterpreted", removed_ids=measurement_ids)
    checks["uninterpreted_measurements_do_not_create_direction"] = all(
        measurement[k] == expected[k] for k in ("S", "R", "D", "conflict")
    )
    coverage_checks = []
    for resource in sorted(resources):
        after = add("remove_resource:" + resource, removed_resources={resource})
        for scope, before in expected["coverage"].items():
            lost = after["coverage"][scope]
            coverage_checks.append(lost["denominator"] == before["denominator"])
            for field in ("acquisition", "interpretation"):
                coverage_checks.append(
                    lost[field] is None if before[field] is None else lost[field] <= before[field]
                )
    checks["removal_preserves_denominators_and_cannot_add_coverage"] = all(coverage_checks)
    empty = add("without_all_assessments", removed_ids={r.id for r in rows})
    checks["no_evidence_is_unavailable_not_neutral_probability"] = empty["D"] is None
    return {
        "schema_version": "case-mechanism-study/1.0.0",
        "kind": "offline_fixed_review_mechanism_comparison",
        "case_count": 1,
        "new_medical_reviews": 0,
        "baseline": expected,
        "assessment_count": len(rows),
        "directional_assessments": [r.id for r in rows if r.support or r.refutation],
        "checks": checks,
        "passed": all(checks.values()),
        "trials": trials,
        "key_sets": current.key_analysis(rows, acquired, interpreted),
        "background_analysis": current.conclusion(rows),
        "limits": [
            "Ablations change only aggregation assumptions in offline copies, not patient facts.",
            "Invariant checks test software properties; ablation deltas have no preferred sign.",
            "One case cannot establish clinical validity, calibration or population performance.",
            "The parameter envelope is reused sensitivity analysis, not a confidence interval.",
            "No tree rewiring, new visual review or tool replacement interpretation was performed.",
        ],
    }


def compare_rounds(base, rounds, *, material_checksum, citation_ids, protocol, prompt_checksum):
    """Compare complete candidate rounds offline, without accepting any of them.

    Material curation and independent-session evidence belong to the coordinator;
    a checksum or a session string alone cannot prove either one.
    """
    research = base.get("schema_version") == "research-review-comparison-input/1.0.0"
    if research:
        from packages.evidence_details.acceptance_models import DomainState
        from packages.evidence_scoring.accepted import build_accepted, rows_from_candidate
        from packages.grounding.models import validate
        from packages.llm_provider.models import ReviewProposal

        state = validate(DomainState, base["state"])
        seed = validate(ReviewProposal, base["candidate"])
        if material_checksum != state.material.scope.material_checksum:
            raise ValueError("review_material_binding")
        original = rows_from_candidate(state, seed)
    else:
        original, _, _, _ = current.rows_from_records(base)
    parsed = [
        validate_round(
            r,
            expected_ids={x.id for x in original},
            material_checksum=material_checksum,
            protocol_checksum=digest(protocol),
            citation_ids=citation_ids,
        )
        for r in rounds
    ]
    if len(parsed) != 3 or {r.round_id for r in parsed} != {"A", "B", "C"}:
        raise ValueError("review_three_distinct_rounds_required")
    if len({r.session_identifier for r in parsed}) != 3:
        raise ValueError("review_sessions_must_differ")
    if any(r.prompt_checksum != prompt_checksum for r in parsed):
        raise ValueError("review_prompt_binding_mismatch")
    if (
        len(
            {
                (
                    r.reviewer_kind,
                    r.model_identifier,
                    r.prompt_checksum,
                    digest(r.generation_settings),
                )
                for r in parsed
            }
        )
        != 1
    ):
        raise ValueError("review_first_comparison_requires_fixed_reviewer_configuration")
    results, by_id = {}, {r.id: [] for r in original}
    for review in sorted(parsed, key=lambda r: r.round_id):
        items = {i.assessment_id: i for i in review.items}
        if research:
            candidate = ReviewProposal(
                scope_checksum=seed.scope_checksum, round=review.round_id, items=tuple(review.items)
            )
            scores = build_accepted(state, candidate)
            for key in by_id:
                by_id[key].append(items[key])
            results[review.round_id] = {
                "review_checksum": digest(review.model_dump(mode="json")),
                "indicators": indicators(scores["root"]),
                "background_analysis": current.conclusion(rows_from_candidate(state, candidate)),
                "scoring_version": scores["schema_version"],
            }
            continue
        rows, acquired, interpreted = [], set(), set()
        for old in original:
            item = items[old.id]
            if item.observation_available != old.observation_available:
                raise ValueError("review_cannot_change_frozen_acquisition")
            fixed_fields = ("value", "lower", "upper", "method")
            if (
                old.judgment.method == "fixed_model_margin"
                and any(getattr(item.judgment, k) != getattr(old.judgment, k) for k in fixed_fields)
            ) or (
                old.judgment.method != "fixed_model_margin"
                and item.judgment.method == "fixed_model_margin"
            ):
                raise ValueError("review_cannot_rejudge_fixed_model_mapping")
            ratings = (item.observation.value, item.relation.value, item.applicability.value)
            weight = ratings[0] * ratings[1] * ratings[2] if None not in ratings else None
            effect = item.judgment.value
            row = _copy(
                old,
                observation=item.observation.model_dump(mode="json"),
                relation=item.relation.model_dump(mode="json"),
                applicability=item.applicability.model_dump(mode="json"),
                judgment=item.judgment.model_dump(mode="json"),
                reason=item.comparison,
                criterion_checked=item.status == "assessed" and old.kind == "criterion",
                weight=weight,
                support=weight * max(effect, 0.0) if effect is not None else 0.0,
                refutation=weight * max(-effect, 0.0) if effect is not None else 0.0,
                direction="unavailable"
                if effect is None
                else "support"
                if effect > 0
                else "refute"
                if effect < 0
                else "neutral",
            )
            rows.append(row)
            by_id[old.id].append(item)
            if item.observation_available and item.status not in {
                "input_missing",
                "observation_ungradable",
                "method_mismatch",
                "intermediate",
            }:
                acquired.add(old.id)
            if item.status == "assessed":
                interpreted.add(old.id)
        results[review.round_id] = {
            "review_checksum": digest(review.model_dump(mode="json")),
            "indicators": indicators(current.state(rows, acquired, interpreted)),
            "background_analysis": current.conclusion(rows),
        }
    flags = protocol["disagreement_flags"]
    differences = []
    for key, items in by_id.items():
        ranges, missingness = {}, {}
        for field in ("observation", "relation", "applicability", "judgment"):
            values = [getattr(i, field).value for i in items]
            missingness[field] = len({v is None for v in values}) > 1
            ranges[field] = max(values) - min(values) if None not in values else None
        statuses = sorted({i.status for i in items})
        directions = sorted(
            {
                "unavailable"
                if i.judgment.value is None
                else "positive"
                if i.judgment.value > 0
                else "negative"
                if i.judgment.value < 0
                else "neutral"
                for i in items
            }
        )
        trigger = len(statuses) > 1 or len(directions) > 1 or any(missingness.values())
        trigger |= any(
            value is not None
            and value
            > flags[
                "judgment_range_greater_than"
                if field == "judgment"
                else "dimension_range_greater_than"
            ]
            for field, value in ranges.items()
        )
        differences.append(
            {
                "assessment_id": key,
                "statuses": statuses,
                "directions": directions,
                "numeric_ranges": ranges,
                "missingness_differs": missingness,
                "investigate": trigger,
            }
        )
    values = [r["indicators"]["D"] for r in results.values()]
    spread = max(values) - min(values) if None not in values else None
    root_directions = {r["indicators"]["direction"] for r in results.values()}
    return {
        "schema_version": "case-review-comparison/1.0.0",
        "protocol_checksum": digest(protocol),
        "material_checksum": material_checksum,
        "case_count": 1,
        "round_count": 3,
        "results": results,
        "items": differences,
        "root_score_range": spread,
        "root_requires_investigation": len(root_directions) > 1
        or (spread is not None and spread > flags["root_score_range_greater_than"]),
        "independence": "requires_separate_session_and_material_delivery_evidence",
        "production_authorization": False,
        "consensus_score": None,
    }

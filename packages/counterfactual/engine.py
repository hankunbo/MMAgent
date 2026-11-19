"""Candidate computation; all numerical work stays in the original scoring owner."""

from packages.evidence_scoring.accepted import rows_from_candidate
from packages.evidence_scoring.simulation import project, reverse
from packages.graph_engine.simulation import removal_closure
from packages.grounding.models import digest


def authorize(policy, fact, candidates):
    if (policy.case_code, policy.run_id, policy.fact_checksum) != (
        fact.binding.case_code,
        fact.binding.run_id,
        fact.checksum,
    ) or policy.rule != next(r for r in fact.binding.locks if r.role == "rule"):
        raise ValueError("simulation_policy_scope")
    if len(candidates) > policy.max_candidates or len({c.id for c in candidates}) != len(
        candidates
    ):
        raise ValueError("simulation_candidate_budget_or_duplicate")
    ids = {i.id for i in fact.state.policy.items}
    resources = {r.code for r in fact.state.resources}
    for candidate in candidates:
        kinds = {i.kind for i in candidate.interventions}
        if any(set(combo) <= kinds for combo in policy.forbidden_combinations):
            raise ValueError("simulation_forbidden_combination")
        occupied = set()
        for intervention in candidate.interventions:
            targets = set(intervention.targets)
            if not targets <= set(policy.allowed.get(intervention.kind, ())):
                raise ValueError("simulation_policy_denied")
            if not targets <= (resources if intervention.kind == "remove_resource" else ids):
                raise ValueError("simulation_unknown_target")
            if occupied & targets:
                raise ValueError("simulation_overlapping_intervention")
            occupied |= targets
            if intervention.alternative and intervention.alternative not in policy.alternatives:
                raise ValueError("simulation_alternative_not_declared")


def paired(before, after):
    if (
        before.binding.case_code != after.binding.case_code
        or before.binding.input_checksum != after.binding.input_checksum
        or before.binding.run_id == after.binding.run_id
    ):
        raise ValueError("simulation_pair_common_input_or_run")
    a, b = ({r.role: r for r in f.binding.locks} for f in (before, after))
    differences = [
        dict(
            field="lock:" + k,
            before=a[k].model_dump(mode="json") if k in a else None,
            after=b[k].model_dump(mode="json") if k in b else None,
        )
        for k in sorted(a.keys() | b.keys())
        if a.get(k) != b.get(k)
    ]
    for name, left, right in (
        ("task_policy", before.state.policy, after.state.policy),
        ("observations", before.state.material.observations, after.state.material.observations),
        ("sources", before.state.material.sources, after.state.material.sources),
        ("resources", before.state.resources, after.state.resources),
        ("reviews", before.candidate.items, after.candidate.items),
    ):
        if digest(left) != digest(right):
            differences.append(dict(field=name, before=digest(left), after=digest(right)))
    return dict(
        identity="fact_fact_comparison_candidate",
        before=before.references(),
        after=after.references(),
        differences=differences,
        attribution="no_single_tool_or_causal_attribution",
        created_runs=0,
    )


def compatible(original, alternative, target):
    old = next(i for i in original.state.policy.items if i.id == target)
    new = next((i for i in alternative.state.policy.items if i.id == target), None)
    if new is None or any(
        getattr(old, k) != getattr(new, k)
        for k in ("node", "kind", "role", "method", "unit", "laterality", "definition")
    ):
        raise ValueError("simulation_alternative_type_unit_domain")
    return new


def compute(fact, candidate, policy, alternatives):
    authorize(policy, fact, (candidate,))
    baseline = fact.baseline()
    rows = {r.id: r for r in rows_from_candidate(fact.state, fact.candidate)}
    evidence, resources, overrides, refs, reasons, affected = set(), set(), {}, {}, [], {}
    for i in candidate.interventions:
        if i.kind == "remove_evidence":
            evidence.update(i.targets)
        elif i.kind == "remove_resource":
            resources.update(i.targets)
        elif i.kind == "reverse":
            for target in i.targets:
                overrides[target] = reverse(rows[target])
                affected[target] = "hypothetical_effect_reversal_original_measurement_unchanged"
        else:
            other = alternatives.get(i.alternative)
            if other is None:
                reasons.append(i.kind + ":missing_alternative_result")
                continue
            if other.checksum != policy.alternatives[i.alternative]:
                raise ValueError("simulation_alternative_drift")
            pair = paired(fact, other)
            if i.kind == "replace_tool" and not any(
                d["field"] == "lock:catalog" for d in pair["differences"]
            ):
                raise ValueError("simulation_tool_requires_actual_catalog_change")
            replacements = {r.id: r for r in rows_from_candidate(other.state, other.candidate)}
            old_obs = {o.reference.code: o for o in fact.state.material.observations}
            new_obs = {o.reference.code: o for o in other.state.material.observations}
            for target in i.targets:
                item = compatible(fact, other, target)
                old_item = next(p for p in fact.state.policy.items if p.id == target)
                row = replacements[target]
                if not row.observation_available or row.judgment.value is None:
                    reasons.append(i.kind + ":missing_observation_or_interpretation:" + target)
                elif i.kind == "replace_tool" and (
                    old_obs[old_item.observation_code].reference
                    == new_obs[item.observation_code].reference
                ):
                    raise ValueError("simulation_tool_requires_new_output_and_review")
                else:
                    overrides[target] = row
                    affected[target] = i.kind + ":independent_accepted_result"
            refs[i.alternative] = pair
    closed, removed_resources = removal_closure(fact.state, evidence=evidence, resources=resources)
    affected.update(closed)
    if set(overrides) & set(closed):
        raise ValueError("simulation_dependency_combination_conflict")
    simulated = (
        None
        if reasons
        else project(
            fact.state,
            fact.candidate,
            fact.scores,
            removed_ids=set(closed),
            removed_resources=removed_resources,
            overrides=overrides,
        )
    )
    comparisons = compare(baseline, simulated)
    classification = classify(baseline, simulated, policy)
    return dict(
        baseline=dict(reference=fact.references(), **baseline),
        simulated=simulated,
        affected=tuple(dict(item=k, reason=v) for k, v in sorted(affected.items())),
        remaining=tuple(simulated["remaining"] if simulated else ()),
        comparisons=comparisons,
        alternatives=refs,
        classification=classification,
        reasons=tuple(reasons),
        counters=dict(
            expected_cost=policy.expected_cost,
            expected_failure_risk=policy.expected_failure_risk,
            actual=dict(model_requests=0, tool_starts=0, created_runs=0),
        ),
    )


def compare(before, after):
    def metrics(value):
        s = value["root"]["summary"]
        return dict(
            score=s["confidence"]["value"],
            **s["combined"],
            **{
                scope + ":" + name: v
                for scope, row in value["root"]["coverage"].items()
                for name, v in row.items()
            },
        )

    a, b = metrics(before), metrics(after) if after else {}
    return {
        k: dict(
            original=v,
            simulated=b.get(k),
            delta=b[k] - v if v is not None and b.get(k) is not None else None,
        )
        for k, v in a.items()
    }


def classify(before, after, policy):
    if after is None or before["root"]["summary"]["confidence"]["value"] is None:
        return "indeterminate"
    a, b = (v["root"]["summary"]["confidence"] for v in (before, after))
    if b["value"] is None:
        return "necessary"
    if a["direction"] != b["direction"]:
        return "conclusion_changed"
    fidelity = metric_fidelity(compare(before, after), policy)
    if fidelity["unavailable"]:
        return "indeterminate"
    return "context_redundant" if fidelity["preserved"] else "components_changed"


def metric_fidelity(comparisons, policy):
    """One metric interpretation for classification and search, no scoring formula."""
    selected = [m for m in policy.metrics if m != "coverage"]
    if "coverage" in policy.metrics:
        # Both acquisition and interpretation for every original coverage scope.
        # Denominators are structural invariants, not tolerance-scaled metrics.
        selected.extend(
            k for k in sorted(comparisons) if k.endswith((":acquisition", ":interpretation"))
        )
    unavailable = [k for k in selected if comparisons[k]["delta"] is None]
    changed = [
        k
        for k in selected
        if comparisons[k]["delta"] is not None and abs(comparisons[k]["delta"]) > policy.tolerance
    ]
    return dict(
        selected=selected,
        unavailable=unavailable,
        changed=changed,
        preserved=bool(selected) and not unavailable and not changed,
    )


def search_fidelity(before, after, comparisons, policy):
    metrics = metric_fidelity(comparisons, policy)
    reasons = []
    if not metrics["preserved"]:
        reasons.append("declared_metrics_not_preserved")
    if after is None:
        return dict(metrics=metrics, eligible=False, reasons=[*reasons, "no_comparison"])
    # Explicit safety constraints of semantics 1.1, regardless of selected metrics.
    a, b = (v["root"]["summary"]["confidence"] for v in (before, after))
    if a["value"] is None or b["value"] is None:
        reasons.append("unavailable_directional_score")
    elif a["direction"] != b["direction"]:
        reasons.append("direction_changed")
    if policy.preserve_roles and before["roles"] != after["roles"]:
        reasons.append("roles_changed")
    for scope, row in sorted(before["root"]["coverage"].items()):
        new = after["root"]["coverage"][scope]
        if row["denominator"] != new["denominator"]:
            reasons.append(scope + ":denominator_changed")
        old_value, new_value = row["interpretation"], new["interpretation"]
        # No division by zero or invented ratio. Unknown cannot prove retention.
        if old_value is None or new_value is None:
            reasons.append(scope + ":retention_unavailable")
        elif old_value == 0:
            if new_value != 0:
                reasons.append(scope + ":zero_baseline_changed")
        elif new_value / old_value < policy.coverage_retention:
            reasons.append(scope + ":coverage_retention")
    return dict(metrics=metrics, eligible=not reasons, reasons=reasons)

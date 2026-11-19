"""Metric contracts share the original compare and original 6.1 computation."""

import copy

import pytest

from packages.counterfactual.engine import classify, compare, metric_fidelity, search_fidelity
from tests.counterfactual.support import candidate, fixture, run


@pytest.mark.parametrize(
    "metrics,found", [(("score",), True), (("support", "refutation", "conflict"), False)]
)
def test_cancellation_differs_between_score_only_and_components(metrics, found):
    x = fixture(
        statuses=("obtained",) * 4,
        directions=(0.8, -0.8, 0.2, -0.2),
        policy_updates=dict(metrics=metrics, coverage_retention=0.25),
    )
    r = x.service.search()
    assert bool(r["minimal_within_explored"]) == found
    assert r["baseline"]["remaining"] == [f"item:{i}" for i in range(4)]
    assert r["baseline"]["fidelity"]["eligible"]
    assert r["outcome"] == ("bounded_minimal_sets_found" if found else "no_feasible_removal_found")
    assert r["complete"] and r["explored"] == 15 and not r["global_optimum"]
    for row in r["minimal_within_explored"]:
        report = x.service.read(candidate_id=row["candidate_id"])
        assert report["classification"] == "context_redundant"
        assert metric_fidelity(report["comparisons"], x.policy)["preserved"]
    assert x.service.search() == r  # same key, same semantics, no re-request
    assert x.f.budget.used == 1


@pytest.mark.parametrize("metric", ["support", "refutation", "conflict", "score"])
def test_single_metric_change_and_unselected_metrics(metric):
    x = fixture()
    before = x.fact.baseline()
    after = copy.deepcopy(before)
    if metric == "score":
        after["root"]["summary"]["confidence"]["value"] += 0.01
    else:
        after["root"]["summary"]["combined"][metric] += 0.01
    policy = x.policy.model_copy(update=dict(metrics=(metric,)))
    comparisons = compare(before, after)
    assert metric_fidelity(comparisons, policy)["changed"] == [metric]
    assert classify(before, after, policy) == "components_changed"
    assert not search_fidelity(before, after, comparisons, policy)["eligible"]
    other = "support" if metric != "support" else "refutation"
    policy = policy.model_copy(update=dict(metrics=(other,)))
    assert classify(before, after, policy) == "context_redundant"
    assert search_fidelity(before, after, comparisons, policy)["eligible"]


@pytest.mark.parametrize("kind", ["redundant", "coverage", "shared", "roles"])
def test_redundancy_coverage_roles_and_shared_dependency(kind):
    x = fixture(
        statuses=("obtained",) * 2,
        same_source=True,
        alternative_slot=kind != "coverage",
        shared_product=kind == "shared",
        policy_updates=dict(metrics=("score",), coverage_retention=0.8),
    )
    if kind == "roles":
        before = x.fact.baseline()
        after = copy.deepcopy(before)
        after["roles"] = []
        fidelity = search_fidelity(before, after, compare(before, after), x.policy)
        assert fidelity["reasons"] == ["roles_changed"]
        policy = x.policy.model_copy(update=dict(preserve_roles=False))
        assert search_fidelity(before, after, compare(before, after), policy)["eligible"]
        return
    r = x.service.search()
    assert bool(r["minimal_within_explored"]) == (kind == "redundant")
    if kind == "coverage":
        assert any(
            "comprehensive_research:coverage_retention" in e["reasons"] for e in r["evaluations"]
        )
    if kind == "shared":
        assert all("unavailable_directional_score" in e["reasons"] for e in r["evaluations"])


@pytest.mark.parametrize(
    "mode", ["acquisition", "interpretation", "null", "zero", "denominator", "direction"]
)
def test_coverage_expansion_null_zero_denominator_and_explicit_direction(mode):
    x = fixture()
    before = x.fact.baseline()
    after = copy.deepcopy(before)
    policy = x.policy.model_copy(update=dict(metrics=("coverage",)))
    scope = "comprehensive_research"
    if mode in {"acquisition", "interpretation"}:
        after["root"]["coverage"][scope][mode] -= 0.01
    elif mode == "null":
        for v in (before, after):
            v["root"]["coverage"][scope].update(
                denominator=0, acquisition=None, interpretation=None
            )
    elif mode == "zero":
        for v in (before, after):
            v["root"]["coverage"][scope]["interpretation"] = 0
    elif mode == "denominator":
        after["root"]["coverage"][scope]["denominator"] += 1
    else:
        after["root"]["summary"]["confidence"]["direction"] = "refute"
    comparisons = compare(before, after)
    fidelity = metric_fidelity(comparisons, policy)
    assert set(fidelity["selected"]) == {
        s + ":" + m for s in before["root"]["coverage"] for m in ("acquisition", "interpretation")
    }
    eligibility = search_fidelity(before, after, comparisons, policy)
    assert eligibility["eligible"] == (mode == "zero")
    if mode == "null":
        assert fidelity["unavailable"] and classify(before, after, policy) == "indeterminate"
        assert comparisons[scope + ":interpretation"]["delta"] is None
    if mode in {"acquisition", "interpretation"}:
        assert classify(before, after, policy) == "components_changed"
    if mode == "direction":
        assert fidelity["preserved"] and eligibility["reasons"] == ["direction_changed"]


def test_all_removed_has_no_score_not_zero_and_no_feasible_is_not_no_evidence():
    x = fixture(policy_updates=dict(metrics=("support",)))
    report = run(x, candidate())["report"]
    assert report["comparisons"]["score"]["simulated"] is None
    assert report["comparisons"]["score"]["delta"] is None
    assert report["classification"] == "necessary"
    x = fixture(policy_updates=dict(metrics=("support",)))
    search = x.service.search()
    assert search["outcome"] == "no_feasible_removal_found"
    assert search["baseline"]["remaining"] == ["item:0"]
    assert search["baseline"]["fidelity"]["eligible"]
    assert search["minimal_within_explored"] == []

"""Known identities must not spend the next question's bounded fetch quota."""

from copy import deepcopy

import pytest

from packages.evidence_reuse.definition_sources import DefinitionMaterialUnavailable
from tests.evaluation_preparation.test_intent_definition_retrieval import owner


def plan(hits):
    decisions = []
    for position, hit in enumerate(hits):
        known = hit["reference"]["code"] == "known"
        decisions.append(
            dict(
                position=position,
                hit_ref=hit["reference"],
                already_acquired_identity=known,
                ambiguous_source_identity=False,
                conflicts=[],
                matched=[
                    dict(
                        index_key="original",
                        source_ref=dict(code="source"),
                        source_version=dict(code="source-version"),
                    )
                ]
                if known
                else [],
            )
        )
    return dict(
        schema_version="definition-fetch-order/1.0.0",
        index_keys=["original"],
        hits=hits,
        decisions=decisions,
        order=sorted(range(len(hits)), key=lambda i: decisions[i]["already_acquired_identity"]),
        eligibility_granted=False,
    )


def test_known_material_preserves_fetch_for_next_query_and_cold_replay():
    sources, _, _ = owner()
    hits = [dict(reference=dict(code=c)) for c in ("new-first", "known")]
    sources.search_public = lambda **kw: deepcopy(hits)
    sources.repository.rank_definition_discovery_hits = lambda **kw: plan(kw["hits"])
    fetched, journal = [], {}

    def fetch(hit):
        fetched.append(hit["reference"]["code"])
        raise DefinitionMaterialUnavailable("synthetic_missing_body")

    sources.fetch_hit = fetch

    def operation(key, kind, call):
        if key not in journal:
            if kind == "fetch_group" and len(fetched) >= 2:
                return dict(status="public_budget_exhausted")
            journal[key] = call()
        return deepcopy(journal[key])

    def run(key):
        return sources.search(
            query="q",
            key=key,
            limit=3,
            operation=operation,
            retrieval_mode="unseen-first-for-owner-gap/1.0.0",
        )

    first = run("first")
    hits[:] = [dict(reference=dict(code="new-second"))]
    run("second")
    assert fetched == ["new-first", "new-second"]
    deferred = [b for b in first["backlog"] if b.get("reason") == "known_source_fetch_deferred"]
    assert len(deferred) == 1 and deferred[0]["hit"]["reference"]["code"] == "known"
    run("first")
    assert fetched == ["new-first", "new-second"]


@pytest.mark.parametrize("mutation", ["missing", "position", "hit", "truthy", "conflict", "index"])
def test_invalid_persisted_deferral_never_dispatches(mutation):
    sources, _, _ = owner()
    hits = [dict(reference=dict(code="known"))]
    value = plan(hits)
    if mutation == "missing":
        value.pop("decisions")
    elif mutation == "position":
        value["decisions"][0]["position"] = True
    elif mutation == "hit":
        value["decisions"][0]["hit_ref"] = dict(code="other")
    elif mutation == "truthy":
        value["decisions"][0]["already_acquired_identity"] = "yes"
    elif mutation == "conflict":
        value["decisions"][0]["conflicts"] = [dict(reason="conflicting identity")]
    else:
        value["decisions"][0]["matched"][0]["index_key"] = "unverified-index"
    sources.search_public = lambda **kw: hits
    sources.repository.rank_definition_discovery_hits = lambda **kw: value
    sources.fetch_hit = lambda hit: pytest.fail("invalid decision must not dispatch")
    with pytest.raises(ValueError, match="definition_fetch_order_binding"):
        sources.search(
            query="q",
            key="bad",
            limit=3,
            operation=lambda k, t, f: f(),
            retrieval_mode="unseen-first-for-owner-gap/1.0.0",
        )


def test_fresh_cold_source_requires_original_pool_but_allows_growth():
    original, _, _ = owner()
    hits = [dict(reference=dict(code="known"))]
    original.search_public = lambda **kw: deepcopy(hits)
    original.repository.rank_definition_discovery_hits = lambda **kw: plan(kw["hits"])
    original.fetch_hit = lambda hit: pytest.fail("known hit must not fetch")
    journal = {}

    def operation(key, kind, call):
        if key not in journal:
            journal[key] = call()
        return deepcopy(journal[key])

    def search(source):
        return source.search(
            query="q",
            key="cold",
            limit=3,
            operation=operation,
            retrieval_mode="unseen-first-for-owner-gap/1.0.0",
        )

    expected = search(original)
    cold, _, _ = owner()
    cold.index_keys = ("other",)
    cold.search_public = lambda **kw: pytest.fail("confirmed search must not repeat")
    cold.repository.rank_definition_discovery_hits = lambda **kw: pytest.fail("no reranking")
    cold.fetch_hit = lambda hit: pytest.fail("no extra wire request")
    with pytest.raises(ValueError, match="definition_fetch_order_binding"):
        search(cold)
    cold.index_keys = ("original", "other")
    assert search(cold)["discovery"]["fetch_order"] == expected["discovery"]["fetch_order"]

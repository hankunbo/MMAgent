"""Discovery order is owner-derived, journaled, and not a medical eligibility grant."""

import pytest

from tests.evaluation_preparation.test_intent_definition_retrieval import owner


def test_new_gap_mode_prioritizes_unknown_hit_without_changing_fetch_identity():
    sources, _, _ = owner()
    hits = [dict(reference=dict(code=code)) for code in ("known-a", "known-b", "new")]
    sources.search_public = lambda **kwargs: hits
    sources.fetch_hit = lambda _: pytest.fail("quota stops before transport")
    sources.repository.rank_definition_discovery_hits = lambda **kwargs: dict(
        schema_version="definition-fetch-order/1.0.0",
        index_keys=["original"],
        order=[2, 0, 1],
        hits=hits,
        eligibility_granted=False,
        decisions=[
            dict(
                position=i,
                hit_ref=hit["reference"],
                already_acquired_identity=i < 2,
                ambiguous_source_identity=False,
                conflicts=[],
                matched=[
                    dict(
                        index_key="original",
                        source_ref=dict(code="source"),
                        source_version=dict(code="version"),
                    )
                ]
                if i < 2
                else [],
            )
            for i, hit in enumerate(hits)
        ],
    )
    calls = []

    def operation(key, kind, call):
        calls.append((key, kind))
        if kind == "fetch_group":
            return dict(status="public_budget_exhausted")
        return call()

    result = sources.search(
        query="q",
        key="gap",
        limit=3,
        operation=operation,
        retrieval_mode="unseen-first-for-owner-gap/1.0.0",
    )
    assert calls[-1] == ("gap:fetch:2", "fetch_group")
    assert result["discovery"]["fetch_order"]["order"] == [2, 0, 1]
    budget = next(
        b for b in result["backlog"] if b.get("reason") == "definition_public_request_budget"
    )
    assert budget["unattempted_hits"] == [hits[2], hits[0], hits[1]]


def test_changed_hit_cannot_reuse_another_persisted_fetch_plan():
    sources, _, _ = owner()
    sources.search_public = lambda **kwargs: [dict(reference=dict(code="new"))]
    sources.fetch_hit = lambda _: pytest.fail("invalid plan never dispatches")

    def operation(key, kind, call):
        if kind == "fetch_order":
            return dict(
                schema_version="definition-fetch-order/1.0.0",
                order=[0],
                hits=[dict(reference=dict(code="old"))],
                eligibility_granted=False,
            )
        return call()

    with pytest.raises(ValueError, match="definition_fetch_order_binding"):
        sources.search(
            query="q",
            key="gap",
            limit=1,
            operation=operation,
            retrieval_mode="unseen-first-for-owner-gap/1.0.0",
        )


def test_new_policy_trial_binding_and_frozen_config(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from packages.multimodal_research import definition_trial
    from packages.multimodal_research.definition_expansion import UnseenFirstExpansionPolicy
    from packages.multimodal_research.implementation import freeze
    from tests.evaluation_preparation.test_component_trial import envelope

    path = "configs/research/multimodal_definition_gate_v16.json"
    value, spec = envelope(tmp_path, monkeypatch, path)
    assert value["schema_version"] == "multimodal-definition-trial/1.11.0"
    assert definition_trial.read(tmp_path) == value
    assert path in freeze()
    raw = json.loads(Path(path).read_bytes())
    assert raw["schema_version"] == "campaign-definition-expansion/1.16.0"
    assert raw["max_fetch_requests"] == raw["max_search_requests"] == 2
    assert UnseenFirstExpansionPolicy.model_validate_json(
        json.dumps(raw)
    ).unresolved_relation_retrieval == ("unseen-first-for-owner-gap/1.0.0")
    value["schema_version"] = "multimodal-definition-trial/1.10.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        definition_trial.checked(value, spec)

"""Local recall alternatives never become medical admission or network retries."""

from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources


def plan(**changes):
    return dict(
        schema_version="definition-passage-plan/1.0.0",
        queries=["RNFL differential", "retinal nerve fiber layer", "RNFL"],
        max_per_document=3,
        context_radius=1,
        max_characters=12000,
        **changes,
    )


def source(code, ordinal=1):
    return dict(
        code=code,
        text="A complete public paragraph.",
        checksum=code,
        source_version=dict(code="PUBLIC:DOC"),
        purpose="research_material",
        position=dict(ordinal=ordinal),
        hit=dict(chunk_ref=dict(code=code)),
    )


def owner(*, error=None):
    seen = []
    a, b = source("A"), source("B", 2)

    def query(key, text, filters, **options):
        assert options == dict(fts_single_pass=True)
        seen.append(text)
        if error:
            raise error
        return dict(hits=[a["hit"]] if text != "RNFL differential" else [])

    repo = SimpleNamespace(
        query_candidate_index=query,
        read_definition_material=lambda hit, *a, **kw: source(hit["chunk_ref"]["code"]),
        read_definition_context=lambda *a, **kw: [b],
    )
    s = KnowledgeDefinitionSources(
        repository=repo,
        store=None,
        index_keys=("original-index",),
        passage_plans={"public question": plan()},
    )
    return s, seen


def test_local_aliases_and_context_reuse_original_material_without_network():
    s, seen = owner()
    value = s.search(
        query="public question",
        key="q",
        limit=3,
        operation=lambda *args: pytest.fail("must reuse source"),
    )
    assert seen == ["RNFL differential", "retinal nerve fiber layer", "RNFL"]
    assert [x["code"] for x in value["materials"]] == ["A", "B"]
    assert value["reused"]
    assert value["retrieval"]["query"] == "public question"
    assert value["retrieval"]["medical_entailment_proven"] is False
    assert value["retrieval"]["selected_codes"] == ["A", "B"]


def test_zero_capacity_never_queries_or_sends():
    s, seen = owner()
    with pytest.raises(ValueError, match="material_limit"):
        s.search(query="public question", key="q", limit=0, operation=None)
    assert not seen


def test_missing_plan_and_technical_failure_are_not_no_literature():
    s, _ = owner()
    with pytest.raises(ValueError, match="passage_plan_missing"):
        s.search(query="unknown", key="q", limit=3, operation=None)
    s, _ = owner(error=RuntimeError("database_unavailable"))
    with pytest.raises(RuntimeError, match="database_unavailable"):
        s.search(query="public question", key="q", limit=3, operation=None)


def test_plan_is_bounded_and_rejects_private_or_duplicate_queries():
    for queries in (["RNFL"] * 2, ["patient id: 123"], ["x"] * 7):
        with pytest.raises(ValueError):
            KnowledgeDefinitionSources(
                repository=None, store=None, passage_plans={"q": {**plan(), "queries": queries}}
            )


def test_old_search_stays_single_query_without_context():
    seen = []
    repo = SimpleNamespace(query_candidate_index=lambda k, q, f: seen.append(q) or dict(hits=[]))
    s = KnowledgeDefinitionSources(repository=repo, store=None, index_keys=("old",))
    value = s.search(query="RNFL differential", key="q", limit=2, operation=None)
    assert seen == ["RNFL differential"]
    assert "retrieval" not in value


def test_current_guard_runs_before_local_reads():
    s, seen = owner()

    def stop():
        raise RuntimeError("cancelled")

    s.before_read = stop
    with pytest.raises(RuntimeError, match="cancelled"):
        s.search(query="public question", key="q", limit=3, operation=None)
    assert not seen


def test_new_policy_and_prompt_are_explicit_old_wire_is_unchanged():
    import json
    from pathlib import Path

    from packages.multimodal_research.campaign_renewal import compact_configuration
    from packages.multimodal_research.definition_expansion import (
        definition_config,
        read_expansion_policy,
    )

    raw = json.loads(Path("configs/research/multimodal_definition_gate_v3.json").read_bytes())
    policy = read_expansion_policy(raw)
    assert policy.schema_version == "campaign-definition-expansion/1.4.0"
    assert policy.max_definitions == policy.max_new_observations == policy.max_rounds == 1
    assert (
        policy.max_candidates == 5 and policy.max_search_requests == policy.max_fetch_requests == 2
    )
    old = definition_config(compact_configuration())
    new = definition_config(compact_configuration(), prompt_version=policy.prompt_version)
    for before, after in zip(old.roles, new.roles, strict=True):
        if before.role != "source":
            assert before == after
        else:
            assert before.prompt_version == "definition-draft-2.0.0"
            assert after.prompt_version == "definition-draft-2.1.0"
            assert before.prompt_checksum != after.prompt_checksum
    raw["passage_plans"].pop(raw["public_queries"][0])
    with pytest.raises(ValueError, match="passage_plan_binding"):
        read_expansion_policy(raw)


def test_per_document_limit_deduplicates_context_and_aliases():
    s, _ = owner()
    s.passage_plans["public question"] = s.passage_plans["public question"].model_copy(
        update={"max_per_document": 1}
    )
    value = s.search(query="public question", key="q", limit=5, operation=None)
    assert [m["code"] for m in value["materials"]] == ["A"]


def test_current_material_exception_not_swallowed_as_an_empty_search():
    s, _ = owner()

    def unavailable(*a, **kw):
        raise ValueError("source_unavailable")

    s.repository.read_definition_material = unavailable
    with pytest.raises(ValueError, match="source_unavailable"):
        s.search(query="public question", key="q", limit=3, operation=None)


@pytest.mark.parametrize("contextual", [False, True])
@pytest.mark.parametrize("reason", ["table_unavailable", "source_unavailable", "chunk_checksum"])
def test_rejected_table_does_not_hide_valid_prose_or_swallow_authority_failure(contextual, reason):
    from packages.knowledge_claims.evidence import Rejected

    s, _ = owner()
    if not contextual:
        s.passage_plans = {}
    a, b = source("A"), source("B")
    s.repository.query_candidate_index = lambda *a, **kw: dict(hits=[a_hit, b_hit])
    a_hit, b_hit = a["hit"], b["hit"]

    def read(hit, *args, **kwargs):
        if hit == a_hit:
            raise Rejected(reason)
        return b

    s.repository.read_definition_material = read
    if reason != "table_unavailable":
        with pytest.raises(Rejected, match=reason):
            s.search(query="public question", key="q", limit=1, operation=None)
        return
    value = s.search(query="public question", key="q", limit=1, operation=None)
    assert value["materials"] == [b]
    assert value["backlog"][0] == dict(reason="source_chunk_table_unavailable", hit=a_hit)
    assert "A" not in s.hits
    if contextual:
        assert value["retrieval"]["selected_codes"] == ["B"]


def test_primary_expression_is_tried_across_documents_before_fallback():
    s, _ = owner()
    s.index_keys = ("first", "second")
    seen = []
    s.repository.query_candidate_index = lambda k, q, f, **kw: seen.append((k, q)) or dict(hits=[])
    s.search(query="public question", key="q", limit=3, operation=None)
    assert seen[:2] == [("first", "RNFL differential"), ("second", "RNFL differential")]


def test_bibliography_hit_does_not_spend_definition_material_capacity():
    s, _ = owner()
    reference = {**source("A"), "position": {"section_path": ["References", "section"]}}
    s.repository.read_definition_material = lambda *a, **kw: reference
    value = s.search(query="public question", key="q", limit=3, operation=None)
    assert value["materials"] == []
    assert any(b["reason"] == "bibliography_not_definition_text" for b in value["backlog"])
    assert value["retrieval"]["selected_codes"] == []


def test_contextual_campaign_reserves_material_capacity_for_second_question(tmp_path):
    import json
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    e = fixture(tmp_path)
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v3.json").read_bytes())
    )
    e.query_plan = lambda: [dict(query="q1", selected=True), dict(query="q2", selected=True)]
    calls = []
    e.sources.search = lambda **kw: (
        calls.append((kw["query"], kw["limit"])) or dict(materials=[], backlog=[])
    )
    e.advance()
    assert calls == [("q1", 3), ("q2", 3)]


@pytest.mark.parametrize("version", ["definition-draft-2.1.0", "definition-draft-2.2.0"])
def test_contextual_prompt_reaches_real_provider_dispatch_and_acceptance(tmp_path, version):
    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.definition_expansion import definition_config
    from tests.evaluation_preparation.test_definition_expansion import draft, fixture

    e = fixture(tmp_path)
    config = definition_config(e.provider.config, prompt_version=version)
    e.provider = DeterministicFakeProvider(
        config,
        [
            dict(
                schema_version="research-definition-proposal/1.0.0",
                definitions=[draft(e.binding["target"])],
                no_addition_reason=None,
            )
        ],
    )
    revision = e.advance()
    assert revision["added"] and e.budget.used == 1
    assert len(e.provider.transport.calls) == 1
    assert e.advance() == revision and len(e.provider.transport.calls) == 1

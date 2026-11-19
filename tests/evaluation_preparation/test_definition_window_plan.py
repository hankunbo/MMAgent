"""Source assembly's complete plan precedes all original index writes."""

from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.knowledge_index.models import ChunkPolicy, ParsePolicy
from tests.knowledge_index.test_window_selection import CHECKSUM, document


def assembly(monkeypatch, count=262):
    from packages.knowledge_index import parsers

    monkeypatch.setattr(
        parsers,
        "parse",
        lambda *a, **k: SimpleNamespace(document=document(count), reason=None, status="parsed"),
    )
    owner = KnowledgeDefinitionSources(
        repository=None, store=None, material_policy="xml-jats-citations-windowed/1.0.0"
    )
    request = SimpleNamespace(
        document=SimpleNamespace(
            response=SimpleNamespace(
                body=b"public fixture", mime="application/xml", response_checksum=CHECKSUM
            )
        ),
        bibliography=SimpleNamespace(language="en"),
    )
    options = dict(
        parse_policy=ParsePolicy(version="xml-jats-citations/1.0.0"),
        chunk_policy=ChunkPolicy(max_chars=3000, max_chunks=250),
    )
    return owner, request, options


def test_complete_plan_is_journaled_before_writes_and_confirmed_windows_not_repeated(monkeypatch):
    owner, request, options = assembly(monkeypatch)
    journal, calls = {}, []

    def operation(key, kind, call):
        if key not in journal:
            journal[key] = call()
        return journal[key]

    def build():
        assert len(journal["source:plan"]["policies"]) == 2
        policy = options["chunk_policy"]
        calls.append(policy.first_ordinal)
        return dict(status="indexed", key=str(policy.first_ordinal))

    result = owner.window_indexes(
        request=request, options=options, build=build, operation=operation, key="source"
    )
    assert calls == [1, 251]
    assert (
        owner.window_indexes(
            request=request, options=options, build=build, operation=operation, key="source"
        )
        == result
    )
    assert calls == [1, 251]


def test_capacity_backlog_has_no_partial_index_writes(monkeypatch):
    owner, request, options = assembly(monkeypatch, 1001)

    def forbidden():
        pytest.fail("incomplete plan must not write indexes")

    indexes, notes = owner.window_indexes(
        request=request,
        options=options,
        build=forbidden,
        operation=lambda key, kind, call: call(),
        key="source",
    )
    assert indexes == []
    assert notes[0]["plan"]["reason"] == "source_window_limit"


def test_window_failure_preserves_confirmed_prefix_not_partial_success(monkeypatch):
    owner, request, options = assembly(monkeypatch)
    journal = {}

    def operation(key, kind, call):
        journal[key] = call()
        return journal[key]

    def build():
        if options["chunk_policy"].first_ordinal == 251:
            raise OSError("local storage unavailable")
        return dict(status="indexed", key="first")

    with pytest.raises(OSError, match="local storage"):
        owner.window_indexes(
            request=request, options=options, build=build, operation=operation, key="source"
        )
    assert journal["source:window:0"]["key"] == "first"
    assert "source:window:1" not in journal


@pytest.mark.parametrize(
    "kind", ["local_search", "local_index_plan", "local_index_window", "fetch_group", "llm"]
)
def test_versioned_local_recovery_does_not_replay_unknown_remote(tmp_path, kind):
    import json
    from pathlib import Path

    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    e = fixture(tmp_path)
    policy = json.loads(Path("configs/research/multimodal_definition_gate_v16.json").read_bytes())
    policy.update(
        schema_version="campaign-definition-expansion/1.17.0",
        material_policy="xml-jats-citations-windowed/1.0.0",
    )
    e.policy = read_expansion_policy(policy)
    e.documents.write(
        "definition:operation:0", (dict(key="pending", kind=kind, binding=e.binding),)
    )
    calls = []
    if kind in {"fetch_group", "llm"}:
        with pytest.raises(ProviderError, match="unknown_no_repeat"):
            e.operation("pending", kind, lambda: calls.append(True))
        assert calls == []
    else:
        assert e.operation("pending", kind, lambda: dict(local=True)) == dict(local=True)
        assert len(e.operations()) == 1
        assert e.operation("pending", kind, lambda: calls.append(True)) == dict(local=True)
        assert calls == []


def test_windowed_trial_uses_explicit_new_policy_and_retains_remote_budget(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial
    from tests.evaluation_preparation.test_component_trial import envelope

    value, spec = envelope(
        tmp_path, monkeypatch, "configs/research/multimodal_definition_gate_v17.json"
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.12.0"
    assert value["expansion_policy"]["schema_version"] == "campaign-definition-expansion/1.17.0"
    assert value["expansion_policy"]["max_operations"] == 36
    assert value["expansion_policy"]["max_search_requests"] == 2
    assert value["expansion_policy"]["max_fetch_requests"] == 2
    assert definition_trial.checked(value, spec) == value


def test_window_pool_restores_eight_indexes_but_only_two_documents():
    from tests.evaluation_preparation.test_intent_definition_retrieval import plan

    receipts = [
        dict(key=str(i), status="indexed", source_version_ref=dict(code=str(i // 4)))
        for i in range(8)
    ]
    repo = SimpleNamespace(replay_document_index=lambda key, store: receipts[int(key)])
    source = KnowledgeDefinitionSources(
        repository=repo,
        store=None,
        index_keys=("old-a", "old-b"),
        passage_plans={"q": plan()},
        material_policy="xml-jats-citations-windowed/1.0.0",
    )
    source.restore_indexes(receipts)
    assert len(source.index_keys) == 10
    source.restore_indexes(receipts)
    assert len(source.index_keys) == 10
    old = KnowledgeDefinitionSources(repository=repo, store=None, passage_plans={"q": plan()})
    with pytest.raises(ValueError, match="definition_index_pool_bound"):
        old.restore_indexes(receipts)
    receipts[0]["source_version_ref"] = dict(code="third")
    with pytest.raises(ValueError, match="definition_index_pool_bound"):
        source.restore_indexes(receipts)


def test_small_document_keeps_whole_index_policy(monkeypatch):
    owner, request, options = assembly(monkeypatch, 3)
    policies = []

    def build():
        policies.append(options["chunk_policy"].version)
        return dict(status="indexed", key="whole")

    owner.window_indexes(
        request=request,
        options=options,
        build=build,
        operation=lambda key, kind, call: call(),
        key="source",
    )
    assert policies == ["2.0.0"]


def test_late_oversized_table_is_explicitly_excluded_after_count_overflow(monkeypatch):
    from packages.knowledge_index import parsers

    owner, request, options = assembly(monkeypatch, 262)
    doc = document(262)
    table = doc.blocks[-1].model_copy(
        update={
            "text": "x" * 4000,
            "position": doc.blocks[-1].position.model_copy(
                update={"kind": "table", "span_end": 4000}
            ),
        }
    )
    doc = doc.model_copy(update={"blocks": (*doc.blocks[:-1], table)})
    monkeypatch.setattr(
        parsers,
        "parse",
        lambda *a, **k: SimpleNamespace(document=doc, reason=None, status="parsed"),
    )
    policies = []

    def build():
        policies.append(options["chunk_policy"])
        return dict(status="indexed", key=str(len(policies)))

    _, notes = owner.window_indexes(
        request=request,
        options=options,
        build=build,
        operation=lambda key, kind, call: call(),
        key="source",
    )
    assert len(policies) == 2 and all(p.tables == "excluded" for p in policies)
    assert notes == [dict(reason="table_excluded_from_prose_index")]
    assert doc.blocks[-1] == table

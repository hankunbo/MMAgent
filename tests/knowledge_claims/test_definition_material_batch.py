"""Call-local computation reuse must not become current-source authorization."""

from types import SimpleNamespace

import pytest

from db.knowledge.repository import claims
from db.knowledge.repository.postgres import KnowledgePermissionError
from packages.knowledge_claims import evidence
from packages.knowledge_index.models import ChunkPolicy, canonical
from packages.knowledge_index.parsers import parse


def test_exact_bytes_and_policy_key_pure_computation(monkeypatch):
    document = parse(b"<article><p>First observation.</p></article>", "application/xml").document
    body = canonical(document)
    memo = evidence._DerivationRead()
    parsed = memo.parse(body)
    assert memo.parse(body) is parsed
    calls = []
    original = evidence.rechunk

    def tracked(value, policy):
        calls.append(canonical(policy))
        return original(value, policy)

    monkeypatch.setattr(evidence, "rechunk", tracked)
    policy = ChunkPolicy()
    first = memo.derive(body, parsed, policy)
    assert memo.derive(body, parsed, policy) is first
    assert len(calls) == 1
    # New repository call gets a new computation context, not a global cache.
    other = evidence._DerivationRead()
    assert other.derive(body, other.parse(body), policy) == first
    assert len(calls) == 2
    memo.derive(body, parsed, ChunkPolicy(max_chars=1000))
    assert len(calls) == 3  # Same bytes with a different chunk policy is not reusable.
    changed = parse(b"<article><p>Another observation.</p></article>", "application/xml").document
    changed_body = canonical(changed)
    assert memo.parse(changed_body) != parsed
    memo.derive(changed_body, memo.parse(changed_body), policy)
    assert len(calls) == 4
    with pytest.raises(evidence.Rejected, match="parse_canonical"):
        memo.parse(body + b" ")


@pytest.mark.parametrize("hits", [[], [{}] * 17, iter([{}]), None])
def test_batch_rejects_unbounded_input_before_connect(hits):
    repository = SimpleNamespace(_role="mmagent_knowledge_editor")
    with pytest.raises(ValueError, match="definition_material_batch_bound"):
        claims.read_definition_materials(repository, hits, None, filters={})


def test_batch_requires_original_editor():
    repository = SimpleNamespace(_role="mmagent_knowledge_reader")
    with pytest.raises(KnowledgePermissionError):
        claims.read_definition_materials(repository, [{}], None, filters={})


def test_source_owner_batch_order_bounds_and_fresh_calls():
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources

    calls = []

    def read(hits, store, *, filters):
        calls.append(list(hits))
        return list(hits)

    owner = KnowledgeDefinitionSources(
        repository=SimpleNamespace(read_definition_materials=read), store=None, index_keys=()
    )
    owner.hits = {str(i): {"marker": i} for i in range(17)}
    codes = list(owner.hits)
    assert owner.read_many(codes) == list(owner.hits.values())
    assert list(map(len, calls)) == [16, 1]
    assert owner.read_many(codes) == list(owner.hits.values())
    assert list(map(len, calls)) == [16, 1, 16, 1]
    with pytest.raises(ValueError, match="definition_source_not_acquired"):
        owner.read_many(["unknown"])
    assert len(calls) == 4
    assert owner.read_many([]) == []

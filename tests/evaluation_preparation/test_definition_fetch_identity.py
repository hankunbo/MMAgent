"""CPU-only real typed Knowledge records; repository I/O replaced by exact maps."""

from types import SimpleNamespace

import pytest

from db.knowledge import models as m
from db.knowledge.repository import KnowledgeConflictError
from db.knowledge.repository.archive import _conflicts
from db.knowledge.repository.index import rank_definition_discovery_hits
from tests.database.knowledge_support_04a import candidate_records


def identities(**values):
    return tuple(
        m.SourceIdentity(identifier_type=m.SourceIdentifierType(k), value=v)
        for k, vs in values.items()
        for v in vs
    )


def changed(record, **values):
    raw = record.model_dump(mode="python", exclude={"checksum"})
    return m.seal(type(record), **{**raw, **values})


def setup(known_ids, hit_ids):
    records, indexes = {}, {}
    for i, ids in enumerate(known_ids):
        base = candidate_records("RANK" + str(i))
        source = changed(base["source"], identifiers=ids)
        version = changed(base["source_version"], source_ref=source.reference())
        records[source.code] = source
        records[version.code] = version
        indexes["index" + str(i)] = dict(
            source_version_ref=version.reference().model_dump(mode="json")
        )
    base = candidate_records("RANKHIT")
    hit = changed(base["hit"], identifiers=hit_ids)
    records[hit.code] = hit
    calls = []

    def get(ref):
        calls.append(ref)
        r = records[ref.code]
        assert r.reference() == ref
        return SimpleNamespace(record=r, withdrawn=False, status=m.KnowledgeStatus.DRAFT)

    repo = SimpleNamespace(get=get, replay_document_index=lambda key, store: indexes[key])
    return repo, [dict(reference=hit.reference().model_dump(mode="json"))], tuple(indexes), calls


def test_multivalued_pmid_cannot_be_called_already_acquired():
    old = identities(PMID=["101"])
    new = identities(PMID=["101", "202"])
    assert _conflicts(old, new)
    repo, hits, keys, _ = setup([old], new)
    value = rank_definition_discovery_hits(repo, hits, keys, None)
    assert value["decisions"][0]["already_acquired_identity"] is False


def test_split_identifier_match_to_two_sources_is_ambiguous_not_acquired():
    known = [identities(PMID=["101"]), identities(DOI=["10.9999/different-source"])]
    new = identities(PMID=["101"], DOI=["10.9999/different-source"])
    assert all(not _conflicts(ids, new) for ids in known)
    repo, hits, keys, _ = setup(known, new)
    value = rank_definition_discovery_hits(repo, hits, keys, None)
    assert value["decisions"][0]["already_acquired_identity"] is False


def test_exact_single_source_identity_is_ordering_only_and_current_each_call():
    ids = identities(PMID=["101"], DOI=["10.9999/exact"])
    repo, hits, keys, calls = setup([ids], ids)
    first = rank_definition_discovery_hits(repo, hits, keys, None)
    assert first["decisions"][0]["already_acquired_identity"] is True
    assert first["eligibility_granted"] is False
    n = len(calls)
    rank_definition_discovery_hits(repo, hits, keys, None)
    assert len(calls) == 2 * n


def test_current_withdrawal_error_propagates_not_empty_ranking():
    repo, hits, keys, _ = setup([identities(PMID=["101"])], identities(PMID=["102"]))
    original = repo.get

    def withdrawn(ref):
        snapshot = original(ref)
        snapshot.withdrawn = True
        return snapshot

    repo.get = withdrawn
    with pytest.raises(KnowledgeConflictError, match="unavailable"):
        rank_definition_discovery_hits(repo, hits, keys, None)

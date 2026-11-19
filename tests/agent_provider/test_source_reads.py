"""Original material reader retains lifecycle and exact-byte checks at scope exit."""

from copy import deepcopy

import pytest

from packages.evidence_details import accepted_read_scope as scoped
from packages.grounding.models import digest
from packages.multimodal_research.material import MedicalSources
from tests.agent_provider.test_current_source_batch import fixture


def sources(monkeypatch):
    from packages.evaluation_preparation import unified_requirements

    reader, records, values, calls = fixture()
    reader.chunks[0]["id"] = "source"
    owner = object.__new__(MedicalSources)
    owner.reader = reader
    owner.catalog = {"version": "synthetic:1", "sources": deepcopy(reader.chunks)}
    monkeypatch.setattr(unified_requirements, "catalog", lambda: owner.catalog)
    return owner, records, values, calls


def test_repeated_history_reads_share_first_but_final_read_is_fresh(monkeypatch):
    owner, _, _, calls = sources(monkeypatch)
    with scoped.historical_fact_read_scope():
        first = owner.read()
        for _ in range(12):
            assert owner.read() == first
        assert len(calls) == 3
    assert len(calls) == 6
    assert scoped._facts.get() is None
    assert owner.read() == first
    assert len(calls) == 9  # Dispatch-facing calls never reuse a past scope.


def test_withdrawal_after_first_read_rejected_before_scope_returns(monkeypatch):
    owner, records, values, calls = sources(monkeypatch)
    with pytest.raises(ValueError, match="source_withdrawn"):
        with scoped.historical_fact_read_scope():
            owner.read()
            values[digest(records["source"].reference())].withdrawn = True
            owner.read()
    assert len(calls) == 6 and scoped._facts.get() is None


def test_bytes_changed_without_reader_identity_change_rejected(monkeypatch):
    owner, _, _, _ = sources(monkeypatch)
    original = owner.reader.knowledge_objects.read_bounded
    data = [original()]
    owner.reader.knowledge_objects.read_bounded = lambda *args: data[0]
    with pytest.raises(ValueError, match="source_object_changed"):
        with scoped.historical_fact_read_scope():
            owner.read()
            data[0] = b"changed"
    assert scoped._facts.get() is None


def test_mutating_returned_mapping_does_not_change_the_next_read(monkeypatch):
    owner, _, _, calls = sources(monkeypatch)
    with scoped.historical_fact_read_scope():
        expected = owner.read()
        first = owner.read()
        first.clear()
        assert owner.read() == expected
    assert len(calls) == 6


def test_reader_binding_change_rejected(monkeypatch):
    owner, _, _, _ = sources(monkeypatch)
    with pytest.raises(ValueError, match="historical_source_owner_changed"):
        with scoped.historical_fact_read_scope():
            owner.read()
            owner.reader.chunks[0]["locator"] = "changed"
            owner.read()
    assert scoped._facts.get() is None


def test_catalog_change_and_technical_error_keep_original_errors(monkeypatch):
    from packages.evaluation_preparation import unified_requirements

    owner, _, _, _ = sources(monkeypatch)
    with pytest.raises(ValueError, match="multimodal_medical_catalog_changed"):
        with scoped.historical_fact_read_scope():
            owner.read()
            monkeypatch.setattr(unified_requirements, "catalog", lambda: {"changed": True})
    assert scoped._facts.get() is None
    owner, _, _, _ = sources(monkeypatch)

    def fail(*args):
        raise OSError("synthetic_source_io")

    owner.reader.knowledge_objects.read_bounded = fail
    with pytest.raises(OSError, match="synthetic_source_io"):
        with scoped.historical_fact_read_scope():
            owner.read()
    assert scoped._facts.get() is None


def test_nested_history_is_one_scope_and_next_scope_starts_fresh(monkeypatch):
    owner, _, _, calls = sources(monkeypatch)
    with scoped.historical_fact_read_scope():
        owner.read()
        with scoped.historical_fact_read_scope():
            owner.read()
        assert len(calls) == 3
    assert len(calls) == 6
    with scoped.historical_fact_read_scope():
        owner.read()
    assert len(calls) == 12


def test_different_owner_instances_do_not_share_source_snapshots(monkeypatch):
    one, _, _, a = sources(monkeypatch)
    two, _, _, b = sources(monkeypatch)
    with scoped.historical_fact_read_scope():
        assert one.read() == two.read()
        one.read()
        two.read()
    assert len(a) == len(b) == 6


def test_more_than_64_source_owners_fall_back_to_fresh_reads(monkeypatch):
    from copy import copy

    owner, _, _, calls = sources(monkeypatch)
    owners = [copy(owner) for _ in range(65)]
    with scoped.historical_fact_read_scope():
        for item in owners:
            item.read()
        assert len(scoped._facts.get().sources) == 64
        assert len(calls) == 195
        owners[-1].read()
        assert len(calls) == 198
    assert len(calls) == 390


def test_recursive_source_reader_is_rejected_and_scope_cleared(monkeypatch):
    owner, _, _, _ = sources(monkeypatch)
    owner._read_current = owner.read
    with pytest.raises(ValueError, match="historical_source_cycle"):
        with scoped.historical_fact_read_scope():
            owner.read()
    assert scoped._facts.get() is None

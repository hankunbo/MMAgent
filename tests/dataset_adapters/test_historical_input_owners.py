"""Shared history edges retain input owners, not saved results or authority."""

from types import SimpleNamespace

import pytest

from packages.dataset_adapters import current_read_scope as scoped
from packages.dataset_adapters.service import DatasetPreparationService, VisibleInputReader
from tests.dataset_adapters.test_current_read_scope import Manifest


def factory_fixture(monkeypatch):
    state = SimpleNamespace(factories=0, reads=0, version=1)

    def read(self, code):
        state.reads += 1
        assert self.repository.store is self.store
        return Manifest(version=state.version, details={"case": code})

    monkeypatch.setattr(VisibleInputReader, "read", read)

    def factory():
        state.factories += 1
        store = object()
        repository = SimpleNamespace(store=store)
        owner = object.__new__(DatasetPreparationService)
        owner.repository, owner.visible_store = repository, store
        return object(), store, owner, object()

    return factory, state


def test_shared_ancestry_uses_one_original_owner_and_fresh_final_input(monkeypatch):
    factory, state = factory_fixture(monkeypatch)
    with scoped._projection_visible_reads():
        for _ in range(24):
            with scoped._projection_visible_reads():
                owners = scoped._historical_input_owners(factory)
                assert owners[2].read_visible("CASE:ONE").version == 1
        assert state.factories == state.reads == 1
    assert state.factories == 1 and state.reads == 2
    assert scoped._scope.get() is None and scoped._history_owners.get() is None
    assert scoped._historical_input_owners(factory) is not owners
    with scoped._projection_visible_reads():
        assert scoped._historical_input_owners(factory) is not owners
    assert state.factories == 3


def test_shared_owner_does_not_mask_current_input_change(monkeypatch):
    factory, state = factory_fixture(monkeypatch)
    with pytest.raises(ValueError, match="projection_input_changed_before_return"):
        with scoped._projection_visible_reads():
            owners = scoped._historical_input_owners(factory)
            owners[2].read_visible("CASE:ONE")
            state.version = 2
            assert scoped._historical_input_owners(factory) is owners
    assert state.reads == 2
    assert scoped._history_owners.get() is None and scoped._scope.get() is None


@pytest.mark.parametrize("change", ["factory", "repository", "visible_store"])
def test_owner_or_factory_replacement_rejected(monkeypatch, change):
    factory, _ = factory_fixture(monkeypatch)
    with pytest.raises(ValueError, match="projection_history_owner_changed"):
        with scoped._projection_visible_reads():
            owners = scoped._historical_input_owners(factory)
            if change == "factory":
                scoped._historical_input_owners(lambda: owners)
            else:
                setattr(owners[2], change, object())
                scoped._historical_input_owners(factory)
    assert scoped._history_owners.get() is None


def test_distinct_input_limit_is_not_raised_or_bypassed(monkeypatch):
    factory, state = factory_fixture(monkeypatch)
    with pytest.raises(ValueError, match="projection_input_read_bound"):
        with scoped._projection_visible_reads():
            owner = scoped._historical_input_owners(factory)[2]
            for n in range(17):
                owner.read_visible(f"CASE:{n}")
    assert state.factories == 1 and state.reads == 16
    assert scoped._history_owners.get() is None and scoped._scope.get() is None


def test_original_factory_exception_is_preserved_and_scope_cleared():
    def fail():
        raise OSError("original_owner_unavailable")

    with pytest.raises(OSError, match="original_owner_unavailable"):
        with scoped._projection_visible_reads():
            scoped._historical_input_owners(fail)
    assert scoped._history_owners.get() is None and scoped._scope.get() is None

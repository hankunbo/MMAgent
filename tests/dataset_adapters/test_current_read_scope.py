"""Single synchronous proof reuse retains original owner revalidation on return."""

from types import SimpleNamespace

import pytest

from packages.dataset_adapters.current_read_scope import _projection_visible_reads
from packages.dataset_adapters.service import DatasetPreparationService, VisibleInputReader
from packages.grounding.models import Value


class Manifest(Value):
    version: int
    details: dict[str, str]


def reader(monkeypatch):
    state = SimpleNamespace(calls=0, value=Manifest(version=1, details={"input": "original"}))
    owner = object.__new__(DatasetPreparationService)
    owner.repository, owner.visible_store = object(), object()

    def read(self, code):
        assert self.repository is owner.repository and self.store is owner.visible_store
        assert code == "CASE:ONE"
        state.calls += 1
        return state.value.model_copy(deep=True)

    monkeypatch.setattr(VisibleInputReader, "read", read)
    return owner, state


def test_twenty_reads_become_first_and_last_full_owner_checks(monkeypatch):
    owner, state = reader(monkeypatch)
    expected = owner.read_visible("CASE:ONE")
    state.calls = 0
    with _projection_visible_reads():
        for _ in range(20):
            assert owner.read_visible("CASE:ONE") == expected
        assert state.calls == 1
    assert state.calls == 2
    assert owner.read_visible("CASE:ONE") == expected and state.calls == 3
    with _projection_visible_reads():
        assert owner.read_visible("CASE:ONE") == expected
    assert state.calls == 5


def test_return_rechecks_change_and_does_not_leak_or_mutate_cached_input(monkeypatch):
    owner, state = reader(monkeypatch)
    with pytest.raises(ValueError, match="projection_input_changed_before_return"):
        with _projection_visible_reads():
            value = owner.read_visible("CASE:ONE")
            value.details["input"] = "caller mutation"
            assert owner.read_visible("CASE:ONE").details["input"] == "original"
            state.value = Manifest(version=2, details={"input": "current changed"})
    assert owner.read_visible("CASE:ONE").version == 2 and state.calls == 3


def test_owner_replacement_original_error_nested_scope_and_fresh_exit(monkeypatch):
    owner, state = reader(monkeypatch)
    with pytest.raises(ValueError, match="projection_input_owner_changed"):
        with _projection_visible_reads():
            owner.read_visible("CASE:ONE")
            owner.repository = object()
    with pytest.raises(OSError, match="original owner failed"):
        with _projection_visible_reads():
            with _projection_visible_reads():
                owner.read_visible("CASE:ONE")
                raise OSError("original owner failed")
    before = state.calls
    owner.read_visible("CASE:ONE")
    assert state.calls == before + 1
    with _projection_visible_reads():
        owner.read_visible("CASE:ONE")
        with _projection_visible_reads():
            owner.read_visible("CASE:ONE")
        assert state.calls == before + 2
    assert state.calls == before + 3


def test_return_repeats_the_actual_input_integrity_check(monkeypatch):
    owner, state = reader(monkeypatch)
    original = VisibleInputReader.read

    def changed(self, code):
        if state.calls:
            raise ValueError("visible_object_checksum_mismatch")
        return original(self, code)

    monkeypatch.setattr(VisibleInputReader, "read", changed)
    with pytest.raises(ValueError, match="visible_object_checksum_mismatch"):
        with _projection_visible_reads():
            owner.read_visible("CASE:ONE")
            owner.read_visible("CASE:ONE")


def component_reader(monkeypatch):
    owner, state = reader(monkeypatch)
    state.component_calls = 0
    state.stored = {"chunks": ["original"]}

    def read_components(self, code):
        assert self.repository is owner.repository and self.store is owner.visible_store
        assert code == "CASE:ONE"
        from copy import deepcopy

        state.component_calls += 1
        return state.value.model_copy(deep=True), deepcopy(state.stored)

    monkeypatch.setattr(VisibleInputReader, "read_components", read_components)
    return owner, state


def test_component_reads_have_separate_identity_and_two_full_checks(monkeypatch):
    owner, state = component_reader(monkeypatch)
    with _projection_visible_reads():
        for _ in range(12):
            assert owner.read_visible("CASE:ONE") == state.value
            visible, stored = owner.read_visible_components("CASE:ONE")
            assert visible == state.value and stored == state.stored
            stored["chunks"].append("caller-only")
        assert state.calls == state.component_calls == 1
    assert state.calls == state.component_calls == 2
    owner.read_visible_components("CASE:ONE")
    assert state.component_calls == 3


@pytest.mark.parametrize("fault", ["chunk", "owner", "read_error"])
def test_component_return_rechecks_chunks_owner_and_original_error(monkeypatch, fault):
    owner, state = component_reader(monkeypatch)
    message = (
        "projection_input_owner_changed"
        if fault == "owner"
        else (
            "multimodal_chunk_changed"
            if fault == "read_error"
            else "projection_input_changed_before_return"
        )
    )
    with pytest.raises(ValueError, match=message):
        with _projection_visible_reads():
            owner.read_visible_components("CASE:ONE")
            if fault == "owner":
                owner.visible_store = object()
            elif fault == "chunk":
                state.stored["chunks"] = ["changed"]
            else:

                def unavailable(self, code):
                    raise ValueError("multimodal_chunk_changed")

                monkeypatch.setattr(VisibleInputReader, "read_components", unavailable)


def test_independent_history_finishes_each_proof_and_preserves_outer_limit():
    from collections import Counter

    from packages.dataset_adapters.current_read_scope import (
        _independent_projection_visible_reads,
        _read_visible,
    )

    owner = SimpleNamespace(repository=object(), visible_store=object())
    calls = Counter()

    def read(code):
        def actual():
            calls[code] += 1
            return {"case": code}

        return _read_visible(owner, code, actual)

    with _projection_visible_reads():
        for n in range(16):
            read(f"current:{n}")
        for n in range(24):
            with _independent_projection_visible_reads():
                read(f"history:{n}")
                read(f"history:{n}")
            assert calls[f"history:{n}"] == 2
        assert calls["current:0"] == 1
        with pytest.raises(ValueError, match="projection_input_read_bound"):
            read("current:17")
    assert all(v == 2 for v in calls.values())


def test_independent_history_revalidates_and_restores_on_error(monkeypatch):
    from packages.dataset_adapters.current_read_scope import _independent_projection_visible_reads

    owner, state = reader(monkeypatch)
    with _projection_visible_reads():
        owner.read_visible("CASE:ONE")
        original = state.value
        with pytest.raises(ValueError, match="projection_input_changed_before_return"):
            with _independent_projection_visible_reads():
                owner.read_visible("CASE:ONE")
                state.value = Manifest(version=2, details={"input": "changed"})
        state.value = original
        before = state.calls
        owner.read_visible("CASE:ONE")
        assert state.calls == before  # Original scope was restored, not discarded.
    assert state.calls == before + 1  # Caller still performs its fresh return check.

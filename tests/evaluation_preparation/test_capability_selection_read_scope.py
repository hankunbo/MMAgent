"""A current selection proof may be shared only until its fresh return check."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_projection as projection
from packages.evaluation_preparation import generic_capability_selection as selection


def fixture(monkeypatch):
    owner = object.__new__(projection.GenericCapabilityOwner)
    owner.campaign = SimpleNamespace(base=object())
    owner.own_claim = {"claim": "current"}
    owner.snapshot = {"selection": ["original"]}
    owner.policy = {"correspondence": "current"}
    owner.previous_sources = []
    state = {"reads": 0, "snapshot": deepcopy(owner.snapshot), "sources": ["current"]}

    def inspect(campaign, *, own_claim):
        assert campaign is owner.campaign
        state["reads"] += 1
        return deepcopy(state["snapshot"])

    def read_sources():
        return deepcopy(state["sources"])

    owner.medical = SimpleNamespace(read=read_sources)
    monkeypatch.setattr(selection, "inspect", inspect)
    monkeypatch.setattr(
        projection, "json", SimpleNamespace(loads=lambda value: {"correspondence": "current"})
    )
    monkeypatch.setattr(
        projection, "Path", lambda path: SimpleNamespace(read_bytes=lambda: b"public-policy")
    )
    return owner, state


def test_nested_reads_are_bounded_and_do_not_authorize_next_operation(monkeypatch):
    owner, state = fixture(monkeypatch)
    with projection._current_selection_reads():
        first = owner.current()
        first[0]["selection"].append("caller-mutation")
        for _ in range(5):
            with projection._current_selection_reads():
                assert owner.current() == ({"selection": ["original"]}, ["current"])
        assert state["reads"] == 1
    assert state["reads"] == 2  # original read and a fresh owner read before return
    owner.current()
    assert state["reads"] == 3  # no cached qualification outside the operation
    assert projection._selection_reads.get() is None


@pytest.mark.parametrize("change", ["selection", "source", "claim", "owner", "reader"])
def test_return_refuses_changed_source_selection_or_owner(monkeypatch, change):
    owner, state = fixture(monkeypatch)
    with pytest.raises(ValueError, match="generic_capability_.*changed"):
        with projection._current_selection_reads():
            owner.current()
            if change == "selection":
                state["snapshot"]["selection"].append("new")
            elif change == "source":
                state["sources"].append("withdrawn")
            elif change == "claim":
                owner.own_claim["claim"] = "changed"
            elif change == "owner":
                owner.campaign.base = object()
            else:
                owner.medical.read = lambda: ["current"]
    assert projection._selection_reads.get() is None


def test_exception_discards_scope_without_hiding_technical_error(monkeypatch):
    owner, state = fixture(monkeypatch)
    with pytest.raises(OSError, match="technical"):
        with projection._current_selection_reads():
            owner.current()
            raise OSError("technical")
    assert projection._selection_reads.get() is None
    owner.current()
    assert state["reads"] == 2


def test_mutated_binding_is_rejected_before_a_memo_hit(monkeypatch):
    owner, _ = fixture(monkeypatch)
    with pytest.raises(ValueError, match="generic_capability_selection_owner_changed"):
        with projection._current_selection_reads():
            owner.current()
            owner.snapshot["selection"].append("changed")
            owner.current()

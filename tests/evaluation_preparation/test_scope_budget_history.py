"""Survey recovery remains accounted history, never a new medical vote."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import scope_successor
from packages.multimodal_research.campaign_renewal import attempted_acquisition_targets


@pytest.mark.parametrize("fault", [None, "count", "checksum", "input", "missing", "owner", "claim"])
def test_recovered_scope_history_uses_bound_original_owners(tmp_path, monkeypatch, fault):
    previous = tmp_path / "previous"
    proof = dict(previous=str(previous), failure_checksum=digest("failure"), requests=1)
    predecessor = SimpleNamespace(output=previous)
    protocol = dict(
        schema_version=scope_successor.VERSION,
        input=digest("input"),
        task=digest("task"),
        scope_reuse=proof,
    )
    owner = SimpleNamespace(
        protocol=protocol, base=SimpleNamespace(scope_reuse_predecessor=predecessor)
    )
    parent = SimpleNamespace(
        protocol=dict(input=protocol["input"], task=protocol["task"]), parent=owner
    )
    attempt = dict(
        directory=str(previous),
        checksum=proof["failure_checksum"],
        requests=1,
        kind="accepted-scope-local-recovery/1.0.0",
    )
    calls = []

    def inspect(value):
        assert value is predecessor
        calls.append("original-request-and-recovery")
        return dict(proof, requests=2) if fault == "owner" else deepcopy(proof)

    def inherited(value):
        assert value is owner
        calls.append("current-source-and-runtime-claim")
        if fault == "claim":
            raise ValueError("scope_successor_claim_binding")
        return {"input-scope:one": {"disease_votes": 0}}

    monkeypatch.setattr(scope_successor, "inspect", inspect)
    monkeypatch.setattr(scope_successor, "inherited", inherited)
    if fault == "count":
        attempt["requests"] = 0
    elif fault == "checksum":
        attempt["checksum"] = digest("wrong")
    elif fault == "input":
        parent.protocol["input"] = digest("other")
    elif fault == "missing":
        parent.parent = None
    if fault:
        with pytest.raises(ValueError, match="scope_budget_history_|scope_successor_claim_binding"):
            attempted_acquisition_targets(parent, [attempt])
    else:
        before = deepcopy((protocol, attempt))
        assert attempted_acquisition_targets(parent, [attempt]) == set()
        assert calls == ["original-request-and-recovery", "current-source-and-runtime-claim"]
        assert (protocol, attempt) == before


def test_unknown_history_kind_still_rejected(tmp_path):
    parent = SimpleNamespace(protocol={}, parent=None)
    attempt = dict(
        directory=str(tmp_path), checksum=digest("x"), requests=1, kind="unqualified-survey"
    )
    with pytest.raises(ValueError, match="renewal_acquisition_history_kind"):
        attempted_acquisition_targets(parent, [attempt])

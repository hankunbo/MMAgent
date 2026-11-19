from datetime import timedelta

import pytest

from packages.grounding.models import validate
from packages.task_runtime.models import Profile, Receipt
from packages.task_runtime.transitions import change, lease_check
from tests.task_runtime.support import configured


def test_closed_schema_and_bounded_policy():
    with pytest.raises(ValueError):
        Profile(lease_seconds=0)
    with pytest.raises(ValueError):
        Profile(synthetic_only=False)
    _, _, runtime, _, r, _ = configured()
    with pytest.raises(ValueError):
        validate(Receipt, {**r.model_dump(), "executable": "arbitrary"})
    with pytest.raises(ValueError, match="dispatch_lease_state"):
        validate(Receipt, {**r.model_dump(), "state": "leased"})
    with pytest.raises(ValueError, match="dispatch_cancel_monotone"):
        validate(Receipt, {**r.model_dump(), "state": "cancelled"})
    assert r.deadline == runtime.state.binding.deadline


def test_lease_expiry_token_heartbeat_and_cancel_monotone():
    _, _, runtime, _, r, _ = configured()
    now = runtime.state.binding.started_at
    r = change(r, "claim", dict(worker="w", token="t"), runtime.state, now)
    for worker, token, time in (
        ("other", "t", now),
        ("w", "bad", now),
        ("w", "t", now + timedelta(seconds=31)),
    ):
        with pytest.raises(ValueError, match="dispatch_lease_invalid"):
            lease_check(r, worker, token, time)
    with pytest.raises(ValueError, match="dispatch_busy"):
        change(r, "claim", dict(worker="other", token="new"), runtime.state, now)
    r = change(r, "cancel", {}, runtime.state, now)
    assert r.cancel_requested and r.state == "leased"
    r = change(r, "heartbeat", dict(worker="w", token="t"), runtime.state, now)
    assert r.cancel_requested
    r = change(
        r, "claim", dict(worker="other", token="new"), runtime.state, now + timedelta(seconds=31)
    )
    assert r.claims == 2 and r.lease.worker == "other" and r.cancel_requested


def test_failure_budget_preserves_unknown_command():
    owner, _, runtime, _, r, _ = configured()
    now = runtime.state.binding.started_at
    for n in range(r.profile.max_failures):
        r = change(r, "claim", dict(worker="w", token=str(n)), runtime.state, now)
        cmd = owner.next_command(r)
        r = change(r, "prepare", dict(worker="w", token=str(n)), runtime.state, now, command=cmd)
        r = change(r, "failure", dict(worker="w", token=str(n)), runtime.state, now)
        assert r.command == cmd and r.lease is None
    assert r.state == "dead_letter" and r.failures == 3


def test_claim_budget_dead_letter_no_execution():
    _, _, runtime, domain, r, _ = configured()
    r = validate(Receipt, {**r.model_dump(), "claims": r.profile.max_claims})
    r = change(
        r, "claim", dict(worker="w", token="t"), runtime.state, runtime.state.binding.started_at
    )
    assert r.state == "dead_letter" and domain.calls == []

"""Pure finite queue transitions; original Planner controls all execution facts."""

from datetime import timedelta

from packages.grounding.models import digest, validate
from packages.planner_control.models import Phase

from .models import TERMINAL, Lease, Receipt


def lease_check(receipt, worker, token, now, *, allow_expired=False):
    lease = receipt.lease
    if (
        receipt.state != "leased"
        or lease is None
        or lease.worker != worker
        or lease.token != token
        or (not allow_expired and lease.expires_at <= now)
    ):
        raise ValueError("dispatch_lease_invalid")


def change(old, action, parameters, state, now, *, command=None):
    values = old.model_dump()
    values.update(revision=old.revision + 1, control_checksum=digest(state))
    if action == "cancel":
        if old.state in TERMINAL:
            return old
        values.update(cancel_requested=True, reason="cancel_requested")
    elif action == "claim":
        if old.state in TERMINAL:
            raise ValueError("dispatch_terminal")
        if old.lease and old.lease.expires_at > now:
            raise ValueError("dispatch_busy")
        if old.claims >= old.profile.max_claims:
            values.update(state="dead_letter", lease=None, reason="claim_budget")
        else:
            values.update(
                state="leased",
                claims=old.claims + 1,
                lease=Lease(
                    worker=parameters["worker"],
                    token=parameters["token"],
                    expires_at=now + timedelta(seconds=old.profile.lease_seconds),
                ),
                reason="claimed_for_confirmation" if old.command else "claimed",
            )
    elif action in {"heartbeat", "prepare", "confirm", "failure"}:
        lease_check(
            old,
            parameters["worker"],
            parameters["token"],
            now,
            allow_expired=action in {"confirm", "failure"},
        )
        if action == "heartbeat":
            values["lease"] = old.lease.model_copy(
                update={"expires_at": now + timedelta(seconds=old.profile.lease_seconds)}
            )
        elif action == "prepare":
            values.update(command=command, reason="command_prepared")
        elif action == "failure":
            # Unknown technical failure never implies work was not started. Keep exact command.
            count = old.failures + 1
            values.update(
                failures=count,
                lease=None,
                state="dead_letter" if count >= old.profile.max_failures else "confirming",
                reason="technical_failure_reconcile_same_command",
            )
        else:
            values.update(lease=None)
            if state.phase == Phase.STOPPED:
                values.update(
                    state="cancelled" if state.cancelled else "completed",
                    cancel_requested=old.cancel_requested or state.cancelled,
                    command=None,
                    reason=state.reason or "planner_stopped",
                )
            elif state.pending:
                values.update(state="confirming", command=state.pending.command, reason="pending")
            else:
                # A returned step must actually have been confirmed by the original owner.
                if old.command is not None and not any(
                    e.command == old.command for e in state.events
                ):
                    raise ValueError("dispatch_confirmation_missing")
                values.update(state="queued", command=None, reason="step_confirmed")
    else:
        raise ValueError("dispatch_action")
    return validate(Receipt, values)

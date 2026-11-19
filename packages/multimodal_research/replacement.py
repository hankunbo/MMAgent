"""Explicit single replacement of cancelled read-only inference, never tool recovery."""

from threading import Event

from agents.planner.service import Planner
from packages.grounding.models import digest


def review_replaceable(phase, control, events):
    completed = [e["record"] for e in events if e.get("phase") == "completed"]
    reserved = sum(e.get("phase") == "reserved" for e in events)
    return bool(
        phase["target"] is None
        and not phase["accepted"]
        and phase.get("acceptance_key") is None
        and control.phase == "stopped"
        and control.cancelled
        and control.pending is None
        and completed
        and len(completed) == reserved
        and all(
            r["error"] in {"timeout", "transport"} and r["api_requests"] == 1 for r in completed
        )
    )


def close_failed_review(runtime, result):
    if result["target"] is not None or result["accepted"]:
        return None
    planner = Planner(runtime=runtime, owner=None)
    before = planner.read(result["run_id"])
    pending = before.pending
    if (
        pending is None
        or pending.command.action != "review"
        or pending.launch_claimed
        or pending.reserved
    ):
        return None
    cancel = Event()
    cancel.set()
    event = planner.step(result["run_id"], pending.command, cancel=cancel)
    after = planner.read(result["run_id"])
    if after.phase != "stopped" or after.pending is not None or not after.cancelled:
        raise ValueError("multimodal_review_cancellation_unconfirmed")
    return dict(
        schema_version="multimodal-review-cancellation/1.0.0",
        before=digest(before),
        after=digest(after),
        event=event.model_dump(mode="json"),
        cancelled=True,
        new_requests=0,
        remote_response_unknown_preserved=True,
    )

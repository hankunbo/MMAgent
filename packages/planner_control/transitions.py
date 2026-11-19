"""Persistence boundary validates transitions as well as individual control envelopes."""

from .models import Control, Phase, SelectionOutcome, SelectionPending, request_checksum

ALLOWED = {
    Phase.INITIALIZE: {"load"},
    Phase.RETRIEVE: {"retrieve"},
    Phase.INSTANTIATE: {"instantiate"},
    Phase.IDENTIFY: {"identify"},
    Phase.SELECT: {"select", "review"},
    Phase.PREPARE: {"prepare"},
    Phase.EXECUTE: {"execute", "recover"},
    Phase.VERIFY: {"verify"},
    Phase.REVIEW: {"review"},
    Phase.FALLBACK: {"fallback", "review"},
    Phase.STOPPED: set(),
}
NEXT = {
    "load": Phase.RETRIEVE,
    "retrieve": Phase.INSTANTIATE,
    "instantiate": Phase.IDENTIFY,
    "identify": Phase.SELECT,
    "select": Phase.PREPARE,
    "prepare": Phase.EXECUTE,
    "execute": Phase.VERIFY,
    "recover": Phase.VERIFY,
    "verify": Phase.REVIEW,
    "review": Phase.STOPPED,
    "fallback": Phase.SELECT,
}


def validate_checkpoint(old: Control | None, new: Control):
    if old is None:
        if new != Control(binding=new.binding, graph=new.binding.graph):
            raise ValueError("initial_control_required")
        return
    if new.binding != old.binding or new.revision != old.revision + 1:
        raise ValueError("immutable_control_binding")
    if old.phase == Phase.STOPPED:
        raise ValueError("terminal_control_immutable")
    if old.cancelled and not new.cancelled:
        raise ValueError("cancellation_cannot_clear")
    if new.events[: len(old.events)] != old.events:
        raise ValueError("history_immutable")
    if len(new.events) == len(old.events):
        if new.phase != old.phase or new.graph != old.graph:
            raise ValueError("transition_requires_event")
        if (
            new.selected,
            new.candidates,
            new.attempted,
            new.branches,
            new.result_refs,
            new.started,
            new.confirmed,
            new.gaps,
            new.steps,
        ) != (
            old.selected,
            old.candidates,
            old.attempted,
            old.branches,
            old.result_refs,
            old.started,
            old.confirmed,
            old.gaps,
            old.steps,
        ):
            raise ValueError("fact_change_requires_event")
        pending = new.pending
        if isinstance(old.pending, SelectionPending) and (
            not isinstance(pending, SelectionPending) or pending.decision != old.pending.decision
        ):
            raise ValueError("pending_selection_immutable")
        if isinstance(pending, SelectionPending):
            from .actions import state_checksum

            if (
                pending.decision.action_set.binding != old.binding
                or pending.decision.action_set.graph != old.graph
                or pending.decision.action_set.state_checksum != state_checksum(old)
            ):
                raise ValueError("pending_selection_scope")
        if pending:
            action = pending.command.action
            if pending.request_checksum != request_checksum(new.binding, pending.command):
                raise ValueError("pending_checksum")
            if action not in ALLOWED[old.phase] | {"stop", "cancel"}:
                raise ValueError("illegal_state_action")
            if old.pending:
                if (pending.command, pending.request_checksum, pending.execution_key) != (
                    old.pending.command,
                    old.pending.request_checksum,
                    old.pending.execution_key,
                ):
                    raise ValueError("pending_binding_immutable")
                if old.pending.launch_claimed and not pending.launch_claimed:
                    raise ValueError("launch_claim_cannot_clear")
                if old.pending.reserved and not pending.reserved:
                    raise ValueError("reservation_cannot_clear")
            elif pending.command.expected_revision != old.revision:
                raise ValueError("pending_revision")
            if pending.launch_claimed and action not in {"execute", "recover"}:
                raise ValueError("launch_claim_phase")
            reservation = int(pending.reserved and not (old.pending and old.pending.reserved))
            if reservation and action != "prepare":
                raise ValueError("reservation_phase")
            if new.reserved != old.reserved + reservation:
                raise ValueError("reservation_count")
        elif old.pending or new.reserved != old.reserved:
            raise ValueError("pending_cannot_disappear")
        return
    if len(new.events) != len(old.events) + 1 or new.pending is not None or old.pending is None:
        raise ValueError("single_confirmation_required")
    event = new.events[-1]
    if (event.command, event.request_checksum, event.before, event.after, event.graph) != (
        old.pending.command,
        old.pending.request_checksum,
        old.phase,
        new.phase,
        new.graph,
    ):
        raise ValueError("event_binding_mismatch")
    if isinstance(event.outcome, SelectionOutcome):
        from .actions import state_checksum

        decision = event.outcome.decision
        if not isinstance(old.pending, SelectionPending) or old.pending.decision != decision:
            raise ValueError("selection_requires_durable_advice")
        if event.command.action != "select" or (
            decision.action_set.binding != old.binding
            or decision.action_set.graph != old.graph
            or decision.action_set.state_checksum != state_checksum(old)
            or decision.decided_at > event.at
            or new.selected != decision.selected.candidate
        ):
            raise ValueError("selection_checkpoint_mismatch")
        if decision.selected.candidate is not None and (
            new.selected not in old.candidates or new.selected.branch in old.attempted
        ):
            raise ValueError("selection_not_owner_admitted")
    if event.outcome.status == "unknown":
        raise ValueError("unknown_is_not_confirmation")
    if new.graph != old.graph:
        if new.phase != Phase.RETRIEVE or new.selected or new.candidates or new.result_refs:
            raise ValueError("graph_change_requires_full_revalidation")
    elif new.phase != Phase.STOPPED:
        expected = Phase.FALLBACK if event.outcome.status != "ok" else NEXT[event.command.action]
        if new.phase != expected:
            raise ValueError("illegal_state_transition")
    if (new.reserved, new.started, new.confirmed) != (
        old.reserved,
        old.started + event.outcome.started,
        old.confirmed + event.outcome.confirmed,
    ):
        raise ValueError("confirmation_counters")

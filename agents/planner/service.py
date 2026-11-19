"""Deterministic single-step scheduler. No SQL, graph writes, or model/process calls."""

from datetime import UTC, datetime

from packages.grounding.models import digest, validate
from packages.planner_control.models import (
    ActionDecision,
    Binding,
    Candidate,
    Command,
    Control,
    Event,
    Gap,
    Graph,
    Outcome,
    Pending,
    Phase,
    SelectionOutcome,
    SelectionPending,
    request_checksum,
)
from packages.planner_control.transitions import ALLOWED, NEXT

from .ports import UNREADY, EmptyMemory, RegisteredActions


class Planner:
    def __init__(self, *, runtime, owner, actions=None, ranking=None, clock=None):
        self.runtime, self.owner = runtime, owner
        self.actions = actions or RegisteredActions()
        self.ranking = ranking
        self.clock = clock or (lambda: datetime.now(UTC))
        self.memory = EmptyMemory()

    def begin(self, binding):
        binding = validate(Binding, binding)
        if validate(Graph, self.owner.current(binding)) != binding.graph:
            raise ValueError("initial_graph_version_mismatch")
        return self.runtime.begin(binding)

    def read(self, run_id):
        return validate(Control, self.runtime.read(run_id))

    def require_agent_work(self, binding):
        """Current original Control gates new requests and new candidate acceptance.

        Public probes share the run's control even though their case is PUBLIC_PROBE.
        This read owns no network lifetime or second control state.
        """
        state = self.read(binding.run_id)
        if state.binding != binding:
            raise ValueError("agent_control_binding")
        reason = self.research_review_limit(state)
        if reason:
            raise ValueError(reason)
        if state.phase == Phase.STOPPED:
            raise ValueError("stopped")
        if state.phase in {Phase.PREPARE, Phase.EXECUTE, Phase.VERIFY}:
            raise ValueError("agent_control_requires_confirmation")
        if state.pending and (
            state.pending.launch_claimed
            or state.pending.reserved
            or state.pending.command.action not in {"select", "review"}
            or state.phase not in {Phase.SELECT, Phase.REVIEW, Phase.FALLBACK}
        ):
            raise ValueError("agent_control_requires_confirmation")
        return state

    def trace(self, run_id):
        return self.read(run_id).events

    def replay(self, run_id):
        """Replay control events only; no provider calls or medical graph reconstruction."""
        state = self.read(run_id)
        phase = Phase.INITIALIZE
        for event in state.events:
            if event.before != phase or event.request_checksum != request_checksum(
                state.binding, event.command
            ):
                raise ValueError("control_replay_mismatch")
            phase = event.after
        if phase != state.phase:
            raise ValueError("control_replay_phase")
        return state

    def report(self, run_id):
        from packages.planner_control.models import ControlReport

        state = self.replay(run_id)
        return ControlReport(
            run_id=run_id,
            control_checksum=digest(state),
            phase=state.phase,
            reason=state.reason,
            graph=state.graph,
            deadline=state.binding.deadline,
            pending_execution_key=state.pending.execution_key if state.pending else None,
            reserved_actions=state.reserved,
            started_processes=state.started,
            confirmed_workflows=state.confirmed,
            result_refs=state.result_refs,
            unresolved=state.gaps,
        )

    def begin_research_review(self, run_id, command, *, cancel=None):
        """Claim the original review command, releasing its session before any model call."""
        return self._begin_review(run_id, command, cancel=cancel, phases={Phase.REVIEW})

    def begin_recovery_review(self, run_id, command, *, cancel=None):
        """09C claims the original review pending at a failed/exhausted choice point."""
        return self._begin_review(
            run_id, command, cancel=cancel, phases={Phase.SELECT, Phase.FALLBACK, Phase.REVIEW}
        )

    def _begin_review(self, run_id, command, *, cancel, phases):
        from packages.planner_control.models import Reference

        command = validate(Command, command)
        if command.action != "review":
            raise ValueError("review_command_required")
        with self.runtime.session(run_id):
            state = self.read(run_id)
            sha = request_checksum(state.binding, command)
            prior = next((e for e in state.events if e.command.key == command.key), None)
            if prior:
                if prior.request_checksum != sha:
                    raise ValueError("step_key_content_conflict")
            elif state.pending:
                if state.pending.request_checksum != sha:
                    raise ValueError("pending_step_requires_same_command")
            else:
                if state.phase not in phases or command.expected_revision != state.revision:
                    raise ValueError("review_control_state")
                state = self._save(
                    state,
                    cancelled=state.cancelled or bool(cancel and cancel.is_set()),
                    reason=self.research_review_limit(state),
                    pending=Pending(
                        command=command,
                        request_checksum=sha,
                        execution_key="planner:" + digest(dict(run=run_id, key=command.key))[7:55],
                    ),
                )
            event_ref = Reference(
                role="review_control",
                code="review:" + sha[7:],
                version="planner-research-review/1.0.0",
                checksum=sha,
            )
        if cancel is not None:
            self.observe_research_control(run_id, event_ref, cancel=cancel)
        return event_ref

    def claim_recovery_fallback(self, run_id, command, *, cancel=None):
        """09C journals the original fallback pending before its additive documents."""
        command = validate(Command, command)
        if command.action != "fallback":
            raise ValueError("fallback_command_required")
        with self.runtime.session(run_id):
            state = self.read(run_id)
            sha = request_checksum(state.binding, command)
            prior = next((e for e in state.events if e.command.key == command.key), None)
            if prior:
                if prior.request_checksum != sha:
                    raise ValueError("step_key_content_conflict")
                return state
            if state.pending:
                if state.pending.request_checksum != sha:
                    raise ValueError("pending_step_requires_same_command")
                return state
            if state.phase != Phase.FALLBACK or command.expected_revision != state.revision:
                raise ValueError("fallback_control_state")
            return self._save(
                state,
                cancelled=state.cancelled or bool(cancel and cancel.is_set()),
                pending=Pending(
                    command=command,
                    request_checksum=sha,
                    execution_key="planner:" + digest(dict(run=run_id, key=command.key))[7:55],
                ),
            )

    def observe_recovery_cancel(self, run_id, command=None, *, cancel=None):
        """Observe outer I/O cancellation without clearing an unknown pending."""
        command = validate(Command, command) if command is not None else None
        with self.runtime.session(run_id):
            state = self.read(run_id)
            sha = request_checksum(state.binding, command) if command else None
            prior = next((e for e in state.events if e.request_checksum == sha), None)
            cancellation_key = "cancel:" + sha[7:] if sha else None
            if prior and state.pending and state.pending.command.key == cancellation_key:
                self._finish(
                    state, Outcome(reason="cancelled"), phase=Phase.STOPPED, cancelled=True
                )
                return self.read(run_id)
            if state.pending and state.pending.request_checksum != sha:
                raise ValueError("recovery_control_binding")
            if (
                prior
                and not state.pending
                and cancel
                and cancel.is_set()
                and state.phase != Phase.STOPPED
            ):
                cancellation = Command(
                    key=cancellation_key, expected_revision=state.revision, action="cancel"
                )
                state = self._save(
                    state,
                    cancelled=True,
                    pending=Pending(
                        command=cancellation,
                        request_checksum=request_checksum(state.binding, cancellation),
                        execution_key="planner:cancel:" + sha[7:55],
                    ),
                )
                self._finish(
                    state, Outcome(reason="cancelled"), phase=Phase.STOPPED, cancelled=True
                )
                return self.read(run_id)
            if (
                command
                and not state.pending
                and not any(e.request_checksum == sha for e in state.events)
                and command.expected_revision != state.revision
            ):
                raise ValueError("recovery_control_binding")
            if cancel and cancel.is_set() and not state.cancelled and state.phase != Phase.STOPPED:
                state = self._save(state, cancelled=True)
            return state

    def research_review_limit(self, state):
        """Only gates new work; historical facts may always be confirmed."""
        if state.cancelled:
            return "cancelled"
        if self.clock() >= state.binding.deadline:
            return "deadline"
        if state.steps >= state.binding.profile.max_steps:
            return "step_budget"
        if state.no_progress >= state.binding.profile.max_no_progress:
            return "no_new_information"
        return None

    def observe_research_control(self, run_id, event_ref, *, cancel=None):
        """Persist cancellation without consuming/rewriting the pending review fact.

        If the review confirmation committed before its reply was lost, cancellation
        is a subsequent original control event, never a second review or new run.
        """
        with self.runtime.session(run_id):
            state = self.read(run_id)
            prior = next(
                (e for e in state.events if e.request_checksum == event_ref.checksum), None
            )
            if not prior and (
                state.pending is None or state.pending.request_checksum != event_ref.checksum
            ):
                raise ValueError("review_control_binding")
            cancellation_key = "cancel:" + event_ref.checksum[7:]
            if prior and state.pending and state.pending.command.key == cancellation_key:
                self._finish(
                    state,
                    Outcome(reason="cancelled", refs=prior.outcome.refs),
                    phase=Phase.STOPPED,
                    cancelled=True,
                )
                return self.read(run_id)
            if cancel and cancel.is_set() and not state.cancelled and state.phase != Phase.STOPPED:
                if prior:
                    if state.pending is not None:
                        raise ValueError("pending_step_requires_same_command")
                    command = Command(
                        key=cancellation_key,
                        expected_revision=state.revision,
                        action="cancel",
                    )
                    state = self._save(
                        state,
                        cancelled=True,
                        pending=Pending(
                            command=command,
                            request_checksum=request_checksum(state.binding, command),
                            execution_key="planner:cancel:" + event_ref.checksum[7:55],
                        ),
                    )
                    self._finish(
                        state,
                        Outcome(reason="cancelled", refs=prior.outcome.refs),
                        phase=Phase.STOPPED,
                        cancelled=True,
                    )
                    return self.read(run_id)
                state = self._save(state, cancelled=True, reason="cancelled")
            return state

    def confirm_research_review(self, run_id, event_ref, result, *, cancel=None, stop_reason=None):
        """Confirm saved research facts only; models never run in this transaction."""
        from packages.evidence_details.current_rejection import current_rejection
        from packages.planner_control.models import Reference

        self.observe_research_control(run_id, event_ref, cancel=cancel)

        with self.runtime.session(run_id):
            state = self.read(run_id)
            for event in state.events:
                if event.request_checksum == event_ref.checksum:
                    if not any(r.checksum == digest(result) for r in event.outcome.refs):
                        raise ValueError("review_confirmation_conflict")
                    return event
            if (
                state.pending is None
                or state.pending.command.action != "review"
                or state.pending.request_checksum != event_ref.checksum
                or result["event"] != event_ref.model_dump(mode="json")
                or result["run_id"] != run_id
            ):
                raise ValueError("review_confirmation_binding")
            ref = Reference(
                role="research_review",
                code=event_ref.code,
                version=result["schema_version"],
                checksum=digest(result),
            )
            changes = {}
            phase = Phase.STOPPED
            reason = self.research_review_limit(state) or stop_reason
            if result.get("continuation") == "stop":
                reason = reason or result["reason"]
            if result["status"] == "accepted" and reason is None:
                try:
                    current = validate(Graph, self.owner.current(state.binding))
                    if (
                        current.model_dump(mode="json", exclude={"kind", "parent"})
                        != result["graph"]
                    ):
                        reason = "review_successor_version_changed"
                    else:
                        phase = Phase.RETRIEVE
                        changes = dict(graph=current, selected=None, candidates=(), result_refs=())
                except ValueError as error:
                    reason = current_rejection(error)
                    if reason is None:
                        raise
            elif result["status"] in {"reject", "alternative"} and reason is None:
                phase = Phase.FALLBACK
            cancelled = state.cancelled or bool(cancel and cancel.is_set())
            if cancelled:
                phase, changes, reason = Phase.STOPPED, {}, "cancelled"
            return self._finish(
                state,
                Outcome(
                    status="rejected" if phase == Phase.FALLBACK else "ok",
                    reason=reason or result["reason"],
                    refs=(ref,),
                ),
                phase=phase,
                changes=changes,
                cancelled=cancelled,
            )

    def _save(self, old, **changes):
        new = validate(Control, old.model_copy(update={**changes, "revision": old.revision + 1}))
        return self.runtime.save(old, new)

    def step(self, run_id, command, *, cancel=None):
        command = validate(Command, command)
        with self.runtime.session(run_id):
            state = self.read(run_id)
            sha = request_checksum(state.binding, command)
            for event in state.events:
                if event.command.key == command.key:
                    if event.request_checksum != sha:
                        raise ValueError("step_key_content_conflict")
                    return event
            if state.pending:
                if state.pending.request_checksum != sha:
                    raise ValueError("pending_step_requires_same_command")
                return self._resume(state, cancel, recovering=True)
            if command.expected_revision != state.revision:
                raise ValueError("control_revision_conflict")
            if state.phase == Phase.STOPPED or (
                command.action not in ALLOWED[state.phase]
                and command.action not in {"stop", "cancel"}
            ):
                raise ValueError("illegal_state_action")
            pending = Pending(
                command=command,
                request_checksum=sha,
                execution_key="planner:" + digest(dict(run=run_id, key=command.key))[7:55],
            )
            state = self._save(state, pending=pending)
            return self._resume(state, cancel, recovering=False)

    def _resume(self, state, cancel, *, recovering):
        pending, profile = state.pending, state.binding.profile
        action = pending.command.action
        cancelled = state.cancelled or action == "cancel" or bool(cancel and cancel.is_set())

        def finish(state, outcome, **kwargs):
            # Every early return samples the signal; stored cancellation is monotone.
            kwargs["cancelled"] = cancelled or bool(cancel and cancel.is_set())
            return self._finish(state, outcome, **kwargs)

        # Claimed execution has uncertain side effects: only recover, even after deadline/cancel.
        if pending.launch_claimed:
            try:
                outcome = validate(Outcome, self.owner.recover(state, pending))
            except Exception:
                outcome = Outcome(status="unknown", reason="recovery_unconfirmed")
            cancelled = cancelled or bool(cancel and cancel.is_set())
            if outcome.status == "unknown":
                return self._unknown(state, outcome, cancelled)
            return finish(state, outcome, cancelled=cancelled)
        if action in {"cancel", "stop"}:
            return finish(state, Outcome(reason=action), phase=Phase.STOPPED, cancelled=cancelled)
        expired = self.clock() >= state.binding.deadline
        # verify is a read/confirmation phase, never launches new work.
        if (cancelled or expired or state.steps >= profile.max_steps) and action != "verify":
            reason = "cancelled" if cancelled else "deadline" if expired else "step_budget"
            return finish(state, Outcome(reason=reason), phase=Phase.STOPPED, cancelled=cancelled)
        try:
            current = validate(Graph, self.owner.current(state.binding))
        except ValueError:
            return finish(
                state,
                Outcome(status="rejected", reason="source_changed_revalidation_required"),
                phase=Phase.STOPPED,
            )
        cancelled = cancelled or bool(cancel and cancel.is_set())
        if current != state.graph:
            return finish(
                state,
                Outcome(reason="graph_changed_reread_revalidate"),
                phase=Phase.RETRIEVE,
                changes=dict(graph=current, selected=None, candidates=(), result_refs=()),
            )
        if self.ranking is not None and action == "select":
            try:
                decision = validate(
                    ActionDecision,
                    self.ranking.resume_selection(state, pending.decision, self.clock)
                    if isinstance(pending, SelectionPending)
                    else self.ranking.select(state, self.clock),
                )
                # Re-read graph/input owners after the advisory call, even for a Fake.
                if validate(Graph, self.owner.current(state.binding)) != state.graph:
                    raise ValueError("selection_graph_changed")
                if self.clock() >= state.binding.deadline:
                    raise ValueError("selection_deadline")
                # Explicit 10D adapter confirms its additive Runtime receipt. The
                # uninstalled 08 path and its serialized events remain unchanged.
                confirm = getattr(self.ranking, "confirm_selection", None)
                if confirm is not None:
                    confirm(state, decision, self.clock)
                if cancel and cancel.is_set():
                    return finish(state, Outcome(reason="cancelled"), phase=Phase.STOPPED)
            except Exception as error:
                return finish(state, self._selection_refusal(state, error), phase=Phase.STOPPED)
            if not isinstance(pending, SelectionPending):
                state = self._save(
                    state, pending=SelectionPending(**pending.model_dump(), decision=decision)
                )
                pending = state.pending
            if self.clock() >= state.binding.deadline:
                return finish(state, Outcome(reason="action_deadline"), phase=Phase.STOPPED)
            selected = decision.selected
            outcome = SelectionOutcome(
                decision=decision, reason=decision.strategy.replace("/", ":")
            )
            if selected.kind == "stop":
                return finish(state, outcome, phase=Phase.STOPPED, changes=dict(selected=None))
            if selected.kind == "fallback":
                return finish(
                    state,
                    outcome.model_copy(update={"status": "rejected"}),
                    phase=Phase.FALLBACK,
                    changes=dict(selected=None),
                )
            return finish(state, outcome, changes=dict(selected=selected.candidate))
        if self.ranking is not None and action in {"prepare", "execute"}:
            try:
                decisions = [
                    e.outcome.decision
                    for e in state.events
                    if isinstance(e.outcome, SelectionOutcome)
                ]
                if not decisions:
                    raise ValueError("versioned_selection_required")
                self.ranking.recheck(state, decisions[-1], self.clock)
            except Exception as error:
                return finish(state, self._selection_refusal(state, error), phase=Phase.STOPPED)
        if action == "select":
            selection_reason = "deterministic_id_order"
            try:
                choices = tuple(validate(Candidate, c) for c in self.actions.candidates(state))
            except Exception:
                # Existing owner-admitted candidates were just revalidated; no model authority.
                choices = state.candidates
                selection_reason = "provider_unavailable_deterministic_baseline"
            # Provider may narrow existing owner-admitted values; cannot invent an executable.
            if len({c.id for c in choices}) != len(choices) or any(
                c not in state.candidates for c in choices
            ):
                return finish(state, Outcome(status="rejected", reason="illegal_provider"))
            choices = sorted(
                (c for c in choices if c.and_ready and c.branch not in state.attempted),
                key=lambda c: c.id,
            )
            if not choices:
                return finish(state, Outcome(reason="no_legal_or"), phase=Phase.STOPPED)
            return finish(
                state, Outcome(reason=selection_reason), changes=dict(selected=choices[0])
            )
        if action == "fallback":
            if (
                state.branches >= profile.max_branches
                or state.no_progress >= profile.max_no_progress
            ):
                return finish(state, Outcome(reason="fallback_budget"), phase=Phase.STOPPED)
            attempted = state.attempted + ((state.selected.branch,) if state.selected else ())
            return finish(
                state,
                Outcome(reason="finite_or_fallback"),
                changes=dict(branches=state.branches + 1, attempted=attempted, selected=None),
            )
        if action == "review":
            reason = (
                "expert_required_unavailable" if profile.expert_required else "expert_not_required"
            )
            return finish(state, Outcome(reason=reason), phase=Phase.STOPPED)
        if action == "prepare":
            if state.selected.operation in UNREADY:
                return finish(
                    state, Outcome(status="unavailable", reason=UNREADY[state.selected.operation])
                )
            if not pending.reserved:
                if state.reserved >= profile.max_actions:
                    return finish(state, Outcome(reason="action_budget"), phase=Phase.STOPPED)
                state = self._save(
                    state,
                    reserved=state.reserved + 1,
                    pending=pending.model_copy(update={"reserved": True}),
                )
                pending = state.pending
        if action in {"execute", "recover"}:
            # A committed claim is a fence, not a claim that a process actually started.
            state = self._save(state, pending=pending.model_copy(update={"launch_claimed": True}))
            if action == "recover":
                outcome = self.owner.recover(state, state.pending)
            else:
                outcome = self._invoke(state, state.pending, cancel)
        else:
            outcome = self._invoke(state, pending, cancel)
        outcome = validate(Outcome, outcome)
        cancelled = cancelled or bool(cancel and cancel.is_set())
        if outcome.status == "unknown":
            return self._unknown(state, outcome, cancelled)
        return finish(state, outcome)

    @staticmethod
    def _selection_refusal(state, error):
        known = {
            "stale_action_set",
            "action_deadline",
            "selection_deadline",
            "selection_graph_changed",
            "stale_pending_selection",
            "selection_source_changed",
            "selection_no_longer_legal",
            "ranking_policy_lock_mismatch",
            "grounding_input_or_policy_lock_mismatch",
            "grounding_graph_scope_mismatch",
            "knowledge_not_published_or_withdrawn",
            "action_scope_mismatch",
        }
        reason = (
            str(error)
            if str(error) in known
            else (
                "action_selection_rejected"
                if isinstance(error, ValueError)
                else "action_source_technical_error"
            )
        )
        return Outcome(
            status="rejected",
            reason=reason,
            gaps=(
                Gap(
                    kind="safety" if isinstance(error, ValueError) else "technical",
                    target=state.graph.code,
                    reason=reason,
                ),
            ),
        )

    def _invoke(self, state, pending, cancel):
        try:
            return self.owner.invoke(state, pending, cancel=cancel)
        except Exception:
            return Outcome(
                status="unknown" if pending.launch_claimed else "rejected",
                reason="owner_unconfirmed" if pending.launch_claimed else "owner_rejected",
                progress=False,
            )

    def _unknown(self, state, outcome, cancelled):
        cancelled = state.cancelled or cancelled
        # Preserve pending command/claim. Callers get a resumable snapshot, never a false event.
        if (
            state.retries < state.binding.profile.max_retries
            or state.cancelled != cancelled
            or state.reason != "unknown_job_no_relaunch"
        ):
            state = self._save(
                state,
                retries=min(state.retries + 1, state.binding.profile.max_retries),
                reason="unknown_job_no_relaunch",
                cancelled=cancelled,
            )
        return state

    def _finish(self, state, outcome, *, phase=None, cancelled=False, changes=None):
        changes = changes or {}
        cancelled = state.cancelled or cancelled
        if cancelled and changes.get("graph", state.graph) != state.graph:
            # Keep confirmed facts on their original graph; cancellation forbids
            # starting a new retrieval cycle. The new graph was not accepted.
            phase, changes = Phase.STOPPED, {}
            outcome = Outcome(status="rejected", reason="graph_changed_after_cancel")
        pending, action = state.pending, state.pending.command.action
        if phase is None:
            phase = Phase.FALLBACK if outcome.status != "ok" else NEXT[action]
        event = Event(
            command=pending.command,
            request_checksum=pending.request_checksum,
            before=state.phase,
            after=phase,
            graph=changes.get("graph", state.graph),
            outcome=outcome,
            at=self.clock(),
        )
        candidates = outcome.candidates if action == "identify" else state.candidates
        no_progress = state.no_progress
        if action in {"execute", "recover", "verify"} or outcome.status != "ok":
            new_facts = outcome.confirmed > 0 or any(
                r not in state.result_refs for r in outcome.refs
            )
            no_progress = (
                0
                if outcome.status == "ok" and outcome.progress and new_facts
                else state.no_progress + 1
            )
        updates = dict(
            phase=phase,
            pending=None,
            steps=state.steps + 1,
            events=state.events + (event,),
            candidates=candidates,
            result_refs=state.result_refs + outcome.refs,
            gaps=state.gaps + outcome.gaps,
            started=state.started + outcome.started,
            confirmed=state.confirmed + outcome.confirmed,
            cancelled=cancelled,
            reason=outcome.reason,
            no_progress=no_progress,
        )
        self._save(state, **{**updates, **changes})
        return event

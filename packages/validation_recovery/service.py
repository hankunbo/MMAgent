"""Research recovery integration; every execution and confirmation stays with 06/08."""

import html
import json
import sys
from types import SimpleNamespace

from agents.expert_agent import ResearchReview
from agents.expert_agent.models import ReviewFacts
from packages.evidence_details.current_rejection import current_rejection
from packages.grounding.models import digest, validate
from packages.path_graph_validation.service import checked_summary
from packages.planner_control.models import Command, Phase, Reference

from .models import (
    Failure,
    RecoveryFallbackReport,
    RecoveryIncident,
    RecoveryPolicy,
    RecoveryStopReport,
    ToolRequirement,
)
from .requirements import requirement_key


def scope(state):
    return SimpleNamespace(
        run_id=state.binding.run_id, case_code=state.binding.case_code, graph=state.graph
    )


def stop_reason(planner, state, actions):
    reason = planner.research_review_limit(state)
    if reason:
        return reason
    if state.branches >= state.binding.profile.max_branches:
        return "branch_budget"
    if state.reserved >= state.binding.profile.max_actions:
        return "action_budget"
    failed = state.selected.branch if state.selected else None
    if not any(a.candidate and a.branch != failed for a in actions.actions):
        return "legal_paths_exhausted"
    return None


class ValidationRecovery:
    """One bounded resolution, not a run loop. Installed readers own failure facts.

    The caller keeps key across failures. Continued work uses original Planner.step;
    this service never invokes a tool, provider role, accept, install or new run.
    """

    def __init__(self, *, planner, review, documents, policy, failure_reader, summary_reader=None):
        self.planner, self.review, self.documents = planner, review, documents
        self.policy = validate(RecoveryPolicy, policy)
        self.failure_reader, self.summary_reader = failure_reader, summary_reader
        if (
            planner.ranking is None
            or getattr(planner.ranking.source, "policy", None) != self.policy
        ):
            raise ValueError("installed_nearest_legal_source_required")
        if review.context.planner is not planner:
            raise ValueError("original_planner_context_required")

    def _read(self, key):
        with self.documents.session():
            return self.documents.read("recovery:" + key)

    def _write(self, key, values, state):
        try:
            with self.documents.session():
                self.documents.validate_scope(scope(state))
                self.documents.write("recovery:" + key, values)
                if self.documents.read("recovery:" + key) != values:
                    raise ValueError("recovery_document_confirmation")
        except Exception as error:
            try:
                self.documents.failure(key, digest(values), "object_or_transaction_failure")
            except Exception as audit_error:
                raise error from audit_error
            raise

    def resume_execution(self, run_id, *, cancel=None):
        state = self.planner.read(run_id)
        if state.pending is None or state.pending.command.action not in {
            "prepare",
            "execute",
            "recover",
            "verify",
        }:
            raise ValueError("original_execution_pending_required")
        return self.planner.step(run_id, state.pending.command, cancel=cancel)

    def _intent(self, run_id, key, cancel):
        state = self.planner.read(run_id)
        if state.binding.graph.kind != "research_artifact" or self.review.context.run_id != run_id:
            raise ValueError("research_recovery_scope")
        self.documents.validate_scope(scope(state))
        self.review.acceptance.documents.validate_scope(scope(state))
        frozen_policy = self._read("policy")
        if frozen_policy and frozen_policy != (
            dict(
                binding_checksum=digest(state.binding), policy=self.policy.model_dump(mode="json")
            ),
        ):
            raise ValueError("recovery_key_content_conflict")
        prior = self._read(key + ":incident")
        if prior:
            incident = validate(RecoveryIncident, prior[0])
            if (
                incident.key != key
                or incident.control.binding != state.binding
                or incident.policy_checksum != digest(self.policy)
            ):
                raise ValueError("recovery_key_content_conflict")
            return incident.command, incident
        # Unknown jobs and unrelated pending work cannot be escaped with a fresh key.
        command_key = "recovery:" + key
        if state.pending and state.pending.command.key != command_key:
            raise ValueError("pending_step_requires_same_command")
        if state.phase not in {Phase.SELECT, Phase.FALLBACK, Phase.REVIEW}:
            raise ValueError("recovery_choice_point_required")
        failure = validate(Failure, self.failure_reader(state))
        reason, actions = None, None
        try:
            if self.planner.owner.current(state.binding) != state.graph:
                reason = "version_conflict"
            else:
                actions = self.planner.ranking.current(state)
        except ValueError as error:
            reason = current_rejection(error)
            if reason is None:
                raise
        reason = reason or (
            stop_reason(self.planner, state, actions) if actions else "version_conflict"
        )
        if failure.category == "version_conflict":
            reason = "version_conflict"
        if state.phase != Phase.FALLBACK:
            reason = reason or "recovery_stop_review"
        if cancel and cancel.is_set():
            reason = "cancelled"
        summary = None
        if self.summary_reader:
            try:
                summary = checked_summary(self.summary_reader())
            except ValueError as error:
                rejected = current_rejection(error)
                if rejected is None:
                    raise
                reason = rejected
        if summary is not None and (
            summary.domain != "research"
            or summary.run_id != run_id
            or summary.case_code != state.binding.case_code
            or summary.binding["current_scope"]["graph"] != state.graph.model_dump(mode="json")
        ):
            raise ValueError("recovery_summary_scope")
        command = (
            state.pending.command
            if state.pending
            else Command(
                key=command_key,
                expected_revision=state.revision,
                action="review" if reason else "fallback",
                input_refs=(
                    Reference(
                        role="recovery_policy",
                        code="POLICY:RECOVERY",
                        version=self.policy.schema_version,
                        checksum=digest(self.policy),
                    ),
                ),
            )
        )
        if not any(
            r.role == "recovery_policy" and r.checksum == digest(self.policy)
            for r in command.input_refs
        ):
            raise ValueError("recovery_key_content_conflict")
        return command, (state, failure, reason, actions, summary)

    def resolve(self, run_id, *, key, cancel=None):
        # Validate the key before it participates in any storage or control identity.
        Command(key="recovery:" + key, action="review", expected_revision=0)
        if self.review.context.run_id != run_id:
            raise ValueError("research_recovery_scope")
        initial = self.planner.read(run_id)
        command = (
            initial.pending.command
            if initial.pending and initial.pending.command.key == "recovery:" + key
            else None
        )
        event = None
        try:
            command, payload = self._intent(run_id, key, cancel)
            if command.action == "review":
                event = self.planner.begin_recovery_review(run_id, command, cancel=cancel)
            else:
                self.planner.claim_recovery_fallback(run_id, command, cancel=cancel)
            self._write(
                "policy",
                (
                    dict(
                        binding_checksum=digest(initial.binding),
                        policy=self.policy.model_dump(mode="json"),
                    ),
                ),
                initial,
            )
            if isinstance(payload, RecoveryIncident):
                incident = payload
            else:
                state, failure, reason, actions, summary = payload
                gaps = tuple(dict.fromkeys(state.gaps + ((failure.gap,) if failure.gap else ())))
                requirements = tuple(
                    ToolRequirement(
                        origin_run_id=run_id,
                        binding_checksum=digest(state.binding),
                        catalog=next(r for r in state.binding.locks if r.role == "catalog"),
                        gap=g,
                        sources=failure.sources,
                        available_inputs=state.selected.input_refs if state.selected else (),
                        attempted_branches=state.attempted,
                    )
                    for g in gaps
                    if g.kind == "capability"
                )
                first_requirements = []
                for requirement in requirements:
                    old = self._read(requirement_key(requirement))
                    if old:
                        original = validate(ToolRequirement, old[0])
                        if requirement_key(original) != requirement_key(requirement):
                            raise ValueError("recovery_requirement_binding")
                        requirement = original
                    first_requirements.append(requirement)
                requirements = tuple(first_requirements)
                incident = RecoveryIncident(
                    key=key,
                    control=state,
                    command=command,
                    failure=failure,
                    policy_checksum=digest(self.policy),
                    action_set_checksum=actions.checksum if actions else None,
                    stop_reason=reason,
                    summary=summary,
                    requirements=requirements,
                )
                self._write(key + ":incident", (incident.model_dump(mode="json"),), state)
            # Content-addressed intentions are durable but never dispatched or awaited.
            for requirement in incident.requirements:
                self._write(
                    requirement_key(requirement),
                    (requirement.model_dump(mode="json"),),
                    incident.control,
                )
            if command.action == "fallback":
                saved = self._read(key + ":fallback-report")
                if saved:
                    report = validate(RecoveryFallbackReport, saved[0])
                    if report.incident != incident or saved[1] != render(report):
                        raise ValueError("recovery_report_binding")
                else:
                    current = self.planner.read(run_id)
                    report = self._report(
                        incident,
                        current,
                        dict(
                            schema_version="recovery-no-request/1.0.0",
                            status="not_requested",
                            reason="legal_alternative_or_control_stop",
                        ),
                        model=RecoveryFallbackReport,
                    )
                    self._write(
                        key + ":fallback-report",
                        (report.model_dump(mode="json"), render(report)),
                        current,
                    )
                self.planner.step(run_id, command, cancel=cancel)
                self.planner.observe_recovery_cancel(run_id, command, cancel=cancel)
                return dict(
                    incident=incident.model_dump(mode="json"),
                    report=report.model_dump(mode="json"),
                    control=self.planner.report(run_id).model_dump(mode="json"),
                )
            return self._stop(run_id, key, event, incident, cancel)
        except BaseException as error:
            try:
                if event is not None:
                    self.review.context.control(event, cancel)
                else:
                    self.planner.observe_recovery_cancel(run_id, command, cancel=cancel)
            except Exception as control_error:
                raise error from control_error
            raise

    def _report(self, incident, current, result, *, model=RecoveryStopReport):
        return model(
            incident=incident,
            control_observed=current,
            expert_result=result,
            provider_requests=self.review.ports.budget.summary().model_dump(mode="json"),
            stop_reason=self.planner.research_review_limit(current)
            or incident.stop_reason
            or "legal_alternative_available",
            unobserved_nodes=unobserved(incident, current),
            retained_evidence=tuple(
                dict.fromkeys(r for e in current.events for r in e.outcome.refs)
            ),
            attempted_alternatives=tuple(
                dict.fromkeys(
                    current.attempted + ((current.selected.branch,) if current.selected else ())
                )
            ),
            unresolved=tuple(
                dict.fromkeys(
                    current.gaps + ((incident.failure.gap,) if incident.failure.gap else ())
                )
            ),
        )

    def _stop(self, run_id, key, event, incident, cancel):
        context = StopContext(self.review.context, incident)
        advisory = ResearchReview(
            acceptance=self.review.acceptance,
            ports=self.review.ports,
            context=context,
            protocol=self.review.protocol,
        )
        # Recover exact original proposed ID; no Provider retry on a historical result.
        proposal = self._read(key + ":proposal")
        if proposal:
            proposed = proposal[0]["action_id"]
        else:
            if self.review.context.control(event, cancel):
                # Original 08 checks control before reading a proposal; no request is issued.
                proposed = "stopped-before-proposal"
            else:
                try:
                    frame = context.read(event)
                    proposed = next(a.id for a in frame.actions.actions if a.kind == "stop")
                except ValueError as error:
                    if current_rejection(error) is None:
                        raise
                    proposed = "source-rejected-before-proposal"
            self._write(key + ":proposal", (dict(action_id=proposed),), incident.control)
        result = advisory.run(event=event, proposed_action_id=proposed, cancel=cancel)
        saved = self._read(key + ":report")
        if saved:
            report = validate(RecoveryStopReport, saved[0])
            if report.incident != incident or report.expert_result != result:
                raise ValueError("recovery_report_binding")
            expected_html = render(report)
            if saved != (report.model_dump(mode="json"), expected_html):
                raise ValueError("recovery_report_html_binding")
        else:
            current = self.planner.read(run_id)
            report = self._report(incident, current, result)
            self._write(key + ":report", (report.model_dump(mode="json"), render(report)), current)
        self.review.context.control(event, cancel)
        # Mandatory stop disposition: an advisory alternative cannot extend an exhausted run.
        self.planner.confirm_research_review(
            run_id,
            event,
            result,
            cancel=cancel,
            stop_reason=incident.stop_reason or "recovery_stop",
        )
        return dict(
            report=report.model_dump(mode="json"),
            control=self.planner.report(run_id).model_dump(mode="json"),
        )


def unobserved(incident, current):
    nodes = {g.target for g in current.gaps}
    if incident.failure.gap:
        nodes.add(incident.failure.target)
    if incident.summary:
        nodes.update(row["target"] for row in incident.summary.unresolved if "target" in row)
    return tuple(sorted(nodes))


def render(report):
    model = (
        RecoveryFallbackReport if isinstance(report, RecoveryFallbackReport) else RecoveryStopReport
    )
    value = validate(model, report).model_dump(mode="json")
    return dict(
        schema_version="recovery-stop-html/1.0.0",
        report_checksum=digest(value),
        html='<!doctype html><meta charset="utf-8"><h1>研究离线安全停止</h1><pre>'
        + html.escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
        + "</pre>",
    )


class StopContext:
    def __init__(self, original, incident):
        self.original, self.incident = original, incident

    def read(self, event):
        frame = self.original.read(event)
        facts = frame.facts
        return frame.model_copy(
            update={
                "facts": ReviewFacts(
                    conflict=facts.conflict or self.incident.failure.category == "version_conflict",
                    uncertain=True,
                    high_risk=facts.high_risk,
                    required=facts.required,
                    no_progress=facts.no_progress,
                )
            }
        )

    def control(self, event, cancel=None):
        primary = sys.exception()
        try:
            reason = self.original.control(event, cancel)
        except Exception as control_error:
            # Original 08 also observes control in finally. Preserve an outstanding
            # document/transaction error if that observation itself is unconfirmed.
            if primary is not None:
                raise primary from control_error
            raise
        return reason or (
            "recovery_version_conflict" if self.incident.stop_reason == "version_conflict" else None
        )

    def counters(self):
        return self.original.counters()

    def checkpoint(self, event, result, *, cancel=None):
        return self.original.control(event, cancel)

    def install(self, *args, **kwargs):
        raise ValueError("recovery_advice_cannot_install")

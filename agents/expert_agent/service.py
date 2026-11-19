"""Recoverable Fake orchestration through original domain and Runtime ports.

Context reads the original control event, current ActionSet and owner facts. Owner
installation is explicit, idempotent and revalidates dependencies, never model text.
No database session is held across Provider calls or owner/checkpoint callbacks.
"""

from packages.evidence_details.acceptance_models import DomainState, Submission
from packages.evidence_details.current_rejection import CurrentResearchRejected, current_rejection
from packages.evidence_details.review_batches import read_submission
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material
from packages.llm_provider.provider import ProviderError

from .expert import advise
from .models import ExpertDecision, ReviewFrame, ReviewResultV2, ReviewResultV3, read_result
from .storage import ReviewStorage


class ReviewCurrentWindow:
    """Synchronous original recheck, never a caller-supplied qualification token."""

    def __init__(self, review, frame, state, cancel):
        self.review, self.frame, self.state, self.cancel = review, frame, state, cancel

    def current(self, agent, documents, state):
        review = self.review
        if (
            type(review) is not ResearchReview
            or review.ports.review is not agent
            or review.acceptance.owner is not agent.materials
            or review.acceptance.documents is not documents
            or review.acceptance.requests is not agent.budget
            or self.state != state
        ):
            raise ValueError("review_current_window_binding")
        return review._recheck(self.frame, state, self.cancel)


class ResearchReview:
    def __init__(self, *, acceptance, ports, context, protocol):
        self.acceptance, self.ports, self.context = acceptance, ports, context
        self.store = ReviewStorage(acceptance.documents)
        self.protocol = protocol
        agents = (ports.trace, ports.expert, ports.source, ports.vision, ports.review)
        if any(
            a.budget is not ports.budget or a.provider is not ports.expert.provider for a in agents
        ):
            raise ValueError("shared_provider_and_budget_required")
        if ports.expert.provider.config.mode != "fake":
            from packages.evidence_details.live_permission import require_live_permission

            require_live_permission(acceptance.live_permission, acceptance.owner.current())
        if acceptance.requests is not ports.budget:
            raise ValueError("acceptance_request_budget_mismatch")

    def _save(self, key, value):
        self.store.save(key, value)
        return value

    def _current_frame(self, event):
        from .planner import PlannerReviewContext

        # Only the installed original context with the very same owner can
        # combine these reads. Arbitrary callbacks cannot supply current facts.
        if (
            type(self.context) is PlannerReviewContext
            and self.context.owner is self.acceptance.owner
        ):
            return self.context.read_with_domain(event)
        return self.context.read(event), self.acceptance.owner.current()

    def _recheck(self, frame, state, cancel):
        try:
            return self._check_current(frame, state, cancel)
        except ValueError as error:
            reason = current_rejection(error)
            if reason is None:
                raise
            raise CurrentResearchRejected(reason) from error

    def _check_current(self, frame, state, cancel):
        from pydantic import ValidationError

        from agents.provider_agents import provider_locks

        reason = self.context.control(frame.event, cancel)
        if reason:
            raise CurrentResearchRejected(reason)
        if frame.actions.budget.steps >= frame.actions.binding.profile.max_steps:
            raise CurrentResearchRejected("step_budget")
        if frame.actions.budget.cancelled:
            raise CurrentResearchRejected("cancelled")
        if frame.facts.no_progress >= min(
            frame.policy.max_no_progress, frame.actions.binding.profile.max_no_progress
        ):
            raise CurrentResearchRejected("no_new_information")
        config = self.ports.expert.provider.config
        if config.mode == "live":
            from packages.evidence_details.live_permission import require_live_permission

            self.ports.budget.check_control()
            require_live_permission(self.acceptance.live_permission, state)
            p = frame.actions.binding.profile
            if p.max_steps > 20 or p.max_no_progress > 2 or p.max_retries > 1 or p.seconds > 2700:
                raise ValueError("bounded_live_planner_limits_required")
        locks = {r.role: r for r in frame.actions.binding.locks}
        if state.config_checksum != digest(config) or any(
            locks.get(r.role) != r for r in provider_locks(config)
        ):
            raise CurrentResearchRejected("provider_configuration_changed")
        if cancel is not None and cancel.is_set():
            raise CurrentResearchRejected("cancelled")
        now = self.ports.budget.clock()
        if now >= min(self.ports.budget.deadline, frame.actions.expires_at):
            raise CurrentResearchRejected("deadline")
        if (
            self.ports.budget.used > self.ports.budget.max_requests
            or self.ports.budget.max_requests != self.ports.expert.provider.config.max_requests
            or self.ports.budget.deadline != frame.actions.binding.deadline
        ):
            raise CurrentResearchRejected("budget_changed")
        # Complete the independent persistence reads before producing the fresh
        # domain window. No document I/O follows that last current-owner read.
        self.acceptance.documents.validate_scope(frame.scope)
        if self.acceptance.documents.consumed(frame.scope.graph):
            raise CurrentResearchRejected("parent_graph_already_consumed")
        current_frame, current = self._current_frame(frame.event)
        if validate(ReviewFrame, current_frame) != frame:
            raise CurrentResearchRejected("review_context_changed")
        try:
            current = validate(DomainState, current)
        except ValidationError as error:
            raise CurrentResearchRejected("review_domain_changed") from error
        if current != state:
            raise CurrentResearchRejected("review_domain_changed")
        return current

    def _candidate(self, key, role, round_id, frame, state, cancel):
        saved = self.store.read(key + ":candidate")
        if saved:
            sub = read_submission(saved[0])
            if sub.scope != frame.scope:
                raise ValueError("saved_candidate_scope")
            return sub
        self._recheck(frame, state, cancel)
        if role == "review" and getattr(self.ports.review.provider.config, "review_batching", None):
            from agents.provider_agents.batched_review import run_batches

            try:
                sub = run_batches(
                    self.ports.review,
                    state=state,
                    documents=self.acceptance.documents,
                    key=key,
                    round_id=round_id,
                    before_request=ReviewCurrentWindow(self, frame, state, cancel),
                )
            except ProviderError as error:
                from packages.evidence_details.acceptance_models import TaskPolicyV3
                from packages.evidence_details.review_batches import collect_partial

                recoverable_batch = error.code in {
                    "timeout",
                    "transport",
                    "rate_limit",
                    "truncated",
                    "schema",
                } or (error.code == "scope" and error.fields == ("batch_candidate",))
                if not isinstance(state.policy, TaskPolicyV3) or not recoverable_batch:
                    raise
                # Only a rejected candidate is local. A changed parent/current
                # authority remains fatal; never turn all scope errors into partials.
                # Original receipts preserve every confirmed batch, not the bad reply.
                self._recheck(frame, state, cancel)
                try:
                    sub = collect_partial(state, self.acceptance.documents, key)
                except ValueError as partial_error:
                    if str(partial_error) == "batch_incomplete_or_reused_request":
                        raise error from partial_error
                    raise
                self._save(
                    key + ":partial-cause",
                    dict(
                        error=error.code,
                        fields=list(error.fields),
                        pending_batch_ids=list(sub.pending_batch_ids),
                        failed_request_not_retried=True,
                    ),
                )
            self._save(key + ":candidate", sub.model_dump(mode="json"))
            return sub
        claim = dict(
            schema_version="research-call-claim/1.0.0",
            role=role,
            round=round_id,
            frame_checksum=digest(frame),
        )
        if not self.store.claim(key + ":call", claim):
            raise CurrentResearchRejected("unconfirmed_request_no_repeat")
        agent = getattr(self.ports, role)
        result = agent.call(
            frame.scope,
            deadline=self.ports.budget.deadline,
            round_id=round_id,
            before_request=lambda: self._recheck(frame, state, cancel),
        )
        self._recheck(frame, state, cancel)
        records = [
            r
            for r in self.ports.budget.records()
            if r.role == role
            and r.scope == frame.scope
            and r.output_checksum == digest(result)
            and r.error is None
        ]
        if not records:
            raise ValueError("candidate_request_missing")
        record = records[-1]
        from packages.evidence_details.input_scope import submission_type
        from packages.llm_provider.input_scope import ScopeVisionProposal

        sub = (submission_type(result) if isinstance(result, ScopeVisionProposal) else Submission)(
            scope=frame.scope,
            actor="PLANNER",
            correlation="SOURCE:AGENT_RESEARCH",
            request_id=record.request_id,
            prompt_checksum=record.prompt_checksum,
            config_checksum=record.config_checksum,
            observation_checksums=tuple(o.reference for o in state.material.observations),
            source_checksums=tuple(s.reference for s in state.material.sources),
            candidate=result,
        )
        self._save(key + ":candidate", sub.model_dump(mode="json"))
        return sub

    def run(self, *, event, proposed_action_id, role=None, abc=False, cancel=None):
        try:
            return self._run(
                event=event,
                proposed_action_id=proposed_action_id,
                role=role,
                abc=abc,
                cancel=cancel,
            )
        finally:
            # Even an unknown object/commit reply cannot discard an observed signal.
            self.context.control(event, cancel)

    def _run(self, *, event, proposed_action_id, role=None, abc=False, cancel=None):
        if role not in {None, "source", "vision", "review"} or (abc and role != "review"):
            raise ValueError("installed_research_role_required")
        if abc and getattr(self.ports.review.provider.config, "review_batching", None):
            raise ValueError("batched_review_single_round_required")
        key = event.code
        command = dict(
            event=event.model_dump(mode="json"),
            proposed_action_id=proposed_action_id,
            role=role,
            abc=abc,
        )
        self.store.claim(key + ":command", command)
        terminal = self.store.read(key + ":result")
        if terminal:
            result = read_result(terminal[0])
            self.context.checkpoint(event, result, cancel=cancel)
            return result  # Historical confirmation, never new authorization.
        intent = self.store.read(key + ":intent")
        if intent:
            frame = validate(ReviewFrame, intent[0]["frame"])
            state = validate(DomainState, intent[0]["state"])
        else:
            reason = self.context.control(event, cancel)
            if reason:
                return self._stop(key, event, reason, cancel)
            try:
                current_frame, current = self._current_frame(event)
                frame = validate(ReviewFrame, current_frame)
                state = validate(DomainState, current)
            except ValueError as error:
                reason = current_rejection(error)
                if reason is None:
                    raise
                return self._stop(key, event, reason, cancel)
            if frame.event != event or state.material.scope != frame.scope:
                raise ValueError("event_domain_binding")
            if frame.policy.comparison_protocol != digest(self.protocol):
                raise ValueError("comparison_protocol_changed")
            self._save(
                key + ":intent",
                dict(frame=frame.model_dump(mode="json"), state=state.model_dump(mode="json")),
            )
        proposed = next((a for a in frame.actions.actions if a.id == proposed_action_id), None)
        if proposed is None:
            raise ValueError("proposal_not_legal")
        if role and (proposed.kind != "request_review" or proposed.branch != "research_" + role):
            raise ValueError("candidate_port_not_authorized")
        selected = self.store.read(key + ":selected")
        accepted_key = key + ":domain"
        # Once accepted, recover successor installation even if its parent is consumed.
        if self.acceptance.read(key=accepted_key):
            return self._finish_acceptance(key, event, state, accepted_key, cancel)
        if frame.facts.no_progress >= frame.policy.max_no_progress:
            return self._stop(key, event, "no_new_information", cancel)
        if not selected:
            try:
                self._recheck(frame, state, cancel)
                if role:
                    rounds = ("A", "B", "C") if abc else (("A",) if role == "review" else (None,))
                    subs = [
                        self._candidate(key + ":" + (r or role), role, r, frame, state, cancel)
                        for r in rounds
                    ]
                    if abc:
                        from .study import compare_candidates

                        comparison = compare_candidates(
                            state,
                            subs,
                            self.protocol,
                            self.ports,
                            live_permission=self.acceptance.live_permission,
                        )
                        self._save(key + ":comparison", comparison)
                        # The fixed ordering is declared policy, not maximizing a medical score.
                    sub = subs[0]
                    selected = (dict(submission=sub.model_dump(mode="json")),)
                else:
                    selected = (dict(submission=None),)
                self._save(key + ":selected", selected[0])
            except (CurrentResearchRejected, ProviderError) as error:
                self._save(
                    key + ":candidate-failure",
                    dict(
                        schema_version="research-candidate-failure/1.0.0",
                        category="provider" if isinstance(error, ProviderError) else "current",
                        code=error.code if isinstance(error, ProviderError) else error.reason,
                        fields=list(error.fields) if isinstance(error, ProviderError) else [],
                    ),
                )
                return self._stop(key, event, "candidate_failed_or_context_changed", cancel)
        sub = read_submission(selected[0]["submission"]) if selected[0]["submission"] else None
        proposal = sub.candidate if sub else proposed
        decision = self.store.read(key + ":decision")
        if not decision:
            try:
                self._recheck(frame, state, cancel)
            except (CurrentResearchRejected, ProviderError):
                return self._stop(key, event, "context_changed_before_expert", cancel)
            if not self.store.claim(
                key + ":expert-call", dict(frame=digest(frame), proposal=digest(proposal))
            ):
                return self._stop(key, event, "unconfirmed_expert_no_repeat", cancel)
            advice_frame = frame
            comparison = self.store.read(key + ":comparison")
            if comparison and (
                comparison[0]["comparison"]["root_requires_investigation"]
                or any(i["investigate"] for i in comparison[0]["comparison"]["items"])
            ):
                advice_frame = frame.model_copy(
                    update={"facts": frame.facts.model_copy(update={"conflict": True})}
                )
            d = advise(
                frame=advice_frame,
                material=state.material,
                proposed=proposed,
                proposal=proposal,
                ports=self.ports,
                recheck=lambda: self._recheck(frame, state, cancel),
            )
            self._save(key + ":decision", d.model_dump(mode="json"))
        else:
            d = validate(ExpertDecision, decision[0])
        # A saved advisory result is historical, not permission to return a stale
        # alternative/acceptance after its persistence acknowledgement was lost.
        if d.decision != "safe_stop":
            try:
                self._recheck(frame, state, cancel)
            except (CurrentResearchRejected, ProviderError):
                return self._stop(key, event, "context_changed_before_decision", cancel)
        if (
            d.decision not in {"accept", "baseline"}
            or d.selected_action_id != proposed.id
            or sub is None
        ):
            return self._complete(
                key,
                event,
                dict(
                    status=d.decision,
                    reason=d.reason,
                    selected_action_id=d.selected_action_id,
                    decision_checksum=digest(d),
                    graph=None,
                ),
                cancel=cancel,
            )
        try:
            self._recheck(frame, state, cancel)
        except CurrentResearchRejected:
            return self._stop(key, event, "context_changed_before_acceptance", cancel)
        # Domain rechecks request, scope, source, observations and task policy again.
        if role == "source":
            from packages.knowledge_claims.research_candidates import submit_research_candidate

            result = submit_research_candidate(self.acceptance, sub, key=accepted_key)
        else:
            result = self.acceptance.submit(sub, key=accepted_key)
        if not isinstance(result, dict) or result["status"] != "submitted_not_accepted":
            return self._stop(key, event, "domain_submission_rejected", cancel)
        # Submission persistence may itself observe cancellation or a changed owner.
        try:
            self._recheck(frame, state, cancel)
        except CurrentResearchRejected:
            return self._stop(key, event, "context_changed_before_acceptance", cancel)
        try:
            self.acceptance.accept(key=accepted_key)
        except ValueError as error:
            reason = current_rejection(error)
            if reason is None:
                raise
            if not self.acceptance.read(key=accepted_key):
                return self._stop(key, event, reason, cancel)
        return self._finish_acceptance(key, event, state, accepted_key, cancel)

    def _finish_acceptance(self, key, event, parent, accepted_key, cancel):
        from .reporting import read_research_report

        report = read_research_report(self.acceptance, key=accepted_key, parent=parent)
        if report["receipt"]["status"] != "accepted":
            return self._stop(key, event, "domain_acceptance_rejected", cancel)
        saved_decision = self.store.read(key + ":decision")
        saved_selection = self.store.read(key + ":selected")
        if saved_decision is None or saved_selection is None:
            raise ValueError("accepted_review_decision_missing")
        decision = validate(ExpertDecision, saved_decision[0])
        submission = read_submission(saved_selection[0]["submission"])
        if (
            decision.decision not in {"accept", "baseline"}
            or decision.selected_action_id != decision.proposed_action_id
            or decision.scope != parent.material.scope
            or decision.proposal_checksum != digest(report["records"]["candidate"])
            or digest(submission) != report["receipt"]["submission_checksum"]
        ):
            raise ValueError("accepted_review_decision_binding")
        # The historical report is saved even if current-use permission has expired.
        self._save(key + ":report", report)
        from .reporting import render_research_html

        self._save(
            key + ":html",
            dict(
                schema_version="research-successor-html/1.0.0",
                report_checksum=digest(report),
                html=render_research_html(report),
            ),
        )
        reason = self.context.control(event, cancel)
        installed, expected = "not_installed", None
        saved = self.store.read(key + ":successor")
        if saved:
            material = validate(Material, saved[0])
            if (
                material.scope.graph.model_dump(mode="json", exclude={"kind", "parent"})
                != report["receipt"]["graph"]
            ):
                raise ValueError("saved_successor_graph")
            expected = validate(DomainState, parent.model_copy(update={"material": material}))
            attempt = self.store.read(key + ":install-attempt")
            confirmation = self.store.read(key + ":installed")
            if any(
                v and v[0]["successor_checksum"] != digest(expected)
                for v in (attempt, confirmation)
            ):
                raise ValueError("saved_install_binding")
            installed = (
                "installed" if confirmation else "unconfirmed" if attempt else "not_installed"
            )
        if reason is None:
            try:
                if not saved:
                    if validate(DomainState, self.acceptance.owner.current()) != parent:
                        raise CurrentResearchRejected("research_parent_changed")
                    material = self.acceptance.successor_material(key=accepted_key)
                    self._save(key + ":successor", material.model_dump(mode="json"))
                expected = validate(DomainState, parent.model_copy(update={"material": material}))
                current = validate(DomainState, self.acceptance.owner.current())
                if current not in (parent, expected):
                    raise CurrentResearchRejected("research_parent_changed")
                reason = self.context.control(event, cancel)
                if reason is None:
                    # Confirmation loss remains an exception; retry uses this exact successor.
                    marker = dict(
                        schema_version="research-install-fact/1.0.0",
                        successor_checksum=digest(expected),
                    )
                    self._save(key + ":install-attempt", marker)
                    reason = self.context.control(event, cancel)
                    if reason is None:
                        installed = "unconfirmed"
                        self.context.install(event, parent=parent, successor=expected)
                        self._save(key + ":installed", marker)
                        installed = "installed"
                        if validate(DomainState, self.acceptance.owner.current()) != expected:
                            raise CurrentResearchRejected("research_successor_version_changed")
                        reason = self.context.control(event, cancel)
            except ValueError as error:
                reason = current_rejection(error)
                if reason is None:
                    raise
        return self._complete(
            key,
            event,
            dict(
                status="safe_stop" if reason else "accepted",
                reason=reason or "research_successor_installed",
                graph=None if reason else report["receipt"]["graph"],
                selected_action_id=None,
                report_checksum=digest(report),
                successor_checksum=digest(expected) if installed == "installed" else None,
                acceptance_key=accepted_key,
                decision_checksum=digest(decision),
                domain_status="accepted",
                successor_status=installed,
            ),
            cancel=cancel,
        )

    def _complete(self, key, event, result, *, cancel=None):
        reason = self.context.control(event, cancel)
        if reason:
            result.update(status="safe_stop", reason=reason, graph=None, selected_action_id=None)
        records = self.ports.budget.records()
        intent = self.store.read(key + ":intent")
        budget = intent[0]["frame"]["actions"]["budget"] if intent else self.context.counters()
        value = dict(
            schema_version="research-review-result/2.0.0",
            event=event.model_dump(mode="json"),
            run_id=self.ports.budget.run_id,
            clinical_acceptance=False,
            new_medical_review=False,
            recorded_agent_invocations=len({r.invocation_id for r in records}),
            original_tool_reservations=budget["reserved"],
            original_tool_starts=budget["started"],
            requests=self.ports.budget.summary().model_dump(mode="json"),
            continuation="continue"
            if result["status"] in {"accepted", "reject", "alternative"}
            else "stop",
            cancelled=reason == "cancelled",
            **{"domain_status": "not_accepted", "successor_status": "not_installed", **result},
        )
        live = self.ports.expert.provider.config.mode == "live"
        if live:
            value.update(
                schema_version="research-review-result/3.0.0",
                permission_checksum=digest(self.acceptance.live_permission.grant),
                new_medical_review=any(
                    r.role == "review"
                    and r.scope.case_code == self.acceptance.live_permission.grant.case_code
                    and not r.simulated
                    and r.error is None
                    for r in records
                ),
            )
        value = validate(ReviewResultV3 if live else ReviewResultV2, value).model_dump(mode="json")
        self._save(key + ":result", value)
        self.context.checkpoint(event, value, cancel=cancel)
        return value

    def _stop(self, key, event, reason, cancel=None):
        return self._complete(
            key,
            event,
            dict(status="safe_stop", reason=reason, graph=None, selected_action_id=None),
            cancel=cancel,
        )

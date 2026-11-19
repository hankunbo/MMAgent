"""One registered case and one installed tool; durable deadline, no UNKNOWN replay."""

from datetime import UTC, datetime
from threading import Event, Timer

from packages.grounding.models import digest


class ResearchSingleCaseService:
    def __init__(
        self,
        *,
        runtime,
        research,
        choice,
        bind_plan,
        config_checksum,
        budget_seconds,
        preparation=None,
        register_request=None,
        completion_goal=None,
    ):
        self.runtime, self.research, self.choice = runtime, research, choice
        self.bind_plan = bind_plan
        self.config_checksum, self.budget_seconds = config_checksum, budget_seconds
        self.preparation, self.register_request = preparation, register_request
        from .task_completion import ResearchTaskCompletion

        self.task_completion = (
            ResearchTaskCompletion(self, goal=completion_goal) if completion_goal else None
        )

    def run(self, sample, *, key, cancel=None):
        """K only: original 07A registration/readback, then registered 06 context."""
        self._validate_key(key)
        if self.preparation is None or self.register_request is None:
            raise ValueError("research_preparation_not_installed")
        registration = self.preparation.register(sample, key=key + ":visible")
        visible = self.preparation.read_visible(registration.visible.case.case_code)
        if visible != registration.visible:
            raise ValueError("research_visible_readback_mismatch")
        request = self.register_request(visible, key)
        return self.run_registered(request, key=key, cancel=cancel)

    def run_registered(self, request, *, key, cancel=None, recover_only=False):
        """Consumes an installed registration, never caller scores or graph evidence."""
        self._validate_key(key)
        with self.runtime.session(request.scope.case_code, "research:" + key):
            context = self.research.create(request.code, request.version, request.checksum, key)
            if self.research.read(context.run_id) != context:
                raise ValueError("research_context_readback")
            # Re-admission is mandatory even during readback/recovery; failure never launches.
            try:
                choice = self.choice(request) if callable(self.choice) else self.choice
                plan = self.research.compile(context.run_id, choice)
            except ValueError:
                if self.runtime.lifecycle(context.run_id) is not None:
                    raise ValueError(
                        "research_current_admission_failed_no_launch_or_recovery"
                    ) from None
                return self.runtime.reject_preparation(context, key, self.config_checksum)
            if self.task_completion is None:
                state = self.runtime.begin_execution(
                    context, plan, key, self.config_checksum, self.budget_seconds
                )
                executor, scores = self.bind_plan(plan, state["binding"]["execution_key"])
            else:
                executor, scores = self.bind_plan(plan, key + ":tool")
                definition = self.task_completion.definition(
                    context, plan, executor, scores, key + ":tool"
                )
                state = self.runtime.begin_execution(
                    context,
                    plan,
                    key,
                    self.config_checksum,
                    self.budget_seconds,
                    task_definition=definition,
                )
            if scores.context_version != "research-run/1.0.0":
                raise ValueError("research_score_port_required")
            if state["phase"] == "FINISHED":
                ledger = scores.read(context.run_id, state["binding"]["score_key"])
                if ledger.checksum != state["report"]["score_checksum"]:
                    raise ValueError("research_report_source_drift")
                return state["report"]
            return self._drive(state, plan, executor, scores, cancel, recover_only)

    def read_task_completion(self, run_id):
        if self.task_completion is None:
            from packages.evidence_details.current_rejection import CurrentResearchRejected

            raise CurrentResearchRejected("task_completion_not_installed")
        return self.task_completion.read(run_id)

    @staticmethod
    def _validate_key(key):
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 80
            or not key.isascii()
            or any(not (c.isalnum() or c in "-_.:") for c in key)
        ):
            raise ValueError("research_single_case_key")

    def _record(self, state, phase, reason=None, target=None, report=None):
        return self.runtime.record_lifecycle(
            state, phase=phase, reason=reason, target_status=target, report=report
        )

    def _drive(self, state, plan, executor, scores, cancel, recover_only):
        rid = state["run_id"]
        execution_key = state["binding"]["execution_key"]
        if state["phase"] == "SCORING":
            return self._finish(state, scores)
        attempt = executor.recover(rid, execution_key)
        if attempt is None and state["phase"] == "UNKNOWN":
            raise ValueError("research_unknown_attempt_missing_no_replay")
        if attempt is None and recover_only:
            raise ValueError("research_recovery_no_attempt")
        if attempt is not None:
            try:
                workflow = executor.recover_commit(plan, execution_key)
            except Exception:
                # The controlled 06 gateway may also report a database/commit
                # failure. Its journal, not the exception type, determines outcome.
                return self._unknown(state)
        else:
            for source, target in (
                ("CREATED", "NORMALIZING"),
                ("NORMALIZING", "PLANNING"),
                ("PLANNING", "EXECUTING"),
            ):
                if self.runtime.run_status(rid) == source:
                    state = self._record(state, "PREPARED", target=target)
            remaining = (
                datetime.fromisoformat(state["deadline"]) - datetime.now(UTC)
            ).total_seconds()
            if remaining <= 0 or (cancel is not None and cancel.is_set()):
                reason = "deadline_exhausted" if remaining <= 0 else "cancelled_before_launch"
                state = self._record(state, "SCORING", reason, "STOPPED_UNSAFE")
                return self._finish(state, scores)
            # Store launch intent first. A crash here is not permission to launch on retry.
            if state["phase"] == "EXECUTING":
                return self._unknown(state)
            state = self._record(state, "EXECUTING")
            cancellation = cancel if cancel is not None else Event()
            timer = Timer(remaining, cancellation.set)
            timer.daemon = True
            timer.start()
            try:
                outcome = executor.execute(plan, execution_key, cancel=cancellation)
            except Exception:
                # The durable 06 attempt determines uncertainty, never an exception string.
                return self._unknown(state)
            finally:
                timer.cancel()
            if (
                getattr(outcome, "state", None) == "technical_rejection"
                and executor.recover(rid, execution_key) is None
            ):
                state = self._record(
                    state, "SCORING", "controlled_tool_rejected_before_attempt", "FAILED_TECHNICAL"
                )
                return self._finish(state, scores)
            try:
                workflow = executor.recover_commit(plan, execution_key)
            except Exception:
                return self._unknown(state)
        status = workflow.status
        if status == "SUCCEEDED":
            state = self._record(state, "EXECUTING", "tool_succeeded", "VERIFYING")
            target, reason = (
                "COMPLETED_INSUFFICIENT_EVIDENCE",
                "research_output_without_medical_graph",
            )
        elif status == "CANCELLED":
            expired = datetime.now(UTC) >= datetime.fromisoformat(state["deadline"])
            target, reason = "STOPPED_UNSAFE", "deadline_exhausted" if expired else "tool_cancelled"
        else:
            target, reason = "FAILED_TECHNICAL", "tool_" + status.lower()
        state = self._record(state, "SCORING", reason, target)
        return self._finish(state, scores)

    def _unknown(self, state):
        if state["phase"] != "UNKNOWN":
            state = self._record(state, "UNKNOWN", "execution_outcome_unknown_no_replay")
        return dict(
            version="research-case-report/1.0.0",
            run_id=state["run_id"],
            run_status=self.runtime.run_status(state["run_id"]),
            terminal=False,
            technical_status="UNKNOWN_JOB",
            reason=state["technical_reason"],
            execution_key=state["binding"]["execution_key"],
            score_key=state["binding"]["score_key"],
            plan_checksum=state["binding"]["plan_checksum"],
            config_checksum=self.config_checksum,
            deadline=state["deadline"],
            score_checksum=None,
            score_missing_reason="execution_outcome_unknown",
            model_outputs=[],
            research_only=True,
            synthetic=self.research.synthetic,
            decisive_evidence=False,
            graph=None,
            graph_scores="unavailable",
            accepted_validations=[],
            recovery="same installed request, plan and key; recover_only=True; never launch",
        )

    def _finish(self, state, scores):
        rid, key = state["run_id"], state["binding"]["score_key"]
        ledger = scores.save(rid, key)
        if scores.read(rid, key) != ledger:
            raise ValueError("research_score_readback")
        report = dict(
            version="research-case-report/1.0.0",
            run_id=rid,
            run_status=self.runtime.run_status(rid),
            terminal=True,
            technical_status=ledger.technical[0].status if ledger.technical else "NOT_EXECUTED",
            reason=state["technical_reason"],
            execution_key=state["binding"]["execution_key"],
            score_key=key,
            score_checksum=ledger.checksum,
            policy_checksum=scores.policy_checksum,
            plan_checksum=state["binding"]["plan_checksum"],
            config_checksum=self.config_checksum,
            deadline=state["deadline"],
            research_only=True,
            synthetic=ledger.synthetic,
            decisive_evidence=False,
            graph=None,
            graph_scores="unavailable",
            accepted_validations=[],
            model_entries=[
                e.model_dump(mode="json") for e in ledger.entries if e.definition.layer == "model"
            ],
            model_outputs=verified_model_outputs(ledger, scores.objects),
        )
        report["checksum"] = digest(report)
        return self._record(state, "FINISHED", state["technical_reason"], report=report)["report"]


def verified_model_outputs(ledger, objects):
    """Only H-confirmed, available source objects can contribute a research prediction."""
    from packages.fairclip_research.models import read_output
    from packages.object_store import checksum

    found = {}
    for entry in ledger.entries:
        source = entry.source
        if source.output_schema_version != "fairclip-output/1.0.0" or entry.status != "available":
            continue
        ref = source.object_ref
        raw = objects.read_bounded(ref.uri, 65536)
        if len(raw) != ref.byte_size or checksum(raw) != ref.checksum:
            raise ValueError("research_model_output_source_drift")
        value = read_output(raw)
        found[ref.checksum] = dict(
            output=value.model_dump(mode="json"), source=source.model_dump(mode="json")
        )
    return [found[k] for k in sorted(found)]

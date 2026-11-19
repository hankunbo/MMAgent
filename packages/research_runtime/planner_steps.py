"""07/06 one-step facade, reusing original owners; no new run/pipeline or SQL."""

from datetime import UTC, datetime
from threading import Event

from packages.grounding.models import digest, validate
from packages.planner_control.models import Candidate, Gap, Outcome, Reference
from packages.tool_execution.models import ExecutionError, WorkflowResult


class DeadlineCancellation:
    def __init__(self, deadline, caller):
        self.deadline, self.caller = deadline, caller or Event()

    def is_set(self):
        return self.caller.is_set() or datetime.now(UTC) >= self.deadline


class ResearchPlannerSteps:
    def __init__(
        self, *, snapshots, research=None, choice=None, executor_for=None, operation="observe"
    ):
        if operation not in {"observe", "tool"}:
            raise ValueError("installed_operation_required")
        self.operation = operation
        self.snapshots, self.research = snapshots, research
        self.choice, self.executor_for = choice, executor_for

    def current(self, binding):
        if binding.graph.kind != "research_artifact":
            raise ValueError("research_graph_required")
        if self.operation == "tool":
            if (
                self.research is None
                or self.research.read(binding.run_id).checksum != binding.run_context_checksum
            ):
                raise ValueError("research_context_changed")
        return self.snapshots.current(binding)

    def _plan(self, state):
        if self.research is None or self.choice is None or self.executor_for is None:
            raise ValueError("tool_ports_not_installed")
        context = self.research.read(state.binding.run_id)
        if context.checksum != state.binding.run_context_checksum:
            raise ValueError("run_context_changed")
        plan = self.research.compile(state.binding.run_id, self.choice)
        locks = {r.role: r for r in state.binding.locks}
        if (locks["catalog"].code, locks["catalog"].version, locks["catalog"].checksum) != (
            plan.document.catalog_ref.code,
            plan.document.catalog_ref.version,
            plan.document.catalog_ref.checksum,
        ):
            raise ValueError("catalog_lock_mismatch")
        if locks["execution"].checksum != plan.document.config.checksum:
            raise ValueError("execution_config_lock_mismatch")
        if state.pending.command.action != "prepare":
            expected = [r for r in state.result_refs if r.role == "plan"]
            if not expected or expected[-1].checksum != plan.checksum:
                raise ValueError("exact_plan_changed")
        return plan

    @staticmethod
    def _result(workflow):
        # Only an owner-confirmed WorkflowResult reaches this projection.
        workflow = validate(WorkflowResult, workflow)
        started = sum("RUNNING" in s.status_history for s in workflow.steps)
        return Outcome(
            status="ok" if workflow.status == "SUCCEEDED" else "rejected",
            reason="tool_" + workflow.status.lower(),
            refs=(
                Reference(
                    role="workflow",
                    code=workflow.attempt_id,
                    version="1.0.0",
                    checksum=digest(workflow),
                ),
            ),
            started=started,
            confirmed=1,
        )

    def invoke(self, state, pending, *, cancel):
        action = pending.command.action
        if action in {"load", "retrieve", "instantiate", "verify"}:
            return self.snapshots.step(state.binding, action)
        if action == "identify":
            return Outcome(
                candidates=(
                    Candidate(
                        id="installed_" + self.operation,
                        branch="installed",
                        operation=self.operation,
                        and_ready=True,
                    ),
                ),
                gaps=(
                    Gap(
                        kind="unavailable_port",
                        target="new_evidence",
                        reason="08C_acceptance_not_ready",
                    ),
                ),
            )
        if action == "prepare":
            if state.selected.operation == "observe":
                return Outcome(reason="read_existing_only")
            plan = self._plan(state)
            return Outcome(
                refs=(
                    Reference(
                        role="plan",
                        code=plan.document.run_id,
                        version="1.0.0",
                        checksum=plan.checksum,
                    ),
                )
            )
        if action == "execute":
            if state.selected.operation == "observe":
                return self.snapshots.step(state.binding, "verify")
            plan = self._plan(state)
            signal = DeadlineCancellation(state.binding.deadline, cancel)
            # compile time consumes the same absolute budget, including immediately before launch.
            if signal.is_set():
                return Outcome(status="rejected", reason="deadline_or_cancel_before_launch")
            executor = self.executor_for(plan)
            returned = executor.execute(plan, pending.execution_key, cancel=signal)
            workflow = executor.execution_result(plan, pending.execution_key, returned)
            if workflow is None:
                return Outcome(status="rejected", reason="tool_prelaunch_validation_rejected")
            return self._result(workflow)
        raise ValueError("owner_action_not_installed")

    def recover(self, state, pending):
        if state.selected.operation == "observe":
            return self.snapshots.step(state.binding, "verify")
        plan = self._plan(state)
        executor = self.executor_for(plan)
        record = executor.recover(state.binding.run_id, pending.execution_key)
        if record is None:
            return Outcome(status="unknown", reason="no_attempt_no_permission_to_relaunch")
        if record["result"] is not None:
            return self._result(
                executor.confirmed_result(
                    plan, pending.execution_key, validate(WorkflowResult, record["result"])
                )
            )
        if state.retries >= state.binding.profile.max_retries:
            return Outcome(status="unknown", reason="recovery_commit_budget_read_only")
        try:
            workflow = executor.recover_commit(plan, pending.execution_key)
        except ExecutionError:
            return Outcome(status="unknown", reason="unknown_job_no_relaunch")
        return self._result(executor.confirmed_result(plan, pending.execution_key, workflow))

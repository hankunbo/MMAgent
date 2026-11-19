"""Parent Task Control owns a component tool action; child v1 owns 06 admission.

No child Planner, lifecycle deadline, action budget, SQL or medical score. The
immutable prepare receipt binds the two identities. Confirmed tool facts remain
facts after cancellation; they are not a new medical-consumption authorization.
"""

import json
from datetime import UTC, datetime

from packages.grounding.models import canonical, digest, validate
from packages.planner_control.models import Candidate, Control, Gap, Outcome, Phase, Reference
from packages.tool_execution.models import WorkflowResult
from packages.tool_workflow import ExecutionPlan

from .component_inputs import ComponentResearchRequests
from .planner_steps import ResearchPlannerSteps
from .service import ResearchRuntimeService


class ParentCancellation:
    """Original parent control is sampled at prelaunch and runner cancellation polls."""

    def __init__(self, sessions, state, deadline, caller):
        self.sessions, self.state = sessions, state
        self.deadline, self.caller = deadline, caller
        self.reason = None

    def is_set(self):
        # Local cancellation/deadline cannot depend on a successful database read.
        if (
            self.reason
            or (self.caller and self.caller.is_set())
            or datetime.now(UTC) >= self.deadline
        ):
            return True
        try:
            current = validate(Control, self.sessions.read(self.state.binding.run_id))
        except Exception:
            # Ask the original runner to terminate normally and journal cleanup.
            # The adapter separately records this as technical failure, not user cancellation.
            self.reason = "component_parent_control_unavailable"
            return True
        if (
            current.binding != self.state.binding
            or current.graph != self.state.graph
            or current.selected != self.state.selected
            or current.pending != self.state.pending
            or current.reserved != self.state.reserved
        ):
            self.reason = "component_parent_control_changed"
            return True
        return bool(
            current.cancelled
            or (self.caller and self.caller.is_set())
            or datetime.now(UTC) >= self.deadline
        )


class ComponentPlannerSteps(ResearchPlannerSteps):
    """Explicit installed one-component/one-method port, not task-run2 admission.

    `work_deadline` is the installed campaign acquisition cutoff, not a renewed
    duration. It cannot exceed the original parent deadline and is frozen in the
    prepare receipt. The original Planner owns selection, reserve and claim.
    """

    def __init__(
        self, *, snapshots, requests, research, choice, executor_for, sessions, work_deadline
    ):
        if (
            type(requests) is not ComponentResearchRequests
            or type(research) is not ResearchRuntimeService
        ):
            raise ValueError("component_original_owners_required")
        if research.research is not requests or research.inputs is not requests:
            raise ValueError("component_research_owner_binding")
        if work_deadline.tzinfo is None:
            raise ValueError("component_absolute_deadline_required")
        super().__init__(
            snapshots=snapshots,
            research=research,
            choice=choice,
            executor_for=executor_for,
            operation="tool",
        )
        self.requests, self.sessions, self.work_deadline = requests, sessions, work_deadline
        self.documents = requests.documents

    def _candidate(self):
        identity = "component_" + digest(self.requests.registration_binding)[7:39]
        return Candidate(id=identity, branch=identity, operation="tool", and_ready=True)

    def current(self, binding):
        parent = self.requests.parent
        locks = {ref.role: ref for ref in binding.locks}
        if "admission" in locks and locks["admission"].checksum != digest(
            self.research.compiler.admission.policy
        ):
            raise ValueError("component_admission_policy_changed")
        if (
            binding.run_id != parent.run_id
            or binding.run_context_checksum != parent.checksum
            or binding.case_code != parent.request.scope.case_code
            or binding.input_checksum != parent.request.input_checksum
            or binding.visible_checksum != parent.request.visible_checksum
            or not binding.started_at < self.work_deadline <= binding.deadline
        ):
            raise ValueError("component_parent_binding")
        # Read-only original source reconstruction; no registration or new child.
        self.requests.for_key(self.requests.request_key)
        return self.snapshots.current(binding)

    def read_current_result(self, state, execution_key):
        """Re-admit an exact confirmed tool fact, without dispatch or commit.

        This is a current technical consumption check, not a new medical
        interpretation or permission to resume an expired producer run. The
        consuming campaign must separately bind its case/task and own its budget.
        Historical ``recover`` deliberately does not require this current gate.
        Artifact bytes are not read here: the downstream consumer must bounded-
        read and verify their original checksums before interpreting them.
        """
        return self._read_result(state, execution_key, consumption=False)

    def read_consumable_result(self, state, execution_key):
        """confirmed-result-consumption/1.0.0: current artifact use, no launch.

        A producer's expired environment probe is preserved as historical
        execution provenance; current source/method/input/QC checks remain.
        The consumer still owns its separate case/task/cancel/time budget.
        """
        return self._read_result(state, execution_key, consumption=True)

    def consume_confirmed_result(self, state, execution_key, reader, *, final_check=None):
        """Bound local interpretation between the original current checks.

        The installed reader may bounded-read/decode original artifacts, but
        must not dispatch, write or wait for a Provider. No qualification token
        escapes this call; every subsequent call repeats both current checks.
        """
        if not callable(reader):
            raise ValueError("component_local_reader_required")
        if final_check is not None and not callable(final_check):
            raise ValueError("component_local_final_check_required")
        return self._read_result(
            state, execution_key, consumption=True, reader=reader, final_check=final_check
        )

    def _read_result(self, state, execution_key, *, consumption, reader=None, final_check=None):
        state = validate(Control, state)
        if (
            self.sessions.read(state.binding.run_id) != state
            or state.pending is not None
            or state.confirmed != 1
        ):
            raise ValueError("component_confirmed_control_required")
        saved, context = self._saved(state)
        if not consumption:
            current, _ = self._current_plan(state, create=False)
            if current.checksum != saved.checksum:
                raise ValueError("component_consumption_plan_changed")
        executor = self.executor_for(saved)
        if not consumption:
            self._consumption_requirements(state, saved, executor)
        record = executor.recover(context.run_id, execution_key)
        if record is None or record["result"] is None:
            raise ValueError("component_confirmed_result_required")
        workflow = executor.confirmed_result(
            saved, execution_key, validate(WorkflowResult, record["result"])
        )
        refs = self._result(workflow).refs
        if workflow.status != "SUCCEEDED" or any(ref not in state.result_refs for ref in refs):
            raise ValueError("component_consumption_result_binding")
        if consumption:
            # Confirmation is proven before using the new source-validation
            # semantics. UNKNOWN/failed/unconfirmed never reach this path.
            projection = None
            original_workflow = digest(workflow) if reader is not None else None
            for index in range(2):
                if self.current(state.binding) != state.graph:
                    raise ValueError("component_graph_changed")
                self._consumption_requirements(state, saved, executor, consumption=True)
                if index == 0 and reader is not None:
                    projection = reader(workflow)
                    if digest(workflow) != original_workflow:
                        raise ValueError("component_consumption_result_changed")
            if final_check is not None:
                # Only bounded local selection/projection checks belong here;
                # no decoding, remote I/O, dispatch or new qualification lease.
                final_check()
                if digest(workflow) != original_workflow:
                    raise ValueError("component_consumption_result_changed")
            if self.current(state.binding) != state.graph:
                raise ValueError("component_graph_changed")
            if self.sessions.read(state.binding.run_id) != state:
                raise ValueError("component_consumption_control_changed")
            return workflow if reader is None else (workflow, projection)
        # Original owner admission is repeated after cold result validation.
        # No external dispatch follows this check. No database transaction is
        # held across object reads/admission. This is not global atomicity.
        current, _ = self._current_plan(state, create=False)
        if current.checksum != saved.checksum:
            raise ValueError("component_consumption_plan_changed")
        self._consumption_requirements(state, saved, executor)
        if self.sessions.read(state.binding.run_id) != state:
            raise ValueError("component_consumption_control_changed")
        return workflow

    def _consumption_requirements(self, state, plan, executor, *, consumption=False):
        prepared = self.documents.read(self._key(state))
        if prepared is None or len(prepared) != 1:
            raise ValueError("component_prepare_receipt_required")
        reader = executor.consumption_requirements if consumption else executor.task_requirements
        current = json.loads(canonical(reader(plan)))
        if current != prepared[0]["execution_requirements"]:
            raise ValueError("component_execution_requirements_changed")

    def _state(self, state, pending):
        state = validate(Control, state)
        if self.sessions.read(state.binding.run_id) != state or state.pending != pending:
            raise ValueError("component_current_control_required")
        if (
            state.selected is None
            or state.selected.operation != "tool"
            or state.selected != self._candidate()
            or state.reserved < 1
            or state.reserved > state.binding.profile.max_actions
        ):
            raise ValueError("component_parent_reservation_required")
        return state

    def _key(self, state):
        return (
            "component-plan:"
            + digest(dict(request_key=self.requests.request_key, selected=state.selected))[7:]
        )

    def _receipt(self, state, plan, context, requirements):
        return dict(
            schema_version="component-parent-execution/1.0.0",
            parent_run_id=state.binding.run_id,
            parent_binding=digest(state.binding),
            selected=state.selected.model_dump(mode="json"),
            reserved_actions=state.reserved,
            registration_binding=digest(self.requests.registration_binding),
            request_key=self.requests.request_key,
            request_checksum=context.request.checksum,
            child_run_id=context.run_id,
            child_context=context.checksum,
            plan_checksum=plan.checksum,
            plan_bundle=json.loads(plan.to_bundle_json()),
            execution_requirements=json.loads(canonical(requirements)),
            work_deadline=self.work_deadline.isoformat(),
            child_budget_created=False,
        )

    def _saved(self, state):
        saved = self.documents.read(self._key(state))
        if saved is None or len(saved) != 1:
            raise ValueError("component_prepare_receipt_required")
        value = saved[0]
        plan = ExecutionPlan.from_bundle_json(json.dumps(value["plan_bundle"]))
        # Historical context read, NOT current admission. Recovery only confirms
        # the exact old attempt; current medical consumption must re-admit.
        context = self.research.runtime.read_research_run(value["child_run_id"])
        context.validate_plan(plan)
        if value != self._receipt(state, plan, context, value["execution_requirements"]):
            raise ValueError("component_prepare_binding_changed")
        if context.request.scope != self.requests.parent.request.scope:
            raise ValueError("component_child_scope")
        return plan, context

    def _current_plan(self, state, *, create):
        if self.current(state.binding) != state.graph:
            raise ValueError("component_graph_changed")
        request = self.requests.for_key(self.requests.request_key)
        context = (
            self.research.create(
                request.code, request.version, request.checksum, self.requests.request_key
            )
            if create
            else self.research.read(request.lock.run_id)
        )
        choice = self.choice(request) if callable(self.choice) else self.choice
        plan = self.research.compile(context.run_id, choice)
        if len(plan.document.steps) != 1:
            raise ValueError("component_one_step_budget_required")
        locks = {r.role: r for r in state.binding.locks}
        ref = plan.document.catalog_ref
        if (locks["catalog"].code, locks["catalog"].version, locks["catalog"].checksum) != (
            ref.code,
            ref.version,
            ref.checksum,
        ) or locks["execution"].checksum != plan.document.config.checksum:
            raise ValueError("component_plan_lock_mismatch")
        if self.current(state.binding) != state.graph:
            raise ValueError("component_graph_changed")
        return plan, context

    def invoke(self, state, pending, *, cancel):
        if pending.command.action == "identify":
            if self.sessions.read(state.binding.run_id) != state or state.pending != pending:
                raise ValueError("component_current_control_required")
            if self.current(state.binding) != state.graph:
                raise ValueError("component_graph_changed")
            return Outcome(candidates=(self._candidate(),))
        if pending.command.action not in {"prepare", "execute"}:
            return super().invoke(state, pending, cancel=cancel)
        state = self._state(state, pending)
        if pending.command.action == "prepare":
            if state.phase != Phase.PREPARE or not pending.reserved or pending.launch_claimed:
                raise ValueError("component_prepare_reservation")
        elif state.phase != Phase.EXECUTE or not pending.launch_claimed:
            raise ValueError("component_launch_claim_required")
        signal = ParentCancellation(self.sessions, state, self.work_deadline, cancel)
        if signal.is_set():
            return self._stopped(signal)
        saved = None
        if pending.command.action == "execute":
            if not pending.launch_claimed:
                raise ValueError("component_launch_claim_required")
            saved, _ = self._saved(state)
        plan, context = self._current_plan(state, create=pending.command.action == "prepare")
        if signal.is_set():
            return self._stopped(signal)
        if pending.command.action == "prepare":
            receipt = self._receipt(
                state, plan, context, self.executor_for(plan).task_requirements(plan)
            )
            with self.documents.session():
                self.documents.write(self._key(state), (receipt,))
            if self.documents.read(self._key(state)) != (receipt,):
                raise ValueError("component_prepare_readback")
            return Outcome(
                refs=(
                    Reference(
                        role="plan", code=context.run_id, version="1.0.0", checksum=plan.checksum
                    ),
                )
            )
        prepared = self.documents.read(self._key(state))[0]
        requirements = json.loads(canonical(self.executor_for(plan).task_requirements(plan)))
        if requirements != prepared["execution_requirements"]:
            raise ValueError("component_execution_requirements_changed")
        refs = [r for r in state.result_refs if r.role == "plan"]
        if saved.checksum != plan.checksum or not refs or refs[-1].checksum != plan.checksum:
            raise ValueError("component_prepared_plan_changed")
        executor = self.executor_for(plan)
        if signal.is_set():
            return self._stopped(signal)
        returned = executor.execute(plan, pending.execution_key, cancel=signal)
        workflow = executor.execution_result(plan, pending.execution_key, returned)
        if workflow is None:
            return Outcome(status="rejected", reason="tool_prelaunch_validation_rejected")
        if signal.reason:
            with self.documents.session():
                self.documents.write(
                    "component-stop:" + pending.execution_key,
                    (
                        dict(
                            schema_version="component-parent-stop/1.0.0",
                            reason=signal.reason,
                            parent_binding=digest(state.binding),
                            child_run_id=context.run_id,
                            execution_key=pending.execution_key,
                            workflow_checksum=digest(workflow),
                        ),
                    ),
                )
        # Preserve the confirmed tool fact even if its current eligibility changed.
        # No medical scoring/observation acceptance happens in this facade.
        return self._outcome(state, pending, workflow, context.run_id)

    def _outcome(self, state, pending, workflow, child_run_id):
        result = self._result(workflow)
        stopped = self.documents.read("component-stop:" + pending.execution_key)
        if stopped is None:
            return result
        if len(stopped) != 1:
            raise ValueError("component_stop_binding")
        saved = stopped[0]
        if (
            saved.get("schema_version") != "component-parent-stop/1.0.0"
            or saved.get("parent_binding") != digest(state.binding)
            or saved.get("execution_key") != pending.execution_key
            or saved.get("workflow_checksum") != digest(workflow)
            or saved.get("child_run_id") != child_run_id
            or saved.get("reason")
            not in {"component_parent_control_unavailable", "component_parent_control_changed"}
        ):
            raise ValueError("component_stop_binding")
        return result.model_copy(
            update=dict(
                status="rejected",
                reason=saved["reason"],
                gaps=(Gap(kind="technical", target=saved["child_run_id"], reason=saved["reason"]),),
            )
        )

    @staticmethod
    def _stopped(signal):
        reason = signal.reason or "deadline_or_cancel_before_launch"
        return Outcome(
            status="rejected",
            reason=reason,
            gaps=(
                (Gap(kind="technical", target=signal.state.binding.run_id, reason=reason),)
                if signal.reason
                else ()
            ),
        )

    def recover(self, state, pending):
        state = self._state(state, pending)
        if pending.command.action not in {"execute", "recover"} or not pending.launch_claimed:
            raise ValueError("component_recovery_claim_required")
        plan, _ = self._saved(state)
        executor = self.executor_for(plan)
        record = executor.recover(plan.document.run_id, pending.execution_key)
        if record is None:
            return Outcome(status="unknown", reason="no_attempt_no_permission_to_relaunch")
        if record["result"] is not None:
            return self._outcome(
                state,
                pending,
                executor.confirmed_result(
                    plan, pending.execution_key, validate(WorkflowResult, record["result"])
                ),
                plan.document.run_id,
            )
        if state.retries >= state.binding.profile.max_retries:
            return Outcome(status="unknown", reason="recovery_commit_budget_read_only")
        # Original journal/commit recovery; never execute when remote state unknown.
        workflow = executor.recover_commit(plan, pending.execution_key)
        return self._outcome(
            state,
            pending,
            executor.confirmed_result(plan, pending.execution_key, workflow),
            plan.document.run_id,
        )

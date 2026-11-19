"""Selected-choice adapter only; 05 owns choices and 06 still admits/compiles/executes."""

from packages.grounding.models import digest, validate
from packages.planner_control.models import Candidate, Outcome
from packages.research_runtime.planner_steps import ResearchPlannerSteps
from packages.tool_workflow import WorkflowChoice


class SelectedResearchSteps:
    """Install trusted current candidate/choice readers, never Agent tool-name strings.

    Each candidate carries a workflow_choice reference produced alongside the
    original typed WorkflowChoice. RankedActions independently rechecks current
    type/input/safety membership; the original research service admits and compiles.
    """

    def __init__(self, *, snapshots, research, candidates, choice_for, executor_for):
        self.snapshots, self.research = snapshots, research
        self.candidates, self.choice_for, self.executor_for = candidates, choice_for, executor_for

    def _steps(self, state=None):
        choice = None
        operation = "tool"
        if state is not None:
            candidate = validate(Candidate, state.selected)
            operation = candidate.operation
            if operation == "tool":
                choice = validate(WorkflowChoice, self.choice_for(state.binding, candidate))
                if not any(
                    r.role == "workflow_choice" and r.checksum == digest(choice)
                    for r in candidate.input_refs
                ):
                    raise ValueError("recovery_workflow_choice_binding")
            elif operation != "observe":
                raise ValueError("installed_operation_required")
        return ResearchPlannerSteps(
            snapshots=self.snapshots,
            research=self.research,
            choice=choice,
            executor_for=self.executor_for,
            operation=operation,
        )

    def current(self, binding):
        return self._steps().current(binding)

    def invoke(self, state, pending, *, cancel):
        if pending.command.action == "identify":
            self.current(state.binding)
            return Outcome(
                candidates=tuple(validate(Candidate, c) for c in self.candidates(state.binding))
            )
        if pending.command.action in {"load", "retrieve", "instantiate", "verify"}:
            return self.snapshots.step(state.binding, pending.command.action)
        return self._steps(state).invoke(state, pending, cancel=cancel)

    def recover(self, state, pending):
        return self._steps(state).recover(state, pending)

"""Narrow installed adapter to the original Planner; no duplicate pipeline or scoring."""

from agents.planner.service import Planner
from packages.grounding.models import digest, validate
from packages.planner_control.models import Command, Phase
from packages.planner_control.transitions import ALLOWED

from .models import DispatchRequest


class PlannerTaskOwner:
    def __init__(self, *, planners, runtime, reviews=None):
        if any(not isinstance(p, Planner) for p in planners.values()):
            raise ValueError("dispatch_owner_required")
        self.planners, self.runtime = dict(planners), runtime
        self.reviews = dict(reviews or {})
        if self.reviews:
            from .expert import ExpertDispatch

            if any(
                not isinstance(port, ExpertDispatch)
                or port.run_id != run_id
                or port.planner is not self.planners.get(run_id)
                for run_id, port in self.reviews.items()
            ):
                raise ValueError("dispatch_owner_required")

    def planner(self, run_id):
        planner = self.planners.get(run_id)
        if planner is None:
            raise ValueError("dispatch_owner_required")
        return planner

    def current(self, request, *, sources=False):
        request = validate(DispatchRequest, request)
        original = self.runtime.read_run(request.run_id)
        context = original.context
        if not context.synthetic:
            raise ValueError("dispatch_synthetic_only")
        if (
            context.run_id != request.run_id
            or context.request.scope.case_code != request.case_code
            or context.checksum != request.context_checksum
        ):
            raise ValueError("dispatch_scope")
        planner = self.planner(request.run_id)
        state = planner.read(request.run_id)
        if (
            state.binding.case_code != request.case_code
            or state.binding.run_context_checksum != request.context_checksum
        ):
            raise ValueError("dispatch_scope")
        if sources:
            # Reuse original domain source/version checks. Planner itself handles graph changes.
            planner.owner.current(state.binding)
        return state

    def next_command(self, receipt):
        state = self.current(receipt.request)
        if receipt.binding_checksum != digest(state.binding):
            raise ValueError("dispatch_binding_changed")
        # An uncertain dispatch confirmation reuses its exact command, including after commit.
        if receipt.command is not None:
            return receipt.command
        if state.pending is not None:
            return state.pending.command
        if state.phase == Phase.STOPPED:
            return None
        if receipt.cancel_requested or state.cancelled:
            action = "cancel"
        else:
            action = {
                Phase.SELECT: "select",
                Phase.EXECUTE: "execute",
                Phase.FALLBACK: "fallback",
            }.get(state.phase)
            if action is None:
                choices = ALLOWED[state.phase]
                if len(choices) != 1:
                    raise ValueError("dispatch_action_unavailable")
                action = next(iter(choices))
        return Command(
            key=f"dispatch-{state.revision}-{action}",
            expected_revision=state.revision,
            action=action,
        )

    def advance(self, receipt, *, cancel):
        planner = self.planner(receipt.request.run_id)
        state = self.current(receipt.request)
        if state.phase == Phase.STOPPED:
            return state
        if receipt.command is None:
            raise ValueError("dispatch_command_required")
        review = self.reviews.get(receipt.request.run_id)
        if receipt.command.action == "review" and review is not None:
            if review.advance(receipt.command, cancel=cancel):
                return self.current(receipt.request)
        planner.step(receipt.request.run_id, receipt.command, cancel=cancel)
        return self.current(receipt.request)

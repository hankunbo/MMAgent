"""Opt-in 08 owner adapter: preserve actual acquisition gaps in original Control.

Does not create actions, runs, request budgets or executors. Newly prepared material
and external searches must be installed by their respective owners before use.
"""

from packages.grounding.models import validate
from packages.planner_control.models import Gap, Outcome

from .planning import plan_acquisition


class AcquisitionPlannerOwner:
    def __init__(self, original_steps, material_owner, requirements):
        self.original = original_steps
        self.material_owner = material_owner
        self.requirements = requirements

    def __getattr__(self, name):
        return getattr(self.original, name)

    def invoke(self, state, pending, *, cancel):
        if pending.command.action != "identify":
            return self.original.invoke(state, pending, cancel=cancel)
        current = self.material_owner.current()
        scope = current.material.scope
        binding = state.binding
        if (
            scope.run_id,
            scope.case_code,
            scope.graph,
            scope.input_checksum,
            scope.visible_checksum,
        ) != (
            binding.run_id,
            binding.case_code,
            state.graph,
            binding.input_checksum,
            binding.visible_checksum,
        ):
            raise ValueError("acquisition_planner_scope")
        plan = plan_acquisition(current, self.requirements(current))
        baseline = self.original.invoke(state, pending, cancel=cancel)
        if self.material_owner.current() != current:
            raise ValueError("acquisition_planner_material_changed")
        gaps = list(baseline.gaps)
        for card in plan.cards:
            if not card.blockers:
                continue
            kind = (
                "missing_data"
                if card.disposition == "current_input_unavailable"
                else "product_demand"
                if "targeted_vision" in card.next_actions
                else "mapping"
            )
            gaps.append(Gap(kind=kind, target=card.assessment_id, reason=card.blockers[0]))
        # Never convert unresolved requirements into an invented CapabilityGap.
        return validate(Outcome, baseline.model_copy(update={"gaps": tuple(gaps)}))

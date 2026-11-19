"""Narrow value-only ports. Installed owners retain all graph and process authority."""

from typing import Protocol

from packages.planner_control.models import Binding, Candidate, Control, Graph, Outcome, Pending


class OwnerSteps(Protocol):
    def current(self, binding: Binding) -> Graph: ...
    def invoke(self, state: Control, pending: Pending, *, cancel) -> Outcome: ...
    def recover(self, state: Control, pending: Pending) -> Outcome: ...


class ActionProvider(Protocol):
    def candidates(self, state: Control) -> tuple[Candidate, ...]: ...


class RegisteredActions:
    """08A baseline only; 08B may replace this value-only legal-provider port."""

    def candidates(self, state):
        return state.candidates


class EmptyMemory:
    def retrieve(self, binding):
        return ()


UNREADY = {
    "new_review": "08C_domain_acceptance_not_ready",
    "new_branch": "08C_relation_acceptance_not_ready",
    "trace_ranking": "08B_not_installed",
    "expert": "08D_not_installed",
}

"""Explicit installed research candidate ports in the original LegalActionSet."""

from packages.grounding.models import seal, validate
from packages.planner_control.actions import action, unknown_components
from packages.planner_control.models import LegalActionSet


class ResearchActionSource:
    def __init__(self, source, owner, *, roles=("source", "vision", "review"), phases=("review",)):
        if not set(roles) <= {"source", "vision", "review"} or len(set(roles)) != len(roles):
            raise ValueError("research_ports_required")
        self.source, self.owner, self.roles = source, owner, roles
        if not phases or not set(phases) <= {"select", "review", "fallback"}:
            raise ValueError("research_action_phase")
        self.phases = phases

    def read(self, control, policy):
        return self.read_with_domain(control, policy)[0]

    def read_with_domain(self, control, policy):
        """Fresh original action and domain read in one synchronous window.

        No supplied DomainState, stored qualification or cross-call cache is
        accepted. The context may reuse this exact owner-produced state only.
        """
        from packages.planner_control.actions import SavedResearchActions
        from packages.research_runtime.live_snapshots import LiveSnapshots

        domain = None
        if (
            type(self.source) is SavedResearchActions
            and type(self.source.snapshots) is LiveSnapshots
            and self.source.snapshots.owner is self.owner
        ):
            original, domain = self.source.read_with_domain(control, policy)
        else:
            original = self.source.read(control, policy)
        original = validate(LegalActionSet, original)
        if control.phase not in self.phases:
            return original, None
        state = self.owner.current() if domain is None else domain
        material = state.material
        if material.scope.graph != control.graph or material.scope.run_id != control.binding.run_id:
            raise ValueError("research_action_scope")
        choices = list(original.actions)
        # These authorize candidate generation only, never tool execution or publication.
        if (
            not original.budget.cancelled
            and original.budget.no_progress < control.binding.profile.max_no_progress
        ):
            for role in self.roles:
                if role == "vision" and not any(o.image for o in material.observations):
                    continue
                if role == "review":
                    from packages.evidence_details.review_tasks import project_review_tasks

                    if not project_review_tasks(state).tasks:
                        continue
                choices.append(
                    action(
                        schema_version="planner-research-action/1.0.0",
                        kind="request_review",
                        target=control.graph,
                        branch="research_" + role,
                        candidate=None,
                        candidate_ref=None,
                        required_inputs=tuple(s.reference for s in material.sources)
                        + tuple(o.reference for o in material.observations),
                        entrypoint="ResearchReview.run",
                        preconditions=("research_owner_verified",),
                        reasons=("installed_" + role + "_candidate_port",),
                        components=unknown_components(),
                        expires_at=original.expires_at,
                    )
                )
        fields = original.model_dump(mode="json", exclude={"checksum"})
        fields["actions"] = tuple(sorted(choices, key=lambda a: a.semantic_key()))
        return seal(LegalActionSet, **fields), state

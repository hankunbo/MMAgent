"""Original Planner control context. Source and install ports remain owner-controlled."""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.current_rejection import current_rejection
from packages.grounding.models import digest, validate
from packages.planner_control.models import Phase

from .models import ReviewFacts, ReviewFrame


class PlannerReviewContext:
    def __init__(
        self,
        *,
        planner,
        run_id,
        domain_owner,
        action_source,
        ranking_policy,
        review_policy,
        facts,
        install,
        documents,
    ):
        self.planner, self.run_id, self.owner = planner, run_id, domain_owner
        self.source, self.ranking = action_source, ranking_policy
        self.policy, self.facts, self.installer = review_policy, facts, install
        self.documents = documents

    def read(self, event):
        return self.read_with_domain(event)[0]

    def read_with_domain(self, event):
        from .actions import ResearchActionSource

        state = self.planner.read(self.run_id)
        if (
            state.phase not in {Phase.REVIEW, Phase.SELECT, Phase.FALLBACK}
            or state.pending is None
            or state.pending.command.action != "review"
            or state.pending.request_checksum != event.checksum
            or event.code != "review:" + event.checksum[7:]
        ):
            raise ValueError("original_review_event_required")
        if state.cancelled:
            raise ValueError("cancelled")
        facts = validate(ReviewFacts, self.facts(state))
        facts = facts.model_copy(update={"no_progress": max(facts.no_progress, state.no_progress)})
        if type(self.source) is ResearchActionSource and self.source.owner is self.owner:
            actions, domain = self.source.read_with_domain(state, self.ranking)
            if domain is None:
                domain = self.owner.current()
        else:
            actions = self.source.read(state, self.ranking)
            # Arbitrary action readers are not trusted current-domain producers.
            # Read the owner after their callback, not before it.
            domain = self.owner.current()
        frame = ReviewFrame(
            event=event,
            scope=domain.material.scope,
            actions=actions,
            policy=self.policy,
            facts=facts,
        )
        return frame, domain

    def install(self, event, *, parent, successor):
        state = self.planner.read(self.run_id)
        if state.pending is None or state.pending.request_checksum != event.checksum:
            raise ValueError("original_review_event_required")
        self.installer(event, parent=parent, successor=successor)

    def control(self, event, cancel=None):
        state = self.planner.observe_research_control(self.run_id, event, cancel=cancel)
        return self.planner.research_review_limit(state)

    def counters(self):
        state = self.planner.read(self.run_id)
        return dict(reserved=state.reserved, started=state.started)

    def checkpoint(self, event, result, *, cancel=None):
        with self.documents.session():
            saved = self.documents.read("review:" + event.code + ":result")
            if saved is None or saved[0] != result:
                raise ValueError("persisted_review_result_required")
        state = self.planner.observe_research_control(self.run_id, event, cancel=cancel)
        reason = None
        # A prior checkpoint is historical and never reauthorizes this successor.
        if state.pending is not None and result["status"] == "accepted" and not state.cancelled:
            try:
                current = validate(DomainState, self.owner.current())
                if digest(current) != result["successor_checksum"]:
                    reason = "research_successor_version_changed"
            except ValueError as error:
                reason = current_rejection(error)
                if reason is None:
                    raise
        return self.planner.confirm_research_review(
            self.run_id, event, result, cancel=cancel, stop_reason=reason
        )

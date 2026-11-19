"""Synthetic, installed HTTP intent over original review documents and Planner commands."""

from agents.expert_agent.planner import PlannerReviewContext
from agents.expert_agent.service import ResearchReview
from packages.grounding.models import digest, validate
from packages.planner_control.models import Command, Phase

from .expert_models import ExpertDispatchRequest


class ExpertDispatch:
    def __init__(self, *, service, runtime):
        if not isinstance(service, ResearchReview) or not isinstance(
            service.context, PlannerReviewContext
        ):
            raise ValueError("expert_owner_required")
        if service.ports.expert.provider.config.mode != "fake":
            raise ValueError("expert_synthetic_only")
        self.service, self.runtime = service, runtime
        self.planner, self.run_id = service.context.planner, service.context.run_id
        self.documents = service.acceptance.documents

    def _scope(self, body):
        context = self.runtime.read_run(self.run_id).context
        if not context.synthetic or self.service.ports.expert.provider.config.mode != "fake":
            raise ValueError("expert_synthetic_only")
        if (context.request.scope.case_code, context.checksum) != (
            body.case_code,
            body.context_checksum,
        ):
            raise ValueError("expert_scope")

    def options(self):
        state = self.planner.read(self.run_id)
        context = self.runtime.read_run(self.run_id).context
        if not context.synthetic or self.service.ports.expert.provider.config.mode != "fake":
            raise ValueError("expert_synthetic_only")
        if (
            state.phase != Phase.REVIEW
            or state.pending
            or self.planner.research_review_limit(state)
        ):
            raise ValueError("expert_control_state")
        actions = self.service.context.source.read(state, self.service.context.ranking)
        return dict(
            schema_version="research-expert-options/1.0.0",
            case_code=state.binding.case_code,
            run_id=self.run_id,
            context_checksum=context.checksum,
            expected_version=state.graph.checksum,
            expected_revision=state.revision,
            idempotency_key=f"dispatch-{state.revision}-review",
            actions_checksum=actions.checksum,
            actions=[dict(id=a.id, kind=a.kind, branch=a.branch) for a in actions.actions],
            current_authorization=False,
        )

    def _current(self, body):
        self._scope(body)
        state = self.planner.read(self.run_id)
        if state.phase != Phase.REVIEW or state.pending or state.cancelled:
            raise ValueError("expert_control_state")
        if (
            state.revision != body.expected_revision
            or state.graph.checksum != body.expected_version
        ):
            raise ValueError("expert_version_changed")
        if self.planner.research_review_limit(state):
            raise ValueError("expert_control_state")
        context = self.service.context
        domain = context.owner.current()
        if (
            domain.material.scope.run_id != self.run_id
            or domain.material.scope.case_code != body.case_code
            or domain.material.scope.graph != state.graph
        ):
            raise ValueError("expert_scope")
        actions = context.source.read(state, context.ranking)
        proposed = next((a for a in actions.actions if a.id == body.proposed_action_id), None)
        if proposed is None or (body.role and proposed.branch != "research_" + body.role):
            raise ValueError("expert_action_scope")
        if body.abc and body.role != "review":
            raise ValueError("expert_action_scope")
        return dict(
            domain_checksum=digest(domain),
            actions_checksum=actions.checksum,
            policy_checksum=digest(context.policy),
            config_checksum=digest(self.service.ports.expert.provider.config),
        )

    @staticmethod
    def command(body):
        return Command(
            key=f"dispatch-{body.expected_revision}-review",
            expected_revision=body.expected_revision,
            action="review",
        )

    @staticmethod
    def _key(command):
        return "platform-expert-command:" + command.key

    def submit(self, body, *, actor_code):
        body = validate(ExpertDispatchRequest, body)
        self._scope(body)
        command = self.command(body)
        if body.idempotency_key != command.key:
            raise ValueError("expert_key_conflict")
        request = dict(body=body.model_dump(mode="json"), actor_code=actor_code, run_id=self.run_id)
        request_key = self._key(command)
        with self.documents.session():
            previous = self.documents.read(request_key)
        if previous:
            if previous[0]["request"] != request:
                raise ValueError("expert_key_conflict")
            # An immutable registration receipt is not current permission to execute.
            return previous[0]
        binding = self._current(body)
        value = dict(
            schema_version="research-expert-intent/1.0.0",
            request=request,
            command=command.model_dump(mode="json"),
            binding=binding,
            status="registered",
            executed=False,
            current_authorization=False,
        )
        # A single immutable bundle is the intent and the idempotency receipt. The original
        # document session is an advisory lock, NOT a transaction spanning separate writes.
        with self.documents.session():
            old = self.documents.read(request_key)
            if old and old[0] != value:
                raise ValueError("expert_key_conflict")
            self.documents.write(request_key, (value,))
        return value

    def advance(self, command, *, cancel):
        with self.documents.session():
            saved = self.documents.read(self._key(command))
        if saved is None:
            return False  # Original deterministic uninstalled-review behavior stays intact.
        intent = saved[0]
        if intent["command"] != command.model_dump(mode="json"):
            raise ValueError("expert_key_conflict")
        body = validate(ExpertDispatchRequest, intent["request"]["body"])
        self._scope(body)
        state = self.planner.read(self.run_id)
        previous = any(e.command == command for e in state.events)
        if not state.pending and not previous:
            if state.cancelled or cancel.is_set() or self.planner.research_review_limit(state):
                self.planner.step(self.run_id, command, cancel=cancel)
                return True
            if self._current(body) != intent["binding"]:
                raise ValueError("expert_version_changed")
        event = self.planner.begin_research_review(self.run_id, command, cancel=cancel)
        if (
            not self.service.store.read(event.code + ":intent")
            and not self.service.store.read(event.code + ":result")
            and not self.service.context.control(event, cancel)
        ):
            # Revalidate after original pending persistence, before the first model request.
            # Once the original review intent/result exists, its owner controls restoration,
            # including installation of a successor whose digest differs from this parent.
            context = self.service.context
            bound = intent["binding"]
            if (
                digest(context.owner.current()) != bound["domain_checksum"]
                or digest(context.policy) != bound["policy_checksum"]
                or digest(self.service.ports.expert.provider.config) != bound["config_checksum"]
            ):
                raise ValueError("expert_version_changed")
        self.service.run(
            event=event,
            proposed_action_id=body.proposed_action_id,
            role=body.role,
            abc=body.abc,
            cancel=cancel,
        )
        return True

import json
from pathlib import Path
from types import SimpleNamespace

from agents.expert_agent import ResearchReview
from agents.expert_agent.actions import ResearchActionSource
from agents.expert_agent.models import ReviewFacts, ReviewFrame, ReviewPolicy
from agents.provider_agents import ProviderTraceAgent, StructuredRoleAgent, provider_locks
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import load_provider_config
from packages.planner_control.models import Control, Phase, RankingPolicy
from tests.agent_provider.support import Documents, budget_for, materials, ref
from tests.planner.test_actions import Source, ranked_binding


def protocol():
    return json.loads(Path("configs/research/evidence_review_protocol_v1.json").read_text())


class Context:
    def __init__(self, reader, frame):
        self.reader, self.frame = reader, frame
        self.installs, self.checkpoints = [], []
        self.fail_install = self.fail_checkpoint = False
        self.cancelled = False

    def control(self, event, cancel=None):
        self.cancelled = self.cancelled or bool(cancel and cancel.is_set())
        return "cancelled" if self.cancelled else None

    def counters(self):
        return self.frame.actions.budget.model_dump()

    def read(self, event):
        assert event == self.frame.event
        self.reader.current()
        return self.frame

    def install(self, event, *, parent, successor):
        current = validate(DomainState, self.reader.current())
        if current not in (parent, successor):
            raise ValueError("owner_install_conflict")
        if current == parent:
            self.reader.state = successor
            self.installs.append(event)
        if self.fail_install:
            self.fail_install = False
            raise OSError("install_confirmation_lost")

    def checkpoint(self, event, result, *, cancel=None):
        self.control(event, cancel)
        if self.checkpoints and self.checkpoints[-1] != result:
            raise ValueError("checkpoint_conflict")
        if not self.checkpoints:
            self.checkpoints.append(result)
        if self.fail_checkpoint:
            self.fail_checkpoint = False
            raise OSError("checkpoint_confirmation_lost")


def setup(
    replies=(),
    *,
    facts=None,
    reader=None,
    binding=None,
    config=None,
    documents=None,
    budget=None,
    role="review",
):
    config = config or load_provider_config(environ={})
    ranking = RankingPolicy()
    b = binding or ranked_binding(ranking)
    if binding is None:
        locks = {r.role: r for r in provider_locks(config)}
        b = b.model_copy(update={"locks": tuple(locks.get(r.role, r) for r in b.locks)})
    reader = reader or materials(b, role=role, config=config)[0]
    source = Source(b)
    control = Control(binding=b, graph=b.graph, phase=Phase.REVIEW, candidates=source.choices)
    actions = ResearchActionSource(source, reader).read(control, ranking)
    event = ref("review_control", "review:synthetic", checksum=digest("original_control_event"))
    frame = ReviewFrame(
        event=event,
        scope=reader.state.material.scope,
        actions=actions,
        policy=ReviewPolicy(comparison_protocol=digest(protocol())),
        facts=ReviewFacts(**(facts or {})),
    )
    context = Context(reader, frame)
    budget = budget or budget_for(b, config)
    provider = DeterministicFakeProvider(config, replies)
    options = dict(provider=provider, budget=budget, materials=reader)
    ports = SimpleNamespace(
        budget=budget,
        trace=ProviderTraceAgent(**options),
        **{
            role: StructuredRoleAgent(role, **options)
            for role in ("expert", "source", "vision", "review")
        },
    )
    docs = documents or Documents()
    acceptance = ResearchAcceptance(owner=reader, documents=docs, requests=budget)
    service = ResearchReview(
        acceptance=acceptance, ports=ports, context=context, protocol=protocol()
    )
    proposed = next(a for a in actions.actions if a.kind == "observe")
    return SimpleNamespace(
        service=service,
        context=context,
        frame=frame,
        reader=reader,
        provider=provider,
        budget=budget,
        documents=docs,
        ports=ports,
        proposed=proposed,
        acceptance=acceptance,
        binding=b,
        source=source,
    )


def run(f, **kwargs):
    proposed = f.proposed
    if kwargs.get("role"):
        proposed = next(
            a for a in f.frame.actions.actions if a.branch == "research_" + kwargs["role"]
        )
    return f.service.run(event=f.frame.event, proposed_action_id=proposed.id, **kwargs)


def expert_reply(action="accept", action_id=None, mutate=None):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        if mutate:
            mutate(payload)
        return dict(
            scope_checksum=digest(payload["material"]["scope"]),
            action=action,
            action_id=(action_id or payload["expert_review"]["proposed_action_id"])
            if action in {"accept", "alternative"}
            else None,
            evidence=[payload["material"]["sources"][0]["reference"]["code"]],
            reason="Synthetic bounded suggestion",
        )

    return reply

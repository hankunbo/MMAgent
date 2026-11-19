"""Six genuine accepted/review stages; original owners, injected HTTP only."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.evidence_details.live_permission import ResearchLiveGrant, ResearchLivePermission
from packages.evidence_details.recorded_finalization import RecordedBatchRequests
from packages.grounding.models import digest
from packages.llm_provider.provider import RequestBudget, StructuredProvider
from packages.llm_provider.transport import HTTPTransport
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.finalization_support import fixture
from tests.agent_provider.support import Reader
from tests.agent_provider.test_accepted_review_reuse import state_from
from tests.agent_provider.test_all_reused_review import one_item
from tests.agent_provider.test_bounded_live import SyntheticJournal, SyntheticWire
from tests.evaluation_preparation.test_directional_review import reply
from tests.expert_agent.test_accepted_continued_postgres import stop, successor
from tests.expert_agent.test_batch_assembly import LiveDocuments
from tests.planner.support import MemoryRuntime


def chain(monkeypatch, pg=None):
    one_item(monkeypatch)
    old = fixture(pg, corrected=True, reasoned=True, dynamic_policy=True, expired_clock=False)
    if pg is None:
        old.docs.consumed = lambda graph: LiveDocuments.consumed(old.docs, graph)
    original = ResearchAcceptance(
        owner=old.owner, documents=old.docs, requests=old.budget, live_permission=old.old
    )
    original.submit(old.submission, key="original")
    assert original.accept(key="original").status == "accepted"
    binding = old.planner.read(old.docs.run_id).binding
    history = {
        old.docs.run_id: SimpleNamespace(
            owner=old.owner, documents=old.docs, budget=old.budget, key="original"
        )
    }

    def reader(run, lineage=()):
        f = history[run]
        return AcceptedReviewReader(
            documents=f.documents,
            requests=f.budget,
            state_reader=f.owner.current,
            acceptance_key=f.key,
            ancestor_reader=reader,
            lineage=lineage,
        )

    prior = old.docs.run_id
    for index in range(1, 6):
        if pg is not None:
            docs, b = successor(
                SimpleNamespace(documents=old.docs, binding=binding), f"LINEAGE_{index}"
            )
            journal = RuntimeAgentJournal(docs)
        else:
            b = binding.model_copy(update={"run_id": f"RUN:LINEAGE_{index}"})
            docs = LiveDocuments()
            docs.run_id = b.run_id
            docs.root = old.docs.root
            docs.sessions = MemoryRuntime()
            journal = SyntheticJournal()
            journal.documents = docs
        stop(docs, b)
        raw = old.state.model_dump(mode="json")
        raw["material"]["scope"].update(run_id=docs.run_id, graph=b.graph.model_dump(mode="json"))
        if index == 5:
            item = deepcopy(raw["policy"]["items"][0])
            item.update(id="zz:new", observation_code="OBS:NEW")
            observation = deepcopy(raw["material"]["observations"][0])
            observation["reference"].update(
                code="OBS:NEW", checksum=digest("new synthetic observed feature")
            )
            observation["value"] = "New synthetic local feature"
            raw["material"]["observations"].append(observation)
            raw["policy"]["items"].append(item)
            requirement = deepcopy(raw["policy"]["requirements"][0])
            requirement["id"] = item["id"]
            raw["policy"]["requirements"].append(requirement)
            group = deepcopy(raw["policy"]["groups"][0])
            group.update(id="group:new", slots=[[item["id"]]])
            raw["policy"]["groups"].append(group)
        owner = Reader(state_from(raw))
        owner.confirmed_review_reader = reader(prior)
        budget = RequestBudget(
            run_id=docs.run_id, deadline=b.deadline, max_requests=40, journal=journal
        )
        grant = old.old.grant.model_dump(mode="python")
        grant.update(
            run_id=docs.run_id,
            policy_checksum=owner.current().material.scope.policy_checksum,
            deadline=b.deadline,
        )
        permission = ResearchLivePermission(
            grant=ResearchLiveGrant(**grant), config=old.config, documents=docs, budget=budget
        )
        wire = SyntheticWire([reply])

        class Transport(HTTPTransport):
            def __init__(self, wire):
                super().__init__()
                self.wire = wire

            def send(self, config, body, **kwargs):
                return self._observed(
                    self.wire.send(config, body, **kwargs),
                    {"http_status": 200},
                    body,
                    "chat_completions",
                )

        agent = StructuredRoleAgent(
            "review",
            provider=StructuredProvider(old.config, live_opt_in=True, transport=Transport(wire)),
            materials=owner,
            budget=budget,
        )
        key = f"lineage{index}"
        docs.write("live-stage-intent:review", ({"event": {"code": key}},))
        docs.write("dynamic-phase-input", (owner.current(),))
        sub = run_batches(
            agent, state=owner.current(), documents=docs, key=key + ":A", round_id="A"
        )
        service = ResearchAcceptance(
            owner=owner, documents=docs, requests=budget, live_permission=permission
        )
        f = SimpleNamespace(
            old=old,
            owner=owner,
            documents=docs,
            budget=budget,
            requests=RecordedBatchRequests(journal),
            wire=wire,
            config=old.config,
            submission=sub,
            key=key,
            acceptance=service,
            history=history,
        )
        if index < 5:
            service.submit(sub, key=key)
            assert service.accept(key=key).status == "accepted"
            assert not wire.calls
            history[docs.run_id] = f
            prior = docs.run_id
        else:
            from packages.evidence_details import accepted_review_reuse as reuse

            with monkeypatch.context() as patch:
                patch.setattr(reuse, "MAX_ACCEPTED_REVIEW_DEPTH", 4)
                with pytest.raises(ValueError, match="^accepted_review_reuse_depth$"):
                    reuse.provenance_batches(sub)
            phase = dict(
                run_id=docs.run_id, target=None, accepted=False, error="accepted_review_reuse_depth"
            )
            docs.write("dynamic-phase-result", (phase,))
            docs.write("multimodal-case-result", (dict(complete=False, phases=[phase]),))
            assert len(wire.calls) == 1
    return f

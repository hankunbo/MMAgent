"""A format-recovered accepted case must support later new-observation work."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import read_records
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.planner_control.models import Control
from tests.agent_provider.support import Documents, Reader, budget_for
from tests.agent_provider.test_accepted_review_reuse import state_from
from tests.agent_provider.test_review_continuation import continuation_fixture
from tests.evaluation_preparation.test_directional_review import reply


def fixture():
    prior = continuation_fixture(format_failure=True, single_batches=True)
    sub = run_batches(
        prior.agent,
        state=prior.owner.current(),
        documents=prior.documents,
        key="format:closure",
        round_id="A",
    )
    prior.acceptance.submit(sub, key="format:accepted")
    assert prior.acceptance.accept(key="format:accepted").status == "accepted"
    prior.documents.run_id = prior.owner.state.material.scope.run_id
    prior_binding = prior.old.binding.model_copy(update={"run_id": prior.documents.run_id})
    control = validate(
        Control, dict(binding=prior_binding, graph=prior_binding.graph, phase="stopped")
    )
    prior.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = prior.owner.current().model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:NEW_FACT_AFTER_FORMAT"
    policy = deepcopy(raw["policy"]["items"][0])
    policy.update(id="item:additional", observation_code="OBS:ADDITIONAL")
    observation = deepcopy(raw["material"]["observations"][0])
    observation["reference"].update(code=policy["observation_code"], checksum=digest("additional"))
    observation["value"] = "Additional synthetic observation; not another copy of the old fact"
    raw["material"]["observations"].append(observation)
    raw["policy"]["items"].append(policy)
    requirement = deepcopy(raw["policy"]["requirements"][0])
    requirement["id"] = policy["id"]
    raw["policy"]["requirements"].append(requirement)
    group = deepcopy(raw["policy"]["groups"][0])
    group.update(id="group:additional", slots=[[policy["id"]]])
    raw["policy"]["groups"].append(group)
    owner = Reader(state_from(raw))
    owner.confirmed_review_reader = AcceptedReviewReader(
        documents=prior.documents,
        requests=prior.budget,
        state_reader=prior.owner.current,
        acceptance_key="format:accepted",
        ancestor_reader=lambda rid, chain: prior.owner.confirmed_review_reader,
    )
    docs = Documents()
    config = prior.agent.provider.config
    budget = budget_for(
        prior_binding.model_copy(update={"run_id": owner.state.material.scope.run_id}), config
    )
    agent = StructuredRoleAgent(
        "review",
        provider=DeterministicFakeProvider(config, [reply] * 10),
        materials=owner,
        budget=budget,
    )
    return SimpleNamespace(
        prior=prior,
        owner=owner,
        agent=agent,
        documents=docs,
        budget=budget,
        acceptance=ResearchAcceptance(owner=owner, documents=docs, requests=budget),
    )


def test_accepted_format_history_survives_next_real_owner_chain():
    f = fixture()
    before = digest(f.prior.documents.saved), digest(f.prior.old.documents.saved)
    origin = f.owner.confirmed_review_reader.current(f.owner.current())
    assert origin.assessment_ids
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new:fact", round_id="A"
    )
    assert sub.schema_version == "research-batched-submission/9.0.0"
    f.acceptance.submit(sub, key="next:accepted")
    assert f.acceptance.accept(key="next:accepted").status == "accepted"
    records = read_records(f.acceptance.read(key="next:accepted")[0])
    assert records.schema_version == "case-observation-interpretation/2.11.0"
    failed = (
        f.prior.owner.confirmed_review_reader.current(f.prior.owner.current())
        .confirmed_format_failures[0]
        .request_id
    )
    assert failed not in records.request_ids
    assert len(records.candidate.items) == 6
    assert before == (digest(f.prior.documents.saved), digest(f.prior.old.documents.saved))


@pytest.mark.parametrize("mutation", ["ancestor", "raw", "source"])
def test_unverified_format_ancestor_cannot_authorize_new_work(mutation):
    f = fixture()
    if mutation == "ancestor":
        f.owner.confirmed_review_reader.ancestor_reader = None
    elif mutation == "raw":
        old = f.prior.old.documents
        key = next(
            k for k in old.saved if ":attempt:" in k and old.saved[k][0].get("error") == "schema"
        )
        old.saved[key][0]["raw_model_text"] += " "
    else:
        raw = f.owner.current().model_dump(mode="json")
        raw["material"]["sources"][0]["text"] += " changed"
        f.owner.state = state_from(raw)
    with pytest.raises(ValueError):
        run_batches(
            f.agent, state=f.owner.current(), documents=f.documents, key="new:fact", round_id="A"
        )
    assert not f.budget.records()


def test_text_successor_preserves_only_unattempted_declared_available_targets():
    from packages.multimodal_research.campaign_renewal import unattempted_targets

    rows = [
        dict(id=i, component="input" if i != "absent" else None, acquisition_method="vision")
        for i in ("done", "unknown", "new", "absent", "not_declared")
    ]
    original = SimpleNamespace(initial_targets=("done", "unknown", "new", "absent"))
    current = SimpleNamespace(
        initial_targets=(), parent=original, base=SimpleNamespace(ledger=rows)
    )
    assert unattempted_targets(current, {"unknown"}, {"done": object()}) == ["new"]
    original.parent = current
    with pytest.raises(ValueError, match="history_cycle"):
        unattempted_targets(current, set(), {})

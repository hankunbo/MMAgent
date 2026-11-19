"""Preflight original batch owner at size three before changing production renewal."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.planner_control.models import Control
from tests.agent_provider.batch_support import generate
from tests.agent_provider.support import Documents, Reader, budget_for
from tests.agent_provider.test_accepted_review_reuse import state_from
from tests.evaluation_preparation.test_directional_review import fixture, reply


def partition_fixture(*, batch_key="new", batch_size=3, pg=None, documents=None, binding=None):
    old = fixture()
    old.config = old.config.model_copy(
        update={
            "review_batching": old.config.review_batching.model_copy(
                update={"batch_size": batch_size}
            )
        }
    )
    old.owner.state = old.owner.state.model_copy(update={"config_checksum": digest(old.config)})
    if pg is not None:
        raw = old.owner.current().model_dump(mode="json")
        raw["material"]["scope"].update(
            case_code=pg.binding.case_code,
            run_id=pg.binding.run_id,
            graph=pg.binding.graph.model_dump(mode="json"),
        )
        old.owner.state = state_from(raw)
        old.binding, old.documents, old.budget = pg.binding, pg.documents, pg.budget
        old.agent.budget = pg.budget
        old.acceptance = ResearchAcceptance(
            owner=old.owner, documents=pg.documents, requests=pg.budget
        )
    old.agent.provider = DeterministicFakeProvider(old.config, [reply] * 10)
    submission = generate(old)
    old.acceptance.submit(submission, key="original")
    assert old.acceptance.accept(key="original").status == "accepted"
    if pg is None:
        old.documents.run_id = old.owner.state.material.scope.run_id
        control = validate(
            Control, dict(binding=old.binding, graph=old.binding.graph, phase="stopped")
        )
        old.documents.sessions = SimpleNamespace(read=lambda _: control)
    before = digest(old.documents.read("accepted:original")), digest(old.budget.journal.read())
    raw = old.owner.state.model_dump(mode="json")
    run_id = binding.run_id if binding else "RUN:BATCH3_INCREMENT"
    raw["material"]["scope"]["run_id"] = run_id
    for number in range(3):
        item = deepcopy(raw["policy"]["items"][0])
        item.update(id=f"item:new{number}", observation_code=f"OBS:NEW{number}")
        obs = deepcopy(raw["material"]["observations"][0])
        obs["reference"].update(
            code=item["observation_code"], checksum=digest(("synthetic", number))
        )
        obs["value"] = f"Additional synthetic observed feature {number}"
        raw["material"]["observations"].append(obs)
        raw["policy"]["items"].append(item)
        requirement = deepcopy(raw["policy"]["requirements"][0])
        requirement["id"] = item["id"]
        raw["policy"]["requirements"].append(requirement)
        group = deepcopy(raw["policy"]["groups"][0])
        group.update(id=f"group:new{number}", slots=[[item["id"]]])
        raw["policy"]["groups"].append(group)
    owner = Reader(state_from(raw))
    owner.confirmed_review_reader = AcceptedReviewReader(
        documents=old.documents,
        requests=old.budget,
        state_reader=old.owner.current,
        acceptance_key="original",
    )
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    persistent = documents is not None
    documents = documents or Documents()
    budget = budget_for(
        binding or old.binding.model_copy(update={"run_id": run_id}),
        old.config,
        journal=RuntimeAgentJournal(documents) if persistent else None,
    )
    provider = DeterministicFakeProvider(old.config, [reply] * 5)
    agent = StructuredRoleAgent("review", provider=provider, materials=owner, budget=budget)
    candidate = run_batches(
        agent, state=owner.current(), documents=documents, key=batch_key, round_id="A"
    )
    pending = set(candidate.plan.assessment_ids) - set(candidate.historical.assessment_ids)
    assert (
        len(candidate.batches)
        == len(budget.records())
        == (len(pending) + batch_size - 1) // batch_size
    )
    assert {i.assessment_id for b in candidate.batches for i in b.candidate.items} == pending
    assert all(len(b.candidate.items) <= batch_size for b in candidate.batches)
    originals = {i.assessment_id: i for i in submission.candidate.items}
    assert all(
        i == originals[i.assessment_id]
        for i in candidate.candidate.items
        if i.assessment_id in candidate.historical.assessment_ids
    )
    acceptance = ResearchAcceptance(owner=owner, documents=documents, requests=budget)
    assert acceptance.submit(candidate, key="new")["status"] == "submitted_not_accepted"
    assert acceptance.accept(key="new").status == "accepted"
    assert before == (
        digest(old.documents.read("accepted:original")),
        digest(old.budget.journal.read()),
    )
    return SimpleNamespace(
        old=old,
        owner=owner,
        agent=agent,
        documents=documents,
        budget=budget,
        acceptance=acceptance,
        submission=candidate,
    )


def test_new_observations_and_old_unassessed_use_only_pending_batches():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.acceptance_models import read_records

    f = partition_fixture()
    records = read_records(f.acceptance.read(key="new")[0])
    assert records.schema_version == "case-observation-interpretation/2.12.0"
    assert records.batch_plan.schema_version == "review-batch-plan/1.1.0"
    report = read_research_report(f.acceptance, key="new", parent=f.owner.current())
    assert "review-batch-plan/1.1.0" in render_research_html(report)
    before = len(f.budget.records()), len(f.documents.audit)
    replay = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    assert replay == f.submission
    f.acceptance.accept(key="new")
    assert before == (len(f.budget.records()), len(f.documents.audit))


@pytest.mark.parametrize("change", ["origin", "reused_ids"])
def test_changed_reuse_partition_is_rejected_without_new_request(change):
    from packages.evidence_details.review_batches import partition_plan
    from packages.evidence_details.review_continuation import check_continued

    f = partition_fixture()
    before = len(f.budget.records()), len(f.documents.audit)
    plan = f.submission.plan
    reused = set(plan.reused_assessment_ids)
    origin = plan.origin_checksum
    if change == "origin":
        origin = digest("wrong-origin")
    else:
        reused.remove(sorted(reused)[0])
    changed = f.submission.model_copy(update={"plan": partition_plan(plan, reused, origin)})
    # Exact request/result binding rejects the changed plan before the later
    # origin-equivalence check; this is the expected stronger early rejection.
    with pytest.raises(ValueError, match="batch_result_binding"):
        check_continued(
            f.owner.current(), changed, owner=f.owner, documents=f.documents, requests=f.budget
        )
    assert before == (len(f.budget.records()), len(f.documents.audit))


def test_partitioned_acceptance_remains_reusable_in_another_successor():
    from tests.agent_provider.test_accepted_review_reuse import accepted_successor

    f = partition_fixture()
    next_case = accepted_successor(f, acceptance_key="new")
    sub = run_batches(
        next_case.agent,
        state=next_case.owner.current(),
        documents=next_case.documents,
        key="third",
        round_id="A",
    )
    assert sub.historical.schema_version == "accepted-review-origin/1.3.0"
    next_case.acceptance.submit(sub, key="third")
    assert next_case.acceptance.accept(key="third").status == "accepted"

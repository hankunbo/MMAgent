"""Accepted local judgments retain old requests; new facts use new requests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState, read_records
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.planner_control.models import Control
from tests.agent_provider.batch_support import generate
from tests.agent_provider.support import Documents, Reader, budget_for
from tests.evaluation_preparation.test_directional_review import fixture, reply


def state_from(raw):
    material = raw["material"]
    material["scope"]["policy_checksum"] = digest(raw["policy"])
    material["policy_ref"]["checksum"] = digest(raw["policy"])
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    return validate(DomainState, raw)


def accepted_fixture(
    *,
    new_observation=False,
    pg=None,
    next_documents=None,
    next_binding=None,
    corrected=False,
    batch_size=1,
):
    old = fixture()
    old.config = old.config.model_copy(
        update={
            "review_batching": old.config.review_batching.model_copy(
                update={"batch_size": batch_size}
            )
        }
    )
    old.owner.state = old.owner.state.model_copy(update={"config_checksum": digest(old.config)})
    if corrected and pg is None:
        raw = old.owner.state.model_dump(mode="json")
        raw["material"]["scope"]["case_code"] = "CASE:CORRECTED_REUSE"
        old.owner.state = state_from(raw)
        old.binding = old.binding.model_copy(update={"case_code": "CASE:CORRECTED_REUSE"})
    if pg is not None:
        raw = old.owner.state.model_dump(mode="json")
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
    replies = [reply] * 10
    if corrected:
        from tests.agent_provider.test_review_optional_null import malformed

        replies.insert(0, malformed)
    old.agent.provider = DeterministicFakeProvider(old.config, replies)
    submission = generate(old)
    old.acceptance.submit(submission, key="old-accepted")
    assert old.acceptance.accept(key="old-accepted").status == "accepted"
    if pg is None:
        old.documents.run_id = old.owner.state.material.scope.run_id
        control = validate(
            Control, dict(binding=old.binding, graph=old.binding.graph, phase="stopped")
        )
        old.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = old.owner.state.model_dump(mode="json")
    run_id = next_binding.run_id if next_binding else "RUN:ACCEPTED_NEXT"
    raw["material"]["scope"]["run_id"] = run_id
    if new_observation:
        p = deepcopy(raw["policy"]["items"][0])
        p["id"], p["observation_code"] = "item:additional", "OBS:ADDITIONAL"
        observation = deepcopy(raw["material"]["observations"][0])
        observation["reference"].update(code=p["observation_code"], checksum=digest("additional"))
        observation["value"] = "Additional synthetic observed feature"
        raw["material"]["observations"].append(observation)
        raw["policy"]["items"].append(p)
        requirement = deepcopy(raw["policy"]["requirements"][0])
        requirement["id"] = p["id"]
        raw["policy"]["requirements"].append(requirement)
        group = deepcopy(raw["policy"]["groups"][0])
        group.update(id="group:additional", slots=[[p["id"]]])
        raw["policy"]["groups"].append(group)
    owner = Reader(state_from(raw))
    owner.confirmed_review_reader = AcceptedReviewReader(
        documents=old.documents,
        requests=old.budget,
        state_reader=old.owner.current,
        acceptance_key="old-accepted",
    )
    docs = next_documents or Documents()
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    budget = budget_for(
        next_binding or old.binding.model_copy(update={"run_id": run_id}),
        old.config,
        journal=RuntimeAgentJournal(docs) if next_documents else None,
    )
    provider = DeterministicFakeProvider(old.config, [reply] * 10)
    agent = StructuredRoleAgent("review", provider=provider, materials=owner, budget=budget)
    return SimpleNamespace(
        old=old,
        owner=owner,
        agent=agent,
        documents=docs,
        budget=budget,
        acceptance=ResearchAcceptance(owner=owner, documents=docs, requests=budget),
    )


@pytest.mark.parametrize("new_observation", [False, True])
def test_original_acceptance_is_reused_without_request_and_report_preserves_origin(new_observation):
    from agents.expert_agent.reporting import read_research_report, render_research_html

    f = accepted_fixture(new_observation=new_observation)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    reused = set(sub.historical.assessment_ids)
    assert len(f.budget.records()) == len(sub.candidate.items) - len(reused)
    assert sub.schema_version == "research-batched-submission/6.0.0"
    originals = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    assert all(
        i == originals[i.assessment_id] for i in sub.candidate.items if i.assessment_id in reused
    )
    assert f.acceptance.submit(sub, key="next")["status"] == "submitted_not_accepted"
    receipt = f.acceptance.accept(key="next")
    assert receipt.status == "accepted"
    records = read_records(f.acceptance.read(key="next")[0])
    assert records.schema_version == "case-observation-interpretation/2.8.0"
    report = read_research_report(f.acceptance, key="next", parent=f.owner.current())
    assert "accepted-review-origin/1.0.0" in render_research_html(report)
    audit, requests = len(f.documents.audit), len(f.budget.records())
    assert f.acceptance.accept(key="next") == receipt
    assert (len(f.documents.audit), len(f.budget.records())) == (audit, requests)
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))


@pytest.mark.parametrize("change", ["observation", "source", "receipt", "unknown", "owner"])
def test_changed_or_unverified_origin_stops_before_new_request(change):
    f = accepted_fixture(new_observation=True)
    if change in {"observation", "source"}:
        raw = f.owner.state.model_dump(mode="json")
        if change == "observation":
            raw["material"]["observations"][0]["value"] += " changed"
        else:
            raw["material"]["sources"][0]["text"] += " changed"
        f.owner.state = state_from(raw)
    elif change == "receipt":
        f.old.documents.saved["accepted:old-accepted"][3]["records_checksum"] = digest("wrong")
    elif change == "unknown":
        f.old.budget.journal.append(dict(phase="reserved", request_id="unknown"))
    else:
        f.owner.withdrawn = True
    with pytest.raises(ValueError):
        run_batches(
            f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
        )
    assert not f.budget.records()


def test_accepted_corrected_batches_reuse_without_replaying_failed_response():
    f = accepted_fixture(new_observation=True, corrected=True)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    assert sum(r.error == "schema" for r in f.old.budget.records()) == 1
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    # Two fixture items have no accepted direction and are deliberately not
    # reusable. Only previously assessed items must avoid another request.
    reused = set(sub.historical.assessment_ids)
    expected = set(sub.plan.assessment_ids) - reused
    assert "item:additional" in expected
    assert len(f.budget.records()) == len(expected)
    assert {i.assessment_id for i in sub.candidate.items if i.status == "assessed"} >= reused
    assert f.acceptance.submit(sub, key="next")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="next").status == "accepted"
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))
    audit = len(f.documents.audit)
    assert (
        run_batches(
            f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
        )
        == sub
    )
    assert len(f.budget.records()) == len(expected) and len(f.documents.audit) == audit


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("raw", "pending_format_response_binding"),
        ("feedback", "review_correction_binding"),
        ("unknown", "accepted_review_reuse_unknown_request"),
        ("finish", "pending_format_response_binding"),
        ("record_output", "review_correction_binding"),
    ],
)
def test_corrected_origin_cannot_hide_unknown_or_altered_error(fault, error):
    f = accepted_fixture(new_observation=True, corrected=True)
    if fault in {"raw", "finish"}:
        key = next(
            k
            for k, v in f.old.documents.saved.items()
            if ":attempt:" in k and v[0].get("error") == "schema"
        )
        if fault == "raw":
            f.old.documents.saved[key][0]["raw_model_text"] += " "
        else:
            f.old.documents.saved[key][0]["finish_reason"] = "length"
    elif fault == "feedback":
        event = next(
            x
            for x in f.old.budget.journal.events
            if x.get("schema_version") == "review-schema-correction/1.0.0"
        )
        event["failed_record_checksum"] = digest("changed")
    elif fault == "record_output":
        event = next(
            x
            for x in f.old.budget.journal.events
            if x.get("phase") == "completed" and x["record"].get("error") == "schema"
        )
        event["record"]["output_checksum"] = digest("changed")
    else:
        f.old.budget.journal.append(dict(phase="reserved", request_id="unknown"))
    with pytest.raises(ValueError, match="^" + error + "$"):
        run_batches(
            f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
        )
    assert not f.budget.records()


def accepted_successor(f, *, acceptance_key="next", documents=None, binding=None):
    """Second successor, retaining original owners for both accepted ancestors."""
    raw = f.owner.current().model_dump(mode="json")
    run_id = binding.run_id if binding else "RUN:ACCEPTED_THIRD"
    raw["material"]["scope"]["run_id"] = run_id
    p = deepcopy(raw["policy"]["items"][0])
    p["id"], p["observation_code"] = "item:third", "OBS:THIRD"
    obs = deepcopy(raw["material"]["observations"][0])
    obs["reference"].update(code=p["observation_code"], checksum=digest("third"))
    obs["value"] = "Third synthetic observed feature"
    raw["material"]["observations"].append(obs)
    raw["policy"]["items"].append(p)
    req = deepcopy(raw["policy"]["requirements"][0])
    req["id"] = p["id"]
    raw["policy"]["requirements"].append(req)
    group = deepcopy(raw["policy"]["groups"][0])
    group.update(id="group:third", slots=[[p["id"]]])
    raw["policy"]["groups"].append(group)
    owner = Reader(state_from(raw))
    ancestor = f.owner.confirmed_review_reader

    def resolve(rid, lineage):
        if rid != ancestor.documents.run_id:
            raise ValueError("unexpected_test_ancestor")
        return AcceptedReviewReader(
            documents=ancestor.documents,
            requests=ancestor.requests,
            state_reader=ancestor.state_reader,
            acceptance_key=ancestor.acceptance_key,
            lineage=lineage,
        )

    if documents is None:
        f.documents.run_id = f.owner.state.material.scope.run_id
        previous_binding = f.old.binding.model_copy(update={"run_id": f.documents.run_id})
        control = validate(
            Control, dict(binding=previous_binding, graph=previous_binding.graph, phase="stopped")
        )
        f.documents.sessions = SimpleNamespace(read=lambda _: control)
    owner.confirmed_review_reader = AcceptedReviewReader(
        documents=f.documents,
        requests=f.budget,
        state_reader=f.owner.current,
        acceptance_key=acceptance_key,
        ancestor_reader=resolve,
    )
    docs = documents or Documents()
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    budget = budget_for(
        binding or f.old.binding.model_copy(update={"run_id": run_id}),
        f.old.config,
        journal=RuntimeAgentJournal(docs) if documents else None,
    )
    agent = StructuredRoleAgent(
        "review",
        provider=DeterministicFakeProvider(f.old.config, [reply] * 10),
        materials=owner,
        budget=budget,
    )
    return SimpleNamespace(
        owner=owner,
        agent=agent,
        documents=docs,
        budget=budget,
        acceptance=ResearchAcceptance(owner=owner, documents=docs, requests=budget),
    )


def test_two_accepted_successors_keep_exact_original_requests_and_bounded_lineage():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.review_batches import request_for_item

    f = accepted_fixture(new_observation=True)
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    f.acceptance.submit(sub, key="next")
    assert f.acceptance.accept(key="next").status == "accepted"
    before = digest(f.old.documents.saved), digest(f.documents.saved)
    g = accepted_successor(f)
    newer = run_batches(
        g.agent, state=g.owner.current(), documents=g.documents, key="third", round_id="A"
    )
    assert newer.schema_version == "research-batched-submission/7.0.0"
    assert len(g.budget.records()) == len(newer.candidate.items) - len(
        newer.historical.assessment_ids
    )
    assert "item:additional" in newer.historical.assessment_ids
    assert request_for_item(newer, "item:additional") == request_for_item(sub, "item:additional")
    for item in sub.historical.assessment_ids:
        assert request_for_item(newer, item) == request_for_item(sub, item)
    g.acceptance.submit(newer, key="third")
    assert g.acceptance.accept(key="third").status == "accepted"
    records = read_records(g.acceptance.read(key="third")[0])
    assert records.schema_version == "case-observation-interpretation/2.9.0"
    report = read_research_report(g.acceptance, key="third", parent=g.owner.current())
    assert "accepted-review-origin/1.1.0" in render_research_html(report)
    assert before == (digest(f.old.documents.saved), digest(f.documents.saved))
    g.owner.confirmed_review_reader.ancestor_reader = None
    with pytest.raises(ValueError, match="ancestor_owner_required"):
        g.owner.confirmed_review_reader.current(g.owner.current())
    g.owner.confirmed_review_reader.lineage = (f.documents.run_id,)
    with pytest.raises(ValueError, match="cycle_or_depth"):
        g.owner.confirmed_review_reader.current(g.owner.current())

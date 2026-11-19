"""Cross-window confirmed responses are provenance, never fresh model requests."""

import pytest

from packages.grounding.models import digest, validate
from tests.evaluation_preparation.test_directional_review import fixture


def test_continuation_keeps_medical_binding_but_not_execution_identity():
    from packages.evidence_details.review_continuation import semantic_binding

    state = fixture().owner.current()
    raw = state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:NEW"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    newer = validate(type(state), raw)
    assert semantic_binding(state) == semantic_binding(newer)
    raw["material"]["goal_summary"] += " changed proposition"
    raw["policy"]["target_summary"] = raw["material"]["goal_summary"]
    raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
    raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    assert semantic_binding(state) != semantic_binding(validate(type(state), raw))


def test_continuation_without_original_owner_is_rejected():
    from packages.evidence_details.review_continuation import require_reader

    with pytest.raises(ValueError, match="original_reader_required"):
        require_reader(object())


def test_production_successor_role_switch_passes_through_continuation_owner():
    from packages.evidence_details.review_continuation import ContinuationOwner
    from packages.evidence_details.successor_owner import ResearchSuccessorOwner
    from tests.agent_provider.support import Documents

    f = fixture()
    owner = ResearchSuccessorOwner(
        initial=f.owner.current(),
        documents=Documents(),
        verify_dependencies=lambda _: None,
        read_image=lambda _: b"",
        config=f.config,
        role="vision",
    )
    wrapped = ContinuationOwner(owner, None)
    assert wrapped.current().prompt_checksum != f.owner.current().prompt_checksum
    wrapped.role = "review"
    assert owner.role == "review"
    assert wrapped.current() == f.owner.current()


def continuation_fixture(
    pg=None, next_documents=None, next_binding=None, format_failure=False, single_batches=False
):
    from types import SimpleNamespace

    from agents.provider_agents import StructuredRoleAgent
    from agents.provider_agents.batched_review import run_batches
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.review_continuation import ConfirmedReviewReader
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.provider import ProviderError
    from packages.planner_control.models import Control
    from tests.agent_provider.support import Documents, Reader, budget_for
    from tests.evaluation_preparation.test_directional_review import reply

    old = fixture()
    config = old.config.model_copy(
        update={
            "review_batching": old.config.review_batching.model_copy(
                update={"batch_size": 1 if single_batches else 2}
            )
        }
    )
    if format_failure:
        config = config.model_copy(
            update={"transport": config.transport.model_copy(update={"max_retries": 0})}
        )
    old.config = config
    old.owner.state = old.owner.state.model_copy(update={"config_checksum": digest(config)})
    if pg is not None:
        raw = old.owner.state.model_dump(mode="json")
        raw["material"]["scope"].update(
            case_code=pg.binding.case_code,
            run_id=pg.binding.run_id,
            graph=pg.binding.graph.model_dump(mode="json"),
        )
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        old.owner.state = validate(type(old.owner.state), raw)
        old.binding, old.documents, old.budget = pg.binding, pg.documents, pg.budget
        old.agent.budget = old.budget
    old.agent.provider = DeterministicFakeProvider(config, [reply] * 3)
    if format_failure:
        from tests.agent_provider.test_review_optional_null import malformed

        old.agent.provider = DeterministicFakeProvider(config, [reply, reply, malformed])

    def stop():
        if not format_failure and old.documents.read("review-batches:origin:A:batch:001:result"):
            raise ProviderError("cancelled")

    with pytest.raises(ProviderError, match="schema|no_result" if format_failure else "cancelled"):
        run_batches(
            old.agent,
            state=old.owner.current(),
            documents=old.documents,
            key="origin:A",
            round_id="A",
            before_request=stop,
        )
    assert len(old.budget.records()) == (3 if format_failure else 2)
    old.documents.write("live-stage-intent:review", ({"event": {"code": "origin"}},))
    if pg is None:
        old.documents.run_id = old.owner.current().material.scope.run_id
        control = validate(
            Control, dict(binding=old.binding, graph=old.binding.graph, phase="stopped")
        )
        old.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = old.owner.current().model_dump(mode="json")
    run_id = next_binding.run_id if next_binding else "RUN:NEXT"
    raw["material"]["scope"]["run_id"] = run_id
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    if format_failure:
        from packages.llm_provider.config import review_format_upgrade

        config = review_format_upgrade(config)
        raw["config_checksum"] = digest(config)
    owner = Reader(validate(type(old.owner.state), raw))
    owner.confirmed_review_reader = ConfirmedReviewReader(
        documents=old.documents,
        requests=old.budget,
        state_reader=old.owner.current,
        format_config=config if format_failure else None,
    )
    docs = next_documents or Documents()
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    budget = budget_for(
        next_binding or old.binding.model_copy(update={"run_id": run_id}),
        config,
        journal=RuntimeAgentJournal(docs) if next_documents else None,
    )
    provider = DeterministicFakeProvider(config, [reply] * (10 if single_batches else 1))
    agent = StructuredRoleAgent("review", provider=provider, materials=owner, budget=budget)
    service = ResearchAcceptance(owner=owner, documents=docs, requests=budget)
    return SimpleNamespace(
        old=old,
        owner=owner,
        agent=agent,
        documents=docs,
        budget=budget,
        acceptance=service,
        provider=provider,
    )


def test_four_original_items_one_new_request_owner_accepts_scores_and_reports():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from agents.provider_agents.batched_review import run_batches
    from packages.evidence_details.acceptance_models import read_records

    f = continuation_fixture()
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new:A", round_id="A"
    )
    assert len(sub.historical.submission.candidate.items) == 4
    assert len(sub.candidate.items) == 5 and len(f.budget.records()) == 1
    assert f.acceptance.submit(sub, key="close")["status"] == "submitted_not_accepted"
    receipt = f.acceptance.accept(key="close")
    assert receipt.status == "accepted"
    records = read_records(f.acceptance.read(key="close")[0])
    assert records.schema_version.endswith("/2.7.0") and len(records.request_ids) == 3
    assert (
        tuple(
            i
            for i in sub.candidate.items
            if i.assessment_id
            in {x.assessment_id for x in sub.historical.submission.candidate.items}
        )
        == sub.historical.submission.candidate.items
    )
    report = read_research_report(f.acceptance, key="close", parent=f.owner.current())
    assert len(report["scores"]["reviews"]) == 5
    assert "confirmed-review-origin" in render_research_html(report)
    audit = len(f.documents.audit)
    assert f.acceptance.accept(key="close") == receipt and len(f.documents.audit) == audit
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))


@pytest.mark.parametrize("change", ["source", "pending_claim", "response", "unknown"])
def test_origin_change_or_unknown_prevents_any_new_request(change):
    from agents.provider_agents.batched_review import run_batches

    f = continuation_fixture()
    if change == "source":
        f.owner.withdrawn = True
    elif change == "pending_claim":
        f.old.documents.write("review-batches:origin:A:batch:002:claim", ({"unknown": True},))
    elif change == "response":
        key = "review-batches:origin:A:batch:000:response"
        f.old.documents.saved[key][0]["raw_model_text"] += " "
    else:
        f.old.budget.journal.append({"phase": "reserved", "request_id": "unknown"})
    with pytest.raises(ValueError):
        run_batches(
            f.agent, state=f.owner.current(), documents=f.documents, key="new:A", round_id="A"
        )
    assert not f.budget.records()

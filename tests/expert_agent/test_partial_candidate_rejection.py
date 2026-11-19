"""A rejected batch never edits its response or discards other confirmed batches."""

from threading import Event

import pytest

from agents.expert_agent.reporting import read_research_report
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from packages.llm_provider.provider import ProviderError
from tests.agent_provider.batch_support import batch_reply, fixture
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture
from tests.expert_agent.support import run, setup


def test_local_candidate_stop_preserves_typed_cause_without_model_text(monkeypatch):
    f = assembled()

    def stopped(*args, **kwargs):
        raise ProviderError("deadline", fields=("final_review",))

    monkeypatch.setattr(f.service, "_candidate", stopped)
    result = run(f, role="review")
    assert result["status"] == "safe_stop"
    key = f.frame.event.code + ":candidate-failure"
    assert f.service.store.read(key) == (
        {
            "schema_version": "research-candidate-failure/1.0.0",
            "category": "provider",
            "code": "deadline",
            "fields": ["final_review"],
        },
    )
    old = f.documents.saved.copy()
    assert run(f, role="review") == result and f.documents.saved == old
    assert f.budget.used == 0


def assembled(*, dynamic=True):
    batch = dynamic_fixture() if dynamic else fixture()
    base = setup(config=batch.config)
    raw = batch.owner.current().model_dump(mode="json")
    material = raw["material"]
    material["scope"].update(
        case_code=base.binding.case_code,
        run_id=base.binding.run_id,
        graph=base.binding.graph.model_dump(mode="json"),
        input_checksum=base.binding.input_checksum,
        visible_checksum=base.binding.visible_checksum,
    )
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    batch.owner.state = validate(DomainState, raw)
    return setup(
        config=batch.config,
        binding=base.binding,
        reader=batch.owner,
        documents=batch.documents,
        budget=base.budget,
    )


@pytest.mark.parametrize("field", ["scope_checksum", "plan_checksum", "batch_id"])
def test_one_rejected_batch_preserves_original_confirmed_results(field):
    f = assembled()
    parent = f.reader.current()

    def wrong_binding(body):
        raw = batch_reply(body)
        raw[field] = "batch:999" if field == "batch_id" else digest("wrong model binding")
        return raw

    f.provider.transport.replies = [batch_reply, wrong_binding]
    result = run(f, role="review")
    assert result["domain_status"] == "accepted"
    assert f.budget.used == 2 and len(f.provider.transport.calls) == 2
    assert f.budget.records()[-1].error == "scope"
    report = read_research_report(f.acceptance, key=result["acceptance_key"], parent=parent)
    assert len(report["scores"]["reviews"]) == 3
    assert len(report["scores"]["review_completion"]["pending_ids"]) == 2
    cause = f.service.store.read(f.frame.event.code + ":A:partial-cause")[0]
    assert cause["error"] == "scope" and cause["fields"] == ["batch_candidate"]
    assert cause["failed_request_not_retried"]
    saved, count = f.documents.saved.copy(), len(f.documents.audit)
    assert run(f, role="review") == result
    assert f.documents.saved == saved and len(f.documents.audit) == count
    assert f.budget.used == 2


@pytest.mark.parametrize("change", ["parent", "cancel", "deadline", "source"])
def test_global_authority_change_cannot_be_downgraded_to_partial(change):
    f = assembled()
    signal = Event()

    def wrong_binding_and_changed_owner(body):
        raw = batch_reply(body)
        raw["scope_checksum"] = digest("wrong model binding")
        if change == "parent":
            f.reader.state = f.reader.state.model_copy(
                update={"nodes": (*f.reader.state.nodes, "GOAL:CHANGED")}
            )
        elif change == "cancel":
            signal.set()
        elif change == "deadline":
            f.budget.clock = lambda: f.binding.deadline
        else:
            original = f.reader.current

            def withdrawn():
                original()
                raise ValueError("source_withdrawn")

            f.reader.current = withdrawn
        return raw

    f.provider.transport.replies = [batch_reply, wrong_binding_and_changed_owner]
    result = run(f, role="review", cancel=signal)
    assert result["status"] == "safe_stop"
    assert not f.context.installs
    assert f.acceptance.read(key=f.frame.event.code + ":domain") is None
    assert f.budget.used == 2


def test_legacy_fixed_policy_does_not_gain_partial_acceptance():
    f = assembled(dynamic=False)

    def wrong_binding(body):
        raw = batch_reply(body)
        raw["scope_checksum"] = digest("wrong binding")
        return raw

    f.provider.transport.replies = [batch_reply, wrong_binding]
    assert run(f, role="review")["status"] == "safe_stop"
    assert not f.context.installs and f.budget.used == 2


def test_no_confirmed_batch_stops_without_inventing_partial_result():
    f = assembled()

    def wrong_binding(body):
        raw = batch_reply(body)
        raw["scope_checksum"] = digest("wrong first reply")
        return raw

    f.provider.transport.replies = [wrong_binding]
    assert run(f, role="review")["status"] == "safe_stop"
    assert not f.context.installs and f.budget.used == 1
    assert f.acceptance.read(key=f.frame.event.code + ":domain") is None

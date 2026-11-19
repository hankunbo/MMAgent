"""Legacy partial + only pending judgments, original owner aggregation, no API."""

from types import SimpleNamespace

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from packages.evaluation_preparation.dynamic_stages import rebuild
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState, read_records
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    AcceptedOriginReader,
    AcceptedReviewSet,
    AggregationPermission,
    AggregationRequests,
    HistoricalRequests,
)
from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.batch_support import generate
from tests.agent_provider.support import budget_for
from tests.agent_provider.test_pending_truncated_review import partial
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture
from tests.expert_agent.test_batch_assembly import LiveDocuments


def move_state(state, run_id):
    raw = state.model_dump(mode="json")
    material = raw["material"]
    material["scope"]["run_id"] = run_id
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    return validate(DomainState, raw)


def combined(monkeypatch, *, resize=False):
    old, partial_sub = partial(monkeypatch)
    old_ids = {i.assessment_id for i in partial_sub.candidate.items}
    pending = {p.id for p in old.owner.current().policy.items} - old_ids
    new = dynamic_fixture(
        17,
        config_path=(
            "configs/agents/fairvlmed_relay_review_single_v1.json"
            if resize
            else "configs/agents/fairvlmed_relay_review_shape_v1.json"
        ),
    )
    state = move_state(old.owner.current(), "RUN:LEGACY_PENDING")
    state = state.model_copy(update={"config_checksum": digest(new.config)})
    policy = state.policy.model_dump(mode="json")
    policy["items"] = [p for p in policy["items"] if p["id"] in pending]
    new.owner.state = rebuild(
        state, validate(type(state.policy), policy), state.material.observations
    )
    new.binding = old.binding.model_copy(update={"run_id": state.material.scope.run_id})
    new.budget = budget_for(new.binding, new.config)
    new.agent.budget = new.budget
    new.acceptance.requests = new.budget
    new.documents.run_id = new.binding.run_id
    control = old.documents.sessions.read(old.binding.run_id)
    new.documents.sessions = SimpleNamespace(
        read=lambda _: control.model_copy(update={"binding": new.binding})
    )
    new_sub = generate(new)
    assert {i.assessment_id for i in new_sub.candidate.items} == pending
    assert new.acceptance.submit(new_sub, key="remaining")["status"] == "submitted_not_accepted"
    assert new.acceptance.accept(key="remaining").status == "accepted"
    current = move_state(old.owner.current(), "RUN:LEGACY_AGGREGATE")
    current = current.model_copy(update={"config_checksum": digest(new.config)})
    readers = [
        AcceptedOriginReader(
            documents=f.documents,
            requests=HistoricalRequests(f.budget.journal),
            state_reader=f.owner.current,
            acceptance_key=key,
            current_config=new.config if resize else None,
        )
        for f, key in ((old, "partial"), (new, "remaining"))
    ]
    source = AcceptedReviewSet(readers, new.config)
    docs = LiveDocuments()
    docs.run_id = current.material.scope.run_id
    docs.root = SimpleNamespace(checksum=current.material.scope.graph.checksum)
    docs.sessions = SimpleNamespace(
        read=lambda _: control.model_copy(
            update={
                "binding": old.binding.model_copy(
                    update={
                        "run_id": docs.run_id,
                        "case_code": current.material.scope.case_code,
                        "input_checksum": current.material.scope.input_checksum,
                        "visible_checksum": current.material.scope.visible_checksum,
                    }
                )
            }
        )
    )
    owner = SimpleNamespace(current=lambda: current, accepted_review_set=source)
    requests = AggregationRequests(source, owner, RuntimeAgentJournal(docs))
    return SimpleNamespace(
        old=old, new=new, source=source, owner=owner, docs=docs, requests=requests
    )


@pytest.mark.parametrize("resize", [False, True])
def test_legacy_judgments_keep_exact_values_and_original_requests(monkeypatch, resize):
    f = combined(monkeypatch, resize=resize)
    before = [(digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)]
    permission, sub = AggregationPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.new.config
    )
    assert sub.schema_version == "research-batched-submission/13.0.0"
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    assert service.submit(sub, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    receipt = service.accept(key=ACCEPTANCE_KEY)
    assert receipt.status == "accepted" and not receipt.new_medical_review
    saved = read_records(service.read(key=ACCEPTANCE_KEY)[0])
    assert saved.schema_version == "case-observation-interpretation/2.15.0"
    assert saved.candidate.items == sub.candidate.items
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=f.owner.current())
    assert not report["scores"]["review_completion"]["pending_ids"]
    assert "零新模型请求" in render_research_html(report)
    original = {
        i.assessment_id: i
        for x, k in ((f.old, "partial"), (f.new, "remaining"))
        for i in read_records(x.acceptance.read(key=k)[0]).candidate.items
    }
    assert {i.assessment_id: i for i in sub.candidate.items} == original
    audit = len(f.docs.audit)
    assert service.accept(key=ACCEPTANCE_KEY) == receipt and len(f.docs.audit) == audit
    assert not f.requests.journal.read()
    assert before == [
        (digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)
    ]
    if resize:
        assert all(
            len(b["assessment_ids"]) == 1
            for b in f.new.documents.read("review-batches:synthetic-batches:plan")[0]["batches"]
        )


@pytest.mark.parametrize("change", ["model", "token_limit", "batch_count", "missing_proof"])
def test_resize_is_not_general_config_compatibility(monkeypatch, change):
    f = combined(monkeypatch, resize=True)
    reader = f.source.readers[0]
    if change == "missing_proof":
        key = next(k for k in f.old.documents.saved if k.endswith(":response"))
        f.old.documents.saved[key][0].pop("provider_config")
    else:
        raw = reader.current_config.model_dump(mode="json")
        if change == "model":
            raw["transport"]["model"] = "different"
        elif change == "token_limit":
            next(r for r in raw["roles"] if r["role"] == "review")["max_output_tokens"] += 1
        else:
            raw["review_batching"]["max_batches"] -= 1
        from packages.llm_provider.config import read_provider_config

        reader.current_config = read_provider_config(raw)
        state = f.owner.current().model_copy(
            update={"config_checksum": digest(reader.current_config)}
        )
        f.owner.current = lambda: state
    with pytest.raises(ValueError):
        reader.read(f.owner.current())
    assert not f.docs.saved


@pytest.mark.parametrize("fault", ["duplicate", "config", "source"])
def test_legacy_aggregation_does_not_weaken_reuse_contract(monkeypatch, fault):
    f = combined(monkeypatch)
    if fault == "duplicate":
        f.source.readers = (f.source.readers[0], f.source.readers[0])
    elif fault == "config":
        f.new.owner.state = f.new.owner.state.model_copy(
            update={"config_checksum": digest("changed")}
        )
    else:
        state = f.new.owner.current()
        f.new.owner.state = state.model_copy(
            update={"material": state.material.model_copy(update={"sources": ()})}
        )
    with pytest.raises(ValueError):
        f.source.current(f.owner.current())
    assert not f.docs.saved

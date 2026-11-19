"""Actual 7.x partial acceptance -> HTTP-only client -> zero-request composition."""

import json
from pathlib import Path

import pytest

from agents.expert_agent.reporting import read_research_report
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    AggregationPermission,
)
from packages.grounding.models import digest
from packages.llm_provider.config import with_http_client, with_network_route
from packages.llm_provider.network_route import prepare_route
from tests.agent_provider import batch_support
from tests.expert_agent import test_legacy_accepted_aggregation as legacy


def fixture(monkeypatch):
    load = batch_support.load_config
    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setattr(batch_support, "load_config", lambda p: with_network_route(load(p), route))
    f = legacy.combined(monkeypatch)
    client = json.loads(Path("configs/agents/system_curl_client_v1.json").read_bytes())
    config = with_http_client(f.new.config, client)
    assert f.old.config.schema_version == "agent-provider-config/7.1.0"
    assert config.schema_version == "agent-provider-config/7.2.0"
    state = f.owner.current().model_copy(update={"config_checksum": digest(config)})
    f.owner.current = lambda: state
    f.source.config = config
    for reader in f.source.readers:
        reader.current_config = config
        reader.client_transition = True
    return f, config


def test_legacy_original_owners_client_only_composition_and_readback(monkeypatch):
    f, config = fixture(monkeypatch)
    before = [(digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)]
    permission, sub = AggregationPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    service.submit(sub, key=ACCEPTANCE_KEY)
    receipt = service.accept(key=ACCEPTANCE_KEY)
    assert receipt.status == "accepted"
    assert receipt.new_medical_review is False
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=f.owner.current())
    assert not report["scores"]["review_completion"]["pending_ids"]
    audit = list(f.docs.audit)
    assert service.accept(key=ACCEPTANCE_KEY) == receipt
    assert f.docs.audit == audit and not f.requests.journal.read()
    assert before == [
        (digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)
    ]
    from packages.evidence_details.acceptance_models import read_records

    original = {
        i.assessment_id: i
        for x, key in ((f.old, "partial"), (f.new, "remaining"))
        for i in read_records(x.acceptance.read(key=key)[0]).candidate.items
    }
    assert {i.assessment_id: i for i in sub.candidate.items} == original
    assert len(sub.candidate.items) == len(report["scores"]["reviews"])


@pytest.mark.parametrize(
    "fault", ["model", "route", "batch_size", "missing_config", "not_installed"]
)
def test_client_binding_is_not_general_configuration_compatibility(monkeypatch, fault):
    f, config = fixture(monkeypatch)
    reader = f.source.readers[0]
    if fault == "not_installed":
        reader.client_transition = False
    elif fault == "missing_config":
        key = next(k for k in f.old.documents.saved if k.endswith(":response"))
        f.old.documents.saved[key][0].pop("provider_config")
    else:
        if fault == "model":
            config = config.model_copy(
                update={"transport": config.transport.model_copy(update={"model": "other"})}
            )
        elif fault == "route":
            config = config.model_copy(
                update={
                    "network_route": prepare_route(environ={"HTTPS_PROXY": "http://localhost:9999"})
                }
            )
        else:
            config = config.model_copy(
                update={
                    "review_batching": config.review_batching.model_copy(update={"batch_size": 1})
                }
            )
        reader.current_config = config
        state = f.owner.current().model_copy(update={"config_checksum": digest(config)})
        f.owner.current = lambda: state
    if fault == "missing_config":
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as error:
            reader.read(f.owner.current())
        assert any(
            e["loc"] == ("provider_config",) and e["type"] == "missing"
            for e in error.value.errors()
        )
    else:
        with pytest.raises(ValueError, match="accepted_aggregation_|accepted_review_"):
            reader.read(f.owner.current())
    assert not f.docs.saved and not f.requests.journal.read()


def test_client_plan_is_explicit_and_old_plan_unchanged():
    from packages.multimodal_research.accepted_aggregation import (
        GenericClientAggregationPlan,
        read_plan,
    )

    plan = read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.10.0",
            facts="local",
            origins=[dict(campaign="prior")],
        )
    )
    assert isinstance(plan, GenericClientAggregationPlan)
    old = read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.2.0",
            facts="local",
            origins=[dict(campaign="prior")],
        )
    )
    assert not isinstance(old, GenericClientAggregationPlan)


@pytest.mark.parametrize("field", ["reasoned_batch_resize", "alias_manifests"])
def test_client_transition_rechecked_after_reader_construction(monkeypatch, field):
    f, _ = fixture(monkeypatch)
    reader = f.source.readers[0]
    setattr(reader, field, True if field == "reasoned_batch_resize" else lambda: None)
    with pytest.raises(ValueError, match="accepted_aggregation_client_combined_transition"):
        reader.read(f.owner.current())
    assert not f.docs.saved


@pytest.mark.parametrize("client", [False, True])
def test_explicit_client_plan_requires_client_bound_facts(monkeypatch, client):
    from packages.evaluation_preparation import generic_pending_review
    from packages.evaluation_preparation.generic_review import GenericReview
    from packages.llm_provider.config import read_provider_config
    from packages.multimodal_research import accepted_aggregation as entry

    c = GenericReview.__new__(GenericReview)
    c.pending_review_proof = {}
    c.config = read_provider_config(
        json.loads(
            Path("configs/agents/fairvlmed_relay_environment_gpt6_review_v1.json").read_bytes()
        )
    )
    if client:
        c.config = with_http_client(
            c.config, json.loads(Path("configs/agents/system_curl_client_v1.json").read_bytes())
        )
    monkeypatch.setattr(generic_pending_review, "restore_campaign", lambda p: (c, {}))
    monkeypatch.setattr(entry, "checked_path", lambda p: Path(p))
    plan = entry.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.10.0",
            facts="facts",
            origins=[dict(campaign="origin")],
        )
    )
    if client:
        assert entry.plan_campaign(plan)[0] is c
    else:
        with pytest.raises(ValueError, match="generic_aggregation_client_version_required"):
            entry.plan_campaign(plan)

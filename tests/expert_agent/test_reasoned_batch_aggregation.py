"""Reasoned partial + new tasks, original acceptance/scoring, no network."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.transport import parse_reply
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_directional_review import fixture as directional
from tests.evaluation_preparation.test_directional_review import reply
from tests.evaluation_preparation.test_http_client_entry import public_config
from tests.expert_agent import test_legacy_accepted_aggregation as legacy


def configured(*, single=False, replies=None):
    from packages.llm_provider.config import read_provider_config

    f = directional()
    raw = public_config().model_dump(mode="json")
    raw["mode"] = "fake"
    raw["review_batching"]["batch_size"] = 1 if single else 3
    config = read_provider_config(raw)
    state = f.owner.current().model_dump(mode="json")
    state["config_checksum"] = digest(config)
    state["prompt_checksum"] = next(r.prompt_checksum for r in config.roles if r.role == "review")
    f.owner.state = validate(type(f.owner.current()), state)
    f.provider = f.agent.provider = DeterministicFakeProvider(config, replies or [reply] * 15)
    f.config = config
    return f


def partial(monkeypatch):
    from packages.evidence_details.review_batches import collect_partial
    from packages.planner_control.models import Control

    truncated = parse_reply(
        json.dumps({"choices": [{"finish_reason": "length", "message": {"content": ""}}]}),
        "chat_completions",
        retain_metadata=True,
    )
    f = configured(replies=[reply, truncated])
    with pytest.raises(ProviderError, match="truncated"):
        generate(f)
    sub = collect_partial(f.owner.current(), f.documents, "synthetic-batches")
    assert sub.schema_version == "research-batched-submission/4.0.0"
    f.acceptance.submit(sub, key="partial")
    assert f.acceptance.accept(key="partial").status == "accepted"
    f.documents.run_id = f.binding.run_id
    control = validate(Control, dict(binding=f.binding, graph=f.binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    return f, sub


def combined(monkeypatch):
    from packages.evidence_details.accepted_aggregation import AcceptedOriginReader

    original_reader = AcceptedOriginReader
    monkeypatch.setattr(legacy, "partial", partial)
    monkeypatch.setattr(
        legacy,
        "dynamic_fixture",
        lambda *_a, config_path=None, **_k: configured(single="single" in config_path),
    )
    monkeypatch.setattr(
        legacy,
        "AcceptedOriginReader",
        lambda **kw: original_reader(**kw, reasoned_batch_resize=True),
    )
    return legacy.combined(monkeypatch, resize=True)


def test_reasoned_partition_aggregates_exact_original_values_without_requests(monkeypatch):
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.accepted_aggregation import ACCEPTANCE_KEY, AggregationPermission

    f = combined(monkeypatch)
    before = [(digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)]
    permission, sub = AggregationPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.new.config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    assert service.submit(sub, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    receipt = service.accept(key=ACCEPTANCE_KEY)
    assert receipt.status == "accepted" and not receipt.new_medical_review
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=f.owner.current())
    assert not report["scores"]["review_completion"]["pending_ids"]
    assert "零新模型请求" in render_research_html(report)
    from packages.evidence_details.acceptance_models import read_records

    original = {
        i.assessment_id: i
        for x, key in ((f.old, "partial"), (f.new, "remaining"))
        for i in read_records(x.acceptance.read(key=key)[0]).candidate.items
    }
    assert {i.assessment_id: i for i in sub.candidate.items} == original
    assert before == [
        (digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)
    ]
    audit = len(f.docs.audit)
    assert service.accept(key=ACCEPTANCE_KEY) == receipt and len(f.docs.audit) == audit
    assert not f.requests.journal.read()


@pytest.mark.parametrize(
    "change", ["model", "token_limit", "batch_count", "missing_proof", "legacy_reader"]
)
def test_reasoned_resize_does_not_allow_other_config_changes(monkeypatch, change):
    f = combined(monkeypatch)
    reader = f.source.readers[0]
    if change == "legacy_reader":
        reader.reasoned_batch_resize = False
    elif change == "missing_proof":
        key = next(k for k in f.old.documents.saved if k.endswith(":response"))
        f.old.documents.saved[key][0].pop("provider_config")
    else:
        from packages.llm_provider.config import read_provider_config

        raw = reader.current_config.model_dump(mode="json")
        if change == "model":
            raw["transport"]["model"] = "changed-model"
        elif change == "token_limit":
            next(r for r in raw["roles"] if r["role"] == "review")["max_output_tokens"] += 1
        else:
            raw["review_batching"]["max_batches"] -= 1
        reader.current_config = read_provider_config(raw)
    state = f.owner.current().model_copy(update=dict(config_checksum=digest(reader.current_config)))
    with pytest.raises(
        ValueError,
        match="provider_config"
        if change == "missing_proof"
        else "accepted_aggregation_|accepted_review_batch_",
    ):
        reader.read(state)
    assert not f.docs.saved

"""Original Review path must retain safe transport diagnostics without changing wire or retries."""

import http.client
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.review_batches import batch_payload, make_plan, proposal_type
from packages.grounding.models import digest
from packages.llm_provider.config import prompt, read_provider_config
from packages.llm_provider.provider import ProviderError, StructuredProvider
from packages.llm_provider.public_shape import PublicReviewMaterials
from packages.llm_provider.transport import HTTPTransport, wire_request
from packages.llm_provider.worker import safe_transport_error
from tests.agent_provider.batch_support import batch_reply
from tests.agent_provider.support import Documents, budget_for
from tests.agent_provider.test_bounded_live import SyntheticJournal
from tests.planner.support import binding


def setup_review(monkeypatch, *, failure=False):
    config = read_provider_config(
        json.loads(Path("configs/agents/fairvlmed_relay_review_shape_v1.json").read_bytes())
    )
    b = binding()
    owner = PublicReviewMaterials(b.run_id, config, task_count=3)
    journal = SyntheticJournal()  # Explicit unit-test journal; no DB/network.
    budget = budget_for(b, config, journal)
    documents = Documents()
    seen = []

    def run(*args, **kwargs):
        envelope = json.loads(kwargs["input"])
        seen.append(envelope)
        if failure:
            result = dict(
                error="transport",
                diagnostic=dict(
                    transport_failure=safe_transport_error(
                        http.client.RemoteDisconnected("secret-placeholder"), "dispatch_boundary"
                    )
                )
                if envelope.get("timing_events")
                else None,
            )
        else:
            result = dict(
                http_status=200,
                body=dict(
                    choices=[
                        dict(
                            finish_reason="stop",
                            message=dict(content=json.dumps(batch_reply(envelope["body"]))),
                        )
                    ]
                ),
            )
        return SimpleNamespace(
            stdout=b"DISPATCH\n" + json.dumps(result).encode(),
            stderr=b'TIMING {"phase":"dispatch_boundary","seconds":0.01}\nprivate stderr',
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", run)
    agent = StructuredRoleAgent(
        "review",
        provider=StructuredProvider(config, transport=HTTPTransport(), live_opt_in=True),
        materials=owner,
        budget=budget,
    )
    return SimpleNamespace(
        agent=agent,
        config=config,
        state=owner.current(),
        budget=budget,
        documents=documents,
        seen=seen,
    )


def execute(f):
    return run_batches(
        f.agent, state=f.state, documents=f.documents, key="public-diagnostics", round_id="A"
    )


def test_legacy_nondirectional_review_keeps_wire_budget_and_safe_timing(monkeypatch):
    f = setup_review(monkeypatch)
    plan = make_plan(f.state, f.config.review_batching, key="public-diagnostics")
    setting, text = prompt(f.config, "review")
    body = wire_request(
        f.config.transport,
        setting,
        text,
        batch_payload(f.state, plan, plan.batches[0]),
        proposal_type(f.config).model_json_schema(),
        (),
        scope_binding=digest(f.state.material.scope),
        wire_policy=f.config.wire_policy,
        wire_mode=f.config.wire_mode,
    )
    result = execute(f)
    assert len(result.batches) == len(f.seen) == f.budget.used == 1
    assert f.seen[0]["body"] == body
    assert f.seen[0].get("timing_events") is True
    events = [
        e
        for e in f.budget.journal.read()
        if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
    ]
    assert len(events) == 1 and events[0]["value"]["transport_timing"]["phases"]
    assert "private stderr" not in json.dumps(events)
    assert execute(f) == result and f.budget.used == len(f.seen) == 1


def test_unknown_review_retains_diagnostic_without_retry_or_new_acceptance(monkeypatch):
    f = setup_review(monkeypatch, failure=True)
    with pytest.raises(ProviderError, match="transport"):
        execute(f)
    assert f.budget.used == len(f.seen) == 1
    events = [
        e
        for e in f.budget.journal.read()
        if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
    ]
    assert len(events) == 1
    value = events[0]["value"]["transport_failure"]
    assert value["category"] == "remote_disconnected"
    assert value["remote_execution"] == "unknown" and not value["automatic_replay_authorized"]
    assert "secret-placeholder" not in json.dumps(events)
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        execute(f)
    assert f.budget.used == len(f.seen) == 1
    assert not any(k.startswith("accepted:") for k in f.documents.saved)


def test_remote_disconnected_not_shadowed_by_connection_reset_superclass():
    closed = safe_transport_error(http.client.RemoteDisconnected("private"), "dispatch_boundary")
    reset = safe_transport_error(ConnectionResetError(104, "private"), "dispatch_boundary")
    assert closed["category"] == "remote_disconnected"
    assert reset["category"] == "connection_reset"
    assert "private" not in json.dumps([closed, reset])

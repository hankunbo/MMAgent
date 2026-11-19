"""Explicit UNKNOWN acknowledgment and one-dispatch boundary, not a replay grant."""

import json
from types import SimpleNamespace as NS

import pytest

from apps.vectorengine_single_case import load_config
from packages.evidence_details.review_batches import batch_payload, make_plan
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError, RequestBudget
from packages.multimodal_research import campaign_renewal as renewal
from tests.agent_provider.batch_support import CONFIG
from tests.agent_provider.test_bounded_live import SyntheticJournal
from tests.expert_agent.batch_case_support import initial_state
from tests.planner.support import binding


def fixture(tmp_path, monkeypatch):
    config, b = load_config(CONFIG), binding()
    state, _ = initial_state(b, config, count=1)
    plan = make_plan(state, config.review_batching, key="synthetic-review")
    origin = dict(path=str(tmp_path / "source"), definition=dict(accepted_interpretation=True))
    old = dict(
        schema_version="multimodal-campaign-renewal/1.13.0",
        mode="review",
        review_definition_increment=True,
        review_target_ids=list(plan.assessment_ids),
        tip=origin,
        provider=digest(config),
        source_revision_checksum=digest("revision"),
        logical_campaign_id="original-campaign",
    )
    request_id = "request:synthetic-unknown"
    record = dict(
        request_id=request_id,
        role="review",
        error="transport",
        retry_index=0,
        api_requests=1,
        simulated=False,
        scope=state.material.scope.model_dump(mode="json"),
        config_checksum=digest(config),
        input_checksum=digest(batch_payload(state, plan, plan.batches[0])),
    )
    events = [
        dict(phase="reserved", request_id=request_id),
        dict(phase="completed", record=record),
        dict(
            schema_version="agent-transport-diagnostic/1.0.0",
            value=dict(transport_failure=dict(remote_execution="unknown")),
        ),
    ]
    prefix = "review-batches:EVENT:REVIEW:A"
    slot = prefix + ":" + plan.batches[0].id
    values = {
        "multimodal-case-result": [
            dict(complete=False, new_requests=1, phases=[dict(accepted=False)])
        ],
        "live-stage-intent:review": [dict(event=dict(code="EVENT:REVIEW"))],
        prefix + ":plan": [plan.model_dump(mode="json")],
        "dynamic-phase-input": [state.model_dump(mode="json")],
        slot + ":attempt:" + request_id: [
            dict(error="transport", raw_model_text="", record_checksum=digest(record))
        ],
    }
    docs = NS(read=values.get)
    control = NS(phase="stopped", pending=None, cancelled=False)
    parent = NS(documents=lambda run: docs, sessions=NS(read=lambda run: control))
    tip = dict(path=str(tmp_path), run_id=b.run_id)
    monkeypatch.setattr(renewal, "stopped_tip", lambda *a, **kw: origin)
    monkeypatch.setattr(renewal, "RuntimeAgentJournal", lambda d: NS(read=lambda: events))

    def call(ack=request_id):
        (tmp_path / "protocol.json").write_text(json.dumps(old))
        return renewal.definition_reassessment(parent, tip, ack)

    return NS(
        call=call,
        old=old,
        events=events,
        record=record,
        values=values,
        slot=slot,
        request_id=request_id,
        state=state,
        config=config,
        b=b,
        control=control,
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "ack",
        "version",
        "targets",
        "accepted",
        "pending",
        "role",
        "scope",
        "input",
        "config",
        "retry",
        "simulated",
        "not_sent",
        "known",
        "response",
        "batch",
        "record_hash",
        "cancelled",
    ],
)
def test_reassessment_requires_exact_old_unknown_and_no_usable_response(
    tmp_path, monkeypatch, fault
):
    f = fixture(tmp_path, monkeypatch)
    ack = f.request_id
    if fault == "ack":
        ack = "request:wrong"
    elif fault == "version":
        f.old["schema_version"] = "multimodal-campaign-renewal/1.14.0"
    elif fault == "targets":
        f.old["review_target_ids"] *= 2
    elif fault == "accepted":
        f.values["accepted:EVENT:REVIEW:domain"] = [{"status": "accepted"}]
    elif fault == "pending":
        f.events.append(dict(phase="reserved", request_id="request:another"))
    elif fault in {"scope", "input", "config"}:
        key = {"scope": "scope", "input": "input_checksum", "config": "config_checksum"}[fault]
        f.record[key] = "changed"
    elif fault == "role":
        f.record["role"] = "vision"
    elif fault == "retry":
        f.record["retry_index"] = 1
    elif fault == "simulated":
        f.record["simulated"] = True
    elif fault == "not_sent":
        f.record["api_requests"] = 0
    elif fault == "known":
        f.events[-1]["value"]["transport_failure"]["remote_execution"] = "confirmed"
    elif fault == "response":
        f.values[f.slot + ":attempt:" + f.request_id][0]["raw_model_text"] = "saved response"
    elif fault == "batch":
        f.values[f.slot + ":result"] = [{"confirmed": True}]
    elif fault == "record_hash":
        f.values[f.slot + ":attempt:" + f.request_id][0]["record_checksum"] = digest("different")
    elif fault == "cancelled":
        f.control.cancelled = True
    if fault is None:
        result = f.call(ack)
        assert result["request_id"] == ack and result["duplicate_cost_possible"]
        assert not result["original_request_replayed"]
    else:
        with pytest.raises(ValueError, match="renewal_definition_reassessment_"):
            f.call(ack)


@pytest.mark.parametrize("version", ["1.13.0", "1.14.0"])
def test_one_real_budget_reservation_blocks_second_and_preserves_old_policy(monkeypatch, version):
    b, config = binding(), load_config(CONFIG)
    state, _ = initial_state(b, config, count=1)
    journal = SyntheticJournal()
    budget = RequestBudget(
        run_id=b.run_id,
        deadline=b.deadline,
        max_requests=40,
        journal=journal,
        clock=lambda: b.started_at,
    )
    monkeypatch.setattr(renewal.MultimodalCampaign, "budget_factory", lambda self, **kw: budget)
    c = object.__new__(renewal.RenewedCampaign)
    c.protocol = dict(
        schema_version="multimodal-campaign-renewal/" + version, max_model_dispatches=1
    )
    c.usage = lambda: budget.used
    assert c.budget_factory() is budget
    budget.reserve(state.material.scope, b.deadline, digest(config))
    before = journal.read()
    if version == "1.13.0":
        budget.reserve(state.material.scope, b.deadline, digest(config))
        assert budget.used == 2
    else:
        with pytest.raises(ProviderError, match="budget"):
            budget.reserve(state.material.scope, b.deadline, digest(config))
        assert journal.read() == before and budget.used == 1


def test_cli_requires_explicit_acknowledged_request(monkeypatch, capsys):
    import sys

    from apps import multimodal_case

    seen = []
    monkeypatch.setattr(renewal, "prepare", lambda **kw: seen.append(kw) or {"prepared": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case",
            "renewal-prepare",
            "--facts",
            "facts",
            "--previous",
            "previous",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "synthetic-only",
            "--review-definition-increment",
            "--acknowledge-unknown-request",
            "request:exact",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen[0]["acknowledge_unknown_request"] == "request:exact"
    assert "prepared" in capsys.readouterr().out

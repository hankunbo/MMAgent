"""Original Planner and Review failure leave exact, bounded successor inventory."""

from threading import Event

import pytest

from apps.vectorengine_live_assembly import run_case_stages
from packages.evidence_details.pending_review import read_failed_review_attempt
from packages.grounding.models import digest
from packages.llm_provider.transport import Reply
from packages.planner_control.models import Command
from tests.expert_agent.test_batch_assembly import CONFIG, assembly


def failed(monkeypatch, tmp_path, *, replies=None, config_path=CONFIG):
    f = assembly(
        monkeypatch,
        replies=replies
        if replies is not None
        else [Reply(error="timeout", api_requests=1, response_confirmed=False)],
        config_path=config_path,
    )
    if config_path != CONFIG:
        monkeypatch.setattr(
            "packages.llm_provider.transport.HTTPTransport.send",
            lambda self, config, body, **kw: self._observed(
                f.wire.send(config, body, **kw), {"http_status": 200}, body, "chat_completions"
            ),
        )
    run_case_stages(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
        selected_roles=("review",),
    )
    current = f.planner.read(f.binding.run_id)
    if current.phase != "stopped":
        f.planner.step(
            f.binding.run_id,
            Command(key="close-failed", expected_revision=current.revision, action="stop"),
        )
    event = f.documents.read("live-stage-intent:review")[0]["event"]["code"]
    state = f.documents.read("review:" + event + ":intent")[0]["state"]
    from packages.evidence_details.acceptance_models import DomainState
    from packages.grounding.models import validate

    return f, validate(DomainState, state)


def test_original_failure_selects_unstarted_and_explicit_ack_only(monkeypatch, tmp_path):
    f, state = failed(monkeypatch, tmp_path)
    before = digest(f.documents.saved), digest(f.ports.budget.journal.read())
    value = read_failed_review_attempt(documents=f.documents, state=state, config=f.config)
    assert value["selected_ids"] and value["quarantined_batches"]
    q = value["quarantined_batches"][0]
    assert not q["new_reassessment_authorized"]
    assert not set(q["assessment_ids"]) & set(value["selected_ids"])
    selected = read_failed_review_attempt(
        documents=f.documents, state=state, config=f.config, acknowledge_request=q["request_ids"][0]
    )
    assert set(selected["selected_ids"]) == set(value["selected_ids"]) | set(q["assessment_ids"])
    assert before == (digest(f.documents.saved), digest(f.ports.budget.journal.read()))
    with pytest.raises(ValueError, match="ack_mismatch"):
        read_failed_review_attempt(
            documents=f.documents, state=state, config=f.config, acknowledge_request="request:wrong"
        )


@pytest.mark.parametrize("change", ["claim", "parent", "cancel"])
def test_failure_cannot_bypass_owner_binding(monkeypatch, tmp_path, change):
    f, state = failed(monkeypatch, tmp_path)
    if change == "claim":
        key = next(
            k for k in f.documents.saved if k.startswith("review-batches:") and k.endswith(":claim")
        )
        f.documents.saved[key][0]["input_checksum"] = digest("wrong")
    elif change == "parent":
        state = state.model_copy(update={"config_checksum": digest("wrong")})
    else:
        current = f.planner.read(f.binding.run_id)
        monkeypatch.setattr(
            f.documents.sessions, "read", lambda _: current.model_copy(update={"cancelled": True})
        )
    with pytest.raises(ValueError, match="pending_attempt"):
        read_failed_review_attempt(documents=f.documents, state=state, config=f.config)


def test_rejected_format_quarantines_only_claimed_batch(monkeypatch, tmp_path):
    reply = Reply(
        content=b'{"bad":true}', api_requests=1, response_confirmed=True, finish_reason="stop"
    )
    f, state = failed(
        monkeypatch,
        tmp_path,
        replies=[reply, reply],
        config_path="configs/agents/vectorengine_batch_integrated.json",
    )
    before = digest(f.documents.saved), digest(f.ports.budget.journal.read())
    value = read_failed_review_attempt(documents=f.documents, state=state, config=f.config)
    q = value["quarantined_batches"][0]
    assert value["selected_ids"] and not q["new_reassessment_authorized"]
    assert not set(q["assessment_ids"]) & set(value["selected_ids"])
    assert "known_format_failure" not in q
    inspected = read_failed_review_attempt(
        documents=f.documents, state=state, config=f.config, inspect_format_failures=True
    )
    assert inspected["selected_ids"] == value["selected_ids"]
    assert inspected["quarantined_batches"][0]["known_format_failure"]
    assert before == (digest(f.documents.saved), digest(f.ports.budget.journal.read()))
    with pytest.raises(ValueError, match="ack_mismatch"):
        read_failed_review_attempt(
            documents=f.documents,
            state=state,
            config=f.config,
            acknowledge_request=q["request_ids"][0],
        )


@pytest.mark.parametrize("change", ["bytes", "config", "finish"])
def test_rejected_format_inventory_validates_bytes_without_reassessment(
    monkeypatch, tmp_path, change
):
    reply = Reply(
        content=b'{"bad":true}', api_requests=1, response_confirmed=True, finish_reason="stop"
    )
    f, state = failed(
        monkeypatch,
        tmp_path,
        replies=[reply, reply],
        config_path="configs/agents/vectorengine_batch_integrated.json",
    )
    key = next(
        k
        for k, v in f.documents.saved.items()
        if ":attempt:" in k and v[0].get("error") == "schema"
    )
    raw = f.documents.saved[key][0]
    if change == "bytes":
        raw["raw_model_text"] += " "
    elif change == "config":
        raw["provider_config"] = {}
    else:
        raw["finish_reason"] = "length"
    with pytest.raises(ValueError, match="format_response_binding"):
        read_failed_review_attempt(documents=f.documents, state=state, config=f.config)

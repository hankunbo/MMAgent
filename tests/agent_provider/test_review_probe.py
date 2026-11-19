"""Public diagnostic ladder: no credentials, real HTTP, patient data or DB in unit tests."""

import json

import pytest

from packages.evidence_details.review_batches import batch_payload
from packages.grounding.models import digest
from packages.llm_provider import review_probe as probe
from packages.llm_provider import review_probe_execution as execution
from packages.llm_provider.transport import Reply
from tests.agent_provider.batch_support import batch_reply


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "implementation", lambda: {"synthetic-unit-freeze": "unchanged"})
    monkeypatch.setattr(execution, "route_opener", lambda *args: None)
    monkeypatch.setattr(execution, "verify_binary", lambda *args: None)
    return tmp_path / "new"


@pytest.mark.parametrize("rung,context,size", probe.RUNGS)
def test_rungs_use_original_plan_full_parent_and_schema(rung, context, size):
    config, state, plan, body, counts = probe.build("gpt-4o-2024-08-06", "a" * 32, rung, 0)
    payload = json.loads(body["messages"][1]["content"][0]["text"])
    assert payload == batch_payload(state, plan, plan.batches[0])
    assert payload["material"] == state.material.model_dump(mode="json")
    assert counts["tasks"] == size and counts["parent_observations"] == context
    assert counts["parent_sources"] == context
    assert counts["wire_bytes"] == len(json.dumps(body).encode())
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == config.transport.model
    assert "OUTPUT_JSON_SCHEMA" in body["messages"][1]["content"][-1]["text"]
    assert all(s.purpose == "research_material" for s in state.material.sources)
    assert not any(o.image for o in state.material.observations)


def test_prepare_check_and_existing_directory_no_overwrite(local):
    plan = probe.prepare(local, "gpt-4o-2024-08-06")
    assert probe.check(local) == plan and len(plan["entries"]) == 9
    before = (local / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.prepare(local, "gpt-4o-2024-08-06")
    assert (local / "plan.json").read_bytes() == before
    assert execution.status(local)["status"] == "prepared"


@pytest.mark.parametrize("change", ["wire", "plan", "parent", "freeze"])
def test_tamper_rejected_before_secret_even_if_checksum_updated(local, change, monkeypatch):
    plan = probe.prepare(local, "gpt-4o-2024-08-06", repetitions=1)
    first = plan["entries"][0]
    if change == "plan":
        plan["http_seconds"] = 999
    elif change == "freeze":
        monkeypatch.setattr(probe, "implementation", lambda: {"changed": "yes"})
    else:
        key = "wire" if change == "wire" else "state"
        path = local / first["name"] / (key + ".json")
        value = json.loads(path.read_bytes())
        if key == "wire":
            value["messages"][0]["content"] = "Private material must never be sent by this probe."
        else:
            value["material"]["goal_summary"] = "private"
        path.write_text(json.dumps(value))
        first["checksums"][key] = digest(value)
    (local / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        execution.execute(local, lambda: pytest.fail("secret must not be read"))
    assert not (local / "execution.claim").exists()


class FakeTransport:
    calls = 0
    error = None

    def __init__(self, **kwargs):
        pass

    def send(self, config, body, **kwargs):
        type(self).calls += 1
        assert kwargs["effective_http_timeout"] == 90
        if self.error:
            return Reply(error=self.error, api_requests=1, response_confirmed=None)
        return Reply(
            content=json.dumps(batch_reply(body)).encode(),
            api_requests=1,
            response_confirmed=True,
            finish_reason="stop",
        )

    def observation(self, reply):
        return {"synthetic_test_only": True}


def test_all_pass_original_decode_and_owner_env_restore_no_real_qualification(local, monkeypatch):
    FakeTransport.calls, FakeTransport.error = 0, None
    monkeypatch.setenv("MMAGENT_VECTORENGINE_API_KEY", "prior-placeholder")
    probe.prepare(local, "gpt-4o-2024-08-06")
    summary = execution.execute(
        local, lambda: "unit-placeholder-key", transport_factory=FakeTransport
    )
    assert summary["three_rung_gate_passed"] and FakeTransport.calls == 9
    assert summary["actual_requests"] == 9  # Synthetic transport counters, not real API evidence.
    assert not summary["clinical_qualification"] and not summary["goal_restarted"]
    import os

    assert os.environ["MMAGENT_VECTORENGINE_API_KEY"] == "prior-placeholder"
    with pytest.raises(ValueError, match="already_claimed"):
        execution.execute(local, lambda: pytest.fail("do not reread secret"))


def test_unknown_stops_each_rung_without_retry_or_false_pass(local):
    FakeTransport.calls, FakeTransport.error = 0, "transport"
    probe.prepare(local, "gpt-4o-2024-08-06")
    summary = execution.execute(
        local, lambda: "unit-placeholder-key", transport_factory=FakeTransport
    )
    assert FakeTransport.calls == summary["actual_requests"] == 3
    assert len(summary["not_sent"]) == 6 and not summary["three_rung_gate_passed"]
    assert summary["qualified_rungs"] == []


def test_cancel_before_key_and_single_pass_not_stability(local):
    probe.prepare(local, "gpt-4o-2024-08-06", repetitions=1)
    (local / "cancel").touch()
    with pytest.raises(ValueError, match="cancelled"):
        execution.execute(local, lambda: pytest.fail("cancel must precede credentials"))
    (local / "cancel").unlink()
    FakeTransport.calls, FakeTransport.error = 0, None
    result = execution.execute(
        local, lambda: "unit-placeholder-key", transport_factory=FakeTransport
    )
    assert FakeTransport.calls == 3 and not result["three_rung_gate_passed"]


def test_crash_keeps_claim_and_readonly_status_never_replays(local):
    class Crash(FakeTransport):
        def send(self, *args, **kwargs):
            raise OSError("simulated_post_dispatch_crash")

    probe.prepare(local, "gpt-4o-2024-08-06", repetitions=1)
    with pytest.raises(OSError, match="simulated"):
        execution.execute(local, lambda: "unit-placeholder-key", transport_factory=Crash)
    status = execution.status(local)
    assert status["entries"][0]["status"] == "claimed_unconfirmed_no_replay"
    assert status["entries"][1]["status"] == "not_sent"
    with pytest.raises(ValueError, match="already_claimed"):
        execution.execute(local, lambda: pytest.fail("must not resend"))


def test_wrong_citations_still_rejected_by_original_owner(local):
    class WrongCitation(FakeTransport):
        def send(self, config, body, **kwargs):
            candidate = batch_reply(body)
            candidate["items"][0]["observation_citations"] = ["OBS:NOT_IN_OWNER"]
            return Reply(
                content=json.dumps(candidate).encode(),
                api_requests=1,
                response_confirmed=True,
                finish_reason="stop",
            )

    probe.prepare(local, "gpt-4o-2024-08-06")
    summary = execution.execute(
        local, lambda: "unit-placeholder-key", transport_factory=WrongCitation
    )
    assert summary["actual_requests"] == 3 and not summary["qualified_rungs"]
    assert all(not r["contract"]["passed"] for r in summary["results"])


def test_cancellation_between_requests_no_additional_dispatch(local):
    class Cancel(FakeTransport):
        def send(self, *args, **kwargs):
            (local / "cancel").touch()
            return Reply(error="transport", api_requests=1)

    probe.prepare(local, "gpt-4o-2024-08-06")
    result = execution.execute(local, lambda: "unit-placeholder-key", transport_factory=Cancel)
    assert result["actual_requests"] == 1 and result["cancelled"]
    assert len(result["not_sent"]) == 8


def test_no_secret_echo_written_or_printed(local, capsys):
    class Echo(FakeTransport):
        def send(self, *args, **kwargs):
            return Reply(content=b"unit-placeholder-key", api_requests=1, response_confirmed=True)

    probe.prepare(local, "gpt-4o-2024-08-06", repetitions=1)
    result = execution.execute(local, lambda: "unit-placeholder-key", transport_factory=Echo)
    assert all(r["error"] == "secret_response" for r in result["results"])
    assert not list(local.glob("*/response.txt"))
    assert "unit-placeholder-key" not in capsys.readouterr().out


def test_invalid_limits_and_model_cannot_create_output(local):
    with pytest.raises(ValueError, match="repetitions"):
        probe.prepare(local, "gpt-4o-2024-08-06", repetitions=4)
    with pytest.raises(ValueError, match="model"):
        probe.prepare(local, "unapproved-model")
    assert not local.exists()

"""Explicit public provider selection must retain original route and current binding gates."""

import json

import pytest

from packages.grounding.models import digest
from packages.llm_provider import review_probe as probe
from packages.llm_provider import review_probe_execution as execution
from packages.llm_provider.network_route import prepare_route
from tests.agent_provider.test_review_probe import FakeTransport


@pytest.fixture
def selected(monkeypatch, tmp_path):
    raw = probe.configuration("gpt-6-sol", 3).model_dump(mode="json")
    raw["network_route"] = prepare_route(
        environ={"HTTPS_PROXY": "http://127.0.0.1:7890"}
    ).model_dump(mode="json")
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "implementation", lambda: {"test-only": "unchanged"})
    folder = tmp_path / "configs/agents"
    folder.mkdir(parents=True)
    path = folder / "selected.json"
    path.write_text(json.dumps(raw))
    return tmp_path / "probe", path, raw


def test_explicit_configuration_bound_to_all_rungs_and_execution(selected, monkeypatch):
    output, path, raw = selected
    plan = probe.prepare(output, "gpt-6-sol", provider_config=str(path), repetitions=1)
    assert plan["schema_version"] == "public-review-load-ladder/1.1.0"
    assert plan["provider_binding"]["checksum"] == digest(raw)
    assert probe.check(output) == plan
    for entry in plan["entries"]:
        config = json.loads((output / entry["name"] / "config.json").read_bytes())
        assert config["network_route"] == raw["network_route"]
        assert config["transport"] == raw["transport"]
        assert config["roles"] == raw["roles"]
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    FakeTransport.calls, FakeTransport.error = 0, None
    result = execution.execute(output, lambda: "synthetic-key", transport_factory=FakeTransport)
    assert result["actual_requests"] == 3 and not result["three_rung_gate_passed"]


def test_route_mismatch_rejects_before_key_and_claim(selected, monkeypatch):
    output, path, _ = selected
    probe.prepare(output, "gpt-6-sol", provider_config=str(path), repetitions=1)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    with pytest.raises(ValueError, match="endpoint_changed"):
        execution.execute(output, lambda: pytest.fail("must not read Key"))
    assert not (output / "execution.claim").exists()


def test_configuration_drift_rejects_before_key_even_updated_manifest(selected):
    output, path, raw = selected
    plan = probe.prepare(output, "gpt-6-sol", provider_config=str(path), repetitions=1)
    raw["transport"]["timeout_seconds"] = 299
    path.write_text(json.dumps(raw))
    plan["provider_binding"]["checksum"] = digest(raw)
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        execution.execute(output, lambda: pytest.fail("must not read Key"))


@pytest.mark.parametrize("change", ["model", "url", "key_env", "path", "route", "mode"])
def test_unapproved_provider_or_path_cannot_prepare(selected, change):
    output, path, raw = selected
    if change == "model":
        raw["transport"]["model"] = "gpt-4o-2024-08-06"
    elif change == "url":
        raw["transport"]["base_url"] = "https://not-authorized.invalid/v1"
    elif change == "key_env":
        raw["transport"]["api_key_env"] = "UNRELATED_API_KEY"
    elif change == "route":
        raw.pop("network_route")
        raw.pop("http_client")
        raw["schema_version"] = "agent-provider-config/7.0.0"
    elif change == "mode":
        raw["mode"] = "fake"
    else:
        path = output.parent / "private-config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        probe.prepare(output, "gpt-6-sol", provider_config=str(path))
    assert not output.exists()


def test_original_urllib_profile_does_not_require_curl(selected, monkeypatch):
    output, path, raw = selected
    raw["schema_version"] = "agent-provider-config/7.1.0"
    raw.pop("http_client")
    path.write_text(json.dumps(raw))
    probe.prepare(output, "gpt-6-sol", provider_config=str(path), repetitions=1)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(execution, "verify_binary", lambda _: pytest.fail("urllib is not curl"))
    FakeTransport.calls, FakeTransport.error = 0, None
    result = execution.execute(output, lambda: "synthetic-key", transport_factory=FakeTransport)
    assert result["actual_requests"] == 3


@pytest.mark.parametrize("name", ["fairvlmed", "multimodal"])
def test_actual_public_profiles_retain_prompt_owner_and_complete_material(name):
    path = f"configs/agents/{name}_relay_local_gpt6_review_v1.json"
    origin, binding = probe.selected_provider(path, "gpt-6-sol")
    config, state, plan, wire, counts = probe.build(
        "gpt-6-sol", "e" * 32, "triple-expanded", 0, provider_config=path
    )
    assert config == origin and binding["checksum"] == digest(config)
    assert getattr(config, "http_client", None) is None
    assert config.transport.max_retries == 0
    assert wire["model"] == "gpt-6-sol" and wire["stream"] is False
    from packages.evidence_details.review_batches import batch_payload
    from packages.llm_provider.config import prompt

    assert wire["messages"][0]["content"] == prompt(config, "review")[1]
    assert json.loads(wire["messages"][1]["content"][0]["text"]) == batch_payload(
        state, plan, plan.batches[0]
    )
    assert counts["parent_observations"] == counts["parent_sources"] == 15
    from packages.evidence_details.review_batches import proposal_type

    assert proposal_type(state.policy) is proposal_type(config)


def test_reasoned_response_reaches_original_domain_owner(monkeypatch, tmp_path):
    from tests.evaluation_preparation.test_directional_review import reply

    class Reasoned(FakeTransport):
        def send(self, config, body, **kwargs):
            from packages.llm_provider.transport import Reply

            return Reply(
                content=json.dumps(reply(body)).encode(),
                api_requests=1,
                response_confirmed=True,
                finish_reason="stop",
            )

    monkeypatch.setattr(probe, "implementation", lambda: {"test-only": "unchanged"})
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    output = tmp_path / "directional-public"
    plan = probe.prepare(
        output,
        "gpt-6-sol",
        repetitions=1,
        provider_config="configs/agents/multimodal_relay_local_gpt6_review_v1.json",
    )
    result = execution.execute(output, lambda: "synthetic-key", transport_factory=Reasoned)
    assert all(row["contract"]["passed"] for row in result["results"])
    assert plan["schema_version"] == "public-review-load-ladder/1.2.0"

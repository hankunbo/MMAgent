"""Explicit shape-only successor; original legacy Review, scoring and history retained."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation import generic_input as g
from packages.evaluation_preparation import generic_review as r
from packages.grounding.models import canonical
from packages.llm_provider.config import prompt, read_provider_config
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.transport import Reply
from tests.agent_provider.batch_support import batch_reply, fixture, generate
from tests.expert_agent.batch_case_support import initial_state


def original():
    return read_provider_config(json.loads(Path(g.RELAY_PROVIDER).read_bytes()))


def test_shape_profile_changes_only_explicit_review_prompt_and_bounded_retry():
    before = original()
    selected = r.review_configuration(before, r.SHAPE_PROFILE)
    raw = selected.model_dump(mode="json")
    raw["transport"]["max_retries"] = 0
    role = next(row for row in raw["roles"] if row["role"] == "review")
    old_role = next(row for row in before.roles if row.role == "review")
    role.update(old_role.model_dump(mode="json"))
    assert raw == before.model_dump(mode="json")
    assert selected.schema_version == "agent-provider-config/7.0.0"
    assert selected.max_requests == 40 and selected.max_extra_requests == 3
    _, text = prompt(selected, "review")
    assert "Review output shape /1.0.0" in text
    assert "Only judgment has method" in text
    assert original() == before


@pytest.mark.parametrize("kind", ["name", "model", "old_prompt", "public_drift"])
def test_shape_profile_rejects_unrelated_change(tmp_path, monkeypatch, kind):
    config = original()
    name = r.SHAPE_PROFILE
    if kind == "name":
        name = "unknown"
    elif kind == "model":
        config = config.model_copy(
            update={"transport": config.transport.model_copy(update={"model": "changed"})}
        )
    elif kind == "old_prompt":
        raw = config.model_dump(mode="json")
        next(row for row in raw["roles"] if row["role"] == "review")["prompt_version"] = "old"
        config = read_provider_config(raw)
    else:
        path = tmp_path / "public.json"
        raw = json.loads(Path(r.SHAPE_PROVIDER).read_bytes())
        raw["review_batching"]["batch_size"] = 2
        path.write_text(json.dumps(raw))
        monkeypatch.setattr(r, "SHAPE_PROVIDER", str(path))
    with pytest.raises(ValueError, match="generic_review_profile_"):
        r.review_configuration(config, name)


def test_constructor_installs_review_only_and_preserves_original_acquisition(tmp_path, monkeypatch):
    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    def setup(self, spec, registered):
        self.config = original()
        self.provider_path = g.RELAY_PROVIDER

    monkeypatch.setattr(GenericCampaign, "__init__", setup)
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r.GenericReview, "prior", lambda self: {"reservations": 7})
    c = r.GenericReview(
        {"schema_version": "fairvlmed-run-input/1.5.0"}, None, "new", review_profile=r.SHAPE_PROFILE
    )
    assert c.config == r.review_configuration(original(), r.SHAPE_PROFILE)
    assert c.acquisition.config == original()
    assert c.provider_path == r.SHAPE_PROVIDER
    assert c.max_vision_requests == 0 and c.max_new_requests == 10
    assert c.duration_seconds == 2700


def malformed(body):
    raw = batch_reply(body)
    raw["items"][0]["observation"]["method"] = "not_assessed"
    return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)


def test_actual_provider_correction_then_original_acceptance_score_and_readback():
    seen = []

    def corrected(body):
        assert "Review output shape /1.0.0" in body["messages"][0]["content"]
        feedback = json.loads(body["messages"][-1]["content"])
        assert "Only judgment" in feedback["rating_contract"]
        seen.append(feedback)
        return batch_reply(body)

    f = fixture(3, replies=[malformed, corrected], config_path=r.SHAPE_PROVIDER)
    f.owner.state = initial_state(f.binding, f.config, 3)[0]
    sub = generate(f)
    assert sub.schema_version == "research-batched-submission/1.0.0"
    assert len(seen) == 1 and f.budget.used == 2
    f.acceptance.submit(sub, key="new-shape-review")
    assert f.acceptance.accept(key="new-shape-review").status == "accepted"
    scores = f.acceptance.read(key="new-shape-review")[2]
    assert scores["scores"]
    failed = [v[0] for k, v in f.documents.saved.items() if ":attempt:" in k and v[0].get("error")]
    assert len(failed) == 1 and "not_assessed" in failed[0]["raw_model_text"]
    count = len(f.documents.audit)
    assert generate(f) == sub and f.budget.used == 2
    assert len(f.documents.audit) == count


def test_second_bad_response_stops_without_indefinite_repair():
    f = fixture(3, replies=[malformed, malformed, batch_reply], config_path=r.SHAPE_PROVIDER)
    f.owner.state = initial_state(f.binding, f.config, 3)[0]
    with pytest.raises(ProviderError):
        generate(f)
    assert f.budget.used == 2


def test_new_profile_unknown_http_is_not_retried(monkeypatch):
    from tests.agent_provider import test_batch_public as public

    monkeypatch.setattr(public, "CONFIG", r.SHAPE_PROVIDER)
    f = public.controlled_batch(monkeypatch, ["unknown"])
    with pytest.raises(ProviderError, match="timeout"):
        public.execute(f)
    with pytest.raises(ProviderError, match="timeout|unknown_no_repeat"):
        public.execute(f)
    assert f.budget.used == len(f.calls) == 1
    assert f.budget.transport_counts()["remote_response_unknown"] == 1


def test_unknown_profile_rejected_before_successor_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "load", lambda _: {})
    monkeypatch.setattr(r, "provider_config", lambda _: original())
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    with pytest.raises(ValueError, match="profile_unknown"):
        r.run_review("parent", "child", review_profile="unknown")
    assert not list(tmp_path.rglob("review-successor.json"))


def test_cli_forwards_explicit_profile(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(r, "run_review", lambda *a, **k: calls.append((a, k)) or {"status": "test"})
    assert (
        r.main(
            [
                "--key",
                "parent",
                "--review-key",
                "child",
                "--review-profile",
                r.SHAPE_PROFILE,
            ]
        )
        == 0
    )
    assert calls == [
        ((), {"key": "parent", "review_key": "child", "review_profile": r.SHAPE_PROFILE})
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "test"}


def test_profile_cli_returns_failure_and_rejects_unknown_without_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        r, "run_review", lambda **k: calls.append(k) or {"result": {"complete": False}}
    )
    args = ["--key", "parent", "--review-key", "child", "--review-profile", r.SHAPE_PROFILE]
    assert r.main(args) == 1 and len(calls) == 1
    with pytest.raises(SystemExit) as error:
        r.main([*args[:-1], "unknown"])
    assert error.value.code == 2 and len(calls) == 1

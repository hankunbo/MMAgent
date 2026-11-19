"""New public Provider binding, legacy bytes and grounded weak-score contract."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation import generic_input as g
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_input import data


def prepared(tmp_path, monkeypatch):
    from packages.evaluation_preparation import report_production, unified_requirements
    from packages.evidence_reuse import integration

    public = {
        path: Path(path).read_bytes()
        for path in (
            g.RELAY_PROVIDER,
            g.SHAPE_PROVIDER,
            "configs/prompts/agents/review-batch-1.9.0.txt",
        )
    }
    monkeypatch.chdir(tmp_path)
    for name, content in public.items():
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    data(tmp_path / "input", "arbitrary.npz")
    monkeypatch.setattr(unified_requirements, "catalog", lambda: {})
    monkeypatch.setattr(integration, "implementation", lambda: {"upstream": "bound"})
    monkeypatch.setattr(report_production, "implementation", lambda: {"report": "bound"})
    options = dict(
        dataset="fairvlmed",
        root=tmp_path / "input",
        sample="arbitrary.npz",
        evidence_profile=unified_requirements.PROFILE,
        upstream_profile=integration.PROFILE,
        report_profile=report_production.PROFILE,
    )
    spec = g.prepare(**options, key="new-route", provider_profile=g.RELAY_PROFILE)
    return spec, options


def test_new_route_is_bound_and_old_input_remains_exact(tmp_path, monkeypatch):
    from packages.llm_provider.config import prompt

    spec, options = prepared(tmp_path, monkeypatch)
    assert spec["schema_version"] == "fairvlmed-run-input/1.5.0"
    assert g.load("new-route") == spec
    config = g.provider_config(spec)
    assert config.transport.base_url == "https://api.relayrouter.ai/v1"
    assert config.transport.model == "gpt-4o-2024-08-06"
    assert config.transport.max_retries == 0 and config.max_requests == 40
    assert digest(config) == spec["provider_configuration_checksum"]
    setting, text = prompt(config, "review")
    assert setting.prompt_version == "review-batch-1.8.0"
    assert "however weak" in text and "Zero does not prove absolute absence" in text
    old = g.prepare(**options, key="legacy-route")
    before = (g.directory("legacy-route") / "input.json").read_bytes()
    assert old["schema_version"] == "fairvlmed-run-input/1.4.0"
    assert old["provider_configuration"] == g.PROVIDER
    assert "provider_profile" not in old and g.load("legacy-route") == old
    assert g.prepare(**options, key="legacy-route") == old
    assert (g.directory("legacy-route") / "input.json").read_bytes() == before
    assert g.owner_key(spec, "phase", "review") != g.owner_key(old, "phase", "review")
    assert not list(tmp_path.rglob("binding.json"))


def test_shape_profile_initial_run_binding_and_old_input_unchanged(tmp_path, monkeypatch):
    from packages.evaluation_preparation.generic_review import review_configuration
    from packages.llm_provider.config import prompt

    old, options = prepared(tmp_path, monkeypatch)
    previous = (g.directory(old["key"]) / "input.json").read_bytes()
    new = g.prepare(**options, key="shape-initial", provider_profile=g.SHAPE_PROFILE)
    assert new["schema_version"] == "fairvlmed-run-input/1.6.0"
    assert g.load(new["key"]) == new
    config = g.provider_config(new)
    assert config == review_configuration(g.provider_config(old), g.SHAPE_PROFILE)
    assert prompt(config, "review")[0].prompt_version == "review-batch-1.9.0"
    assert config.max_requests == new["max_requests"] == 40
    assert new["seconds"] == 5400 and config.transport.max_retries == 1
    assert g.owner_key(new, "phase", "review") != g.owner_key(old, "phase", "review")
    assert g.load(old["key"]) == old
    assert (g.directory(old["key"]) / "input.json").read_bytes() == previous
    assert not list(tmp_path.rglob("binding.json"))


@pytest.mark.parametrize("mutation", ["downgrade", "profile", "checksum", "vision"])
def test_shape_initial_cannot_mix_versions_or_change_acquisition(tmp_path, monkeypatch, mutation):
    _, options = prepared(tmp_path, monkeypatch)
    new = g.prepare(**options, key="shape-initial", provider_profile=g.SHAPE_PROFILE)
    if mutation == "downgrade":
        new["schema_version"] = "fairvlmed-run-input/1.5.0"
    elif mutation == "profile":
        new["provider_profile"] = g.RELAY_PROFILE
    elif mutation == "checksum":
        new["provider_configuration_checksum"] = digest("unbound")
    else:
        from packages.llm_provider.config import read_provider_config

        path = Path(g.SHAPE_PROVIDER)
        raw = json.loads(path.read_bytes())
        next(role for role in raw["roles"] if role["role"] == "vision")["max_output_tokens"] = 2048
        path.write_text(json.dumps(raw))
        new["provider_configuration_checksum"] = digest(read_provider_config(raw))
    (g.directory(new["key"]) / "input.json").write_text(json.dumps(new))
    with pytest.raises(ValueError, match="generic_(configuration|provider|review_profile)"):
        g.load(new["key"])


def test_shape_initial_successor_does_not_revert_provider(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_review
    from packages.evaluation_preparation.generic_campaign import GenericCampaign
    from packages.llm_provider.config import read_provider_config

    config = read_provider_config(json.loads(Path(g.SHAPE_PROVIDER).read_bytes()))

    def setup(self, spec, registered):
        self.config, self.provider_path = config, g.SHAPE_PROVIDER

    monkeypatch.setattr(GenericCampaign, "__init__", setup)
    monkeypatch.setattr(generic_review, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(generic_review.GenericReview, "prior", lambda self: {"reservations": 7})
    successor = generic_review.GenericReview(
        {"schema_version": "fairvlmed-run-input/1.6.0"}, None, "future-review"
    )
    assert successor.config == config and successor.provider_path == g.SHAPE_PROVIDER
    assert successor.max_vision_requests == 0 and successor.max_new_requests == 10


def test_initial_cli_forwards_shape_profile(monkeypatch, capsys):
    import sys

    from apps import fairvlmed_case

    calls = []
    monkeypatch.setattr(fairvlmed_case, "prepare", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fairvlmed_case",
            "prepare",
            "--root",
            "input",
            "--sample",
            "arbitrary.npz",
            "--key",
            "new-shape",
            "--provider-profile",
            g.SHAPE_PROFILE,
        ],
    )
    assert fairvlmed_case.main() == 0
    assert calls[0]["provider_profile"] == g.SHAPE_PROFILE


@pytest.mark.parametrize("mutation", ["url", "model", "checksum", "legacy_extension", "path"])
def test_public_config_drift_and_legacy_extension_rejected(tmp_path, monkeypatch, mutation):
    spec, _ = prepared(tmp_path, monkeypatch)
    if mutation in {"url", "model"}:
        path = Path(g.RELAY_PROVIDER)
        config = json.loads(path.read_bytes())
        config["transport"]["base_url" if mutation == "url" else "model"] = (
            "https://invalid.example/v1" if mutation == "url" else "other-model"
        )
        path.write_text(json.dumps(config))
    else:
        if mutation == "checksum":
            spec["provider_configuration_checksum"] = digest("wrong")
        elif mutation == "path":
            spec["provider_configuration"] = g.PROVIDER
        else:
            spec.update(
                schema_version="fairvlmed-run-input/1.4.0", provider_configuration=g.PROVIDER
            )
        (g.directory(spec["key"]) / "input.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="generic_(provider|configuration|legacy_provider)"):
        g.load(spec["key"])


def test_new_wire_accepts_weak_signed_value_and_keeps_original_domain_owner():
    from tests.agent_provider.batch_support import batch_reply, fixture, generate

    calls = []

    def response(body):
        calls.append(body)
        assert "Grounded weak-direction review" in body["messages"][0]["content"]
        assert "REVIEW_CITATION_CODES" in str(body)
        value = batch_reply(body)
        for item in value["items"]:
            if item["status"] == "assessed":
                item["judgment"].update(
                    value=0.02,
                    lower=-0.2,
                    upper=0.2,
                    reason=(
                        "Synthetic visible feature weakly supports the specified comparison, "
                        "with alternatives."
                    ),
                )
        return value

    f = fixture(3, replies=[response], config_path=g.RELAY_PROVIDER)
    sub = generate(f)
    f.acceptance.submit(sub, key="grounded-profile")
    assert f.acceptance.accept(key="grounded-profile").status == "accepted"
    saved = f.acceptance.read(key="grounded-profile")
    assert len(calls) == 1 and saved[2]["scores"]
    assert any(row.judgment.value == 0.02 for row in sub.candidate.items)
    count = len(f.documents.audit)
    f.acceptance.accept(key="grounded-profile")
    assert len(f.documents.audit) == count and len(calls) == 1


def test_review_successor_keeps_explicit_route_and_old_successor_contract(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_review
    from packages.evaluation_preparation.generic_campaign import GenericCampaign
    from packages.llm_provider.config import read_provider_config

    new = read_provider_config(json.loads(Path(g.RELAY_PROVIDER).read_bytes()))
    old = read_provider_config(json.loads(Path(g.PROVIDER).read_bytes()))

    # Only acquisition setup is replaced: the actual successor constructor must
    # retain the bound route instead of silently restoring the historical one.
    def setup(self, spec, registered):
        self.config = new
        self.provider_path = g.RELAY_PROVIDER

    monkeypatch.setattr(GenericCampaign, "__init__", setup)
    monkeypatch.setattr(generic_review, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(generic_review.GenericReview, "prior", lambda self: {"reservations": 4})
    current = generic_review.GenericReview(
        {"schema_version": "fairvlmed-run-input/1.5.0"}, None, "new-review"
    )
    assert current.config == new and current.provider_path == g.RELAY_PROVIDER
    assert current.max_vision_requests == 0 and current.max_new_requests == 10
    legacy = generic_review.GenericReview(
        {"schema_version": "fairvlmed-run-input/1.4.0"}, None, "old-review"
    )
    assert legacy.config == old and legacy.provider_path == g.PROVIDER

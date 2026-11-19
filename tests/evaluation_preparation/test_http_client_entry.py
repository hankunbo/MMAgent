"""Explicit client selection reaches the original terminal/entry/frozen protocol."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CLIENT = "configs/agents/system_curl_client_v1.json"
PUBLIC = "configs/agents/multimodal_relay_local_gpt6_review_curl_v1.json"


def test_cli_exposes_explicit_client_not_silent_default():
    reply = subprocess.run(
        [sys.executable, "-m", "apps.multimodal_case", "live-prepare", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--http-client" in reply.stdout


def public_config():
    from packages.llm_provider.config import read_provider_config

    return read_provider_config(json.loads(Path(PUBLIC).read_bytes()))


def test_exact_client_delta_and_original_config_unchanged():
    from packages.llm_provider.client_selection import apply_selection, load_selection
    from packages.llm_provider.config import read_provider_config

    raw = json.loads(Path("configs/agents/multimodal_relay_local_gpt6_review_v4.json").read_bytes())
    origin = read_provider_config(raw)
    before = origin.model_dump(mode="json")
    selected = apply_selection(origin, load_selection(CLIENT))
    assert selected == public_config()
    assert origin.model_dump(mode="json") == before
    assert apply_selection(origin, None) is origin
    assert selected.transport == origin.transport
    assert selected.roles == origin.roles


@pytest.mark.parametrize("mutation", ["digest", "client", "path", "scope", "provider", "protocol"])
def test_client_binding_rejects_drift_or_missing_authority(mutation):
    from packages.llm_provider.client_selection import check_entry_binding, load_selection

    binding = load_selection(CLIENT)
    spec = dict(
        schema_version="multimodal-live-entry/1.29.0",
        execution_profile="relay-root-question-v1",
        http_client_selection=binding,
        provider=public_config().model_dump(mode="json"),
    )
    protocol = dict(http_client_selection=copy.deepcopy(binding))
    if mutation == "digest":
        binding["client_checksum"] = "sha256:" + "0" * 64
    elif mutation == "client":
        binding["client"]["max_retries"] = 1
    elif mutation == "path":
        binding["path"] = ".local/vectorengine/config.json"
    elif mutation == "scope":
        spec["schema_version"] = "multimodal-live-entry/1.26.0"
    elif mutation == "provider":
        spec["provider"].pop("http_client")
    else:
        protocol.clear()
    with pytest.raises(ValueError):
        check_entry_binding(spec, protocol)


def test_private_client_path_rejected_before_read(tmp_path):
    from packages.llm_provider.client_selection import load_selection

    with pytest.raises(ValueError, match="public_configuration"):
        load_selection(tmp_path / "missing-private.json")


def test_undeclared_client_and_legacy_no_client():
    from packages.llm_provider.client_selection import check_entry_binding

    assert check_entry_binding(dict(provider={})) is None
    with pytest.raises(ValueError, match="missing_entry_binding"):
        check_entry_binding(dict(provider=public_config().model_dump(mode="json")))


def test_original_prepare_preflight_source_successor_and_cold_context(tmp_path, monkeypatch):
    from tests.evaluation_preparation.test_vision_feedback_entry import (
        test_actual_prepare_preflight_provider_and_cold_binding as exercise,
    )

    exercise(tmp_path, monkeypatch, review_contract="rating-presence/1.1.0", http_client=CLIENT)


def test_manifest_real_cli_frozen_client_and_legacy_unchanged(tmp_path, monkeypatch):
    from packages.evaluation_preparation import terminal_cohort as cohort
    from packages.grounding.models import digest

    raw = json.loads(
        Path("docs/acceptance/terminal-local-route-d3xklcNp/cohort/manifest.json").read_bytes()
    )
    old = cohort.read_manifest(copy.deepcopy(raw))
    raw.update(schema_version="terminal-case-cohort/1.12.0", multimodal_http_client=CLIENT)
    current = cohort.read_manifest(raw)
    before, after = cohort.provider_bindings(old), cohort.provider_bindings(current)
    for case in current.cases:
        argv = cohort.commands(current, case, tmp_path, "prepare")[0]
        legacy = cohort.commands(old, case, tmp_path, "prepare")[0]
        assert "--http-client" not in legacy
        if case.dataset == "fairvlmed":
            assert before[case.id] == after[case.id]
            assert "--http-client" not in argv
        else:
            assert argv[argv.index("--http-client") + 1] == CLIENT
            assert after[case.id]["provider_checksum"] == digest(public_config())
            assert before[case.id]["provider_checksum"] != after[case.id]["provider_checksum"]
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(raw))
    cohort.initialize(source, tmp_path / "cohort")
    _, saved, frozen = cohort.load(tmp_path / "cohort", check_current=True)
    assert saved == current
    assert CLIENT in frozen["configurations"]
    assert frozen["provider_bindings"] == after
    raw["schema_version"] = "terminal-case-cohort/1.11.0"
    with pytest.raises(ValueError):
        cohort.read_manifest(raw)


def test_original_provider_assembles_selected_transport():
    from packages.llm_provider.provider import StructuredProvider
    from packages.llm_provider.transport import HTTPTransport

    config = public_config()
    # No request or credential loading: inspect the production Provider assembly.
    provider = StructuredProvider(config, live_opt_in=True)
    assert isinstance(provider.transport, HTTPTransport)
    assert provider.transport.http_client == config.http_client


def test_actual_campaign_freezes_the_selected_client(tmp_path, monkeypatch):
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.grounding.models import digest
    from packages.llm_provider.client_selection import load_selection
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.single_proposition import root_direction_spec

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    identity = SimpleNamespace(
        sample="synthetic", adapter_version="multimodal-adapters/1.2.0", checksum=digest("input")
    )
    config = public_config()
    binding = load_selection(CLIENT)
    base = SimpleNamespace(
        output=tmp_path,
        manifest=identity,
        visible_value=identity,
        task=task,
        inputs=object(),
        definition_expansion_policy={},
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route=config.network_route.model_dump(mode="json"),
        vision_feedback="state-consistency-once/1.0.0",
        review_contract="rating-presence/1.1.0",
        http_client_selection=binding,
        root_direction_review=root_direction_spec(task),
        sources=SimpleNamespace(initial=[]),
        view_checksum=digest("views"),
    )
    monkeypatch.setattr(
        "db.runtime.repository.postgres.validate_runtime_principal", lambda target: None
    )
    monkeypatch.setattr(MultimodalCampaign, "available_targets", staticmethod(lambda base: ()))
    campaign = MultimodalCampaign(base, object(), campaign_id="client-selection-test")
    assert campaign.config == config
    frozen = campaign.freeze()
    assert frozen["http_client_selection"] == binding
    assert frozen["provider"] == digest(config)
    assert campaign.freeze() == frozen
    before = (tmp_path / "protocol.json").read_bytes()
    base.http_client_selection = None
    with pytest.raises(ValueError):
        campaign.freeze()
    assert (tmp_path / "protocol.json").read_bytes() == before

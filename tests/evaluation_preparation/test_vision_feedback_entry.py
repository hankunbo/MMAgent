"""Versioned feedback reaches original prepare/preflight and cold recovery."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import terminal_cohort as batch
from packages.grounding.models import digest
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research import entry
from packages.multimodal_research.report_recovery import restore_execution_context
from tests.dataset_adapters.test_multimodal import gdp
from tests.evaluation_preparation.test_terminal_routed_recovery import routed_manifest

POLICY = "state-consistency-once/1.0.0"


def test_manifest_real_cli_and_source_successor_keep_explicit_policy(tmp_path):
    raw = routed_manifest().model_dump(mode="json")
    raw.update(schema_version="terminal-case-cohort/1.5.0", vision_feedback=POLICY)
    m = batch.read_manifest(raw)
    for action in ("prepare", "source-prepare"):
        argv = batch.commands(m, m.cases[0], tmp_path, action)[0]
        assert argv[:4] == [batch.sys.executable, "-u", "-m", "apps.multimodal_case"]
        assert argv[argv.index("--vision-feedback") + 1] == POLICY
    old = routed_manifest()
    assert "--vision-feedback" not in batch.commands(old, old.cases[0], tmp_path, "prepare")[0]
    with pytest.raises(ValueError):
        batch.read_manifest(dict(raw, vision_feedback="unbounded"))


def test_actual_prepare_preflight_provider_and_cold_binding(
    tmp_path, monkeypatch, *, review_contract=None, http_client=None
):
    from packages.dataset_adapters.multimodal.tasks import load_intent

    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("packages.llm_provider.network_route.load_route", lambda path: route)
    monkeypatch.setattr(
        "packages.multimodal_research.entry_preflight.check_local_route", lambda *a: "offline"
    )
    out = tmp_path / "fresh"
    common = dict(
        dataset="harvard_gdp",
        root=gdp(tmp_path / "data"),
        sample="data_0001.npz",
        task="advanced_glaucoma",
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_harvard_visual_priority_v5.json",
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="harvard-gdp-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route="configs/agents/environment_https_route_v1.json",
        vision_feedback=POLICY,
    )
    if review_contract:
        common["review_contract"] = review_contract
    if http_client:
        common["http_client"] = http_client
    result = entry.prepare_entry(**common, output=out)
    assert result["api_requests"] == result["database_writes"] == 0
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["vision_feedback"] == POLICY
    vision = next(r for r in spec["provider"]["roles"] if r["role"] == "vision")
    assert vision["prompt_version"] == "vision-partial-multimodal-1.3.0"
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    protocol = dict(
        schema_version="multimodal-live-campaign/1.16.0",
        execution_profile=spec["execution_profile"],
        root_direction_review=spec["root_direction_review"],
        review_batch_size=3,
        vision_feedback=POLICY,
    )
    if review_contract:
        protocol["review_contract"] = review_contract
        assert spec["review_contract"] == review_contract
        review = next(r for r in spec["provider"]["roles"] if r["role"] == "review")
        from packages.llm_provider.review_rating import POLICIES

        assert review["prompt_version"] == POLICIES[review_contract]
    if http_client:
        protocol["http_client_selection"] = spec["http_client_selection"]
        assert spec["provider"]["schema_version"] == "agent-provider-config/9.2.0"
        assert spec["provider"]["http_client"] == spec["http_client_selection"]["client"]
    base = SimpleNamespace(task=load_intent(common["task"], common["registry"])[0])
    restore_execution_context(base, spec, protocol)
    assert base.vision_feedback == POLICY
    if http_client:
        assert base.http_client_selection == spec["http_client_selection"]
        with pytest.raises(ValueError, match="client_selection_protocol"):
            restore_execution_context(base, spec, dict(protocol, http_client_selection=None))
    if review_contract:
        assert base.review_contract == review_contract
        with pytest.raises(ValueError, match="review_rating_protocol"):
            restore_execution_context(base, spec, dict(protocol, review_contract=None))
    with pytest.raises(ValueError, match="vision_feedback"):
        restore_execution_context(base, spec, dict(protocol, vision_feedback=None))
    # The original Source successor must preserve this policy in both envelopes.
    from packages.multimodal_research.discovery_continuation import definition_parent
    from tests.evaluation_preparation.test_routed_source_backlog import routed_parent

    _, ref, _, plan, failure, _, _ = routed_parent(tmp_path)
    plan.update(
        input=spec["input_checksum"],
        implementation=spec["implementation"],
        provider=digest(spec["provider"]),
        vision_feedback=POLICY,
    )
    if review_contract:
        plan["review_contract"] = review_contract
    if http_client:
        plan["http_client_selection"] = spec["http_client_selection"]
    failure.update(input_checksum=spec["input_checksum"], protocol_checksum=digest(plan))
    parent = Path(ref["parent"])
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (parent / (name + ".json")).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in parent.glob("*.json")}
    child = tmp_path / "vision-successor"
    entry.prepare_entry(
        **common,
        output=child,
        definition_failure_of=parent,
        defer_failed_definition_expansion=True,
        source_fallback_seconds=1800,
        source_fallback_authorization="test-no-source-replay",
    )
    successor = json.loads((child / "entry.json").read_bytes())
    assert successor["provider"] == spec["provider"] and successor["vision_feedback"] == POLICY
    if review_contract:
        assert successor["review_contract"] == review_contract
    if http_client:
        assert successor["http_client_selection"] == spec["http_client_selection"]
    assert entry.assemble_entry(child, preflight_only=True)["status"] == "local_preflight_passed"
    assert {p.name: p.read_bytes() for p in parent.glob("*.json")} == before
    plan.pop("vision_feedback")
    (parent / "protocol.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="vision_feedback_protocol"):
        definition_parent(
            parent,
            checksum=spec["input_checksum"],
            intent=spec["intent"],
            defer=True,
            root_backlog=True,
            routed_backlog=True,
        )
    spec.pop("vision_feedback")
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="vision_feedback"):
        entry.assemble_entry(out, preflight_only=True)

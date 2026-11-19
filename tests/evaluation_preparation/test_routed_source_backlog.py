"""Explicit no-Source-replay successor, preserving original owners and counts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.config import with_network_route
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research.discovery_continuation import (
    ROUTED_BACKLOG,
    ConfirmedDefinitionContinuation,
    check_fallback_window,
    definition_successor_policy,
    fallback_window,
    reuse_failed_native,
)
from tests.evaluation_preparation.test_root_definition_backlog import root_parent


def routed_parent(tmp_path, *, final_error="transport"):
    c, ref, spec, plan, failure, records, writes = root_parent(tmp_path)
    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    config = with_network_route(c.config, route)
    path = "configs/research/multimodal_grape_active_requirements_v6.json"
    spec.update(
        schema_version="multimodal-live-entry/1.29.0",
        provider=config.model_dump(mode="json"),
        network_route=route.model_dump(mode="json"),
        active_acquisition=dict(path=path, checksum=digest(json.loads(Path(path).read_bytes()))),
    )
    c.config = config
    c.base.manifest = SimpleNamespace(checksum=spec["input_checksum"])
    plan["provider"] = digest(config)
    failure.update(
        error_code=final_error,
        protocol_checksum=digest(plan),
        failure_kind=(
            "confirmed_response_or_domain_refusal"
            if final_error == "schema"
            else "dispatched_unknown"
        ),
    )
    failure["transport"]["remote_response_unknown"] = int(final_error != "schema")
    # Match the real incident: schema rejection, then one correction whose
    # response is unknown. No accepted draft and no definition revision.
    for index, error in ((1, "schema"), (3, final_error)):
        record = records[f"journal:{index}"][0]["record"]
        text = '{"definitions":[{"wrong":"shape"}]}' if error == "schema" else ""
        record.update(error=error, output_checksum=digest(text.encode().hex()) if text else None)
        records["definition:response:" + record["request_id"]][0]["raw_text"] = text
    records.pop("definition:gap:0:decision")
    parent = Path(ref["parent"])
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (parent / (name + ".json")).write_text(json.dumps(value))
    ref.update(
        entry=digest(spec), protocol=digest(plan), failure=digest(failure), mode=ROUTED_BACKLOG
    )
    return c, ref, spec, plan, failure, records, writes


@pytest.mark.parametrize("error", ["schema", "timeout", "transport"])
def test_owner_claim_preserves_failed_response_and_cumulative_budget(tmp_path, error):
    c, ref, spec, plan, failure, records, writes = routed_parent(tmp_path, final_error=error)
    before = copy.deepcopy(records)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    assert continuation.routed_backlog and continuation.defer
    assert continuation.check_attempts()[0]["requests"] == 3  # 2 model + 1 public HTTP
    assert len(writes) == 1
    assert all(records[k] == v for k, v in before.items())
    assert ConfirmedDefinitionContinuation(c, ref, read_only=True).attempts == continuation.attempts
    assert ConfirmedDefinitionContinuation(c, ref).attempts == continuation.attempts
    assert len(writes) == 1
    assert json.loads((Path(ref["parent"]) / "protocol.json").read_bytes()) == plan
    assert json.loads((Path(ref["parent"]) / "stage-failure.json").read_bytes()) == failure
    definition_successor_policy(
        spec,
        json.loads(Path(spec["active_acquisition"]["path"]).read_bytes()),
        defer=True,
        routed_backlog=True,
    )


@pytest.mark.parametrize(
    "change",
    [
        "wire_unknown",
        "earlier_unknown",
        "missing_response",
        "changed_response",
        "uncommitted_revision",
        "acquisition_started",
        "route_changed",
        "cancelled",
    ],
)
def test_unsafe_parent_cannot_claim_or_authorize_dispatch(tmp_path, change):
    c, ref, spec, plan, failure, records, writes = routed_parent(tmp_path)
    if change == "wire_unknown":
        records.pop("definition:wire:0:outcome")
    elif change == "earlier_unknown":
        records["journal:1"][0]["record"]["error"] = "transport"
    elif change == "missing_response":
        records.pop("definition:response:r0")
    elif change == "changed_response":
        records["definition:response:r0"][0]["raw_text"] = "{}"
    elif change == "uncommitted_revision":
        records["definition:revision"] = ({"saved": True},)
    elif change == "acquisition_started":
        c.contexts.read_research_run = lambda rid: {"exists": True}
    elif change == "route_changed":
        spec["network_route"]["endpoint_checksum"] = digest("other")
        (Path(ref["parent"]) / "entry.json").write_text(json.dumps(spec))
        ref["entry"] = digest(spec)
    else:
        (Path(ref["parent"]) / "cancel").touch()
    with pytest.raises(ValueError, match="definition_successor"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


def test_new_window_explicit_bounded_and_not_a_parent_deadline_rewrite(tmp_path):
    c, ref, _, parent, _, _, writes = routed_parent(tmp_path)
    original = (Path(ref["parent"]) / "protocol.json").read_bytes()
    window = fallback_window(parent, seconds=5400, authorization="goal-bounded-source-fallback")
    c.base.source_fallback_window = window
    c.base.active_start = check_fallback_window(window, parent)
    owner = ConfirmedDefinitionContinuation(c, ref)
    assert owner.claim["successor_window"] == window
    assert owner.attempts[0]["requests"] == 3
    assert (Path(ref["parent"]) / "protocol.json").read_bytes() == original
    assert ConfirmedDefinitionContinuation(c, ref, read_only=True).claim == owner.claim
    assert len(writes) == 1
    bad = dict(window, seconds=5401)
    with pytest.raises(ValueError, match="definition_fallback_window"):
        check_fallback_window(bad, parent)
    with pytest.raises(ValueError, match="definition_fallback_window"):
        fallback_window(parent, seconds=True, authorization="not-an-integer-budget")
    with pytest.raises(ValueError, match="definition_fallback_window"):
        check_fallback_window(dict(window, deadline=parent["deadline"]), parent)


@pytest.mark.parametrize("closed_public_wire", [False, True])
def test_same_native_fact_can_be_reused_despite_independent_source_unknown(
    tmp_path, monkeypatch, closed_public_wire
):
    c, ref, spec, _, _, _, _ = routed_parent(tmp_path)
    if closed_public_wire:
        ref["public_wire_policy"] = "closed-public-wire-backlog/1.0.0"
    c.base.output = tmp_path / "native-successor"
    c.base.output.mkdir()
    requirement = dict(id="check:one", component="image:one")
    record = dict(
        schema_version="native-fundus-acceptance/1.1.0",
        plan=dict(target=requirement["id"], component=requirement["component"]),
        bundle_key="original-native-bundle",
    )
    suffix = digest((requirement["id"], requirement["component"]))[7:39]
    parent_folder = Path(ref["parent"]) / "native-fundus" / suffix
    parent_folder.mkdir(parents=True)
    (parent_folder / "accepted.json").write_text(json.dumps(record))
    reads = []
    monkeypatch.setattr(
        "packages.multimodal_research.native_fundus.recheck",
        lambda base, value: reads.append((base, copy.deepcopy(value))),
    )
    runner = reuse_failed_native(c.base, ref)
    assert (
        runner(
            c.base,
            requirement,
            before_dispatch=lambda: pytest.fail("new GPU"),
            component_scoped=True,
        )
        == record
    )
    cold = reuse_failed_native(c.base, ref, read_only=True)
    assert (
        cold(
            c.base,
            requirement,
            before_dispatch=lambda: pytest.fail("new GPU"),
            component_scoped=True,
        )
        == record
    )
    assert len(reads) == 2
    receipt = json.loads(next(c.base.output.glob("native-failure-reuse-*.json")).read_bytes())
    assert receipt["new_model_runs"] == 0 and receipt["original_failure"] == ref


def test_original_definition_owner_records_backlog_and_schedules_baseline(tmp_path):
    from packages.multimodal_research.definition_expansion import DefinitionExpansion
    from tests.evaluation_preparation.test_definition_expansion import fixture

    c, ref, spec, _, _, _, _ = routed_parent(tmp_path)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    profile = json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())
    seed = fixture(tmp_path)
    seed.campaign.definition_continuation = continuation
    seed.campaign.base.task.target_code = profile["definition_expansion"]["public_target"]
    expansion = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=seed.sources,
        provider=seed.provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=profile["definition_expansion"],
    )
    saved = expansion.advance()
    assert saved["expansion_stop"] == "terminal_source_technical_backlog"
    assert saved["rows"] == seed.campaign.base.ledger
    assert saved["rejected"][0]["reason"] == "source_failure_quarantined_not_replayed"
    assert saved["rejected"][0]["medical_no_addition_claimed"] is False
    assert saved["added"] == [] and seed.budget.used == 0
    assert expansion.advance() == expansion.read_revision() == saved


@pytest.mark.parametrize("provider_transition", [False, True])
@pytest.mark.parametrize("closed_public_wire", [False, True])
def test_original_cli_preparation_preflight_and_cold_context_preserve_route_and_window(
    tmp_path, monkeypatch, provider_transition, closed_public_wire
):
    from packages.multimodal_research import entry
    from packages.multimodal_research.report_recovery import restore_execution_context
    from tests.dataset_adapters.test_multimodal import gdp

    _, ref, spec, plan, failure, _, _ = routed_parent(tmp_path)
    profile_path = "configs/research/multimodal_harvard_visual_priority_v5.json"
    spec["active_acquisition"] = dict(
        path=profile_path, checksum=digest(json.loads(Path(profile_path).read_bytes()))
    )
    root = gdp(tmp_path / "data")
    policy = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setattr("packages.llm_provider.network_route.load_route", lambda path: policy)
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "packages.multimodal_research.entry_preflight.check_local_route", lambda *a: "offline"
    )
    common = dict(
        dataset="harvard_gdp",
        root=root,
        sample="data_0001.npz",
        task="advanced_glaucoma",
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition=spec["active_acquisition"]["path"],
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="harvard-gdp-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route="configs/agents/environment_https_route_v1.json",
    )
    entry.prepare_entry(**common, output=tmp_path / "seed")
    current = json.loads((tmp_path / "seed/entry.json").read_bytes())
    spec.update(current)
    plan.update(input=spec["input_checksum"], implementation=spec["implementation"])
    failure.update(input_checksum=spec["input_checksum"], protocol_checksum=digest(plan))
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (Path(ref["parent"]) / (name + ".json")).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in Path(ref["parent"]).glob("*.json")}
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("DB during preparation/preflight"))
    output = tmp_path / "explicit-window"
    transition_options = {}
    if provider_transition:
        policy = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8890"})
        monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8890")
        transition_options["review_contract"] = "rating-presence/1.1.0"
    result = entry.prepare_entry(
        **common,
        **transition_options,
        output=output,
        definition_failure_of=ref["parent"],
        defer_failed_definition_expansion=True,
        retain_closed_public_wire_failures=closed_public_wire,
        source_fallback_seconds=3600,
        source_fallback_authorization="test_explicit_new_window",
    )
    value = json.loads((output / "entry.json").read_bytes())
    assert result["api_requests"] == result["database_writes"] == 0
    assert value["schema_version"] == "multimodal-live-entry/1.30.0"
    if provider_transition:
        from packages.multimodal_research.discovery_continuation import (
            definition_provider_transition,
        )

        proof = definition_provider_transition(spec, value["provider"], value["review_contract"])
        assert value["definition_failure_of"]["provider_transition"] == proof
        assert value["provider"] != spec["provider"]
    else:
        assert value["provider"] == spec["provider"]
        assert "provider_transition" not in value["definition_failure_of"]
    assert value["definition_failure_of"]["mode"] == ROUTED_BACKLOG
    assert value["definition_failure_of"].get("public_wire_policy") == (
        "closed-public-wire-backlog/1.0.0" if closed_public_wire else None
    )
    assert value["source_fallback_window"]["parent_deadline"] == plan["deadline"]
    assert entry.assemble_entry(output, preflight_only=True)["status"] == "local_preflight_passed"
    from packages.dataset_adapters.multimodal.tasks import load_intent

    task = load_intent("advanced_glaucoma", common["registry"])[0]
    base = SimpleNamespace(task=task)
    frozen = dict(
        schema_version="multimodal-live-campaign/1.19.0",
        execution_profile=value["execution_profile"],
        root_direction_review=value["root_direction_review"],
        review_batch_size=3,
        definition_continuation=value["definition_failure_of"],
        source_fallback_window=value["source_fallback_window"],
        **check_fallback_window(value["source_fallback_window"], plan),
    )
    if provider_transition:
        frozen["review_contract"] = value["review_contract"]
    restore_execution_context(base, value, frozen)
    assert base.network_route == value["network_route"]
    assert base.source_fallback_window == value["source_fallback_window"]
    with pytest.raises(ValueError, match="saved_report_fallback_window"):
        restore_execution_context(base, value, dict(frozen, deadline=plan["deadline"]))
    assert {p.name: p.read_bytes() for p in Path(ref["parent"]).glob("*.json")} == before
    broken = copy.deepcopy(value)
    broken["source_fallback_window"]["seconds"] = 1
    (output / "entry.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="definition_fallback_window"):
        entry.assemble_entry(output, preflight_only=True)


def test_actual_campaign_freeze_keeps_new_window_and_original_attempts(tmp_path, monkeypatch):
    from packages.multimodal_research import implementation
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.report_recovery import restore_execution_context

    parent_campaign, ref, spec, old, _, _, _ = routed_parent(tmp_path)
    window = fallback_window(old, seconds=1800, authorization="explicit-bounded-test")
    parent_campaign.base.source_fallback_window = window
    parent_campaign.base.active_start = check_fallback_window(window, old)
    continuation = ConfirmedDefinitionContinuation(parent_campaign, ref)
    current = object.__new__(MultimodalCampaign)
    current.output = tmp_path / "frozen"
    current.output.mkdir()
    current.campaign_id = "routed-backlog"
    sha = SimpleNamespace(checksum=spec["input_checksum"])
    current.base = SimpleNamespace(
        manifest=sha,
        visible_value=sha,
        task=parent_campaign.base.task,
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
        execution_profile=spec["execution_profile"],
        root_direction_review=spec["root_direction_review"],
        review_batch_size=3,
        source_fallback_window=window,
        network_route=spec["network_route"],
        active_start=check_fallback_window(window, old),
    )
    current.config = parent_campaign.config
    current.targets = ()
    current.definition_continuation = continuation
    current.previous_attempts = continuation.check_attempts
    monkeypatch.setattr(implementation, "freeze", lambda: {})
    frozen = current.freeze()
    assert frozen["schema_version"] == "multimodal-live-campaign/1.19.0"
    assert frozen["source_fallback_window"] == window
    assert frozen["deadline"] == window["deadline"] != old["deadline"]
    assert frozen["previous_attempts"][0]["requests"] == 3
    assert current.freeze() == frozen
    value = dict(
        spec,
        schema_version="multimodal-live-entry/1.30.0",
        definition_failure_of=ref,
        source_fallback_window=window,
    )
    base = SimpleNamespace(task=current.base.task)
    restore_execution_context(base, value, frozen)
    assert base.source_fallback_window == window
    assert base.network_route == spec["network_route"]

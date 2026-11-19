"""A rejected optional definition cannot erase qualified baseline requirements."""

import copy
import json
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.definition_expansion import DefinitionExpansion
from packages.multimodal_research.discovery_continuation import (
    ConfirmedDefinitionContinuation,
    definition_successor_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_definition_failure_successor import parent_fixture


def qualified_parent(tmp_path, profile_version=2):
    c, ref, spec, plan, failure, records, writes = parent_fixture(tmp_path)
    spec.update(
        schema_version="multimodal-live-entry/1.23.0",
        component_consumer=None,
        input_semantics="grape-v1",
        dataset="grape",
    )
    profile_path = f"configs/research/multimodal_grape_active_requirements_v{profile_version}.json"
    profile = json.loads(Path(profile_path).read_bytes())
    spec["active_acquisition"] = dict(path=profile_path, checksum=digest(profile))
    c.base.component_consumer_selection = None
    failure["new_requests"] = 2
    failure["operations"].append(dict(kind="wire", key="definition:wire:0"))
    records["definition:wire:0:outcome"] = (
        dict(response_checksum=digest("public source"), http_status=200),
    )
    records["multimodal-stage-failure"] = (failure,)
    parent = Path(ref["parent"])
    (parent / "entry.json").write_text(json.dumps(spec))
    (parent / "stage-failure.json").write_text(json.dumps(failure))
    ref.update(
        entry=digest(spec),
        failure=digest(failure),
        mode="confirmed_schema_backlog_no_source_replay",
    )
    return c, ref, spec, plan, failure, records, writes, profile


def preceding_empty_proposal(tmp_path):
    from packages.evidence_reuse.definitions import OwnerBoundDefinitionProposal

    c, ref, spec, plan, failure, records, writes, profile = qualified_parent(tmp_path, 3)
    proposal = OwnerBoundDefinitionProposal(
        definitions=(), no_addition_reason="Available materials do not add a supported definition"
    )
    raw = proposal.model_dump_json()
    record = dict(
        records["journal:1"][0]["record"],
        error=None,
        request_id="r0",
        output_checksum=digest(proposal),
    )
    records["journal:2"], records["journal:3"] = records["journal:0"], records["journal:1"]
    records["journal:1"] = (dict(phase="completed", record=record),)
    records["definition:response:r0"] = (dict(record=record, raw_text=raw),)
    records["definition:gap:0:decision"] = (dict(candidate=dict(added=[], replaced=[])),)
    failure["new_requests"] = 3
    failure["transport"]["dispatched_observed"] = 2
    (Path(ref["parent"]) / "stage-failure.json").write_text(json.dumps(failure))
    ref["failure"] = digest(failure)
    return c, ref, failure, records, writes


def test_confirmed_empty_first_round_then_schema_failure_preserves_same_window(tmp_path):
    c, ref, failure, records, writes = preceding_empty_proposal(tmp_path)
    before = copy.deepcopy(records)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    assert continuation.check_attempts()[0]["requests"] == 3
    assert (
        ConfirmedDefinitionContinuation(c, ref, read_only=True).check_attempts()
        == continuation.attempts
    )
    assert len(writes) == 1
    assert all(records[key] == value for key, value in before.items())
    assert json.loads((Path(ref["parent"]) / "stage-failure.json").read_bytes()) == failure


@pytest.mark.parametrize(
    "mutation", ["missing_decision", "increment", "response_changed", "last_not_failed"]
)
def test_mixed_confirmed_history_requires_empty_resolved_first_round(tmp_path, mutation):
    c, ref, _, records, writes = preceding_empty_proposal(tmp_path)
    if mutation == "missing_decision":
        records.pop("definition:gap:0:decision")
    elif mutation == "increment":
        records["definition:gap:0:decision"][0]["candidate"]["added"] = ["new"]
    elif mutation == "response_changed":
        records["definition:response:r0"][0]["raw_text"] = "{}"
    else:
        records["journal:1"], records["journal:3"] = records["journal:3"], records["journal:1"]
    with pytest.raises(ValueError):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


@pytest.mark.parametrize("profile_version", [2, 3])
def test_confirmed_schema_backlog_uses_original_revision_owner_and_budget(
    tmp_path, profile_version
):
    c, ref, spec, plan, failure, records, writes, profile = qualified_parent(
        tmp_path, profile_version
    )
    before = copy.deepcopy((spec, plan, failure))
    definition_successor_policy(spec, profile, defer=True)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    seed = fixture(tmp_path)
    seed.campaign.definition_continuation = continuation
    seed.campaign.base.task.target_code = profile["definition_expansion"]["public_target"]
    e = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=seed.sources,
        provider=seed.provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=profile["definition_expansion"],
    )
    revision = e.advance()
    assert revision["expansion_stop"] == "terminal_source_technical_backlog"
    assert revision["rejected"][0]["reason"] == "source_schema_confirmed_not_replayed"
    assert revision["rejected"][0]["medical_no_addition_claimed"] is False
    assert revision["rows"] == seed.campaign.base.ledger
    assert revision["added"] == [] and seed.budget.used == 0
    assert e.advance() == revision and e.read_revision() == revision
    assert continuation.check_attempts()[0]["requests"] == 2
    assert (
        ConfirmedDefinitionContinuation(c, ref, read_only=True).check_attempts()
        == continuation.attempts
    )
    assert len(writes) == 1 and (spec, plan, failure) == before
    records["definition:wire:0:outcome"] = (dict(response_checksum=None, http_status=None),)
    with pytest.raises(ValueError, match="history_changed"):
        continuation.check_attempts()


@pytest.mark.parametrize(
    "mutation",
    [
        "wire_unknown",
        "wire_bool",
        "wire_duplicate",
        "accounting",
        "unknown",
        "increment",
        "profile",
    ],
)
def test_invalid_confirmed_backlog_cannot_claim(tmp_path, mutation):
    c, ref, spec, _, failure, records, writes, profile = qualified_parent(tmp_path)
    if mutation == "wire_unknown":
        records.pop("definition:wire:0:outcome")
    elif mutation == "wire_bool":
        records["definition:wire:0:outcome"] = (
            dict(response_checksum=digest("public source"), http_status=True),
        )
    elif mutation == "wire_duplicate":
        failure["operations"].append(dict(kind="wire", key="definition:wire:0"))
        failure["new_requests"] = 3
    elif mutation == "accounting":
        failure["new_requests"] = 1
    elif mutation == "unknown":
        failure["transport"]["remote_response_unknown"] = 1
    elif mutation == "increment":
        records["definition:gap:0:decision"] = (dict(candidate=dict(added=["x"], replaced=[])),)
    else:
        profile["checks"].pop()
        with pytest.raises(ValueError, match="plan_delta"):
            definition_successor_policy(spec, profile, defer=True)
        return
    (Path(ref["parent"]) / "stage-failure.json").write_text(json.dumps(failure))
    ref["failure"] = digest(failure)
    with pytest.raises(ValueError, match="definition_successor_"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


def test_confirmed_schema_with_reused_local_source_needs_no_new_public_wire(tmp_path):
    c, ref, _, _, failure, records, writes, _ = qualified_parent(tmp_path, 3)
    failure["operations"] = [o for o in failure["operations"] if o["kind"] != "wire"]
    failure["new_requests"] = 1
    records.pop("definition:wire:0:outcome")
    records["multimodal-stage-failure"] = (failure,)
    (Path(ref["parent"]) / "stage-failure.json").write_text(json.dumps(failure))
    ref["failure"] = digest(failure)
    value = ConfirmedDefinitionContinuation(c, ref)
    assert value.check_attempts()[0]["requests"] == 1
    assert value.wire_proofs == [] and len(writes) == 1
    assert ConfirmedDefinitionContinuation(c, ref).check_attempts() == value.attempts
    assert len(writes) == 1


@pytest.mark.parametrize("profile_version", [2, 3])
def test_original_prepare_and_preflight_versioned_same_window_entry(
    tmp_path, monkeypatch, profile_version
):
    from packages.multimodal_research.entry import assemble_entry, prepare_entry
    from tests.dataset_adapters.test_multimodal import grape

    c, ref, spec, plan, failure, _, _, profile = qualified_parent(tmp_path, profile_version)
    root = grape(tmp_path / "data")
    # Use the actual selected input/intent, rather than accepting fixture aliases.
    common = dict(
        dataset="grape",
        root=root,
        sample="1:OD",
        cutoff_visit=1,
        task="advanced_glaucoma",
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition=spec["active_acquisition"]["path"],
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="grape-v1",
        execution_profile="relay-snapshot-context-v1",
    )
    seed_path = tmp_path / "seed"
    prepare_entry(**common, output=seed_path)
    real = json.loads((seed_path / "entry.json").read_bytes())
    spec.update(real)
    plan.update(
        input=real["input_checksum"],
        task=real["intent"]["checksum"],
        implementation=real["implementation"],
    )
    failure.update(input_checksum=real["input_checksum"], protocol_checksum=digest(plan))
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (Path(ref["parent"]) / (name + ".json")).write_text(json.dumps(value))

    def forbidden():
        pytest.fail("preparation/preflight may not install live owners")

    monkeypatch.setattr("packages.multimodal_research.entry.owners", forbidden)
    out = tmp_path / "actual-successor"
    result = prepare_entry(
        **common,
        output=out,
        definition_failure_of=ref["parent"],
        defer_failed_definition_expansion=True,
    )
    new = json.loads((out / "entry.json").read_bytes())
    assert result["api_requests"] == result["database_writes"] == 0
    assert new["schema_version"] == "multimodal-live-entry/1.24.0"
    assert new["definition_failure_of"]["mode"] == "confirmed_schema_backlog_no_source_replay"
    assert (
        new["provider"] == spec["provider"]
        and new["active_acquisition"] == spec["active_acquisition"]
    )
    assert assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    new["definition_failure_of"]["mode"] = "technical_backlog_no_source_replay"
    (out / "entry.json").write_text(json.dumps(new))
    with pytest.raises(ValueError, match="deferral_version"):
        assemble_entry(out, preflight_only=True)

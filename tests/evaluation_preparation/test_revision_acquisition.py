"""Accepted interpretation revisions must reach eligible original acquisition."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from packages.evidence_reuse.definition_acceptance import requirement_rows
from packages.grounding.models import digest
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    definition_config,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_owner_dependency_definition import raw


def policy(version="1.33.0"):
    value = json.loads(
        Path("configs/research/multimodal_definition_raw_revision_v1.json").read_bytes()
    )
    value.update(
        schema_version="campaign-definition-expansion/" + version,
        prompt_version="definition-draft-6.7.0"
        if version == "1.35.0"
        else "definition-draft-6.6.0"
        if version == "1.34.0"
        else "definition-draft-6.4.0",
        unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0",
    )
    if version in {"1.32.0", "1.34.0", "1.35.0"}:
        value.update(
            index_keys=["synthetic-current-index"],
            unresolved_relation_retrieval=None,
            retrieval_scope="confirmed-index-only/1.0.0",
        )
    return value


def campaign(tmp_path, *, attempted=(), confirmed=(), version="1.33.0"):
    parent_result = dict(phases=[dict(target=t, accepted=False) for t in attempted])
    old = tmp_path / "prior"
    old.mkdir()
    (old / "result.json").write_text(json.dumps(parent_result))
    history = [dict(directory=str(old), checksum=digest(parent_result), requests=len(attempted))]
    values = {t: {"synthetic_confirmed": t} for t in confirmed}
    return NS(
        protocol=dict(
            expansion_policy=policy(version),
            confirmed_observations_checksum=digest(values),
        ),
        parent=NS(accepted_observations=lambda: copy.deepcopy(values)),
        previous_attempts=lambda: history,
        base=NS(observations={}),
        targets=("check:unattempted", "check:other"),
    )


def revision():
    return dict(
        added=[],
        replaced=[dict(id="check:unattempted")],
        rows=[dict(id="check:unattempted", review_allowed=True)],
    )


@pytest.mark.parametrize("version", ["1.32.0", "1.33.0", "1.34.0", "1.35.0"])
def test_new_version_schedules_revised_unobserved_check(tmp_path, version):
    c = campaign(tmp_path, version=version)
    assert read_expansion_policy(c.protocol["expansion_policy"])
    assert renewal.gate_selection(c, revision()) == ("check:unattempted",)
    assert c.targets == ("check:unattempted",)


@pytest.mark.parametrize("state", ["confirmed", "attempted_unknown", "native_fact"])
def test_no_implicit_repeat_of_prior_work(tmp_path, state):
    c = campaign(
        tmp_path,
        attempted=("check:unattempted",) if state == "attempted_unknown" else (),
        confirmed=("check:unattempted",) if state == "confirmed" else (),
    )
    if state == "native_fact":
        c.base.observations["check:unattempted"] = object()
    renewal.gate_selection(c, revision())
    assert c.targets == ()


def test_restored_current_phase_remains_in_schedule_for_original_receipt_read(tmp_path):
    c = campaign(tmp_path)
    c.accepted_observations = lambda: {"check:unattempted": object()}
    renewal.gate_selection(c, revision())
    # The phase owner reads its exact saved receipt; removing this target would
    # also remove its observation from MultimodalCampaign.accepted_observations.
    assert c.targets == ("check:unattempted",)


def test_missing_input_does_not_get_synthesized_by_selection(tmp_path):
    c = campaign(tmp_path)
    c.targets = ()  # Original select_targets rejected the absent component.
    renewal.gate_selection(c, revision())
    assert c.targets == ()


@pytest.mark.parametrize("mutation", ["history", "confirmed"])
def test_current_owner_history_or_fact_drift_rejects(tmp_path, mutation):
    c = campaign(tmp_path)
    if mutation == "history":
        (tmp_path / "prior" / "result.json").write_text('{"phases": [] , "changed": true}')
    else:
        c.parent.accepted_observations = lambda: {"unexpected": "fact"}
    with pytest.raises(ValueError, match="history_changed|observations_changed"):
        renewal.gate_selection(c, revision())


def test_old_policy_preserves_its_frozen_added_only_schedule(tmp_path):
    c = campaign(tmp_path, version="1.30.0")
    renewal.gate_selection(c, revision())
    assert c.targets == ()


@pytest.mark.parametrize("version", ["1.32.0", "1.33.0", "1.34.0", "1.35.0"])
def test_new_policy_uses_original_runtime_and_exact_retrieval_scope(version):
    from packages.multimodal_research.execution_lease import PROFILE

    parsed = read_expansion_policy(policy(version))
    assert read_expansion_policy(parsed.model_dump(mode="json")) == parsed
    plan = dict(
        schema_version=renewal.RUNTIME_DEFINITION_VERSION,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=parsed.model_dump(mode="json"),
        execution_lease_profile=PROFILE,
        new_public_search_allowed=version not in {"1.32.0", "1.34.0", "1.35.0"},
    )
    renewal.validate_definition_runtime(plan)
    if version in {"1.32.0", "1.34.0", "1.35.0"}:
        from packages.llm_provider.provider import ProviderError

        e = object.__new__(DefinitionExpansion)
        e.policy = parsed
        with pytest.raises(ProviderError, match="definition_index_only_no_public_dispatch"):
            e.reserve_wire("PUBLIC", kind="search")
        with pytest.raises(ValueError, match="definition_index_only_no_public_dispatch"):
            renewal.validate_definition_runtime(dict(plan, new_public_search_allowed=True))


def test_unresolved_dependency_cannot_become_dispatchable(tmp_path):
    c = campaign(tmp_path)
    r = revision()
    r["rows"][0].update(dependency_assessment={"items": []}, review_allowed=False)
    assert renewal.gate_selection(c, r) == ()
    assert c.targets == ()


def test_increment_acquisition_cap_is_not_expanded(tmp_path):
    c = campaign(tmp_path)
    r = revision()
    r["added"] = ["check:other"]
    with pytest.raises(ValueError, match="definition_gate_increment_bound"):
        renewal.gate_selection(c, r)


@pytest.mark.parametrize("version", ["1.33.0", "1.34.0", "1.35.0"])
@pytest.mark.parametrize("implicit_vision_method", [False, True])
def test_original_definition_owner_replacement_reaches_renewal_dispatch(
    tmp_path, version, implicit_vision_method
):
    old_owner, manifest, sources, _, input_base, value = raw()
    value.update(schema_version="source-definition-draft/6.1.0", target="GOAL:ADVANCED_GLAUCOMA")
    for excerpt in value["excerpts"]:
        excerpt.pop("start", None)
        excerpt.pop("end", None)
    old_owner.binding["target"] = value["target"]
    old_receipt = old_owner.accept("old", value)
    row = requirement_rows(old_receipt, manifest, fact_reader=old_owner.fact_reader)[0]
    row.update(id="check:unattempted", parent="EXISTING")
    row.pop("definition_receipt_checksum")
    if implicit_vision_method:
        row.pop("expected_method")
        row.pop("expected_unit")
    original = copy.deepcopy(row)
    value["interpretation"] = "New source-grounded synthetic relation; not a clinical rule."
    seed = fixture(tmp_path)
    base = seed.campaign.base
    base.manifest, base.ledger = manifest, [row]
    base.task.target_code = value["target"]
    base.read_dependency_input = lambda: manifest
    base.read_dependency_fact = old_owner.fact_reader
    base.prepared = input_base.prepared
    selected_policy = policy(version)
    selected_policy["query_requirements"] = {
        q: [row["id"]] for q in selected_policy["public_queries"]
    }
    sources.index_keys = ("synthetic-index",)
    sources.restore_indexes = lambda _: None
    sources.search = lambda **kw: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[sources.read(sources.value["code"])],
        acquired_indexes=[],
        backlog=[],
    )
    sources.reconcile_pool = lambda **kw: dict(
        schema_version="research-definition-pool/1.0.0",
        materials=[sources.read(sources.value["code"])],
        backlog=[],
        retrieval={"synthetic": True},
    )
    provider = DeterministicFakeProvider(
        definition_config(
            seed.provider.config,
            prompt_version=selected_policy["prompt_version"],
            source_output_tokens=8192,
        ),
        [
            dict(
                schema_version="research-definition-proposal/5.0.0",
                definitions=[value],
                no_addition_reason=None,
            )
        ],
    )
    c = object.__new__(renewal.RenewedCampaign)
    c.__dict__.update(seed.campaign.__dict__)
    c.parent = NS(accepted_observations=lambda: {})
    c.previous_attempts = lambda: []
    c.guard = lambda **kw: None
    c.guard_definition_materials = lambda **kw: None
    c.protocol = dict(
        c.protocol,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=selected_policy,
        confirmed_observations_checksum=digest({}),
    )
    e = DefinitionExpansion(
        c,
        documents=seed.documents,
        sources=sources,
        provider=provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=selected_policy,
    )
    c.definition_expansion = e
    revised = c.expand_definitions()
    assert not revised["added"] and [r["id"] for r in revised["replaced"]] == [row["id"]]
    assert revised["denominator_before"] == revised["denominator_after"] == 1
    assert c.targets == (row["id"],) and c.base.review_target_ids == (row["id"],)
    assert base.observations == {} and row == original
    assert e.owner.read(revised["replaced"][0]["definition"]) in revised["accepted"].values()
    schedule = json.loads((tmp_path / "increment-schedule.json").read_bytes())
    assert schedule["scheduled"] == [row["id"]]
    assert c.expand_definitions() == revised and seed.budget.used == 1
    assert old_owner.read("old", historical=True) == old_receipt
    assert c.expand_definitions(restore_only=True, report_only=True) == revised
    assert seed.budget.used == 1 and row == original

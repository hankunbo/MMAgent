"""Confirmed Source-format failure: explicit successor, not UNKNOWN retry."""

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research.definition_expansion import (
    definition_config,
    read_expansion_policy,
)
from packages.multimodal_research.discovery_continuation import (
    ConfirmedDefinitionContinuation,
    definition_successor_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_owner_dependency_definition import confirmed


def test_new_prompt_actual_provider_to_original_definition_owner(tmp_path):
    owner, value, _, _, _ = confirmed(tmp_path)
    seed = fixture(tmp_path)
    policy = json.loads(Path("configs/research/multimodal_structural_context_v2.json").read_bytes())
    parsed = read_expansion_policy(policy["definition_expansion"])
    assert parsed.prompt_version == "definition-draft-6.3.0"

    def response(body):
        prompt = body["messages"][0]["content"]
        assert "consecutive" in prompt and "dependencies" in prompt
        return dict(
            schema_version="research-definition-proposal/5.0.0",
            definitions=[value],
            no_addition_reason=None,
        )

    provider = DeterministicFakeProvider(
        definition_config(seed.provider.config, prompt_version=parsed.prompt_version), [response]
    )
    result = provider.generate_definitions(
        payload={"public_material": "synthetic"},
        scope=seed.scope,
        budget=seed.budget,
        before_request=lambda: None,
    )
    assert result.schema_version == "research-definition-proposal/5.0.0"
    candidate = result.definitions[0].model_dump(mode="json")
    receipt = owner.accept("encoded", candidate)
    assert owner.read("encoded") == receipt
    assert seed.budget.used == 1


@pytest.mark.parametrize("mutation", ["aliases", "duplicate_index"])
def test_original_bad_dependency_shape_remains_rejected(tmp_path, mutation):
    from packages.evidence_reuse.definitions import OwnerBoundDefinitionProposal

    _, value, _, _, _ = confirmed(tmp_path)
    if mutation == "aliases":
        value["dependencies"] = ["requirement:0"] * len(value["dependencies"])
    else:
        value["dependency_contracts"][1]["index"] = 0
    reason = (
        "definition_dependency_binding"
        if mutation == "aliases"
        else "definition_dependency_coverage"
    )
    with pytest.raises(ValueError, match=reason):
        OwnerBoundDefinitionProposal.model_validate_json(
            json.dumps(
                dict(
                    schema_version="research-definition-proposal/5.0.0",
                    definitions=[value],
                    no_addition_reason=None,
                )
            )
        )


def parent_fixture(tmp_path, *, defer=False):
    parent = tmp_path / "original"
    parent.mkdir()
    (parent / "lock").touch()
    intent = dict(checksum=digest("task"))
    profile_path = (
        "configs/research/multimodal_structural_context_v2.json"
        if defer
        else "configs/research/multimodal_structural_context_v1.json"
    )
    spec = dict(
        schema_version="multimodal-live-entry/1.21.0" if defer else "multimodal-live-entry/1.20.0",
        input_checksum=digest("input"),
        intent=intent,
        implementation={"synthetic": "freeze"},
        provider={},
        component_consumer={"proof": True},
        active_acquisition=dict(
            path=profile_path, checksum=digest(json.loads(Path(profile_path).read_bytes()))
        ),
    )
    protocol = dict(
        input=spec["input_checksum"],
        task=intent["checksum"],
        implementation=spec["implementation"],
        deadline="2026-10-17T10:31:59+00:00",
        targets=["one"],
        previous_attempts=[],
    )
    rid = identity(
        "RUN", "task-research:mm-live:" + digest((parent.name, "definition-expansion"))[7:55]
    )
    failure = dict(
        stage="definitions",
        error_code="timeout" if defer else "schema",
        failure_kind="dispatched_unknown" if defer else "confirmed_response_or_domain_refusal",
        protocol_checksum=digest(protocol),
        input_checksum=spec["input_checksum"],
        campaign_id=parent.name,
        run_id=rid,
        deadline=protocol["deadline"],
        previous_attempts=[],
        definition_revision_saved=False,
        confirmed_proposal_saved=False,
        new_requests=1,
        operations=[dict(kind="llm")],
        transport=dict(
            remote_response_unknown=1 if defer else 0, dispatch_unknown=0, dispatched_observed=1
        ),
    )
    for name, data in (("entry", spec), ("protocol", protocol), ("stage-failure", failure)):
        (parent / (name + ".json")).write_text(json.dumps(data))
    raw = "" if defer else '{"invalid":"definition shape"}'
    record = dict(
        role="source",
        error="timeout" if defer else "schema",
        simulated=False,
        request_id="r1",
        api_requests=1,
        output_checksum=None if defer else digest(raw.encode().hex()),
    )
    records = {
        "multimodal-stage-failure": (failure,),
        "journal:0": (dict(phase="reserved"),),
        "journal:1": (dict(phase="completed", record=record),),
        "definition:response:r1": (dict(record=record, raw_text=raw),),
    }
    writes = []

    def write(key, value):
        assert key not in records
        records[key] = value
        writes.append(key)

    docs = SimpleNamespace(run_id=rid, read=records.get, session=nullcontext, write=write)

    def no_run(rid):
        raise ValueError("research_context_required")

    campaign = SimpleNamespace(
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=spec["input_checksum"]),
            task=SimpleNamespace(model_dump=lambda **kw: intent),
            component_consumer_selection=spec["component_consumer"],
        ),
        config=SimpleNamespace(model_dump=lambda **kw: {}),
        documents=lambda rid: docs,
        sessions=SimpleNamespace(read=lambda rid: None),
        contexts=SimpleNamespace(read_research_run=no_run),
        output=tmp_path / "successor",
    )
    campaign.output.mkdir()
    reference = dict(
        parent=str(parent), entry=digest(spec), protocol=digest(protocol), failure=digest(failure)
    )
    if defer:
        reference["mode"] = "technical_backlog_no_source_replay"
    return campaign, reference, spec, protocol, failure, records, writes


def test_explicit_unknown_deferral_no_source_replay_and_original_revision_owner(tmp_path):
    from packages.multimodal_research.definition_expansion import DefinitionExpansion

    c, ref, spec, _, _, _, writes = parent_fixture(tmp_path, defer=True)
    profile = json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())
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
    assert revision["rejected"][0]["reason"] == "source_timeout_unknown_not_replayed"
    assert not revision["rejected"][0]["medical_no_addition_claimed"]
    assert revision["added"] == [] and seed.budget.used == 0
    assert e.advance() == revision and e.read_revision() == revision
    assert len(writes) == 1 and continuation.check_attempts()[0]["requests"] == 1
    assert (
        ConfirmedDefinitionContinuation(c, ref, read_only=True).check_attempts()
        == continuation.attempts
    )


def test_unknown_cannot_use_format_retry_or_discard_uncommitted_increment(tmp_path):
    c, ref, _, _, _, records, writes = parent_fixture(tmp_path, defer=True)
    wrong = {k: v for k, v in ref.items() if k != "mode"}
    with pytest.raises(ValueError, match="parent_binding"):
        ConfirmedDefinitionContinuation(c, wrong)
    records["definition:gap:0:decision"] = (dict(candidate=dict(added=["new"], replaced=[])),)
    with pytest.raises(ValueError, match="uncommitted_increment"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


def test_claim_idempotence_read_only_restore_budget_and_original_clock(tmp_path):
    c, ref, spec, protocol, failure, records, writes = parent_fixture(tmp_path)
    before = copy.deepcopy((spec, protocol, failure))
    continuation = ConfirmedDefinitionContinuation(c, ref)
    assert continuation.check_attempts()[0]["requests"] == 1
    assert (
        ConfirmedDefinitionContinuation(c, ref, read_only=True).check_attempts()
        == continuation.attempts
    )
    assert len(writes) == 1 and (spec, protocol, failure) == before
    c.output = tmp_path / "other"
    with pytest.raises(ValueError, match="already_claimed"):
        ConfirmedDefinitionContinuation(c, ref)
    records["journal:2"] = (dict(phase="reserved"),)
    with pytest.raises(ValueError, match="history_changed"):
        continuation.check_attempts()


def test_read_only_does_not_create_claim(tmp_path):
    c, ref, _, _, _, _, writes = parent_fixture(tmp_path)
    with pytest.raises(ValueError, match="claim_missing"):
        ConfirmedDefinitionContinuation(c, ref, read_only=True)
    assert writes == []


@pytest.mark.parametrize(
    "mutation", ["unknown", "cancel", "wrong_run", "input", "revision", "owner", "acquisition"]
)
def test_invalid_parent_or_owner_cannot_claim(tmp_path, mutation):
    c, ref, spec, _, failure, records, writes = parent_fixture(tmp_path)
    parent = Path(ref["parent"])
    if mutation == "unknown":
        failure["transport"]["remote_response_unknown"] = 1
    elif mutation == "cancel":
        (parent / "cancel").touch()
    elif mutation == "wrong_run":
        failure["run_id"] = "other"
    elif mutation == "input":
        failure["input_checksum"] = digest("other")
    elif mutation == "revision":
        failure["definition_revision_saved"] = True
    elif mutation == "owner":
        records["multimodal-stage-failure"] = ({"different": True},)
    else:
        c.contexts.read_research_run = lambda rid: object()
    (parent / "stage-failure.json").write_text(json.dumps(failure))
    ref["failure"] = digest(failure)
    with pytest.raises(ValueError):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


def test_only_explicit_prompt_delta_no_budget_or_check_changes(tmp_path):
    _, _, spec, _, _, _, _ = parent_fixture(tmp_path)
    profile = json.loads(
        Path("configs/research/multimodal_structural_context_v2.json").read_bytes()
    )
    definition_successor_policy(spec, profile)
    profile["checks"].pop()
    with pytest.raises(ValueError, match="plan_delta"):
        definition_successor_policy(spec, profile)

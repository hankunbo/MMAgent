"""Owner evidence proves availability, never clinical reliability or method approval."""

import copy
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_acceptance import requirement_rows
from packages.grounding.models import digest
from packages.multimodal_research.campaign import InputEvidence, MultimodalCampaign
from packages.object_store import checksum
from tests.evaluation_preparation.test_definition_dependency_scope import candidate, fixture


def setup():
    owner, manifest, sources, docs = fixture()
    raw = b"synthetic image bytes; no patient data"
    manifest.payloads = (SimpleNamespace(id="p", checksum=checksum(raw), byte_size=len(raw)),)
    base = SimpleNamespace(
        read_dependency_input=lambda: manifest,
        prepared=SimpleNamespace(payload_bytes=(("p", raw),)),
    )
    owner.fact_reader = lambda contract: InputEvidence.read_dependency_fact(base, contract)
    value = candidate(blocks=("acquisition", "interpretation"))
    value["schema_version"] = "source-definition-draft/4.0.0"
    value["dependencies"] = ["Input image is present; no reliability claim"]
    value["dependency_contracts"][0].update(
        text=value["dependencies"][0],
        modality="new_image",
        concept="texture",
        predicate="input_available",
        method=None,
        unit=None,
    )
    return owner, manifest, sources, docs, base, value


def test_new_dependency_owner_projection_and_scheduler_positive():
    owner, manifest, _, _, _, value = setup()
    saved = owner.accept("available", value)
    assert saved["schema_version"] == "accepted-research-definition/3.0.0"
    assessment = saved["dependency_assessment"]
    assert assessment["items"][0]["satisfied"]
    assert assessment["review_allowed"] and assessment["acquisition_allowed"]
    rows = requirement_rows(saved, manifest, fact_reader=owner.fact_reader)
    assert MultimodalCampaign.select_targets(SimpleNamespace(ledger=rows), {"image-a"})
    assert owner.read("available") == saved


def test_new_contract_without_original_fact_owner_refuses():
    owner, _, _, docs, _, value = setup()
    owner.fact_reader = None
    with pytest.raises(ValueError, match="definition_dependency_fact_owner_missing"):
        owner.accept("no-owner", value)
    assert docs.read("definition:no-owner") is None


def test_payload_drift_is_technical_refusal_not_missing_data():
    owner, _, _, _, base, value = setup()
    saved = owner.accept("before", value)
    base.prepared.payload_bytes = (("p", b"changed"),)
    with pytest.raises(ValueError, match="multimodal_dependency_payload_binding"):
        owner.read("before")
    assert owner.read("before", historical=True) == saved


@pytest.mark.parametrize("kind", ["method", "quality"])
def test_unverified_qualification_is_not_resolved_by_presence(kind):
    owner, manifest, _, _, _, value = setup()
    value["dependency_contracts"][0].update(
        kind=kind,
        predicate="unresolved",
        modality=None,
        concept=None,
    )
    receipt = owner.accept("blocked", value)
    assert not receipt["dependency_assessment"]["review_allowed"]
    assert not requirement_rows(receipt, manifest, fact_reader=owner.fact_reader)[0][
        "review_allowed"
    ]


@pytest.mark.parametrize("mutation", ["quality_presence", "method_presence", "caller_flag"])
def test_unqualified_predicate_cannot_be_promoted(mutation):
    owner, _, _, docs, _, value = setup()
    c = value["dependency_contracts"][0]
    if mutation == "caller_flag":
        c["satisfied"] = True
    else:
        c["kind"] = mutation.split("_")[0]
    with pytest.raises(ValueError):
        owner.accept("invalid", value)
    assert docs.read("definition:invalid") is None


@pytest.mark.parametrize("number", [0.0, -1.5, 4.5])
def test_original_numeric_zero_negative_and_unit_binding(number):
    from packages.dataset_adapters.multimodal.models import Component

    owner, manifest, _, _, _, value = setup()
    scalar = Component(
        id="metric",
        kind="scalar",
        concept="metric",
        modality="measurement",
        status="obtained",
        value=number,
        unit="unit-x",
        unit_basis="synthetic source",
        source_locator="synthetic",
    )
    manifest.components += (scalar,)
    value["dependency_contracts"][0].update(
        predicate="source_numeric_available",
        concept="metric",
        modality="measurement",
        method="source_numeric_read",
        unit="unit-x",
    )
    receipt = owner.accept("number", value)
    assert receipt["dependency_assessment"]["items"][0]["satisfied"]
    proof = receipt["dependency_assessment"]["items"][0]["evidence"]
    assert proof["numeric"]["value"] == number
    value["dependency_contracts"][0]["unit"] = "other-unit"
    mismatch = owner.accept("mismatch", value)
    assert not mismatch["dependency_assessment"]["items"][0]["satisfied"]
    assert mismatch["dependency_assessment"]["items"][0]["status"] == "numeric_method_unit_mismatch"


def test_missing_ambiguous_and_historical_receipt_are_not_rewritten():
    owner, manifest, _, _, _, value = setup()
    before = owner.accept("old", value)
    manifest.components = ()
    after = owner.accept("new", value)
    assert not after["dependency_assessment"]["review_allowed"]
    assert owner.read("old", historical=True) == before
    with pytest.raises(ValueError, match="definition_current_binding"):
        owner.read("old")
    owner, manifest, _, _, _, value = setup()
    manifest.components += (manifest.components[0].model_copy(update={"id": "duplicate"}),)
    ambiguous = owner.accept("ambiguous", value)
    assert ambiguous["dependency_assessment"]["items"][0]["status"] == "dependency_input_ambiguous"


def test_projection_cannot_trust_saved_satisfied_flag():
    owner, manifest, _, _, _, value = setup()
    receipt = owner.accept("valid", value)
    changed = copy.deepcopy(receipt)
    changed["dependency_assessment"]["items"][0]["satisfied"] = False
    with pytest.raises(ValueError, match="definition_dependency_projection_binding"):
        requirement_rows(changed, manifest, fact_reader=owner.fact_reader)
    with pytest.raises(ValueError, match="definition_dependency_fact_owner_missing"):
        requirement_rows(receipt, manifest)


def test_production_provider_revision_current_and_gate(tmp_path):
    import json
    from pathlib import Path

    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.campaign_renewal import gate_selection
    from packages.multimodal_research.definition_expansion import (
        definition_config,
        read_expansion_policy,
    )
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    owner, manifest, _, _, _, value = setup()
    expansion = expansion_fixture(tmp_path)
    expansion.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v10.json").read_bytes())
    )
    expansion.binding.update(
        policy_checksum=digest(expansion.policy),
        target=expansion.policy.public_target,
        input_checksum=manifest.checksum,
    )
    expansion.owner.binding.update(
        target=expansion.policy.public_target, input_checksum=manifest.checksum
    )
    expansion.campaign.base.manifest = manifest
    expansion.campaign.base.read_dependency_input = lambda: manifest
    expansion.campaign.base.read_dependency_fact = owner.fact_reader
    value.update(
        schema_version="source-definition-draft/4.1.0", target=expansion.policy.public_target
    )
    for excerpt in value["excerpts"]:
        excerpt.pop("start")
        excerpt.pop("end")

    def response(body):
        assert body["max_completion_tokens"] == 8192
        assert "source-definition-draft/4.1.0" in body["messages"][0]["content"]
        assert "source_numeric_available" in body["messages"][0]["content"]
        return dict(
            schema_version="research-definition-proposal/3.0.0",
            definitions=[value],
            no_addition_reason=None,
        )

    expansion.provider = DeterministicFakeProvider(
        definition_config(
            expansion.provider.config,
            prompt_version=expansion.policy.prompt_version,
            source_output_tokens=expansion.policy.source_output_tokens,
        ),
        [response],
    )
    revision = expansion.advance()
    assert revision["added"] and expansion.campaign.targets == tuple(revision["added"])
    assert gate_selection(expansion.campaign, revision) == tuple(revision["added"])
    assert expansion.current() == revision
    assert expansion.advance() == revision
    # An old saved assessment cannot be rewritten in place after an input changes.
    manifest.components = ()
    with pytest.raises(ValueError, match="definition_current_binding"):
        expansion.current()


def test_trial_config_dispatch_and_legacy_rejection(monkeypatch):
    import json
    from pathlib import Path

    from packages.multimodal_research import definition_trial as trial
    from packages.multimodal_research.definition_expansion import read_expansion_policy

    policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v10.json").read_bytes())
    )
    monkeypatch.setattr(trial, "freeze", lambda: {"synthetic": "test"})
    spec = dict(
        implementation=trial.freeze(),
        schema_version="multimodal-live-entry/1.13.0",
        intent=dict(target_code=policy.public_target),
    )
    envelope = trial.EvidenceTrial(
        entry_checksum=digest(spec),
        authorization="offline test only",
        expansion_policy=policy.model_dump(mode="json"),
    ).model_dump(mode="json")
    assert trial.checked(envelope, spec) == envelope
    envelope["schema_version"] = "multimodal-definition-trial/1.4.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        trial.checked(envelope, spec)

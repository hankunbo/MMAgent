"""Phase-specific dependencies never turn input availability into medical authority."""

import copy
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner, requirement_rows
from packages.grounding.models import digest
from packages.multimodal_research.campaign import MultimodalCampaign
from tests.agent_provider.support import Documents
from tests.evaluation_preparation.test_definition_expansion import Sources, draft


def candidate(*, blocks=("interpretation",)):
    value = draft("GOAL:DEPENDENCY_TEST")
    value["schema_version"] = "source-definition-draft/3.0.0"
    value["dependencies"] = ["Independent functional input"]
    value["dependency_contracts"] = [
        dict(
            index=0,
            text=value["dependencies"][0],
            kind="patient_input",
            blocks=list(blocks),
            modality="functional",
            concept="function",
            rationale="This relation requires functional input for direction, not visible texture.",
            excerpt_indices=[0],
        )
    ]
    return value


def fixture():
    from packages.dataset_adapters.multimodal.models import Component

    image = Component(
        id="image-a",
        kind="image",
        concept="texture",
        modality="new_image",
        status="obtained",
        payload_ids=("p",),
        source_locator="synthetic",
        shape=(8, 8),
    )
    manifest = SimpleNamespace(checksum=digest("dependency-input"), components=(image,))
    sources, documents = Sources(), Documents()
    owner = ResearchDefinitionOwner(
        documents=documents,
        sources=sources,
        campaign_id="synthetic-dependency",
        target="GOAL:DEPENDENCY_TEST",
        input_checksum=manifest.checksum,
        guard=lambda: None,
        input_reader=lambda: manifest,
    )
    return owner, manifest, sources, documents


def test_acquisition_is_distinct_from_review_and_old_draft_stays_blocked():
    owner, manifest, _, _ = fixture()
    value = owner.accept("new", candidate())
    assert value["schema_version"] == "accepted-research-definition/2.0.0"
    assert value["dependency_assessment"]["acquisition_allowed"]
    assert not value["dependency_assessment"]["review_allowed"]
    assert value["definition"]["interpretation_status"] == "missing"
    rows = requirement_rows(value, manifest)
    assert rows[0]["review_allowed"] is False
    assert rows[0]["acquisition_allowed"] is True
    base = SimpleNamespace(ledger=rows, observations={})
    assert MultimodalCampaign.select_targets(base, {"image-a"}) == (rows[0]["id"],)
    old = draft("GOAL:DEPENDENCY_TEST")
    old["dependencies"] = ["Independent functional input"]
    receipt = owner.accept("old", old)
    with pytest.raises(ValueError, match="definition_dependency_unresolved"):
        requirement_rows(receipt, manifest)
    assert owner.read("new") == value


def test_acquisition_dependency_is_not_scheduled_and_missing_input_is_retained():
    owner, manifest, _, _ = fixture()
    value = owner.accept("blocked", candidate(blocks=("acquisition", "interpretation")))
    rows = requirement_rows(value, manifest)
    assert rows[0]["missing_reason"] == "definition_acquisition_dependency_unresolved"
    assert not MultimodalCampaign.select_targets(SimpleNamespace(ledger=rows), {"image-a"})
    assert value["dependency_assessment"]["items"][0]["status"] == "patient_input_missing"


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "text", "excerpt", "satisfied"])
def test_dependency_declarations_are_complete_exact_and_not_caller_authority(mutation):
    owner, _, _, documents = fixture()
    value = candidate()
    if mutation == "omit":
        value["dependency_contracts"] = []
    elif mutation == "duplicate":
        value["dependency_contracts"] *= 2
    elif mutation == "text":
        value["dependency_contracts"][0]["text"] = "different"
    elif mutation == "excerpt":
        value["dependency_contracts"][0]["excerpt_indices"] = [1]
    else:
        value["dependency_contracts"][0]["satisfied"] = True
    with pytest.raises(ValueError):
        owner.accept("invalid", value)
    assert documents.read("definition:invalid") is None


def test_input_presence_is_not_quality_or_interpretation_authority():
    owner, manifest, sources, _ = fixture()
    value = candidate()
    value["dependency_contracts"][0].update(modality="new_image", concept="texture")
    saved = owner.accept("present", value)
    assert (
        saved["dependency_assessment"]["items"][0]["status"] == "patient_input_present_unverified"
    )
    assert not saved["dependency_assessment"]["review_allowed"]
    before = copy.deepcopy(saved)
    manifest.components = ()
    with pytest.raises(ValueError, match="definition_current_binding"):
        owner.read("present")
    assert owner.read("present", historical=True) == before
    sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        owner.read("present")


def test_new_contract_requires_installed_input_owner():
    owner, _, _, _ = fixture()
    owner.input_reader = None
    with pytest.raises(ValueError, match="definition_input_owner_missing"):
        owner.accept("absent", candidate())


@pytest.mark.parametrize(
    "tamper", ["permission", "delete_scope", "rename", "method", "drop", "replacement", "quality"]
)
def test_production_policy_prompt_provider_scheduler_and_current_gate(tmp_path, tamper):
    import json
    from pathlib import Path

    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.campaign_renewal import gate_selection
    from packages.multimodal_research.definition_expansion import (
        definition_config,
        read_expansion_policy,
    )
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    e = expansion_fixture(tmp_path)
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v8.json").read_bytes())
    )
    e.binding["policy_checksum"] = digest(e.policy)
    # This fixture's graph scope is synthetic; the public proposition is not a label.
    e.binding["target"] = e.policy.public_target
    e.owner.binding["target"] = e.policy.public_target
    owner, manifest, _, _ = fixture()
    e.campaign.base.manifest = manifest
    e.campaign.base.read_dependency_input = lambda: manifest
    e.binding["input_checksum"] = manifest.checksum
    e.owner.binding["input_checksum"] = manifest.checksum
    value = candidate()
    value.update(schema_version="source-definition-draft/3.1.0", target=e.policy.public_target)
    for excerpt in value["excerpts"]:
        excerpt.pop("start")
        excerpt.pop("end")

    def response(body):
        assert "dependency_contracts" in body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        assert payload["input_capabilities"][0]["concept"] == "texture"
        return dict(
            schema_version="research-definition-proposal/2.0.0",
            definitions=[value],
            no_addition_reason=None,
        )

    e.provider = DeterministicFakeProvider(
        definition_config(e.provider.config, prompt_version="definition-draft-3.0.0"), [response]
    )
    revision = e.advance()
    assert revision["added"] and e.campaign.targets == tuple(revision["added"])
    assert gate_selection(e.campaign, revision) == ()
    assert e.campaign.targets == tuple(revision["added"])
    assert e.advance() == revision and e.budget.used == 1
    saved = e.documents.saved["definition:revision"][0]
    row = saved["rows"][0]
    if tamper == "permission":
        row["review_allowed"] = True
    elif tamper in {"delete_scope", "rename"}:
        for k in ("dependency_assessment", "acquisition_allowed", "review_allowed"):
            row.pop(k)
        if tamper == "rename":
            row["id"] = "arbitrary:new"
            saved["added"] = [row["id"]]
    elif tamper == "method":
        row["expected_method"] = "unapproved_method"
    elif tamper == "drop":
        saved["rows"] = []
    elif tamper == "replacement":
        saved["replaced"] = [dict(id="new_alias", definition=next(iter(saved["accepted"])))]
    else:
        row["quality"] = 1.0
    with pytest.raises(ValueError, match="definition_dependency_projection_binding"):
        e.current()


def test_first_acceptance_binds_the_original_visible_manifest():
    from packages.multimodal_research.campaign import InputEvidence
    from packages.object_store import checksum

    owner, manifest, _, documents = fixture()
    visible = SimpleNamespace(components=manifest)
    data = b"synthetic view"
    original = {"source": "synthetic"}
    base = SimpleNamespace(
        inputs=SimpleNamespace(read_visible=lambda case: visible),
        binding={"case_code": "CASE:SYNTHETIC"},
        visible_value=visible,
        manifest=manifest,
        prepared=SimpleNamespace(manifest=manifest),
        sources=SimpleNamespace(read=lambda: original, initial=original),
        views_path=SimpleNamespace(read_bytes=lambda: data),
        view_checksum=checksum(data),
    )
    owner.input_reader = lambda: InputEvidence.read_dependency_input(base)
    assert (
        owner.accept("valid-input", candidate())["dependency_assessment"]["items"][0]["status"]
        == "patient_input_missing"
    )
    base.manifest = SimpleNamespace(checksum=manifest.checksum, components=())
    with pytest.raises(ValueError, match="multimodal_dependency_manifest_binding"):
        owner.accept("wrong-input", candidate())
    assert documents.read("definition:wrong-input") is None


@pytest.mark.parametrize("kind", ["method", "quality"])
def test_method_and_quality_are_not_resolved_by_input_presence(kind):
    owner, _, _, _ = fixture()
    value = candidate()
    value["dependency_contracts"][0].update(kind=kind, concept=None, modality=None)
    receipt = owner.accept(kind, value)
    assert receipt["dependency_assessment"]["acquisition_allowed"]
    assert not receipt["dependency_assessment"]["review_allowed"]
    assert not receipt["dependency_assessment"]["items"][0]["satisfied"]


def test_numeric_missing_reason_rebuild_uses_original_measurement_owner(tmp_path):
    import json
    from pathlib import Path

    from packages.dataset_adapters.multimodal.models import Component
    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.definition_expansion import (
        definition_config,
        read_expansion_policy,
    )
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    e = expansion_fixture(tmp_path)
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v8.json").read_bytes())
    )
    e.binding.update(policy_checksum=digest(e.policy), target=e.policy.public_target)
    e.owner.binding["target"] = e.policy.public_target
    _, manifest, _, _ = fixture()
    manifest.components = (
        Component(
            id="numeric",
            kind="scalar",
            concept="texture",
            modality="new_image",
            status="obtained",
            value=0,
            unit="mm",
            unit_basis="synthetic",
            source_locator="synthetic",
        ),
    )
    e.campaign.base.manifest = manifest
    e.campaign.base.read_dependency_input = lambda: manifest
    e.binding["input_checksum"] = e.owner.binding["input_checksum"] = manifest.checksum
    value = candidate()
    value.update(
        schema_version="source-definition-draft/3.1.0",
        target=e.policy.public_target,
        acquisition="source_measurement",
        method="source_numeric_read",
        unit="different_unit",
    )
    for excerpt in value["excerpts"]:
        excerpt.pop("start")
        excerpt.pop("end")
    e.provider = DeterministicFakeProvider(
        definition_config(e.provider.config, prompt_version="definition-draft-3.0.0"),
        [
            dict(
                schema_version="research-definition-proposal/2.0.0",
                definitions=[value],
                no_addition_reason=None,
            )
        ],
    )
    revision = e.advance()
    assert revision["added"]
    assert (
        revision["rows"][0]["missing_reason"]
        == "definition_measurement_method_unit_mismatch_or_missing"
    )
    assert e.current() == revision
    assert not e.campaign.base.observations
    e.documents.saved["definition:revision"][0]["rows"][0]["missing_reason"] = None
    with pytest.raises(ValueError, match="definition_dependency_projection_binding"):
        e.current()

"""CPU contracts for numeric method reuse; synthetic sources are not real D1."""

import copy
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.models import Component
from packages.grounding.models import digest, validate
from packages.llm_provider.models import TaskDefinition
from packages.multimodal_research.campaign import InputEvidence
from packages.multimodal_research.confirmed_numeric_observations import ConfirmedNumericObservations
from packages.multimodal_research.material import numeric, observation
from tests.evaluation_preparation.test_confirmed_interpretation_definition import candidate
from tests.evaluation_preparation.test_definition_dependency_scope import fixture
from tests.evaluation_preparation.test_definition_expansion import draft


def setup(value=-0.37, *, target="GOAL:DEPENDENCY_TEST"):
    owner, _, sources, documents = fixture()
    owner.binding["target"] = target
    old = owner.accept("previous", draft(owner.binding["target"]))
    c = Component(
        id="numeric-md",
        kind="scalar",
        concept="md",
        modality="new_image",
        status="obtained",
        value=value,
        unit="dB",
        unit_basis="synthetic unit statement",
        source_locator="synthetic",
    )
    manifest = SimpleNamespace(
        checksum=owner.binding["input_checksum"],
        components=(c,),
        payloads=(),
        adapter_version="multimodal-adapters/1.1.0",
    )
    base = object.__new__(InputEvidence)
    base.manifest = manifest
    base.task = SimpleNamespace(target_code=owner.binding["target"], checksum=digest("task"))
    base.prepared = SimpleNamespace(payload_bytes=())
    base.visible_value = SimpleNamespace(checksum=digest("visible"))
    base.read_dependency_input = lambda: manifest
    row = dict(
        id="check:numeric-md",
        component=c.id,
        acquisition_method="source_measurement",
        input_modalities=["new_image"],
        definition=validate(TaskDefinition, old["definition"]),
    )
    base.ledger = [row]
    base.observations = {row["id"]: observation(row["id"], numeric(c, {}), unit="dB")}
    numeric_owner = ConfirmedNumericObservations(base)
    owner.confirmed_observations = numeric_owner
    owner.requirement_ids = (row["id"],)
    owner.input_reader = base.read_dependency_input
    owner.fact_reader = lambda _: (_ for _ in ()).throw(AssertionError("no extra dependency"))
    value = candidate()
    value.update(target=target, method="source_measurement_projection", unit="dB")
    return owner, numeric_owner, value, documents, sources


@pytest.mark.parametrize("number", [0, -0.37, 5])
def test_same_id_original_method_zero_negative_and_positive_remain_exact(number):
    owner, numeric_owner, value, documents, _ = setup(number)
    before = copy.deepcopy(numeric_owner.base.observations)
    receipt = owner.accept("new", value)
    assert receipt["confirmed_basis"]["method"] == "source_measurement_projection"
    assert (
        receipt["confirmed_basis"]["origin"]["qualification"]
        == "original_source_projection_not_clinical_accuracy"
    )
    assert owner.read("new") == receipt
    assert numeric_owner.base.observations == before
    assert documents.read("definition:previous") is not None


def test_current_input_change_rejects_but_history_remains():
    owner, numeric_owner, value, _, _ = setup()
    receipt = owner.accept("new", value)
    numeric_owner.base.manifest.checksum = digest("other input")
    with pytest.raises(ValueError, match="definition_confirmed_requirement_scope"):
        owner.read("new")
    assert owner.read("new", historical=True) == receipt


@pytest.mark.parametrize(
    "field,value", [("method", "source_numeric_read"), ("unit", "mm"), ("examination", "other")]
)
def test_method_unit_examination_mismatch_no_partial_write(field, value):
    owner, _, candidate_, documents, _ = setup()
    candidate_[field] = value
    with pytest.raises(ValueError, match="definition_confirmed_requirement_method"):
        owner.accept("bad", candidate_)
    assert documents.read("definition:bad") is None


def test_projection_install_and_repeat_keep_observation_and_no_new_write():
    owner, numeric_owner, value, documents, _ = setup()
    before = copy.deepcopy(numeric_owner.base.observations)
    owner.accept("new", value)
    adoption = numeric_owner.prepare_interpretation(owner, "new")
    rows = copy.deepcopy(numeric_owner.base.ledger)
    rows[0].update(adoption["changes"])
    numeric_owner.install_interpretation(owner, adoption, rows)
    numeric_owner.install_interpretation(owner, copy.deepcopy(adoption), copy.deepcopy(rows))
    assert owner.read("new")["confirmed_basis"]["observation"] == before[
        rows[0]["id"]
    ].reference.model_dump(mode="json")
    assert numeric_owner.base.observations == before
    adoption["changes"]["review_allowed"] = "tampered"
    numeric_owner.check_projection(rows[0]["id"])
    assert documents.read("definition:new") is not None


def test_projection_other_field_change_rejected_and_rolled_back():
    owner, numeric_owner, value, _, _ = setup()
    owner.accept("new", value)
    adoption = numeric_owner.prepare_interpretation(owner, "new")
    before = copy.deepcopy(numeric_owner.base.ledger)
    rows = copy.deepcopy(before)
    rows[0].update(adoption["changes"], acquisition_method="vision")
    with pytest.raises(ValueError, match="numeric_interpretation_projection_changed"):
        numeric_owner.install_interpretation(owner, adoption, rows)
    assert numeric_owner.base.ledger == before
    assert numeric_owner.adoptions == {}


def test_original_source_change_rejects_current_receipt():
    owner, _, value, _, sources = setup()
    old = owner.accept("new", value)
    sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        owner.read("new")
    assert owner.read("new", historical=True) == old


def test_numeric_capabilities_are_explicit_and_do_not_advertise_mask_qc():
    from packages.multimodal_research.definition_context import (
        candidate_contract_choices,
        owner_capabilities,
        requirement_context,
    )

    owner, numeric_owner, _, _, _ = setup()
    b = numeric_owner.base
    payload = dict(existing_requirement_context=requirement_context(b.ledger, [], b.observations))
    assert (
        owner_capabilities(copy.deepcopy(payload), b, b.ledger)[
            "confirmed_observation_capabilities"
        ]
        == []
    )
    result = owner_capabilities(payload, b, b.ledger, confirmed_owner=numeric_owner)
    item = result["confirmed_observation_capabilities"][0]
    assert item["resolvable_conditions"] == ["confirmed_method"]
    assert result["dependency_capability_version"] == "definition-owner-capabilities/1.1.0"
    choice = candidate_contract_choices(result)["candidate_contract_choices"][
        "confirmed_observation"
    ][0]
    assert choice["copy_fields"]["method"] == "source_measurement_projection"
    assert numeric_owner.dependency_basis(b.ledger[0]["id"])["technical_checks"] == {}


def test_expansion_installs_numeric_owner_only_for_explicit_new_policy(tmp_path):
    import json
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import (
        DefinitionExpansion,
        NumericInterpretationPublicPolicy,
        read_expansion_policy,
    )
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    owner, numeric_owner, value, _, _ = setup(target="GOAL:ADVANCED_GLAUCOMA")
    original = expansion_fixture(tmp_path)
    original.campaign.base = numeric_owner.base
    original.campaign.base.views = {"views": []}
    original.campaign.base.definition_revision = None
    policy = json.loads(
        Path("configs/research/multimodal_definition_raw_revision_v1.json").read_bytes()
    )
    policy.update(
        schema_version="campaign-definition-expansion/1.36.0",
        prompt_version="definition-draft-6.7.0",
        public_target=owner.binding["target"],
        unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0",
    )
    assert isinstance(read_expansion_policy(policy), NumericInterpretationPublicPolicy)
    exp = DefinitionExpansion(
        original.campaign,
        documents=original.documents,
        sources=owner.sources,
        provider=original.provider,
        budget=original.budget,
        scope=original.scope,
        policy=policy,
    )
    assert type(exp.owner.confirmed_observations) is ConfirmedNumericObservations
    receipt = exp.owner.accept("numeric-interpretation", value)
    expected = exp.owner.project_confirmed(receipt)
    row = copy.deepcopy(original.campaign.base.ledger[0])
    fields = (
        "definition",
        "basis_kind",
        "definition_receipt_checksum",
        "dependency_assessment",
        "acquisition_allowed",
        "review_allowed",
        "expected_method",
        "expected_unit",
    )
    row.update({k: expected[k] for k in fields}, definition_receipt_key="numeric-interpretation")
    revision = {
        "rows": [row],
        "replaced": [{"id": row["id"], "definition": "numeric-interpretation"}],
        "accepted": {"numeric-interpretation": receipt},
        "added": [],
        "materials": receipt["sources"],
    }
    revision["review_source_scope"] = exp.source_scope_record(revision)
    before = copy.deepcopy(original.campaign.base.observations)
    exp.install(revision)
    assert original.campaign.base.observations == before and original.campaign.targets == ()
    assert original.campaign.base.ledger[0]["expected_method"] == "source_measurement_projection"
    assert original.campaign.base.ledger[0]["definition_receipt_key"] == "numeric-interpretation"
    assert exp.owner.read("numeric-interpretation") == receipt


@pytest.mark.parametrize(
    "condition,allowed",
    [("method", True), ("mask_measurement_available", False), ("artifact_integrity", False)],
)
def test_typed_source_contract_confirms_method_but_never_invents_mask_qc(condition, allowed):
    from tests.evaluation_preparation.test_owner_dependency_definition import contract

    owner, numeric_owner, value, _, _ = setup()
    value["schema_version"] = "source-definition-draft/7.1.0"
    dependency = contract(0, condition, method=value["method"], unit=value["unit"])
    value["dependencies"] = [dependency["text"]]
    value["dependency_contracts"] = [dependency]
    receipt = owner.accept("typed", value)
    assert receipt["schema_version"] == "accepted-research-definition/6.0.0"
    assert receipt["dependency_assessment"]["review_allowed"] is allowed
    assert receipt["dependency_assessment"]["items"][0]["satisfied"] is allowed
    assert owner.read("typed") == receipt
    assert numeric_owner.base.observations

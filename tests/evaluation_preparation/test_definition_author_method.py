"""Public author methods bind definitions without exporting patient measurements."""

import copy
import json

import pytest

from packages.dataset_adapters.multimodal import grape_semantics as author
from packages.dataset_adapters.multimodal.models import Component
from packages.grounding.models import canonical, digest
from packages.multimodal_research.confirmed_numeric_observations import ConfirmedNumericObservations
from packages.multimodal_research.definition_context import owner_capabilities, requirement_context
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    read_expansion_policy,
)
from packages.multimodal_research.material import numeric as measurement
from packages.multimodal_research.material import observation
from tests.evaluation_preparation.test_confirmed_numeric_definition import setup
from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture
from tests.evaluation_preparation.test_whole_definition_material import new_policy


def prepared(*, enabled=True):
    domain, old, candidate, documents, sources = setup(target="GOAL:ADVANCED_GLAUCOMA")
    base = old.base
    row = base.ledger[0]
    component = Component(
        id="private-component-identifier",
        kind="scalar",
        modality="structured_measurement",
        concept="cct",
        status="obtained",
        value=572.123456789,
        unit=None,
        unit_basis=None,
        source_locator="private-spreadsheet-location",
        semantic_limits=(
            "semantic_profile:grape-v1",
            "source_document_checksum:" + author.SOURCE_CHECKSUM,
        ),
    )
    base.manifest.components = (component,)
    base.manifest.dataset = "grape"
    base.manifest.adapter_version = author.ADAPTER
    row.update(component=component.id, input_modalities=[component.modality])
    base.observations = {
        row["id"]: observation(
            row["id"], measurement(component, {}), unit="source_units_unqualified"
        )
    }
    numeric = ConfirmedNumericObservations(
        base,
        **(
            dict(method_context_profile=author.METHOD_PROFILE, method_context_targets=(row["id"],))
            if enabled
            else {}
        ),
    )
    domain.confirmed_observations = numeric
    candidate.update(unit="source_units_unqualified", modalities=[component.modality])
    return domain, numeric, candidate, documents, sources


def policy(target):
    value = new_policy()
    value.update(
        schema_version="campaign-definition-expansion/1.40.0",
        confirmed_method_metadata_targets=[target],
    )
    value["query_requirements"] = {q: [target] for q in value["public_queries"]}
    return value


def test_new_version_scope_and_legacy_do_not_silently_gain_metadata():
    domain, old, _, _, _ = prepared(enabled=False)
    assert "author_method" not in old.interpretation_basis(old.base.ledger[0]["id"])["origin"]
    target = old.base.ledger[0]["id"]
    new = policy(target)
    parsed = read_expansion_policy(new)
    from packages.multimodal_research.campaign_renewal import (
        INDEX_ONLY_POLICIES,
        REVISED_ACQUISITION_POLICIES,
    )

    assert parsed.schema_version in INDEX_ONLY_POLICIES & REVISED_ACQUISITION_POLICIES
    for value in [
        dict(new, schema_version="campaign-definition-expansion/1.39.0"),
        dict(new, confirmed_method_metadata_targets=["not-requested"]),
        dict(new, confirmed_method_metadata_targets=[target, target]),
    ]:
        with pytest.raises(ValueError):
            read_expansion_policy(value)


def test_author_name_bound_to_same_measurement_accepts_and_reads_without_unit_upgrade():
    domain, numeric, candidate, documents, _ = prepared()
    before = copy.deepcopy(numeric.base.observations)
    receipt = domain.accept("author-method", candidate)
    basis = receipt["confirmed_basis"]
    assert (
        basis["origin"]["author_method"]["locator"]["section"] == "Central corneal thickness (CCT)"
    )
    assert basis["unit"] == "source_units_unqualified"
    assert "unit_bridge" not in basis["origin"]["author_method"]
    assert domain.read("author-method") == receipt
    assert numeric.base.observations == before
    assert documents.read("definition:previous") is not None


def test_public_source_payload_has_author_semantics_not_values_ids_or_patient_locator():
    _, numeric, _, _, _ = prepared()
    base = numeric.base
    payload = dict(
        existing_requirement_context=requirement_context(base.ledger, [], base.observations)
    )
    current = owner_capabilities(payload, base, base.ledger, confirmed_owner=numeric)
    raw = canonical(current).decode()
    assert "Central corneal thickness (CCT)" in raw
    for forbidden in [
        "572.123456789",
        "private-component-identifier",
        "private-spreadsheet-location",
    ]:
        assert forbidden not in raw
    assert current["dependency_capability_version"] == "definition-owner-capabilities/1.2.0"
    assert numeric.dependency_basis(base.ledger[0]["id"])["technical_checks"] == {}
    assert current["confirmed_observation_capabilities"][0]["resolvable_conditions"] == [
        "confirmed_method"
    ]
    from packages.llm_provider.config import prompt
    from packages.llm_provider.definition_passages import SelectedDefinitionProposal
    from packages.llm_provider.transport import wire_request
    from packages.multimodal_research.campaign_renewal import compact_configuration
    from packages.multimodal_research.definition_expansion import definition_config

    config = definition_config(compact_configuration(), prompt_version="definition-draft-6.8.0")
    role = next(r for r in config.roles if r.role == "source")
    body = wire_request(
        config.transport,
        role,
        prompt(config, "source")[1],
        current,
        SelectedDefinitionProposal.model_json_schema(),
        (),
        wire_policy=config.wire_policy,
        wire_mode=config.wire_mode,
    )
    wire = json.dumps(body)
    assert "author_method_metadata" in wire and "Central corneal thickness (CCT)" in wire
    assert "572.123456789" not in wire and "private-component-identifier" not in wire


def test_author_source_change_rejects_current_receipt_but_preserves_history(monkeypatch):
    domain, _, candidate, _, _ = prepared()
    receipt = domain.accept("author-method", candidate)
    monkeypatch.setattr(author, "SOURCE_CHECKSUM", digest("changed-public-author-document"))
    with pytest.raises(ValueError, match="original_scalar|source_binding"):
        domain.read("author-method")
    assert domain.read("author-method", historical=True) == receipt


@pytest.mark.parametrize("fault", ["dataset", "observation", "unknown_target", "profile"])
def test_metadata_never_substitutes_for_original_input_or_method(fault):
    _, numeric, _, _, _ = prepared()
    base = numeric.base
    target = base.ledger[0]["id"]
    kwargs = dict(method_context_profile=author.METHOD_PROFILE, method_context_targets=(target,))
    if fault == "dataset":
        base.manifest.dataset = "other"
    elif fault == "observation":
        base.observations[target] = observation(target, 1, unit="source_units_unqualified")
    elif fault == "unknown_target":
        kwargs["method_context_targets"] = ("absent",)
    else:
        kwargs["method_context_profile"] = "unknown"
    with pytest.raises(ValueError):
        ConfirmedNumericObservations(base, **kwargs)


def test_original_expansion_installs_exact_configured_owner(tmp_path):
    domain, numeric, candidate, _, _ = prepared()
    original = expansion_fixture(tmp_path)
    original.campaign.base = numeric.base
    original.campaign.base.views = {"views": []}
    original.campaign.base.definition_revision = None
    target = numeric.base.ledger[0]["id"]
    exp = DefinitionExpansion(
        original.campaign,
        documents=original.documents,
        sources=domain.sources,
        provider=original.provider,
        budget=original.budget,
        scope=original.scope,
        policy=policy(target),
    )
    assert type(exp.owner.confirmed_observations) is ConfirmedNumericObservations
    receipt = exp.owner.accept("current-author-method", candidate)
    assert exp.owner.read("current-author-method") == receipt
    assert receipt["confirmed_basis"]["origin"]["author_method"]["concept"] == "cct"
    assert original.budget.used == 0

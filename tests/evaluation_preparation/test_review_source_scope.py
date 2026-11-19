"""Versioned dependency projection; unused search history is not Review context."""

import copy
import json
from pathlib import Path

import pytest

from packages.grounding.models import digest, validate
from packages.llm_provider.models import Source
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    read_expansion_policy,
)
from packages.multimodal_research.material import requirement_source_refs, requirement_sources
from tests.evaluation_preparation.test_component_trial import envelope
from tests.evaluation_preparation.test_definition_expansion import draft, fixture, source
from tests.evaluation_preparation.test_directional_review import fixture as review_fixture


def policy():
    return read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v14.json").read_bytes())
    )


def test_new_trial_is_explicit_and_old_mixing_rejected(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial as trial

    value, spec = envelope(
        tmp_path, monkeypatch, "configs/research/multimodal_definition_gate_v14.json"
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.9.0"
    assert trial.read(tmp_path) == value
    value["schema_version"] = "multimodal-definition-trial/1.8.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        trial.checked(value, spec)


def test_public_configuration_is_in_implementation_freeze():
    from packages.multimodal_research.implementation import freeze
    from packages.object_store import checksum

    path = Path("configs/research/multimodal_definition_gate_v14.json")
    assert freeze()[str(path)] == checksum(path.read_bytes())


def test_new_schemas_are_exact():
    from packages.multimodal_research.definition_expansion import RequirementScopeExpansionPolicy
    from packages.multimodal_research.definition_trial import RequirementScopeTrial

    for model in (RequirementScopeTrial, RequirementScopeExpansionPolicy):
        path = Path("packages/llm_provider/schemas") / (model.__name__ + ".schema.json")
        assert json.loads(path.read_bytes()) == model.model_json_schema()


def test_exact_requirement_closure_keeps_missing_input_context_not_search_pool():
    state = review_fixture().owner.current()
    row = dict(
        id="missing-input",
        definition=state.policy.items[0].definition,
        missing_reason="patient_input_missing",
    )
    materials = {s.reference.code: s for s in state.material.sources}
    extra = validate(Source, source()["material"])
    materials[extra.reference.code] = extra
    selected = requirement_sources(materials, [row])
    assert set(selected) == {r.code for r in row["definition"].source_refs}
    assert extra.reference.code not in selected
    assert requirement_source_refs([row, row]) == requirement_source_refs([row])
    for code in selected:
        missing = dict(materials)
        missing.pop(code)
        with pytest.raises(ValueError, match="source_missing_or_changed"):
            requirement_sources(missing, [row])
        changed = dict(materials)
        changed[code] = changed[code].model_copy(
            update={
                "reference": changed[code].reference.model_copy(
                    update={"checksum": digest("other")}
                )
            }
        )
        with pytest.raises(ValueError, match="source_missing_or_changed"):
            requirement_sources(changed, [row])


def test_conflicting_versions_are_not_silently_deduplicated():
    state = review_fixture().owner.current()
    definition = state.policy.items[0].definition
    ref = definition.source_refs[0]
    changed = definition.model_copy(
        update={"source_refs": (ref.model_copy(update={"version": "99.0.0"}),)}
    )
    with pytest.raises(ValueError, match="source_version_conflict"):
        requirement_source_refs([dict(definition=definition), dict(definition=changed)])


def test_revision_new_scope_keeps_history_without_reading_unused_pool(tmp_path):
    e = fixture(tmp_path)
    e.policy = policy()  # Unit branch probe; public version binding tested separately.
    revision = dict(
        binding=e.binding,
        materials=[source()],
        accepted={},
        rows=[],
        added=[],
        replaced=[],
        denominator_before=0,
        denominator_after=0,
    )
    revision["review_source_scope"] = e.source_scope_record(revision)
    e.documents.write("definition:revision", (revision,))
    old = digest(revision)
    e.sources.read = lambda code: pytest.fail("unused pool is not a consumed dependency")
    assert e.read_revision() == revision
    e.install(revision)
    assert e.campaign.base.definition_materials == {}
    assert e.campaign.base.definition_revision["materials"] == [source()]
    assert digest(e.documents.read("definition:revision")[0]) == old
    revision["review_source_scope"]["retrieval_history_checksum"] = digest("tampered")
    e.documents.saved["definition:revision"] = (copy.deepcopy(revision),)
    with pytest.raises(ValueError, match="definition_review_source_scope_binding"):
        e.read_revision()
    revision["review_source_scope"] = e.source_scope_record(revision)
    e.documents.saved["definition:revision"] = (copy.deepcopy(revision),)
    # Old version still checks its declared full pool, no retroactive relaxation.
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v13.json").read_bytes())
    )
    called = []
    e.sources.read = lambda code: called.append(code) or source()
    assert e.read_revision() == revision
    assert called == [source()["code"]]


def test_retained_definition_cannot_hide_changed_source():
    state = review_fixture().owner.current()
    definition = state.policy.items[0].definition
    s = next(s for s in state.material.sources if s.reference == definition.source_refs[0])
    item = dict(
        code=s.reference.code, checksum=s.reference.checksum, material=s.model_dump(mode="json")
    )
    revision = dict(rows=[dict(definition=definition)], materials=[item, source()])
    e = object.__new__(DefinitionExpansion)
    e.policy = policy()
    assert e.required_materials(revision) == [item]
    changed = copy.deepcopy(revision)
    changed["materials"][0]["checksum"] = digest("changed")
    with pytest.raises(ValueError, match="revision_source_changed"):
        e.required_materials(changed)


@pytest.mark.parametrize("forged_published", [True, 0])
def test_unused_accepted_receipt_is_history_not_current_authority(tmp_path, forged_published):
    e = fixture(tmp_path)
    accepted = e.owner.accept("unused", draft(e.binding["target"]))
    e.policy = policy()
    revision = dict(
        binding=e.binding,
        materials=[source()],
        accepted={"unused": accepted},
        rows=[],
        added=[],
        replaced=[],
        denominator_before=0,
        denominator_after=0,
    )
    revision["review_source_scope"] = e.source_scope_record(revision)
    e.documents.write("definition:revision", (revision,))
    original = digest(e.documents.read("definition:unused"))

    def withdrawn(code):
        raise ValueError("definition_source_changed")

    e.sources.read = withdrawn
    assert e.read_revision() == revision
    assert digest(e.documents.read("definition:unused")) == original
    # Unused is not a renewed grant. Direct current use must still fail.
    with pytest.raises(ValueError, match="definition_source_changed"):
        e.owner.read_for_report("unused")
    # Even historical receipts must match the original immutable owner record.
    revision["accepted"]["unused"]["published"] = forged_published
    e.documents.saved["definition:revision"] = (copy.deepcopy(revision),)
    with pytest.raises(ValueError, match="definition_revision_source_changed"):
        e.read_revision()
    revision["accepted"]["unused"] = accepted
    e.documents.saved["definition:revision"] = (copy.deepcopy(revision),)
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v13.json").read_bytes())
    )
    with pytest.raises(ValueError, match="definition_source_changed"):
        e.read_revision()


@pytest.mark.parametrize("receipt_key", [None, "not-saved"])
def test_added_row_cannot_drop_or_invent_its_receipt(tmp_path, receipt_key):
    e = fixture(tmp_path)
    revision = copy.deepcopy(e.advance())
    e.policy = policy()
    e.campaign.base.read_dependency_input = lambda: e.campaign.base.manifest
    revision["rows"][0]["definition_receipt_key"] = receipt_key
    revision["review_source_scope"] = e.source_scope_record(revision)
    e.documents.saved["definition:revision"] = (revision,)
    with pytest.raises(ValueError, match="definition_dependency_projection_binding"):
        e.read_revision()

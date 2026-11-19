"""Method hints must match target/representation, not dataset or patient values."""

from copy import deepcopy
from types import SimpleNamespace as S

import pytest

from packages.memory.procedure_context import contract, mismatch
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.reuse import CAPABILITY_POLICY
from tests.evaluation_preparation.test_multimodal_reuse import fixture


def context(**changes):
    component = S(
        kind="map",
        modality="OCT_RNFLT",
        concept="rnflt",
        unit="um",
        unit_basis="source_measurement",
        axis_semantics="source_index_only",
    )
    task = S(target_code="GOAL:ADVANCED_GLAUCOMA", checksum="task-v1")
    value = dict(
        contract=contract(task, {"reference": {}, "interpretation": "sourced"}, component),
        patient_group=None,
        provenance={"parent_checksum": "parent"},
    )
    value.update(changes)
    return value


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "GOAL:EARLY_GLAUCOMA"),
        ("task_checksum", "task-v2"),
        ("definition_checksum", "different_relation"),
        ("pair_binding", "paired"),
    ],
)
def test_proposition_definition_and_pairing_are_not_interchangeable(field, value):
    old, current = context(), context()
    current["contract"][field] = value
    assert mismatch(old, current) == "memory_method_" + field + "_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("representation", "volume"),
        ("modality", "OCT"),
        ("concept", "gcc"),
        ("unit", "mm"),
        ("unit_basis", "different_measurement_definition"),
        ("unit_qualified", False),
        ("axis_semantics", "unknown"),
    ],
)
def test_map_volume_units_and_axes_are_precise_prerequisites(field, value):
    old, current = context(), context()
    current["contract"]["component"][field] = value
    assert mismatch(old, current) == "memory_method_component_mismatch"


def test_dataset_patient_values_and_quality_estimates_do_not_define_method():
    old, current = context(patient_group="old"), context(patient_group="new")
    for value, dataset in ((old, "gamma"), (current, "other")):
        value["dataset"] = dataset
    assert mismatch(old, current) is None
    assert (
        mismatch(old, dict(current, patient_group="old")) == "same_patient_not_external_experience"
    )
    assert mismatch(context(), context()) is None
    assert "dataset" not in old["contract"] and "patient_group" not in old["contract"]


def install(s, monkeypatch):
    s.capability_policy = CAPABILITY_POLICY
    proof = dict(contract=context()["contract"], context_checksum="proof", provenance={})
    monkeypatch.setattr(s, "method_context", lambda record, row: (deepcopy(proof), None))
    return proof


def test_new_plan_freezes_context_and_cold_restore_never_requeries(monkeypatch):
    first = fixture()
    proof = install(first, monkeypatch)
    plan = first.prepare()
    assert plan["schema_version"] == "multimodal-procedure-reuse/2.0.0"
    assert plan["plans"][0]["selected"]["method_context"] == proof
    second = fixture()
    install(second, monkeypatch)
    monkeypatch.setattr(second.reader, "procedure_keys", lambda **kw: pytest.fail("requery"))
    assert second.restore(plan) == plan and second.check() == ("saved",)
    monkeypatch.setattr(second, "method_context", lambda *args: (None, "changed"))
    assert second.check() == () and second.events[-1]["action"] == "current_owner_baseline"


def test_mismatch_has_reason_and_legacy_plan_cannot_claim_new_policy(monkeypatch):
    old = fixture().prepare()
    new = fixture()
    install(new, monkeypatch)
    with pytest.raises(ValueError, match="reuse_restore_binding"):
        new.restore(old)
    monkeypatch.setattr(new, "method_context", lambda *a: (None, "representation_mismatch"))
    plan = new.prepare()
    assert plan["plans"][0]["selected"] is None
    assert plan["plans"][0]["rejected"][0]["reason"] == "representation_mismatch"


def test_original_owners_required_and_new_profile_is_explicit():
    from packages.memory.procedure_context import read

    with pytest.raises(ValueError, match="original_owners_required"):
        read(S(), S(), inputs=S(), contexts=S())
    assert "procedure_reuse_policy" not in load(
        "configs/research/multimodal_catalog_acquisition_v4.json"
    )
    assert (
        load("configs/research/multimodal_catalog_acquisition_v5.json")["procedure_reuse_policy"]
        == CAPABILITY_POLICY
    )


def test_original_alias_roundtrip_preserves_definition_and_rejects_wrong_map():
    from packages.evidence_acquisition.source_aliases import alias_sources
    from packages.memory.procedure_context import original_definition
    from packages.multimodal_research.material import definition
    from tests.evaluation_preparation.test_active_acquisition import source_fixture

    source = next(iter(source_fixture().values()))
    old = definition("check:x", "sourced check", "relation", (source,)).model_dump(mode="json")
    aliases, proof = alias_sources((source,))
    aliased = deepcopy(old)
    aliased["reference"]["version"] = "research-source-alias/1.0.0"
    aliased["source_refs"] = [aliases[0].reference.model_dump(mode="json")]
    materials = [s.model_dump(mode="json") for s in aliases]
    rebuilt = original_definition(aliased, materials, (proof,))
    assert {k: v for k, v in rebuilt.items() if k != "reference"} == {
        k: v for k, v in old.items() if k != "reference"
    }
    assert original_definition(old, materials, None) == old
    with pytest.raises(ValueError, match="alias_provenance_missing"):
        original_definition(aliased, materials, None)
    wrong = deepcopy(proof)
    wrong["bindings"][0]["original"]["code"] = "SOURCE:OTHER"
    with pytest.raises(ValueError, match="alias_binding"):
        original_definition(aliased, materials, (wrong,))


def test_current_definition_replacement_is_not_hidden_by_captured_row():
    from packages.memory.procedure_context import query

    component = S(
        kind="scalar",
        modality="IOP",
        concept="iop",
        unit="mmHg",
        unit_basis="source",
        axis_semantics="not_applicable",
        id="c",
        status="obtained",
    )
    old = dict(id="r", component="c", definition={"reference": {}, "interpretation": "old"})
    base = S(
        read_dependency_input=lambda: None,
        ledger=[old],
        manifest=S(components=[component], patient_group=None),
        task=S(target_code="GOAL:P", checksum="task"),
    )
    first = query(base, old)
    base.ledger = [dict(old, definition={"reference": {}, "interpretation": "new"})]
    assert mismatch(first, query(base, old)) == "memory_method_definition_checksum_mismatch"
    base.ledger = []
    with pytest.raises(ValueError, match="current_requirement_missing"):
        query(base, old)


def test_removed_requirement_retires_hint_without_retry(monkeypatch):
    s = fixture()
    install(s, monkeypatch)
    s.prepare()

    def missing(*args):
        raise ValueError("memory_method_current_requirement_missing")

    monkeypatch.setattr(s, "method_context", missing)
    assert s.check() == ()
    assert s.events[-1]["reason"] == "memory_method_current_requirement_missing"


def test_repository_reuses_installed_archive_owner_and_rejects_other_database(monkeypatch):
    from apps import case_memory_store
    from packages.multimodal_research.reuse import repository

    target = S(host="synthetic", port=1234, database="research")
    installed = S(target=target)
    calls = []
    monkeypatch.setattr(case_memory_store, "PRIVATE", S(exists=lambda: True))
    monkeypatch.setattr(
        case_memory_store,
        "assembly",
        lambda role: (calls.append(role) or installed, object()),
    )
    campaign = S(
        target=target, base=S(objects=S(read_bounded=lambda *args: pytest.fail("input-only")))
    )
    assert repository(campaign) is installed and calls == ["reader"]
    with pytest.raises(ValueError, match="memory_target"):
        repository(S(target=S(host="other", port=1234, database="research")))


def test_method_context_unwraps_original_archive_owners_only(monkeypatch):
    from packages.memory import procedure_context
    from packages.memory.archive import ArchiveSources
    from packages.memory.execution_archive import ArchiveOwners

    owner = ArchiveSources(runtime=None, read_bytes=None, read_run_case=None, read_view=None)
    s = fixture()
    s.reader.sources = ArchiveOwners(owner, None)
    s.campaign.base.inputs = object()
    s.campaign.contexts = object()
    seen = []
    monkeypatch.setattr(
        procedure_context, "read", lambda current, *a, **k: seen.append(current) or context()
    )
    monkeypatch.setattr(procedure_context, "query", lambda *a: context())
    method, reason = s.method_context(object(), s.campaign.base.ledger[0])
    assert seen == [owner] and reason is None and method["context_checksum"]


def test_method_input_proof_does_not_accumulate_other_cases_in_current_projection(monkeypatch):
    from collections import Counter

    from packages.dataset_adapters.current_read_scope import (
        _projection_visible_reads,
        _read_visible,
    )
    from packages.memory import procedure_context

    inputs = S(repository=object(), visible_store=object())
    counts = Counter()

    def original_read(owner, record, *, inputs, contexts):
        def fresh():
            counts[record] += 1
            return {"case": record}

        return _read_visible(inputs, record, fresh)

    # Isolate the original archive decoding while exercising the public method
    # reader's production scope and the real bounded/fresh input-read mechanism.
    monkeypatch.setattr(procedure_context, "_read", original_read)
    with _projection_visible_reads():
        for n in range(24):
            result = procedure_context.read(None, str(n), inputs=inputs, contexts=None)
            assert result == {"case": str(n)} and counts[str(n)] == 2
    assert len(counts) == 24 and set(counts.values()) == {2}

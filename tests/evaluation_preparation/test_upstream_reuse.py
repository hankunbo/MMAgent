"""Contract tests; synthetic owner fixtures, no credentials/API or patient tools."""

import copy
import json
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.components import (
    available_concepts,
    papila_components,
    visible_components,
)
from packages.evidence_reuse.cache import ShadowCache
from packages.evidence_reuse.models import Component, InputComponents
from packages.evidence_reuse.requirements import audit
from packages.evidence_reuse.skills import compile_plans, recheck_plans
from packages.grounding.models import digest
from tests.dataset_adapters.test_papila import dataset


def inputs(*, age=True, modality="SLO", scope="CASE:QUERY"):
    parts = [
        Component(
            id="image",
            kind="image",
            concept=modality,
            status="obtained",
            source_checksum=digest("image"),
            source_locator="ASSET:IMAGE",
            shape=(2, 3),
        )
    ]
    if age:
        parts.append(
            Component(
                id="age",
                kind="scalar",
                concept="age",
                status="obtained",
                value=0.0,
                unit="year",
                source_checksum=digest("age"),
                source_locator="ASSET:AGE",
            )
        )
    return InputComponents(
        dataset="synthetic",
        adapter_version="fixture/1.0",
        scope=scope,
        input_checksum=digest("input"),
        split="training",
        patient_group=None,
        components=tuple(parts),
        runtime_ready=True,
    )


def plan():
    source = dict(code="SOURCE:A", checksum=digest("source"), version="1.0.0")
    definition = dict(interpretation_status="installed", source_refs=[source])
    return dict(
        input_checksum=digest("image"),
        sources=[source],
        requirements=[
            dict(
                id="criterion:AGE",
                definition=definition,
                input_modalities=["age"],
                basis_kind="quantitative",
            ),
            dict(
                id="criterion:VCDR_GE_07",
                definition=definition,
                input_modalities=["SLO"],
                basis_kind="qualitative",
            ),
        ],
    )


def test_papila_common_components_real_reader_and_negative_zero_missing(tmp_path):
    owner = dataset(tmp_path)
    value = papila_components(owner, "RET003OD")
    by_id = {c.id: c for c in value.components}
    assert by_id["clinical:dioptre_1"].value == 0.0
    assert by_id["clinical:vf_md"].value == -3.2
    assert by_id["clinical:iop_perkins"].reason == "not_recorded"
    assert by_id["clinical:astigmatism"].unit is None
    assert "SLO" not in available_concepts(value) and "age" in available_concepts(value)
    assert not value.runtime_ready and value.patient_group == "PAPILA:RET003"
    assert "diagnosis" not in value.model_dump_json() and "contour" not in value.model_dump_json()
    assert "OCT_volume" not in available_concepts(value)
    assert value == papila_components(owner, "RET003OD")


def test_visible_projection_cannot_mix_another_cases_age():
    artifact = SimpleNamespace(
        asset_code="ASSET:IMAGE",
        artifact_type_code="ARTIFACT:SLO_IMAGE",
        checksum=digest("image"),
        dimensions=(2, 3),
        laterality="UNKNOWN",
    )
    visible = SimpleNamespace(
        case=SimpleNamespace(case_code="CASE:A", artifacts=[artifact]),
        checksum=digest("visible"),
        adapter_version="test/1",
        dataset_ref=SimpleNamespace(code="test"),
    )
    context = dict(
        case_code="CASE:B",
        visible_checksum=visible.checksum,
        image_checksum=artifact.checksum,
        hidden_fields_used=False,
        age=dict(value=0.0, unit="year", status="obtained", reason=None),
        source_field="age",
    )
    with pytest.raises(ValueError, match="context_scope"):
        visible_components(visible, context)
    context["case_code"] = "CASE:A"
    value = visible_components(visible, context)
    assert "age" in available_concepts(value)
    assert value.components[-1].value == 0


@pytest.mark.parametrize("age,modality", [(True, "SLO"), (False, "SLO"), (True, "color_fundus")])
def test_skills_use_components_and_never_guess_modality(age, modality):
    c, p = inputs(age=age, modality=modality), plan()
    result = compile_plans(p, c)
    assert len(result["plans"]) == 2
    assert not result["clinical_evidence"]
    recheck_plans(result, p, c, memory=None)
    rows = {r["requirement_id"]: r for r in result["plans"]}
    assert rows["criterion:AGE"]["missing_components"] == ([] if age else ["age"])
    assert rows["criterion:VCDR_GE_07"]["missing_components"] == (
        [] if modality == "SLO" else ["SLO"]
    )
    assert all(not r["execution_authorized"] for r in result["plans"])


@pytest.mark.parametrize("change", ["source", "case", "port", "availability"])
def test_skill_replay_rechecks_current_binding(change):
    c, p = inputs(), plan()
    value = compile_plans(p, c)
    if change == "source":
        p["requirements"][0]["definition"]["source_refs"] = []
    elif change == "case":
        c = inputs(scope="CASE:OTHER")
    elif change == "availability":
        value["plans"][0]["missing_components"] = ["age"]
    else:
        value["plans"][0]["steps"][0]["port"] = "segman"
    with pytest.raises(ValueError, match="skill_"):
        recheck_plans(value, p, c, memory=None)


def test_inventory_distinguishes_context_missing_input_and_definition():
    p = plan()
    catalog = dict(
        tasks={
            "criterion:AGE": {},
            "criterion:VCDR_GE_07": {},
            "quality": dict(use="measurement_quality_or_input_context"),
            "definition": dict(selected_source_ids=["a"]),
            "source": dict(missing_exact_source_ids=["missing"]),
            "alias": dict(use="observation_only", related_definition="quality"),
        }
    )
    result = audit(catalog, p, inputs(age=False))
    rows = {r["id"]: r for r in result["rows"]}
    assert rows["quality"]["disposition"] == "context_or_dependency"
    assert rows["alias"]["target"] == "quality"
    assert rows["definition"]["next_action"] == "extract_definition_from_current_sources"
    assert rows["source"]["next_action"] == "bounded_source_search"
    assert rows["criterion:AGE"]["reasons"] == ["patient_input_missing"]
    assert result["frozen_denominator"] == 2
    assert sum(r["counts_in_coverage"] for r in result["rows"]) == 2
    p["sources"] = []
    with pytest.raises(ValueError, match="source_binding"):
        audit(catalog, p, inputs())


def test_context_roles_are_not_extra_votes_or_fake_measurements():
    p = plan()
    catalog = dict(tasks={r["id"]: {} for r in p["requirements"]})
    catalog["tasks"]["criterion:ANGLE"] = dict(use="measurement_quality_or_input_context")
    result = audit(catalog, p, inputs())
    row = next(r for r in result["rows"] if r["id"] == "criterion:ANGLE")
    assert row["disposition"] == "context_or_dependency"
    assert not row["counts_in_coverage"] and not row["independent_disease_vote"]


def test_skill_prefilter_does_not_replace_current_source_owner():
    calls = []

    def keys(**kwargs):
        calls.append(kwargs)
        return ("retracted", "valid")

    def read(key):
        if key == "retracted":
            raise ValueError("memory_source_withdrawn")
        raise RuntimeError("source_database_unavailable")

    memory = SimpleNamespace(repository=SimpleNamespace(procedure_keys=keys, read=read))
    with pytest.raises(RuntimeError, match="source_database_unavailable"):
        compile_plans(plan(), inputs(), memory=memory)
    assert calls == [dict(targets=("criterion:AGE",), modality="age", limit=20)]


def test_shadow_cache_always_rechecks_and_never_returns_mutable_prior():
    cache, calls = ShadowCache(limit=1), []

    def current():
        calls.append(1)
        return dict(version=1)

    for _ in range(2):
        value = cache.project(
            namespace="method",
            bindings=dict(purpose="research"),
            compute=lambda: dict(port="segman"),
            revalidate=current,
        )
        value["port"] = "caller_changed"
    assert len(calls) == 4 and cache.events[-1]["candidate_hit"]
    assert all(e["recomputed"] and not e["authorization_cached"] for e in cache.events)
    with pytest.raises(ValueError, match="shadow_mismatch"):
        cache.project(
            namespace="method",
            bindings=dict(purpose="research"),
            compute=lambda: dict(port="different"),
            revalidate=current,
        )


def test_cache_return_time_withdrawal_and_technical_error():
    c = ShadowCache()
    values = iter([dict(active=True), dict(active=False)])
    with pytest.raises(ValueError, match="current_changed"):
        c.project(namespace="x", bindings={}, compute=lambda: {}, revalidate=lambda: next(values))
    assert not c.values

    def unavailable():
        raise OSError("synthetic owner unavailable")

    with pytest.raises(OSError):
        c.project(namespace="x", bindings={}, compute=lambda: {}, revalidate=unavailable)


def test_saved_score_probe_no_writes_and_original_kernel():
    from packages.evidence_reuse.evaluation import evaluate_saved
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture, generate

    f = dynamic_fixture()
    f.acceptance.submit(generate(f), key="upstream-eval")
    f.acceptance.accept(key="upstream-eval")
    scores = f.acceptance.read(key="upstream-eval")[2]["scores"]
    report = dict(schema_version="dynamic-case-report/1.0.0", scores=scores)
    before = copy.deepcopy(report)
    result = evaluate_saved(report)
    assert result["exact_duplicate_equal"] and report == before
    assert result["original_requirement_denominator"] == 11
    assert not result["clinical_effectiveness_established"] and not result["labels_read"]
    report["scores"]["root"]["summary"]["combined"]["support"] = 0.12345
    with pytest.raises(ValueError, match="checksum"):
        evaluate_saved(report)


def test_upstream_opt_in_config_drift_and_legacy_not_upgraded(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_input as g
    from packages.evidence_reuse.integration import PROFILE
    from tests.evaluation_preparation.test_generic_input import data

    data(tmp_path / "input", "any.npz")
    monkeypatch.setattr(g, "directory", lambda key: tmp_path / key)
    s = g.prepare(
        dataset="fairvlmed",
        root=tmp_path / "input",
        sample="any.npz",
        key="new",
        evidence_profile="source-complete-view/1.0.0",
        upstream_profile=PROFILE,
    )
    assert s["schema_version"] == "fairvlmed-run-input/1.3.0" and g.load("new") == s
    assert len(g.owner_key(s, "phase", "review")) <= 80
    s["upstream_implementation"] = {}
    (tmp_path / "new/input.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="implementation_changed"):
        g.load("new")
    old = g.prepare(dataset="fairvlmed", root=tmp_path / "input", sample="any.npz", key="old")
    assert old["schema_version"] == "fairvlmed-run-input/1.1.0" and "upstream_profile" not in old

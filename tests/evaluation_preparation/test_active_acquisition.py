"""Public source-bound planning and native quality recovery, without live calls."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.llm_provider.models import Source
from packages.multimodal_research.active_requirements import discover, load, project
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.native_fundus import preparation_guard
from tests.dataset_adapters.test_multimodal import gamma, gdp
from tests.evaluation_preparation.test_multimodal_campaign import sources


def source_fixture():
    value = sources().initial
    profile = load()
    for key in {c["source"] for c in profile["checks"]}:
        text = "\n".join(c["quote"] for c in profile["checks"] if c["source"] == key)
        value[key] = Source(
            reference=ref("source", "SOURCE:" + key.upper(), digest(text)),
            locator="Synthetic explicit passage fixture",
            purpose="research_material",
            state="research_only",
            text=text,
        )
    return value


def task():
    return load_intent("early_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]


def test_source_bound_inventory_missing_inputs_and_correlation_not_seven(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    rows, trace = project(p.manifest, source_fixture(), task(), load())
    assert len(rows) == 18
    assert sum(r["component"] is not None for r in rows) == 16
    assert {r["id"] for r in rows if r["component"] is None} == {
        "active:cup_asymmetry",
        "active:functional_stage",
    }
    assert (
        next(r for r in rows if r["id"] == "active:cup_geometry")["category"]
        == next(r for r in rows if r["id"] == "active:cup_size")["category"]
    )
    assert len({r["category"] for r in rows}) < len(rows)
    assert all("e>0支持这个命题" in r["definition"].applicability_conditions for r in rows)
    assert trace["scope"] == "registered_source_catalog_not_exhaustive_literature"


def test_unbound_definition_is_backlog_not_an_invented_result(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    medical = source_fixture()
    del medical["aao_signs"]
    rows, trace = project(p.manifest, medical, task(), load())
    assert 0 < len(rows) < 18
    assert any(r["state"] == "definition_backlog" for r in trace["rows"])
    assert not any(r["id"] == "active:cup_geometry" for r in rows)


@pytest.mark.parametrize("mutation", ["cycle", "duplicate", "unknown_parent", "budget"])
def test_invalid_registry_rejected(tmp_path, mutation):
    profile = copy.deepcopy(load())
    if mutation == "cycle":
        profile["nodes"][0]["parent"] = profile["nodes"][0]["id"]
    elif mutation == "duplicate":
        profile["checks"].append(profile["checks"][0])
    elif mutation == "unknown_parent":
        profile["checks"][0]["parent"] = "not_known"
    else:
        profile["max_queries"] = 50
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="active_requirement"):
        load(path)


def test_search_original_receipts_exact_recovery_no_patient_query(tmp_path):
    calls = []

    def searcher(topic, key):
        calls.append((topic, key))
        return dict(topic=topic, request_key=key, status="empty", results=[])

    first = discover(tmp_path, load(), searcher=searcher)
    assert discover(tmp_path, load(), searcher=searcher) == first
    assert len(calls) == 3 and first["automatic_medical_acceptance"] is False
    path = tmp_path / "rnfl.json"
    value = json.loads(path.read_bytes())
    value["request_key"] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="active_search_binding"):
        discover(tmp_path, load(), searcher=searcher)


@pytest.mark.parametrize("stop", ["cancel", "result.json", "protocol.json", "deadline"])
def test_preparation_guard_stopped_never_launches(tmp_path, stop):
    base = SimpleNamespace(output=tmp_path, manifest=SimpleNamespace(checksum=digest("input")))
    guard = preparation_guard(base)
    guard()
    if stop == "deadline":
        base.active_start["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    else:
        (tmp_path / stop).write_text("{}")
    with pytest.raises(ValueError, match="active_preparation_stopped"):
        guard()


def test_native_measurement_is_not_resent_to_vision_and_missing_falls_back():
    base = SimpleNamespace(
        active_owner=True,
        observations={"native": object()},
        ledger=[
            dict(id="native", component="fundus", acquisition_method="disc_cup_measurement"),
            dict(id="other", component="fundus", acquisition_method="vision"),
        ],
    )
    assert MultimodalCampaign.select_targets(base, {"fundus"}) == ("other",)
    base.observations = {}
    assert MultimodalCampaign.select_targets(base, {"fundus"}) == ("native", "other")


def test_new_profile_preserves_numeric_inputs(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    # Apply the same disease proposition to numeric/map inputs without removing them.
    rows, _ = project(p.manifest, source_fixture(), load_intent("glaucoma_detection")[0], load())
    numeric = [r for r in rows if r["acquisition_method"] == "source_measurement"]
    by_id = {c.id: c.concept for c in p.manifest.components}
    assert any(by_id[r["component"]] == "age" for r in numeric)
    assert any(by_id[r["component"]] == "md" for r in numeric)


def test_medical_hierarchy_is_preserved_without_new_votes(tmp_path):
    from packages.multimodal_research.active_acquisition import structure

    p = prepare("gamma", gamma(tmp_path), "0001")
    rows, plan = project(p.manifest, source_fixture(), task(), load())
    owner = SimpleNamespace(
        check=lambda: None, report=lambda: {"plan": plan}, plan=plan, native=None
    )
    report = dict(
        requirements=[dict(id=r["id"]) for r in rows],
        report_production={},
        scores={"unchanged": 0.5},
    )
    result = structure(report, SimpleNamespace(active_owner=owner))
    edges = result["presentation_structure"]["edges"]
    assert dict(source="root", target="medical:STRUCTURE", relation="contains") in edges
    assert dict(source="medical:STRUCTURE", target="medical:CUPDISC", relation="contains") in edges
    assert (
        dict(source="medical:CUPDISC", target="check:active:cup_geometry", relation="contains")
        in edges
    )
    assert result["scores"] == {"unchanged": 0.5}


def test_native_geometry_empty_masks_remain_gap_and_zero_quality(tmp_path):
    from packages.dataset_adapters.reading import grayscale_image
    from packages.multimodal_research.fundus_measurements import measure

    raw = grayscale_image((512, 512), bytes(512 * 512))
    for name in ("disc", "cup", "rim", "labels"):
        (tmp_path / (name + ".png")).write_bytes(raw)
    result = measure(tmp_path)
    assert result["status"] == "missing" and result["quality"] == 0
    assert result["values"]["vertical_cdr"] is None
    assert result["direction"] is None and result["qc"]["stop_case"] is False
    (tmp_path / "cup.png").write_bytes(
        grayscale_image((512, 512), bytes([255]) + bytes(512 * 512 - 1))
    )
    with pytest.raises(ValueError, match="fundus_mask_class_binding"):
        measure(tmp_path)


def test_native_current_readback_uses_original_object_bound(tmp_path, monkeypatch):
    import copy

    from packages.grounding.models import canonical
    from packages.multimodal_research import native_fundus
    from packages.object_store import LocalObjectStore, checksum

    store = LocalObjectStore(tmp_path / "objects")
    manifest = {"tool": "registered_fixture"}
    plan = {"manifest_checksum": digest(manifest)}
    measurement = {"status": "obtained", "quality": 0.6, "values": {"vertical_cdr": 0.5}}
    prediction = {"input_checksum": digest("pixels")}
    payloads = {
        name: b"confirmed image fixture"
        for name in ("cup.png", "disc.png", "rim.png", "labels.png", "overlay.png")
    }
    payloads.update(
        plan=canonical(plan), measurements=canonical(measurement), prediction=canonical(prediction)
    )
    artifacts = []
    for name, raw in payloads.items():
        sha = checksum(raw)
        uri = store.promote(store.stage(raw), sha)
        document = dict(
            uri=uri,
            checksum=sha,
            extensions={"output_port": name},
            scope={"case_code": "CASE:FIXTURE"},
            derived_from=["ASSET:INPUT"],
        )
        artifacts.append(
            SimpleNamespace(
                uri=uri,
                checksum=sha,
                extensions=document["extensions"],
                scope=SimpleNamespace(case_code="CASE:FIXTURE"),
                derived_from=["ASSET:INPUT"],
                model_dump=lambda d=document, **kw: d,
            )
        )
    monkeypatch.setattr(native_fundus, "current_manifest", lambda: manifest)
    base = SimpleNamespace(
        binding={"case_code": "CASE:FIXTURE"},
        manifest=SimpleNamespace(checksum=digest("input")),
        root=SimpleNamespace(asset_code="ASSET:INPUT"),
        objects=store,
        runtime=SimpleNamespace(read_observation_bundle=lambda key: tuple(artifacts)),
    )
    record = dict(
        case_code="CASE:FIXTURE",
        input_checksum=digest("input"),
        plan=plan,
        measurement=measurement,
        prediction=prediction,
        status="obtained",
        bundle_key="TASK:NATIVE",
        artifacts=[a.model_dump() for a in artifacts],
        artifact=next(
            a.model_dump() for a in artifacts if a.extensions["output_port"] == "measurements"
        ),
    )
    native_fundus.recheck(base, record)
    altered = copy.deepcopy(record)
    altered["measurement"]["values"]["vertical_cdr"] = 0.9
    with pytest.raises(ValueError, match="native_fundus_record_content_binding"):
        native_fundus.recheck(base, altered)
    altered = copy.deepcopy(record)
    altered["prediction"]["input_checksum"] = digest("other")
    with pytest.raises(ValueError, match="native_fundus_record_content_binding"):
        native_fundus.recheck(base, altered)

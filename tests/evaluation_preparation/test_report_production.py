"""Producer, not renderer, owns explicit source/check/execution semantics."""

import copy
import json

import pytest

from agents.expert_agent.reporting import read_research_report
from packages.evaluation_preparation.evidence_tree_v3 import build, render
from packages.evaluation_preparation.report_production import PROFILE, media_links, produce
from packages.grounding.models import digest
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture, generate


def report_for(f, key="producer-review"):
    state = f.owner.current()
    f.acceptance.submit(generate(f), key=key)
    assert f.acceptance.accept(key=key).status == "accepted"
    detail = read_research_report(f.acceptance, key=key, parent=state)
    return dict(
        schema_version="dynamic-case-report/1.0.0",
        result=dict(case_code=state.material.scope.case_code, sample="synthetic-new-dataset"),
        original_review=detail,
        scores=detail["scores"],
        requirements=[
            dict(
                id=r.id,
                examination=r.definition.examination,
                modality=list(r.input_modalities),
                status="source_backed",
                sources=[ref.model_dump(mode="json") for ref in r.definition.source_refs],
            )
            for r in state.policy.requirements
        ],
    )


def test_production_preserves_original_acceptance_scores_and_missing():
    f = dynamic_fixture(5)
    report = report_for(f)
    before = copy.deepcopy(report)
    result = produce(report, f.owner.current())
    assert report == before
    assert result["scores"] == report["scores"]
    assert result["original_review"] == report["original_review"]
    assert result["report_production"]["schema_version"] == PROFILE
    assert len(result["requirements"]) == 5
    assert all(r["presentation_sections"]["rationale"] for r in result["requirements"])
    packet = build(result, input_checksum=digest(result))
    assert all(
        n["parent"].startswith("organization:") for n in packet["navigation"] if n.get("check_id")
    )
    assert packet["scores"] == before["scores"]
    assert (
        "<script>" not in result["requirements"][0]["presentation_sections"]["rationale"]["purpose"]
    )
    assert "evidence-view" in render(result, input_checksum=digest(result))


@pytest.mark.parametrize("fault", ["case", "source", "requirement", "artifact"])
def test_production_wrong_scope_rejected(fault):
    f = dynamic_fixture(1)
    value, assets = report_for(f), []
    if fault == "case":
        value["result"]["case_code"] = "CASE:OTHER"
    elif fault == "source":
        value["requirements"][0]["sources"] = []
    elif fault == "requirement":
        value["requirements"].append(value["requirements"][0])
    else:
        assets = [dict(asset_code="x", scope=dict(case_code="CASE:OTHER"))]
    with pytest.raises(ValueError, match="report_production_"):
        produce(value, f.owner.current(), artifacts=assets)


def test_explicit_media_binding_no_name_guess_and_cross_check_not_borrowed():
    from tests.evaluation_preparation.test_evidence_tree_v2 import packet as legacy_media

    f = dynamic_fixture(3)
    report = produce(report_for(f), f.owner.current())
    old = legacy_media(digest(report))
    # Fixture media's identity must be referenced, not merely given a familiar tool name.
    first = next(iter(old["images"].values()))
    first["reference"] = dict(asset_code="ASSET:SYNTHETIC")
    report["requirements"][0]["execution_binding"]["artifacts"] = [
        dict(asset_code="ASSET:SYNTHETIC", checksum=first["checksum"])
    ]
    old["source_checksum"] = digest(report)
    old["case_code"] = report["result"]["case_code"]
    old["input_checksum"] = f.owner.current().material.scope.input_checksum
    media = media_links(report, old)
    assert media["links"]
    assert {link["node_id"] for link in media["links"]} == {
        "check:" + report["requirements"][0]["id"]
    }
    assert build(report, input_checksum=digest(report), media=media)["assets"]["links"]


def test_zero_observation_report_retains_requirements_without_filling_scores():
    f = dynamic_fixture(3, empty=True)
    state = f.owner.current()
    value = dict(
        schema_version="dynamic-case-report/1.0.0",
        result=dict(case_code=state.material.scope.case_code),
        original_review=None,
        scores=None,
        requirements=[
            dict(id=r.id, sources=[s.model_dump(mode="json") for s in r.definition.source_refs])
            for r in state.policy.requirements
        ],
    )
    result = produce(value, state)
    packet = build(result, input_checksum=digest(result))
    assert packet["confidence"] is None and len(packet["checks"]) == 3
    assert all(r["execution_binding"]["method"] is None for r in result["requirements"])


def test_profile_version_and_uncovered_code_drift(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_input as module
    from packages.evaluation_preparation import report_production, unified_requirements
    from packages.evidence_reuse import integration

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "inspect_input", lambda *a, **kw: (None, dict(input="synthetic")))
    monkeypatch.setattr(unified_requirements, "catalog", lambda: {})
    monkeypatch.setattr(integration, "implementation", lambda: {"upstream": "unchanged"})
    monkeypatch.setattr(report_production, "implementation", lambda: {"production": "v1"})
    spec = module.prepare(
        dataset="fairvlmed",
        root=tmp_path,
        sample="arbitrary.npz",
        key="report-test",
        evidence_profile=unified_requirements.PROFILE,
        upstream_profile=integration.PROFILE,
        report_profile=PROFILE,
    )
    assert spec["schema_version"] == "fairvlmed-run-input/1.4.0"
    assert module.load(spec["key"]) == spec
    assert module.owner_key(spec, "phase")
    monkeypatch.setattr(report_production, "implementation", lambda: {"production": "v2"})
    with pytest.raises(ValueError, match="report_implementation_changed"):
        module.load(spec["key"])


def test_display_recovery_never_constructs_dataset_or_paid_run(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_run

    path = tmp_path / "saved.json"
    path.write_text(json.dumps(dict(report_production=dict(schema_version=PROFILE))))
    monkeypatch.setattr(
        generic_run, "export", lambda k: dict(json=str(path), html=str(tmp_path / "saved.html"))
    )
    monkeypatch.setattr(generic_run, "load", lambda _: pytest.fail("no execution input"))
    calls = []

    def convert(key, destination):
        calls.append(key)
        if len(calls) == 1:
            raise ValueError("synthetic_display_failure")
        return dict(html=str(destination), new_requests=0)

    monkeypatch.setattr(generic_run, "export_presentation", convert)
    with pytest.raises(ValueError, match="display_failure"):
        generic_run.finish_export("saved")
    assert generic_run.finish_export("saved")["tree"]["new_requests"] == 0


@pytest.mark.parametrize("number", [0, -0.4])
def test_native_result_link_requires_exact_projection_and_full_record(number):
    f = dynamic_fixture(1)
    report = report_for(f)
    state = f.owner.current().model_dump(mode="json")
    item = state["policy"]["items"][0]
    observation = state["material"]["observations"][0]
    identifier = item["id"]
    measured = dict(value=number, method="synthetic-calculator", unit="ratio")
    item["dependency_paths"] = [["ASSET:FULL"]]
    observation["value"] = json.dumps(dict(measured, full_record="ASSET:FULL"))
    native = dict(observation=dict(inputs=dict(mask=dict(sha256="a" * 64))))
    report["native_tools"] = dict(results=dict(calculator=native))
    artifact = dict(
        asset_code="ASSET:FULL", scope=dict(case_code=state["material"]["scope"]["case_code"])
    )
    result = produce(
        report,
        state,
        artifacts=[artifact],
        measured={identifier: measured},
        origins={identifier: "calculator"},
    )
    binding = result["requirements"][0]["execution_binding"]
    assert binding["native_task"] == "calculator" and binding["native_record"] == native
    assert any(e["relation"] == "produces" for e in result["presentation_structure"]["edges"])
    observation["value"] = json.dumps(dict(measured, value=number + 1, full_record="ASSET:FULL"))
    changed = produce(
        report,
        state,
        artifacts=[artifact],
        measured={identifier: measured},
        origins={identifier: "calculator"},
    )
    assert changed["requirements"][0]["execution_binding"]["native_task"] is None
    assert changed["scores"] == report["scores"]


def test_campaign_producer_uses_original_model_store_and_bounded_visible_reads(tmp_path):
    from types import SimpleNamespace

    from packages.evaluation_preparation.report_production import campaign_report
    from packages.object_store import LocalObjectStore, checksum

    f = dynamic_fixture(1)
    report = report_for(f)
    state = f.owner.current()
    store = LocalObjectStore(tmp_path / "visible")
    raw = b"synthetic original visible bytes"
    uri = store.promote(store.stage(raw), checksum(raw))
    resource = state.resources[0]
    a = dict(
        asset_code=resource.code,
        checksum=checksum(raw),
        uri=uri,
        scope=dict(case_code=state.material.scope.case_code),
    )
    model_resource = resource.model_copy(update=dict(code="ASSET:MODEL", checksum=digest("model")))
    model_asset = dict(
        asset_code=model_resource.code,
        checksum=model_resource.checksum,
        uri="object://separate-owner",
        scope=a["scope"],
    )
    state = state.model_copy(
        update=dict(
            resources=(resource.model_copy(update=dict(checksum=checksum(raw))), model_resource)
        )
    )
    assets = {resource.code: a, model_resource.code: model_asset}

    def read(case, code):
        value = assets[code]
        return SimpleNamespace(**value, model_dump=lambda **kw: value)

    model = dict(
        source=dict(artifact_code=model_resource.code, artifact_checksum=model_resource.checksum)
    )
    calls = []
    campaign = SimpleNamespace(
        base=SimpleNamespace(runtime=SimpleNamespace(read_case_artifact=read), objects=store),
        read_model=lambda: calls.append(1) or model,
    )
    report["model_output"] = model
    produced = campaign_report(report, campaign, state)
    assert len(calls) == 1 and len(produced["production_inputs"]["artifacts"]) == 2

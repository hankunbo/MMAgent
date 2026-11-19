"""Saved-result structure and display contracts; no live clinical calls."""

import copy
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from packages.evaluation_preparation.evidence_assets import VERSION as ASSETS
from packages.evaluation_preparation.evidence_presentation import (
    STRUCTURE,
    forest,
    project,
    resolve,
    validate_packet,
)
from packages.evaluation_preparation.evidence_tree_v3 import build, bundle, render, unbundle
from packages.evaluation_preparation.evidence_view import convert
from packages.grounding.models import digest
from packages.object_store import checksum
from tests.evaluation_preparation.test_evidence_tree_v2 import packet as old_media
from tests.evaluation_preparation.test_unified_view import report


def test_no_disease_profile_and_scores_unchanged():
    value = report()
    before = copy.deepcopy(value)
    packet = project(value, source_checksum="sha256:test")
    assert packet["schema_version"] == "evidence-presentation/1.0.0"
    assert packet["scores"] == value["scores"] and value == before
    assert set(packet["checks"]) == {r["id"] for r in value["requirements"]}
    assert packet["coverage"]["complete"]
    assert not packet["memory"]["current_authorization"]


def test_mixed_legacy_definition_not_reused_as_two_fields():
    value = report()
    definition = dict(examination="Requirement with old reference", interpretation="Mixed old text")
    value["original_review"]["records"]["review_tasks"]["tasks"] = [
        dict(assessment_id="risk:age", definition=definition)
    ]
    packet = project(value, source_checksum="sha256:test")
    semantic = packet["checks"]["risk:age"]["sections"]
    assert semantic["rationale"] is None and semantic["reference"] is None
    assert semantic["legacy_definition"]["interpretation"] == "Mixed old text"


def structured():
    value = report()
    value["presentation_structure"] = dict(
        schema_version=STRUCTURE,
        nodes=[
            dict(id="new-goal", type="goal", title="Synthetic non-ocular goal"),
            dict(id="multi", type="goal", title="Independent branch"),
        ],
        edges=[
            dict(source="root", target="new-goal", relation="contains"),
            dict(source="root", target="multi", relation="contains"),
            dict(source="new-goal", target="check:risk:age", relation="contains"),
            dict(source="multi", target="check:risk:age", relation="contains"),
            dict(source="check:feature:normal", target="check:risk:age", relation="refutes"),
        ],
    )
    return value


def test_arbitrary_owner_hierarchy_shared_edges_and_orphans():
    value = structured()
    before = copy.deepcopy(value)
    p = project(value, source_checksum="sha256:test")
    parent = {n["id"]: n["parent"] for n in p["navigation"]}
    assert parent["check:risk:age"] == "new-goal"
    assert parent["check:feature:unknown"] == "root"
    assert len([e for e in p["edges"] if e["target"] == "check:risk:age"]) == 3
    assert p["counts"]["nonzero_contributors"] == 2
    assert value == before
    for n in p["nodes"]:
        assert digest(resolve(value, n["origin"]["pointer"])) == n["origin"]["checksum"]


def test_cycle_is_reported_not_recursed_or_deleted_from_saved_relations():
    value = structured()
    value["presentation_structure"]["edges"].append(
        dict(source="check:risk:age", target="new-goal", relation="contains")
    )
    p = project(value, source_checksum="sha256:test")
    assert any(d["code"] == "contains_cycle_or_downstream" for d in p["diagnostics"])
    assert next(n for n in p["navigation"] if n["id"] == "check:risk:age")["parent"] == "root"
    validate_packet(p)


@pytest.mark.parametrize("fault", ["endpoint", "node", "version", "task", "observation"])
def test_bad_references_and_ambiguous_identity_rejected(fault):
    value = structured()
    s = value["presentation_structure"]
    if fault == "endpoint":
        s["edges"][0]["target"] = "not-exported"
    elif fault == "node":
        s["nodes"].append(s["nodes"][0])
    elif fault == "version":
        s["schema_version"] = "unknown"
    elif fault == "task":
        value["original_review"]["records"]["review_tasks"]["tasks"] = [dict(assessment_id="x")] * 2
    else:
        value["original_review"]["material"] = dict(
            observations=[dict(reference=dict(code="obs"))] * 2
        )
    with pytest.raises(ValueError, match="presentation_"):
        project(value, source_checksum="sha256:test")


def test_unknown_node_type_has_visible_fallback():
    value = structured()
    value["presentation_structure"]["nodes"][0]["type"] = "future-type"
    p = project(value, source_checksum="sha256:test")
    assert any(n["id"] == "new-goal" for n in p["navigation"])
    assert any(d["code"] == "unknown_node_type" for d in p["diagnostics"])


def test_empty_and_missing_only_reports_are_valid_not_fake_scores():
    for count in (0, 1):
        value = dict(
            schema_version="dynamic-case-report/1.0.0",
            requirements=[dict(id="no-data")] * count,
            scores=None,
        )
        p = build(value, input_checksum="sha256:test")
        assert p["confidence"] is None and p["counts"]["reviewed"] == 0
        assert len(p["checks"]) == count
        validate_packet(p)


def test_legacy_parent_tree_retained_without_guessing_check_mapping():
    value = dict(
        schema="evidence-graph-html-snapshot/2.1.0",
        current_scoring=report()["scores"],
        nodes=[
            dict(key="old-root", title="Old", parent=None),
            dict(key="child", title="Saved leaf", parent="old-root"),
        ],
    )
    p = project(value, source_checksum="sha256:test")
    assert (
        next(n for n in p["navigation"] if n["id"] == "historical:child")["parent"]
        == "historical:old-root"
    )
    assert p["scores"] == value["current_scoring"]
    assert any(
        d["code"] == "historical_tree_and_review_ids_not_guessed_merged" for d in p["diagnostics"]
    )


def test_explicit_six_section_semantics_and_distinct_state_dimensions():
    value = report()
    value["requirements"][0]["presentation_sections"] = dict(
        schema_version="evidence-sections/1.0.0",
        rationale="Relation",
        acquisition="Input only",
        reference="Comparator",
    )
    value["requirement_inventory"] = dict(rows=[dict(id="risk:age", definition_installed=False)])
    p = project(value, source_checksum="sha256:test")
    s = p["checks"]["risk:age"]["sections"]
    assert (s["rationale"], s["acquisition"], s["reference"]) == (
        "Relation",
        "Input only",
        "Comparator",
    )
    assert p["checks"]["risk:age"]["status"]["review_present"]
    assert any(d["code"] == "catalog_and_current_review_are_distinct" for d in p["diagnostics"])


def test_unknown_fields_are_accessible_sensitive_fields_redacted_and_no_mutation():
    value = report()
    value["future_extension"] = dict(
        note="Public extension", api_key="synthetic-secret", nested=dict(raw_note="hidden-note")
    )
    before = copy.deepcopy(value)
    p = build(value, input_checksum="sha256:test")
    page = render(value, input_checksum="sha256:test")
    assert "synthetic-secret" not in page and "hidden-note" not in page
    assert p["appendices"]["future_extension"]["note"] == "Public extension"
    assert p["coverage"]["complete"] and len(p["coverage"]["restricted"]) == 2
    assert value == before


def test_nonfinite_and_depth_limits():
    value = report()
    value["extra"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        project(value, source_checksum="sha256:test")
    node = value["extra"] = {}
    for _ in range(90):
        node["next"] = {}
        node = node["next"]
    with pytest.raises(ValueError, match="depth_limit"):
        project(value, source_checksum="sha256:test")


def test_large_deep_forest_uses_iterative_processing():
    nodes = [dict(id="root")] + [dict(id=str(i)) for i in range(1500)]
    edges = [
        dict(source="root" if i == 0 else str(i - 1), target=str(i), relation="contains")
        for i in range(1500)
    ]
    parents, warnings = forest(nodes, edges)
    assert parents["1499"] == "1498" and not warnings


def multi_media():
    value = report()
    old = old_media()["images"]["input"]
    value["result"]["case_code"] = "CASE:TEST"
    value["native_tools"] = dict(image_checksum="sha256:input")
    value["requirements"][0]["observation"] = dict(
        status="obtained",
        image=dict(reference=dict(code="asset:a", checksum=old["checksum"])),
    )
    media = dict(
        schema_version=ASSETS,
        source_checksum="sha256:test",
        case_code="CASE:TEST",
        input_checksum="sha256:input",
        items=[
            dict(
                old,
                id="image:a",
                kind="image",
                role="frame",
                laterality="LEFT",
                timepoint="t1",
                reference=dict(asset_code="asset:a"),
            ),
            dict(
                old,
                id="image:b",
                kind="image",
                role="frame",
                laterality="RIGHT",
                timepoint="t2",
                reference=dict(asset_code="asset:b"),
            ),
            dict(
                id="table",
                kind="table",
                title="Table",
                columns=["x", "y"],
                rows=[[0, -1], [2, 0]],
                reference={},
            ),
            dict(
                id="series",
                kind="series",
                title="Curve",
                points=[[0, -1], [1, 0]],
                axes=["s", "unit"],
                reference={},
            ),
            dict(
                id="stack",
                kind="image_sequence",
                title="Saved slices",
                frames=[old, old],
                reference={},
            ),
            dict(id="unknown", kind="future-volume", reference=dict(code="future")),
        ],
        links=[
            dict(
                node_id="check:risk:age",
                asset_id="image:a",
                source_pointer="/requirements/0/observation",
                source_value_checksum=digest(value["requirements"][0]["observation"]),
            )
        ],
    )
    return value, media


def test_multi_assets_same_role_different_eye_and_time_with_exact_binding():
    value, media = multi_media()
    p = build(value, input_checksum="sha256:test", media=media)
    assert len(p["assets"]["items"]) == 6
    assert p["assets"]["links"][0]["asset_id"] == "image:a"
    assert any(d["code"] == "unsupported_media_kind" for d in p["assets"]["diagnostics"])
    assert any(d.get("asset_id") == "image:b" for d in p["assets"]["diagnostics"])
    assert p["assets"]["items"][0]["timepoint"] != p["assets"]["items"][1]["timepoint"]


def test_identical_pixels_cannot_replace_another_eye_artifact():
    value, media = multi_media()
    media["links"][0]["asset_id"] = "image:b"
    with pytest.raises(ValueError, match="asset_link_not_referenced"):
        build(value, input_checksum="sha256:test", media=media)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_checksum", "sha256:other"),
        ("case_code", "CASE:OTHER"),
        ("input_checksum", "sha256:other"),
    ],
)
def test_wrong_media_scope_is_fatal(field, value):
    source, media = multi_media()
    media[field] = value
    with pytest.raises(ValueError, match="asset_.*binding"):
        build(source, input_checksum="sha256:test", media=media)


@pytest.mark.parametrize("change", ["pointer", "check", "id", "item_scope"])
def test_link_forgery_and_duplicate_identity_rejected(change):
    value, media = multi_media()
    if change == "pointer":
        media["links"][0]["source_value_checksum"] = "sha256:wrong"
    elif change == "check":
        media["links"][0]["node_id"] = "check:feature:normal"
    elif change == "id":
        media["items"][1]["id"] = media["items"][0]["id"]
    else:
        media["items"][0]["case_code"] = "CASE:OTHER"
    with pytest.raises(ValueError, match="asset_"):
        build(value, input_checksum="sha256:test", media=media)


@pytest.mark.parametrize("bad", ["url", "checksum", "table", "series"])
def test_bad_single_asset_does_not_erase_valid_report(bad):
    value, media = multi_media()
    if bad == "url":
        media["items"][0]["data_uri"] = "javascript:alert(1)"
    elif bad == "checksum":
        media["items"][0]["checksum"] = "sha256:wrong"
        media["links"] = []
    elif bad == "table":
        media["items"][2]["rows"] = [[1]]
    else:
        media["items"][3]["points"] = [[0, float("inf")]]
    p = build(value, input_checksum="sha256:test", media=media)
    assert p["scores"] == value["scores"]
    assert p["assets"]["diagnostics"]
    assert any(a["display_status"] == "ready" for a in p["assets"]["items"])


def test_legacy_binding_uses_dependencies_and_no_medical_id_map():
    value = report()
    media = old_media()
    media["images"]["input"]["reference"] = dict(asset_code="asset:any")
    value["scores"]["reviews"][0]["dependency_paths"] = [["asset:any"]]
    value["original_review"]["scores"] = copy.deepcopy(value["scores"])
    p = build(value, input_checksum="sha256:test", media=media)
    assert p["assets"]["links"] == [
        dict(
            node_id="check:risk:age",
            asset_id="legacy:input",
            relation="dependency_asset",
            origin=dict(kind="exact_saved_reference", matched=["asset:any"]),
        )
    ]


def test_bundle_repeat_conversion_preserves_scores_media_and_source(tmp_path):
    value, media = multi_media()
    source, mp, output = tmp_path / "source.json", tmp_path / "media.json", tmp_path / "one.html"
    source.write_text(json.dumps(value))
    media["source_checksum"] = checksum(source.read_bytes())
    mp.write_text(json.dumps(media))
    result = convert(source, output, layout="tree-v3", media_path=mp)
    assert result == convert(source, output, layout="tree-v3", media_path=mp)
    exported = Path(result["presentation"])
    again = convert(exported, tmp_path / "two.html", layout="tree-v3")
    assert output.read_bytes() == Path(again["html"]).read_bytes()
    assert again["new_requests"] == 0 and not again["rescored"]
    with pytest.raises(ValueError, match="overwrite"):
        convert(source, source, layout="tree-v3")
    with pytest.raises(ValueError, match="already_has_media"):
        convert(exported, tmp_path / "third.html", layout="tree-v3", media_path=mp)


def test_output_conflict_checked_before_partial_html_write(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "out.html"
    source.write_text(json.dumps(report()))
    output.with_suffix(".view.json").write_text("original unrelated output")
    with pytest.raises(ValueError, match="exists"):
        convert(source, output, layout="tree-v3")
    assert not output.exists()


def test_bundle_tampering_and_cross_component_score_mismatch():
    p = build(report(), input_checksum="sha256:test")
    saved = bundle(p)
    saved["packet"]["scores"]["root"] = {}
    with pytest.raises(ValueError, match="bundle_integrity"):
        unbundle(saved)
    p = build(report(), input_checksum="sha256:test")
    p["checks"]["risk:age"]["sections"]["score"] = None
    with pytest.raises(ValueError, match="check_score"):
        validate_packet(p)


def test_html_escapes_active_content_and_keeps_shared_original_shell():
    value = report()
    attack = '</script><img src=x onerror="alert(1)">'
    value["result"]["sample"] = attack
    page = render(value, input_checksum="sha256:test")
    soup = BeautifulSoup(page, "html.parser")
    assert not soup.find_all("img")
    assert "fetch(" not in page and "innerHTML" not in page
    assert "connect-src 'none'" in page
    assert soup.title.string == attack + " · 病例证据图"
    assert json.loads(soup.find(id="evidence-view").string)["sample"] == attack


def test_current_saved_case_has_exact_counts_and_all_assets_reachable():
    path = Path("docs/exports/FairVLMed_fv-unified-live-r2-20261005.json")
    value = json.loads(path.read_bytes())
    media = json.loads(
        Path(
            "docs/exports/FairVLMed_fv-unified-live-r2-20261005_路径树与图像版.media.json"
        ).read_bytes()
    )
    p = build(value, input_checksum=checksum(path.read_bytes()), media=media)
    assert len(p["checks"]) == 24 and p["counts"]["reviewed"] == 16
    assert p["counts"]["numeric"] == 14 and p["counts"]["nonzero_contributors"] == 6
    assert p["scores"] == value["scores"] and len(p["assets"]["items"]) == 12
    assert p["memory"]["saved"] == value["native_tools"]["memory"]
    assert not any(link["node_id"] == "check:criterion:AGE" for link in p["assets"]["links"])
    assert p["coverage"]["complete"]
    assert any(link["node_id"] == "tool:av_metrics" for link in p["assets"]["links"])


def test_owner_bound_export_to_presentation_without_mocking_export(monkeypatch, tmp_path):
    from packages.evaluation_preparation import generic_run
    from packages.grounding.models import canonical
    from tests.evaluation_preparation.test_generic_export import fixture

    _, raw, artifacts = fixture(tmp_path, monkeypatch)
    value = report()
    value["result"]["case_code"] = "CASE:SYNTHETIC"
    first = json.loads(raw[artifacts[0].uri])
    first["value"] = value
    second = json.loads(raw[artifacts[1].uri])
    second["value"]["report_checksum"] = digest(value)
    for artifact, envelope in zip(artifacts, [first, second], strict=True):
        raw[artifact.uri] = canonical(envelope)
        artifact.checksum = checksum(raw[artifact.uri])
    saved = generic_run.export("export-test")
    media = old_media(checksum(Path(saved["json"]).read_bytes()))
    media["case_code"] = "CASE:SYNTHETIC"
    mp = tmp_path / "media.json"
    mp.write_text(json.dumps(media))
    result = generic_run.export_presentation("export-test", tmp_path / "view.html", media_path=mp)
    assert result["counts"]["reviewed"] == 4 and result["new_requests"] == 0
    assert generic_run.export("export-test") == saved


def test_original_runtime_export_port_is_reused(monkeypatch, tmp_path):
    from packages.evaluation_preparation import generic_run

    source = tmp_path / "original.json"
    source.write_text(json.dumps(report()))
    mp = tmp_path / "media.json"
    mp.write_text(json.dumps(old_media(checksum(source.read_bytes()))))
    calls = []

    def original(key):
        calls.append(key)
        return dict(json=str(source))

    monkeypatch.setattr(generic_run, "export", original)
    result = generic_run.export_presentation("same-run", tmp_path / "view.html", media_path=mp)
    assert calls == ["same-run"] and result["new_requests"] == 0


def test_actual_historical_html_keeps_unknown_requirement_count(tmp_path):
    from packages.evaluation_preparation.evidence_view import load_report

    source = Path("docs/exports/青光眼证据图.html")
    original, _ = load_report(source)
    output = tmp_path / "legacy.html"
    result = convert(source, output, layout="tree-v3")
    assert result["counts"]["requirements"] is None
    packet = unbundle(json.loads(output.with_suffix(".view.json").read_text()))
    assert packet["scores"] == original["current_scoring"]
    assert len([n for n in packet["nodes"] if n["id"].startswith("historical:")]) == len(
        original["nodes"]
    )
    assert packet["assets"]["diagnostics"] == [dict(code="media_not_packaged")]


def test_envelope_schema_and_business_binding_are_separate():
    from packages.evaluation_preparation.presentation_contracts import (
        DisplayExport,
        EvidenceAssets,
        PresentationStructure,
        check,
    )

    value, media = multi_media()
    check(PresentationStructure, structured()["presentation_structure"])
    check(EvidenceAssets, media)
    check(DisplayExport, bundle(build(value, input_checksum="sha256:test", media=media)))
    media["items"][0]["id"] = None
    with pytest.raises(ValueError, match="presentation_contract:EvidenceAssets:items/0/id"):
        build(value, input_checksum="sha256:test", media=media)


@pytest.mark.parametrize("fault", ["id", "endpoint"])
def test_bundle_rechecks_media_identity_even_with_recomputed_checksum(fault):
    value, media = multi_media()
    saved = bundle(build(value, input_checksum="sha256:test", media=media))
    assets = saved["packet"]["assets"]
    if fault == "id":
        assets["items"].append(assets["items"][0])
    else:
        assets["links"][0]["node_id"] = "not-present"
    saved["packet_checksum"] = digest(saved["packet"])
    with pytest.raises(ValueError, match="presentation_bundle_asset"):
        unbundle(saved)

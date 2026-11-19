"""Clinical navigation and saved-media integrity, without new medical inference."""

import base64
import copy
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from packages.evaluation_preparation.evidence_media import build, checked_file, validate_images
from packages.evaluation_preparation.evidence_tree_v2 import PROFILE, project, render
from packages.evaluation_preparation.evidence_view import convert
from packages.object_store import checksum
from tests.evaluation_preparation.test_unified_view import report

CASE = Path("docs/exports/FairVLMed_fv-unified-live-r2-20261005.json")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg=="
)


def packet(digest="sha256:test"):
    return dict(
        schema_version="evidence-media/1.0.0",
        source_checksum=digest,
        images={
            "input": dict(
                title="Synthetic",
                checksum=checksum(PNG),
                shape=[1, 1],
                reference={},
                note="Fixture, not patient evidence",
                data_uri="data:image/png;base64," + base64.b64encode(PNG).decode(),
            )
        },
    )


def test_navigation_is_clinical_tree_not_sections_and_preserves_scores():
    source = json.loads(CASE.read_bytes())
    before = copy.deepcopy(source)
    view = project(source, input_checksum="sha256:test")
    nodes = {n["id"]: n for n in view["reader"]["navigation"]}
    assert {n["payload"]["id"] for n in nodes.values() if n["type"] == "requirement"} == {
        r["id"] for r in source["requirements"]
    }
    assert all(n["type"] in {"goal", "requirement"} for n in nodes.values())
    assert not any(k.endswith((":fact", ":review", ":quality", ":dependencies")) for k in nodes)
    current, path = "check:criterion:VCDR_GE_07", []
    while current:
        path.insert(0, nodes[current]["title"])
        current = nodes[current]["parent"]
    assert path == [
        "青光眼评估",
        "视神经结构",
        "盘杯及盘沿形态",
        "同图杯盘比与盘沿测量",
        "垂直杯盘比",
    ]
    assert view["scores"] == source["scores"] and source == before
    assert view["reader"]["media_roles"]["check:criterion:AGE"] == []
    assert view["reader"]["media_roles"]["check:criterion:VCDR_GE_07"] == [
        "input",
        "segman_overlay",
        "segman_disc",
        "segman_cup",
    ]


def test_other_goal_does_not_inherit_glaucoma_or_fixed_result_slots():
    value = report()
    view = project(value, input_checksum="sha256:test")
    nav = view["reader"]["navigation"]
    assert nav[0]["title"] == "病例评估"
    assert len(nav) == len(value["requirements"]) + 1
    assert not any(n["id"].startswith("nav:") for n in nav)


def test_unknown_requirement_retained_empty_branches_pruned_cycle_rejected():
    value = report()
    profile = json.loads(PROFILE.read_bytes())
    view = project(value, input_checksum="sha256:test", profile=profile)
    assert len(view["reader"]["navigation"]) == 6
    profile["checks"]["missing"] = ["CUPDISC", "Missing test", []]
    profile["groups"]["STRUCTURE"] = ["CUPDISC", "Cycle"]
    with pytest.raises(ValueError, match="navigation_group_parent"):
        project(value, input_checksum="sha256:test", profile=profile)


def test_media_binds_report_and_exact_bytes_with_safe_data_uri():
    media = packet()
    page = render(report(), input_checksum="sha256:test", media=media)
    soup = BeautifulSoup(page, "html.parser")
    snapshot = json.loads(soup.find(id="evidence-view").string)
    assert snapshot["reader"]["media"] == media
    assert "image-dialog" in page and "section-5" in page
    with pytest.raises(ValueError, match="evidence_media_report_binding"):
        render(report(), input_checksum="sha256:other", media=media)
    media["images"]["input"]["checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="media_image_integrity"):
        validate_images(media)
    media["images"]["input"]["data_uri"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="media_data_uri"):
        validate_images(media)


def test_native_file_scope_and_checksum(tmp_path):
    path = tmp_path / "mask.png"
    path.write_bytes(PNG)
    assert checked_file(tmp_path, path.as_uri(), checksum(PNG)) == PNG
    with pytest.raises(ValueError, match="media_file_outside_run"):
        checked_file(tmp_path / "other", path.as_uri(), checksum(PNG))
    with pytest.raises(ValueError, match="media_checksum_mismatch"):
        checked_file(tmp_path, path.as_uri(), "sha256:bad")
    with pytest.raises(ValueError, match="media_local_uri"):
        checked_file(tmp_path, "https://example.com/x.png", checksum(PNG))


def test_offline_media_cannot_switch_case_even_with_same_report_checksum():
    value = json.loads(CASE.read_bytes())
    media = packet()
    media["case_code"] = "CASE:OTHER"
    with pytest.raises(ValueError, match="evidence_media_case_input_binding"):
        project(value, input_checksum="sha256:test", media=media)


def test_wrong_case_binding_rejected_before_reading_owner(tmp_path):
    root = Path(".local/fairvlmed-runs/fv-unified-live-r2-20261005")
    binding = json.loads((root / "binding.json").read_bytes())
    value = json.loads(CASE.read_bytes())
    value["result"]["case_code"] = "CASE:OTHER"
    with pytest.raises(ValueError, match="media_case_input_binding"):
        build(
            value,
            source_checksum="sha256:test",
            root=tmp_path,
            binding=binding,
            runtime=None,
            objects=None,
        )


def test_offline_package_conversion_and_explicit_missing_images(tmp_path):
    source, output = tmp_path / "report.json", tmp_path / "tree.html"
    source.write_text(json.dumps(report()))
    image_file = tmp_path / "media.json"
    image_file.write_text(json.dumps(packet(checksum(source.read_bytes()))))
    result = convert(source, output, layout="tree-v2", media_path=image_file)
    assert result["image_count"] == 1 and result["new_requests"] == 0 and not result["rescored"]
    assert result == convert(source, output, layout="tree-v2", media_path=image_file)
    with pytest.raises(ValueError, match="view_media_requires_tree_v2"):
        convert(source, output, layout="tree-v1", media_path=image_file)
    view = project(report(), input_checksum="sha256:test")
    assert not view["reader"]["media"]["images"] and view["reader"]["media"]["missing"]

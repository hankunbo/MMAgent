"""Refresh display labels from the same saved report, without recomputing evidence."""

import json

import pytest

from packages.evaluation_preparation.evidence_tree_v3 import build, bundle, convert_saved, unbundle
from packages.grounding.models import digest
from tests.evaluation_preparation.test_evidence_tree_v2 import packet as old_media
from tests.evaluation_preparation.test_unified_view import report


def test_saved_examination_precedes_internal_score_title():
    value = report()
    value["requirements"][0]["examination"] = "Human readable measurement"
    p = build(value, input_checksum=digest(value))
    assert p["checks"]["risk:age"]["title"] == "Human readable measurement"
    assert (
        next(n for n in p["navigation"] if n["id"] == "check:risk:age")["title"]
        == "Human readable measurement"
    )
    assert p["scores"] == value["scores"]


def test_same_report_can_reuse_validated_bundle_media_without_overwriting_it(tmp_path):
    value = report()
    sha = digest(value)
    media = old_media(sha)
    p = build(value, input_checksum=sha, media=media)
    saved = bundle(p)
    source, assets, output = (
        tmp_path / "report.json",
        tmp_path / "saved.view.json",
        tmp_path / "new.html",
    )
    source.write_text(json.dumps(value))
    assets.write_text(json.dumps(saved))
    original = assets.read_bytes()
    result = convert_saved(source, output, media_path=assets)
    refreshed = unbundle(json.loads(output.with_suffix(".view.json").read_text()))
    assert result["image_count"] == 1 and result["rescored"] is False
    assert refreshed["assets"] == p["assets"]
    assert refreshed["scores"] == p["scores"]
    assert assets.read_bytes() == original
    value["result"]["sample"] = "other"
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="presentation_media_source_binding"):
        convert_saved(source, tmp_path / "wrong.html", media_path=assets)

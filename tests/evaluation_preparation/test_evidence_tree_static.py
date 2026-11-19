"""Readable saved trees must not depend on an IDE permitting JavaScript."""

import copy
import json

from bs4 import BeautifulSoup

from packages.evaluation_preparation.evidence_tree_v3 import build, render_packet
from packages.grounding.models import digest
from tests.evaluation_preparation.test_evidence_tree_v3 import multi_media, structured
from tests.evaluation_preparation.test_unified_view import report


def test_static_tree_contains_real_content_before_any_script_runs():
    value, media = multi_media()
    packet = build(value, input_checksum="sha256:test", media=media)
    before = copy.deepcopy(packet)
    soup = BeautifulSoup(render_packet(packet), "html.parser")
    tree = soup.find(id="saved-tree")
    body = soup.find(id="saved-detail")
    assert tree and body, "HTML must contain the tree and findings, not empty JS placeholders"
    assert "55.0000 / 100" in body.get_text()
    assert len(body.select("[data-check-id]")) == len(packet["checks"])
    assert len(body.select("[data-section]")) == 6 * len(packet["checks"])
    assert body.find("img") and body.find("table")
    assert not soup.find(id="saved-reader").has_attr("hidden")
    assert soup.find(id="interactive-reader").has_attr("hidden")
    assert packet == before
    assert digest(json.loads(soup.find(id="evidence-view").string)) == digest(packet)
    for a in soup.select("#saved-reader a[href^='#']"):
        assert soup.find(id=a["href"][1:]), a


def test_static_tree_preserves_owner_hierarchy_and_safe_unknown_nodes():
    value = structured()
    value["presentation_structure"]["nodes"][0]["title"] = '<img src=x onerror="attack()">'
    packet = build(value, input_checksum="sha256:test")
    soup = BeautifulSoup(render_packet(packet), "html.parser")
    link = soup.select_one('#saved-tree [data-node-id="new-goal"]')
    assert link and link.find_parent("li").find(attrs={"data-node-id": "check:risk:age"})
    assert not soup.find("img") and not soup.select("[onerror]")
    assert '<img src=x onerror="attack()">' in soup.find(id="saved-tree").get_text()


def test_static_empty_and_missing_reports_do_not_invent_scores():
    value = dict(schema_version="dynamic-case-report/1.0.0", requirements=[], scores=None)
    soup = BeautifulSoup(render_packet(build(value, input_checksum="sha256:test")), "html.parser")
    assert "未评定" in soup.find(id="saved-detail").get_text()
    assert "0.0000 / 100" not in soup.find(id="saved-detail").get_text()
    assert "图片" in soup.find(id="saved-detail").get_text()


def test_static_zero_negative_and_unknown_are_not_conflated():
    packet = build(report(), input_checksum="sha256:test")
    soup = BeautifulSoup(render_packet(packet), "html.parser")
    for key, expected in [
        ("feature:neutral", "0"),
        ("feature:normal", "-0.2"),
        ("feature:unknown", "未评定"),
    ]:
        check = soup.find(attrs={"data-check-id": key})
        assert check
        cells = check.select("[data-score-table] tbody td")
        assert cells[3].get_text() == expected

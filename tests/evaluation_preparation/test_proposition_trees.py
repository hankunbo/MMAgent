"""Independent proposition views preserve accepted scores and source identity."""

import copy

import pytest

from packages.evaluation_preparation.evidence_tree_v3 import build, bundle, render_packet, unbundle
from packages.evaluation_preparation.report_production import produce
from packages.grounding.models import digest
from packages.multimodal_research.proposition_trees import split_packet
from tests.evaluation_preparation.test_hypothesis_scoring import fixture
from tests.evaluation_preparation.test_report_production import report_for


def test_one_case_one_proposition_no_rescore_and_six_sections():
    f = fixture()
    report = produce(report_for(f), f.owner.state)
    packet = build(report, input_checksum=digest(report))
    before = copy.deepcopy(report)
    trees = split_packet(report, packet)
    assert len(trees) == 4 and report == before
    assert len({v["proposition_tree"]["tree_id"] for _, v in trees}) == 4
    for goal, value in trees:
        assert value["nodes"][0]["goal_summary"] == goal["meaning"]
        assert all(e["origin"]["kind"] for e in value["edges"])
        assert value["scores"]["target"] == goal["code"]
        assert {r["target"] for r in value["scores"]["reviews"]} == {goal["code"]}
        assert value["counts"]["requirements"] == 1
        assert all(n["parent"] == "root" for n in value["navigation"] if n["id"] != "root")
        assert {"rationale", "acquisition", "reference", "score"} <= set(
            next(iter(value["checks"].values()))["sections"]
        )
        assert unbundle(bundle(value)) == value
        assert "evidence-view" in render_packet(value)
        assert value["proposition_tree"]["new_acceptance"] is False


def test_changed_source_and_wrong_hypothesis_binding_rejected():
    f = fixture()
    report = produce(report_for(f), f.owner.state)
    packet = build(report, input_checksum=digest(report))
    report["scores"]["reviews"][0]["target"] = "GOAL:FOREIGN"
    with pytest.raises(ValueError, match="proposition_source_binding"):
        split_packet(report, packet)

"""Synthetic normalization tests; no additional patient/model/clinical judgments."""

import copy
import json

import pytest

from apps.evidence_graph_records import attach, inspect, validate_view
from apps.evidence_graph_scores import attach as scored
from packages.evidence_details.models import digest
from packages.evidence_details.records import build, validate
from packages.evidence_details.records_models import Datum, EvidenceRecords
from tests.test_evidence_scoring_interpretation import interpreted  # noqa: F401
from tests.test_evidence_scoring_participation import graded  # noqa: F401
from tests.test_glaucoma_structure_ui import browser


@pytest.fixture(scope="module")
def parent(interpreted):  # noqa: F811
    base, _, scores = interpreted
    return scored(base, scores)


def seal(value):
    value["checksum"] = digest({k: v for k, v in value.items() if k != "checksum"})
    return value


def test_typed_pairs_are_complete_and_preserve_original_scoring(parent):
    before = copy.deepcopy(parent)
    records = build(parent)
    view = attach(parent, records)
    assert parent == before == validate_view(view)
    assert view["research_scoring"] == parent["research_scoring"]
    assert len(records["observations"]) == len(records["interpretations"]) == 46
    assert len(records["node_index"]) == 63
    assert records["inventory"]["direction_assessed"] == 7
    assert (
        EvidenceRecords.model_validate_json(json.dumps(records)).model_dump(mode="json") == records
    )
    observations = {o["assessment_id"]: o for o in records["observations"]}
    assert observations["criterion:FAMILY"]["acquisition"] == "not_obtained"
    assert observations["criterion:FAMILY"]["values"] == []
    assert all(
        v["name"] != "vertical_cup_disc_ratio"
        for v in observations["criterion:RIM_NOTCH"]["values"]
    )
    assert observations["visual:rim_continuity"]["acquisition"] == "obtained"
    assert observations["visual:rim_continuity"]["result_kind"] == "visual_review"
    assert (
        inspect(view, assessment="criterion:IOP")["interpretations"][0]["status"] == "input_missing"
    )
    with pytest.raises(ValueError, match="unknown_node"):
        inspect(view, node="not-a-node")
    with pytest.raises(ValueError, match="unknown_assessment"):
        inspect(view, assessment="not-an-assessment")
    resolved = inspect(view, assessment="criterion:VCDR_GE_07", resolve_bindings=True)
    assert resolved["resolved_bindings"] and len(resolved["observations"]) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [float("-inf")], [1.0] * 17])
def test_invalid_numeric_measurements_rejected(value):
    with pytest.raises(ValueError):
        Datum(name="fixture", value=value, unit="1", missing_reason=None)


@pytest.mark.parametrize("change", ["missing_reason", "unknown_field", "none_without_reason"])
def test_explicit_missing_value_and_strict_fields(change):
    v = {"name": "fixture", "value": 1.0, "unit": "1", "missing_reason": None}
    if change == "missing_reason":
        v["missing_reason"] = "not acquired"
    elif change == "unknown_field":
        v["caller_probability"] = 0.9
    else:
        v["value"] = None
    with pytest.raises(ValueError):
        Datum.model_validate(v)


@pytest.mark.parametrize(
    "change", ["case", "input", "binding", "unit", "link", "omit", "score", "construct"]
)
def test_rehashed_or_constructed_records_cannot_replace_sources(parent, change):
    value = build(parent)
    if change == "case":
        value["case_code"] = "CASE:ANON_OTHER"
    elif change == "input":
        value["input_checksum"] = "sha256:" + "a" * 64
    elif change == "binding":
        value["observations"][0]["input_bindings"][0]["checksum"] = "sha256:" + "a" * 64
    elif change == "unit":
        next(o for o in value["observations"] if o["values"])["values"][0]["unit"] = "invented-unit"
    elif change == "link":
        value["interpretations"][0]["observation_id"] = value["observations"][-1]["id"]
    elif change == "omit":
        value["interpretations"].pop()
    elif change == "score":
        j = next(j for j in value["interpretations"] if j["status"] == "input_missing")
        j["judgment"] = copy.deepcopy(
            next(x for x in value["interpretations"] if x["status"] == "assessed")["judgment"]
        )
    else:
        value = EvidenceRecords.model_construct(
            **{**value, "clinical_acceptance": True}
        ).model_dump()
    with pytest.raises(ValueError):
        validate(parent, seal(value))


def test_changed_parent_requires_a_new_binding_and_cannot_reuse_old_records(parent):
    changed = copy.deepcopy(parent)
    changed["case_evidence"]["sample"] = "synthetic-new-binding"
    seal(changed)
    with pytest.raises(ValueError, match="source_reconstruction"):
        validate(changed, build(parent))


def test_new_measurements_have_values_without_inheriting_old_scores(parent):
    # Shape-only synthetic adapter fixture: not the production v1.17 validator/runner.
    p = copy.deepcopy(parent)
    nodes = ["VASCULAR_SKELETON_DENSITY", "VASCULAR_FRACTAL", "AV_TORTUOSITY", "ARTERIOLAR_CALIBER"]
    p["vascular_followup"] = {
        "record": {
            "input_bindings": {"fixture": True},
            "policy": {"fixture": True},
            "classes": {
                c: {
                    "whole_image": {"skeleton_density": 0.01},
                    "skeleton_fractal": {"value": 1.1},
                    "tortuosity": {"mean": 1.2},
                    "annulus": {"width_proxy_disc_ratio": 0.05},
                }
                for c in ["artery", "vein"]
            },
        },
        "artifact": {"synthetic": True},
        "execution": {"synthetic": True},
        "policy": {"synthetic": True},
        "sources": {"fixture": {"synthetic": True}},
        "comparisons": [
            {
                "node": node,
                "state": "proxy_measured" if node == nodes[-1] else "measured_reference_unmatched",
                "observed": "synthetic acquired values",
                "reference": "synthetic source",
                "interpretation": "comparison unavailable",
                "remaining": "matched reference required",
                "source_ids": ["fixture"],
            }
            for node in nodes
        ],
    }
    seal(p)
    records = build(p)
    rows = [
        j
        for j in records["interpretations"]
        if j["node"] in nodes and j["assessment_id"].startswith("association:")
    ]
    assert len(rows) == 4
    for j in rows:
        o = next(o for o in records["observations"] if o["id"] == j["observation_id"])
        assert o["acquisition"] == "obtained" and len(o["values"]) == 2
        assert j["judgment"]["value"] is None and j["observation_reliability"]["value"] is None
        assert j["score_action"] == "new_observation_not_rescored"
        assert j["dependency_semantics"] == "legacy_not_rebased"
    assert p["research_scoring"] == parent["research_scoring"]


def test_browser_keeps_six_sections_with_separate_observation_and_judgment(parent):
    browser(
        attach(parent, build(parent)),
        r"""
check(text(elements.detail).includes('观察取得与医学判读'),'standard overview missing');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const h=elements.detail.children.filter(x=>x.tag==='h3'&&/^\d ·/.test(x.textContent));
 check(h.length===6,'six sections');
}
run("show('CUP_RIM_MEASURE')");
check(fifth().includes('连续判读 e'),'typed interpretation missing');
check(fifth().includes('可靠性'),'ratings missing');
""",
    )

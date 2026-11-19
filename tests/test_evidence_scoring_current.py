"""Current-record scoring tests use synthetic records, never new patient reviews."""

import base64
import copy

import pytest

from apps.evidence_graph_records import attach as typed
from apps.evidence_graph_scores import attach, unscored
from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from packages.evidence_details.models import digest
from packages.evidence_details.records import build as records
from packages.evidence_scoring.current import (
    build,
    completeness,
    rows_from_records,
    state,
    validate,
)
from packages.evidence_scoring.current_models import CurrentScores
from packages.evidence_scoring.replacement import build as replacement
from packages.evidence_scoring.replacement import metrics
from packages.object_store import checksum as bytes_checksum
from tests.test_evidence_records import parent  # noqa: F401
from tests.test_evidence_scoring_interpretation import interpreted  # noqa: F401
from tests.test_evidence_scoring_participation import graded  # noqa: F401
from tests.test_glaucoma_disease_conditions import mask_bodies
from tests.test_glaucoma_structure_ui import browser


@pytest.fixture(scope="module")
def current(parent):  # noqa: F811
    base = typed(parent, records(parent))
    return base, build(base)


def test_exact_historical_scores_and_measurement_direction_separation(current):
    base, score = current
    v = attach(base, score)
    assert unscored(v) == base
    assert v["research_scoring"] == base["research_scoring"]
    assert v["evidence_records"] == base["evidence_records"]
    assert (
        score["root"]["summary"]["confidence"]
        == base["research_scoring"]["nodes"]["ROOT"]["confidence"]
    )
    assert len(score["nodes"]) == 63 and len(score["reviews"]) == 46
    assert score["conclusion"]["disease_probability"] is None
    c = score["conclusion"]
    assert c["age_changes_direction"] == (
        {c["all_evidence"]["confidence"]["direction"], c["without_age"]["confidence"]["direction"]}
        == {"support_side", "refutation_side"}
    )
    assert c["age_delta"] > 0
    assert (
        score["root"]["coverage"]["slo_image_research"]["acquisition"]
        > score["root"]["coverage"]["slo_image_research"]["interpretation"]
    )


def test_new_vascular_values_change_coverage_and_dependencies_not_direction(parent):  # noqa: F811
    p = copy.deepcopy(parent)
    nodes = ["VASCULAR_SKELETON_DENSITY", "VASCULAR_FRACTAL", "AV_TORTUOSITY", "ARTERIOLAR_CALIBER"]
    p["vascular_followup"] = {
        "record": {
            "input_bindings": {"synthetic": True},
            "policy": {"synthetic": True},
            "classes": {
                c: {
                    "whole_image": {"skeleton_density": 0.01},
                    "skeleton_fractal": {"value": 1.1},
                    "tortuosity": {"mean": 1.2},
                    "annulus": {"width_proxy_disc_ratio": 0.05},
                }
                for c in ("artery", "vein")
            },
        },
        "artifact": {"synthetic": True},
        "execution": {"synthetic": True},
        "policy": {"synthetic": True},
        "sources": {"fixture": {"synthetic": True}},
        "comparisons": [
            {
                "node": n,
                "state": "proxy_measured" if n == nodes[-1] else "measured_reference_unmatched",
                "observed": "synthetic measurement",
                "reference": "synthetic source",
                "interpretation": "no matched comparator",
                "remaining": "reference",
                "source_ids": ["fixture"],
            }
            for n in nodes
        ],
    }
    p["checksum"] = snapshot_checksum({k: v for k, v in p.items() if k != "checksum"})
    rows, resources, acquired, judged = rows_from_records(typed(p, records(p)))
    root = state(rows, acquired, judged)
    assert "av_measurement" in resources
    assert "association:ARTERIOLAR_CALIBER" not in acquired
    for resource in ("slo_artery", "slo_vein", "segman_disc", "av_measurement"):
        after = state(rows, acquired, judged, removed_resources={resource})
        assert (
            after["coverage"]["slo_image_research"]["acquisition"]
            < root["coverage"]["slo_image_research"]["acquisition"]
        )
        if resource != "segman_disc":
            assert after["summary"]["confidence"] == root["summary"]["confidence"]
    assert all(
        r.judgment.value is None
        for r in rows
        if r.node in nodes and r.id.startswith("association:")
    )


def test_missing_scenarios_and_exhaustive_sets_are_not_medical_counterfactuals(current):
    _, score = current
    assert score["key_analysis"]["evaluated_subsets"] == 128
    assert score["key_analysis"]["smallest_direction_flip_removals"]
    for s in score["key_analysis"]["smallest_fidelity_sets"]:
        assert len(s["assessment_ids"]) >= 4  # not an age-only shortcut
        assert s["interpretation_retention"] >= 0.8
    assert all(s["disease_effect"] is None for s in score["missing_scenarios"])
    for r in score["removals"]:
        assert r["after"]["coverage"]["slo_image_research"]["denominator"] == 19.0
    no_image = next(r for r in score["removals"] if r["id"] == "resource:image")
    assert no_image["after"]["summary"]["confidence"]["basis"] == ["risk"]
    assert completeness([])["slo_image_research"]["acquisition"] is None


@pytest.mark.parametrize("mutation", ["case", "coverage", "dependency", "probability", "construct"])
def test_forged_current_scores_rejected(current, mutation):
    base, s = current
    forged = copy.deepcopy(s)
    if mutation == "case":
        forged["case_code"] = "CASE:OTHER"
    elif mutation == "coverage":
        forged["root"]["coverage"]["slo_image_research"]["acquisition"] = 1.0
    elif mutation == "dependency":
        forged["reviews"][0]["dependency_paths"] = [["image"]]
    elif mutation == "probability":
        forged["conclusion"]["disease_probability"] = 0.9
    else:
        forged = (
            CurrentScores.model_validate(forged)
            .model_copy(update={"clinical_acceptance": True})
            .model_dump(mode="json")
        )
    forged["checksum"] = digest({k: v for k, v in forged.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate(base, forged)


def test_replacement_metric_checks_nested_shapes():
    shape = (15, 15)
    disc = bytes(255 if 2 <= i // 15 <= 12 and 2 <= i % 15 <= 12 else 0 for i in range(225))
    cup = bytes(255 if 5 <= i // 15 <= 9 and 5 <= i % 15 <= 9 else 0 for i in range(225))
    assert metrics(shape, disc, cup)["vertical_cup_disc_ratio"] == 5 / 11
    with pytest.raises(ValueError, match="not_nested"):
        metrics(shape, cup, disc)


@pytest.mark.parametrize("change", [None, "bytes", "case", "ancestry", "shape", "laterality"])
def test_replacement_verifies_exact_observed_inputs(change):
    bodies = mask_bodies()
    case = {
        "case_code": "CASE:FIXTURE",
        "input": {"asset_code": "ASSET:IMAGE", "dimensions": [24, 24], "laterality": "UNKNOWN"},
        "artifacts": [],
        "images": {},
    }
    for key, kind, body in [
        ("segman_disc", "ARTIFACT:SLO_SEGMAN_DISC", bodies["disc"]),
        ("disc_mask", "ARTIFACT:SLO_DISC_MASK", bodies["disc"]),
        ("segman_cup", "ARTIFACT:SLO_SEGMAN_CUP", bodies["cup"]),
    ]:
        case["images"][key] = "data:image/png;base64," + base64.b64encode(body).decode()
        case["artifacts"].append(
            {
                "asset_code": "ASSET:" + key.upper(),
                "artifact_type_code": kind,
                "scope": {"case_code": case["case_code"]},
                "dimensions": [24, 24],
                "laterality": "UNKNOWN",
                "qc_status": "PASS",
                "checksum": bytes_checksum(body),
                "derived_from": ["ASSET:IMAGE"],
            }
        )
    a = case["artifacts"][1]
    if change == "bytes":
        case["images"]["disc_mask"] = case["images"]["segman_cup"]
    elif change == "case":
        a["scope"]["case_code"] = "CASE:OTHER"
    elif change == "ancestry":
        a["derived_from"] = ["ASSET:OTHER_IMAGE"]
    elif change == "shape":
        a["dimensions"] = [12, 48]
    elif change == "laterality":
        a["laterality"] = "LEFT"
    if change:
        with pytest.raises(ValueError, match="replacement_"):
            replacement({"case_evidence": case})
    else:
        out = replacement({"case_evidence": case})
        assert out["status"] == "measured_requires_new_interpretation"
        assert all(v == 0 for v in out["deltas"].values())
        assert out["disease_score_after_replacement"] is None


def test_current_ui_keeps_six_sections_and_explicit_root_conclusion(current):
    base, score = current
    browser(
        attach(base, score),
        r"""
check(text(elements.detail).includes('2 · 结论与关键分数'),'current root');
check(text(elements.detail).includes('年龄是否改变方向：'),'age flag');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const hs=elements.detail.children.filter(x=>x.tag==='h3'&&/^\d ·/.test(x.textContent));
 check(hs.length===6,'six sections');
}
run("show('VASCULAR_SKELETON_DENSITY')");
check(fifth().includes('本节点汇总与逐项定位'),'current node scoring');
""",
    )

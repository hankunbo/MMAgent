"""Synthetic protocol/ablation tests, not additional patient reviews."""

import copy
import json

import pytest

from apps import evidence_review_study as cli
from apps.evidence_graph_scores import attach
from packages.evidence_details.models import digest
from packages.evidence_scoring import current as scoring
from packages.evidence_scoring.review_protocol import ReviewItem, validate_round
from packages.evidence_scoring.review_study import compare_rounds, indicators, run
from tests.test_evidence_records import parent  # noqa: F401
from tests.test_evidence_scoring_current import current  # noqa: F401
from tests.test_evidence_scoring_interpretation import interpreted  # noqa: F401
from tests.test_evidence_scoring_participation import graded  # noqa: F401


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def rounds(current):  # noqa: F811
    base, _ = current
    rows, _, _, _ = scoring.rows_from_records(base)
    status = {r["assessment_id"]: r["status"] for r in base["evidence_records"]["interpretations"]}
    protocol = cli.read_json(cli.PROTOCOL)[1]
    items = [
        {
            "assessment_id": row.id,
            "status": "reference_missing" if status[row.id] == "not_reviewed" else status[row.id],
            "observation_available": row.observation_available,
            "observation_citations": ["fixture:observation"] if row.observation_available else [],
            "reference_citations": ["fixture:reference"],
            "comparison": "synthetic software fixture, not an independent clinical review",
            "observation": row.observation.model_dump(mode="json"),
            "relation": row.relation.model_dump(mode="json"),
            "applicability": row.applicability.model_dump(mode="json"),
            "judgment": row.judgment.model_dump(mode="json"),
            "limitation_allocation": "synthetic fixture",
            "unresolved": [],
        }
        for row in rows
    ]
    return [
        {
            "schema_version": "case-review-round/1.0.0",
            "protocol_checksum": digest(protocol),
            "material_checksum": digest("synthetic material"),
            "round_id": key,
            "reviewer_kind": "assistant_manual_simulation",
            "model_identifier": "synthetic-not-an-actual-reviewer",
            "session_identifier": "synthetic-" + key,
            "prompt_checksum": digest("synthetic prompt"),
            "generation_settings": {"mode": "fixture"},
            "prior_answers_visible": False,
            "items": copy.deepcopy(items),
            "purpose": "candidate_review_not_production_authorization",
        }
        for key in ("A", "B", "C")
    ]


def compare(base, rounds):
    return compare_rounds(
        base,
        rounds,
        material_checksum=digest("synthetic material"),
        citation_ids={"fixture:observation", "fixture:reference"},
        protocol=cli.read_json(cli.PROTOCOL)[1],
        prompt_checksum=digest("synthetic prompt"),
    )


def test_mechanisms_preserve_original_patient_and_policy(current):  # noqa: F811
    base, score = current
    before = digest(base), digest(scoring.POLICY)
    result = run(base)
    assert result["passed"] and all(result["checks"].values())
    assert result["baseline"] == indicators(score["root"])
    assert result["new_medical_reviews"] == 0 and result["case_count"] == 1
    assert result["background_analysis"] == score["conclusion"]
    assert result["key_sets"] == score["key_analysis"]
    assert before == (digest(base), digest(scoring.POLICY))
    trials = {t["id"]: t for t in result["trials"]}
    assert trials["without_age"]["D_delta"] < 0
    assert trials["without_all_assessments"]["indicators"]["D"] is None


def test_freeze_replay_is_read_only_and_refuses_adopting_old_directory(current, tmp_path):  # noqa: F811
    base, score = current
    snapshot = tmp_path / "snapshot.json"
    write_json(snapshot, attach(base, score))
    output = tmp_path / "study"
    assert cli.freeze(snapshot, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    write_json(snapshot, {"later_export": "does not change frozen source bytes"})
    assert cli.verify(output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="already_exists"):
        cli.freeze(snapshot, output)
    manifest = cli.read_json(output / "manifest.json")[1]
    assert manifest["review_materials"]["status"] == "not_released"
    assert set(manifest["independent_rounds"].values()) == {"not_started"}
    assert cli.main(["verify", "--study", str(output)]) == 0


def test_tampering_detected_and_no_output_on_invalid_snapshot(current, tmp_path, monkeypatch):  # noqa: F811
    base, score = current
    snapshot = tmp_path / "snapshot.json"
    write_json(snapshot, attach(base, score))
    output = tmp_path / "study"
    cli.freeze(snapshot, output)
    monkeypatch.setattr(cli, "code_inventory", lambda: {})
    with pytest.raises(ValueError, match="implementation_drift"):
        cli.verify(output)
    view = cli.read_json(snapshot)[1]
    view["current_scoring"]["root"]["summary"]["combined"]["support"] = 0.99
    write_json(snapshot, view)
    with pytest.raises(ValueError, match="snapshot_checksum"):
        cli.freeze(snapshot, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_candidate_rounds_compare_without_production_acceptance(current, rounds):  # noqa: F811
    base, score = current
    result = compare(base, rounds)
    assert result["root_score_range"] == 0
    assert result["production_authorization"] is False
    assert result["consensus_score"] is None
    assert len(result["items"]) == 46 and result["case_count"] == 1
    assert result["results"]["A"]["indicators"] == indicators(score["root"])


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "material", "citation", "session", "model"]
)
def test_round_scope_and_independence_metadata_checks(current, rounds, mutation):  # noqa: F811
    edited = copy.deepcopy(rounds)
    if mutation == "missing":
        edited[0]["items"].pop()
    elif mutation == "duplicate":
        edited[0]["items"].append(edited[0]["items"][0])
    elif mutation == "material":
        edited[0]["material_checksum"] = digest("other")
    elif mutation == "citation":
        edited[0]["items"][0]["reference_citations"] = ["invented"]
    elif mutation == "session":
        edited[0]["session_identifier"] = edited[1]["session_identifier"]
    else:
        edited[0]["model_identifier"] = "another-model"
    with pytest.raises(ValueError, match="review_"):
        compare(current[0], edited)


def test_score_disagreement_is_reported_not_averaged(current, rounds):  # noqa: F811
    edited = copy.deepcopy(rounds)
    item = next(i for i in edited[0]["items"] if i["assessment_id"] == "criterion:AGE")
    item["judgment"].update(value=-0.5, lower=-0.8, upper=0.3)
    result = compare(current[0], edited)
    assert result["root_requires_investigation"]
    age = next(i for i in result["items"] if i["assessment_id"] == "criterion:AGE")
    assert age["investigate"] and len(age["directions"]) == 2
    assert result["consensus_score"] is None


@pytest.mark.parametrize("mutation", ["nan", "range", "observation", "status", "construct"])
def test_invalid_reviews_cannot_become_evidence(rounds, mutation):
    edited = copy.deepcopy(rounds[0])
    item = next(i for i in edited["items"] if i["assessment_id"] == "criterion:AGE")
    if mutation == "nan":
        item["observation"]["value"] = float("nan")
    elif mutation == "range":
        item["judgment"]["upper"] = -1.0
    elif mutation == "observation":
        item["observation_available"] = False
    elif mutation == "status":
        item["status"] = "input_missing"
    else:
        copied = ReviewItem.model_validate(item).model_copy(update={"observation_available": False})
        edited["items"][edited["items"].index(item)] = copied.model_dump(mode="json")
    with pytest.raises(ValueError):
        validate_round(
            edited,
            expected_ids={i["assessment_id"] for i in rounds[0]["items"]},
            material_checksum=edited["material_checksum"],
            protocol_checksum=edited["protocol_checksum"],
            citation_ids={"fixture:observation", "fixture:reference"},
        )


def test_round_cannot_invent_acquisition_or_replace_model_mapping(current, rounds):  # noqa: F811
    edited = copy.deepcopy(rounds)
    missing = next(i for i in edited[0]["items"] if not i["observation_available"])
    missing.update(observation_available=True, status="reference_missing")
    with pytest.raises(ValueError, match="frozen_acquisition"):
        compare(current[0], edited)
    edited = copy.deepcopy(rounds)
    model = next(
        i for i in edited[0]["items"] if i["assessment_id"] == "descriptive:CLASSIFICATION"
    )
    model["judgment"].update(value=0.9, lower=0.8, upper=1.0)
    with pytest.raises(ValueError, match="fixed_model_mapping"):
        compare(current[0], edited)


def test_fresh_model_explanation_allowed_but_wrong_prompt_rejected(current, rounds):  # noqa: F811
    edited = copy.deepcopy(rounds)
    item = next(i for i in edited[0]["items"] if i["assessment_id"] == "descriptive:CLASSIFICATION")
    item["judgment"]["reason"] = "Independently worded explanation of the same fixed calculation."
    assert compare(current[0], edited)["root_score_range"] == 0
    for row in edited:
        row["prompt_checksum"] = digest("wrong-but-common-prompt")
    with pytest.raises(ValueError, match="prompt_binding"):
        compare(current[0], edited)

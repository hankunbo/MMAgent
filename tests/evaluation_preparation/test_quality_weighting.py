import pytest

from packages.evaluation_preparation.quality import mask_quality
from packages.evaluation_preparation.quality_materials import merge_review, upgrade
from packages.evidence_scoring.accepted import build_accepted
from packages.evidence_scoring.api import validate as validate_scores
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture, generate
from tests.tool_execution.test_geometry_qc import mask_png


def png(values, size=(6, 6)):
    return mask_png((size[1], size[0]), bytes(values))


def test_soft_qc_disconnected_is_numeric_not_case_stop():
    disc = [255] * 36
    cup = [0] * 36
    cup[7] = cup[8] = cup[35] = 255
    result = mask_quality(png(disc), png(cup))
    assert result["measurement_available"] and not result["stop_case"]
    assert result["quality_factor"] == pytest.approx((2 / 3) * (1 / 5))
    assert result["components"]["span_stability"] == pytest.approx(1 / 5)
    assert not result["masks_modified"]


def test_empty_cup_has_zero_quality_not_fake_cdr_or_case_failure():
    value = mask_quality(png([255] * 36), png([0] * 36))
    assert not value["measurement_available"]
    assert value["quality_factor"] == value["availability_score"] == 0
    assert value["next_action"] == "bounded_alternative_or_visual_observation"
    assert not value["stop_case"] and value["missing_disease_direction"] is None


def test_shapes_remain_integrity_errors():
    with pytest.raises(ValueError, match="shape"):
        mask_quality(png([0] * 36), png([0] * 25, (5, 5)))


def test_projection_keeps_value_short_and_full_diagnostic_separate(monkeypatch):
    import json

    from packages.evaluation_preparation.dynamic_materials import DynamicCaseEvidence
    from packages.evaluation_preparation.dynamic_stages import rebuild
    from packages.evaluation_preparation.quality_materials import QualityCaseEvidence
    from packages.grounding.models import canonical, digest, validate
    from packages.llm_provider.models import ObservationInput

    seed = dynamic_fixture(1).owner.current()
    body = seed.material.observations[0].model_dump(mode="json")
    body["value"] = canonical(
        dict(value=0.9653333333333334, input_checksum=seed.material.scope.input_checksum)
    ).decode()
    body["reference"]["checksum"] = digest({k: v for k, v in body.items() if k != "reference"})
    seed = rebuild(seed, seed.policy, (validate(ObservationInput, body),))
    owner = object.__new__(QualityCaseEvidence)
    owner.policies = {}
    owner.soft_quality = mask_quality(png([255] * 36), png([255] * 3 + [0] * 33))
    monkeypatch.setattr(DynamicCaseEvidence, "project", lambda self, **kwargs: (seed, {}))
    state, manifest = owner.project(run_id=seed.material.scope.run_id, config=None)
    value = state.material.observations[0].value
    assert len(value) <= 1000
    assert json.loads(value)["value"] == 0.9653333333333334
    assert manifest["quality"] == owner.soft_quality
    assert json.loads(value)["quality"]["diagnostic_checksum"] == digest(manifest["quality"])


def test_unknown_quality_revision_rejected_before_any_run():
    from packages.evaluation_preparation.quality_campaign import QualityCampaign

    with pytest.raises(ValueError, match="quality_revision"):
        QualityCampaign("not-a-case", revision="arbitrary")


@pytest.mark.parametrize("q", [0.0, 0.2, 0.6, 1.0])
def test_q_multiplies_once_and_replay_and_removal(q):
    f = dynamic_fixture(1)
    old = f.owner.current()
    factor = dict(
        value=q,
        method="test_quality",
        reason="synthetic mask quality",
        source=old.resources[0].model_dump(mode="json"),
    )
    f.owner.state = upgrade(old, factors={old.policy.items[0].id: factor})
    submission = generate(f)
    f.acceptance.submit(submission, key="quality")
    assert f.acceptance.accept(key="quality").status == "accepted"
    scores = f.acceptance.read(key="quality")[2]["scores"]
    assert scores["schema_version"] == "case-research-scoring/8.0.0"
    row = scores["reviews"][0]
    ratings = scores["quality_weighting"]["raw_observation_ratings"][row["id"]]
    assert row["weight"] == pytest.approx(
        ratings["value"] * q * row["relation"]["value"] * row["applicability"]["value"]
    )
    validate_scores(dict(state=f.owner.current(), candidate=submission.candidate), scores)
    assert scores["removals"][0]["summary"]["contributing_ids"] == []


def test_measured_seed_is_not_erased_by_visual_merge():
    seed = upgrade(dynamic_fixture(1).owner.current())
    result = merge_review(seed, {})
    assert result.policy.items == seed.policy.items
    assert result.material.observations == seed.material.observations


def test_zero_evidence_still_has_gap_coverage():
    seed = upgrade(dynamic_fixture(1, empty=True).owner.current())
    result = build_accepted(seed, None)
    assert result["root"]["summary"]["confidence"]["value"] is None
    assert result["root"]["coverage"]["slo_image_research"]["acquisition"] == 0
    row = result["requirement_evaluations"][0]
    assert row["availability_score"] == row["quality_factor"] == 0
    assert row["gap_score"] == 1 and row["disease_direction"] is None


def test_visual_fallback_is_separate_observation_not_zero_mask_measurement():
    from packages.evaluation_preparation.dynamic_stages import rebuild
    from packages.evidence_details.acceptance_models import TaskPolicyV4
    from packages.grounding.models import digest, validate
    from packages.llm_provider.models import ObservationInput

    original = dynamic_fixture(1).owner.current()
    seed = upgrade(dynamic_fixture(1, empty=True).owner.current())
    raw = seed.policy.model_dump(mode="json")
    target = "criterion:VCDR_GE_07"
    raw["requirements"][0]["id"] = target
    raw["groups"][0]["slots"] = [[target]]
    seed = rebuild(seed, validate(TaskPolicyV4, raw), ())
    value = original.material.observations[0].model_dump(mode="json")
    value.update(
        method="live_image_observation",
        unit="qualitative",
        value="Synthetic image estimate: cup/disc height approximately 0.4 to 0.6.",
    )
    value["reference"]["checksum"] = digest({k: v for k, v in value.items() if k != "reference"})
    result = merge_review(seed, {target: validate(ObservationInput, value)})
    assert len(result.policy.items) == 1
    assert result.policy.items[0].method == "live_image_observation"
    assert result.policy.items[0].quality.value == 1  # no mask penalty for a different observation
    assert result.material.observations[0].value == value["value"]
    value.update(status="ungradable", value=None)
    value["reference"]["checksum"] = digest({k: v for k, v in value.items() if k != "reference"})
    absent = merge_review(seed, {target: validate(ObservationInput, value)})
    assert absent.policy.items == ()
    assert absent.policy.requirements[0].missing_reason == "observation_ungradable"

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.inspection_views import contrast_view
from packages.evaluation_preparation.partial_observation import evaluate, schedule
from packages.grounding.models import digest, validate
from packages.llm_provider.models import PartialObservation, PartialVisionProposal, VisionProposal


def detail(**changes):
    return dict(
        coverage=0.5,
        clarity=0.8,
        basis="disc_arc",
        basis_detail="visible half rim",
        visible="superior arc",
        unseen="inferior arc",
        signal="absent_in_visible_region",
        estimate=None,
        lower=None,
        upper=None,
        **changes,
    )


def proposal(**changes):
    return PartialVisionProposal(
        scope_checksum=digest("s"),
        image_checksum=digest("i"),
        preprocessing_checksum=digest("p"),
        location=(0, 0, 32, 32),
        status="observed",
        finding="No local notch in the inspected superior arc",
        partial=detail(),
        **changes,
    )


@pytest.mark.parametrize(
    "signal,expected", [("present", 0.8), ("absent_in_visible_region", 0.4), ("indeterminate", 0.4)]
)
def test_local_positive_and_local_absence_have_distinct_coverage_factors(signal, expected):
    p = proposal()
    p = p.model_copy(update=dict(partial=p.partial.model_copy(update=dict(signal=signal))))
    result = evaluate(SimpleNamespace(status="obtained", value=p.observation_value()))
    assert result["availability_score"] == 1
    assert result["effective_quality"] == expected
    assert result["gap_score"] == pytest.approx(0.6)
    assert result["unseen_disease_probability"] is None


def test_zero_coverage_cannot_be_positive_or_measured():
    raw = detail()
    raw.update(coverage=0, signal="present")
    with pytest.raises(ValueError, match="partial_unseen"):
        validate(PartialObservation, raw)
    raw.update(signal="indeterminate", estimate=0.5, lower=0.4, upper=0.6)
    with pytest.raises(ValueError, match="partial_unseen"):
        validate(PartialObservation, raw)


def test_explicit_version_preserves_legacy_and_validates_new_schema():
    p = proposal()
    assert len(p.observation_value()) <= 1000
    assert (
        json.loads(p.observation_value())["coverage_origin"] == "vlm_estimate_not_pixel_measurement"
    )
    with pytest.raises(ValueError):
        validate(VisionProposal, p.model_dump(mode="json"))
    from packages.llm_provider.transport import strict_schema
    from packages.llm_provider.wire_compatibility import normalize

    normalize(strict_schema(PartialVisionProposal.model_json_schema()))


def test_invalid_interval_and_nonfinite_rejected():
    raw = detail()
    for a, lo, hi in ((0.4, 0.5, 0.7), (0, None, 1), (float("nan"), 0, 1)):
        with pytest.raises(ValueError):
            validate(PartialObservation, dict(raw, estimate=a, lower=lo, upper=hi))


def test_contrast_is_bounded_geometry_preserving_and_reproducible():
    from packages.dataset_adapters.reading import grayscale_image
    from packages.tool_execution.qc import grayscale_png

    raw = grayscale_image((4, 4), bytes(range(16)))
    view, policy = contrast_view(raw)
    assert contrast_view(raw) == (view, policy)
    shape, pixels = grayscale_png(view, 100)
    assert shape == (4, 4) and max(pixels) == 255 and min(pixels) == 0
    assert not policy["independent_evidence"] and not policy["synthetic_content"]


def test_component_view_is_explicit_and_never_creates_foreground():
    from packages.dataset_adapters.reading import grayscale_image
    from packages.evaluation_preparation.component_view import select_component
    from packages.tool_execution.qc import grayscale_png

    pixels = bytes(255 if i in (0, 28, 29, 36, 37) else 0 for i in range(64))
    raw = grayscale_image((8, 8), pixels)
    view, p = select_component(raw)
    shape, values = grayscale_png(view, 100)
    assert shape == (8, 8) and p["retained_fraction"] == 0.8
    assert p["excluded_pixels"] == 1 and not p["boundary_contact"]
    assert all(not v or pixels[i] for i, v in enumerate(values))
    assert select_component(raw) == (view, p)
    empty, p = select_component(grayscale_image((8, 8), bytes(64)))
    assert not any(grayscale_png(empty, 100)[1]) and p["retained_fraction"] == 0


def test_scheduler_enumerates_new_source_backed_target_without_fixed_four_list():
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

    req = dynamic_fixture(1, empty=True).owner.current().policy.requirements[0]
    assert schedule([req])[0]["action"] == "observe_partial"
    assert schedule([req], measured=[req.id])[0]["action"] == "review_measurement"
    missing = req.model_copy(update=dict(input_modalities=("OCT",)))
    assert schedule([missing])[0]["action"] == "record_input_gap"


def test_partial_merge_retains_visible_observation_and_factor_once():
    from packages.evaluation_preparation.comprehensive_materials import merge_partial
    from packages.evaluation_preparation.quality_materials import upgrade
    from packages.llm_provider.models import ObservationInput
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture, generate

    f = dynamic_fixture(1)
    seed = upgrade(f.owner.current())
    obs = seed.material.observations[0].model_copy(
        update=dict(
            value=proposal().observation_value(),
            method="live_image_observation",
            unit="qualitative",
        )
    )
    state = merge_partial(seed, {seed.policy.items[0].id: validate(ObservationInput, obs)})
    assert state.policy.items[0].quality.value == 0.4
    f.owner.state = state
    s = generate(f)
    f.acceptance.submit(s, key="partial-visible")
    assert f.acceptance.accept(key="partial-visible").status == "accepted"
    scores = f.acceptance.read(key="partial-visible")[2]["scores"]
    row = scores["reviews"][0]
    raw = scores["quality_weighting"]["raw_observation_ratings"][row["id"]]["value"]
    assert row["weight"] == pytest.approx(
        raw * 0.4 * row["relation"]["value"] * row["applicability"]["value"]
    )


def test_partial_wire_original_planner_acceptance_and_legacy_dispatch(monkeypatch, tmp_path):
    from pathlib import Path
    from threading import Event

    import tests.expert_agent.test_batch_assembly as support
    from apps.vectorengine_live_assembly import run_case_stages
    from packages.evaluation_preparation.comprehensive_materials import PartialReader
    from tests.expert_agent.batch_case_support import vision_reply

    original = support.assemble

    def install(**kwargs):
        kwargs["evidence"].wrap_owner = PartialReader
        return original(**kwargs)

    def reply(body):
        result = vision_reply(body)
        result.update(schema_version="visual-observation-candidate/2.0.0", partial=detail())
        return result

    monkeypatch.setattr(support, "assemble", install)
    f = support.assembly(
        monkeypatch,
        count=1,
        replies=[reply],
        config_path=Path("configs/agents/vectorengine_full_evidence_v1.json"),
    )
    completed, roles = run_case_stages(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
        selected_roles=("vision",),
    )
    assert roles == ("vision",) and completed[0]["domain_status"] == "accepted"
    records = f.review.acceptance.read(key=completed[0]["acceptance_key"])[0]
    assert records["candidate"]["schema_version"] == "visual-observation-candidate/2.0.0"
    assert json.loads(records["observations"][0]["value"])["coverage"] == 0.5
    assert len(f.wire.calls) == f.ports.budget.used == 1

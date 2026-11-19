"""New types reject illegal construction; these are not invented historical red tests."""

import pytest

from packages.case_score_ledger.models import ScoreDefinition, ScoreEntry, ScoreSource
from packages.contracts import Laterality
from packages.grounding.models import digest, validate


def source():
    return ScoreSource(
        scope=dict(
            case_code="CASE:ANON_ABCDEF",
            run_id="RUN:ANON_ABCDEF",
            encounter_code="ENCOUNTER:ANON_ABCDEF",
            patient_hash="sha256:" + "1" * 64,
            laterality=Laterality.UNKNOWN,
        ),
        qualification="technical",
        synthetic=True,
        correlation_group=digest("input"),
        policy_checksum=digest("policy"),
    )


def definition():
    return ScoreDefinition(
        code="fixture.logit",
        layer="model",
        metric="logit",
        definition="Synthetic raw logit; no disease probability",
        unit="logit",
        direction="higher",
    )


def entry():
    return ScoreEntry(
        definition=definition(),
        target="GOAL:SYNTHETIC",
        status="available",
        value=-3.0,
        missing_reason=None,
        source=source(),
        calibration="uncalibrated",
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(layer="disease"),
        dict(minimum=0.0, maximum=1.0),
        dict(unit="1"),
        dict(minimum=float("nan")),
        dict(maximum=float("inf")),
    ],
)
def test_definition_refuses_wrong_layer_and_fake_probability(change):
    with pytest.raises(ValueError):
        validate(ScoreDefinition, definition().model_copy(update=change))


@pytest.mark.parametrize(
    "change",
    [
        dict(value=float("nan")),
        dict(value=float("inf")),
        dict(value=-float("inf")),
        dict(value=None),
        dict(missing_reason="not_provided"),
        dict(status="unavailable"),
        dict(value=0.5, calibration="not_applicable"),
        dict(numerator=0),
        dict(schema_version="score-entry/0.0.0"),
        dict(note="private label"),
    ],
)
def test_entry_revalidates_copy_construct_and_json(change):
    value = entry()
    for changed in (
        value.model_copy(update=change),
        {**value.model_dump(mode="json"), **change},
        ScoreEntry.model_construct(**{**value.__dict__, **change})
        if "note" not in change
        else value.model_copy(update=change),
    ):
        with pytest.raises(ValueError):
            validate(ScoreEntry, changed)


def test_missing_value_is_never_default_zero():
    value = entry()
    missing = validate(
        ScoreEntry,
        value.model_copy(
            update=dict(status="unavailable", value=None, missing_reason="not_provided")
        ),
    )
    assert missing.value is None
    assert value.value == -3.0


def test_scalar_layer_cannot_take_graph_component():
    from packages.case_score_ledger.models import Component

    with pytest.raises(ValueError, match="wrong_value_layer"):
        validate(
            ScoreEntry,
            entry().model_copy(
                update=dict(
                    value=Component(
                        contributing_nodes=(),
                        rule_refs=(),
                        required_count=0,
                        resolved_count=0,
                        blocked_count=0,
                        explanations=(),
                    )
                )
            ),
        )


def test_no_model_probability_or_single_case_dice_without_sources():
    with pytest.raises(ValueError):
        ScoreDefinition(
            code="dice",
            layer="measurement",
            metric="dice",
            definition="not permitted without truth",
            unit="1",
            direction="higher",
        )
    d = ScoreDefinition(
        code="p",
        layer="disease",
        metric="probability",
        definition="No calibrated model installed",
        unit="1",
        direction="higher",
        minimum=0.0,
        maximum=1.0,
    )
    with pytest.raises(ValueError, match="calibrator_not_implemented"):
        validate(ScoreEntry, entry().model_copy(update=dict(definition=d, value=0.5)))


def test_zero_technical_denominator():
    d = ScoreDefinition(
        code="coverage",
        layer="technical",
        metric="coverage",
        definition="successful distinct workflows / registered distinct workflows",
        unit="1",
        direction="higher",
        minimum=0.0,
        maximum=1.0,
    )
    value = validate(
        ScoreEntry,
        entry().model_copy(
            update=dict(
                definition=d,
                value=None,
                status="not_applicable",
                missing_reason="zero_denominator",
                numerator=0,
                denominator=0,
                calibration="not_applicable",
            )
        ),
    )
    assert value.value is None
    with pytest.raises(ValueError):
        validate(
            ScoreEntry,
            value.model_copy(update=dict(value=0.0, status="available", missing_reason=None)),
        )


@pytest.mark.parametrize(
    "field,unit", [("area_px", "px2"), ("width_px", "px"), ("height_px", "px")]
)
def test_geometry_measurements_preserve_units(field, unit):
    import json

    from packages.case_score_ledger.service import CaseScoreService

    service = object.__new__(CaseScoreService)
    data = json.dumps(
        dict(
            schema_version="disc-geometry/1.0.0",
            input_checksum=digest("mask"),
            height=10,
            width=10,
            area_px=4,
            bbox_xyxy=[0, 0, 2, 2],
            width_px=2,
            height_px=2,
        )
    ).encode()
    d = ScoreDefinition(
        code=field,
        layer="measurement",
        metric="measurement",
        definition="Observed pixel geometry",
        unit=unit,
        direction="none",
        minimum=0.0,
    )
    assert service._decode(data, "disc-geometry/1.0.0", field, d) == (
        4.0 if field == "area_px" else 2.0
    )
    with pytest.raises(ValueError, match="measurement_unit"):
        service._decode(data, "disc-geometry/1.0.0", field, d.model_copy(update={"unit": "mm"}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_scalar_probability_map_confidence_range(value):
    from packages.tool_execution.scalar_output import ScalarOutput

    raw = dict(
        metric="segmentation_confidence", value=value, unit="1", input_checksum=digest("input")
    )
    with pytest.raises(ValueError):
        ScalarOutput(**raw)
    with pytest.raises(ValueError):
        validate(ScalarOutput, ScalarOutput.model_construct(**raw))

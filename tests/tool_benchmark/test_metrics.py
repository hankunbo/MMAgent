import pytest

from packages.tool_benchmark.metrics import failure_rate, overlap, patient_macro_mean, repeatability
from packages.tool_benchmark.models import BenchmarkError


def test_known_confusion_counts_and_measurement_error():
    result = overlap(bytes([1, 1, 0, 0]), bytes([1, 0, 1, 0]), (2, 2))
    assert result["dice"] == 0.5
    assert result["iou"] == 1 / 3
    assert result["area_error_px"] == 0  # Equal area does not imply correct spatial segmentation.


@pytest.mark.parametrize(
    "p,t,status,dice,iou,relative",
    [
        (b"\x00", b"\x00", "both_empty", 1.0, 1.0, None),
        (b"\x00", b"\x01", "empty_prediction", 0.0, 0.0, 1.0),
        (b"\x01", b"\x00", "empty_truth", 0.0, 0.0, None),
    ],
)
def test_empty_policy_explicit(p, t, status, dice, iou, relative):
    r = overlap(p, t, (1, 1))
    assert (r["status"], r["dice"], r["iou"], r["relative_area_error"]) == (
        status,
        dice,
        iou,
        relative,
    )


def test_missing_truth_not_fabricated_score():
    assert overlap(b"\x01", None, (1, 1))["dice"] is None
    assert failure_rate(0, 0) is None
    assert failure_rate(1, 4) == 0.25


@pytest.mark.parametrize("p,t,shape", [(b"\x01", b"\x01", (2, 1)), (b"\xff", b"\x01", (1, 1))])
def test_shapes_and_non_binary_labels_rejected(p, t, shape):
    with pytest.raises(BenchmarkError):
        overlap(p, t, shape)


def test_patient_and_expert_aggregation_does_not_weight_bilateral_twice():
    rows = [
        ("PAPILA", "A", "OD_A", "exp1", 1.0),
        ("PAPILA", "A", "OD_A", "exp2", 0.0),
        ("PAPILA", "A", "OS_A", "exp1", 0.5),
        ("PAPILA", "B", "OD_B", "exp1", 1.0),
    ]
    assert patient_macro_mean(rows) == 0.75
    with pytest.raises(BenchmarkError):
        patient_macro_mean(rows + [rows[0]])
    assert patient_macro_mean([]) is None


def test_repeatability_requires_actual_repeats():
    assert repeatability([b"\x00", b"\x00"])["pixel_exact"]
    assert not repeatability([b"\x00", b"\x01"])["pixel_exact"]
    with pytest.raises(BenchmarkError):
        repeatability([b"\x00"])


def test_metric_rows_cannot_change_patient_between_experts():
    with pytest.raises(BenchmarkError, match="group_mismatch"):
        patient_macro_mean([("PAPILA", "A", "OD", "exp1", 1.0), ("PAPILA", "B", "OD", "exp2", 0.0)])

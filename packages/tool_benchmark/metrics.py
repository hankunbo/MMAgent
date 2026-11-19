"""Offline binary segmentation evaluation, not an executable catalog metric runner."""

from .models import BenchmarkError

IMPLEMENTATION = "binary_overlap_area/1.0.0"


def overlap(prediction, truth, shape):
    if truth is None:
        return {
            "status": "missing_truth",
            "dice": None,
            "iou": None,
            "area_error_px": None,
            "relative_area_error": None,
        }
    if (
        len(shape) != 2
        or any(type(n) is not int or n < 1 for n in shape)
        or shape[0] * shape[1] > 16777216
    ):
        raise BenchmarkError("metric_shape_invalid")
    if len(prediction) != len(truth) or len(truth) != shape[0] * shape[1]:
        raise BenchmarkError("metric_shape_mismatch")
    if not set(prediction) <= {0, 1} or not set(truth) <= {0, 1}:
        raise BenchmarkError("metric_labels_invalid")
    predicted, expected = sum(prediction), sum(truth)
    intersection = sum(a & b for a, b in zip(prediction, truth, strict=True))
    union = predicted + expected - intersection
    return {
        "status": "both_empty"
        if not union
        else "empty_prediction"
        if not predicted
        else "empty_truth"
        if not expected
        else "computed",
        "dice": 2 * intersection / (predicted + expected) if union else 1.0,
        "iou": intersection / union if union else 1.0,
        "area_error_px": abs(predicted - expected),
        "relative_area_error": abs(predicted - expected) / expected if expected else None,
    }


def repeatability(predictions):
    if len(predictions) < 2:
        raise BenchmarkError("repeatability_requires_repeated_prediction")
    if any(len(p) != len(predictions[0]) or not set(p) <= {0, 1} for p in predictions):
        raise BenchmarkError("repeatability_shape_or_labels_invalid")
    return {
        "repetitions": len(predictions),
        "pixel_exact": all(p == predictions[0] for p in predictions[1:]),
    }


def failure_rate(failures, attempted):
    if type(failures) is not int or type(attempted) is not int or not 0 <= failures <= attempted:
        raise BenchmarkError("failure_denominator_invalid")
    return failures / attempted if attempted else None


def patient_macro_mean(rows):
    """Expert mean within image, image mean within patient, then patient mean.

    Each row is (dataset, group, sample, expert, score); missing scores excluded,
    never silently converted into successes. Failure rate is reported separately.
    """
    patients, samples, seen, identities = {}, {}, set(), {}
    for dataset, group, sample, expert, score in rows:
        identity = (dataset, sample, expert)
        sample_key = (dataset, sample)
        if sample_key in identities and identities[sample_key] != group:
            raise BenchmarkError("metric_sample_group_mismatch")
        identities[sample_key] = group
        if identity in seen:
            raise BenchmarkError("duplicate_metric_row")
        seen.add(identity)
        if score is None:
            continue
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise BenchmarkError("metric_score_invalid")
        samples.setdefault((dataset, group, sample), []).append(score)
    for (dataset, group, _), scores in samples.items():
        patients.setdefault((dataset, group), []).append(sum(scores) / len(scores))
    return sum(sum(v) / len(v) for v in patients.values()) / len(patients) if patients else None

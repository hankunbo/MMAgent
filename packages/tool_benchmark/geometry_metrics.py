"""Development mask metrics in native pixels; no tool registration or clinical rules."""

import math

from .metrics import overlap
from .models import BenchmarkError

VERSION = "native-mask-geometry/1.0.0"


def boundary(mask, shape):
    h, w = shape
    return [
        (i // w, i % w)
        for i, v in enumerate(mask)
        if v
        and (
            i // w in (0, h - 1)
            or i % w in (0, w - 1)
            or not (mask[i - w] and mask[i + w] and mask[i - 1] and mask[i + 1])
        )
    ]


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def geometry(prediction, truth, shape):
    result = overlap(prediction, truth, shape)
    result.update(boundary_status="empty_mask", assd_px=None, hd95_px=None)
    a, b = boundary(prediction, shape), boundary(truth, shape)
    if a and b:
        if len(a) * len(b) > 4000000:
            result["boundary_status"] = "pair_budget_exceeded"
        else:
            forward = [math.sqrt(min((y - v) ** 2 + (x - u) ** 2 for v, u in b)) for y, x in a]
            backward = [math.sqrt(min((y - v) ** 2 + (x - u) ** 2 for v, u in a)) for y, x in b]
            distances = forward + backward
            result.update(
                boundary_status="computed",
                assd_px=sum(distances) / len(distances),
                hd95_px=percentile(distances, 0.95),
            )
    result["area_fraction_error"] = result["area_error_px"] / (shape[0] * shape[1])
    return result


def ratios(labels, shape):
    if len(labels) != shape[0] * shape[1] or not set(labels) <= {0, 1, 2}:
        raise BenchmarkError("ratio_labels_invalid")
    disc = [i for i, v in enumerate(labels) if v > 0]
    cup = [i for i, v in enumerate(labels) if v == 2]
    if not cup or not disc:
        return dict(vertical_cdr=None, area_cdr=None)

    def height(indices):
        return max(i // shape[1] for i in indices) - min(i // shape[1] for i in indices) + 1

    return dict(vertical_cdr=height(cup) / height(disc), area_cdr=len(cup) / len(disc))


def compare(prediction, truth, shape):
    if not set(prediction) <= {0, 1, 2} or not set(truth) <= {0, 1, 2}:
        raise BenchmarkError("comparison_labels_invalid")
    result = {}
    for target, codes in (("disc", {1, 2}), ("cup", {2}), ("rim", {1})):
        result[target] = geometry(
            bytes(v in codes for v in prediction), bytes(v in codes for v in truth), shape
        )
    a, b = ratios(prediction, shape), ratios(truth, shape)
    result["ratios"] = {
        k: dict(
            predicted=a[k],
            reference=b[k],
            absolute_error=abs(a[k] - b[k]) if a[k] is not None and b[k] is not None else None,
        )
        for k in a
    }
    result["qc_flags"] = [
        "empty_predicted_" + k
        for k, codes in (("cup", {2}), ("rim", {1}), ("disc", {1, 2}))
        if not any(v in codes for v in prediction)
    ]
    if all(prediction):
        result["qc_flags"].append("full_image_disc")
    return result

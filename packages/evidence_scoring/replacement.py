"""Existing-mask replacement only; no inference and no inherited new judgments."""

import base64

from apps.glaucoma_rim_radial import radial_profile
from packages.object_store import checksum
from packages.tool_execution.qc import grayscale_png

SOURCES = {
    "original_disc": ("ARTIFACT:SLO_SEGMAN_DISC", "segman_disc"),
    "alternative_disc": ("ARTIFACT:SLO_DISC_MASK", "disc_mask"),
    "fixed_cup": ("ARTIFACT:SLO_SEGMAN_CUP", "segman_cup"),
}


def _mask(case, kind, image):
    assets = [a for a in case["artifacts"] if a["artifact_type_code"] == kind]
    if len(assets) != 1:
        raise ValueError("replacement_ambiguous_asset")
    a = assets[0]
    if a["scope"]["case_code"] != case["case_code"] or a["qc_status"] != "PASS":
        raise ValueError("replacement_scope_or_qc")
    if a["laterality"] != case["input"]["laterality"]:
        raise ValueError("replacement_laterality")
    encoded = case["images"][image]
    prefix = "data:image/png;base64,"
    if not encoded.startswith(prefix) or len(encoded) > 2_800_000:
        raise ValueError("replacement_mask_bound")
    data = base64.b64decode(encoded[len(prefix) :], validate=True)
    if checksum(data) != a["checksum"]:
        raise ValueError("replacement_mask_checksum")
    shape, mask = grayscale_png(data, 1_048_576)
    if list(shape) != case["input"]["dimensions"][:2] or list(shape) != a["dimensions"][:2]:
        raise ValueError("replacement_shape")
    # Graph ancestry must trace to this exact input, not merely the same patient.
    by_id = {x["asset_code"]: x for x in case["artifacts"]}
    pending, seen = list(a["derived_from"]), set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        if key in by_id:
            pending.extend(by_id[key]["derived_from"])
    if case["input"]["asset_code"] not in seen:
        raise ValueError("replacement_image_ancestry")
    return shape, mask, {"code": a["asset_code"], "checksum": a["checksum"]}


def metrics(shape, disc, cup):
    radial = radial_profile(shape, disc, cup)
    w = shape[1]

    def height(mask):
        ys = [i // w for i, v in enumerate(mask) if v]
        return max(ys) - min(ys) + 1

    return {
        "vertical_cup_disc_ratio": height(cup) / height(disc),
        "minimum_radial_ratio": radial["minimum"]["ratio"],
        "disc_pixels": sum(bool(v) for v in disc),
        "cup_pixels": sum(bool(v) for v in cup),
    }


def build(base):
    case = base["case_evidence"]
    result = {
        "schema_version": "existing-mask-replacement/1.0.0",
        "substitution": "SegMAN-B disc -> SLOctolyzer disc; fixed SegMAN-B cup",
        "status": "inputs_unavailable",
        "source_refs": {},
        "original": None,
        "alternative": None,
        "deltas": None,
        "error": None,
        "disease_score_after_replacement": None,
        "next_action": "替换测量须独立判读，不自动沿用原e与可靠性。",
    }
    if any(image not in case.get("images", {}) for _, image in SOURCES.values()):
        result["error"] = "replacement_embedded_inputs_missing"
        return result
    masks = {}
    for name, (kind, image) in SOURCES.items():
        shape, masks[name], ref = _mask(case, kind, image)
        result["source_refs"][name] = ref
    result["original"] = metrics(shape, masks["original_disc"], masks["fixed_cup"])
    try:
        result["alternative"] = metrics(shape, masks["alternative_disc"], masks["fixed_cup"])
    except ValueError as error:
        if str(error) not in {
            "rim_masks_not_nested",
            "rim_disc_centroid_outside_cup",
            "rim_mask_touches_image_edge",
            "rim_boundary_not_found",
        }:
            raise
        result.update(
            status="qc_rejected",
            error=str(error),
            next_action="两工具组合未通过几何QC，不裁剪杯掩膜制造通过；保留原结果。",
        )
        return result
    result["status"] = "measured_requires_new_interpretation"
    result["deltas"] = {k: result["alternative"][k] - v for k, v in result["original"].items()}
    return result

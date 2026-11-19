"""Measure existing A/V masks, never infer absent OCT/IOP or run a model."""

import argparse
import io
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage.morphology import skeletonize

from Algorithm_Tool.av_vascular_metrics.contract import POLICY, VERSION, checksum, validate
from Algorithm_Tool.slo_observations.runner import segments


def read(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to("/opt/data/test") or not 0 < path.stat().st_size <= 32_000_000:
        raise ValueError("av_input_path_or_size")
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "L" or not 16 <= min(image.size) or max(image.size) > 4096 or math.prod(image.size) > 4_000_000:
            raise ValueError("av_binary_png_required")
        a = np.array(image)
    if not np.isin(a, [0, 255]).all():
        raise ValueError("av_nonbinary_input")
    import hashlib

    return a > 0, {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "dimensions_hw": list(a.shape)}


def density(mask, skeleton, roi, diameter):
    den, area, length = int(roi.sum()), int((mask & roi).sum()), int((skeleton & roi).sum())
    width = area/length if length else None
    return {"roi_pixels": den, "vessel_pixels": area, "skeleton_pixels": length,
            "area_density": area/den if den else None, "skeleton_density": length/den if den else None,
            "width_proxy_pixels": width, "width_proxy_disc_ratio": width/diameter if width is not None else None}


def fractal(skeleton):
    boxes = []
    k = 4
    while k <= min(skeleton.shape)//4:
        count = int(np.count_nonzero(np.add.reduceat(np.add.reduceat(skeleton.astype(np.int32), np.arange(0,skeleton.shape[0],k), axis=0), np.arange(0,skeleton.shape[1],k), axis=1)))
        if count:
            boxes.append({"size_pixels": k, "occupied": count})
        k *= 2
    if len(boxes) < 3:
        return {"value": None, "r_squared": None, "boxes": boxes, "reason": "fewer_than_three_occupied_scales"}
    x, y = np.log([1/b["size_pixels"] for b in boxes]), np.log([b["occupied"] for b in boxes])
    slope, intercept = np.polyfit(x,y,1)
    residual, total = float(np.sum((y-(slope*x+intercept))**2)), float(np.sum((y-y.mean())**2))
    return {"value": max(0.0,float(slope)), "r_squared": max(0.0,1-residual/total) if total else 1.0,
            "boxes": boxes, "reason": None}


def tortuosity(skeleton, diameter):
    minimum = max(10.0, .1*diameter)
    retained, short, closed = [], 0, 0
    for points in segments(skeleton):
        arc = sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        chord = math.dist(points[0],points[-1])
        if chord == 0:
            closed += 1
        elif arc < minimum:
            short += 1
        else:
            retained.append({"arc_pixels":arc, "chord_pixels":chord, "ratio":arc/chord})
    if len(retained) > 10000:
        raise ValueError("av_segment_budget")
    values = [r["ratio"] for r in retained]
    return {"count":len(values), "excluded_short":short, "excluded_closed":closed,
            "minimum_length_pixels":minimum, "segments":retained,
            "mean":statistics.mean(values) if values else None,
            "median":statistics.median(values) if values else None,
            "min":min(values) if values else None, "max":max(values) if values else None}


def compute(artery, vein, disc, bindings):
    if artery.shape != vein.shape or artery.shape != disc.shape:
        raise ValueError("av_input_alignment")
    if not 16 <= disc.sum() < disc.size/2 or label(disc, np.ones((3,3)))[1] != 1:
        raise ValueError("av_single_disc_required")
    ys, xs = np.nonzero(disc)
    x,y = float(xs.mean()),float(ys.mean())
    d = math.sqrt(4*int(disc.sum())/math.pi)
    yy,xx = np.indices(disc.shape)
    radius = np.hypot(xx-x,yy-y)
    annulus = (radius >= d) & (radius < 1.5*d)
    overlap = artery & vein
    classes = {}
    for name, mask in [("artery", artery & ~overlap), ("vein", vein & ~overlap)]:
        sk = skeletonize(mask, method="zhang")
        classes[name] = {"whole_image":density(mask,sk,np.ones_like(mask),d),
                         "annulus":density(mask,sk,annulus,d),
                         "tortuosity":tortuosity(sk,d), "skeleton_fractal":fractal(sk)}
    v = {"schema_version":VERSION, "policy":POLICY, "policy_checksum":checksum(POLICY),
         "input_bindings":bindings, "overlap_pixels":int(overlap.sum()), "research_only":True,
         "disc":{"area_pixels":int(disc.sum()), "diameter_pixels":d, "center_xy":[x,y],
                 "annulus_pixels":int(annulus.sum()),
                 "annulus_fraction_in_image":min(1.0,float(annulus.sum())/(math.pi*(2.25-1)*d*d))},
         "classes":classes}
    return validate({**v,"checksum":checksum(v)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for port in ("artery", "vein", "disc"):
        p.add_argument("--"+port+"-mask", type=Path, required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    output = a.output.resolve()
    if not output.is_relative_to("/opt/data/test") or output.exists():
        raise ValueError("av_new_output_required")
    arrays, bindings = {}, {}
    for port in ("artery_mask","vein_mask","disc_mask"):
        arrays[port], bindings[port] = read(getattr(a,port))
    result = compute(*arrays.values(),bindings)
    with output.open("x") as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2)


if __name__ == "__main__":
    main()

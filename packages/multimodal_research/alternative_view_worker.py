"""Deterministic source-pixel view; run only in the existing image environment.

This does not estimate anatomy, a disease direction, or a missing measurement.
"""

import hashlib
import io
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

METHOD = "original_plus_registered_disc_roi_rgb_green/1.0.0"
TILED_METHOD = "original_plus_full_frame_quadrants_rgb_green/1.0.0"


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def geometry(shape, analysis_shape, bbox):
    if (
        len(shape) != 2
        or len(analysis_shape) != 2
        or len(bbox) != 4
        or any(type(v) is not int for v in (*shape, *analysis_shape, *bbox))
        or not all(1 <= v <= 16384 for v in (*shape, *analysis_shape))
    ):
        raise ValueError("alternative_view_geometry")
    h, w = shape
    ah, aw = analysis_shape
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= aw or not 0 <= y0 < y1 <= ah:
        raise ValueError("alternative_view_bbox")
    cx, cy = (x0 + x1) * w / (2 * aw), (y0 + y1) * h / (2 * ah)
    side = 3 * max((x1 - x0) * w / aw, (y1 - y0) * h / ah)
    return [
        max(0, math.floor(cx - side / 2)),
        max(0, math.floor(cy - side / 2)),
        min(w, math.ceil(cx + side / 2)),
        min(h, math.ceil(cy + side / 2)),
    ]


def render(raw, proof):
    if proof["method"] == TILED_METHOD:
        return render_tiles(raw, proof)
    if proof["method"] != METHOD or sha(raw) != proof["source_checksum"]:
        raise ValueError("alternative_view_source")
    box = geometry(proof["source_shape"], proof["analysis_shape"], proof["disc_bbox_xyxy"])
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        original = opened.convert("RGB")
    if [original.height, original.width] != proof["source_shape"]:
        raise ValueError("alternative_view_source_shape")
    crop = original.crop(box)
    green = crop.getchannel("G")
    panel = 768
    view = Image.new("RGB", (panel * 3, panel + 48), "#171717")
    drawing = ImageDraw.Draw(view)
    positions = []
    for i, (label, source) in enumerate(
        (
            ("ORIGINAL RGB", original),
            ("DISC ROI RGB", crop),
            ("SAME ROI GREEN CHANNEL (not red-free)", green.convert("RGB")),
        )
    ):
        display = ImageOps.contain(source, (panel, panel), Image.Resampling.LANCZOS)
        left = i * panel + (panel - display.width) // 2
        top = 48 + (panel - display.height) // 2
        view.paste(display, (left, top))
        drawing.text((i * panel + 10, 16), label, fill="white")
        positions.append([left, top, left + display.width, top + display.height])
    stream = io.BytesIO()
    view.save(stream, format="PNG")
    content = stream.getvalue()
    if len(content) > 4194304:
        raise ValueError("alternative_view_size")
    metadata = dict(
        method=METHOD,
        source_checksum=proof["source_checksum"],
        source_shape=proof["source_shape"],
        native_checksum=proof["native_checksum"],
        roi_source_xyxy=box,
        display_panels_xyxy=positions,
        crop_pixels_checksum=sha(crop.tobytes()),
        green_pixels_checksum=sha(green.tobytes()),
        checksum=sha(content),
        width=view.width,
        height=view.height,
        modality="color_fundus",
        indices=[0],
        available_members=1,
        inspected_fraction=1.0,
        transform=dict(
            method=METHOD,
            roi_source_xyxy=box,
            display_panels_xyxy=positions,
            localization="owner_verified_disc_bbox_fixed_three_diameters",
            channel="unchanged_RGB_green_not_acquired_red_free",
            display_resize="Pillow_LANCZOS_no_sharpening_no_histogram_equalization",
        ),
        limits=[
            "Original RGB remains visible; ROI and channel share one source "
            "and are not independent evidence.",
            "Green channel is not acquired red-free photography; "
            "no red-free clinical calibration is claimed.",
            "Disc mask only localizes the crop; "
            "it does not segment RNFL or restore obscured tissue.",
            "Display interpolation adds no source resolution "
            "and cannot establish physical thickness.",
        ],
    )
    return content, metadata


def render_tiles(raw, proof):
    """Cover every source pixel without pretending a failed mask localizes anatomy."""
    if proof["method"] != TILED_METHOD or sha(raw) != proof["source_checksum"]:
        raise ValueError("alternative_view_source")
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        original = opened.convert("RGB")
    h, w = original.height, original.width
    if [h, w] != proof["source_shape"] or not 2 <= min(h, w) <= max(h, w) <= 16384:
        raise ValueError("alternative_view_source_shape")
    mx, my = w // 2, h // 2
    boxes = [[0, 0, mx, my], [mx, 0, w, my], [0, my, mx, h], [mx, my, w, h]]
    tiles = [original.crop(box) for box in boxes]
    sources = [original, original.getchannel("G").convert("RGB"), *tiles]
    labels = [
        "ORIGINAL RGB",
        "ORIGINAL GREEN (not red-free)",
        "TOP LEFT",
        "TOP RIGHT",
        "BOTTOM LEFT",
        "BOTTOM RIGHT",
    ]
    # A finite deterministic display-size fallback, never a repeated model call.
    for panel in (768, 640, 512):
        view = Image.new("RGB", (panel * 3, (panel + 40) * 2), "#171717")
        drawing, positions = ImageDraw.Draw(view), []
        for i, (label, source) in enumerate(zip(labels, sources, strict=True)):
            x, y = (i % 3) * panel, (i // 3) * (panel + 40)
            shown = ImageOps.contain(source, (panel, panel), Image.Resampling.LANCZOS)
            left, top = x + (panel - shown.width) // 2, y + 40 + (panel - shown.height) // 2
            view.paste(shown, (left, top))
            drawing.text((x + 8, y + 12), label, fill="white")
            positions.append([left, top, left + shown.width, top + shown.height])
        stream = io.BytesIO()
        view.save(stream, format="PNG")
        content = stream.getvalue()
        if len(content) <= 4194304:
            break
    else:
        raise ValueError("alternative_view_size")
    metadata = dict(
        method=TILED_METHOD,
        source_checksum=proof["source_checksum"],
        source_shape=proof["source_shape"],
        native_checksum=proof["native_checksum"],
        native_localization_used=False,
        tile_source_xyxy=boxes,
        display_panels_xyxy=positions,
        tile_pixels_checksums=[sha(t.tobytes()) for t in tiles],
        checksum=sha(content),
        width=view.width,
        height=view.height,
        modality="color_fundus",
        indices=[0],
        available_members=1,
        inspected_fraction=1.0,
        transform=dict(
            method=TILED_METHOD,
            tile_source_xyxy=boxes,
            display_panels_xyxy=positions,
            localization="none_full_frame_grid",
            channel="RGB_and_unchanged_green",
            panel_pixels=panel,
            resize="Pillow_LANCZOS",
        ),
        limits=[
            "All panels share one image, not independent evidence.",
            "Full-frame quadrants are not anatomical regions or a successful segmentation.",
            "Green channel is not acquired red-free photography.",
            "Interpolation restores neither hidden tissue nor physical measurements.",
        ],
    )
    return content, metadata


def main():
    source, spec, output = map(Path, sys.argv[1:])
    if any(p.is_symlink() for p in (source, spec, output, *output.parents)):
        raise ValueError("alternative_view_symlink")
    content, metadata = render(source.read_bytes(), json.loads(spec.read_bytes()))
    output.mkdir(mode=0o700)
    with (output / "view.png").open("xb") as stream:
        stream.write(content)
    with (output / "view.json").open("x") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(checksum=metadata["checksum"], bytes=len(content), model_calls=0)))


if __name__ == "__main__":
    main()

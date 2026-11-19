"""Isolated display-only transform over approved payloads; no model, labels or network.

Run with the existing image environment, not inside the API environment. Source
indices remain indices, never anatomical coordinates or a segmented thickness.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def selected(count, limit=12):
    return (
        sorted({round(i * (count - 1) / (min(limit, count) - 1)) for i in range(min(limit, count))})
        if count > 1
        else [0]
    )


def generate(preparation, output, profile="overview-v1"):
    if profile not in {"overview-v1", "detail-v1"}:
        raise ValueError("view_profile_unknown")
    detailed = profile == "detail-v1"
    value = json.loads(Path(preparation).read_bytes())
    qualified = value["manifest"].get("adapter_version") == "multimodal-adapters/1.1.0"
    files = {f["id"]: f for f in value["payload_files"]}
    out = Path(output)
    out.mkdir(mode=0o700, exist_ok=False)

    def payload(key):
        f = files[key]
        expected_path = "payloads/" + sha(key.encode())[7:] + ".bin"
        if f["path"] != expected_path:
            raise ValueError("view_payload_path")
        path = Path(preparation).parent / expected_path
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("view_symlink")
        raw = path.read_bytes()
        if len(raw) != f["byte_size"] or sha(raw) != f["checksum"]:
            raise ValueError("view_payload_changed")
        return raw

    results = []
    for c in value["manifest"]["components"]:
        if c["kind"] not in {"image", "image_series", "volume", "map"} or not c["payload_ids"]:
            continue
        if c["status"] not in {"obtained", "partial"}:
            continue
        ids = c["payload_ids"]
        limits = list(c["semantic_limits"])
        if c["kind"] in {"map", "volume"}:
            a = np.load(io.BytesIO(payload(ids[0])), allow_pickle=False)
            if list(a.shape) != c["shape"] or a.dtype.kind not in "fiu":
                raise ValueError("view_array_shape")
            valid = a[np.isfinite(a)]
            thickness = qualified and c["concept"] == "rnflt" and c["unit"] == "um"
            if thickness:
                valid = valid[valid >= 0]
            if not valid.size:
                results.append(
                    dict(component=c["id"], status="ungradable", reason="no_finite_pixels")
                )
                continue
            low, high = float(np.min(valid)), float(np.max(valid))
            if c["kind"] == "volume":
                indices = selected(a.shape[0], 24 if detailed else 12)
                count = a.shape[0]
                arrays = [a[i] for i in indices]
                limits.append("axis_0_source_slices_not_anatomical_registration")
            else:
                indices, count, arrays = [0], 1, [a]
            images = [
                Image.fromarray(
                    np.uint8(np.clip(np.nan_to_num((v - low) / max(high - low, 1e-9)), 0, 1) * 255)
                ).convert("RGB")
                for v in arrays
            ]
            if thickness:
                # Nonphysical negative codes remain visible as magenta, never tissue/zero.
                pixels = np.asarray(images[0]).copy()
                pixels[~np.isfinite(a) | (a < 0)] = (255, 0, 255)
                images[0] = Image.fromarray(pixels)
                limits.append("magenta_negative_or_nonfinite_not_measured_tissue")
            transform = dict(
                kind="global_finite_minmax_display",
                minimum=low,
                maximum=high,
                diagnostic_scale=False,
            )
            if thickness:
                transform.update(
                    unit="um",
                    invalid_display="magenta",
                    normative_probability=False,
                    scalar_scale="linear_valid_thickness_not_normative_color_code",
                )
        else:
            count = len(ids)
            indices = (
                selected(count, 24 if detailed else 12) if c["kind"] == "image_series" else [0]
            )
            images = []
            for i in indices:
                with Image.open(io.BytesIO(payload(ids[i]))) as im:
                    im.load()
                    if im.width * im.height > 40000000:
                        raise ValueError("view_image_bound")
                    images.append(im.convert("RGB"))
            if c["kind"] == "image_series":
                indices = [c["member_indices"][i] for i in indices]
                count = c["expected_members"]
            transform = dict(kind="RGB_thumbnail_no_crop")
        if len(images) == 1:
            image = ImageOps.contain(images[0], (2048, 2048) if detailed else (1024, 1024))
        else:
            cell_size = 320 if qualified and detailed else 512 if detailed else 256
            height = 240 if qualified and detailed else cell_size + 24
            image = Image.new(
                "RGB", (4 * cell_size, math.ceil(len(images) / 4) * height), "#181818"
            )
            draw = ImageDraw.Draw(image)
            for j, im in enumerate(images):
                cell = (
                    ImageOps.contain(im, (cell_size - 12, height - 28))
                    if qualified
                    else ImageOps.contain(im, (cell_size, cell_size))
                )
                x, y = (j % 4) * cell_size, (j // 4) * height
                image.paste(
                    cell,
                    (
                        x + (cell_size - cell.width) // 2,
                        y + ((height - 28 if qualified else cell_size) - cell.height) // 2,
                    ),
                )
                draw.text(
                    (x + 4, y + (height - 20 if qualified else cell_size + 3)),
                    "source index " + str(indices[j]),
                    fill="white",
                )
            limits.append("sampled_panels_not_complete_volume_inspection")
        if detailed:
            transform["profile"] = profile
            transform["source_shapes"] = [[im.height, im.width] for im in images]
            transform["panel_pixels"] = (320 if qualified else 512) if len(images) > 1 else None
            transform["anatomical_registration"] = False
            if qualified:
                transform["layout_version"] = "native-aspect-gutters/1.0.0"
        raw = io.BytesIO()
        image.save(raw, format="PNG")
        data = raw.getvalue()
        if len(data) > 4194304:
            raise ValueError("view_output_bound")
        filename = sha(c["id"].encode())[7:31] + ".png"
        with (out / filename).open("xb") as stream:
            stream.write(data)
        results.append(
            dict(
                component=c["id"],
                status="obtained",
                file=filename,
                checksum=sha(data),
                width=image.width,
                height=image.height,
                indices=indices,
                available_members=count,
                available_payloads=len(ids),
                inspected_fraction=len(indices) / count,
                input_refs=[dict(id=k, checksum=files[k]["checksum"]) for k in ids],
                transform=transform,
                limits=limits,
                unit=c["unit"],
                modality=c["modality"],
            )
        )
    result = dict(
        schema_version="multimodal-inspection-views/1.2.0"
        if qualified and detailed
        else "multimodal-inspection-views/1.1.0"
        if detailed
        else "multimodal-inspection-views/1.0.0",
        input_checksum=value["manifest"]["checksum"],
        preparation_checksum=sha(Path(preparation).read_bytes()),
        views=results,
        model_executions=0,
        labels_read=False,
    )
    with (out / "views.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preparation")
    parser.add_argument("output")
    parser.add_argument("--profile", choices=("overview-v1", "detail-v1"), default="overview-v1")
    args = parser.parse_args()
    generate(args.preparation, args.output, args.profile)

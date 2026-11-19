"""Deterministic material interpretation and explicit sampled views; no diagnosis."""

import io
import json
import math
from collections import defaultdict

from PIL import Image, ImageStat

from packages.dataset_adapters.multimodal.formats import npy
from packages.grounding.models import digest
from packages.object_store import checksum
from packages.tool_execution.process import read_file

UNITS = {
    "year": "year",
    "years": "year",
    "um": "µm",
    "μm": "µm",
    "µm": "µm",
    "mm": "mm",
    "mmHg": "mmHg",
    "dB": "dB",
    "D": "D",
}


def unit_semantics(component):
    """Normalize explicit unit spelling only; never derive units from value ranges."""
    original = component.unit
    normalized = UNITS.get(original)
    categorical = original is not None and "=" in original
    return dict(
        source_unit=original,
        unit_basis=component.unit_basis,
        normalized_unit=normalized,
        normalized_value=component.value if normalized else None,
        unit_state="source_declared_spelling_normalized"
        if normalized
        else "source_declared_code_mapping_not_physical_unit"
        if categorical
        else "source_declared_unmapped"
        if original
        else "unknown",
        normalization_rule="unit-spelling/1.0.0" if normalized else None,
        transform="identity" if normalized else None,
        category_mapping=original if categorical else None,
    )


def indices(count, limit):
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def _array_image(array, index):
    # Axis 0 is a storage axis, never an asserted anatomical axis.
    shape = array.shape
    if len(shape) == 3:
        count, h, w = shape
        size = int(array.dtype[2:])
        block = array.data[index * h * w * size : (index + 1) * h * w * size]
        array = type(array)(array.raw, block, (h, w), array.dtype)
    elif len(shape) == 2:
        h, w = shape
    else:
        raise ValueError("qualitative_array_view_dimensions")
    if h * w > 16777216:
        raise ValueError("qualitative_array_view_limit")
    values = list(array.numbers())
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        raise ValueError("qualitative_array_no_finite_values")
    low, high = min(finite), max(finite)
    pixels = bytes(
        0 if not math.isfinite(v) or high == low else round(255 * (v - low) / (high - low))
        for v in values
    )
    return Image.frombytes("L", (w, h), pixels), dict(
        kind="linear_display_only",
        source_min=low,
        source_max=high,
        constant=low == high,
        nonfinite_rendered_black=len(values) - len(finite),
        physical_scale="not_inferred",
        axis_semantics="source_index_only",
    )


def views(directory, component, payloads, config):
    candidates = []
    if component.kind in {"scalar", "vector"} or not component.payload_ids:
        return [], []
    if component.parse_status == "invalid_payload":
        raise ValueError("qualitative_source_pixel_decode_failed")
    selected = indices(len(component.payload_ids), config.views_per_component)
    for position in selected:
        p = payloads[component.payload_ids[position]]
        raw = read_file(directory / p.path, p.byte_size)
        if checksum(raw) != p.checksum:
            raise ValueError("qualitative_payload_changed")
        if p.media_type.startswith("image/"):
            im = Image.open(io.BytesIO(raw))
            im.load()
            candidates.append(
                (
                    im,
                    p,
                    dict(
                        payload_position=position,
                        slice_index=None,
                        source_member_index=component.member_indices[position]
                        if component.member_indices
                        else None,
                        transform="color_decode",
                        original_size=list(im.size),
                    ),
                )
            )
        elif p.media_type == "application/x-npy":
            a = npy(raw)
            selected_slices = (
                indices(a.shape[0], config.views_per_component) if len(a.shape) == 3 else [0]
            )
            for index in selected_slices:
                im, transform = _array_image(a, index)
                candidates.append(
                    (
                        im,
                        p,
                        dict(
                            payload_position=position,
                            slice_index=index if len(a.shape) == 3 else None,
                            source_member_index=None,
                            transform=transform,
                            original_size=list(im.size),
                        ),
                    )
                )
        else:
            raise ValueError("qualitative_visual_payload_unsupported")
    descriptions, images = [], []
    for i, (im, p, source) in enumerate(candidates):
        im = im.convert("RGB")
        im.thumbnail((config.image_edge, config.image_edge), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        raw = stream.getvalue()
        descriptions.append(
            dict(
                view_id=f"V{i + 1}",
                payload_id=p.id,
                payload_checksum=p.checksum,
                view_checksum=checksum(raw),
                displayed_size=list(im.size),
                source=source,
                intensity_summary=dict(
                    mean_rgb=ImageStat.Stat(im).mean,
                    interpretation="display_statistics_not_clinical_quality",
                ),
            )
        )
        images.append(raw)
    return descriptions, images


def deterministic(packet, scope, directory):
    payloads = {p.id: p for p in packet.payloads}
    components = []
    duplicates = defaultdict(list)
    selected = set(scope.component_ids)
    for c in packet.components:
        if c.id not in selected:
            continue
        source = c.model_dump(mode="json")
        # Raw labels/notes, file paths, dataset selectors and restricted references
        # are never imported into this data-level result.
        row = dict(
            component_id=c.id,
            component_checksum=digest(c),
            source_declared=source,
            type_modality=dict(
                kind=c.kind,
                modality=c.modality,
                basis="source_declared",
                actual_content_confirmation="pending"
                if c.payload_ids and c.kind != "vector"
                else "not_applicable",
            ),
            content_regions=[],
            visibility=[],
            semantics=dict(
                concept=c.concept,
                value=c.value,
                source_unit=c.unit,
                normalized_unit=None,
                normalized_value=None,
                unit_state="source_declared" if c.unit else "unknown",
                unit_basis=c.unit_basis,
                inference_promoted_to_fact=False,
                semantic_limits=list(c.semantic_limits),
            ),
            execution=dict(structural="completed", visual="not_requested"),
            interpretation_state="source_declared_not_independently_confirmed",
        )
        row["semantics"].update(unit_semantics(c))
        if c.kind == "vector" and c.payload_ids:
            p = payloads[c.payload_ids[0]]
            raw = read_file(directory / p.path, p.byte_size)
            if p.media_type == "application/x-npy":
                a = npy(raw)
                finite, total = a.finite_counts()
                row["semantics"]["numeric_inventory"] = dict(
                    shape=list(a.shape),
                    finite=finite,
                    total=total,
                    nonfinite=total - finite,
                    index_meaning="source_declared_only",
                )
            elif p.media_type == "application/json":
                row["semantics"]["saved_values"] = json.loads(raw)
        if c.payload_ids:
            duplicates[tuple(payloads[p].checksum for p in c.payload_ids)].append(c.id)
        components.append(row)
    backgrounds = []
    for b in packet.background:
        if b.id in scope.background_ids:
            backgrounds.append(
                dict(
                    source_declared=b.model_dump(mode="json"),
                    semantic_state="pending_missing_marker_confirmation"
                    if b.value == "/"
                    else b.meaning_status,
                    normalized_value=None if b.value == "/" else b.value,
                    limitation="slash retained in source, not interpreted as a clinical value"
                    if b.value == "/"
                    else "no categorical code mapping inferred",
                )
            )
    return dict(
        components=components,
        background=backgrounds,
        relations=dict(
            eye_visit_groups=[
                dict(
                    v.model_dump(mode="json"),
                    component_ids=[
                        i for i in v.component_ids if i in selected or i in scope.background_ids
                    ],
                )
                for v in packet.visits
                if set(v.component_ids) & (selected | set(scope.background_ids))
            ],
            identical_payload_groups=[
                dict(component_ids=ids, relation="identical_payload_bytes_not_same_clinical_event")
                for ids in duplicates.values()
                if len(ids) > 1
            ],
            constraints=[
                "equal_visit_numbers_do_not_prove_same_date",
                "sample_identity_does_not_prove_cross_dataset_patient_identity",
                "derived_measurements_not_independent_examinations",
            ],
        ),
        scope_excluded=[c.id for c in packet.components if c.id not in selected],
    )

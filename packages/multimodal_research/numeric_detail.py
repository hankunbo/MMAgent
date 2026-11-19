"""Bounded, lossless vector chunks; map coordinates are not anatomical sectors."""

import json
import math

from packages.grounding.models import canonical, digest

from . import material


def expand(ledger, observations, manifest, payloads):
    rows, values = [], dict(observations)
    for row in ledger:
        component = next((c for c in manifest.components if c.id == row["component"]), None)
        if (
            component is None
            or component.kind not in {"vector", "map"}
            or not component.payload_ids
            or component.status not in {"obtained", "partial"}
        ):
            rows.append(row)
            continue
        raw = payloads[component.payload_ids[0]]
        if raw.startswith(b"\x93NUMPY"):
            from packages.dataset_adapters.multimodal.formats import npy

            numbers = list(npy(raw).numbers())
        else:
            decoded = json.loads(raw)
            numbers = decoded["values"] if isinstance(decoded, dict) else decoded
        numbers = [float(n) if n is not None and math.isfinite(float(n)) else None for n in numbers]
        if component.kind == "map":
            # Keep acquired summary; append local source-coordinate contrasts, not extra votes.
            rows.append(row)
            shape = component.shape
            if len(shape) != 2 or shape[0] * shape[1] != len(numbers):
                raise ValueError("numeric_detail_map_shape")
            h, w = shape
            for y in range(2):
                for x in range(2):
                    y0, y1, x0, x1 = y * h // 2, (y + 1) * h // 2, x * w // 2, (x + 1) * w // 2
                    chosen = [numbers[i * w + j] for i in range(y0, y1) for j in range(x0, x1)]
                    finite = [n for n in chosen if n is not None]
                    if not finite:
                        continue
                    identifier = row["id"] + f":region:{y}:{x}"
                    body = dict(
                        source_shape=list(shape),
                        rows=[y0, y1],
                        columns=[x0, x1],
                        finite_count=len(finite),
                        total_count=len(chosen),
                        mean=sum(finite) / len(finite),
                        minimum=min(finite),
                        maximum=max(finite),
                        meaning="source_array_region_not_anatomical_sector_or_normative_percentile",
                        source_checksum=digest(numbers),
                        unit=component.unit,
                    )
                    rows.append(dict(row, id=identifier))
                    values[identifier] = material.observation(identifier, body)
            continue
        # Replace the lossy vector summary with complete bounded slices.
        # Every finite value, zero, negative value and null stays in source order.
        values.pop(row["id"], None)
        start = 0
        while start < len(numbers):
            end = min(start + 24, len(numbers))
            while True:
                body = dict(
                    values=numbers[start:end],
                    source_range=[start, end],
                    total=len(numbers),
                    source_checksum=digest(numbers),
                    unit=component.unit,
                    spatial_layout="unbound_source_order_not_anatomy",
                )
                if "semantic_profile:grape-v1" in component.semantic_limits:
                    from packages.dataset_adapters.multimodal.grape_semantics import VF_MEANING

                    selected = numbers[start:end]
                    body.update(
                        semantic_profile="grape-v1",
                        meaning=VF_MEANING,
                        coded_meaning="minus_one_unseen_or_blind_point_not_negative_measurement",
                        coded_count=sum(n == -1 for n in selected),
                        valid_count=sum(n is not None and n >= 0 for n in selected),
                        missing_count=sum(n is None for n in selected),
                        invalid_negative_count=sum(
                            n is not None and n < 0 and n != -1 for n in selected
                        ),
                    )
                if len(canonical(body).decode()) <= 1000:
                    break
                end -= 1
                if end <= start:
                    raise ValueError("numeric_detail_value_bound")
            identifier = row["id"] + f":values:{start}:{end}"
            rows.append(dict(row, id=identifier))
            values[identifier] = material.observation(identifier, body)
            start = end
    return rows, values

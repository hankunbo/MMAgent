"""Versioned nonempty disc and strict geometry output adapters; no performance gate."""

import json
from typing import Literal

from pydantic import Field

from packages.contracts import QCStatus
from packages.grounding.models import Ref, Term, Value, digest
from packages.object_store import checksum

from .models import ExecutionError
from .qc import PNGPolicy, QCObservation, TechnicalQCRegistry, grayscale_png


class DiscPNGPolicy(PNGPolicy):
    implementation: Literal["disc_png_nonempty/1.0.0"] = "disc_png_nonempty/1.0.0"
    labels: tuple[Literal[0, 255], ...] = (0, 255)


class GeometryJSONPolicy(Value):
    implementation: Literal["disc_geometry_json/1.0.0"] = "disc_geometry_json/1.0.0"
    artifact_ref: Ref
    artifact_type_code: str
    unit: Literal["pixel"] = "pixel"
    unit_term: Term
    role: Literal["derived_product"] = "derived_product"
    max_bytes: int = Field(default=65536, ge=1, le=65536)


class GeometryQCRegistry(TechnicalQCRegistry):
    def __init__(self, policies, objects):
        super().__init__(policies)
        self.objects = objects

    def evaluate(self, output, rule, data, inputs):
        policy = self.policy_for(output)
        if isinstance(policy, DiscPNGPolicy):
            observed = super().evaluate(output, rule, data, inputs)
            _, pixels = grayscale_png(data, policy.max_pixels)
            if 255 not in pixels:
                raise ExecutionError("empty_disc_mask")
            return observed
        if not isinstance(policy, GeometryJSONPolicy):
            return super().evaluate(output, rule, data, inputs)
        if (
            len(data) > policy.max_bytes
            or "application/json" not in rule.media_types
            or len(inputs) != 1
        ):
            raise ExecutionError("geometry_output_size_or_input")

        def pairs(values):
            result = {}
            for k, v in values:
                if k in result:
                    raise ValueError("duplicate")
                result[k] = v
            return result

        try:
            obj = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
            )
            keys = {
                "schema_version",
                "input_checksum",
                "height",
                "width",
                "area_px",
                "bbox_xyxy",
                "width_px",
                "height_px",
            }
            if (
                type(obj) is not dict
                or set(obj) != keys
                or obj["schema_version"] != "disc-geometry/1.0.0"
            ):
                raise ValueError("schema")
            if any(
                type(obj[k]) is not int
                for k in ("height", "width", "area_px", "width_px", "height_px")
            ):
                raise ValueError("integer")
            box = obj["bbox_xyxy"]
            if type(box) is not list or len(box) != 4 or any(type(x) is not int for x in box):
                raise ValueError("bbox")
            a = inputs[0]
            raw = self.objects.read_bounded(a.uri, 4194304)
            if checksum(raw) != a.checksum or obj["input_checksum"] != a.checksum:
                raise ValueError("source")
            (h, w), pixels = grayscale_png(raw, 16777216)
            if (
                (h, w) != a.dimensions
                or a.qc_status != QCStatus.PASS
                or a.laterality != output.constraint.laterality
                or a.coordinate_space != output.constraint.coordinate_space
            ):
                raise ValueError("input_scope")
            if not set(pixels) <= {0, 255}:
                raise ValueError("labels")
            count = 0
            x0 = w
            y0 = h
            x1 = y1 = 0
            for i, v in enumerate(pixels):
                if v:
                    y, x = divmod(i, w)
                    count += 1
                    x0 = min(x0, x)
                    y0 = min(y0, y)
                    x1 = max(x1, x + 1)
                    y1 = max(y1, y + 1)
            if not count or (
                obj["height"],
                obj["width"],
                obj["area_px"],
                box,
                obj["width_px"],
                obj["height_px"],
            ) != (h, w, count, [x0, y0, x1, y1], x1 - x0, y1 - y0):
                raise ValueError("geometry")
            if rule.dimensions_min != (1,) or rule.dimensions_max != (1,):
                raise ValueError("dimensions")
        except (ValueError, TypeError, RecursionError, KeyError, UnicodeError) as exc:
            raise ExecutionError("geometry_output_invalid") from exc
        return QCObservation(
            status=QCStatus.PASS,
            dimensions=(1,),
            media_type="application/json",
            policy_checksum=digest(policy),
            content_checksum=checksum(data),
            implementation=policy.implementation,
        )

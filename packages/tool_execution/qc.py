"""Versioned technical QC. No evidence validity, diagnosis, or tool upgrade authority."""

import struct
import zlib
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from packages.contracts import QCStatus
from packages.grounding.models import Ref, Term, Value, digest
from packages.object_store import checksum

from .models import ExecutionError


class PNGPolicy(Value):
    implementation: Literal["png_grayscale8_labels/1.0.0"] = "png_grayscale8_labels/1.0.0"
    artifact_ref: Ref
    artifact_type_code: str
    unit: str | None
    unit_term: Term
    role: Literal["model_prediction", "derived_product"]
    labels: tuple[Annotated[int, Field(ge=0, le=255)], ...] = Field(min_length=1, max_length=256)
    same_shape_as_first_input: Literal[True] = True
    max_pixels: int = Field(ge=1, le=16777216)


@dataclass(frozen=True)
class QCObservation:
    status: QCStatus
    dimensions: tuple[int, ...]
    media_type: str
    policy_checksum: str
    content_checksum: str
    implementation: str


def grayscale_png(data, max_pixels):
    """Full bounded CRC/decompression/filter parsing; unsupported PNG variants reject."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ExecutionError("output_format_invalid")
    offset, compressed, shape, ended = 8, bytearray(), None, False
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + size]
        end = offset + 12 + size
        if end > len(data) or zlib.crc32(kind + body) != int.from_bytes(data[end - 4 : end], "big"):
            raise ExecutionError("output_format_invalid")
        if kind == b"IHDR" and offset == 8 and size == 13:
            w, h, bits, color, comp, filtering, interlace = struct.unpack(">IIBBBBB", body)
            if (
                not w
                or not h
                or w * h > max_pixels
                or (bits, color, comp, filtering, interlace) != (8, 0, 0, 0, 0)
            ):
                raise ExecutionError("output_format_unsupported")
            shape = (h, w)
        elif kind == b"IDAT" and shape and not ended:
            compressed.extend(body)
        elif kind == b"IEND" and shape and not body and end == len(data):
            ended = True
        else:
            raise ExecutionError("output_format_unsupported")
        offset = end
    if not ended or offset != len(data) or not compressed:
        raise ExecutionError("output_format_invalid")
    h, w = shape
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(bytes(compressed), (w + 1) * h + 1)
    except zlib.error:
        raise ExecutionError("output_format_invalid") from None
    if len(raw) != (w + 1) * h or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ExecutionError("output_format_invalid")
    pixels, previous = bytearray(), bytearray(w)
    for y in range(h):
        row = bytearray(raw[y * (w + 1) + 1 : (y + 1) * (w + 1)])
        method = raw[y * (w + 1)]
        if method > 4:
            raise ExecutionError("output_format_invalid")
        for x in range(w):
            a, b, c = row[x - 1] if x else 0, previous[x], previous[x - 1] if x else 0
            if method == 1:
                predictor = a
            elif method == 2:
                predictor = b
            elif method == 3:
                predictor = (a + b) // 2
            elif method == 4:
                p = a + b - c
                predictor = min((a, b, c), key=lambda v: abs(p - v))
            else:
                predictor = 0
            row[x] = (row[x] + predictor) % 256
        pixels.extend(row)
        previous = row
    return shape, pixels


class TechnicalQCRegistry:
    port = "technical_output_qc/1.0.0"

    def __init__(self, policies):
        self._policies = dict(policies)
        for ref, policy in self._policies.items():
            if ref.kind != "policy" or ref.checksum != digest(policy):
                raise ExecutionError("qc_policy_content_mismatch")

    def policy_for(self, output):
        policy = self._policies.get(output.qc_policy_ref)
        if policy is None or output.qc_port != self.port:
            raise ExecutionError("qc_implementation_missing")
        if (
            policy.artifact_ref != output.constraint.artifact_ref
            or policy.unit_term != output.constraint.unit
        ):
            raise ExecutionError("qc_output_binding_mismatch")
        return policy

    def evaluate(self, output, rule, data, inputs):
        policy = self.policy_for(output)
        shape, pixels = grayscale_png(data, policy.max_pixels)
        if (
            "image/png" not in rule.media_types
            or len(shape) != len(rule.dimensions_min)
            or any(
                not low <= n <= high
                for n, low, high in zip(
                    shape, rule.dimensions_min, rule.dimensions_max, strict=True
                )
            )
        ):
            raise ExecutionError("output_dimensions_invalid")
        passed = (
            bool(inputs)
            and shape == inputs[0].dimensions
            and inputs[0].laterality == output.constraint.laterality
            and inputs[0].coordinate_space == output.constraint.coordinate_space
            and set(pixels) <= set(policy.labels)
        )
        return QCObservation(
            status=QCStatus.PASS if passed else QCStatus.FAIL,
            dimensions=shape,
            media_type="image/png",
            policy_checksum=digest(policy),
            content_checksum=checksum(data),
            implementation=policy.implementation,
        )

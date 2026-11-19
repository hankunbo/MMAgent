"""Single-file AutoMorph research output; technical integrity, not medical QC.

New artifact/implementation registration is required. The old directory output
and its installation lock are intentionally not changed or silently upgraded.
"""

import base64
import binascii
import json
from typing import Annotated, Literal

from pydantic import Field

from packages.contracts import QCStatus
from packages.grounding.models import Checksum, Ref, Term, Value, canonical, digest, validate
from packages.object_store import checksum
from packages.tool_benchmark.formats import jpeg_dimensions

from .models import ExecutionError
from .qc import QCObservation, TechnicalQCRegistry, grayscale_png

MAX_BYTES = 4_194_304
MAX_PNG = 1_048_576
COMMIT = "9a953e5edfa419b454e9fb7b06235eb413828d05"
ENGINE = "automorph-eight-wnet-single-file/1.0.0"
TRANSFORM = "RGB_bilinear_512_tensor_0_1_no_normalization"
Pixel = Annotated[int, Field(ge=1, le=16777216)]


class EncodedFundusPNG(Value):
    checksum: Checksum
    data_base64: str = Field(min_length=1, max_length=4 * ((MAX_PNG + 2) // 3))


class FundusMasks(Value):
    labels: EncodedFundusPNG
    disc: EncodedFundusPNG
    cup: EncodedFundusPNG
    rim: EncodedFundusPNG


class FundusBundle(Value):
    schema_version: Literal["automorph-fundus-bundle/1.0.0"] = "automorph-fundus-bundle/1.0.0"
    engine: Literal["automorph-eight-wnet-single-file/1.0.0"] = ENGINE
    source_commit: Literal["9a953e5edfa419b454e9fb7b06235eb413828d05"] = COMMIT
    source_checksum: Checksum
    input_checksum: Checksum
    input_shape: tuple[Pixel, Pixel]
    analysis_shape: tuple[Literal[512], Literal[512]] = (512, 512)
    transform: Literal["RGB_bilinear_512_tensor_0_1_no_normalization"] = TRANSFORM
    masks: FundusMasks
    device: Literal["cpu", "cuda:0"]
    ensemble_std_mean: float = Field(ge=0, le=1, allow_inf_nan=False)
    physical_scale: None = None
    disease_probability: None = None
    masks_modified: Literal[False] = False
    quality_is_calibrated_accuracy: Literal[False] = False
    training_overlap: Literal["GAMMA_REFUGE_in_upstream_training_config_not_external_test"] = (
        "GAMMA_REFUGE_in_upstream_training_config_not_external_test"
    )


def strict_json(data, limit=MAX_BYTES):
    if not isinstance(data, bytes) or not 0 < len(data) <= limit:
        raise ValueError("fundus_bundle_size")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("fundus_duplicate_key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("fundus_nonfinite")

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def decode_bundle(data):
    """Bounded full PNG decoding; returns exact committed bytes, never repairs masks."""
    try:
        value = validate(FundusBundle, strict_json(data))
        if value.input_shape[0] * value.input_shape[1] > 16777216:
            raise ValueError("input_pixels")
        raws, pixels = {}, {}
        for name in ("labels", "disc", "cup", "rim"):
            encoded = getattr(value.masks, name)
            raw = base64.b64decode(encoded.data_base64, validate=True)
            if (
                not 0 < len(raw) <= MAX_PNG
                or base64.b64encode(raw).decode("ascii") != encoded.data_base64
                or checksum(raw) != encoded.checksum
            ):
                raise ValueError("png_binding")
            shape, content = grayscale_png(raw, 512 * 512)
            if shape != value.analysis_shape:
                raise ValueError("analysis_shape")
            raws[name], pixels[name] = raw, content
        labels = pixels["labels"]
        if not set(labels) <= {0, 1, 2}:
            raise ValueError("labels")
        for name, selected in (("disc", {1, 2}), ("cup", {2}), ("rim", {1})):
            if pixels[name] != bytes(255 if p in selected else 0 for p in labels):
                raise ValueError("mask_class_binding")
        return value, raws
    except (ValueError, TypeError, UnicodeError, RecursionError, binascii.Error) as exc:
        raise ExecutionError("fundus_bundle_invalid") from exc


def encode_bundle(
    *, input_checksum, input_shape, source_checksum, masks, device, ensemble_std_mean
):
    """Construct the new envelope without changing any model-produced PNG bytes."""
    if set(masks) != {"labels", "disc", "cup", "rim"}:
        raise ExecutionError("fundus_bundle_mask_inventory")
    for raw in masks.values():
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_PNG:
            raise ExecutionError("fundus_bundle_mask_size")
    value = FundusBundle(
        input_checksum=input_checksum,
        input_shape=tuple(input_shape),
        source_checksum=source_checksum,
        device=device,
        ensemble_std_mean=ensemble_std_mean,
        masks=validate(
            FundusMasks,
            {
                name: dict(
                    checksum=checksum(raw), data_base64=base64.b64encode(raw).decode("ascii")
                )
                for name, raw in masks.items()
            },
        ),
    )
    raw = canonical(value)
    decode_bundle(raw)
    return raw


class FundusBundlePolicy(Value):
    implementation: Literal["automorph_fundus_bundle_qc/1.0.0"] = "automorph_fundus_bundle_qc/1.0.0"
    artifact_ref: Ref
    artifact_type_code: str = Field(min_length=1, max_length=128)
    input_artifact_type_code: str = Field(min_length=1, max_length=128)
    source_checksum: Checksum
    unit: None = None
    unit_term: Term
    role: Literal["model_prediction"] = "model_prediction"
    max_bytes: int = Field(default=MAX_BYTES, ge=1, le=MAX_BYTES)


class FundusBundleQCRegistry(TechnicalQCRegistry):
    def __init__(self, policies, objects):
        super().__init__(policies)
        self.objects = objects

    def evaluate(self, output, rule, data, inputs):
        policy = self.policy_for(output)
        if not isinstance(policy, FundusBundlePolicy):
            return super().evaluate(output, rule, data, inputs)
        if (
            len(inputs) != 1
            or len(data) > policy.max_bytes
            or "application/json" not in rule.media_types
            or rule.dimensions_min != (1,)
            or rule.dimensions_max != (1,)
            or policy.unit_term.state != "not_applicable"
        ):
            raise ExecutionError("fundus_bundle_rule_or_input")
        bundle, _ = decode_bundle(data)
        original = inputs[0]
        if (
            original.artifact_type_code != policy.input_artifact_type_code
            or original.media_type != "image/jpeg"
            or original.coordinate_space != "PIXEL_2D"
            or original.unit is not None
            or original.laterality != output.constraint.laterality
            or original.coordinate_space != output.constraint.coordinate_space
            or bundle.source_checksum != policy.source_checksum
            or bundle.input_checksum != original.checksum
            or bundle.input_shape != original.dimensions
        ):
            raise ExecutionError("fundus_bundle_source_binding")
        raw = self.objects.read_bounded(original.uri, MAX_BYTES)
        if checksum(raw) != original.checksum or jpeg_dimensions(raw) != bundle.input_shape:
            raise ExecutionError("fundus_bundle_input_changed")
        # Empty masks are technically valid. Geometry availability and medical
        # quality/direction are separate original owners, not inferred here.
        return QCObservation(
            status=QCStatus.PASS,
            dimensions=(1,),
            media_type="application/json",
            policy_checksum=digest(policy),
            content_checksum=checksum(data),
            implementation=policy.implementation,
        )

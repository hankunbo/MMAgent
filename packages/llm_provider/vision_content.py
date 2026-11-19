"""Explicit content-only Vision wire; identities come from the current request.

Never repairs a legacy candidate. Only the opt-in prompt uses this new envelope.
The short binding token detects mismatched responses; original full identities
remain in RequestRecord, image validation and the final domain proposal.
"""

from typing import Literal

from pydantic import Field

from packages.grounding.models import Value, digest, validate
from packages.llm_provider.models import PartialObservation, PartialVisionProposal, Text
from packages.object_store import checksum

PROMPT = "vision-partial-multimodal-1.1.0"


class VisionContent(Value):
    schema_version: Literal["visual-observation-content/1.0.0"]
    binding_code: str = Field(pattern=r"^V:[0-9a-f]{16}$")
    status: Literal["observed", "ungradable", "unknown"]
    finding: Text
    partial: PartialObservation


def prepare(scope, images, output_schema):
    if output_schema is not PartialVisionProposal or len(images) != 1:
        raise ValueError("vision_content_contract")
    image = images[0][0]
    binding = dict(
        scope_checksum=digest(scope),
        image_checksum=image.reference.checksum,
        preprocessing_checksum=image.preprocessing.checksum,
        location=image.location,
    )
    code = "V:" + digest(binding)[7:23]
    schema = VisionContent.model_json_schema()
    schema["properties"]["binding_code"]["enum"] = [code]
    return binding, code, schema


def decode(raw, binding, code):
    from .provider import strict_json

    content = validate(VisionContent, strict_json(raw))
    if content.binding_code != code:
        raise ValueError("vision_content_binding")
    result = PartialVisionProposal(
        **binding, status=content.status, finding=content.finding, partial=content.partial
    )
    proof = dict(
        schema_version="vision-content-binding/1.0.0",
        raw_checksum=checksum(raw),
        binding_code=code,
        binding=binding,
        candidate_checksum=digest(result),
        medical_content_modified=False,
    )
    return result, proof

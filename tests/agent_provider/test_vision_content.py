"""Content-only protocol never rewrites a legacy failed model response."""

import pytest

from packages.grounding.models import canonical, digest
from packages.llm_provider.models import PartialVisionProposal
from packages.llm_provider.vision_content import decode, prepare
from tests.evaluation_preparation.test_partial_observation import detail


def content(binding, code):
    return dict(
        schema_version="visual-observation-content/1.0.0",
        binding_code=code,
        status="observed",
        finding="局部可见，保留范围限制",
        partial=detail(),
    )


def values():
    from types import SimpleNamespace

    binding = SimpleNamespace(
        reference=SimpleNamespace(checksum=digest("image")),
        preprocessing=SimpleNamespace(checksum=digest("transform")),
        location=(0, 0, 8, 8),
    )
    return prepare(dict(run="one"), ((binding, b"unused"),), PartialVisionProposal)


def test_content_preserves_all_medical_fields_and_binds_request():
    binding, code, schema = values()
    raw = content(binding, code)
    value, proof = decode(canonical(raw), binding, code)
    assert schema["properties"]["binding_code"]["enum"] == [code]
    assert (
        value.finding == raw["finding"] and value.partial.model_dump(mode="json") == raw["partial"]
    )
    assert value.scope_checksum == binding["scope_checksum"]
    assert proof["medical_content_modified"] is False
    with pytest.raises(ValueError):
        PartialVisionProposal.model_validate(raw)


@pytest.mark.parametrize("field", ["binding_code", "image_checksum", "coverage", "finding"])
def test_wrong_binding_extra_identity_and_invalid_content_still_rejected(field):
    binding, code, _ = values()
    raw = content(binding, code)
    if field == "binding_code":
        raw[field] = "V:" + "0" * 16
    elif field == "image_checksum":
        raw[field] = digest("other image")
    elif field == "coverage":
        raw["partial"][field] = 2
    else:
        raw[field] = "a" * 241
    with pytest.raises(ValueError):
        decode(canonical(raw), binding, code)

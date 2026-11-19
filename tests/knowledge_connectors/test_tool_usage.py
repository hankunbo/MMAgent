import json

import pytest
from pydantic import ValidationError

from apps import knowledge_tool_profiles as app
from packages.knowledge_connectors.usage import ToolUsageProfile, seal_profile


@pytest.mark.parametrize("key", tuple(app.DETAILS))
def test_profiles_match_exact_installed_definition(key):
    p = app.profile(key)
    assert p.provider_ref.code == app.provider(key).code
    assert p.provider_ref.checksum == app.provider(key).checksum
    assert p.best_for and p.inputs and p.outputs and p.examples and p.limitations
    assert ToolUsageProfile.model_validate_json(p.model_dump_json()) == p
    assert "@" not in json.dumps(p.model_dump(mode="json"))


def test_tampered_description_cannot_keep_old_checksum():
    p = app.profile("aao").model_copy(update={"purpose": "fake"})
    with pytest.raises(ValidationError, match="tool_usage_checksum"):
        ToolUsageProfile.model_validate_json(p.model_dump_json())


def test_wrong_reference_or_unknown_fields_rejected():
    raw = app.profile("aao").model_dump(mode="json", exclude={"checksum"})
    raw["provider_ref"]["code"] = "SOURCE:OTHER:FAKE"
    with pytest.raises(ValidationError, match="tool_usage_requires_provider"):
        seal_profile(**raw)

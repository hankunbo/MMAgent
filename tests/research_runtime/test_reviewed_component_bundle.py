"""Profile sealing for the optional bundle fixture; old PNG default stays exact."""

import pytest

from packages.grounding.models import validate
from packages.tool_admission.models import JSONOutputRule
from tests.research_runtime.reviewed_component_tool_support import deployment
from tests.tool_execution.test_fundus_bundle import material as material


@pytest.mark.parametrize("bundle", [False, True])
def test_output_rule_is_explicit_before_profile_sealing(tmp_path, material, bundle):
    ready = deployment(tmp_path / "deployment", bundle=material.bundle if bundle else None)
    profile = validate(type(ready.profile), ready.profile.model_dump(mode="json"))
    rule = profile.output_rules[0]
    assert isinstance(rule, JSONOutputRule) is bundle
    assert rule.media_types == (("application/json",) if bundle else ("image/png",))
    assert rule.dimensions_min == rule.dimensions_max == ((1,) if bundle else (2, 3))
    assert (ready.bundle_qc is not None) is bundle

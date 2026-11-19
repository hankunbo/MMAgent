"""Fresh GRAPE uses existing exact dependency encoding; old failure is immutable."""

import copy
import json
from pathlib import Path

from packages.llm_provider.config import prompt
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.definition_expansion import (
    definition_config,
    read_expansion_policy,
)
from packages.multimodal_research.implementation import freeze


def test_new_grape_definition_profile_keeps_acquisition_and_budgets():
    old_path = Path("configs/research/multimodal_grape_active_requirements_v2.json")
    new_path = Path("configs/research/multimodal_grape_active_requirements_v3.json")
    original_bytes = old_path.read_bytes()
    old, new = load(old_path), load(new_path)
    assert {k: v for k, v in old.items() if k != "definition_expansion"} == {
        k: v for k, v in new.items() if k != "definition_expansion"
    }
    expected = copy.deepcopy(old["definition_expansion"])
    expected.update(
        schema_version="campaign-definition-expansion/1.27.0",
        prompt_version="definition-draft-6.3.0",
        unresolved_relation_retrieval=None,
    )
    for plan in expected["passage_plans"].values():
        plan["schema_version"] = "definition-passage-plan/5.0.0"
    policy = read_expansion_policy(new["definition_expansion"])
    assert new["definition_expansion"] == expected
    config = definition_config(
        configuration(directional=True), prompt_version=policy.prompt_version
    )
    role, text = prompt(config, "source")
    assert role.prompt_version == "definition-draft-6.3.0"
    assert "CONDITION TEXT strings" in text and "text=dependencies[i]" in text
    assert old_path.read_bytes() == original_bytes
    assert (
        json.loads(original_bytes)["definition_expansion"]["prompt_version"]
        == "definition-draft-6.1.0"
    )
    assert str(new_path) in freeze()

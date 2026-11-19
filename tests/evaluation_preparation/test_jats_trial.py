"""Explicit policy/entry dispatch; prior targeted definitions retain prior parsing."""

import json
from pathlib import Path

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.multimodal_research.definition_expansion import read_expansion_policy


def test_jats_policy_selects_actual_parser_and_old_profile_is_unchanged():
    for version, parser in ((6, "xml-scripts/1.0.0"), (7, "xml-jats/1.0.0")):
        p = read_expansion_policy(
            json.loads(
                Path(f"configs/research/multimodal_definition_gate_v{version}.json").read_bytes()
            )
        )
        s = KnowledgeDefinitionSources(
            repository=None, store=None, material_policy=p.material_policy
        )
        assert s.index_options()["parse_policy"].version == parser
        assert p.max_fetch_requests == p.max_search_requests == 2
        assert p.max_definitions == p.max_new_observations == 1


def test_jats_trial_version_does_not_accept_wrong_target_or_old_policy(monkeypatch):
    import pytest

    from packages.grounding.models import digest
    from packages.multimodal_research import definition_trial as t

    monkeypatch.setattr(t, "freeze", lambda: {"test": "freeze"})
    spec = dict(
        implementation=t.freeze(),
        schema_version="multimodal-live-entry/1.13.0",
        intent=dict(target_code="GOAL:ADVANCED_GLAUCOMA"),
    )
    policy = json.loads(Path("configs/research/multimodal_definition_gate_v7.json").read_bytes())
    value = t.JatsTrial(
        entry_checksum=digest(spec),
        authorization="offline contract test only",
        expansion_policy=policy,
    ).model_dump(mode="json")
    assert t.checked(value, spec) == value
    value["expansion_policy"]["public_target"] = "GOAL:EARLY_GLAUCOMA"
    with pytest.raises(ValueError, match="public_target_binding"):
        t.checked(value, spec)

"""New automatic fetch profile is explicit; old plans never silently change."""

import json
from pathlib import Path

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.grounding.models import digest
from packages.knowledge_index.parsers import parse
from packages.multimodal_research import definition_trial as trial
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.knowledge_index.test_jats_citation_profile import BODY


def policy():
    value = json.loads(Path("configs/research/multimodal_definition_gate_v10.json").read_bytes())
    value.update(
        schema_version="campaign-definition-expansion/1.12.0",
        material_policy="xml-jats-citations-whole-text/1.0.0",
    )
    return value


def test_new_policy_reaches_original_parser_without_manual_index():
    current = read_expansion_policy(policy())
    source = KnowledgeDefinitionSources(
        repository=None, store=None, material_policy=current.material_policy
    )
    options = source.index_options()
    assert options["parse_policy"].version == "xml-jats-citations/1.0.0"
    result = parse(BODY, "application/xml", policy=options["parse_policy"])
    assert result.status == "parsed"
    assert result.document.parser_version == "2.3.0"
    assert current.prompt_version == "definition-draft-4.0.0"
    assert current.source_output_tokens == 8192


def test_new_trial_and_old_trial_policy_binding(monkeypatch):
    monkeypatch.setattr(trial, "freeze", lambda: {"test": "unchanged"})
    spec = dict(
        schema_version="multimodal-live-entry/1.13.0",
        implementation=trial.freeze(),
        intent=dict(target_code="GOAL:ADVANCED_GLAUCOMA"),
    )
    value = dict(
        schema_version="multimodal-definition-trial/1.6.0",
        entry_checksum=digest(spec),
        authorization="offline synthetic parser contract",
        expansion_policy=policy(),
        max_new_requests=8,
        seconds=1800,
    )
    checked = trial.checked(value, spec)
    assert checked["expansion_policy"] == policy()
    value["schema_version"] = "multimodal-definition-trial/1.5.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        trial.checked(value, spec)


def test_old_policy_cannot_opt_in_without_new_version():
    value = policy()
    value["schema_version"] = "campaign-definition-expansion/1.11.0"
    with pytest.raises(ValueError):
        read_expansion_policy(value)


def test_old_policy_keeps_unsupported_historical_parsing():
    value = json.loads(Path("configs/research/multimodal_definition_gate_v10.json").read_bytes())
    old = read_expansion_policy(value)
    source = KnowledgeDefinitionSources(
        repository=None, store=None, material_policy=old.material_policy
    )
    result = parse(BODY, "application/xml", policy=source.index_options()["parse_policy"])
    assert result.status == "unsupported"
    assert result.reason == "numeric_markup_unsupported"


def test_public_configuration_and_prepare_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "freeze", lambda: {"test": "unchanged"})
    spec = dict(
        schema_version="multimodal-live-entry/1.13.0",
        implementation=trial.freeze(),
        intent=dict(target_code="GOAL:ADVANCED_GLAUCOMA"),
    )
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    value = trial.prepare(
        tmp_path,
        expansion_policy="configs/research/multimodal_definition_gate_v11.json",
        authorization="offline software gate, no actual API",
        max_new_requests=8,
        seconds=1800,
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.6.0"
    assert value["expansion_policy"] == policy()
    assert trial.read(tmp_path) == value
    assert {p.name for p in tmp_path.iterdir()} == {"entry.json", "definition-trial.json"}


def test_new_config_is_in_campaign_implementation_freeze():
    from packages.multimodal_research.implementation import freeze
    from packages.object_store import checksum

    path = Path("configs/research/multimodal_definition_gate_v11.json")
    assert freeze()[str(path)] == checksum(path.read_bytes())


def test_new_policy_and_trial_schemas_are_exact():
    from packages.multimodal_research.definition_expansion import CitationExpansionPolicy

    for model in (CitationExpansionPolicy, trial.CitationTrial):
        path = Path("packages/llm_provider/schemas") / (model.__name__ + ".schema.json")
        assert json.loads(path.read_bytes()) == model.model_json_schema()

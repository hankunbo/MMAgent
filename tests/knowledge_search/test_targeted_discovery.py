"""Proposition words actually reach every wire template, not just local FTS."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.knowledge_search import KnowledgeQueryPlan, broad_plan
from packages.knowledge_search.planning import provider_query
from packages.knowledge_search.targeted import TARGETS, targeted_plan
from tests.knowledge_search.support import config, policy


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("adapter", ["pubmed", "europe_pmc", "crossref"])
def test_target_reaches_provider_query(target, adapter):
    c = config(adapter)
    p = targeted_plan((c.provider_code,), target=target, topics=("oct", "gcc"))
    q = provider_query(p, c, policy())
    assert all(term in q.texts[0] for term in TARGETS[target])
    assert "optical coherence tomography" in q.texts[0]
    assert p == KnowledgeQueryPlan.model_validate_json(p.model_dump_json())


@pytest.mark.parametrize("mutation", ["version", "target_missing", "private", "two_targets"])
def test_wrong_or_unbound_plan_rejected(mutation):
    p = targeted_plan((config().provider_code,), target="GOAL:ADVANCED_GLAUCOMA", topics=("rnfl",))
    raw = p.model_dump(mode="json")
    target = next(v for v in raw["concepts"] if v["code"].startswith("CONCEPT:TARGET:"))
    if mutation == "version":
        raw["query_plan_version"] = "5.0.0"
    elif mutation == "target_missing":
        raw["concepts"].remove(target)
    elif mutation == "private":
        target["synonyms"].append("patient Alice")
    else:
        raw["concepts"].append(
            dict(target, code="CONCEPT:TARGET:EARLY_GLAUCOMA", label="early glaucoma", synonyms=[])
        )
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate_json(json.dumps(raw))


def test_no_complement_or_unknown_patient_target():
    for target in ("GOAL:NO_GLAUCOMA", "GAMMA 0004", "patient Alice", ""):
        with pytest.raises(ValueError, match="unsupported_public_proposition"):
            targeted_plan((config().provider_code,), target=target)


def test_guideline_target_query_fails_explicitly_not_false_empty():
    c = config("guideline")
    p = targeted_plan((c.provider_code,), target="GOAL:ADVANCED_GLAUCOMA", topics=("oct",))
    with pytest.raises(ValueError, match="targeted_guideline_search_not_supported"):
        provider_query(p, c, policy())
    # Legacy title-only lookup retains its prior meaning.
    assert provider_query(broad_plan((c.provider_code,)), c, policy()).texts == ("glaucoma",)


def test_legacy_query_and_frozen_sql_preserved():
    from db.knowledge.validation.build_broad import TARGET, artifact
    from db.knowledge.validation.build_targeted import TARGET as NEW
    from db.knowledge.validation.build_targeted import artifact as new_artifact

    p = broad_plan((config("crossref").provider_code,), topics=("rnfl",))
    assert provider_query(p, config("crossref"), policy()).texts == (
        "glaucoma retinal nerve fiber layer rnfl",
    )
    assert TARGET.read_text() == artifact()
    assert NEW.read_text() == new_artifact()


def test_actual_definition_connector_passes_bound_target(monkeypatch):
    from apps import knowledge_literature_tools
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
    from packages.multimodal_research.definition_expansion import read_expansion_policy

    p = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v6.json").read_text())
    )
    calls = []

    def search(repo, keys, topic, **kwargs):
        calls.append(kwargs)
        real_plan = targeted_plan(
            (config().provider_code,), target=kwargs["public_target"], topics=kwargs["topics"]
        )
        q = provider_query(real_plan, config(), policy())
        assert "advanced glaucoma" in q.texts[0]
        return dict(status="empty", runs=[], results=[])

    monkeypatch.setattr(knowledge_literature_tools, "search", search)
    expansion = SimpleNamespace(policy=p, binding={"target": p.public_target})
    c = LiteratureDefinitionConnector(None, expansion, p.public_topic_sets)
    assert c.search(query=p.public_queries[0], key="bounded", limit=3) == []
    assert len(calls) == 1
    expansion.binding["target"] = "GOAL:EARLY_GLAUCOMA"
    with pytest.raises(ValueError, match="definition_public_target_binding"):
        c.search(query=p.public_queries[0], key="wrong", limit=3)
    assert len(calls) == 1


def test_new_trial_and_policy_versions_do_not_upgrade_old_envelopes(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial as t

    monkeypatch.setattr(t, "freeze", lambda: {"frozen": "same"})
    spec = dict(
        schema_version="multimodal-live-entry/1.13.0",
        implementation=t.freeze(),
        intent=dict(target_code="GOAL:ADVANCED_GLAUCOMA"),
    )
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    fresh = t.prepare(
        tmp_path,
        expansion_policy="configs/research/multimodal_definition_gate_v6.json",
        authorization="offline test authorization",
    )
    assert fresh["schema_version"] == "multimodal-definition-trial/1.1.0"
    assert t.read(tmp_path) == fresh
    from packages.grounding.models import digest

    wrong = dict(spec, intent=dict(target_code="GOAL:EARLY_GLAUCOMA"))
    with pytest.raises(ValueError, match="definition_trial_public_target_binding"):
        t.checked(dict(fresh, entry_checksum=digest(wrong)), wrong)
    fresh["schema_version"] = "multimodal-definition-trial/1.0.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        t.checked(fresh, spec)

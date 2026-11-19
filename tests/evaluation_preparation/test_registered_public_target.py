import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.knowledge_search.models import KnowledgeQueryPlan
from packages.knowledge_search.registered_targets import registered_plan
from packages.knowledge_search.targeted import targeted_plan
from packages.knowledge_search.template import render


def test_public_query_reuses_existing_owner_versions_and_preserves_staging():
    old = targeted_plan(
        ("SOURCE:EUROPEPMC",), target="GOAL:ADVANCED_GLAUCOMA", topics=("rim",), limit=5
    )
    current = registered_plan(
        ("SOURCE:EUROPEPMC",), target="GOAL:GLAUCOMA_DETECTION", topics=("rim",), limit=5
    )
    assert old.query_plan_version == "6.0.0" and current.query_plan_version == "5.0.0"
    assert old.privacy.term_version == "3.0.0" and current.privacy.term_version == "2.0.0"
    text = json.dumps(render(current.model_dump(mode="json"), "europe_pmc"))
    assert "glaucoma" in text and "neuroretinal rim" in text
    assert "advanced" not in text and "severe" not in text
    assert (
        targeted_plan(
            ("SOURCE:EUROPEPMC",), target="GOAL:ADVANCED_GLAUCOMA", topics=("rim",), limit=5
        )
        == old
    )
    assert KnowledgeQueryPlan.model_validate_json(current.model_dump_json()) == current
    with pytest.raises(ValueError, match="unsupported_public_proposition"):
        targeted_plan(("SOURCE:EUROPEPMC",), target="GOAL:GLAUCOMA_DETECTION", topics=("rim",))


@pytest.mark.parametrize("change", ["target", "label", "policy", "target_term"])
def test_no_patient_text_or_wrong_vocabulary_can_enter_public_query(change):
    plan = registered_plan(
        ("SOURCE:EUROPEPMC",), target="GOAL:GLAUCOMA_DETECTION", topics=("rim",)
    ).model_dump(mode="json")
    if change == "target":
        with pytest.raises(ValueError, match="unsupported_registered_public_proposition"):
            registered_plan(("SOURCE:EUROPEPMC",), target="GOAL:PATIENT_FREE_TEXT")
        return
    if change == "label":
        plan["concepts"][-1]["label"] = "patient Alice age 40"
    elif change == "policy":
        plan["privacy"]["version"] = "patient-free/5.0.0"
    else:
        plan["concepts"].append(
            dict(
                terminology_version="2.0.0",
                code="CONCEPT:TARGET:ADVANCED_GLAUCOMA",
                label="advanced glaucoma",
                synonyms=[],
            )
        )
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate_json(json.dumps(plan))


def test_original_definition_connector_sends_bound_question_to_original_app(monkeypatch):
    from apps import knowledge_literature_tools as app
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
    from packages.knowledge_search.planning import provider_query
    from packages.multimodal_research.automatic_profile import derive
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.knowledge_search.support import config, policy

    registry = Path("configs/research/multimodal_tasks_v2.json")
    intent, _ = load_intent("glaucoma_detection", registry)
    raw, _ = derive("configs/research/multimodal_catalog_acquisition_v8.json", intent)
    p = read_expansion_policy(raw["definition_expansion"])
    calls = []

    def search(repo, keys, topic, **kwargs):
        calls.append(kwargs)
        assert kwargs["public_target_vocabulary"] == "registered-task-public/1.0.0"
        plan = registered_plan(
            (config().provider_code,), target=kwargs["public_target"], topics=kwargs["topics"]
        )
        wire = provider_query(plan, config(), policy())
        assert "glaucoma" in wire.texts[0] and "severe" not in wire.texts[0]
        return dict(status="empty", runs=[], results=[])

    monkeypatch.setattr(app, "search", search)
    expansion = SimpleNamespace(policy=p, binding={"target": intent.target_code})
    connector = LiteratureDefinitionConnector(None, expansion, p.public_topic_sets)
    assert connector.search(query=p.public_queries[0], key="candidate", limit=3) == []
    assert len(calls) == 1
    expansion.binding["target"] = "GOAL:ADVANCED_GLAUCOMA"
    with pytest.raises(ValueError, match="definition_public_target_binding"):
        connector.search(query=p.public_queries[0], key="wrong", limit=3)
    assert len(calls) == 1

"""Broad discovery keeps v4 replay and admits only fixed public topic concepts."""

import json
from pathlib import Path

import pytest

from packages.knowledge_search import KnowledgeQueryPlan, broad_plan
from packages.knowledge_search.broad import BROAD_CHECKSUM, MODE, TOPICS
from packages.knowledge_search.planning import provider_query
from tests.knowledge_search.support import config, plan, policy


@pytest.mark.parametrize("topic", tuple(TOPICS))
def test_each_topic_is_public_bounded_and_no_placeholder_and(topic):
    c = config()
    p = broad_plan((c.provider_code,), topics=(topic,))
    q = provider_query(p, c, policy())
    assert q.constraints_mode == MODE
    assert '"glaucoma"' in q.texts[0]
    assert not any(
        s in q.texts[0]
        for s in ('"en"', '"research cohort"', '"original research"', '"association"')
    )
    assert p == KnowledgeQueryPlan.model_validate_json(p.model_dump_json())


@pytest.mark.parametrize(
    "changes",
    [
        {"query_plan_version": "4.0.0"},
        {"template_version": "2.0.0"},
        {"population": ("this patient age 86",)},
        {"date_from": "2026-01-01"},
        {"modalities": ("slo",)},
        {"age_min": 86},
    ],
)
def test_broad_postconstruction_mutations_fail(changes):
    p = broad_plan((config().provider_code,))
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate(p.model_copy(update=changes))


def test_unknown_topic_and_raw_patient_text_not_admitted():
    for topics in (("mrn 123",), ("glaucoma; DROP TABLE",), ("rnfl", "rnfl")):
        with pytest.raises(ValueError):
            broad_plan((config().provider_code,), topics=topics)
    p = broad_plan((config().provider_code,), topics=("rnfl",))
    raw = p.model_dump(mode="json")
    raw["concepts"][0]["synonyms"] = ["patient Alice Example"]
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate_json(json.dumps(raw))
    raw = p.model_dump(mode="json")
    raw["privacy"]["checksum"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate_json(json.dumps(raw))
    assert p.privacy.checksum == BROAD_CHECKSUM


def test_legacy_v4_bytes_unchanged_and_rejects_new_terms():
    # Frozen deployed 0005 schema still accepts every serialized legacy field.
    from db.knowledge.validation.build_v5 import TARGET, artifact

    p = plan(config())
    spec = json.loads(Path("db/knowledge/validation/v5_spec.json").read_text())
    assert set(p.model_dump()) == set(spec["plan_schema"]["properties"])
    assert spec["plan_schema"]["properties"]["query_plan_version"]["const"] == "4.0.0"
    assert TARGET.read_text() == artifact()
    assert p.query_plan_version == "4.0.0" and p.template_version == "2.0.0"
    q = provider_query(p, config(), policy())
    assert '"research' in q.texts[0] or '"adults"' in q.texts[0]
    broad = broad_plan((config().provider_code,), topics=("rim",))
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate(p.model_copy(update={"concepts": broad.concepts}))


def test_guideline_is_catalog_search_not_global_site_search():
    c = config("guideline")
    p = broad_plan((c.provider_code,), topics=("rnfl",))
    q = provider_query(p, c, policy())
    assert q.constraints_mode == "catalog_title_only" and q.texts == ("glaucoma",)


def test_crossref_uses_bibliographic_words_and_preferences_are_not_claimed_filters():
    c = config("crossref")
    p = broad_plan((c.provider_code,), topics=("rnfl",))
    q = provider_query(p, c, policy())
    assert q.texts == ("glaucoma retinal nerve fiber layer rnfl",)
    assert q.constraints_mode == MODE
    altered = KnowledgeQueryPlan.model_validate(
        p.model_copy(update={"languages": ("fr",), "source_types": ("guideline",)})
    )
    assert provider_query(altered, c, policy()).texts == q.texts
    assert altered.checksum() != p.checksum()


def test_new_migration_and_model_schema_are_exact():
    from db.knowledge.validation.build_broad import TARGET, artifact

    assert TARGET.read_text() == artifact()
    schema = json.loads(
        Path("packages/knowledge_search/schemas/KnowledgeQueryPlan.schema.json").read_text()
    )
    assert schema == KnowledgeQueryPlan.model_json_schema()

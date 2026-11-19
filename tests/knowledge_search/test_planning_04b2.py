import json
from datetime import date

import pytest
from pydantic import ValidationError

from packages.knowledge_search import (
    CodedTerm,
    ExpansionConfig,
    ExpansionSuggestion,
    KnowledgeQueryPlan,
    ProviderBudget,
    QueryIdentifier,
    expand_plan,
)
from packages.knowledge_search.planning import provider_query
from tests.knowledge_search.support import config, plan, policy


def test_single_multi_source_templates_are_generic_and_bound():
    a, b = config(), config("pubmed")
    for p in (plan(a), plan(a, b)):
        assert (
            p.checksum() == KnowledgeQueryPlan.model_validate_json(p.model_dump_json()).checksum()
        )
        for c in (a, b)[: len(p.providers)]:
            q = provider_query(p, c, policy())
            assert q.plan_checksum == p.checksum()
            assert q.identity.config_checksum == c.checksum()
            assert "synthetic condition" in q.texts[0]
            assert "association" in q.texts[0]
            assert "glaucoma" not in q.texts[0]


def test_set_order_synonyms_identifiers_dates_have_one_semantic_checksum():
    a, b = config(), config("pubmed")
    p = plan(
        a,
        b,
        population=(" Adults ", "adults", "ELDERLY"),
        languages=("fr", "en"),
        diseases=(CodedTerm(code="concept:test:condition", label="BETA", synonyms=("alpha",)),),
        date_from="2020/01/02",
        date_to="2025-02-03",
        identifiers=(QueryIdentifier(kind="DOI", value="https://doi.org/10.1234/ABC"),),
    )
    q = plan(
        b,
        a,
        population=("elderly", "adults"),
        languages=("en", "fr"),
        diseases=(CodedTerm(code="CONCEPT:TEST:CONDITION", label="alpha", synonyms=("beta",)),),
        date_from=date(2020, 1, 2),
        date_to=date(2025, 2, 3),
        identifiers=(QueryIdentifier(kind="DOI", value="10.1234/abc"),),
    )
    assert p == q and p.checksum() == q.checksum()
    assert provider_query(p, a, policy()) == provider_query(q, a, policy())


@pytest.mark.parametrize(
    "changes",
    [
        {"goal": "How to diagnose glaucoma?"},
        {"claim": "diagnose glaucoma"},
        {"population": ()},
        {"age_min": 70, "age_max": 20},
        {"date_from": "2026-01-01", "date_to": "2025-01-01"},
        {"date_from": "2026"},
        {"sexes": ("anything",)},
        {"source_types": ("medical_truth",)},
        {"total_max_hits": True},
        {"max_concurrency": 0},
        {"total_max_requests": -1},
        {"languages": ("english",)},
        {"population": ("mailto=secret",)},
        {"anatomy": ("x@example.invalid",)},
        {"source_version": "https://example.org?q=1"},
        {"population": ('x" OR *',)},
        {"tool": "disallowed"},
    ],
)
def test_invalid_conditions_are_rejected(changes):
    with pytest.raises((ValidationError, ValueError)):
        plan(config(), **changes)


def test_revalidate_copies_and_no_implicit_scope_repair():
    p = plan(config())
    with pytest.raises(ValidationError):
        KnowledgeQueryPlan.model_validate(p.model_copy(update={"total_max_hits": "10"}))
    q = provider_query(p, config(), policy())
    from packages.knowledge_search.models import ProviderQuery

    raw = q.model_dump(mode="json")
    del raw["identity"]["config_checksum"]
    with pytest.raises(ValidationError):
        ProviderQuery.model_validate_json(json.dumps(raw))


class FakeLLM:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def suggest(self, plan, config):
        self.calls += 1
        return self.output


def test_expansion_is_opt_in_allowlisted_versioned_and_cannot_change_conditions():
    p = plan(config())
    suggestion = ExpansionSuggestion(
        concept_code=p.diseases[0].code, kind="synonym", term="related spelling"
    )
    llm = FakeLLM(json.dumps({"suggestions": [suggestion.model_dump()]}))
    assert expand_plan(p, ExpansionConfig(), llm) == p and llm.calls == 0
    settings = ExpansionConfig(enabled=True, model="fake-model/1", allowlist=(suggestion,))
    expanded = expand_plan(p, settings, llm)
    assert (
        expanded.goal == p.goal
        and expanded.claim == p.claim
        and expanded.population == p.population
    )
    assert expanded.expansion.model == "fake-model/1"
    assert expanded.expansion.config_checksum.startswith("sha256:")
    assert "related spelling" in provider_query(expanded, config(), policy()).texts[0]
    assert expanded.checksum() == expand_plan(p, settings, llm).checksum()


@pytest.mark.parametrize(
    "output",
    [
        '{"suggestions":[],"goal":"changed"}',
        '{"suggestions":[],"age_min":0}',
        '{"suggestions":[],"suggestions":[]}',
        '{"suggestions":"bad"}',
        "x" * 9000,
        '{"suggestions":[{"concept_code":"CONCEPT:OTHER","kind":"synonym","term":"new"}]}',
        '{"suggestions":[{"concept_code":"CONCEPT:TEST:CONDITION","kind":"synonym",'
        '"term":"not allowed"}]}',
        '{"suggestions":[{"concept_code":"CONCEPT:TEST:CONDITION","kind":"rule","term":"new"}]}',
    ],
)
def test_fake_llm_illegal_extensions(output):
    with pytest.raises((ValidationError, ValueError)):
        expand_plan(plan(config()), ExpansionConfig(enabled=True), FakeLLM(output))


def test_duplicate_provider_and_deployment_workers_rejected():
    c = config()
    with pytest.raises(ValidationError):
        plan(c, providers=(ProviderBudget(provider_code=c.provider_code),) * 2)
    from packages.knowledge_search import RatePolicy

    with pytest.raises(ValidationError):
        RatePolicy.model_validate(policy().model_copy(update={"deployment_workers": 2}))


def test_public_plan_schema_snapshot_matches_current_model():
    from pathlib import Path

    schema = json.loads(
        Path("packages/knowledge_search/schemas/KnowledgeQueryPlan.schema.json").read_text()
    )
    assert schema == KnowledgeQueryPlan.model_json_schema()

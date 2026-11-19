"""Patient-free provenance, recursive encoding and complete live-result boundaries."""

from urllib.parse import quote

import pytest

from apps.knowledge_federated_search import KnowledgeSearchService
from packages.knowledge_search import (
    CodedTerm,
    CohortConditions,
    ExpansionConfig,
    FederatedResult,
    KnowledgeQueryPlan,
    PersistedProviderOutcome,
    PrivacyError,
    ProviderOutcome,
    QueryIdentifier,
    derive_federated_status,
    expand_plan,
)
from packages.knowledge_search.privacy import safe_text, safe_url
from tests.knowledge_search.support import assembly, config, exchange, plan
from tests.knowledge_search.test_planning_04b2 import FakeLLM
from tests.knowledge_search.test_repair_red_04b2 import success


@pytest.mark.parametrize(
    "text",
    [
        "patient john smith",
        "MRN 12345678",
        "DOB 1970-01-02",
        "phone 1234567890",
        "home address 100 synthetic road",
        "subject id 12345",
        "visit date 2026-01-01",
        "current patient reports blurred vision",
        "张三 住院号 12345678",
        "unknown unreviewed term",
    ],
)
def test_untrusted_terms_cannot_reach_expander_connector_or_repository(text):
    p = plan(config()).model_copy(update={"population": (text,)})
    llm = FakeLLM('{"suggestions":[]}')
    with pytest.raises(ValueError):
        expand_plan(p, ExpansionConfig(enabled=True), llm)
    assert llm.calls == 0
    executor, transports = assembly((config(),), {})

    class Repository:
        def lookup_search(self, *args):
            pytest.fail("patient input reached cache")

        def commit_search(self, *args):
            pytest.fail("patient input reached database/Audit")

    with executor:
        app = KnowledgeSearchService(executor, Repository(), ())
        with pytest.raises(ValueError):
            app.search(p, idempotency_key="synthetic-patient-rejection")
    assert transports[config().provider_code].calls == 0


def test_cohort_catalog_and_explicit_legacy_rejection():
    p = plan(
        config(),
        cohort=CohortConditions(code="ADULT_FEMALES"),
        age_min=18,
        sexes=("female",),
        population=("research cohort", "adults"),
        diseases=(CodedTerm(code="CONCEPT:DISEASE:GLAUCOMA", label="glaucoma"),),
        modalities=("oct",),
        anatomy=("optic nerve",),
        source_version="1.2.3",
    )
    assert p.privacy.version == "patient-free/3.0.0"
    for change in ({"age_min": 37}, {"sexes": ("male",)}, {"query_plan_version": "1.0.0"}):
        with pytest.raises(ValueError):
            KnowledgeQueryPlan.model_validate(p.model_copy(update=change))
    with pytest.raises(ValueError):
        CohortConditions.model_validate({"provenance": "case_observation", "code": "ADULTS"})
    with pytest.raises(ValueError):
        CodedTerm(code="CONCEPT:TEST:CONDITION", label="patient john smith")


@pytest.mark.parametrize("layers", [1, 2, 3, 4, 8, 16, 20])
@pytest.mark.parametrize(
    "payload",
    [
        "https://example.invalid?x=1",
        "email=fixture@example.invalid",
        "mailto:fixture@example.invalid",
        "api_key=fixture",
        "Authorization: fixture",
    ],
)
def test_recursive_encoding_rejects_at_any_depth(layers, payload):
    for _ in range(layers):
        payload = quote(payload, safe="")
    with pytest.raises(PrivacyError):
        safe_text(payload)


def test_mixed_encoding_and_valid_discovery_values():
    for value in (
        "ｅｍａｉｌ%3dfixture&#64;example.invalid",
        "api%5fkey&#61;value",
        "Authorization&colon; value",
        "%u0040",
        "\\u0040",
        "%2525252540",
    ):
        with pytest.raises(PrivacyError):
            safe_text(value)
    assert safe_text("Retinal structure & function in adults")
    assert safe_text("10.1234/abc")
    assert QueryIdentifier(kind="DOI", value="https://doi.org/10.1234/ABC").value == "10.1234/abc"
    assert safe_url("https://doi.org/10.1234/abc")
    for url in (
        "https://x.example/path?email=x",
        "https://x.example/path#fragment",
        "https://user@x.example/path",
        "https://x.example/%2565mail",
        "http://x.example/path",
    ):
        with pytest.raises(PrivacyError):
            safe_url(url)


@pytest.mark.parametrize("kind", ["missing", "reordered", "old_execution", "query"])
def test_result_complete_ordered_binding_revalidated_after_construct(kind):
    a, b = config(), config(provider_code="SOURCE:PROVIDER:PEER")
    p = plan(a, b)
    ex, _ = assembly(
        (a, b), {a.provider_code: (exchange(a, p),), b.provider_code: (exchange(b, p),)}
    )
    with ex:
        result = ex.execute(p)
    updates = {
        "missing": {"providers": result.providers[:1]},
        "reordered": {"providers": result.providers[::-1]},
        "old_execution": {"execution_checksum": "sha256:" + "0" * 64},
        "query": {
            "providers": (
                result.providers[0].model_copy(update={"query": result.providers[1].query}),
                result.providers[1],
            )
        },
    }[kind]
    forged = FederatedResult.model_construct(**(vars(result) | updates))
    with pytest.raises(ValueError):
        FederatedResult.model_validate(forged)


def test_explicit_persisted_cannot_masquerade_as_live():
    live = success().providers[0]
    persisted = live.persisted()
    assert persisted.hit_count == 1 and persisted.hit_set_checksum
    assert "hits" not in persisted.model_dump()
    with pytest.raises(ValueError):
        PersistedProviderOutcome.model_validate_json(live.model_dump_json())
    with pytest.raises(ValueError):
        ProviderOutcome.model_validate_json(persisted.model_dump_json())
    with pytest.raises(ValueError):
        ProviderOutcome.model_validate(live.model_copy(update={"hits": (), "hit_count": 99}))
    assert FederatedResult.model_validate_json(success().model_dump_json()).status == "succeeded"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("complete", "succeeded"),
        ("empty", "empty"),
        ("provider_error", "failed"),
        ("cancelled", "cancelled"),
        ("timeout", "timeout"),
        ("request_budget", "budget_exhausted"),
        ("hit_budget", "budget_exhausted"),
        ("page_limit", "budget_exhausted"),
        ("duplicate_page", "failed"),
    ],
)
def test_single_rule_for_every_stop_reason(reason, expected):
    source = success().providers[0]
    outcome = ProviderOutcome.model_validate(
        source.model_copy(
            update={
                "stop_reason": reason,
                "hits": source.hits if reason == "complete" else (),
                "hit_count": 1 if reason == "complete" else 0,
                "error_category": "http" if reason == "provider_error" else None,
            }
        )
    )
    assert derive_federated_status((outcome,)) == expected
    assert derive_federated_status((outcome.persisted(),)) == expected
    if reason not in ("complete", "empty"):
        assert derive_federated_status((source, outcome)) == "partial"


def registered_fixture(c):
    from apps.knowledge_provider_registration import register_source_provider
    from db.knowledge import KnowledgeSnapshot, KnowledgeStatus

    class Registry:
        def put(self, record):
            return KnowledgeSnapshot(record=record, status=KnowledgeStatus.DRAFT, revision=0)

    return register_source_provider(c, Registry(), synthetic=True)


def test_service_rejects_other_valid_execution_before_commit():
    c = config()
    p = plan(c)
    changed = plan(c, total_max_hits=2)
    ex, _ = assembly((c,), {c.provider_code: (exchange(c, changed),)})
    with ex:
        wrong = ex.execute(changed)

        snapshot = registered_fixture(c)

        class Repository:
            def get(self, ref):
                assert ref == snapshot.record.reference()
                return snapshot

            def lookup_search(self, *args):
                return None

            def commit_search(self, *args):
                pytest.fail("forged result committed")

        app = KnowledgeSearchService(ex, Repository(), (snapshot,))
        ex.execute = lambda *args, **kwargs: wrong
        with pytest.raises(ValueError, match="does not belong"):
            app.search(p, idempotency_key="synthetic-forgery")


def test_direct_query_model_rejects_uncontrolled_template_and_public_status_revalidates():
    from db.knowledge import SearchQuery, seal

    result = success()
    query = result.providers[0].query.model_copy(update={"texts": ("unreviewed arbitrary words",)})
    provider = registered_fixture(config()).record
    ref = provider.reference()
    with pytest.raises(ValueError, match="patient-free template"):
        seal(
            SearchQuery,
            code="SOURCE:QUERY:UNCONTROLLED",
            version="1.0.0",
            provider_ref=ref,
            provider_snapshot=provider,
            query_plan=result.binding.plan,
            provider_query=query,
            query_text=query.texts[0],
            config_version="4.0.0",
        )
    with pytest.raises(ValueError):
        derive_federated_status(
            (result.providers[0].model_copy(update={"hits": (), "hit_count": 99}),)
        )

"""Minimal independently reproducible counterexamples; synthetic values only."""

from urllib.parse import quote

import pytest

from packages.knowledge_search import FederatedResult, KnowledgeQueryPlan
from packages.knowledge_search.models import ProviderOutcome, safe_text
from tests.knowledge_search.support import assembly, config, exchange, plan


@pytest.mark.parametrize(
    "changes",
    [
        {"population": ("patient john smith mrn 12345678",)},
        {"source_version": "date of birth 1970-01-02"},
        {"anatomy": ("current patient reports blurred vision since yesterday",)},
        {"modalities": ("张三 住院号 12345678",)},
    ],
)
def test_red_patient_plan(changes):
    with pytest.raises(ValueError):
        plan(config(), **changes)


@pytest.mark.parametrize(
    "payload",
    [
        quote(
            quote(
                quote(quote("https://x.example?email=fixture@example.invalid", safe=""), safe=""),
                safe="",
            ),
            safe="",
        ),
        "ｅｍａｉｌ＝fixture＠example.invalid",
        "email&#61;fixture&#64;example.invalid",
        "Authorization&#58; secret",
    ],
)
def test_red_encoded_text(payload):
    with pytest.raises(ValueError):
        safe_text(payload)


def success():
    c = config()
    p = plan(c)
    executor, _ = assembly((c,), {c.provider_code: (exchange(c, p),)})
    with executor:
        return executor.execute(p)


def test_red_federated_status():
    with pytest.raises(ValueError):
        FederatedResult.model_validate(success().model_copy(update={"status": "failed"}))


def test_red_live_count():
    with pytest.raises(ValueError):
        ProviderOutcome.model_validate(
            success().providers[0].model_copy(update={"hits": (), "hit_count": 99})
        )


def test_red_duplicate_provider():
    result = success()
    with pytest.raises(ValueError):
        FederatedResult.model_validate(
            result.model_copy(update={"providers": result.providers * 2})
        )


def test_red_old_plan_binding():
    with pytest.raises(ValueError):
        FederatedResult.model_validate(
            success().model_copy(update={"plan_checksum": "sha256:" + "0" * 64})
        )


def test_red_constructed_patient_plan_revalidated():
    p = plan(config())
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate(p.model_copy(update={"population": ("MRN 12345678",)}))

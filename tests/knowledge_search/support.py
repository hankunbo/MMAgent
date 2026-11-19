"""Offline transport fixtures exercise real 04B1 adapters and the 04B2 governor."""

import json
from pathlib import Path

from packages.knowledge_connectors import (
    ConnectorConfiguration,
    FakeClock,
    FixtureExchange,
    ProviderConfig,
    RecordedTransport,
    SafeHTTPClient,
    load_configuration,
)
from packages.knowledge_search import (
    CodedTerm,
    FederatedSearch,
    KnowledgeQueryPlan,
    ProviderBudget,
    load_rate_policy,
)
from packages.knowledge_search.planning import provider_query


def config(adapter="europe_pmc", **changes):
    current = next(
        p
        for p in load_configuration(Path("configs/knowledge_sources.toml")).providers
        if p.adapter == adapter
    )
    return ProviderConfig.model_validate(current.model_copy(update=changes))


def policy():
    return load_rate_policy(Path("configs/knowledge_search.toml"))


def plan(*configs, **changes):
    base = dict(
        goal="GOAL:TEST:EVIDENCE",
        claim="CLAIM:TEST:ASSOCIATION",
        evidence_need="association",
        diseases=(CodedTerm(code="CONCEPT:TEST:CONDITION", label="synthetic condition"),),
        population=("adults",),
        source_types=("original_research",),
        providers=tuple(
            ProviderBudget(provider_code=c.provider_code, page_size=2, max_pages=3) for c in configs
        ),
    )
    return KnowledgeQueryPlan(**(base | changes))


def exchange(
    cfg,
    query_plan,
    ids=("123",),
    *,
    cursor="*",
    next_cursor=None,
    status=200,
    headers=(),
    error=None,
    title="Synthetic discovery",
):
    text = provider_query(query_plan, cfg, policy()).texts[0]
    budget = next(b for b in query_plan.providers if b.provider_code == cfg.provider_code)
    body = {
        "hitCount": len(ids),
        "resultList": {
            "result": [{"source": "MED", "id": uid, "title": title, "pmid": uid} for uid in ids]
        },
    }
    if next_cursor is not None:
        body["nextCursorMark"] = next_cursor
    from urllib.parse import urlsplit

    return FixtureExchange(
        host=urlsplit(cfg.search_endpoint).hostname,
        path=urlsplit(cfg.search_endpoint).path,
        query=(
            ("query", text),
            ("format", "json"),
            ("resultType", "core"),
            ("pageSize", str(budget.page_size)),
            ("cursorMark", cursor),
        ),
        status=status,
        headers=(("content-type", "application/json"), *headers),
        body_text=json.dumps(body),
        error=error,
    )


def assembly(configs, tapes, *, clock=None, environment=None):
    transports = {}

    def factory(cfg, governor):
        transport = RecordedTransport(tapes.get(cfg.provider_code, ()))
        transports[cfg.provider_code] = transport
        return SafeHTTPClient(
            cfg,
            transport,
            resolver=lambda host: ("93.184.216.34",),
            clock=governor.clock,
            governor=governor,
            environment=environment
            or (
                lambda name: (
                    "fixture@example.invalid"
                    if name == cfg.contact_email_env
                    else "synthetic-secret-key"
                )
            ),
        )

    executor = FederatedSearch(
        ConnectorConfiguration(config_version="2.0.0", providers=tuple(configs)),
        policy(),
        factory,
        clock=clock or FakeClock(),
    )
    return executor, transports

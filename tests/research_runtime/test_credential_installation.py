"""Deployment-only connection resolution, no real credentials or actual databases."""

from types import SimpleNamespace

import pytest
from psycopg.conninfo import conninfo_to_dict

from packages.integrations.medgraph.models import AdapterConfig
from packages.integrations.medgraph.source import PostgresSource
from packages.research_runtime.deployment import medgraph_readonly_dsn


def test_native_resolver_uses_original_target_without_mutating_environment(monkeypatch):
    monkeypatch.delenv("MEDGRAPH_READONLY_DSN", raising=False)
    boundary = SimpleNamespace(
        inputs=lambda: (
            {"host": "127.0.0.1", "port": 5432},
            {"POSTGRES_DB": "medgraph", "POSTGRES_LLM_PASSWORD": "synthetic-only-secret"},
        )
    )
    resolved = conninfo_to_dict(medgraph_readonly_dsn(boundary))
    assert resolved == dict(
        host="127.0.0.1",
        port="5432",
        dbname="medgraph",
        user="medgraph_llm",
        password="synthetic-only-secret",
    )
    import os

    assert "MEDGRAPH_READONLY_DSN" not in os.environ


def test_native_resolver_rejects_different_target():
    boundary = SimpleNamespace(
        inputs=lambda: (
            {"host": "192.0.2.1", "port": 5432},
            {"POSTGRES_DB": "medgraph", "POSTGRES_LLM_PASSWORD": "synthetic-only-secret"},
        )
    )
    with pytest.raises(ValueError, match="research_original_medgraph_target_required"):
        medgraph_readonly_dsn(boundary)


def test_source_resolves_at_each_connection_and_preserves_readonly_role(monkeypatch):
    calls, statements = [], []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, statement, params=None):
            statements.append(statement)

    def connect(dsn, **kwargs):
        calls.append(dsn)
        return Connection()

    monkeypatch.setattr("packages.integrations.medgraph.source.psycopg.connect", connect)
    monkeypatch.delenv("MEDGRAPH_READONLY_DSN", raising=False)
    observed = iter(("synthetic:first", "synthetic:second"))
    source = PostgresSource(
        AdapterConfig(
            version="1.0.0", source_code="LOCAL_MEDGRAPH", dsn_env="MEDGRAPH_READONLY_DSN"
        ),
        dsn_resolver=lambda: next(observed),
    )
    with source.transaction():
        pass
    with source.transaction():
        pass
    assert calls == ["synthetic:first", "synthetic:second"]
    assert statements.count("SET LOCAL ROLE medgraph_reader") == 2
    assert statements.count("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY") == 2

"""Unmodified structural SQL via psql; only session-owned PostgreSQL is initialized."""

from pathlib import Path

import pytest
from psycopg.conninfo import make_conninfo

from db import manage
from packages.integrations.medgraph import (
    AdapterConfig,
    MedgraphCatalogReader,
    PostgresSource,
    ProjectionPolicy,
    SnapshotStore,
)
from tests.database.knowledge_support_04a import free_port, sql

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def medgraph_database(database_lease):
    with database_lease.cluster(free_port()):
        admin = manage.test_target("bootstrap")
        # Precreate fixed role in our exclusive cluster, preserve independent ownership.
        sql(admin, "CREATE ROLE medgraph_owner NOLOGIN; CREATE ROLE medgraph_reader NOLOGIN")
        from psycopg import sql as pgsql

        from tests.database.knowledge_support_04a import connect

        with connect(admin) as connection:
            connection.execute(
                pgsql.SQL("GRANT CREATE ON DATABASE {} TO medgraph_owner").format(
                    pgsql.Identifier(admin.database)
                )
            )
        for name in (
            "init/001_schema.sql",
            "migrations/003_execution_registry.sql",
            "migrations/005_execution_validation.sql",
        ):
            # psql consumes \set; the original bytes and explicit transactions remain intact.
            manage._psql(
                admin,
                "SET ROLE medgraph_owner;\n"
                + (ROOT / "algorithm-capability-db/sql" / name).read_text(),
            )
        sql(
            admin,
            "CREATE ROLE medgraph_fixture_reader LOGIN; "
            "GRANT medgraph_reader TO medgraph_fixture_reader",
        )
        yield admin


@pytest.fixture
def adapter(medgraph_database, monkeypatch, tmp_path):
    admin = medgraph_database
    from tests.database.knowledge_support_04a import connect

    with connect(admin) as connection:
        connection.execute(Path(__file__).with_name("seed.sql").read_text())
    monkeypatch.setenv(
        "MEDGRAPH_06A_TEST_DSN",
        make_conninfo(
            host=admin.host, port=admin.port, dbname=admin.database, user="medgraph_fixture_reader"
        ),
    )
    config = AdapterConfig(source_code="FIXTURE", dsn_env="MEDGRAPH_06A_TEST_DSN")
    source = PostgresSource(config)
    reader = MedgraphCatalogReader(source, SnapshotStore(tmp_path), config, ProjectionPolicy())
    return reader, admin

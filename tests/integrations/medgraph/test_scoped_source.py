"""Original owner exact scope: no model/API/actual database work."""

import pytest

from packages.integrations.medgraph import (
    AdapterConfig,
    AdapterError,
    MedgraphCatalogReader,
    ProjectionPolicy,
)
from packages.integrations.medgraph.scoped_source import DirectToolSelection, ExactPostgresSource
from tests.database.knowledge_support_04a import sql


def source(version="1.0.0"):
    selected = DirectToolSelection(
        tool_code="fixture_tool", version=version, implementation_key="same_key"
    )
    config = AdapterConfig(
        source_code=ExactPostgresSource.source_code(selected), dsn_env="MEDGRAPH_06A_TEST_DSN"
    )
    return ExactPostgresSource(config, selected)


@pytest.mark.parametrize("bad", ["latest", "LATEST", " latest", "", "1.0.0 ", None])
def test_selection_is_explicit(bad):
    with pytest.raises((AdapterError, ValueError)):
        source(bad)


def test_scope_is_bound_to_original_configuration():
    a, b = source(), source("2.0.0")
    assert a.config.source_code != b.config.source_code
    with pytest.raises(AdapterError, match="exact_scope_config_binding"):
        ExactPostgresSource(a.config, b.selection)
    with pytest.raises(AdapterError, match="outside_deployment_selected_scope"):
        a.exact_manifest("fixture_tool", "2.0.0", "same_key")


def test_scoped_original_pg_retains_selected_bounds_and_current_revocation(adapter):
    full, admin = adapter
    sql(
        admin,
        "INSERT INTO medgraph.implementation_resource "
        "(implementation_id,resource_key,resource_role,uri,byte_size,metadata) "
        "SELECT 2,'extra_'||g,'source','file:///synthetic/extra_'||g,1,'{}' "
        "FROM generate_series(1,306) g",
    )
    with pytest.raises(AdapterError, match="invalid_or_unsafe_native_document"):
        full.source.capture()
    exact = source()
    scoped = MedgraphCatalogReader(exact, full.store, exact.config, ProjectionPolicy())
    catalog = scoped.capture_snapshot(synthetic=True)
    native = scoped.read_native(catalog.reference())
    assert len(native.tools) == len(native.versions) == len(native.implementations) == 1
    assert len(native.ports) == len(native.artifacts) == len(native.resources) == 2
    assert not native.workflows and not native.steps and not native.edges
    tool = catalog.tools[0]
    assert not tool.enabled and not tool.workflows  # no automatic research review
    manifest = scoped.read_manifest(catalog.reference(), tool.tool_ref, tool.implementation_ref)
    assert manifest.native_manifest == exact.exact_manifest("fixture_tool", "1.0.0", "same_key")
    assert not scoped.current_safety(catalog.reference()).disabled_implementations
    with pytest.raises(AdapterError, match="invalid_or_unsafe_native_document"):
        source("2.0.0").capture()  # selected oversized manifest is never ignored
    assert not source("9.0.0").capture().tools
    with exact.transaction() as connection:
        assert (
            connection.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"]
            == "on"
        )
        assert (
            connection.execute("SELECT current_user").fetchone()["current_user"]
            == "medgraph_reader"
        )
    sql(admin, "UPDATE medgraph.tool_version SET enabled=false WHERE id=1")
    assert scoped.current_safety(catalog.reference()).disabled_implementations == (
        tool.implementation_ref,
    )
    sql(admin, "UPDATE medgraph.tool_version SET enabled=true WHERE id=1")
    sql(
        admin,
        "UPDATE medgraph.implementation_resource SET sha256=repeat('b',64) "
        "WHERE implementation_id=1",
    )
    assert scoped.current_safety(catalog.reference()).disabled_implementations == (
        tool.implementation_ref,
    )
    sql(admin, "UPDATE medgraph.tool_implementation SET enabled=false WHERE id=1")
    assert not exact.capture().manifests
    assert scoped.current_safety(catalog.reference()).disabled_implementations == (
        tool.implementation_ref,
    )
    with pytest.raises(AdapterError, match="not_found"):
        exact.exact_manifest("fixture_tool", "1.0.0", "same_key")
    sql(admin, "DELETE FROM medgraph.tool_implementation WHERE id=1")
    assert not exact.capture().tools
    assert scoped.current_safety(catalog.reference()).disabled_implementations == (
        tool.implementation_ref,
    )
    assert scoped.read_snapshot(catalog.reference()) == catalog  # exact historical bytes

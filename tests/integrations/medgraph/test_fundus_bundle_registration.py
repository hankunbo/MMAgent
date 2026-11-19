"""Append-only native version tests; no real model or actual research database."""

import pytest
from psycopg.types.json import Jsonb

from apps import fundus_bundle_registry as registry
from packages.grounding.models import canonical, validate
from packages.integrations.medgraph.models import Tool
from packages.object_store import checksum
from tests.database.knowledge_support_04a import connect


def declaration(tmp_path, tool=None):
    resource = tmp_path / "synthetic-inventory.json"
    resource.write_bytes(b'{"synthetic":true}')
    tool = tool or Tool(
        id=1,
        code=registry.CODE,
        name_zh="synthetic tool",
        kind="segmenter",
        description="old row",
        metadata={"old_installation": "unchanged"},
    )
    return validate(
        registry.BundleRegistration,
        dict(
            tool=tool.model_dump(mode="json"),
            version_metadata={"research_only": True},
            manifest=dict(
                tool_code=registry.CODE,
                tool_name_zh=tool.name_zh,
                capability_kind=tool.kind,
                version=registry.VERSION,
                implementation_key=registry.IMPLEMENTATION,
                implementation_kind="python_cli",
                entrypoint_uri=resource.as_uri(),
                working_directory=str(tmp_path),
                runtime_environment={"synthetic": True},
                invocation={"shell": False, "argv": ["not-executed"], "timeout_seconds": 180},
                formula_spec=None,
                hardware_requirements={"concurrency": 1},
                provenance={"synthetic_fixture": True},
                validation_status="smoke_tested",
                validation_report={
                    "clinical_validated": False,
                    "formal_06_bundle_installed": False,
                    "intended_use": "research_only",
                },
                resources=[
                    dict(
                        resource_key="inventory",
                        resource_role="inventory_reference",
                        uri=resource.as_uri(),
                        sha256=checksum(resource.read_bytes())[7:],
                        byte_size=resource.stat().st_size,
                        required=True,
                        metadata={},
                    )
                ],
            ),
        ),
    )


def test_declaration_pin_and_local_resources(tmp_path):
    spec = declaration(tmp_path)
    path = tmp_path / "declaration.json"
    path.write_bytes(canonical(spec))
    pin = checksum(path.read_bytes())
    assert registry.load_verified(path, pin) == spec
    with pytest.raises(ValueError, match="declaration_drift"):
        registry.load_verified(path, "sha256:" + "0" * 64)
    resource = tmp_path / "synthetic-inventory.json"
    resource.write_bytes(b'{"synthetic":True}')  # same size, altered content
    with pytest.raises(ValueError, match="resource_drift"):
        registry.load_verified(path, pin)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "1.0.0"),
        ("validation_status", "validated"),
        ("implementation_key", "local_automorph_eight_wnet_v1"),
        ("validation_report", {"clinical_validated": True}),
        ("invocation", {"shell": True}),
        ("resources", []),
    ],
)
def test_declaration_rejects_upgrade_or_qualification_claim(tmp_path, field, value):
    data = declaration(tmp_path).model_dump(mode="json")
    data["manifest"][field] = value
    with pytest.raises(ValueError, match="fundus_bundle_registration_identity"):
        validate(registry.BundleRegistration, data)


def rows(connection):
    return {
        table: connection.execute(
            "SELECT to_jsonb(t) FROM medgraph." + table + " t ORDER BY id"
        ).fetchall()
        for table in (
            "algorithm_tool",
            "tool_version",
            "artifact_type",
            "tool_port",
            "tool_implementation",
            "implementation_resource",
        )
    }


def test_append_exact_readback_drift_rollback_and_original_projection(
    adapter, tmp_path, monkeypatch
):
    reader, admin = adapter
    with connect(admin) as c:
        c.execute("SET ROLE medgraph_owner")
        tool = declaration(tmp_path).tool
        tid = c.execute(
            "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind,description,metadata) "
            "VALUES(%s,%s,%s,%s,%s) RETURNING id",
            (tool.code, tool.name_zh, tool.kind, tool.description, Jsonb(tool.metadata)),
        ).fetchone()[0]
        tool = tool.model_copy(update={"id": tid})
        c.execute("INSERT INTO medgraph.tool_version(tool_id,version) VALUES(%s,'1.0.0')", (tid,))
        # Deliberately preserve the old output contract, not silently reinterpret it.
        c.execute(
            "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) VALUES(%s,'old',%s)",
            ("fundus:AutoMorphprediction_bundle", Jsonb({"schema_version": "old-directory"})),
        )
        spec = declaration(tmp_path, tool)
        before = rows(c)
        with monkeypatch.context() as patch:

            def reject_after_insert(*args):
                raise RuntimeError("post_insert_probe")

            patch.setattr(registry, "current", reject_after_insert)
            with pytest.raises(RuntimeError, match="post_insert_probe"):
                registry.register(c, spec)
        assert rows(c) == before
        with pytest.raises(RuntimeError, match="rollback_probe"):
            with c.transaction():
                c.execute(
                    "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                    "VALUES(%s,'incompatible output','{}')",
                    (registry.OUTPUT_CODE,),
                )
                incompatible = rows(c)
                with pytest.raises(ValueError, match="fundus_bundle_artifact_conflict"):
                    registry.register(c, spec)
                assert rows(c) == incompatible
                raise RuntimeError("rollback_probe")
        assert rows(c) == before
        assert registry.register(c, spec)["status"] == "registered"
        after = rows(c)
        for table, original in before.items():
            assert all(row in after[table] for row in original)
        assert registry.register(c, spec)["status"] == "already_registered"
        assert rows(c) == after  # timestamps and row contents unchanged
        assert registry.current(c, spec) == spec.manifest.model_dump(mode="json")
        vid = c.execute(
            "SELECT id FROM medgraph.tool_version WHERE tool_id=%s AND version=%s",
            (tid, registry.VERSION),
        ).fetchone()[0]
        mutations = [
            ("UPDATE medgraph.tool_version SET enabled=false WHERE id=%s", (vid,)),
            ("UPDATE medgraph.tool_version SET runtime='{}' WHERE id=%s", (vid,)),
            ("UPDATE medgraph.tool_version SET metadata='{}' WHERE id=%s", (vid,)),
            (
                "UPDATE medgraph.tool_version SET metadata='{\"research_only\":1}' WHERE id=%s",
                (vid,),
            ),
            ("UPDATE medgraph.tool_port SET required=false WHERE tool_version_id=%s", (vid,)),
            (
                "UPDATE medgraph.tool_port SET constraints='{\"research_only\":1}' "
                "WHERE tool_version_id=%s",
                (vid,),
            ),
            (
                "UPDATE medgraph.artifact_type SET schema_json='{}' WHERE code=%s",
                (registry.OUTPUT_CODE,),
            ),
            (
                "UPDATE medgraph.tool_implementation SET enabled=false WHERE tool_version_id=%s",
                (vid,),
            ),
            (
                "UPDATE medgraph.tool_implementation SET entrypoint_uri='file:///wrong' "
                "WHERE tool_version_id=%s",
                (vid,),
            ),
            (
                "UPDATE medgraph.tool_implementation SET "
                "invocation=jsonb_set(invocation,'{shell}','0') WHERE tool_version_id=%s",
                (vid,),
            ),
            (
                "UPDATE medgraph.tool_implementation SET "
                "validation_report=jsonb_set(validation_report,'{clinical_validated}','0') "
                "WHERE tool_version_id=%s",
                (vid,),
            ),
            (
                "DELETE FROM medgraph.implementation_resource WHERE implementation_id IN "
                "(SELECT id FROM medgraph.tool_implementation WHERE tool_version_id=%s)",
                (vid,),
            ),
            ("UPDATE medgraph.algorithm_tool SET metadata='{}' WHERE id=%s", (tid,)),
        ]
        for statement, args in mutations:
            with pytest.raises(RuntimeError, match="rollback_probe"):
                with c.transaction():
                    c.execute(statement, args)
                    altered = rows(c)
                    with pytest.raises(ValueError, match="fundus_bundle_.*drift"):
                        registry.register(c, spec)
                    assert rows(c) == altered
                    raise RuntimeError("rollback_probe")
            assert rows(c) == after
        c.execute("SET ROLE medgraph_fixture_reader")
        with pytest.raises(ValueError, match="native_owner_required"):
            registry.register(c, spec)
    native = reader.source.capture()
    assert any(m.implementation_key == registry.IMPLEMENTATION for m in native.manifests)
    catalog = reader.capture_snapshot(synthetic=True)
    matches = [
        t for t in catalog.tools if t.implementation_ref.code.endswith(registry.IMPLEMENTATION)
    ]
    # Projection uses exact hashed native references, never a registration-as-approval shortcut.
    if not matches:
        matches = [
            t
            for t in catalog.tools
            if reader.read_manifest(
                catalog.reference(), t.tool_ref, t.implementation_ref
            ).native_manifest.implementation_key
            == registry.IMPLEMENTATION
        ]
    assert len(matches) == 1 and not matches[0].enabled
    manifest = reader.read_manifest(
        catalog.reference(), matches[0].tool_ref, matches[0].implementation_ref
    )
    assert "validation_gates_missing" in manifest.binding.gaps
    with connect(admin) as c:
        c.execute("SET ROLE medgraph_owner")
        c.execute(
            "UPDATE medgraph.tool_implementation SET enabled=false WHERE implementation_key=%s",
            (registry.IMPLEMENTATION,),
        )
    assert (
        matches[0].implementation_ref
        in reader.current_safety(catalog.reference()).disabled_implementations
    )
    assert reader.read_snapshot(catalog.reference()) == catalog

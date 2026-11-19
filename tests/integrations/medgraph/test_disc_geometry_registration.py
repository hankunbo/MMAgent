"""Two geometry tools: isolated real PG, no production catalog mutation."""

import json

import pytest
from psycopg import errors

from apps import disc_geometry_tool_registry as registry
from tests.database.knowledge_support_04a import connect


def test_atomic_native_registration_manifests_and_repeat(medgraph_database, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "LOCK", tmp_path / "lock.json")
    evidence = tmp_path / "synthetic-registration-contract.xml"
    evidence.write_text(
        '<testsuite name="synthetic_registration_fixture_not_real_validation" '
        'tests="20" failures="0" errors="0" skipped="0"/>'
    )
    spec = registry.prepare(evidence)
    assert registry.load_verified() == spec
    assert registry.prepare(evidence) == spec
    with connect(medgraph_database) as c:
        c.execute("SET ROLE medgraph_owner")
        before = c.execute("SELECT count(*) FROM medgraph.algorithm_tool").fetchone()[0]
        with pytest.raises(RuntimeError, match="owned_transaction_failure"), c.transaction():
            assert registry.register(c, spec)["status"] == "registered"
            raise RuntimeError("owned_transaction_failure")
        assert c.execute("SELECT count(*) FROM medgraph.algorithm_tool").fetchone()[0] == before
        assert registry.register(c, spec)["status"] == "registered"
        assert registry.register(c, spec)["status"] == "already_registered"
        assert c.execute("SELECT count(*) FROM medgraph.algorithm_tool").fetchone()[0] == before + 2
        for task, (code, _, _) in registry.TOOLS.items():
            manifest = c.execute(
                "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                (code, registry.VERSION, registry.IMPLEMENTATION),
            ).fetchone()[0]
            assert manifest["validation_status"] == "smoke_tested"
            assert manifest["formula_spec"] == registry.FORMULAS[task]
            assert manifest["invocation"]["argv"] == registry.argv_template(task)
            ports = c.execute(
                "SELECT direction,port_name FROM medgraph.v_tool_capability WHERE tool_code=%s",
                (code,),
            ).fetchall()
            assert len(ports) == (3 if task == "rim" else 2)
            disc, cup = tmp_path / "disc.png", tmp_path / "cup.png"
            disc.write_bytes(b"pixels are checked by the measurement runner")
            cup.write_bytes(b"same rule")
            output = tmp_path / task / "output.json"
            plan = registry.resolve(c, task, disc, cup if task == "rim" else None, output)
            assert not output.parent.exists()  # dry-run has no directory side effect
            assert plan["argv"][0] == str(registry.PYTHON)
        with pytest.raises(ValueError, match="state_or_content_conflict"):
            registry.register(c, {**spec, "altered": True})
        c.execute(
            "UPDATE medgraph.tool_implementation SET enabled=false WHERE implementation_key=%s",
            (registry.IMPLEMENTATION,),
        )
        with pytest.raises(ValueError, match="unavailable_or_changed"):
            registry.resolve(c, "ellipse", disc, None, output)
        with pytest.raises(ValueError, match="state_or_content_conflict"):
            registry.register(c, spec)
        c.execute("SET ROLE medgraph_fixture_reader")
        assert len(registry.rows(c)) == 2
        with pytest.raises(errors.InsufficientPrivilege), c.transaction():
            c.execute(
                "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind) "
                "VALUES('unauthorized','unauthorized','other')"
            )
    data = json.loads(registry.LOCK.read_bytes())
    data["resources"][0]["sha256"] = "0" * 64
    registry.LOCK.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resource_drift"):
        registry.load_verified()


@pytest.mark.parametrize(
    "field,value", [("failures", 1), ("errors", 1), ("skipped", 1), ("tests", 0)]
)
def test_failed_validation_does_not_freeze(field, value, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "LOCK", tmp_path / "lock.json")
    attributes = {"tests": 20, "failures": 0, "errors": 0, "skipped": 0, field: value}
    evidence = tmp_path / "synthetic.xml"
    evidence.write_text(
        "<testsuite " + " ".join(f'{k}="{v}"' for k, v in attributes.items()) + "/>"
    )
    with pytest.raises(ValueError, match="passing_geometry_test_evidence_required"):
        registry.prepare(evidence)
    assert not registry.LOCK.exists()

"""Seven native component tools: atomic insertion, exact manifests and idempotence."""

import subprocess
from pathlib import Path

import pytest
from psycopg import errors

from apps import slo_tool_registry as registry
from tests.database.knowledge_support_04a import connect


def fixture_spec():
    return {
        "lock": {
            "environment": {},
            "source_repository": "synthetic-contract-fixture",
            "upstream_commit": "fixture",
            "resources": [],
        },
        "lock_sha256": "a" * 64,
        "evidence": "synthetic-contract-fixture",
        "evidence_sha256": "b" * 64,
    }


def test_register_read_repeat_conflict_and_rollback(medgraph_database, monkeypatch, tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    monkeypatch.setattr(registry, "LOCK", lock)
    spec = fixture_spec()
    spec["lock_sha256"] = registry.digest(lock)
    with connect(medgraph_database) as c:
        c.execute("SET ROLE medgraph_owner")
        before = c.execute("SELECT count(*) FROM medgraph.algorithm_tool").fetchone()[0]
        with pytest.raises(RuntimeError, match="rollback_probe"):
            with c.transaction():
                assert registry.register(c, spec)["status"] == "registered"
                raise RuntimeError("rollback_probe")
        assert c.execute("SELECT count(*) FROM medgraph.algorithm_tool").fetchone()[0] == before
        assert registry.register(c, spec)["status"] == "registered"
        assert c.execute("SELECT count(*) FROM medgraph.algorithm_tool").fetchone()[0] == before + 7
        assert registry.register(c, spec)["status"] == "already_registered"
        for task, (code, _, _, extra) in registry.TOOLS.items():
            m = c.execute(
                "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                (code, registry.VERSION, registry.IMPLEMENTATION),
            ).fetchone()[0]
            assert m["validation_status"] == "smoke_tested"
            assert m["runtime_environment"]["device"] == "cpu"
            assert m["invocation"]["shell"] is False
            assert task in m["invocation"]["argv"]
            ports = c.execute(
                "SELECT direction,port_name,required FROM medgraph.v_tool_capability "
                "WHERE tool_code=%s",
                (code,),
            ).fetchall()
            assert len(ports) == len(extra) + 2 and all(p[2] for p in ports)
            assert not m["validation_report"]["clinical_validated"]
            image = tmp_path / "synthetic-input.png"
            image.write_bytes(b"input checked by runner, not dry-run")
            output_dir = tmp_path / task
            plan = registry.resolve_component(c, task, image, output_dir)
            assert plan["research_only"] and task in plan["argv"]
            assert plan["argv"][0] == str(registry.PYTHON)  # preserve venv, not base Python
            assert not output_dir.exists()
        with pytest.raises(ValueError, match="content_conflict"):
            registry.register(c, {**spec, "evidence_sha256": "c" * 64})
        c.execute(
            "UPDATE medgraph.tool_implementation SET enabled=false WHERE implementation_key=%s",
            (registry.IMPLEMENTATION,),
        )
        with pytest.raises(ValueError, match="state_conflict"):
            registry.register(c, spec)
        c.execute("SET ROLE medgraph_fixture_reader")
        assert len(registry.rows(c)) == 7
        with pytest.raises(errors.InsufficientPrivilege), c.transaction():
            c.execute(
                "INSERT INTO medgraph.algorithm_tool(code,name_zh,kind) VALUES('bad','bad','other')"
            )


def test_formula_suite_in_owned_model_environment():
    python = registry.PYTHON
    if not python.is_file():
        pytest.skip("SLO component environment is a separately deployed optional tool")
    result = subprocess.run(
        [str(python), "-m", "unittest", "-v", "test_runner"],
        cwd=Path(registry.RUNNER).parent,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr

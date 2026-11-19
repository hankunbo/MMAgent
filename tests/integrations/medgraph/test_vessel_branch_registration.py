"""Native Tool registration, rollback and disabled execution in an owned PG lease."""

import pytest

from apps import vessel_branch_tool as t
from tests.database.knowledge_support_04a import connect


def test_registration_rollback_repeat_and_state(medgraph_database, tmp_path, monkeypatch):
    monkeypatch.setattr(t, "LOCK", tmp_path / "installation.json")
    evidence = tmp_path / "synthetic-not-real-validation.xml"
    evidence.write_text(
        '<testsuite name="synthetic_fixture" tests="28" failures="0" errors="0" skipped="0"/>'
    )
    spec = t.prepare(evidence)
    assert t.load_verified() == spec
    with connect(medgraph_database) as c:
        c.execute("SET ROLE medgraph_owner")
        with pytest.raises(RuntimeError, match="abort"), c.transaction():
            t.register(c, spec)
            raise RuntimeError("abort")
        assert t.rows(c) == []
        assert t.register(c, spec)["status"] == "registered"
        assert t.register(c, spec)["status"] == "already_registered"
        m = c.execute(
            "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
            (t.CODE, t.VERSION, t.IMPLEMENTATION),
        ).fetchone()[0]
        assert m["invocation"]["argv"] == t.argv()
        assert m["formula_spec"] == t.FORMULA
        assert m["validation_status"] == "smoke_tested"
        vein, disc, output = (
            tmp_path / "vein.png",
            tmp_path / "disc.png",
            tmp_path / "uncreated" / "result.json",
        )
        vein.write_bytes(b"pixels validated by runner")
        disc.write_bytes(b"pixels validated by runner")
        assert t.resolve(c, vein, disc, output)["argv"][0] == str(t.PYTHON)
        assert not output.parent.exists()
        c.execute(
            "UPDATE medgraph.tool_implementation SET enabled=false WHERE implementation_key=%s",
            (t.IMPLEMENTATION,),
        )
        with pytest.raises(ValueError, match="unavailable"):
            t.resolve(c, vein, disc, output)
        with pytest.raises(ValueError, match="conflict"):
            t.register(c, spec)
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="resource_drift"):
        t.load_verified()


@pytest.mark.parametrize(
    "field,value", [("tests", 0), ("failures", 1), ("errors", 1), ("skipped", 1)]
)
def test_failed_evidence_cannot_freeze(field, value, tmp_path, monkeypatch):
    monkeypatch.setattr(t, "LOCK", tmp_path / "lock.json")
    attributes = {"tests": 28, "failures": 0, "errors": 0, "skipped": 0, field: value}
    path = tmp_path / "evidence.xml"
    path.write_text("<testsuite " + " ".join(f'{k}="{v}"' for k, v in attributes.items()) + "/>")
    with pytest.raises(ValueError, match="passing_branch_tests_required"):
        t.prepare(path)
    assert not t.LOCK.exists()

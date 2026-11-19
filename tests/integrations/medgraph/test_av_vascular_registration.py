"""Native schema/manifest/current-state verification in an isolated PG lease."""

import pytest

from apps import av_vascular_tool as t
from tests.database.knowledge_support_04a import connect


def test_native_registration_atomic_repeat_and_current_disable(
    medgraph_database, tmp_path, monkeypatch
):
    monkeypatch.setattr(t, "LOCK", tmp_path / "installation.json")
    evidence = tmp_path / "synthetic-fixture-not-clinical.xml"
    evidence.write_text('<testsuite tests="30" failures="0" errors="0" skipped="0"/>')
    spec = t.prepare(evidence)
    assert t.load_verified() == spec
    with connect(medgraph_database) as c:
        c.execute("SET ROLE medgraph_owner")
        with pytest.raises(RuntimeError), c.transaction():
            t.register(c, spec)
            raise RuntimeError("rollback")
        assert not t.rows(c)
        assert t.register(c, spec)["status"] == "registered"
        assert t.register(c, spec)["status"] == "already_registered"
        m = c.execute(
            "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
            (t.CODE, t.VERSION, t.IMPLEMENTATION),
        ).fetchone()[0]
        assert m["invocation"]["argv"] == t.argv() and m["formula_spec"] == t.POLICY
        inputs = {k: tmp_path / (k + ".png") for k in ["artery_mask", "vein_mask", "disc_mask"]}
        for p in inputs.values():
            p.write_bytes(b"runner validates actual pixels")
        plan = t.resolve(c, inputs, tmp_path / "new.json")
        assert len(plan["input_checksums"]) == 3
        c.execute(
            "UPDATE medgraph.tool_implementation SET enabled=false WHERE implementation_key=%s",
            (t.IMPLEMENTATION,),
        )
        with pytest.raises(ValueError, match="unavailable"):
            t.resolve(c, inputs, tmp_path / "new.json")
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="resource_drift"):
        t.load_verified()

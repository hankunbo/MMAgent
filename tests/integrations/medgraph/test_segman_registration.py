"""Native Tool registration contract, isolated real PostgreSQL, no GPU."""

import pytest

from apps import segman_tool_registry as r
from tests.database.knowledge_support_04a import connect


def test_segman_atomic_register_exact_repeat_disabled_and_ports(
    medgraph_database, monkeypatch, tmp_path
):
    spec = {
        "environment": {},
        "resources": [],
        "output_contract": {"synthetic": True},
        "validation_report": {"synthetic": True, "clinical_validated": False},
    }
    with connect(medgraph_database) as c:
        c.execute("SET ROLE medgraph_owner")
        with pytest.raises(RuntimeError, match="rollback"), c.transaction():
            r.register(c, spec)
            raise RuntimeError("rollback")
        assert (
            c.execute(
                "SELECT count(*) FROM medgraph.algorithm_tool WHERE code=%s", (r.CODE,)
            ).fetchone()[0]
            == 0
        )
        assert r.register(c, spec)["status"] == "registered"
        assert r.register(c, spec)["status"] == "already_registered"
        m = r.manifest(c)
        assert m["invocation"]["argv"] == r.argv_template()
        assert m["invocation"]["shell"] is False
        assert m["validation_status"] == "smoke_tested"
        assert m["hardware_requirements"]["required"] is True
        ports = c.execute(
            "SELECT direction,port_name FROM medgraph.v_tool_capability WHERE tool_code=%s",
            (r.CODE,),
        ).fetchall()
        assert set(ports) == {(direction, port) for port, (direction, _) in r.PORTS.items()}
        from packages.dataset_adapters.reading import grayscale_image

        image = tmp_path / "input.png"
        image.write_bytes(grayscale_image((2, 3), b"123456"))
        monkeypatch.setattr(r, "load_verified", lambda: spec)
        output = tmp_path / "new_component_job"
        call = r.resolve_component(c, image, output)
        assert call["argv"][0] == str(r.PYTHON)
        assert set(call["outputs"]) == set(r.PORTS) - {"image"}
        assert call["shape_hw"] == [2, 3] and not output.exists()
        with pytest.raises(ValueError, match="manifest_drift"), c.transaction():
            c.execute(
                "UPDATE medgraph.tool_implementation SET invocation='{}'::jsonb "
                "WHERE implementation_key=%s",
                (r.IMPLEMENTATION,),
            )
            r.resolve_component(c, image, output)
        with pytest.raises(ValueError, match="conflict"):
            r.register(c, {**spec, "environment": {"changed": True}})
        c.execute(
            "UPDATE medgraph.tool_implementation SET enabled=false WHERE implementation_key=%s",
            (r.IMPLEMENTATION,),
        )
        with pytest.raises(ValueError, match="disabled"):
            r.manifest(c)
        with pytest.raises(ValueError, match="conflict"):
            r.register(c, spec)

"""Physical DELETE with real native schema in a session-owned database."""

from pathlib import Path

import pytest

from apps.remove_medagent_cup import remove_exact
from tests.database.knowledge_support_04a import connect


def seed(connection):
    connection.execute(Path(__file__).with_name("seed.sql").read_text())
    connection.execute(
        "UPDATE medgraph.algorithm_tool SET code='medical_sam_optic_cup_segmenter' WHERE id=1"
    )
    connection.execute("UPDATE medgraph.tool_version SET version='1.0.1' WHERE id=2")
    connection.execute(
        "UPDATE medgraph.tool_implementation SET implementation_key='local_medagent_pro' WHERE id=1"
    )
    connection.execute(
        "UPDATE medgraph.tool_implementation "
        "SET implementation_key='local_medagent_pro_complete_v2' WHERE id=2"
    )


def test_physical_delete_and_unrelated_records_preserved(medgraph_database):
    backups = []
    with connect(medgraph_database) as c:
        seed(c)
        result = remove_exact(c, backups.append)
        assert result["deleted_implementations"] == 2
        assert result["deleted_resource_bindings"] == 4
        assert c.execute("SELECT id FROM medgraph.tool_implementation ORDER BY id").fetchall() == [
            (3,)
        ]
        assert c.execute("SELECT count(*) FROM medgraph.workflow_step").fetchone()[0] == 1
        assert len(backups[0]["implementations"]) == 2
        assert remove_exact(c, backups.append)["already_absent"]


def test_backup_failure_prevents_deletion(medgraph_database):
    with connect(medgraph_database) as c:
        seed(c)

        def failed(_):
            raise OSError("owned fixture backup failure")

        with pytest.raises(OSError):
            remove_exact(c, failed)
        assert c.execute("SELECT count(*) FROM medgraph.tool_implementation").fetchone()[0] == 3

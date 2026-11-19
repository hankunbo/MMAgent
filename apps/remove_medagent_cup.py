"""Physical deletion of exactly two native callable rows, not enabled=false.

Keep tool/version descriptors referenced by historical runs; delete the two
implementations and their resource bindings only. Does not remove model files.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import psycopg

from packages.integrations.medgraph.retirement import RETIRED_IMPLEMENTATIONS


def remove_exact(connection, save_backup):
    connection.execute(
        "LOCK TABLE medgraph.tool_implementation, medgraph.implementation_resource "
        "IN SHARE ROW EXCLUSIVE MODE"
    )
    refs = connection.execute(
        "SELECT conrelid::regclass::text FROM pg_constraint WHERE contype='f' "
        "AND confrelid='medgraph.tool_implementation'::regclass"
    ).fetchall()
    if {r[0] for r in refs} != {"medgraph.implementation_resource"}:
        raise ValueError("unexpected dependent table; inspect before deleting")
    query = (
        "SELECT t.code,v.version,i.implementation_key,to_jsonb(i) "
        "FROM medgraph.algorithm_tool t JOIN medgraph.tool_version v ON v.tool_id=t.id "
        "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id ORDER BY i.id"
    )
    before = connection.execute(query).fetchall()
    targets = [r for r in before if tuple(r[:3]) in RETIRED_IMPLEMENTATIONS]
    if not targets:
        return {"deleted_implementations": 0, "already_absent": True}
    if {tuple(r[:3]) for r in targets} != RETIRED_IMPLEMENTATIONS:
        raise ValueError("partial target set; inspect before deleting")
    ids = [r[3]["id"] for r in targets]
    resources = connection.execute(
        "SELECT to_jsonb(r) FROM medgraph.implementation_resource r ORDER BY id"
    ).fetchall()
    versions = connection.execute(
        "SELECT to_jsonb(v) FROM medgraph.tool_version v ORDER BY id"
    ).fetchall()
    save_backup(
        {
            "implementations": targets,
            "resources": [r[0] for r in resources if r[0]["implementation_id"] in ids],
        }
    )
    deleted_resources = connection.execute(
        "DELETE FROM medgraph.implementation_resource WHERE implementation_id=ANY(%s)", (ids,)
    ).rowcount
    deleted = connection.execute(
        "DELETE FROM medgraph.tool_implementation WHERE id=ANY(%s)", (ids,)
    ).rowcount
    assert deleted == 2
    assert connection.execute(query).fetchall() == [r for r in before if r not in targets]
    assert connection.execute(
        "SELECT to_jsonb(r) FROM medgraph.implementation_resource r ORDER BY id"
    ).fetchall() == [r for r in resources if r[0]["implementation_id"] not in ids]
    assert (
        connection.execute("SELECT to_jsonb(v) FROM medgraph.tool_version v ORDER BY id").fetchall()
        == versions
    )
    for identity in sorted(RETIRED_IMPLEMENTATIONS):
        assert (
            connection.execute(
                "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)", identity
            ).fetchone()[0]
            is None
        )
    return {
        "deleted_implementations": deleted,
        "deleted_resource_bindings": deleted_resources,
        "historical_version_descriptors_preserved": True,
        "model_files_deleted": False,
    }


def private_json(path, data):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", required=True)
    parser.parse_args()
    from apps.research_single_case import credential_boundary

    b = credential_boundary()
    state, values = b.inputs()
    output = Path(tempfile.mkdtemp(prefix="cup-db-delete-", dir="/opt/data/test/MMAgent/.local"))
    try:
        with psycopg.connect(
            host=state["host"],
            port=state["port"],
            dbname="medgraph",
            user=values["POSTGRES_USER"],
            password=values["POSTGRES_PASSWORD"],
            connect_timeout=3,
            options="-c statement_timeout=10000 -c lock_timeout=5000",
        ) as c:
            assert (
                c.execute("SELECT system_identifier::text FROM pg_control_system()").fetchone()[0]
                == state["cluster_identifier"]
            )
            result = remove_exact(c, lambda data: private_json(output / "before.json", data))
        result.update(committed=True, evidence_directory=str(output))
        private_json(output / "result.json", result)
        print(json.dumps(result))
        return 0
    except (psycopg.Error, ValueError):
        result = {
            "committed": False,
            "reason": "database_unavailable_or_delete_precondition_failed",
        }
        private_json(output / "not-completed.json", result)
        print(json.dumps(result))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

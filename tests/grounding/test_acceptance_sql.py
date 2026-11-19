"""05C SQL identity alignment counterexamples within VAL-001 ownership."""

import psycopg
import pytest
from psycopg.types.json import Jsonb

from db import manage
from packages.retrieval.knowledge.models import RetrievalContext
from tests.database.knowledge_support_04a import connect, sql
from tests.grounding.test_database import database as database
from tests.grounding.test_database import key
from tests.grounding.test_database import setup as setup


@pytest.mark.parametrize("field", ["kind", "schema_version", "code", "version", "checksum"])
def test_sql_mapping_identity_cannot_be_json_null(database, setup, field):
    mapping, writer, _, _, _ = setup
    writer.register_draft(mapping, RetrievalContext(), request_key=key())
    payload = mapping.model_dump(mode="json")
    payload["version"] = "9.0.0"
    payload[field] = None
    with connect(manage.test_target("grounding_writer")) as conn:
        conn.execute("SET ROLE mmagent_grounding_writer")
        # Failed assertions roll back the savepoint too; no invalid red-test row survives.
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction(force_rollback=True):
            conn.execute(
                "INSERT INTO grounding.mapping_version "
                "SELECT code,'9.0.0',checksum,policy_code,policy_version,policy_checksum,"
                "requirement_code,requirement_version,requirement_checksum,"
                "catalog_code,catalog_version,catalog_checksum,%s "
                "FROM grounding.mapping_version WHERE code=%s AND version=%s",
                (Jsonb(payload), mapping.code, mapping.version),
            )


@pytest.mark.parametrize("field", ["kind", "code", "version", "checksum", "type_compatibility"])
def test_sql_policy_identity_cannot_be_json_null(database, setup, field):
    _, writer, _, _, _ = setup
    policy = writer.policy
    payload = policy.model_dump(mode="json")
    payload["version"] = "9.0.0"
    payload[field] = None
    with connect(manage.test_target("grounding_writer")) as conn:
        conn.execute("SET ROLE mmagent_grounding_writer")
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction(force_rollback=True):
            conn.execute(
                "INSERT INTO grounding.policy VALUES(%s,'9.0.0',%s,%s)",
                (policy.code, policy.checksum, Jsonb(payload)),
            )


@pytest.mark.parametrize("field", ["code", "version", "sequence", "state", "reason"])
def test_sql_review_identity_cannot_be_json_null(database, setup, field):
    mapping, writer, _, _, _ = setup
    writer.register_draft(mapping, RetrievalContext(), request_key=key())
    payload = dict(
        mapping_ref=mapping.reference().model_dump(mode="json"),
        policy_ref=writer.policy.reference().model_dump(mode="json"),
        sequence=1,
        state="Active",
        reason="machine_pass",
        request_key=key(),
    )
    if field in {"code", "version"}:
        payload["mapping_ref"][field] = None
    else:
        payload[field] = None
    with connect(manage.test_target("grounding_writer")) as conn:
        conn.execute("SET ROLE mmagent_grounding_writer")
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction(force_rollback=True):
            conn.execute(
                "INSERT INTO grounding.review VALUES(%s,%s,1,%s)",
                (mapping.code, mapping.version, Jsonb(payload)),
            )


@pytest.mark.parametrize("channel", ["grounding_writer", "grounding_reader", "runtime"])
def test_grounding_dml_is_reserved_to_trusted_writer_service(database, channel):
    # OIDs avoid requiring schema USAGE merely to resolve names during ACL inspection.
    identifiers = sql(
        database.admin,
        "SELECT 'grounding.mapping_version'::regclass::oid,"
        "'runtime.run_grounding_lock'::regclass::oid,'audit.event'::regclass::oid",
    )[0]
    with connect(manage.test_target(channel)) as conn:
        conn.execute(
            "SET ROLE " + ("mmagent_runtime_app" if channel == "runtime" else "mmagent_" + channel)
        )
        row = conn.execute(
            "SELECT has_table_privilege(current_user,%s::oid,'INSERT'),"
            "has_table_privilege(current_user,%s::oid,'INSERT'),"
            "has_table_privilege(current_user,%s::oid,'INSERT')",
            identifiers,
        ).fetchone()
        assert row == (channel == "grounding_writer", False, False)

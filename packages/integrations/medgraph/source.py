"""Read-only PostgreSQL connection boundary using medgraph's reader surface."""

import os
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from packages.grounding.models import canonical

from .models import (
    AdapterConfig,
    AdapterError,
    ArtifactType,
    Edge,
    Implementation,
    NativeManifest,
    NativeSnapshot,
    Port,
    Resource,
    Step,
    Tool,
    ToolVersion,
    Workflow,
    parse,
)
from .retirement import retired

# These tables are explicitly SELECT-granted to medgraph_reader by 001/003.
# Fixed identifiers only; caller values always use query parameters.
TABLES = (
    ("artifacts", "artifact_type", ArtifactType),
    ("tools", "algorithm_tool", Tool),
    ("versions", "tool_version", ToolVersion),
    ("ports", "tool_port", Port),
    ("implementations", "tool_implementation", Implementation),
    ("resources", "implementation_resource", Resource),
    ("workflows", "workflow_template", Workflow),
    ("steps", "workflow_step", Step),
    ("edges", "workflow_edge", Edge),
)


class PostgresSource:
    def __init__(self, config: AdapterConfig, *, dsn_resolver=None):
        self.config = parse(AdapterConfig, config)
        # Optional deployment-owned resolver; never a request or serialized credential.
        self._dsn_resolver = dsn_resolver

    @contextmanager
    def transaction(self):
        cfg = self.config
        try:
            dsn = self._dsn_resolver() if self._dsn_resolver else os.environ[cfg.dsn_env]
            with psycopg.connect(
                dsn, connect_timeout=cfg.connect_timeout_seconds, row_factory=dict_row
            ) as connection:
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                connection.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(cfg.statement_timeout_ms),),
                )
                connection.execute(
                    "SELECT set_config('transaction_timeout', %s, true)",
                    (str(cfg.transaction_timeout_ms),),
                )
                connection.execute(
                    "SELECT set_config('idle_in_transaction_session_timeout', %s, true)",
                    (str(cfg.transaction_timeout_ms),),
                )
                # Restrict even a misconfigured privileged connection for subsequent SQL.
                connection.execute("SET LOCAL ROLE medgraph_reader")
                yield connection
        except (psycopg.Error, KeyError):
            raise AdapterError("medgraph_read_unavailable") from None

    def _rows(self, connection, statement, parameters):
        bounded = sql.SQL(
            "SELECT CASE WHEN octet_length(row_to_json(r)::text) <= %s "
            "THEN row_to_json(r) ELSE NULL END AS payload FROM ({}) r"
        ).format(statement)
        raw = connection.execute(bounded, (262144, *parameters)).fetchmany(self.config.max_rows + 1)
        if any(r["payload"] is None for r in raw):
            raise AdapterError("native_record_byte_limit")
        rows = [r["payload"] for r in raw]
        if len(rows) > self.config.max_rows:
            raise AdapterError("native_row_limit")
        if len(canonical(rows)) > self.config.max_snapshot_bytes:
            raise AdapterError("native_byte_limit")
        return rows

    def _capture(self, connection):
        data = {}
        for name, table, model in TABLES:
            columns = sql.SQL(", ").join(
                sql.Identifier(field.alias or name) for name, field in model.model_fields.items()
            )
            statement = sql.SQL("SELECT {} FROM medgraph.{} ORDER BY id LIMIT %s").format(
                columns, sql.Identifier(table)
            )
            data[name] = [
                parse(model, r)
                for r in self._rows(connection, statement, (self.config.max_rows + 1,))
            ]
        tools = {t.id: t for t in data["tools"]}
        versions = {v.id: v for v in data["versions"]}
        data["manifests"] = []
        for implementation in data["implementations"]:
            version = versions[implementation.tool_version_id]
            tool = tools[version.tool_id]
            row = connection.execute(
                "SELECT CASE WHEN octet_length(m::text) <= 262144 "
                "THEN m ELSE NULL END AS manifest, "
                "COALESCE(octet_length(m::text) > 262144, false) AS oversized "
                "FROM (SELECT medgraph.get_tool_execution_manifest(%s, %s, %s) AS m) f",
                (tool.code, version.version, implementation.implementation_key),
            ).fetchone()
            if row["oversized"]:
                raise AdapterError("native_record_byte_limit")
            manifest = row["manifest"]
            if implementation.enabled != (manifest is not None):
                raise AdapterError("native_manifest_visibility_drift")
            if manifest is not None:
                data["manifests"].append(parse(NativeManifest, manifest))
        if len(canonical(data)) > self.config.max_snapshot_bytes:
            raise AdapterError("native_byte_limit")
        return parse(NativeSnapshot, data)

    def capture(self) -> NativeSnapshot:
        with self.transaction() as connection:
            return self._capture(connection)

    def exact_manifest(self, tool_code: str, version: str, implementation_key: str):
        if retired(tool_code, version, implementation_key):
            raise AdapterError("implementation_withdrawn_by_project")
        if any(
            type(x) is not str or not x or x.lower() == "latest" or len(x) > 512
            for x in (tool_code, version, implementation_key)
        ):
            raise AdapterError("exact_native_identity_required")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT CASE WHEN octet_length(m::text) <= 262144 "
                "THEN m ELSE NULL END AS manifest, "
                "COALESCE(octet_length(m::text) > 262144, false) AS oversized "
                "FROM (SELECT medgraph.get_tool_execution_manifest(%s, %s, %s) AS m) f",
                (tool_code, version, implementation_key),
            ).fetchone()
            if row["oversized"]:
                raise AdapterError("native_record_byte_limit")
            value = row["manifest"]
            if value is None:
                raise AdapterError("exact_enabled_manifest_not_found")
            result = parse(NativeManifest, value)
            if (result.tool_code, result.version, result.implementation_key) != (
                tool_code,
                version,
                implementation_key,
            ):
                raise AdapterError("native_manifest_identity_drift")
            return result

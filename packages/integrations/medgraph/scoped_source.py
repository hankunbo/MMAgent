"""Exact direct-tool projection using the original read-only source boundary.

This deployment-selected scope is not a workflow catalog and grants no execution
qualification. Unrelated native records are not loaded or silently repaired.
Every current read repeats the same exact joins; selected malformed records fail
the unchanged native validators. The old whole-catalog reader is unchanged.
"""

from psycopg import sql
from pydantic import model_validator

from packages.grounding.models import canonical, digest

from .models import AdapterError, NativeManifest, NativeSnapshot, NativeValue, Text, parse
from .source import TABLES, PostgresSource


class DirectToolSelection(NativeValue):
    tool_code: Text
    version: Text
    implementation_key: Text

    @model_validator(mode="after")
    def exact(self):
        if any(
            value.lower() == "latest" or value != value.strip()
            for value in (self.tool_code, self.version, self.implementation_key)
        ):
            raise ValueError("exact_native_identity_required")
        return self


class ExactPostgresSource(PostgresSource):
    """One exact direct implementation; original transaction, role and limits."""

    def __init__(self, config, selection: DirectToolSelection, *, dsn_resolver=None):
        super().__init__(config, dsn_resolver=dsn_resolver)
        self.selection = parse(DirectToolSelection, selection)
        if self.config.source_code != self.source_code(self.selection):
            raise AdapterError("exact_scope_config_binding")

    @staticmethod
    def source_code(selection):
        # The original immutable envelope already binds AdapterConfig. Give the
        # exact scope its own stable identity without changing old config bytes.
        value = parse(DirectToolSelection, selection)
        return "EXACT_" + digest(value)[7:31].upper()

    def _capture(self, connection):
        selected = self.selection
        row = connection.execute(
            "SELECT t.id AS tool_id,v.id AS version_id,i.id AS implementation_id "
            "FROM medgraph.algorithm_tool t "
            "JOIN medgraph.tool_version v ON v.tool_id=t.id "
            "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id "
            "WHERE t.code=%s AND v.version=%s AND i.implementation_key=%s",
            (selected.tool_code, selected.version, selected.implementation_key),
        ).fetchone()
        # Missing identity remains an empty scoped snapshot so original
        # current_safety marks the old implementation unavailable, not current.
        data = {name: [] for name, _, _ in TABLES}
        data["manifests"] = []
        if row is None:
            return parse(NativeSnapshot, data)
        filters = {
            "artifacts": (
                "id IN (SELECT artifact_type_id FROM medgraph.tool_port WHERE tool_version_id=%s)",
                row["version_id"],
            ),
            "tools": ("id=%s", row["tool_id"]),
            "versions": ("id=%s", row["version_id"]),
            "ports": ("tool_version_id=%s", row["version_id"]),
            "implementations": ("id=%s", row["implementation_id"]),
            "resources": ("implementation_id=%s", row["implementation_id"]),
        }
        for name, table, model in TABLES:
            if name not in filters:
                continue  # No workflow claim or workflow dependency is projected.
            condition, value = filters[name]
            columns = sql.SQL(", ").join(
                sql.Identifier(field.alias or key) for key, field in model.model_fields.items()
            )
            statement = sql.SQL("SELECT {} FROM medgraph.{} WHERE {} ORDER BY id LIMIT %s").format(
                columns, sql.Identifier(table), sql.SQL(condition)
            )
            data[name] = [
                parse(model, record)
                for record in self._rows(connection, statement, (value, self.config.max_rows + 1))
            ]
        manifest_row = connection.execute(
            "SELECT CASE WHEN octet_length(m::text)<=262144 THEN m ELSE NULL END AS manifest,"
            " COALESCE(octet_length(m::text)>262144,false) AS oversized "
            "FROM (SELECT medgraph.get_tool_execution_manifest(%s,%s,%s) AS m) f",
            (selected.tool_code, selected.version, selected.implementation_key),
        ).fetchone()
        if manifest_row["oversized"]:
            raise AdapterError("native_record_byte_limit")
        manifest = manifest_row["manifest"]
        if data["implementations"][0].enabled != (manifest is not None):
            raise AdapterError("native_manifest_visibility_drift")
        if manifest is not None:
            data["manifests"].append(parse(NativeManifest, manifest))
        if len(canonical(data)) > self.config.max_snapshot_bytes:
            raise AdapterError("native_byte_limit")
        return parse(NativeSnapshot, data)

    def exact_manifest(self, tool_code, version, implementation_key):
        if (tool_code, version, implementation_key) != (
            self.selection.tool_code,
            self.selection.version,
            self.selection.implementation_key,
        ):
            raise AdapterError("outside_deployment_selected_scope")
        return super().exact_manifest(tool_code, version, implementation_key)

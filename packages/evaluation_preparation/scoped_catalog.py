"""Current safety snapshot for one explicitly installed standalone tool only.

Unrelated new native tools may exceed the historical whole-catalog wire limits.
This is not a complete catalog discovery port: unselected implementations are
absent/disabled to the original safety checker, never assumed valid.
"""

from psycopg import sql

from packages.integrations.medgraph.models import (
    AdapterError,
    NativeManifest,
    NativeSnapshot,
    parse,
)
from packages.integrations.medgraph.source import TABLES, PostgresSource


class SingleToolSource(PostgresSource):
    def __init__(self, config, *, identity, dsn_resolver):
        super().__init__(config, dsn_resolver=dsn_resolver)
        if len(identity) != 3 or any(
            not isinstance(v, str) or not v or len(v) > 512 for v in identity
        ):
            raise ValueError("scoped_native_identity")
        self.identity = tuple(identity)

    def capture(self):
        tool, version, implementation = self.identity
        tool_id = "SELECT id FROM medgraph.algorithm_tool WHERE code=%s"
        version_id = (
            "SELECT id FROM medgraph.tool_version WHERE tool_id=(" + tool_id + ") AND version=%s"
        )
        impl_id = (
            "SELECT id FROM medgraph.tool_implementation WHERE tool_version_id=("
            + version_id
            + ") AND implementation_key=%s"
        )
        predicates = {
            "artifacts": ("TRUE", ()),
            "tools": ("code=%s", (tool,)),
            "versions": ("tool_id=(" + tool_id + ") AND version=%s", (tool, version)),
            "implementations": (
                "tool_version_id=(" + version_id + ") AND implementation_key=%s",
                self.identity,
            ),
            "ports": ("tool_version_id=(" + version_id + ")", (tool, version)),
            "resources": ("implementation_id=(" + impl_id + ")", self.identity),
        }
        data = {}
        with self.transaction() as connection:
            for name, table, model in TABLES:
                if name not in predicates:
                    data[name] = []
                    continue
                predicate, params = predicates[name]
                columns = sql.SQL(", ").join(
                    sql.Identifier(f.alias or k) for k, f in model.model_fields.items()
                )
                statement = sql.SQL(
                    "SELECT {} FROM medgraph.{} WHERE {} ORDER BY id LIMIT %s"
                ).format(columns, sql.Identifier(table), sql.SQL(predicate))
                data[name] = [
                    parse(model, row)
                    for row in self._rows(
                        connection, statement, (*params, self.config.max_rows + 1)
                    )
                ]
            row = connection.execute(
                "SELECT CASE WHEN octet_length(m::text)<=262144 THEN m ELSE NULL END AS manifest, "
                "COALESCE(octet_length(m::text)>262144,false) AS oversized "
                "FROM (SELECT medgraph.get_tool_execution_manifest(%s,%s,%s) AS m) f",
                self.identity,
            ).fetchone()
            if row["oversized"]:
                raise AdapterError("native_record_byte_limit")
            manifest = row["manifest"]
            enabled = bool(data["implementations"] and data["implementations"][0].enabled)
            if enabled != (manifest is not None):
                raise AdapterError("native_manifest_visibility_drift")
            data["manifests"] = [parse(NativeManifest, manifest)] if manifest else []
        return parse(NativeSnapshot, data)

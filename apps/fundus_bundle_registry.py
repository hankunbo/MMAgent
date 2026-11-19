"""Append an operator-reviewed bundle version to the original native Tool schema.

No connection loader, migration, model execution or clinical authorization lives
here. The caller supplies the native owner's transaction and a pinned deployment
declaration. Runtime use still requires the original ResearchReview and 06 gates.
"""

from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from psycopg.types.json import Jsonb
from pydantic import Field, model_validator

from packages.grounding.models import canonical, digest, validate
from packages.integrations.medgraph.models import NativeManifest, NativeValue, Object, Tool
from packages.object_store import checksum
from packages.tool_execution.fundus_bundle import strict_json

CODE = "automorph_fundus_disc_cup_ensemble"
VERSION = "2.0.0"
IMPLEMENTATION = "local_automorph_single_file_v2"
INPUT_CODE = "fundus:AutoMorphimage"
OUTPUT_CODE = "fundus:AutoMorphBundleV1"
INPUT_SCHEMA = {"format": "image/jpeg", "modality": "color_fundus", "mode": "RGB"}
OUTPUT_SCHEMA = {
    "format": "application/json",
    "schema_version": "automorph-fundus-bundle/1.0.0",
    "masks": "512_grid_unmodified_labels",
}


class BundleRegistration(NativeValue):
    schema_version: Literal["fundus-bundle-registration/1.0.0"] = "fundus-bundle-registration/1.0.0"
    tool: Tool
    manifest: NativeManifest
    version_metadata: Object = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity(self):
        m = self.manifest
        if (
            self.tool.code != CODE
            or (m.tool_code, m.version, m.implementation_key) != (CODE, VERSION, IMPLEMENTATION)
            or (m.tool_name_zh, m.capability_kind) != (self.tool.name_zh, self.tool.kind)
            or m.implementation_kind != "python_cli"
            or m.formula_spec is not None
            or m.validation_status != "smoke_tested"
            or m.validation_report.get("clinical_validated") is not False
            or m.validation_report.get("formal_06_bundle_installed") is not False
            or m.validation_report.get("intended_use") != "research_only"
            or m.invocation.get("shell") is not False
            or not m.resources
            or len({r.resource_key for r in m.resources}) != len(m.resources)
            or len({r.uri for r in m.resources}) != len(m.resources)
            or [r.resource_key for r in m.resources] != sorted(r.resource_key for r in m.resources)
        ):
            raise ValueError("fundus_bundle_registration_identity")
        return self


def load_verified(path, expected_checksum):
    """Integrity of an operator-selected public declaration, not medical approval.

    The bounded native resources include the full environment inventory *file*,
    not every environment member.  The original trusted runner must verify that
    full inventory before any dispatch; this function does not replace it.
    """
    with Path(path).open("rb") as stream:
        raw = stream.read(262145)
    if checksum(raw) != expected_checksum:
        raise ValueError("fundus_bundle_declaration_drift")
    spec = validate(BundleRegistration, strict_json(raw, 262144))
    total = 0
    for resource in spec.manifest.resources:
        parsed = urlparse(resource.uri)
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or resource.sha256 is None
            or resource.byte_size is None
            or not resource.required
        ):
            raise ValueError("fundus_bundle_resource_declaration")
        item = Path(unquote(parsed.path))
        if not item.is_absolute() or item.is_symlink() or not item.is_file():
            raise ValueError("fundus_bundle_resource_path")
        total += resource.byte_size
        if total > 67108864 or item.stat().st_size != resource.byte_size:
            raise ValueError("fundus_bundle_resource_size")
        with item.open("rb") as stream:
            raw = stream.read(resource.byte_size + 1)
        if len(raw) != resource.byte_size or checksum(raw)[7:] != resource.sha256:
            raise ValueError("fundus_bundle_resource_drift")
    return spec


def _ports(connection, version_id):
    return connection.execute(
        "SELECT p.direction,p.port_name,a.code,a.schema_json,p.required,"
        "p.cardinality_min,p.cardinality_max,p.constraints "
        "FROM medgraph.tool_port p JOIN medgraph.artifact_type a ON a.id=p.artifact_type_id "
        "WHERE p.tool_version_id=%s ORDER BY p.direction,p.port_name",
        (version_id,),
    ).fetchall()


def _expected_ports():
    return [
        (direction, name, code, schema, True, 1, 1, {"research_only": True})
        for direction, name, code, schema in (
            ("input", "image", INPUT_CODE, INPUT_SCHEMA),
            ("output", "prediction_bundle", OUTPUT_CODE, OUTPUT_SCHEMA),
        )
    ]


def _tool(connection, spec):
    row = connection.execute(
        "SELECT id,code,name_zh,kind,description,metadata "
        "FROM medgraph.algorithm_tool WHERE code=%s",
        (CODE,),
    ).fetchone()
    expected = spec.tool
    if canonical(row) != canonical(
        (
            expected.id,
            expected.code,
            expected.name_zh,
            expected.kind,
            expected.description,
            expected.metadata,
        )
    ):
        raise ValueError("fundus_bundle_parent_tool_drift")


def current(connection, spec):
    """Exact current native state, including fields omitted by the SQL manifest."""
    spec = validate(BundleRegistration, spec.model_dump(mode="json"))
    _tool(connection, spec)
    m = spec.manifest
    version = connection.execute(
        "SELECT id,runtime,metadata,enabled FROM medgraph.tool_version "
        "WHERE tool_id=%s AND version=%s",
        (spec.tool.id, VERSION),
    ).fetchone()
    if version is None or canonical(version[1:]) != canonical(
        (m.runtime_environment, spec.version_metadata, True)
    ):
        raise ValueError("fundus_bundle_version_drift")
    states = connection.execute(
        "SELECT implementation_key,enabled FROM medgraph.tool_implementation "
        "WHERE tool_version_id=%s ORDER BY implementation_key",
        (version[0],),
    ).fetchall()
    if states != [(IMPLEMENTATION, True)] or canonical(_ports(connection, version[0])) != canonical(
        _expected_ports()
    ):
        raise ValueError("fundus_bundle_ports_or_state_drift")
    raw = connection.execute(
        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
        (CODE, VERSION, IMPLEMENTATION),
    ).fetchone()[0]
    if raw is None or canonical(validate(NativeManifest, raw)) != canonical(m):
        raise ValueError("fundus_bundle_manifest_drift")
    return raw


def register(connection, spec):
    """Owner-only atomic append. Conflicts never repair or reactivate old rows.

    The explicit transaction also provides a savepoint when the operator already
    has a transaction. Loading/hashing deployment files must precede this call.
    """
    spec = validate(BundleRegistration, spec.model_dump(mode="json"))
    with connection.transaction():
        owner = connection.execute(
            "SELECT pg_get_userbyid(nspowner)=current_user "
            "FROM pg_namespace WHERE nspname='medgraph'"
        ).fetchone()
        if owner != (True,):
            raise ValueError("native_owner_required")
        connection.execute("SELECT pg_advisory_xact_lock(7150926)")
        _tool(connection, spec)
        exists = connection.execute(
            "SELECT id FROM medgraph.tool_version WHERE tool_id=%s AND version=%s",
            (spec.tool.id, VERSION),
        ).fetchone()
        if exists:
            current(connection, spec)
            status = "already_registered"
        else:
            _append(connection, spec)
            current(connection, spec)
            status = "registered"
        return dict(
            status=status,
            tool=CODE,
            version=VERSION,
            installation_checksum=digest(spec),
            clinical_validated=False,
            formal_06_bundle_installed=False,
        )


def _append(connection, spec):
    m = spec.manifest
    vid = connection.execute(
        "INSERT INTO medgraph.tool_version(tool_id,version,runtime,metadata) "
        "VALUES(%s,%s,%s,%s) RETURNING id",
        (spec.tool.id, VERSION, Jsonb(m.runtime_environment), Jsonb(spec.version_metadata)),
    ).fetchone()[0]
    for direction, name, code, schema, required, low, high, constraints in _expected_ports():
        artifact = connection.execute(
            "SELECT id,schema_json FROM medgraph.artifact_type WHERE code=%s",
            (code,),
        ).fetchone()
        if artifact is not None and canonical(artifact[1]) != canonical(schema):
            raise ValueError("fundus_bundle_artifact_conflict")
        aid = (
            artifact[0]
            if artifact
            else connection.execute(
                "INSERT INTO medgraph.artifact_type(code,name_zh,schema_json) "
                "VALUES(%s,%s,%s) RETURNING id",
                (code, name, Jsonb(schema)),
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT INTO medgraph.tool_port(tool_version_id,direction,port_name,artifact_type_id,"
            "required,cardinality_min,cardinality_max,constraints) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
            (vid, direction, name, aid, required, low, high, Jsonb(constraints)),
        )
    iid = connection.execute(
        "INSERT INTO medgraph.tool_implementation(tool_version_id,implementation_key,"
        "implementation_kind,entrypoint_uri,working_directory,runtime_environment,invocation,"
        "formula_spec,hardware_requirements,provenance,validation_status,validation_report) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            vid,
            IMPLEMENTATION,
            m.implementation_kind,
            m.entrypoint_uri,
            m.working_directory,
            Jsonb(m.runtime_environment),
            Jsonb(m.invocation),
            None,
            Jsonb(m.hardware_requirements),
            Jsonb(m.provenance),
            m.validation_status,
            Jsonb(m.validation_report),
        ),
    ).fetchone()[0]
    for resource in m.resources:
        connection.execute(
            "INSERT INTO medgraph.implementation_resource(implementation_id,resource_key,"
            "resource_role,uri,sha256,byte_size,required,metadata) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                iid,
                resource.resource_key,
                resource.resource_role,
                resource.uri,
                resource.sha256,
                resource.byte_size,
                resource.required,
                Jsonb(resource.metadata),
            ),
        )

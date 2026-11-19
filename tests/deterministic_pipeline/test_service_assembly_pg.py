"""P2 real PG acceptance, exclusively under the central session database lease.

No patient pipeline or model is executed. Synthetic persistence fixtures are used
only inside this lease; the production factory's pinned target is not altered.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql

from apps import patient_service_assembly as assembly
from apps.deterministic_case_pipeline import assemble_visible_pipeline
from apps.patient_service_assembly import (
    AssemblyUnavailable,
    KnowledgeObjectReader,
    PatientServices,
    assemble_patient_services,
    check_runtime_bridge,
)
from db import manage
from db.audit.repository import (
    AuditEvent,
    PostgresAuditReader,
    PostgresAuditWriter,
)
from db.grounding.repository import GroundingRepository
from db.knowledge.repository import KnowledgePermissionError, KnowledgeRepository
from db.runtime.repository import IdempotencyConflictError
from db.runtime.repository.case_pipeline import CasePipelineRepository, identity
from packages.contracts import CaseBundle
from packages.deterministic_pipeline.models import CaseInput, PipelineConfig
from packages.disease_packs import DiseasePackCatalog
from packages.graph_engine import canonical_checksum
from packages.grounding import GroundingError
from packages.grounding.synthetic import SyntheticCatalogReader, fixture
from packages.object_store import LocalObjectStore
from packages.retrieval.knowledge.models import RetrievalContext
from tests.database.knowledge_support_04a import connect, free_port
from tests.database.test_audit_repository_02c import _example

SIX = (*manage.KNOWLEDGE_CHANNELS, *manage.GROUNDING_CHANNELS)


@pytest.fixture
def owned_object_roots(tmp_path, monkeypatch):
    """Real root validation with deployment files unavailable, never repaired."""
    deployment_roots = (
        *assembly.OBJECT_ROOTS.values(),
        assembly.ROOT / ".local/07-patient-evaluation-v1",
    )
    attempted_access = []

    def guard(method):
        def checked(path, *args, **kwargs):
            # Lexical only: resolving a deployment path would itself access it.
            paths = [path]
            if method.__name__ in ("rename", "replace"):
                paths.append(args[0] if args else kwargs["target"])
            for candidate in paths:
                absolute = Path(os.path.abspath(candidate))
                if any(absolute.is_relative_to(root) for root in deployment_roots):
                    attempted_access.append((method.__name__, str(candidate)))
                    raise FileNotFoundError("deployment object roots unavailable in this test")
            return method(path, *args, **kwargs)

        return checked

    for name in (
        "stat",
        "lstat",
        "open",
        "resolve",
        "iterdir",
        "mkdir",
        "chmod",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "touch",
    ):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    roots = {purpose: tmp_path / purpose for purpose in ("visible", "knowledge")}
    for purpose, root in roots.items():
        root.mkdir(mode=0o700)
        for child in ("objects", "staging"):
            (root / child).mkdir(mode=0o700)
        marker = root / "owner.json"
        marker.write_text(json.dumps({"owner": assembly.OWNER, "purpose": purpose}))
        marker.chmod(0o600)
    monkeypatch.setattr(assembly, "OBJECT_ROOTS", roots)
    print(f"P2 owned object roots parent: {tmp_path}")
    yield roots
    # Also catches forbidden accesses that the factory wrapped as unavailable.
    assert attempted_access == []


@pytest.mark.parametrize("defect", ["root_mode", "owner", "purpose", "missing_marker"])
def test_owned_object_roots_still_require_real_validation(owned_object_roots, defect):
    root = owned_object_roots["visible"]
    marker = root / "owner.json"
    if defect == "root_mode":
        root.chmod(0o755)
        reason = "object_root_ownership_or_mode"
    elif defect == "missing_marker":
        marker.unlink()
        reason = "service_assembly_unavailable"
    else:
        document = json.loads(marker.read_text())
        document[defect] = "not-the-authorized-" + defect
        marker.write_text(json.dumps(document))
        reason = "object_root_owner_marker"
    requested = []

    def resolve(channel):
        requested.append(channel)
        raise AssertionError("invalid roots must be rejected before resolving targets")

    with pytest.raises(AssemblyUnavailable, match=reason):
        assemble_patient_services(resolve)
    assert requested == []


@pytest.fixture(scope="module")
def deployed(database_lease):
    with database_lease.cluster(free_port()):
        targets = {
            c: manage.test_target(c)
            for c in ("bootstrap", "migration", "runtime", "audit_writer", "audit_reader", *SIX)
        }
        admin, migration = targets["bootstrap"], targets["migration"]
        manage.bootstrap(admin, migration.user)
        manage.bootstrap_runtime_principal(admin, targets["runtime"].user)
        manage.bootstrap_audit_principals(
            admin, targets["audit_writer"].user, targets["audit_reader"].user
        )
        manage.bootstrap_knowledge_principals(
            admin, *(targets[c].user for c in manage.KNOWLEDGE_CHANNELS)
        )
        manage.bootstrap_grounding_principals(
            admin, *(targets[c].user for c in manage.GROUNDING_CHANNELS)
        )
        with pytest.raises(AssemblyUnavailable, match="runtime_0010_ports_unavailable"):
            check_runtime_bridge(targets["runtime"])
        manage.migrate(migration)
        first = {d: manage.recorded_migrations(migration, d) for d in manage.DOMAINS}
        manage.migrate(migration)
        second = {d: manage.recorded_migrations(migration, d) for d in manage.DOMAINS}
        check_runtime_bridge(targets["runtime"])
        yield SimpleNamespace(
            targets=targets,
            first=first,
            second=second,
            runtime=CasePipelineRepository(targets["runtime"]),
            audit=PostgresAuditWriter(targets["audit_writer"]),
            audit_reader=PostgresAuditReader(targets["audit_reader"]),
            knowledge={
                c: KnowledgeRepository(targets["knowledge_" + c], capability=c)
                for c in ("editor", "reviewer", "publisher", "reader")
            },
        )


def test_first_and_repeated_migration_keep_exact_checksums(deployed):
    assert deployed.first == deployed.second
    assert len(deployed.first["runtime"]) == 14
    for domain, ledger in deployed.first.items():
        for path in (manage.DB_ROOT / domain / "migrations").glob("*.sql"):
            assert (
                ledger[path.name[:4]] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            )


@pytest.mark.parametrize("channel", SIX)
def test_six_real_logins_have_only_one_capability_and_no_owner_switch(deployed, channel):
    target = deployed.targets[channel]
    role = "mmagent_" + channel
    validator = (
        manage.validate_knowledge_principal
        if channel.startswith("knowledge_")
        else manage.validate_grounding_principal
    )
    validator(target, role)
    with connect(target) as connection:
        attrs = connection.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles WHERE rolname=session_user"
        ).fetchone()
        assert attrs == (True, False, False, False, False, False, False)
        for denied in (*manage.OWNER_ROLES, "mmagent_runtime_app", "mmagent_migrator"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with connection.transaction():
                    connection.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(denied)))
        connection.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
        for table in ("runtime.case_record", "audit.event", "knowledge.entity_version"):
            assert not connection.execute(
                "SELECT has_table_privilege(current_user,c.oid,'INSERT,UPDATE,DELETE') "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=%s AND c.relname=%s",
                tuple(table.split(".")),
            ).fetchone()[0]
        if channel == "knowledge_reader":
            for forbidden in (
                "SELECT * FROM knowledge.entity_version",
                "SELECT knowledge.put_version('{}'::jsonb,NULL)",
            ):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    with connection.transaction():
                        connection.execute(forbidden)
        if channel == "grounding_reader":
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with connection.transaction():
                    connection.execute("DELETE FROM grounding.mapping_version WHERE false")


def test_wrong_actual_login_rejected_by_public_repositories(deployed):
    with pytest.raises(KnowledgePermissionError):
        KnowledgeRepository(deployed.targets["knowledge_editor"], capability="reader")
    with pytest.raises(manage.DatabaseCommandError):
        CasePipelineRepository(deployed.targets["audit_reader"])
    with pytest.raises(manage.DatabaseCommandError):
        PostgresAuditWriter(deployed.targets["audit_reader"])
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        check_runtime_bridge(deployed.targets["audit_reader"])


def test_real_pg_visible_composition_without_production_target_override(
    deployed, owned_object_roots
):
    cfg = PipelineConfig.model_validate_json(
        (manage.ROOT / "configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    requested = []

    def resolve(channel):
        requested.append(channel)
        return deployed.targets[channel]

    # The production factory must continue to reject a valid but unapproved target.
    with pytest.raises(AssemblyUnavailable, match="service_target_or_identity_conflict"):
        assemble_patient_services(resolve)
    assert requested == ["runtime", "audit_writer", "audit_reader", "knowledge_reader"]
    # Validate before LocalObjectStore can create any directories of its own.
    visible = LocalObjectStore(owned_object_roots["visible"])
    body = KnowledgeObjectReader(LocalObjectStore(owned_object_roots["knowledge"]))
    pipeline = assemble_visible_pipeline(
        runtime_target=resolve("runtime"),
        audit_writer_target=resolve("audit_writer"),
        audit_reader_target=resolve("audit_reader"),
        knowledge_reader=deployed.knowledge["reader"],
        knowledge_store=body,
        visible_store=visible,
        catalog=DiseasePackCatalog.from_directory(manage.ROOT / "configs/disease_packs"),
        configs=(cfg,),
    )
    assert pipeline.runtime.read_request("P2:missing") is None
    assert pipeline.audit_reader.get_event("MESSAGE:OID_P2MISSING") is None
    assert pipeline.objects is pipeline.visible_reader.store is visible
    assert visible.root == owned_object_roots["visible"]
    assert pipeline.disease._published._store is body
    assert pipeline.disease._repository._role == "mmagent_knowledge_reader"
    assert not hasattr(body, "stage")
    assert not set(vars(pipeline)) & {"evaluation", "evaluator_store", "resolve_target"}
    assert pipeline.tools is None and pipeline.mappings is None
    readiness = PatientServices(pipeline, cfg).readiness()
    assert readiness["service_assembly_available"] and not readiness["patient_run_ready"]


def test_grounding_public_writer_reader_use_real_transactions(deployed):
    mapping, policy, catalog, knowledge, _ = fixture(code="BINDING:OID_P2ISOLATED")
    ports = {
        c: GroundingRepository(
            deployed.targets["grounding_" + c],
            capability=c,
            policy=policy,
            knowledge=knowledge,
            catalogs=SyntheticCatalogReader([catalog]),
        )
        for c in ("writer", "reader")
    }
    with pytest.raises(GroundingError, match="writer_required"):
        ports["reader"].register_draft(mapping, RetrievalContext(), request_key="P2-denied")
    draft = ports["writer"].register_draft(mapping, RetrievalContext(), request_key="P2-draft")
    assert ports["reader"].get_exact(mapping.reference()) == draft
    assert (
        ports["writer"].register_draft(mapping, RetrievalContext(), request_key="P2-draft") == draft
    )


@pytest.fixture
def request_input(deployed):
    case = _example(CaseBundle, "CaseBundle.valid.json").model_copy(
        update={"observations": (), "artifacts": (), "report_uri": None, "extensions": {}}
    )
    case = CaseBundle.model_validate_json(case.model_dump_json())
    deployed.runtime.create_case_bundle(case, "TASK:OID_P2CASE")
    cfg = PipelineConfig.model_validate_json(
        (manage.ROOT / "configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    return CaseInput(
        synthetic=True,
        case=case,
        pack_ref=cfg.pack_ref,
        input_checksum=canonical_checksum(case),
        visible=None,
    ), cfg


def event(key, case_code, *, run_id=None, event_type="REQUEST_RECEIVED", payload=None):
    return AuditEvent(
        event_id=identity("MESSAGE", key),
        event_type=event_type,
        actor="PLANNER",
        case_code=case_code,
        run_id=run_id,
        correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
        idempotency_key=identity("TASK", key),
        occurred_at=datetime.now(UTC),
        payload=payload or {"synthetic": True},
    )


def test_0010_request_idempotency_and_audit_failure_roll_back(deployed, request_input):
    value, cfg = request_input
    runtime = deployed.runtime
    key = runtime.key(value.case.case_code, "P2-good")
    state = runtime.begin(key, value, cfg)
    assert runtime.begin(key, value, cfg) == state
    assert deployed.audit_reader.get_event(identity("MESSAGE", key + ":0")) is not None
    next_state = runtime.checkpoint(key, state, actions=("P2:synthetic",))
    with pytest.raises(IdempotencyConflictError):
        runtime.checkpoint(key, state, actions=("P2:stale",))
    assert runtime.read_request(key) == next_state

    failed_key = runtime.key(value.case.case_code, "P2-conflict")
    collision = event("P2-seed-conflict", value.case.case_code).model_copy(
        update={"idempotency_key": identity("TASK", "audit:" + failed_key + ":0")}
    )
    deployed.audit.append_event(collision)
    with pytest.raises(psycopg.errors.UniqueViolation):
        runtime.begin(failed_key, value, cfg)
    assert runtime.read_request(failed_key) is None
    assert deployed.audit_reader.get_event(identity("MESSAGE", failed_key + ":0")) is None

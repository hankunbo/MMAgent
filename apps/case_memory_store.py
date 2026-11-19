"""Local research archive deployment and original-owner assembly; no model calls."""

import json
import os
import secrets
from pathlib import Path

import psycopg
from psycopg import sql

from apps.research_single_case import credential_boundary
from db import manage
from db.memory.admin import bootstrap_memory_principals
from db.memory.repository import MemoryRepository
from db.runtime.repository.case_evidence import CaseEvidenceRepository
from db.runtime.repository.research import ResearchRunRepository
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.memory.archive import ArchiveSources
from packages.memory.execution_archive import ArchiveOwners, ExecutionArchiveSources

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local/memory-case/config.json"


def configuration():
    if PRIVATE.is_symlink() or PRIVATE.parent.is_symlink() or PRIVATE.stat().st_mode & 0o077:
        raise ValueError("memory_credential_permissions")
    return json.loads(PRIVATE.read_bytes())


def deploy():
    """Explicitly invoked only. Add Memory to the current database, not another DB."""
    original = credential_boundary().resolve_target("runtime")
    migration = credential_boundary().resolve_target("migration")
    if (original.host, original.port, original.database) != (
        migration.host,
        migration.port,
        migration.database,
    ):
        raise ValueError("memory_deployment_target_conflict")
    admin = manage.ConnectionTarget(
        str(ROOT / "algorithm-capability-db/.postgres/socket"),
        original.port,
        original.database,
        "medgraph_admin",
        "UNUSED_LOCAL_SOCKET",
    )
    if not manage.target_is_ready(admin):
        raise ValueError("memory_local_database_unavailable")
    before = {d: manage.recorded_migrations(migration, d) for d in manage.DOMAINS}
    # Preserve old SQL checksums and refuse a mismatched deployed migration.
    for p in manage._migration_files("memory"):
        if p.name[:4] in before["memory"] and before["memory"][p.name[:4]] != manage._checksum(p):
            raise ValueError("memory_deployed_checksum_changed")
    if not PRIVATE.exists():
        PRIVATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if PRIVATE.parent.is_symlink() or PRIVATE.parent.stat().st_mode & 0o077:
            raise ValueError("memory_credential_permissions")
        values = dict(
            host=original.host,
            port=original.port,
            database=original.database,
            credentials={
                c: dict(user="mmagent_case_memory_" + c, password=secrets.token_urlsafe(32))
                for c in ("writer", "reviewer", "reader")
            },
        )
        fd = os.open(PRIVATE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(values, stream)
    values = configuration()
    if (values["host"], values["port"], values["database"]) != (
        original.host,
        original.port,
        original.database,
    ):
        raise ValueError("memory_deployment_target_conflict")
    with psycopg.connect(
        host=admin.host, port=admin.port, dbname=admin.database, user=admin.user, connect_timeout=5
    ) as conn:
        for credentials in values["credentials"].values():
            exists = conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname=%s", (credentials["user"],)
            ).fetchone()
            if not exists:
                conn.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                        "NOREPLICATION NOBYPASSRLS PASSWORD {}"
                    ).format(
                        sql.Identifier(credentials["user"]), sql.Literal(credentials["password"])
                    )
                )
            conn.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(original.database), sql.Identifier(credentials["user"])
                )
            )
    # Bootstrap validates three independent LOGINs over the existing local socket.
    bootstrap_memory_principals(
        admin, *(values["credentials"][c]["user"] for c in ("writer", "reviewer", "reader"))
    )
    manage._psql(admin, (manage.DB_ROOT / "memory/publication_bootstrap.sql").read_text())
    prerequisite = manage.DB_ROOT / "runtime/migrations/0015_memory_references.sql"
    if "0015" not in before["runtime"]:
        manage.apply_migration(migration, "runtime", prerequisite)
    for path in manage._migration_files("memory"):
        if path.name[:4] not in before["memory"]:
            manage.apply_migration(migration, "memory", path)
    # Audit 0004 is needed by original Memory writer; do not migrate unrelated domains.
    for path in manage._migration_files("audit"):
        if path.name[:4] not in before["audit"]:
            if path.name != "0004_memory_audit.sql":
                raise ValueError("memory_audit_prerequisite_missing")
            manage.apply_migration(migration, "audit", path)
    after = {d: manage.recorded_migrations(migration, d) for d in manage.DOMAINS}
    if any(
        before[d] != after[d] for d in manage.DOMAINS if d not in {"memory", "audit", "runtime"}
    ):
        raise ValueError("memory_unrelated_migration_drift")
    if {k: v for k, v in after["runtime"].items() if k != "0015"} != {
        k: v for k, v in before["runtime"].items() if k != "0015"
    }:
        raise ValueError("memory_unrelated_runtime_migration_drift")
    return dict(
        database=original.database,
        before=before,
        after=after,
        new_database=False,
        real_publication_enabled=False,
    )


def assembly(capability="reader"):
    from apps.glaucoma_case_storage import read_view
    from apps.patient_service_assembly import OBJECT_ROOTS
    from packages.memory.archive_objects import ArchiveObjectReader
    from packages.object_store import LocalObjectStore
    from packages.research_runtime.deployment import (
        existing_store,
        load_configuration,
        read_installed_bundle,
    )

    target = credential_boundary().resolve_target("runtime")
    runtime = CaseEvidenceRepository(target)
    contexts = ResearchRunRepository(target)
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    config = load_configuration(Path("configs/research/fairclip_single_case.json"))
    installed = read_installed_bundle(config["installed_bundle"])
    output = existing_store(installed["output_root"], "research_output")
    object_reader = ArchiveObjectReader(store, output)

    def case_for_run(run):
        context = contexts.read_research_run(run)
        if context is None:
            raise ValueError("memory_source_missing")
        return context.request.scope.case_code

    artifacts = ArchiveSources(
        runtime=runtime,
        read_bytes=object_reader.read,
        read_run_case=case_for_run,
        read_view=lambda c, g: read_view(c, g, repo=runtime, store=store),
    )
    executions = ExecutionArchiveSources(ResearchLifecycleRepository(target))
    owners = ArchiveOwners(artifacts, executions)
    conf = configuration()
    secret = conf["credentials"][capability]
    connection = manage.ConnectionTarget(
        conf["host"],
        conf["port"],
        conf["database"],
        secret["user"],
        "UNUSED_PRIVATE_MEMORY_CONFIG",
        secret["password"],
    )
    return MemoryRepository(connection, capability=capability, sources=owners), owners


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("deploy", "list", "suggest", "archive", "execution"))
    p.add_argument("--case")
    p.add_argument("--run")
    p.add_argument("--graph")
    p.add_argument("--key")
    p.add_argument(
        "--kind", choices=("graph_snapshot", "tool_experience"), default="graph_snapshot"
    )
    p.add_argument("--target", default="glaucoma")
    a = p.parse_args()
    if a.action == "deploy":
        result = deploy()
    else:
        repo, owners = assembly("writer" if a.action in {"archive", "execution"} else "reader")
        if a.action == "archive":
            result = repo.create(
                owners.artifacts.prepare(
                    key=a.key,
                    case_code=a.case,
                    run_id=a.run,
                    origin_code=a.graph,
                    kind=a.kind,
                    target=a.target,
                )
            ).model_dump(mode="json")
        elif a.action == "execution":
            result = repo.create(owners.executions.prepare(key=a.key, run_id=a.run)).model_dump(
                mode="json"
            )
        elif a.action == "suggest":
            from packages.memory.archive import WorkflowMemory

            result = WorkflowMemory(repo).suggest(
                target=a.target,
                modality="SLO",
                query_case=a.case,
                legal_tools={"segman_slo_disc_cup_segmenter", "cdr_metrics"},
            )
        else:
            result = [
                dict(
                    key=k,
                    kind=(v := repo.read(k)).record.kind,
                    state=v.state,
                    outcome=v.record.detail.outcome,
                )
                for k in repo.archive_keys()
            ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

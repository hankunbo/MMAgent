"""Install only the tested task-context port into the existing research database."""

from pathlib import Path

from apps.vectorengine_single_case import write_once
from db import manage


def install(output):
    from apps.research_single_case import credential_boundary

    boundary = credential_boundary()
    runtime = boundary.resolve_target("runtime")
    target = boundary.resolve_target("migration")
    if (runtime.host, runtime.port, runtime.database) != (
        target.host,
        target.port,
        target.database,
    ):
        raise ValueError("multimodal_database_target_conflict")
    before = {d: manage.recorded_migrations(target, d) for d in manage.DOMAINS}
    for domain in manage.DOMAINS:
        for path in manage._migration_files(domain):
            if path.name[:4] in before[domain] and before[domain][
                path.name[:4]
            ] != manage._checksum(path):
                raise ValueError("multimodal_deployed_migration_drift")
    if not {"0012", "0014"} <= set(before["runtime"]):
        raise ValueError("multimodal_task_prerequisites_missing")

    def existing():
        query = (
            "SET ROLE mmagent_runtime_owner; SELECT count(*)::text || '|' || "
            "md5(coalesce(string_agg(run_id || ':' || context_version || ':' || "
            "coalesce(research_context::text,''),'|' ORDER BY run_id),'')) FROM runtime.run;"
        )
        return manage._psql(target, query, tuples_only=True).stdout.strip()

    previous = existing()
    path = Path("db/runtime/migrations/0023_task_research_context.sql")
    applied = "0023" not in before["runtime"]
    if applied:
        manage.apply_migration(target, "runtime", path)
    after = {d: manage.recorded_migrations(target, d) for d in manage.DOMAINS}
    unchanged = (
        all(before[d] == after[d] for d in manage.DOMAINS if d != "runtime")
        and {k: v for k, v in after["runtime"].items() if k != "0023"}
        == {k: v for k, v in before["runtime"].items() if k != "0023"}
        and previous == existing()
    )
    result = dict(
        schema_version="multimodal-task-port-installation/1.0.0",
        applied=applied,
        migration=path.name,
        checksum=manage._checksum(path),
        before=before,
        after=after,
        old_run_contexts_unchanged=unchanged,
        new_database=False,
        new_network_service=False,
        clinical_acceptance=False,
    )
    write_once(Path(output), result)
    if not unchanged:
        raise ValueError("multimodal_deployment_unexpected_drift")
    return result

"""Original Memory candidate storage without a classifier-owned installation."""

from db import manage
from db.memory.repository import MemoryRepository
from packages.memory.archive import ArchiveSources


def archive_saved_snapshot(directory, *, commit=False):
    """Explicit zero-request archival, not a stopped campaign resume.

    Select the saved result by locator only. Original Runtime/acceptance and
    Memory sources rebuild the snapshot; old result/exports are never changed.
    Local composition and reassessment do not invent new executable workflows.
    """
    from pathlib import Path

    from apps.case_memory_store import assembly
    from db.runtime.repository.case_pipeline import identity
    from db.runtime.repository.planner import PlannerRepository
    from packages.evaluation_preparation.cohort_recovery import read_file
    from packages.grounding.models import digest
    from packages.memory.aggregation_archive import Documents
    from packages.memory.multimodal_archive import ArchiveSourceCapacityError

    from .report_storage import unpack

    result = read_file(Path(directory) / "result.json")
    version = result.get("schema_version")
    if (
        version
        not in {
            "multimodal-case-result/1.0.0",
            "generic-accepted-aggregation-result/1.0.0",
            "multimodal-accepted-aggregation-result/1.0.0",
        }
        or result.get("complete") is not True
    ):
        raise ValueError("memory_snapshot_completed_result_required")
    phase = result["phases"][-1]
    if phase.get("accepted") is not True or phase.get("target") is not None:
        raise ValueError("memory_snapshot_review_required")
    report_key = (
        "accepted-aggregation-report" if "aggregation" in version else "dynamic-case-report"
    )
    repo, owners = assembly("writer" if commit else "reader")
    owner = owners.artifacts
    case, run = result["case_code"], phase["run_id"]
    artifacts = owner.runtime.read_observation_bundle(
        identity("TASK", run + ":agent:" + report_key)
    )
    if not artifacts:
        raise ValueError("memory_snapshot_report_missing")
    import json

    origin, raw = owner.artifact(case, artifacts[0].asset_code)
    envelope = json.loads(raw)
    sessions = PlannerRepository(owner.runtime._target)
    docs = Documents(
        owner, case, run, envelope["root_checksum"], {}, sessions, phase["acceptance_key"]
    )
    control = sessions.read(run)
    report, _ = unpack(docs.read(report_key), key=report_key)
    if (
        control is None
        or control.phase != "stopped"
        or control.pending
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-phase-result") != (phase,)
        or report["result"] != result
    ):
        raise ValueError("memory_snapshot_original_result_binding")
    target = report["original_review"]["material"]["goal_code"]
    key = "recovered-snapshot:" + digest((run, report_key, origin.asset_code, target))[7:55]
    base = dict(
        schema_version="case-result-memory-snapshot/1.0.0",
        run_id=run,
        case_code=case,
        key=key,
        report_key=report_key,
        report_checksum=digest(report),
        source_result_checksum=digest(result),
        new_api_requests=0,
        new_tool_executions=0,
        new_scores=0,
        eligible_for_planning=False,
        clinical_publication=False,
        new_executable_workflows=0,
    )
    try:
        record = owner.prepare(
            key=key,
            case_code=case,
            run_id=run,
            origin_code=origin.asset_code,
            kind="graph_snapshot",
            target=target,
        )
    except ArchiveSourceCapacityError as error:
        return dict(
            base,
            status="not_archived_source_capacity",
            committed=False,
            required_sources=error.required_sources,
            maximum_sources=error.maximum_sources,
            source_set_checksum=error.source_set_checksum,
            source_references_truncated=False,
        )
    if sessions.read(run) != control:
        raise ValueError("memory_snapshot_control_changed")
    if commit:
        saved = repo.create(record)
        if saved.record != record or repo.read(key) != saved or saved.eligible_for_planning:
            raise ValueError("memory_snapshot_readback")
    return dict(
        base,
        status="candidate_snapshot_archived" if commit else "snapshot_prepared",
        committed=commit,
        sources=len(record.sources),
        record_checksum=digest(record),
        archive_version=record.schema_version,
        source_capacity_warning=len(record.sources) > 128,
    )


def method_archive_status(campaign, result):
    """Explicit unsupported archive contract, not an empty library or publication.

    Method reassessments retain their original Runtime report. The Memory owner
    has not implemented their versioned historical-Review lineage; do not coerce
    them into an older completed-case schema or call its writer prematurely.
    """
    versions = {
        "multimodal-campaign-renewal/1.35.0": "multimodal-method-reviewed-case-result/1.0.0",
        "multimodal-campaign-renewal/1.40.0": "multimodal-method-reviewed-case-result/1.0.0",
        "multimodal-campaign-renewal/1.36.0": "multimodal-method-continued-case-result/1.0.0",
    }
    version = getattr(campaign, "protocol", {}).get("schema_version")
    if version not in versions:
        return None
    if result.get("schema_version") != versions[version]:
        raise ValueError("multimodal_memory_method_result_binding")
    return dict(
        status="deferred_unsupported_method_result_contract",
        reason="memory_result_version_not_supported",
        result_version=result["schema_version"],
        retained_in="original_runtime_case_result_and_report",
        candidate_archived=False,
        eligible_for_planning=False,
        clinical_correctness="not_established",
    )


def archive_reader(campaign, documents):
    """Use the installed component owner's stores, not a search of arbitrary roots."""
    component = getattr(campaign.base, "component_observations", None)
    if component is None:
        return documents.read_artifact
    from .component_observations import ComponentObservations

    if type(component) is not ComponentObservations:
        raise ValueError("multimodal_memory_component_owner")
    objects = component.owner.research.compiler.admission.objects
    # CombinedResearchObjects enforces visible/output separation and collisions;
    # ArchiveSources separately rechecks every artifact checksum and provenance.
    return lambda artifact: objects.read_bounded(artifact.uri, 4194304)


def archive(campaign, result):
    from apps.case_memory_store import configuration

    if result.get("review_closure_scope", {}).get("local_confirmed_only"):
        return dict(
            status="not_archived_whole_case_local_partial_closure",
            eligible_for_planning=False,
        )
    if getattr(campaign, "protocol", {}).get("material_context_review"):
        return dict(status="not_archived_context_reassessment", eligible_for_planning=False)
    if getattr(campaign, "protocol", {}).get("review_acquired_only"):
        return dict(status="not_archived_acquired_review_increment", eligible_for_planning=False)
    if getattr(campaign, "protocol", {}).get("acquisition_increment_only"):
        return dict(
            status="not_archived_whole_case_acquisition_increment", eligible_for_planning=False
        )
    if getattr(campaign, "protocol", {}).get("definition_increment_only"):
        return dict(status="not_archived_whole_case_increment_only", eligible_for_planning=False)
    if getattr(campaign, "protocol", {}).get("review_increment_only"):
        return dict(status="not_archived_pending_review_increment", eligible_for_planning=False)
    if not result["complete"] or not result["phases"][-1]["accepted"]:
        return dict(status="not_archived_incomplete", eligible_for_planning=False)
    method = method_archive_status(campaign, result)
    if method is not None:
        return method
    conf = configuration()
    target = campaign.target
    if (conf["host"], conf["port"], conf["database"]) != (
        target.host,
        target.port,
        target.database,
    ):
        raise ValueError("multimodal_memory_database_binding")
    secret = conf["credentials"]["writer"]
    connection = manage.ConnectionTarget(
        conf["host"],
        conf["port"],
        conf["database"],
        secret["user"],
        "UNUSED_PRIVATE_MEMORY_CONFIG",
        secret["password"],
    )
    final = result["phases"][-1]
    docs = campaign.documents(final["run_id"])
    sources = ArchiveSources(
        runtime=campaign.base.runtime,
        read_bytes=archive_reader(campaign, docs),
        read_run_case=lambda run: campaign.contexts.read_research_run(run).request.scope.case_code,
        read_view=lambda *_: None,
    )
    repo = MemoryRepository(connection, capability="writer", sources=sources)
    return persist(
        repo,
        sources,
        docs,
        result,
        report_key=getattr(campaign, "report_key", "dynamic-case-report"),
    )


def persist(repo, sources, docs, result, *, report_key="dynamic-case-report"):
    from packages.memory.multimodal_archive import ArchiveSourceCapacityError

    from .report_storage import unpack

    report = unpack(docs.read(report_key), key=report_key)[0]
    origin = docs.repo.read_observation_bundle(docs.key(report_key))[0]
    items = []
    targets = [("graph_snapshot", result["task"]["target_code"])]
    units = report.get("production_inputs", {}).get("policy", {}).get("evidence_units", {})
    seen = set()
    for r in report["requirements"]:
        unit = units.get(r["id"], r["id"])
        if r.get("observation") and r["observation"]["status"] == "obtained" and unit not in seen:
            seen.add(unit)
            targets.append(("tool_experience", r["id"]))
    from packages.grounding.models import digest

    for kind, target in targets:
        key = (
            "mm-memory:"
            + digest(
                (docs.run_id, kind, target)
                if report_key == "dynamic-case-report"
                else (docs.run_id, report_key, kind, target)
            )[7:55]
        )
        try:
            record = sources.prepare(
                key=key,
                case_code=result["case_code"],
                run_id=docs.run_id,
                origin_code=origin.asset_code,
                kind=kind,
                target=target,
            )
        except ArchiveSourceCapacityError as exc:
            return dict(
                status="partially_archived_source_capacity"
                if items
                else "not_archived_source_capacity",
                items=items,
                deferred=dict(
                    key=key,
                    kind=kind,
                    target=target,
                    required_sources=exc.required_sources,
                    maximum_sources=exc.maximum_sources,
                    source_set_checksum=exc.source_set_checksum,
                    report_key=report_key,
                    origin_code=origin.asset_code,
                ),
                source_references_truncated=False,
                eligible_for_planning=False,
                clinical_correctness="not_established",
            )
        if record.key != key:
            raise ValueError("multimodal_memory_prepared_key_binding")
        saved = repo.create(record)
        if repo.read(key) != saved or saved.record != record:
            raise ValueError("multimodal_memory_readback")
        items.append(
            dict(
                key=key,
                kind=kind,
                target=target,
                checksum=digest(record),
                eligible_for_planning=saved.eligible_for_planning,
            )
        )
    return dict(
        status="candidate_archived",
        items=items,
        eligible_for_planning=False,
        clinical_correctness="not_established",
    )

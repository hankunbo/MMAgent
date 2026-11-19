"""Task-v2 report archival: current original bytes, never reusable patient scores."""

import json

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest

from .archive_models import ArchiveSourceCapacityError as ArchiveSourceCapacityError
from .archive_models import archive_record_class
from .scope_archive import VERSIONS as SCOPE_VERSIONS


def _committed_binding(owner, read, refs, envelope, parent, result, case_code, run_id):
    """Archive an original commit, never invent a second finalization grant."""
    from types import SimpleNamespace

    from agents.expert_agent.committed_review import read_committed_review_binding
    from packages.grounding.models import validate
    from packages.multimodal_research.stage_models import LocalReviewClosure
    from packages.research_runtime.agent_records import RuntimeDocuments

    original = read("multimodal-case-result")[0]
    closure = validate(LocalReviewClosure, result["local_review_closure"])
    final = result["phases"][-1]
    if (
        digest(original) != result["original_result_checksum"]
        or closure.original_phase != original["phases"][-1]
        or result["phases"][:-1] != original["phases"][:-1]
        or result.get("confirmed_observations") != original.get("confirmed_observations")
        or result.get("previous_attempts") != original.get("previous_attempts")
        or result["task"] != original["task"]
        or result["original_requests"] != original["new_requests"]
        or result["new_requests"] != 0
        or result.get("finalization_grant_checksum") is not None
        or result["execution_kind"] != "committed_review_export_not_reacceptance"
        or final["acceptance_key"] != closure.confirmation.acceptance_key
        or final["run_id"] != closure.confirmation.run_id
    ):
        raise ValueError("memory_committed_original_binding")

    class Documents:
        # Narrow historical read adapter. No session/write/dispatch capability.
        envelope = RuntimeDocuments.envelope
        reference = RuntimeDocuments.reference

        def read(self, name):
            values = read(name, optional=True)
            return tuple(values) if values is not None else None

        def validate_scope(self, scope):
            if scope.run_id != run_id or scope.case_code != case_code:
                raise ValueError("memory_committed_scope")
            graph, raw = owner.artifact(case_code, scope.graph.code)
            value = json.loads(raw)
            if graph.checksum != scope.graph.checksum:
                raise ValueError("memory_committed_graph")
            if graph.checksum != envelope["root_checksum"]:
                if (value.get("run_id"), value.get("root_checksum")) != (
                    run_id,
                    envelope["root_checksum"],
                ):
                    raise ValueError("memory_committed_graph_lineage")
                value = value["value"]
            if value.get("schema_version", value.get("schema")) != scope.graph.version:
                raise ValueError("memory_committed_graph_version")
            refs[graph.asset_code] = graph

    documents = Documents()
    documents.run_id = run_id
    documents.root = SimpleNamespace(checksum=envelope["root_checksum"])
    # Candidate archival reconstructs exact historical ancestry through the
    # original acceptance owner; it never grants current execution authority.
    from packages.evidence_details.review_batches import read_submission

    submitted = documents.read("submission:" + final["acceptance_key"])
    submission = read_submission(submitted[0]["submission"]) if submitted else None
    history = getattr(submission, "historical", None)
    kwargs = {}
    if history is not None:
        kwargs["confirmed_review_reader"] = _archive_ancestor_reader(
            owner, read, refs, envelope, case_code, history
        )
    proof = read_committed_review_binding(
        documents, parent=parent, config_checksum=parent["config_checksum"], **kwargs
    )
    if proof != closure.confirmation.model_dump(mode="json"):
        raise ValueError("memory_committed_confirmation_binding")


def _archive_ancestor_reader(owner, read, refs, envelope, case_code, history, lineage=()):
    """Read-only original accepted ancestor; no caller-supplied validity flags."""
    from types import SimpleNamespace

    from db.runtime.repository.planner import PlannerRepository
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.accepted_review_reuse import (
        AcceptedReviewOrigin,
        AcceptedReviewReader,
    )
    from packages.grounding.models import validate
    from packages.llm_provider.models import read_request_record
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    if (
        not isinstance(history, AcceptedReviewOrigin)
        or history.run_id in lineage
        or len(lineage) > 4
    ):
        raise ValueError("memory_committed_ancestor_version_or_cycle")
    run_id, key = history.run_id, history.acceptance_key

    class Documents:
        envelope = RuntimeDocuments.envelope
        reference = RuntimeDocuments.reference

        def read(self, name):
            value = read(name, source_run=run_id, optional=True)
            return tuple(value) if value is not None else None

        def validate_scope(self, scope):
            if (scope.case_code, scope.run_id) != (case_code, run_id):
                raise ValueError("memory_committed_ancestor_scope")
            graph, raw = owner.artifact(case_code, scope.graph.code)
            value = json.loads(raw)
            if graph.checksum != scope.graph.checksum:
                raise ValueError("memory_committed_ancestor_graph")
            if graph.checksum != envelope["root_checksum"]:
                if (value.get("run_id"), value.get("root_checksum")) != (
                    run_id,
                    envelope["root_checksum"],
                ):
                    raise ValueError("memory_committed_ancestor_lineage")
                value = value["value"]
            if value.get("schema_version", value.get("schema")) != scope.graph.version:
                raise ValueError("memory_committed_ancestor_graph_version")
            refs[graph.asset_code] = graph

        def consumed(self, graph):
            values = self.read("accepted:" + key)
            return bool(
                values
                and values[-1].get("status") == "accepted"
                and values[-1].get("parent_graph", {}).get("code") == graph.code
                and values[-1].get("parent_graph", {}).get("checksum") == graph.checksum
            )

    docs = Documents()
    docs.run_id = run_id
    docs.root = SimpleNamespace(checksum=envelope["root_checksum"])
    original_sessions = PlannerRepository(owner.runtime._target)
    docs.sessions = SimpleNamespace(read=original_sessions.read)
    journal = RuntimeAgentJournal(docs)
    requests = SimpleNamespace(
        journal=journal,
        records=lambda: tuple(
            read_request_record(e["record"])
            for e in journal.read()
            if e.get("phase") == "completed"
        ),
    )

    def state():
        current = validate(DomainState, docs.read("dynamic-phase-input")[0])
        for resource in current.resources:
            artifact, _ = owner.artifact(case_code, resource.code)
            if artifact.checksum != resource.checksum:
                raise ValueError("memory_committed_ancestor_resource")
            refs[artifact.asset_code] = artifact
        return current

    def ancestor(rid, chain):
        nested = getattr(history.submission, "historical", None)
        if nested is None or nested.run_id != rid:
            raise ValueError("memory_committed_ancestor_binding")
        return _archive_ancestor_reader(owner, read, refs, envelope, case_code, nested, chain)

    return AcceptedReviewReader(
        documents=docs,
        requests=requests,
        state_reader=state,
        acceptance_key=key,
        ancestor_reader=ancestor,
        lineage=lineage,
    )


def prepare(owner, *, key, case_code, run_id, origin, envelope, kind, target):
    from packages.evidence_details.partial_finalization import GRANT_KEY, REPORT_KEY, RESULT_KEY

    storage_refs = {}
    from packages.multimodal_research.report_storage import VERSION, unpack

    if envelope.get("value", {}).get("schema_version") == VERSION:
        report_key = envelope["value"]["report_key"]
        if report_key not in {
            "dynamic-case-report",
            REPORT_KEY,
            "multimodal-recorded-finalization-report",
        } or envelope.get("key_checksum") != digest(report_key):
            raise ValueError("memory_report_storage_key")
        artifacts = owner.runtime.read_observation_bundle(
            identity("TASK", run_id + ":agent:" + report_key)
        )
        values = []
        if not artifacts or artifacts[0].asset_code != origin.asset_code:
            raise ValueError("memory_report_storage_origin")
        for index, artifact in enumerate(artifacts):
            a, raw = owner.artifact(case_code, artifact.asset_code)
            wrapped = json.loads(raw)
            if (
                wrapped.get("run_id"),
                wrapped.get("root_checksum"),
                wrapped.get("key_checksum"),
                wrapped.get("index"),
            ) != (run_id, envelope["root_checksum"], digest(report_key), index):
                raise ValueError("memory_report_storage_scope")
            storage_refs[a.asset_code] = a
            values.append(wrapped["value"])
        envelope = dict(envelope, value=unpack(tuple(values), key=report_key)[0])

    partial = (
        envelope.get("value", {}).get("result", {}).get("schema_version")
        == "multimodal-partial-finalization-result/1.0.0"
    )
    version = envelope.get("value", {}).get("result", {}).get("schema_version")
    recorded = version in {
        "multimodal-recorded-finalization-result/1.0.0",
        "multimodal-recorded-finalization-result/1.1.0",
        "multimodal-recorded-finalization-result/1.2.0",
    }
    committed = version == "multimodal-recorded-finalization-result/1.2.0"
    if recorded:
        from packages.evidence_details.recorded_finalization import (
            GRANT_KEY,
            REPORT_KEY,
            RESULT_KEY,
        )
    if (
        kind not in {"graph_snapshot", "tool_experience"}
        or envelope.get("run_id") != run_id
        or envelope.get("key_checksum")
        != digest(REPORT_KEY if partial or recorded else "dynamic-case-report")
        or envelope.get("index") != 0
        or owner.read_run_case(run_id) != case_code
    ):
        raise ValueError("memory_multimodal_report_binding")
    report = envelope["value"]
    result = report["result"]
    final = result["phases"][-1]
    if (
        report["schema_version"] != "dynamic-case-report/1.0.0"
        or result["schema_version"]
        not in SCOPE_VERSIONS
        | {
            "multimodal-case-result/1.0.0",
            "multimodal-case-result/1.1.0",
            "multimodal-case-result/1.2.0",
            "multimodal-partial-finalization-result/1.0.0",
            "multimodal-recorded-finalization-result/1.0.0",
            "multimodal-recorded-finalization-result/1.1.0",
            "multimodal-recorded-finalization-result/1.2.0",
        }
        or result["case_code"] != case_code
        or final["run_id"] != run_id
        or not final["accepted"]
        or final["target"] is not None
    ):
        raise ValueError("memory_multimodal_not_completed")
    refs = {origin.asset_code: origin, **storage_refs}

    def read(name, *, source_run=run_id, optional=False):
        if owner.read_run_case(source_run) != case_code:
            raise ValueError("memory_multimodal_source_case")
        artifacts = owner.runtime.read_observation_bundle(
            identity("TASK", source_run + ":agent:" + name)
        )
        if not artifacts:
            if optional:
                return None
            raise ValueError("memory_source_missing")
        values = []
        for index, a in enumerate(artifacts):
            artifact, raw = owner.artifact(case_code, a.asset_code)
            doc = json.loads(raw)
            if (
                doc.get("run_id"),
                doc.get("root_checksum"),
                doc.get("key_checksum"),
                doc.get("index"),
            ) != (source_run, envelope["root_checksum"], digest(name), index):
                raise ValueError("memory_multimodal_document_binding")
            refs[a.asset_code] = artifact
            values.append(doc["value"])
        return values

    parent = read("dynamic-phase-input")[0]
    stored = read("accepted:" + final["acceptance_key"])
    detail = report["original_review"]
    if (
        read(RESULT_KEY if partial or recorded else "multimodal-case-result")[0] != result
        or detail["material"] != parent["material"]
        or (detail["records"], detail["graph"], report["scores"], detail["receipt"])
        != (stored[0], stored[1], stored[2]["scores"], stored[3])
    ):
        raise ValueError("memory_multimodal_content_binding")
    scope_context = None
    if version in SCOPE_VERSIONS:
        from .scope_archive import read_context

        scope_context = read_context(owner, read, refs, result, report, parent)
    if committed:
        _committed_binding(owner, read, refs, envelope, parent, result, case_code, run_id)
    if partial or (recorded and not committed):
        from packages.evidence_details.acceptance_models import read_records
        from packages.evidence_details.partial_finalization import (
            SUBMISSION_KEY,
            PartialFinalizationGrant,
        )
        from packages.evidence_details.review_batches import read_submission
        from packages.grounding.models import validate

        grant_type = PartialFinalizationGrant
        if recorded:
            from packages.evidence_details.recorded_finalization import (
                SUBMISSION_KEY,
                RecordedReviewGrant,
            )

            grant_type = RecordedReviewGrant
            from packages.evidence_details import recorded_finalization as recorded_owner

            if result.get("finalization_policy") == recorded_owner.PARTITION_POLICY:
                GRANT_KEY = recorded_owner.PARTITION_GRANT_KEY
                SUBMISSION_KEY = recorded_owner.PARTITION_SUBMISSION_KEY
                grant_type = recorded_owner.recorded_grant_type(read(GRANT_KEY)[0])
        grant = validate(grant_type, read(GRANT_KEY)[0])
        if recorded and isinstance(grant, recorded_owner.RecordedAllReusedGrant):
            from types import SimpleNamespace

            def reused_read(name):
                values = read(name, optional=True)
                return tuple(values) if values is not None else None

            proof = recorded_owner.all_reused_failure_proof(
                SimpleNamespace(run_id=run_id, read=reused_read)
            )
            if proof is None or proof != grant.all_reused_origin_checksum:
                raise ValueError("memory_recorded_all_reused_binding")
        if recorded and isinstance(grant, recorded_owner.RecordedOriginCompatibilityGrant):
            from types import SimpleNamespace

            def documents_read(name):
                values = read(name, optional=True)
                return tuple(values) if values is not None else None

            proof = recorded_owner.origin_compatibility_proof(
                SimpleNamespace(run_id=run_id, read=documents_read)
            )
            if proof is None or proof != grant.origin_compatibility_checksum:
                raise ValueError("memory_recorded_origin_compatibility_binding")
        if recorded and isinstance(grant, recorded_owner.RecordedLineageCapacityGrant):
            from types import SimpleNamespace

            def lineage_read(name):
                values = read(name, optional=True)
                return tuple(values) if values is not None else None

            proof = recorded_owner.lineage_capacity_proof(
                SimpleNamespace(run_id=run_id, read=lineage_read)
            )
            if proof is None or proof != grant.lineage_capacity_checksum:
                raise ValueError("memory_recorded_lineage_capacity_binding")
        if recorded and isinstance(grant, recorded_owner.RecordedPartitionGrant):
            for prior_key, expected in (
                (recorded_owner.GRANT_KEY, grant.previous_local_grant_checksum),
                (
                    "accepted:" + recorded_owner.ACCEPTANCE_KEY,
                    grant.previous_local_receipt_checksum,
                ),
            ):
                if expected is not None:
                    previous = read(prior_key)
                    actual = (
                        digest(previous[0])
                        if prior_key == recorded_owner.GRANT_KEY
                        else digest(tuple(previous))
                    )
                    if actual != expected:
                        raise ValueError("memory_recorded_previous_local_attempt_changed")
        records = read_records(stored[0])
        submission = read_submission(read(SUBMISSION_KEY)[0])
        original = read("multimodal-case-result")[0]
        if recorded and (
            result.get("confirmed_observations") != original.get("confirmed_observations")
            or result.get("previous_attempts") != original.get("previous_attempts")
            or result.get("task") != original.get("task")
            or result.get("original_requests") != original.get("new_requests")
        ):
            raise ValueError("memory_recorded_original_binding")
        if (
            digest(original) != result["original_result_checksum"]
            or grant.failed_result_checksum != digest(original)
            or grant.state_checksum != digest(parent)
            or grant.aggregate_checksum != digest(submission)
            or digest(grant) != result["finalization_grant_checksum"]
            or records.permission_checksum != digest(grant)
            or stored[3].get("permission_checksum") != digest(grant)
            or records.candidate != submission.candidate
            or (partial and records.pending_batch_ids != submission.pending_batch_ids)
            or (recorded and hasattr(submission, "pending_batch_ids"))
            or stored[3]["status"] != "accepted"
            or result["new_requests"] != 0
            or result["execution_kind"]
            != (
                "complete_recorded_fact_finalization_not_run_resume"
                if recorded
                else "partial_historical_fact_finalization_not_run_resume"
            )
        ):
            raise ValueError("memory_multimodal_partial_binding")
    if result["schema_version"] == "multimodal-case-result/1.2.0" or (
        recorded and "confirmed_observations" in result
    ):
        provenance = result.get("confirmed_observations")
        if (
            not isinstance(provenance, dict)
            or report.get("confirmed_observations") != provenance
            or provenance.get("schema_version") != "confirmed-visual-review-input/1.0.0"
            or provenance.get("new_vision_requests") != 0
            or provenance.get("unknown_vision_replayed") is not False
            or not provenance.get("origins")
        ):
            raise ValueError("memory_confirmed_visual_provenance")
        from packages.evidence_details.acceptance_models import read_receipt, read_records

        observations = {}
        for source in provenance["origins"]:
            values = read("accepted:" + source["acceptance_key"], source_run=source["run_id"])
            records, receipt = read_records(values[0]), read_receipt(values[3])
            if (
                receipt.status != "accepted"
                or digest(receipt) != source["receipt_checksum"]
                or digest(records) != source["records_checksum"]
                or receipt.records_checksum != digest(records)
                or len(records.observations) != 1
                or source["target"] in observations
                or records.scope.case_code != case_code
                or records.scope.run_id != source["run_id"]
            ):
                raise ValueError("memory_confirmed_visual_source_binding")
            observation = records.observations[0]
            if observation.model_dump(mode="json") not in parent["material"]["observations"]:
                raise ValueError("memory_confirmed_visual_observation_binding")
            observations[source["target"]] = observation
        if digest(observations) != provenance["observations_checksum"]:
            raise ValueError("memory_confirmed_visual_set_binding")
    for resource in parent["resources"]:
        a, _ = owner.artifact(case_code, resource["code"])
        if a.checksum != resource["checksum"]:
            raise ValueError("memory_multimodal_resource")
        refs[a.asset_code] = a
    steps = []
    if kind == "graph_snapshot":
        if target != result["task"]["target_code"]:
            raise ValueError("memory_multimodal_target")
        modalities = sorted(
            {m for r in parent["policy"]["requirements"] for m in r["input_modalities"]}
        )
    else:
        item = next((i for i in parent["policy"]["items"] if i["id"] == target), None)
        requirement = next((r for r in parent["policy"]["requirements"] if r["id"] == target), None)
        if item is None or requirement is None:
            raise ValueError("memory_source_missing")
        modalities = requirement["input_modalities"]
        for operation in ("read_input", "review", "report"):
            steps.append(
                dict(
                    operation=operation,
                    tool="RESEARCH:MULTIMODAL_TASK",
                    version="1.0.0",
                    implementation=item["method"],
                    source_checksum=digest(item),
                    input_roles=tuple(modalities),
                    output_roles=(operation + "_record",),
                    artifact_codes=(origin.asset_code,),
                )
            )
    sources = tuple(
        owner.reference(case_code=case_code, run_id=run_id, code=k) for k in sorted(refs)
    )
    return archive_record_class(sources)(
        key=key,
        kind=kind,
        case_code=case_code,
        run_id=run_id,
        applicability=("USE:RESEARCH_WORKFLOW",),
        nature="source_fact",
        sources=sources,
        detail=dict(
            type="case_snapshot" if kind == "graph_snapshot" else "branch_workflow",
            target=target,
            modality="+".join(modalities),
            modality_source="original_multimodal_task_report",
            origin_code=origin.asset_code,
            steps=tuple(steps),
            outcome="partial" if partial else "recorded",
            **({"execution_snapshot": scope_context} if scope_context is not None else {}),
        ),
    )

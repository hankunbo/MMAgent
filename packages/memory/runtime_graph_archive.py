"""Archive an original Runtime report and tool workflow, not caller-authored values."""

import json

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.memory.archive_models import ArchiveRecord


def prepare(owner, *, key, case_code, run_id, origin, envelope, kind, target):
    if envelope.get("key_checksum") == digest("accepted-aggregation-report"):
        from .aggregation_archive import prepare as prepare_aggregation

        return prepare_aggregation(
            owner,
            key=key,
            case_code=case_code,
            run_id=run_id,
            origin=origin,
            envelope=envelope,
            kind=kind,
            target=target,
        )
    if envelope.get("value", {}).get("multimodal_profile") == "multimodal-task-report/1.0.0":
        from .multimodal_archive import prepare as prepare_multimodal

        return prepare_multimodal(
            owner,
            key=key,
            case_code=case_code,
            run_id=run_id,
            origin=origin,
            envelope=envelope,
            kind=kind,
            target=target,
        )
    if (
        kind not in {"graph_snapshot", "tool_experience"}
        or envelope.get("run_id") != run_id
        or envelope.get("key_checksum")
        not in {digest("dynamic-case-report"), digest("dynamic-case-confirmed-report")}
        or envelope.get("index") != 0
        or owner.read_run_case(run_id) != case_code
    ):
        raise ValueError("memory_runtime_report_binding")
    report = envelope["value"]
    if report.get("schema_version") != "dynamic-case-report/1.0.0":
        raise ValueError("memory_runtime_report_version")
    result = report["result"]
    final = result["phases"][-1]
    if result["case_code"] != case_code or final["run_id"] != run_id:
        raise ValueError("memory_runtime_report_scope")
    refs = {origin.asset_code: origin}

    def read_document(name):
        artifacts = owner.runtime.read_observation_bundle(
            identity("TASK", run_id + ":agent:" + name)
        )
        if not artifacts:
            raise ValueError("memory_source_missing")
        values = []
        for index, artifact in enumerate(artifacts):
            a, raw = owner.artifact(case_code, artifact.asset_code)
            doc = json.loads(raw)
            if (
                doc.get("run_id"),
                doc.get("root_checksum"),
                doc.get("key_checksum"),
                doc.get("index"),
            ) != (run_id, envelope["root_checksum"], digest(name), index):
                raise ValueError("memory_runtime_document_binding")
            refs[a.asset_code] = a
            values.append(doc["value"])
        return values

    if not final["accepted"]:
        from .runtime_zero_archive import prepare_zero

        return prepare_zero(
            owner,
            key=key,
            case_code=case_code,
            run_id=run_id,
            origin=origin,
            envelope=envelope,
            kind=kind,
            target=target,
            refs=refs,
            read_document=read_document,
        )
    saved = read_document("accepted:" + final["acceptance_key"])
    parent = read_document("dynamic-phase-input")[0]
    detail = report["original_review"]
    if (saved[0], saved[1], saved[2]["scores"], saved[3]) != (
        detail["records"],
        detail["graph"],
        report["scores"],
        detail["receipt"],
    ):
        raise ValueError("memory_runtime_report_content")
    if detail["material"] != parent["material"] or saved[3]["status"] != "accepted":
        raise ValueError("memory_runtime_parent_binding")
    if envelope["key_checksum"] == digest("dynamic-case-confirmed-report"):
        from packages.evidence_details.finalization import GRANT_KEY, FinalizationGrant
        from packages.grounding.models import validate

        grant = validate(FinalizationGrant, read_document(GRANT_KEY)[0])
        original = read_document("dynamic-case-result")[0]
        finalized = read_document("dynamic-case-confirmed-result")[0]
        proof = report.get("recovery", {})
        if (
            proof.get("schema_version") != "confirmed-review-report-provenance/1.0.0"
            or finalized != result
            or result.get("recovery") != proof
            or proof.get("grant_checksum") != digest(grant)
            or saved[3].get("permission_checksum") != digest(grant)
            or proof.get("failed_result_checksum") != digest(original)
            or grant.failed_result_checksum != digest(original)
            or grant.state_checksum != digest(parent)
            or proof.get("original_candidate_checksum") != digest(saved[0]["candidate"])
            or proof.get("new_requests") != 0
            or proof.get("stopped_run_resumed") is not False
        ):
            raise ValueError("memory_runtime_finalization_binding")
    elif "recovery" in report:
        raise ValueError("memory_runtime_finalization_binding")
    # The original report is evidence of what happened, not present-day release authority.
    for resource in parent["resources"]:
        try:
            asset, _ = owner.artifact(case_code, resource["code"])
        except ValueError:
            raise ValueError("memory_runtime_resource_binding") from None
        if asset.checksum != resource["checksum"]:
            raise ValueError("memory_runtime_resource_binding")
        refs[asset.asset_code] = asset
    steps = []
    if kind == "tool_experience":
        selected = next((i for i in parent["policy"]["items"] if i["id"] == target), None)
        if selected is None:
            raise ValueError("memory_source_missing")
        if target == "criterion:AGE":
            from .rule_archive import prepare_age

            return prepare_age(
                owner,
                key=key,
                case_code=case_code,
                run_id=run_id,
                origin=origin,
                parent=parent,
                refs=refs,
                selected=selected,
            )
        tool_records = [
            a
            for a in refs.values()
            if a.artifact_type_code == "ARTIFACT:RESEARCH_FULL_TOOL_EVIDENCE"
        ]
        if len(tool_records) != 1:
            raise ValueError("memory_source_missing")
        source = tool_records[0]
        record = json.loads(owner.read_bytes(source))
        tasks = {
            "association:VASCULAR_SKELETON_DENSITY": ("avod", "av_metrics"),
            "association:VASCULAR_FRACTAL": ("avod", "av_metrics"),
            "association:AV_TORTUOSITY": ("avod", "av_metrics"),
            "association:ARTERIOLAR_CALIBER": ("avod", "av_metrics"),
            "association:VENULAR_BRANCH_ANGLE": ("avod", "branch"),
            "association:DISC_SHAPE_CONTEXT": ("ellipse",),
        }.get(target)
        if tasks is None:
            raise ValueError("memory_workflow_target_not_installed")
        for task in tasks:
            r = record["results"][task]
            if r["execution"]["exit_code"] != 0 or not r["observation"]:
                raise ValueError("memory_workflow_not_completed")
            manifest = r.get("manifest", r["execution"].get("manifest"))
            # The original manifest, not a label selected by the caller, names the tool.
            code = manifest.get("tool_code", manifest.get("code"))
            if not code:
                raise ValueError("memory_workflow_manifest_identity")
            steps.append(
                dict(
                    operation="segment" if task == "avod" else "measure",
                    tool=code,
                    version=manifest.get("version", "1.0.0"),
                    implementation=manifest.get("implementation_key", "native_registered"),
                    source_checksum=digest(manifest),
                    input_roles=("current_case_aligned_inputs",),
                    output_roles=(task,),
                    artifact_codes=(source.asset_code,),
                )
            )
    return ArchiveRecord(
        key=key,
        kind=kind,
        case_code=case_code,
        run_id=run_id,
        applicability=("USE:RESEARCH_WORKFLOW",),
        nature="source_fact",
        sources=tuple(
            owner.reference(case_code=case_code, run_id=run_id, code=k) for k in sorted(refs)
        ),
        detail=dict(
            type="case_snapshot" if kind == "graph_snapshot" else "branch_workflow",
            target=target,
            modality="SLO",
            modality_source="original_bound_runtime_report",
            origin_code=origin.asset_code,
            steps=tuple(steps),
            outcome="recorded",
        ),
    )

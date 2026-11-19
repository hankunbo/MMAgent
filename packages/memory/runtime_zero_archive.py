"""Partial historical snapshot with zero acquired items, never a medical acceptance."""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_scoring.accepted import build_accepted
from packages.grounding.models import digest, validate
from packages.memory.archive_models import ArchiveRecord


def prepare_zero(
    owner, *, key, case_code, run_id, origin, envelope, kind, target, refs, read_document
):
    report = envelope["value"]
    result, phase = report["result"], report["result"]["phases"][-1]
    if (
        kind != "graph_snapshot"
        or target != "glaucoma"
        or envelope["key_checksum"] != digest("dynamic-case-report")
        or report.get("original_review") is not None
        or "recovery" in report
        or result.get("complete") is not True
        or phase.get("accepted") is not False
        or phase.get("evidence_limited") is not True
        or phase.get("error") is not None
        or phase.get("acceptance_key") is not None
    ):
        raise ValueError("memory_runtime_zero_scope")
    parent = validate(DomainState, read_document("dynamic-phase-input")[0])
    zero = read_document("dynamic-zero-evidence")[0]
    if (
        parent.material.scope.case_code != case_code
        or parent.material.scope.run_id != run_id
        or parent.policy.items
        or read_document("dynamic-phase-result") != [phase]
        or read_document("dynamic-case-result") != [result]
        or zero.get("status") != "evidence_limited"
        or zero.get("no_review_requests") is not True
        or digest(zero.get("requirements")) != digest(parent.policy.requirements)
        or digest(zero.get("scores")) != digest(build_accepted(parent, None))
        or digest(report.get("scores")) != digest(zero["scores"])
    ):
        raise ValueError("memory_runtime_zero_binding")
    for resource in parent.resources:
        asset, _ = owner.artifact(case_code, resource.code)
        if asset.checksum != resource.checksum:
            raise ValueError("memory_runtime_resource_binding")
        refs[asset.asset_code] = asset
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
            type="case_snapshot",
            target=target,
            modality="SLO",
            modality_source="original_bound_runtime_report",
            origin_code=origin.asset_code,
            steps=(),
            outcome="partial",
        ),
    )

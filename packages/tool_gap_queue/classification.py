"""Project only re-evaluated original owner results, preserving full materials separately."""

from packages.grounding.models import digest

from .models import EXTENSIBLE, Finding

# 06 categories are broader than extension eligibility. Source binding failures must
# never become a way to replace rejected material under a fresh task identity.
SOURCE_BLOCKS = frozenset(
    {
        "manifest_binding_mismatch",
        "profile_binding_mismatch",
        "probe_binding_mismatch",
        "environment_manifest_mismatch",
        "interpreter_checksum_mismatch",
        "resource_declaration_mismatch",
        "processing_policy_mismatch",
        "invocation_binding_missing",
    }
)
ADMISSION_EXTENSION_REASONS = frozenset(
    {
        "tool_outside_catalog",
        "manifest_missing",
        "manifest_unavailable",
        "tool_validation_rejected",
        "environment_unknown",
        "environment_unavailable",
        "probe_stale",
        "resource_unavailable",
        "manifest_port_mismatch",
        "profile_port_mismatch",
        "output_type_mismatch",
        "output_port_scope_mismatch",
        "output_anatomical_target_mismatch",
    }
)


def grounding(result, policy):
    blocked_inputs = {f.source for f in result.gaps if f.reason == "required_input_missing"}
    blocked_sources = {f.source for f in result.gaps if f.category == "safety_restriction"}
    extensions = {
        checksum: s.target_capability_ref.model_dump(mode="json")
        for s in result.suggestions
        for checksum in s.finding_checksums
    }
    findings = []
    for f in result.gaps:
        kind = f.category
        if f.asset_code and f.reason == "artifact_type_mismatch":
            kind = "artifact_type_mismatch"
        capability = extensions.get(digest(f))
        # 05 causal attribution already removes source vetoes and satisfied OR branches.
        findings.append(
            Finding(
                kind=kind,
                source_checksum=digest(f),
                capability=capability,
                queueable=capability is not None
                and f.source not in blocked_inputs
                and f.source not in blocked_sources
                and kind in policy.allowed
                and result.status != "truncated"
                and "decisive_research" in policy.purposes,
                reason=f.reason,
            )
        )
    return tuple(findings)


def admission(result, policy):
    vetoed = {
        r.candidate_ref
        for r in result.rejected
        if r.category == "policy_rejection" or r.reason in SOURCE_BLOCKS
    }
    missing = {
        r.candidate_ref
        for r in result.rejected
        if r.category in {"patient_data_missing", "input_unsatisfied"}
    }
    findings = []
    for r in result.rejected:
        kind = r.category
        if r.reason in SOURCE_BLOCKS:
            kind = "source_safety_blocked"
        if r.asset_code and r.reason == "artifact_type_mismatch":
            kind = "artifact_type_mismatch"
        # Input-port mismatch is a data/format request, never an invented converter.
        eligible = r.port is None or kind == "output_incompatible"
        findings.append(
            Finding(
                kind=kind,
                source_checksum=digest(r),
                reason=r.reason,
                capability=r.capability_ref.model_dump(mode="json") if kind in EXTENSIBLE else None,
                queueable=eligible
                and r.reason in ADMISSION_EXTENSION_REASONS
                and kind in policy.allowed
                and not result.truncated
                and r.candidate_ref not in vetoed | missing
                and r.candidate_ref not in result.retained_roots
                and result.purpose in policy.purposes,
            )
        )
    return tuple(findings)

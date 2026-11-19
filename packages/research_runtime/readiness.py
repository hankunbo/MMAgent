"""Deployment checks with no patient/run/plan identity and no tool launch."""

from pathlib import Path
from types import SimpleNamespace

from packages.fairclip_research.core import CONFIG, WEIGHT_BYTES, WEIGHT_SHA256
from packages.grounding.models import digest, validate
from packages.tool_admission.research_metadata import FAIRCLIP_METADATA, limited_metadata
from packages.tool_admission.service import Budget
from packages.tool_execution.trusted_models import TrustedLocalPolicy
from packages.tool_execution.trusted_process import NvidiaObservation, TrustedLocalProcessRunner
from packages.tool_workflow.compiler import check_command, local_path


class UnboundSingleCasePolicy(TrustedLocalPolicy):
    # A distinct installation type. The original execution policy still requires real plans.
    allowed_plan_checksums: tuple[()] = ()


def check_installation(installed, catalogs, owners, admission, compiler, config):
    policy = validate(UnboundSingleCasePolicy, installed["execution_policy"])
    if len(policy.gpu_pool) != 1 or len(policy.reviews) != 1:
        raise ValueError("research_single_tool_GPU_required")
    if policy.timeout_seconds > config["budget_seconds"]:
        raise ValueError("research_tool_exceeds_run_budget")
    candidate = admission.research.candidate
    invocation = admission.research.invocation
    safety = catalogs.current_safety(catalogs.catalog.reference())
    if candidate.implementation_ref in safety.disabled_implementations:
        raise ValueError("research_installed_tool_disabled")
    manifest = catalogs.read_manifest(
        catalogs.catalog.reference(), candidate.tool_ref, candidate.implementation_ref
    )
    if config.get("metadata_requirements") != FAIRCLIP_METADATA or not limited_metadata(
        manifest, candidate, "shadow"
    ):
        raise ValueError("research_fixed_metadata_review_required")
    # Original 06 deployment checks need only declared anatomical scope, no case identity.
    declaration = SimpleNamespace(satisfied_terms=admission.research.scope_terms)
    _, profile, _, _ = admission._deployment(
        declaration, candidate, invocation, Budget(admission.policy), manifest
    )
    allowlist = owners.read_allowlist(manifest, profile, compiler.config)
    check_command(manifest, profile, allowlist, compiler.config, catalogs.catalog, candidate)
    if local_path(allowlist.entrypoint_uri) != config["adapter_entrypoint"]:
        raise ValueError("research_fixed_entrypoint_required")
    if (
        allowlist.synthetic_only
        or allowlist.manifest_checksum != manifest.checksum
        or allowlist.profile_checksum != profile.checksum
        or allowlist.compiler_config_checksum != compiler.config.checksum
        or profile.runtime.gpu != "required"
        or len(profile.input_rules) != 1
        or len(profile.output_rules) != 1
    ):
        raise ValueError("research_real_single_tool_binding")
    if (
        digest(manifest.preprocessing_policy.get("fixed_preprocess"))
        != digest(CONFIG["preprocess"])
        or manifest.postprocessing_policy.get("classification_checksum")
        != config["classification_checksum"]
    ):
        raise ValueError("research_manifest_fixed_semantics")
    review = policy.reviews[0]
    if (review.implementation_ref, review.manifest_checksum, review.allowlist_checksum) != (
        candidate.implementation_ref,
        manifest.checksum,
        allowlist.checksum,
    ):
        raise ValueError("research_installed_review_binding")
    if not any(
        f.path == config["weight_path"]
        and f.checksum == "sha256:" + WEIGHT_SHA256
        and f.byte_size == WEIGHT_BYTES
        for f in review.files
    ):
        raise ValueError("research_fixed_weight_missing_from_review")
    if not any(v["checksum"] == "sha256:" + WEIGHT_SHA256 for v in installed["model_versions"]):
        raise ValueError("research_model_source_binding")
    if (
        review.executable.path,
        review.executable.checksum,
        review.entrypoint.path,
        review.entrypoint.checksum,
    ) != (
        local_path(allowlist.executable_uri),
        allowlist.executable_checksum,
        local_path(allowlist.entrypoint_uri),
        allowlist.entrypoint_checksum,
    ):
        raise ValueError("research_fixed_command_review")
    if (
        compiler.config.environment_variables
        or compiler.config.cpu_count > policy.cpu_count
        or compiler.config.memory_bytes > policy.ram_budget_bytes
        or compiler.config.timeout_seconds > policy.timeout_seconds
        or compiler.config.max_output_bytes > 4194304
    ):
        raise ValueError("research_installed_execution_budget")
    files = {f.path: f.model_dump(mode="json") for f in review.files}
    for record in config["adapter_files"]:
        if files.get(record["path"]) != record:
            raise ValueError("research_fixed_adapter_review")
    # Reuse original read-only runner checks without constructing/mutating its workspace.
    reader = SimpleNamespace(policy=policy, observation=NvidiaObservation())
    TrustedLocalProcessRunner.check_capabilities(reader)
    TrustedLocalProcessRunner.verify_review(reader, review)
    TrustedLocalProcessRunner.gpu_ready(reader)
    workspace = Path(installed["execution_workspace"])
    from packages.tool_execution.trusted_process import path_checked

    path_checked(workspace)
    if not workspace.is_dir() or workspace.stat().st_mode & 0o077:
        raise ValueError("research_execution_workspace_not_installed")
    return policy

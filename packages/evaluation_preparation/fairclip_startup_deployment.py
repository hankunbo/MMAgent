"""Explicit reviewed launcher revision through original deployment owners.

The native catalog and historical entrypoint remain byte-identical. The new
research overlay derives a new implementation identity; it is not publication
or a medical validation upgrade. No request can install an overlay implicitly.
"""

import json
from pathlib import Path

from apps.research_single_case import credential_boundary
from apps.vectorengine_single_case import write_once
from packages.grounding.models import Candidate, digest, seal, validate
from packages.integrations.medgraph import AdapterConfig, MedgraphCatalogReader, SnapshotStore
from packages.integrations.medgraph.configuration import load_projection_policy
from packages.integrations.medgraph.research import ResearchReview, ReviewedCatalog
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.research_runtime.deployment import (
    ROOT,
    load_configuration,
    medgraph_readonly_dsn,
    read_record,
)
from packages.research_runtime.readiness import UnboundSingleCasePolicy
from packages.tool_admission.models import DeploymentProfile, InvocationInput, OutputTargetBinding
from packages.tool_execution.trusted_models import ReviewedFile, TrustedToolReview
from packages.tool_workflow.models import CommandAllowlist

from .fairclip_requalification import pin, prepare
from .scoped_catalog import SingleToolSource


def reseal(model, value, **changes):
    return seal(model, **{k: v for k, v in dict(value, **changes).items() if k != "checksum"})


def install(output, *, selected_gpu, base_configuration=None, generic_training=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    config_path = base_configuration or prepare(
        output / "base", selected_gpu=selected_gpu, generic_training=generic_training
    )
    config = load_configuration(Path(config_path))
    bundle = json.loads(read_record(config["installed_bundle"], limit=8388608))
    if bundle["execution_policy"]["selected_gpu"] != selected_gpu:
        raise ValueError("fairclip_reviewed_host_mismatch")
    previous = bundle["execution_policy"]["reviews"][0]
    entry = pin(ROOT / "apps/fairclip_tool_v2.py")
    old_entry = previous["entrypoint"]
    if old_entry != pin(ROOT / "apps/fairclip_tool.py"):
        raise ValueError("fairclip_historical_entrypoint_drift")
    note = output / "review.md"
    note_text = (
        "# FairCLIP launcher revision 2 deployment review\n\n"
        "User-authorized two-image FairVLMed research repair. New absolute-script launcher "
        "removes only apps/ from sys.path, installs the repository root, and calls the "
        "unchanged fixed backend. --check-runtime imports dependencies without weights or "
        "images and checks the stdlib origin. No scores/weights/preprocessing/class names "
        "are changed. Original launcher and all historical manifests remain unchanged. "
        "The original deployment owner creates a new research implementation identity, "
        "allowlist and exact inventory. Current native safety, GPU and resource checks "
        "remain required. Native Tool remains smoke_tested, not clinically validated.\n\n"
        "Original preparation consumed a 300s total deadline before a process could start. "
        "New runs have 900s immutable total lifecycle budget including preparation; the "
        "original 300s subprocess limit remains. No old deadline is renewed.\n"
    )
    if generic_training:
        note_text = note_text.replace(
            "User-authorized two-image FairVLMed research repair.",
            "User-authorized generic Training FairVLMed input integration; "
            "each selected SLO input and bounded run is separately registered.",
        )
    note.write_text(note_text)
    raw = json.loads(read_record(bundle["research_review"]))
    native = raw["deployment_declaration"]
    native["entrypoint_uri"] = Path(entry["path"]).as_uri()
    native["invocation"]["argv"][1] = entry["path"]
    matches = [r for r in native["resources"] if r["uri"] == Path(old_entry["path"]).as_uri()]
    if len(matches) != 1:
        raise ValueError("fairclip_entrypoint_resource_count")
    matches[0].update(
        uri=native["entrypoint_uri"], sha256=entry["checksum"][7:], byte_size=entry["byte_size"]
    )
    # Provenance belongs to the native catalog. The separate review record and
    # derived implementation identify this launcher; never alter native facts.
    for resolution in raw["resolutions"]:
        resolution["evidence"] = pin(note)
    review = reseal(ResearchReview, raw)
    write_once(output / "research-review.json", review)
    record = pin(output / "research-review.json")
    boundary = credential_boundary()
    adapter = validate(AdapterConfig, bundle["adapter_config"])
    identity = tuple(native[k] for k in ("tool_code", "version", "implementation_key"))
    source = SingleToolSource(
        adapter, identity=identity, dsn_resolver=lambda: medgraph_readonly_dsn(boundary)
    )
    legacy = MedgraphCatalogReader(
        source,
        SnapshotStore(Path(bundle["snapshot_root"])),
        adapter,
        load_projection_policy(Path(bundle["projection_policy"]["path"]), ROOT),
    )
    resources = LocalResourcePolicy(
        allowed_roots=tuple(bundle["resource_roots"]), max_bytes=2147483648, timeout_seconds=60
    )
    catalogs = ReviewedCatalog(legacy, validate(ReviewedFile, record), resources)
    manifest, tool = catalogs.manifest, catalogs.tool
    source_ref = dict(
        kind="source_version",
        code="SOURCE:FAIRCLIP:LAUNCHER_V2",
        version="2.0.0",
        checksum=review.checksum,
    )
    profile = bundle["profile"]
    profile["output_target_bindings"] = [
        reseal(OutputTargetBinding, v, source_ref=source_ref).model_dump(mode="json")
        for v in profile["output_target_bindings"]
    ]
    profile = reseal(
        DeploymentProfile,
        profile,
        manifest_checksum=manifest.checksum,
        config_checksum=manifest.config_checksum,
        policy_checksum=manifest.policy_checksum,
    )
    candidate = validate(
        Candidate, dict(bundle["candidate"], implementation_ref=tool.implementation_ref)
    )
    invocation = validate(
        InvocationInput,
        dict(
            bundle["invocation"],
            implementation_ref=tool.implementation_ref,
            manifest_checksum=manifest.checksum,
            profile_checksum=profile.checksum,
        ),
    )
    allowlist = reseal(
        CommandAllowlist,
        bundle["allowlist"],
        source_ref=source_ref,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        entrypoint_uri=native["entrypoint_uri"],
        entrypoint_checksum=entry["checksum"],
        resources=native["resources"],
        argv=native["invocation"]["argv"],
    )
    execution = bundle["execution_policy"]
    execution["reviews"] = [
        validate(
            TrustedToolReview,
            dict(
                previous,
                implementation_ref=tool.implementation_ref,
                manifest_checksum=manifest.checksum,
                allowlist_checksum=allowlist.checksum,
                review_record=pin(note),
                entrypoint=entry,
            ),
        ).model_dump(mode="json")
    ]
    execution["reference"] = dict(
        execution["reference"],
        version="2.0.0",
        checksum=digest({k: v for k, v in execution.items() if k != "reference"}),
    )
    bundle.update(
        research_review=record,
        profile=profile.model_dump(mode="json"),
        candidate=candidate.model_dump(mode="json"),
        invocation=invocation.model_dump(mode="json"),
        allowlist=allowlist.model_dump(mode="json"),
        execution_policy=validate(UnboundSingleCasePolicy, execution).model_dump(mode="json"),
        startup_probe="fairclip-absolute-entry/1.0.0",
    )
    write_once(output / "installed-bundle.json", bundle)
    config.update(
        installed_bundle=pin(output / "installed-bundle.json"),
        adapter_entrypoint=entry["path"],
        budget_seconds=900,
        adapter_files=[entry if v == old_entry else v for v in config["adapter_files"]],
    )
    config["checksum"] = digest({k: v for k, v in config.items() if k != "checksum"})
    write_once(output / "deployment-config.json", config)
    write_once(
        output / "revision.json",
        dict(
            old_entry=old_entry,
            new_entry=entry,
            old_implementation=previous["implementation_ref"],
            new_implementation=tool.implementation_ref,
            classification_unchanged=True,
            native_catalog_not_written=True,
            execution_seconds=execution["timeout_seconds"],
            lifecycle_seconds=900,
        ),
    )
    return str(output / "deployment-config.json")

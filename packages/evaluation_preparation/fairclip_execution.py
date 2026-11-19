"""New visible cases through the installed original 06/FairCLIP lifecycle.

No mask dependency, new executor, checkpoint change, or hidden-label read.
The original request key owns attempts, UNKNOWN recovery, admission and scores.
"""

import json
import re
import time
from pathlib import Path

from apps.research_single_case import credential_boundary
from apps.vectorengine_single_case import write_once
from db.runtime.repository.dataset_input import DatasetInputRepository
from packages.dataset_adapters.service import VisibleInputReader
from packages.grounding.models import digest
from packages.research_runtime import deployment

from .case_execution import register
from .scoped_catalog import SingleToolSource


def execute(
    sample_id,
    *,
    recover_only=False,
    configuration="configs/research/fairclip_single_case.json",
    revision=None,
    registered=None,
    request_key=None,
    before_dispatch=None,
    cancel=None,
):
    started = time.monotonic()
    if revision is not None and not re.fullmatch(r"[a-z][a-z0-9-]{0,23}", revision):
        raise ValueError("fairclip_execution_revision")
    if registered is not None:
        from .generic_input import validate_registered

        registered = validate_registered(sample_id, registered)
    directory, binding, runtime, objects = registered or register(sample_id)
    visible = VisibleInputReader(DatasetInputRepository(runtime), objects).read(
        binding["case_code"]
    )
    if len(visible.case.artifacts) != 1:
        raise ValueError("fairclip_single_image_required")
    config = deployment.load_configuration(Path(configuration))
    installed = deployment.read_installed_bundle(config["installed_bundle"])
    declaration = json.loads(deployment.read_record(installed["research_review"]))[
        "deployment_declaration"
    ]
    from packages.research_runtime.fairclip_probe import CurrentFairCLIPProfiles

    output = directory / ("fairclip-" + revision if revision else "fairclip-v1")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if revision:
        write_once(
            output / "execution-input.json",
            dict(
                sample=sample_id,
                case_code=binding["case_code"],
                input_checksum=visible.case.artifacts[0].checksum,
                configuration_checksum=config["checksum"],
                revision=revision,
                previous_key="13-source-driven-fairclip-v1:" + sample_id,
            ),
        )

    def source_factory(adapter, *, dsn_resolver):
        return SingleToolSource(
            adapter,
            identity=tuple(declaration[k] for k in ("tool_code", "version", "implementation_key")),
            dsn_resolver=dsn_resolver,
        )

    def profile_factory(record, catalogs):
        from packages.grounding.models import validate
        from packages.tool_admission.models import DeploymentProfile

        owner = CurrentFairCLIPProfiles(
            record, catalogs, proof_path=output / "environment-probe.json"
        )
        candidate = validate(deployment.Candidate, installed["candidate"])
        manifest = owner.read_manifest(
            catalogs.catalog.reference(), candidate.tool_ref, candidate.implementation_ref
        )
        profile = validate(DeploymentProfile, installed["profile"])
        # A new environment observation hashes the full installed closure. Do
        # that bounded preparation BEFORE the original 30-second admission budget.
        # Admission still checks the exact, current, unexpired proof itself.
        owner.read_probe(manifest, profile)
        return owner

    assembly = deployment.assemble(
        config,
        credential_boundary(),
        source_factory=source_factory,
        profile_factory=profile_factory,
        deployment_profile="installed_fairclip/1.0.0",
    )
    assembled = time.monotonic()
    key = request_key or (
        ("13-fairclip-" + revision if revision else "13-source-driven-fairclip-v1")
        + ":"
        + sample_id
    )
    request = assembly.requests.register(visible, key)
    if assembly.requests.for_key(key) != request or request.scope.case_code != binding["case_code"]:
        raise ValueError("fairclip_case_binding")
    # The existing lifecycle handles precise recovery and current authorization.
    # A local output file is never used as permission to execute or accept evidence.
    if before_dispatch is not None:
        before_dispatch()
    options = dict(key=key, recover_only=recover_only)
    if cancel is not None:
        options["cancel"] = cancel
    result = assembly.service.run_registered(request, **options)
    finished = time.monotonic()
    write_once(output / "request.json", request)
    write_once(output / "result.json", result)
    write_once(
        output / "binding.json",
        dict(
            schema_version="new-case-fairclip-binding/1.0.0",
            case_code=binding["case_code"],
            sample_id=sample_id,
            request_checksum=digest(request),
            image_checksum=visible.case.artifacts[0].checksum,
            result_checksum=digest(result),
            configuration_checksum=config["checksum"],
            original_executor=True,
            mask_required=False,
            clinical_probability=False,
        ),
    )
    if revision and not (output / "timing.json").exists():
        write_once(
            output / "timing.json",
            dict(
                assembly_seconds=assembled - started,
                lifecycle_seconds=finished - assembled,
                total_seconds=finished - started,
                budget_seconds=config["budget_seconds"],
                immutable_deadline=result.get("deadline"),
                recover_only=recover_only,
            ),
        )
    return result

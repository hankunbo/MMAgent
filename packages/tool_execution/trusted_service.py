"""Explicit assembly for trusted GPU research; reuses 06B/C and 06D transactions.

No backend string, command, review, policy or approved flag on the request port.
The deployment owner installs the compiler/readers/reviews when constructing this
service. A hash by itself never authorizes a plan or closes a manifest gap.
"""

import json
import re
from pathlib import Path

from packages.grounding.models import digest, validate
from packages.object_store import checksum
from packages.tool_workflow import ExecutionPlan
from packages.tool_workflow.compiler import local_path
from packages.tool_workflow.models import TrustedResearchCompilerConfig

from .models import ExecutionError
from .service import ToolExecutionService
from .trusted_models import TrustedLocalPolicy, TrustedTechnicalOutcome
from .trusted_process import TrustedLocalProcessRunner, path_checked


class TrustedLocalGPUResearchService(ToolExecutionService):
    """Separate service type, with legacy result and commit-recovery implementation."""

    def __init__(self, *, compiler, catalogs, objects, qc, repository, policy, workspace):
        self._compiler, self._catalogs, self._objects, self._qc, self._repo = (
            compiler,
            catalogs,
            objects,
            qc,
            repository,
        )
        self._policy = validate(TrustedLocalPolicy, policy)
        self._runner = TrustedLocalProcessRunner(workspace, self._policy)

    def _validate(self, plan):
        if type(plan) is not ExecutionPlan:
            raise ExecutionError("complete_plan_bundle_required")
        self._compiler.validate_plan(plan)
        doc = plan.document
        if (
            doc.checksum not in self._policy.allowed_plan_checksums
            or doc.purpose != self._policy.purpose
            or json.loads(plan._context_json).get("kind") != "research"
        ):
            raise ExecutionError("trusted_registered_research_required")
        if type(doc.config) is not TrustedResearchCompilerConfig:
            raise ExecutionError("trusted_explicit_compiler_capabilities_required")
        if (
            doc.config.environment_variables
            or doc.config.cpu_count > self._policy.cpu_count
            or doc.config.memory_bytes > self._policy.ram_budget_bytes
            or doc.config.timeout_seconds > self._policy.timeout_seconds
            or doc.config.max_output_bytes > 4194304
        ):
            raise ExecutionError("trusted_plan_budget_or_environment_rejected")
        catalog = self._catalogs.read_snapshot(doc.catalog_ref)
        if catalog.synthetic:
            raise ExecutionError("trusted_real_catalog_required")
        if not any(s.deployment.profile.runtime.gpu == "required" for s in doc.steps):
            raise ExecutionError("trusted_gpu_requirement_missing")
        for step in doc.steps:
            if step.deployment.allowlist.synthetic_only or step.deployment.manifest.binding.gaps:
                raise ExecutionError("trusted_review_does_not_override_admission")
            review = self._runner.review_for(step)
            allowlist = step.deployment.allowlist
            if (
                review.executable.path != local_path(allowlist.executable_uri)
                or review.executable.checksum != allowlist.executable_checksum
                or review.entrypoint.path != local_path(allowlist.entrypoint_uri)
                or review.entrypoint.checksum != allowlist.entrypoint_checksum
            ):
                raise ExecutionError("trusted_review_command_mismatch")
            files = {f.path: f for f in review.files}
            for resource in allowlist.resources:
                resource_path = local_path(resource.uri)
                if resource_path in review.inventory_roots and Path(resource_path).is_dir():
                    if review.version != "trusted-tool-review/1.1.0":
                        raise ExecutionError("tree_resource_requires_review_1_1")
                    from packages.integrations.medgraph.resources import (
                        LocalResourcePolicy,
                        observe_resource,
                    )

                    observed = observe_resource(
                        resource.uri,
                        LocalResourcePolicy(
                            allowed_roots=review.inventory_roots,
                            max_bytes=self._policy.max_resource_bytes,
                            max_files=16384,
                            timeout_seconds=60,
                        ),
                    )
                    if (
                        not resource.required
                        or resource.sha256 is None
                        or observed.checksum != "sha256:" + resource.sha256
                        or observed.byte_size != resource.byte_size
                    ):
                        raise ExecutionError("trusted_review_resource_mismatch")
                    continue
                item = files.get(resource_path)
                if (
                    item is None
                    or not resource.required
                    or (
                        resource.sha256 is None
                        or item.checksum != "sha256:" + resource.sha256
                        or item.byte_size != resource.byte_size
                    )
                ):
                    raise ExecutionError("trusted_review_resource_mismatch")
            for output in step.outputs:
                self._qc.policy_for(output)
        self._runner.check_capabilities()
        return doc

    def execution_result(self, plan, idempotency_key, returned):
        """Unwrap only the installed technical contract; None proves no start.

        Called on the synchronous execute return by the domain adapter. Recovery
        without that return must keep using recover/recover_commit, never infer a
        rejection merely from a missing attempt.
        """
        if type(returned) is not TrustedTechnicalOutcome:
            raise ExecutionError("trusted_return_type_invalid")
        returned = validate(TrustedTechnicalOutcome, returned)
        if (
            returned.policy_checksum != digest(self._policy)
            or returned.capabilities != self._policy.capabilities
        ):
            raise ExecutionError("trusted_return_policy_mismatch")
        if returned.state == "recorded":
            return self.confirmed_result(plan, idempotency_key, returned.workflow)
        record = self._return_record(plan, idempotency_key)
        if (
            returned.state == "technical_rejection"
            and returned.reason == "controlled_validation_rejected"
            and record is None
        ):
            # Corroborate the pre-admission rejection with the installed owner.
            # Other rejections, quarantines and ACTIVE attempts remain uncertain.
            try:
                self._validate(plan)
            except (ValueError, KeyError):
                return None
        raise ExecutionError("trusted_return_unconfirmed")

    def execute(self, plan, idempotency_key, *, cancel=None):
        state, reason, workflow = "recorded", None, None
        try:
            self._validate(plan)
        except (ValueError, KeyError):
            return TrustedTechnicalOutcome(
                policy_checksum=digest(self._policy),
                capabilities=self._policy.capabilities,
                state="technical_rejection",
                reason="controlled_validation_rejected",
            )
        try:
            workflow = super().execute(plan, idempotency_key, cancel=cancel)
        except ExecutionError as exc:
            state = "technical_rejection"
            reason = str(exc)
            if "quarantined" in reason or "unconfirmed" in reason:
                state = "uncertain"
        except Exception:
            # Unknown launch/commit facts stay in the repository/journal. No replay,
            # no invented v1 terminal status, and no online human waiting state.
            state, reason = "uncertain", "attempt_outcome_uncertain_offline_recovery"
        if reason is not None and not re.fullmatch(r"[a-z0-9_]{1,128}", reason):
            reason = "controlled_execution_rejected"
        return TrustedTechnicalOutcome(
            policy_checksum=digest(self._policy),
            capabilities=self._policy.capabilities,
            state=state,
            reason=reason,
            workflow=workflow,
        )

    def _materialize(self, attempt, step, doc, selected):
        base, root, copied = self._runner.prepare(attempt["attempt_id"], step, doc.config)
        mapping, bindings = {}, {}
        for resource in step.deployment.allowlist.resources:
            source = local_path(resource.uri)
            path_checked(source)
            mapping[resource.uri] = source
            copied[resource.uri] = "sha256:" + resource.sha256
        index = 0
        for bound in step.inputs:
            for source, uri in zip(bound.sources, bound.logical_uris, strict=True):
                artifact = selected(source, uri)
                data = self._objects.read_bounded(artifact.uri, 4194304)
                if checksum(data) != artifact.checksum:
                    raise ExecutionError("input_checksum_mismatch")
                path = root / "in" / str(index)
                index += 1
                with path.open("xb") as stream:
                    stream.write(data)
                path.chmod(0o400)
                mapping[uri] = bindings[uri] = str(path)
                copied[uri] = checksum(data)
        for output in step.outputs:
            filename = output.logical_uri.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,95}", filename):
                raise ExecutionError("output_path_invalid")
            path = root / "out" / filename
            if path.exists() or path.is_symlink():
                raise ExecutionError("output_already_exists")
            mapping[output.logical_uri] = bindings[output.logical_uri] = str(path)
        argv = [bindings.get(token, token) for token in step.argv]
        if any(token.startswith(("object://", "plan-output://")) for token in argv):
            raise ExecutionError("logical_token_unbound")
        if Path(argv[0]) != path_checked(argv[0]):
            raise ExecutionError("trusted_executable_path_invalid")
        return base, root, copied, mapping, argv

"""Deterministic filtering of an already admitted search result. No search or execution."""

import time
from datetime import UTC, datetime

from packages.contracts import CapabilityGap, ExecutionPolicy, IntendedUse, validate_tool_use
from packages.grounding.compatibility import constraint_compatibility, tool_compatibility
from packages.grounding.models import CatalogSnapshot, canonical, digest, seal, validate
from packages.grounding.ports import ToolSafety
from packages.grounding.query import normalized_request
from packages.grounding.query_models import QueryResult
from packages.integrations.medgraph import AdapterError
from packages.integrations.medgraph.models import parse_deployment
from packages.integrations.medgraph.resources import (
    LocalResourcePolicy,
    observe_manifest_resources,
    observe_resource,
)

from .inputs import check_input
from .method_metadata import MetadataRequirements, metadata_requirements
from .models import (
    AdmissionError,
    AdmissionPolicy,
    AdmissionResult,
    AdmissionSelection,
    AdmittedPort,
    DeploymentGap,
    DeploymentProfile,
    EnvironmentProbe,
    Rejection,
    ResearchRequest,
    RetainedCandidate,
)
from .parameters import validate_parameters


def reason_category(reason):
    if reason.startswith("output_anatomical_target_"):
        return "output_incompatible"
    if reason == "patient_data_missing":
        return "patient_data_missing"
    if reason == "required_input_missing":
        return "input_unsatisfied"
    if reason in {"tool_outside_catalog", "manifest_missing", "manifest_unavailable"}:
        return "implementation_missing"
    if reason == "tool_validation_rejected":
        return "validation_insufficient"
    if reason == "resource_path_or_budget_rejected":
        return "policy_rejection"
    if reason.startswith(("environment_", "probe_", "resource_", "interpreter_")):
        return "environment_unavailable"
    if reason in {
        "synthetic_policy_on_real_catalog",
        "tool_not_active",
        "tool_safety_revoked",
        "intended_use_rejected",
        "contract_use_rejected",
        "manifest_gaps",
        "precondition_rejected",
        "deployment_profile_missing",
        "processing_policy_unknown",
        "research_metadata_declaration_binding",
        "research_method_metadata_binding",
    }:
        return "policy_rejection"
    return "input_incompatible"


class Budget:
    def __init__(self, policy):
        self.policy, self.bytes, self.checks, self.nodes = policy, 0, 0, 0
        self.deadline = time.monotonic() + policy.timeout_seconds

    def consume(self, size=0, *, node=False):
        self.bytes += size
        self.checks += 1
        self.nodes += int(node)
        if (
            time.monotonic() > self.deadline
            or self.bytes > self.policy.max_total_bytes
            or self.checks > self.policy.max_checks
            or self.nodes > self.policy.max_nodes
        ):
            raise AdmissionError("admission_budget_exhausted")


class ToolAdmissionService:
    """Assemble only with controlled services. Returned seals do not grant permissions."""

    def __init__(
        self,
        *,
        grounding,
        catalogs,
        manifests,
        profiles,
        probes,
        inputs,
        objects,
        resources,
        research=None,
        policy=None,
        clock=None,
    ):
        self.grounding, self.catalogs, self.manifests = grounding, catalogs, manifests
        self.profiles, self.probes, self.inputs, self.objects = profiles, probes, inputs, objects
        self.resources = validate(LocalResourcePolicy, resources)
        self.research = research
        self.policy = validate(AdmissionPolicy, policy or AdmissionPolicy())
        self.clock = clock or (lambda: datetime.now(UTC))

    def admit(self, request, result, selection):
        """Formal entry: mandatory Runtime/Published/lock/current safety revalidation."""
        request = normalized_request(request)
        selection = validate(AdmissionSelection, selection)
        selection = validate(
            AdmissionSelection,
            dict(
                candidate_refs=sorted(selection.candidate_refs),
                invocations=sorted(
                    selection.invocations, key=lambda i: digest(i.implementation_ref)
                ),
            ),
        )
        result = validate(QueryResult, self.grounding.validate_result(request, result))
        if result.request != request:
            raise AdmissionError("query_request_mismatch")
        roots = {c.identity: c for c in result.candidates}
        if not set(selection.candidate_refs) <= roots.keys():
            raise AdmissionError("candidate_outside_official_result")
        chosen = tuple(
            roots[c.identity] for c in result.candidates if c.identity in selection.candidate_refs
        )
        return self._evaluate(request, chosen, selection, result=result)

    def admit_research(self, request):
        return self._research_admission(request)

    def _research_admission(self, request, *, result_consumption=False):
        request = validate(ResearchRequest, request)
        if self.research is None:
            raise AdmissionError("research_registration_required")
        registered = validate(
            ResearchRequest,
            self.research.read_request(request.code, request.version, request.checksum),
        )
        if registered != request:
            raise AdmissionError("research_registration_mismatch")
        selection = AdmissionSelection(
            candidate_refs=(digest(request.candidate),), invocations=(request.invocation,)
        )
        return self._evaluate(request, (request,), selection, result_consumption=result_consumption)

    def validate_research_consumption(self, request, result):
        """Check current source/input/method, not permission to launch again.

        Only the expiry of the original execution environment observation is
        inapplicable to a confirmed artifact. Its identity and historical facts,
        current resources, registration, safety, purpose and inputs still match.
        No admission result is returned as an execution capability.
        """
        observed = validate(AdmissionResult, result)
        if observed != self._research_admission(request, result_consumption=True):
            raise AdmissionError("stale_or_forged_consumption_source")

    def validate_result(self, request, query_result, selection, result):
        observed = validate(AdmissionResult, result)
        if observed != self.admit(request, query_result, selection):
            raise AdmissionError("stale_or_forged_admission_result")
        return observed

    def validate_research_result(self, request, result):
        observed = validate(AdmissionResult, result)
        if observed != self.admit_research(request):
            raise AdmissionError("stale_or_forged_admission_result")
        return observed

    def inspect_research_gap_inputs(self, request):
        """11A metadata/source recheck even when deployment fails before input admission.

        This does not authorize execution or infer whether an unknown product is a raw
        patient modality. Format conversion remains a separately admitted 05/06 path.
        """
        from packages.grounding.compatibility import artifact_compatibility
        from packages.object_store import checksum

        from .models import InputRecord

        request = validate(ResearchRequest, request)
        if self.research is None:
            raise AdmissionError("research_registration_required")
        if (
            validate(
                ResearchRequest,
                self.research.read_request(request.code, request.version, request.checksum),
            )
            != request
        ):
            raise AdmissionError("research_registration_mismatch")
        facts, missing = [], []
        budget = Budget(self.policy)
        metadata = self._gap_method_metadata(request)
        for port in request.candidate.inputs:
            budget.consume()
            codes = next(i.asset_codes for i in request.inputs if i.port == port.port)
            if len(codes) < port.cardinality_min:
                missing.append(
                    dict(port=port.port.model_dump(mode="json"), reason="required_input_missing")
                )
            for code in codes:
                budget.consume()
                try:
                    record = validate(
                        InputRecord,
                        self.inputs.read_input(request.scope, request.lock.run_id, code),
                    )
                except KeyError:
                    missing.append(dict(asset_code=code, reason="patient_data_missing"))
                    continue
                expected = next((a for a in request.artifacts if a.asset_code == code), None)
                if record.metadata != expected:
                    raise AdmissionError("input_record_mismatch")
                reasons = artifact_compatibility(
                    record.metadata,
                    port.constraint,
                    request,
                    laterality_required=metadata.laterality_required,
                    acquisition_time_required=metadata.acquisition_time_required,
                )
                for reason in reasons:
                    missing.append(dict(asset_code=code, reason=reason))
                if record.content_kind not in {"image", "array"}:
                    missing.append(dict(asset_code=code, reason="input_content_forbidden"))
                # Bounded object read belongs to this owner; I/O failures propagate.
                budget.consume(record.byte_size)
                raw = self.objects.read_bounded(record.artifact.uri, self.policy.max_input_bytes)
                if checksum(raw) != record.artifact.checksum or len(raw) != record.byte_size:
                    raise AdmissionError("input_object_changed")
                facts.append(
                    dict(
                        asset_code=code,
                        source=record.source_ref.model_dump(mode="json"),
                        record_checksum=digest(record),
                    )
                )
        return dict(inputs=facts, missing=missing)

    def _gap_method_metadata(self, request):
        """Only new installed method reviews change legacy input diagnostics.

        A missing deployment remains diagnosable under the original strict input
        rules. Drift, revoked qualification and technical owner failures are not
        converted to missing patient data. This never grants execution authority.
        """
        from .method_metadata import VERSION

        try:
            manifest = self._manifest(request, request.candidate, None, diagnostic=True)
        except AdmissionError as exc:
            if str(exc) == "manifest_missing":
                return MetadataRequirements()
            raise
        raw = (manifest.preprocessing_policy or {}).get("metadata_requirements")
        if not isinstance(raw, dict) or raw.get("schema_version") != VERSION:
            return MetadataRequirements()
        if request.invocation is None:
            raise AdmissionError("invocation_binding_missing")
        if manifest.checksum != request.invocation.manifest_checksum:
            raise AdmissionError("manifest_binding_mismatch")
        catalog = validate(CatalogSnapshot, self.catalogs.read_snapshot(request.lock.catalog_ref))
        if catalog.reference() != request.lock.catalog_ref:
            raise AdmissionError("catalog_binding_mismatch")
        safety = validate(ToolSafety, self.catalogs.current_safety(request.lock.catalog_ref))
        if safety.catalog_ref != request.lock.catalog_ref:
            raise AdmissionError("safety_catalog_mismatch")
        reasons = tool_compatibility(
            request.candidate, catalog, request.compatibility_policy, safety
        )
        if reasons:
            raise AdmissionError(reasons[0])
        matches = [
            t
            for t in catalog.tools
            if (t.tool_ref, t.implementation_ref)
            == (request.candidate.tool_ref, request.candidate.implementation_ref)
        ]
        if len(matches) != 1:
            raise AdmissionError("tool_outside_catalog")
        return metadata_requirements(manifest, request.candidate, request.purpose, matches[0])

    def _manifest(self, request, candidate, invocation, *, diagnostic=False):
        if invocation is None and not diagnostic:
            raise AdmissionError("invocation_binding_missing")
        try:
            manifest = parse_deployment(
                self.manifests.read_manifest(
                    request.lock.catalog_ref, candidate.tool_ref, candidate.implementation_ref
                ),
            )
        except KeyError:
            raise AdmissionError("manifest_missing") from None
        if (
            manifest.catalog_ref,
            manifest.tool_ref,
            manifest.implementation_ref,
        ) != (
            request.lock.catalog_ref,
            candidate.tool_ref,
            candidate.implementation_ref,
        ):
            raise AdmissionError("manifest_binding_mismatch")
        if invocation is not None and manifest.checksum != invocation.manifest_checksum:
            raise AdmissionError("manifest_binding_mismatch")
        return manifest

    def _deployment(
        self, request, candidate, invocation, budget, manifest, *, result_consumption=False
    ):
        if manifest.binding.gaps:
            raise AdmissionError("manifest_gaps")
        if manifest.execution_manifest is None:
            raise AdmissionError("manifest_unavailable")
        expected_ports = {
            (i.port.direction, i.port.port_name, i.required, i.cardinality_min, i.cardinality_max)
            for i in candidate.inputs
        }
        actual_ports = {
            (p.direction, p.port_name, p.required, p.cardinality_min, p.cardinality_max)
            for p in manifest.ports
            if p.direction == "input"
        }
        if (
            expected_ports != actual_ports
            or len(actual_ports) != len(candidate.inputs)
            or (sum(p.direction == "input" for p in manifest.ports) != len(actual_ports))
            or not any(
                p.direction == "output" and p.port_name == candidate.output_port.port_name
                for p in manifest.ports
            )
        ):
            raise AdmissionError("manifest_port_mismatch")
        try:
            profile = validate(DeploymentProfile, self.profiles.read_profile(manifest))
        except KeyError:
            raise AdmissionError("deployment_profile_missing") from None
        if (
            profile.checksum,
            profile.manifest_checksum,
            profile.config_checksum,
            profile.policy_checksum,
        ) != (
            invocation.profile_checksum,
            manifest.checksum,
            manifest.config_checksum,
            manifest.policy_checksum,
        ):
            raise AdmissionError("profile_binding_mismatch")
        # Deterministic output semantics precede probes, resources and input bytes.
        profile.check_output_targets(candidate, request.satisfied_terms)
        if {p.port for p in profile.input_rules} != {i.port for i in candidate.inputs}:
            raise AdmissionError("profile_port_mismatch")
        if any(
            not set(p.anatomical_targets) <= set(request.satisfied_terms)
            for p in profile.input_rules
        ):
            raise AdmissionError("anatomical_target_mismatch")
        for policy, expected in (
            (manifest.preprocessing_policy, profile.preprocessing_checksum),
            (manifest.postprocessing_policy, profile.postprocessing_checksum),
        ):
            if not policy or not policy.get("version") or not policy.get("policy_code"):
                raise AdmissionError("processing_policy_unknown")
            if digest(policy) != expected:
                raise AdmissionError("processing_policy_mismatch")
        if manifest.parameter_schema is None:
            raise AdmissionError("parameter_schema_missing")
        validate_parameters(manifest.parameter_schema, invocation.parameters)
        probe = validate(EnvironmentProbe, self.probes.read_probe(manifest, profile))
        if (probe.manifest_checksum, probe.profile_checksum, probe.runtime_checksum) != (
            manifest.checksum,
            profile.checksum,
            digest(profile.runtime),
        ):
            raise AdmissionError("probe_binding_mismatch")
        if manifest.execution_manifest.runtime_environment.get(
            "admission_runtime"
        ) != profile.runtime.model_dump(mode="json"):
            raise AdmissionError("environment_manifest_mismatch")
        now = self.clock()
        if now < probe.observed_at or (not result_consumption and now >= probe.expires_at):
            raise AdmissionError("probe_stale")
        facts = (probe.interpreter, probe.dependencies, probe.resources)
        if "unknown" in facts or probe.preconditions == "unknown" or probe.gpu == "unknown":
            raise AdmissionError("environment_unknown")
        if any(f != "available" for f in facts) or probe.gpu != (
            "available" if profile.runtime.gpu == "required" else "not_required"
        ):
            raise AdmissionError("environment_unavailable")
        if probe.preconditions != "satisfied":
            raise AdmissionError("precondition_rejected")
        bounded_resources = validate(
            LocalResourcePolicy,
            {
                **self.resources.model_dump(mode="json"),
                "max_bytes": min(self.resources.max_bytes, self.policy.max_total_bytes),
                "timeout_seconds": min(self.resources.timeout_seconds, self.policy.timeout_seconds),
            },
        )
        try:
            interpreter = observe_resource(profile.runtime.interpreter_uri, bounded_resources)
        except FileNotFoundError:
            raise AdmissionError("environment_unavailable") from None
        except AdapterError:
            raise AdmissionError("resource_path_or_budget_rejected") from None
        budget.consume(interpreter.byte_size)
        if (
            interpreter.kind != "file"
            or interpreter.checksum != profile.runtime.interpreter_checksum
        ):
            raise AdmissionError("interpreter_checksum_mismatch")
        try:
            observed = observe_manifest_resources(manifest, bounded_resources)
        except FileNotFoundError:
            raise AdmissionError("resource_unavailable") from None
        except AdapterError:
            raise AdmissionError("resource_path_or_budget_rejected") from None
        budget.consume(sum(o.observation.byte_size for o in observed.observations))
        if any(
            o.declared_checksum_matches is not True or o.declared_size_matches is not True
            for o in observed.observations
        ):
            raise AdmissionError("resource_declaration_mismatch")
        return manifest, profile, probe, observed

    def _evaluate(self, request, roots, selection, *, result=None, result_consumption=False):
        catalog = validate(CatalogSnapshot, self.catalogs.read_snapshot(request.lock.catalog_ref))
        if catalog.reference() != request.lock.catalog_ref:
            raise AdmissionError("catalog_binding_mismatch")
        safety = validate(ToolSafety, self.catalogs.current_safety(request.lock.catalog_ref))
        if safety.catalog_ref != request.lock.catalog_ref:
            raise AdmissionError("safety_catalog_mismatch")
        if result and digest(safety) != result.safety_checksum:
            raise AdmissionError("safety_changed_during_admission")
        budget = Budget(self.policy)
        invocations = {i.implementation_ref: i for i in selection.invocations}
        retained, rejected, seen, active = {}, [], {}, set()
        deployment_gaps = []
        purpose = "decisive_research" if result else request.purpose

        def reject(identity, c, reason, port=None, asset=None):
            budget.consume()
            rejected.append(
                Rejection(
                    candidate_ref=identity,
                    tool_ref=c.tool_ref,
                    implementation_ref=c.implementation_ref,
                    capability_ref=c.capability_ref,
                    port=port,
                    asset_code=asset,
                    constraint=c.output
                    if port is None or port == c.output_port
                    else next(i.constraint for i in c.inputs if i.port == port),
                    reason=reason,
                    category=reason_category(reason),
                )
            )

        def visit(node, depth=0):
            identity = node.identity if result else digest(node.candidate)
            if identity in seen:
                return seen[identity]
            if identity in active or depth > 12:
                raise AdmissionError("admission_budget_exhausted")
            budget.consume(node=True)
            active.add(identity)
            c = node.candidate
            policy = request.lock.policy if result else request.compatibility_policy
            reasons = list(tool_compatibility(c, catalog, policy, safety))
            if policy.synthetic_only and not catalog.synthetic:
                reasons.append("synthetic_policy_on_real_catalog")
            matches = [
                t
                for t in catalog.tools
                if (t.tool_ref, t.implementation_ref) == (c.tool_ref, c.implementation_ref)
            ]
            tool = matches[0] if len(matches) == 1 else None
            if tool is not None:
                if not tool.enabled or tool.catalog_status.value != "active":
                    reasons.append("tool_not_active")
                if tool.implementation_ref in safety.disabled_implementations:
                    reasons.append("tool_safety_revoked")
                if tool.validation_status != c.validation_required:
                    reasons.append("tool_validation_rejected")
                if (
                    c.intended_use != IntendedUse.RESEARCH_ONLY
                    or tool.intended_use != IntendedUse.RESEARCH_ONLY
                    or c.execution_policy != ExecutionPolicy.RESEARCH
                ):
                    reasons.append("intended_use_rejected")
                try:
                    validate_tool_use(
                        tool.catalog_status,
                        tool.validation_status,
                        tool.intended_use,
                        c.execution_policy,
                        decisive_evidence=bool(result),
                    )
                except (ValueError, TypeError):
                    reasons.append("contract_use_rejected")
            manifest = None
            metadata = MetadataRequirements()
            try:
                manifest = self._manifest(request, c, invocations.get(c.implementation_ref))
                metadata = metadata_requirements(
                    manifest, c, None if result else request.purpose, tool
                )
                if manifest.binding.gaps:
                    reasons.append("manifest_gaps")
                    deployment_gaps.append(
                        DeploymentGap(
                            candidate_ref=identity,
                            tool_ref=c.tool_ref,
                            implementation_ref=c.implementation_ref,
                            manifest_checksum=manifest.checksum,
                            gaps=manifest.binding.gaps,
                        )
                    )
            except AdmissionError as error:
                reasons.append(str(error))
            for constraint in (c.output, *(i.constraint for i in c.inputs)):
                reasons.extend(
                    constraint_compatibility(
                        constraint, request, laterality_required=metadata.laterality_required
                    )
                )
            for reason in dict.fromkeys(reasons):
                reject(identity, c, reason)
            if reasons:
                active.remove(identity)
                seen[identity] = False
                return False
            try:
                manifest, profile, probe, resources = self._deployment(
                    request,
                    c,
                    invocations.get(c.implementation_ref),
                    budget,
                    manifest,
                    result_consumption=result_consumption,
                )
            except AdmissionError as error:
                if str(error) == "admission_budget_exhausted":
                    raise
                reject(
                    identity,
                    c,
                    str(error),
                    c.output_port if str(error).startswith("output_anatomical_target_") else None,
                )
                active.remove(identity)
                seen[identity] = False
                return False
            ports, failed = [], False
            for i in c.inputs:
                rule = next(p for p in profile.input_rules if p.port == i.port)
                if result:
                    dep = next(d for d in node.dependencies if d.port_name == i.port.port_name)
                    codes, producers = dep.artifact_options, dep.producer_options
                else:
                    codes = next(d.asset_codes for d in request.inputs if d.port == i.port)
                    producers = ()
                if not result and len(codes) > i.cardinality_max:
                    reject(identity, c, "input_cardinality_exceeded", i.port)
                    failed = True
                    continue
                artifacts, producer_refs = [], []
                for code in codes:
                    budget.consume()
                    try:
                        artifacts.append(
                            check_input(
                                self.inputs,
                                self.objects,
                                request,
                                code,
                                i.constraint,
                                rule,
                                self.policy,
                                budget,
                                metadata=metadata,
                            )
                        )
                    except AdmissionError as error:
                        if str(error) == "admission_budget_exhausted":
                            raise
                        reject(identity, c, str(error), i.port, code)
                for producer in producers:
                    if visit(producer, depth + 1):
                        produced = retained[producer.identity].output_rule
                        if (
                            not set(produced.media_types) <= set(rule.media_types)
                            or len(produced.dimensions_min) != len(rule.dimensions_min)
                            or any(
                                a < b
                                for a, b in zip(
                                    produced.dimensions_min, rule.dimensions_min, strict=True
                                )
                            )
                            or any(
                                a > b
                                for a, b in zip(
                                    produced.dimensions_max, rule.dimensions_max, strict=True
                                )
                            )
                            or not set(rule.anatomical_targets) <= set(produced.anatomical_targets)
                        ):
                            reject(identity, c, "producer_format_or_dimensions_mismatch", i.port)
                        else:
                            producer_refs.append(producer.identity)
                if len(artifacts) + bool(producer_refs) < i.cardinality_min:
                    reject(identity, c, "required_input_missing", i.port)
                    failed = True
                    continue
                ports.append(
                    AdmittedPort(
                        port=i.port,
                        required=i.required,
                        cardinality_min=i.cardinality_min,
                        cardinality_max=i.cardinality_max,
                        artifacts=tuple(artifacts),
                        producer_refs=tuple(producer_refs),
                        optional_missing=not i.required and not artifacts and not producer_refs,
                    )
                )
            active.remove(identity)
            seen[identity] = not failed
            if not failed:
                retained[identity] = RetainedCandidate(
                    identity=identity,
                    candidate=c,
                    manifest_checksum=manifest.checksum,
                    profile_checksum=profile.checksum,
                    probe_checksum=probe.checksum,
                    resources_checksum=resources.checksum,
                    invocation_checksum=digest(invocations[c.implementation_ref]),
                    inputs=tuple(ports),
                    output_rule=next(p for p in profile.output_rules if p.port == c.output_port),
                )
            return not failed

        kept, truncated = [], False
        try:
            # Retain 06A gaps even when 05 rejected the formal candidate. This is
            # a diagnostic read of exact catalog members, never fallback admission.
            if result:
                diagnosed = set()
                for refusal in result.rejected:
                    c = refusal.source.candidate
                    key = (c.tool_ref, c.implementation_ref)
                    if key in diagnosed or not any(
                        (t.tool_ref, t.implementation_ref) == key for t in catalog.tools
                    ):
                        continue
                    diagnosed.add(key)
                    budget.consume()
                    try:
                        m = self._manifest(request, c, None, diagnostic=True)
                    except AdmissionError as error:
                        if str(error) == "manifest_missing":
                            continue
                        raise
                    if m.binding.gaps:
                        deployment_gaps.append(
                            DeploymentGap(
                                origin="official_rejection",
                                candidate_ref=digest(refusal.source),
                                tool_ref=c.tool_ref,
                                implementation_ref=c.implementation_ref,
                                manifest_checksum=m.checksum,
                                gaps=m.binding.gaps,
                            )
                        )
            for root in roots:
                if visit(root):
                    kept.append(root.identity if result else digest(root.candidate))
        except AdmissionError as error:
            if str(error) != "admission_budget_exhausted":
                raise
            truncated = True
            kept, retained = [], {}
        if result and result.status == "truncated":
            truncated = True
            kept, retained = [], {}
        # Remove isolated accepted subpaths of failed roots. Preserve every viable OR option.
        reachable = set()

        def collect(identity):
            if identity in reachable:
                return
            reachable.add(identity)
            for port in retained[identity].inputs:
                for producer in port.producer_refs:
                    collect(producer)

        for identity in kept:
            collect(identity)
        retained = {k: v for k, v in retained.items() if k in reachable}
        if validate(ToolSafety, self.catalogs.current_safety(request.lock.catalog_ref)) != safety:
            raise AdmissionError("safety_changed_during_admission")
        if result:
            # Published state and Runtime locks may have changed during file inspection.
            self.grounding.validate_result(request, result)
        else:
            if (
                validate(
                    ResearchRequest,
                    self.research.read_request(request.code, request.version, request.checksum),
                )
                != request
            ):
                raise AdmissionError("research_registration_changed")
        gaps = list(s.capability_gap for s in result.suggestions) if result else []
        if result and not truncated:
            blocked = set()
            unsatisfied = {
                (r.candidate_ref, r.port.port_name)
                for r in rejected
                if r.reason == "required_input_missing" and r.port
            }

            def trace_blocked(node):
                if node.identity in blocked or seen.get(node.identity) is not False:
                    return
                blocked.add(node.identity)
                for dep in node.dependencies:
                    if (node.identity, dep.port_name) in unsatisfied:
                        for producer in dep.producer_options:
                            trace_blocked(producer)

            for root in roots:
                trace_blocked(root)
            gaps.extend(
                resolved_gaps(
                    request, [r for r in rejected if r.candidate_ref in blocked], kept, retained
                )
            )
        output = seal(
            AdmissionResult,
            purpose=purpose,
            request_checksum=digest(request),
            selection_checksum=digest(selection),
            query_checksum=result.checksum if result else None,
            grounding_status=result.status if result else None,
            catalog_ref=request.lock.catalog_ref,
            scope=request.scope,
            run_id=request.lock.run_id,
            policy_checksum=digest(self.policy),
            safety_checksum=digest(safety),
            status="truncated" if truncated else "preparation_only" if kept else "rejected",
            retained_roots=kept,
            retained=list(retained.values()),
            rejected=rejected,
            deployment_gaps=deployment_gaps,
            upstream_findings=result.gaps if result else (),
            product_demands=result.product_demands if result else (),
            capability_gaps=gaps,
            truncated=truncated,
        )
        if len(canonical(output)) > self.policy.max_result_bytes:
            raise AdmissionError("admission_result_byte_limit")
        return output


def resolved_gaps(request, rejected, roots, retained):
    """Only own implementation/environment failures of blocked paths name a capability."""
    groups = {}
    for r in rejected:
        if r.port is None:
            groups.setdefault(r.candidate_ref, []).append(r)
    gaps = []
    gap_codes = {
        "implementation_missing": "NO_IMPLEMENTATION",
        "validation_insufficient": "LOW_RELIABILITY",
        "environment_unavailable": "EXECUTION_RESOURCE_UNAVAILABLE",
    }
    for identity, failures in groups.items():
        # Safety/policy vetoes dominate extensible conditions on this exact source.
        if any(r.category == "policy_rejection" for r in failures):
            continue
        codes = sorted({gap_codes[r.category] for r in failures if r.category in gap_codes})
        if not codes or identity in retained:
            continue
        r = failures[0]
        gaps.append(
            validate(
                CapabilityGap,
                dict(
                    schema_version="1.0",
                    contract_type="CapabilityGap",
                    gap_id="REQUIREMENT:OID_"
                    + digest((digest(request), failures)).split(":")[1][:32].upper(),
                    origin_run_id=request.lock.run_id,
                    target_node_code=request.target_node_code,
                    capability_code=r.capability_ref.code,
                    gap_codes=codes,
                    observation="UNOBSERVED_TOOL_GAP",
                    capability="UNAVAILABLE_CAPABILITY",
                    validity="UNASSESSED",
                    clinical_effect="NONE",
                    alternative_paths=sorted({retained[i].candidate.code for i in roots}),
                    blocks_required_path=not bool(roots),
                    scope=dict(
                        scope_type="RUN",
                        run_id=request.lock.run_id,
                        case_code=request.scope.case_code,
                    ),
                    laterality=request.scope.laterality,
                    coordinate_space=r.constraint.coordinate_space,
                    unit=None,
                    provenance=dict(
                        actor="RULE_ENGINE",
                        created_at=request.snapshot_at,
                        generator_version="06B.admission.1.0.0",
                        source_codes=["SOURCE:TOOL:ADMISSION"],
                        input_checksums=[digest(request)],
                    ),
                ),
            )
        )
    return gaps

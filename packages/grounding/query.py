"""Deterministic bounded AND/OR description. No I/O, LLM, execution or graph updates."""

from .compatibility import artifact_compatibility, constraint_compatibility
from .models import digest, seal, validate
from .query_models import (
    METRICS,
    CatalogProducer,
    Component,
    DeferredCandidate,
    Dependency,
    FeasibleCandidate,
    Finding,
    PathSource,
    QueryRequest,
    QueryResult,
    RankingEstimate,
    RejectedCandidate,
    category,
)
from .validation import GroundingError, MappingPaths


def normalized_request(request):
    request = validate(QueryRequest, request)
    data = request.model_dump(mode="json")
    data["artifacts"] = sorted(data["artifacts"], key=lambda item: item["asset_code"])
    for field in ("environment", "patient_data_types", "satisfied_terms"):
        data[field] = sorted(data[field], key=lambda item: digest(item))
    for key, value in data["context"].items():
        if isinstance(value, list):
            data["context"][key] = sorted(value)
    for artifact in data["artifacts"]:
        artifact["scope_terms"] = sorted(artifact["scope_terms"])
    return validate(QueryRequest, data)


def rank_key(candidate, config):
    components = {c.name: c for c in candidate.components}
    key = []
    for name in config.order:
        if name == "information_gain" and not config.information_gain_enabled:
            continue
        component = components[name]
        known = component.state == "known"
        value = component.value if known else 0
        key.extend((not known, -value if name in {"information_gain", "directness"} else value))
    return (
        *key,
        candidate.mapping_ref.model_dump_json(),
        candidate.candidate.code,
        candidate.identity,
    )


def evaluate_query(request, mappings, *, estimates=None) -> QueryResult:
    """Pure evaluation of already admitted current mapping projections.

    Public runtime entry is GroundingQueryService.query: it obtains these projections
    from current Published ports and verifies the persisted lock and patient scope.
    This pure function does not grant execution or persistence authority.
    """
    request = normalized_request(request)
    mappings = tuple(validate(MappingPaths, m) for m in mappings)
    if {m.record.reference() for m in mappings} != set(request.lock.mapping_refs):
        raise GroundingError("locked_mapping_set_mismatch")
    if len({digest(m.safety) for m in mappings}) != 1:
        raise GroundingError("safety_changed_during_query")
    safety_checksum = digest(mappings[0].safety)
    targets = [m for m in mappings if m.record.requirement_ref == request.requirement_ref]
    if not targets:
        raise GroundingError("requirement_mapping_missing")
    catalog = mappings[0].catalog
    if (
        any(m.catalog != catalog for m in mappings)
        or catalog.reference() != request.lock.catalog_ref
    ):
        raise GroundingError("catalog_version_mismatch")
    all_paths = sorted(
        ((m.record.reference(), p) for m in mappings for p in m.paths),
        key=lambda x: (x[0].model_dump_json(), x[1].candidate.code),
    )
    target_paths = [
        (ref, p) for ref, p in all_paths if ref in {m.record.reference() for m in targets}
    ]
    findings, rejected, deferred = {}, {}, {}
    evaluations = 0
    operations = 0

    class SearchLimit(Exception):
        pass

    def tick():
        nonlocal operations
        if operations >= request.config.max_operations:
            raise SearchLimit
        operations += 1

    def finding(
        reason,
        path,
        constraint=None,
        asset=None,
        *,
        source=None,
        input_port=None,
        catalog_producers=(),
    ):
        result = Finding(
            reason=reason,
            category=category(reason),
            path=path,
            scope=request.scope,
            constraint=constraint,
            asset_code=asset,
            catalog_ref=request.lock.catalog_ref,
            source=source,
            input_port=input_port,
            catalog_producers=catalog_producers,
        )
        key = digest(result)
        if key in findings:
            return findings[key]
        if reason == "budget_exhausted" and len(findings) >= request.config.max_findings:
            return next(f for f in findings.values() if f.reason == "budget_exhausted")
        if reason != "budget_exhausted" and len(findings) >= request.config.max_findings - 1:
            raise SearchLimit
        findings[key] = result
        return result

    def available(constraint, path, *, source=None, input_port=None):
        result = []
        # Other types are absent inputs, not misleading type-mismatch errors for every asset.
        for artifact in request.artifacts:
            tick()
            if artifact.artifact_ref.code != constraint.artifact_ref.code:
                continue
            reasons = artifact_compatibility(artifact, constraint, request)
            if not reasons:
                result.append(artifact.asset_code)
            else:
                for reason in reasons:
                    finding(
                        reason,
                        path,
                        constraint,
                        artifact.asset_code,
                        source=source,
                        input_port=input_port,
                    )
        return tuple(sorted(result))

    existing = set()
    try:
        for mapping in targets:
            for constraint in mapping.record.accepted_artifacts:
                if constraint.artifact_ref not in catalog.artifacts:
                    finding("artifact_outside_catalog", ("existing_artifact",), constraint)
                else:
                    existing.update(available(constraint, ("existing_artifact",)))
                if existing:
                    break
            if existing:
                break
    except SearchLimit:
        ref, admission = target_paths[0]
        item = DeferredCandidate(
            mapping_ref=ref,
            candidate_code=admission.candidate.code,
            path=("existing_artifact",),
            reason="budget_exhausted",
        )
        deferred[digest(item)] = item
        finding("budget_exhausted", item.path)

    def components(candidate):
        if candidate.estimate_ref is None:
            values = tuple(
                Component(name=name, state="unknown", value=None, source_ref=None)
                for name in METRICS
            )
        else:
            if estimates is None:
                raise GroundingError("estimate_reader_required")
            estimate = validate(
                RankingEstimate,
                estimates.read_estimate(catalog.reference(), candidate.estimate_ref),
            )
            if (estimate.catalog_ref, estimate.estimate_ref) != (
                catalog.reference(),
                candidate.estimate_ref,
            ):
                raise GroundingError("estimate_binding_mismatch")
            values = estimate.components
        return tuple(
            Component(name=c.name, state="disabled", value=None, source_ref=None)
            if c.name == "information_gain" and not request.config.information_gain_enabled
            else c
            for c in values
        )

    def expand(ref, admission, path, ancestors):
        nonlocal evaluations
        tick()
        candidate = admission.candidate
        source = PathSource(
            mapping_ref=ref,
            candidate=candidate,
            path=path,
            mapping_path=tuple(a[0] for a in ancestors) + (ref,),
        )
        identity = (ref, candidate.code)
        reasons = []
        if (
            len(ancestors) >= request.config.max_depth
            or evaluations >= request.config.max_evaluations
        ):
            reason = (
                "depth_exhausted"
                if len(ancestors) >= request.config.max_depth
                else "budget_exhausted"
            )
            item = DeferredCandidate(
                mapping_ref=ref, candidate_code=candidate.code, path=path, reason=reason
            )
            deferred[digest(item)] = item
            finding(reason, path, candidate.output, source=source)
            return None
        elif identity in ancestors:
            evaluations += 1
            reasons.append(finding("cycle_detected", path, candidate.output, source=source))
        else:
            evaluations += 1
            for reason in (
                *admission.reasons,
                *constraint_compatibility(candidate.output, request),
            ):
                reasons.append(finding(reason, path, candidate.output, source=source))
            environment = next(
                (
                    e
                    for e in request.environment
                    if e.implementation_ref == candidate.implementation_ref
                ),
                None,
            )
            if environment is None or environment.availability == "unknown":
                reasons.append(
                    finding("environment_unknown", path, candidate.output, source=source)
                )
            elif environment.availability == "unavailable":
                reasons.append(
                    finding("environment_unavailable", path, candidate.output, source=source)
                )
            if environment is None or environment.preconditions == "unknown":
                reasons.append(
                    finding("required_condition_unknown", path, candidate.output, source=source)
                )
            elif environment.preconditions == "rejected":
                reasons.append(
                    finding("precondition_rejected", path, candidate.output, source=source)
                )
        dependencies = []
        # A cycle has no useful independent input expansion. Other failures retain all AND gaps.
        if not any(r.reason == "cycle_detected" for r in reasons):
            for input_ in sorted(candidate.inputs, key=lambda i: i.port.port_name):
                tick()
                input_path = (*path, input_.port.port_name)
                artifacts = available(
                    input_.constraint, input_path, source=source, input_port=input_.port
                )
                producers = []
                if input_.required and len(artifacts) < input_.cardinality_min:
                    options = [
                        (r, p) for r, p in all_paths if p.candidate.output == input_.constraint
                    ]
                    for option_index, (producer_ref, producer) in enumerate(options):
                        node = expand(
                            producer_ref,
                            producer,
                            (*input_path, producer_ref.code + "/" + producer.candidate.code),
                            (*ancestors, identity),
                        )
                        if node is not None:
                            producers.append(node)
                        if evaluations >= request.config.max_evaluations:
                            if option_index + 1 < len(options):
                                pending_ref, pending = options[option_index + 1]
                                item = DeferredCandidate(
                                    mapping_ref=pending_ref,
                                    candidate_code=pending.candidate.code,
                                    path=(
                                        *input_path,
                                        pending_ref.code + "/" + pending.candidate.code,
                                    ),
                                    reason="budget_exhausted",
                                )
                                deferred[digest(item)] = item
                                finding("budget_exhausted", item.path, input_.constraint)
                            break
                    if not options:
                        discovered = []
                        if input_.constraint.artifact_ref in request.patient_data_types:
                            reason = "patient_data_missing"
                        else:
                            # Inspect this exact catalog only; discovery does not admit a path.
                            for tool in catalog.tools:
                                for port, output in zip(
                                    tool.output_ports, tool.outputs, strict=True
                                ):
                                    tick()
                                    if output.artifact_ref == input_.constraint.artifact_ref:
                                        discovered.append(
                                            CatalogProducer(
                                                tool_ref=tool.tool_ref,
                                                implementation_ref=tool.implementation_ref,
                                                capability_refs=tuple(
                                                    sorted(tool.capabilities, key=digest)
                                                ),
                                                output_port=port,
                                                output=output,
                                            )
                                        )
                            reason = (
                                "unmapped_in_locked_view" if discovered else "unresolved_producer"
                            )
                        reasons.append(
                            finding(
                                reason,
                                input_path,
                                input_.constraint,
                                source=source,
                                input_port=input_.port,
                                catalog_producers=tuple(sorted(discovered, key=digest)),
                            )
                        )
                if len(artifacts) + int(bool(producers)) < input_.cardinality_min:
                    # Search limits are explicitly represented; never claim exhaustive absence.
                    reasons.append(
                        finding(
                            "required_input_missing",
                            input_path,
                            input_.constraint,
                            source=source,
                            input_port=input_.port,
                        )
                    )
                else:
                    dependencies.append(
                        Dependency(
                            port_name=input_.port.port_name,
                            required=input_.required,
                            cardinality_min=input_.cardinality_min,
                            cardinality_max=input_.cardinality_max,
                            constraint=input_.constraint,
                            artifact_options=artifacts,
                            producer_options=tuple(producers),
                            optional_missing=not input_.required
                            and not artifacts
                            and not producers,
                        )
                    )
        if reasons:
            result = RejectedCandidate(
                source=source,
                mapping_ref=ref,
                capability_ref=candidate.capability_ref,
                candidate_code=candidate.code,
                path=path,
                reasons=tuple({digest(r): r for r in reasons}.values()),
            )
            rejected[digest(dict(mapping_ref=ref, candidate_code=candidate.code, path=path))] = (
                result
            )
            return None
        values = dict(
            mapping_ref=ref,
            candidate=candidate,
            dependencies=tuple(dependencies),
            components=components(candidate),
        )
        return FeasibleCandidate(identity=digest(values), **values)

    candidates = []
    if not existing and not deferred:
        for ref, admission in target_paths:
            path = (ref.code + "/" + admission.candidate.code,)
            try:
                result = expand(ref, admission, path, ())
                if result is not None:
                    candidates.append(result)
            except SearchLimit:
                item = DeferredCandidate(
                    mapping_ref=ref,
                    candidate_code=admission.candidate.code,
                    path=path,
                    reason="budget_exhausted",
                )
                deferred[digest(item)] = item
                finding("budget_exhausted", path)
                break
            if evaluations >= request.config.max_evaluations:
                # The next root is explicitly deferred by expand, then stop traversal.
                position = target_paths.index((ref, admission))
                if position + 1 < len(target_paths):
                    next_ref, next_admission = target_paths[position + 1]
                    item = DeferredCandidate(
                        mapping_ref=next_ref,
                        candidate_code=next_admission.candidate.code,
                        path=(next_ref.code + "/" + next_admission.candidate.code,),
                        reason="budget_exhausted",
                    )
                    deferred[digest(item)] = item
                    finding("budget_exhausted", item.path)
                break
    candidates.sort(key=lambda c: rank_key(c, request.config))
    enumeration = (
        "existing_artifact_short_circuit"
        if existing
        else "budget_exhausted"
        if any(d.reason == "budget_exhausted" for d in deferred.values())
        else "depth_exhausted"
        if deferred
        else "complete"
    )
    gaps = tuple(findings[key] for key in sorted(findings))
    from .attribution import derive_attribution

    product_demands, suggestions = derive_attribution(
        request, tuple(candidates), tuple(rejected.values()), gaps
    )
    result = seal(
        QueryResult,
        request=request,
        binding_checksum=digest(request),
        safety_checksum=safety_checksum,
        status="truncated"
        if deferred
        else "existing_artifact"
        if existing
        else "candidates"
        if candidates
        else "gaps",
        existing_artifacts=tuple(sorted(existing)),
        candidates=tuple(candidates),
        top_k=tuple(c.identity for c in candidates[: request.config.top_k]),
        rejected=tuple(rejected[k] for k in sorted(rejected)),
        deferred=tuple(deferred[k] for k in sorted(deferred)),
        gaps=gaps,
        suggestions=suggestions,
        product_demands=product_demands,
        evaluations=evaluations,
        operations=operations,
        enumeration=enumeration,
        expert_review_recommended=not bool(existing or candidates),
    )
    return validate(QueryResult, result)

"""05B causal product attribution shared by construction and result validation.

Only failed required input edges propagate producer failures. Diagnostic failures
on satisfied OR inputs never become extension targets. No I/O or task creation.
"""

from .models import digest, validate
from .query_models import ExtensionSuggestion, InputImpact, ProductDemand

EXTENSIBLE = {
    "capability_unregistered": "NO_TOOL",
    "implementation_missing": "NO_IMPLEMENTATION",
    "validation_insufficient": "LOW_RELIABILITY",
    "input_incompatible": "INPUT_MISMATCH",
    "environment_unavailable": "EXECUTION_RESOURCE_UNAVAILABLE",
}

# Existing hard-filter reasons, not a new admission policy. Catalog revocation or
# inactive status applies to the exact implementation in this catalog; explicit
# use/precondition refusal applies only to its candidate occurrence. Unknown
# conditions still fail execution admission, but are not explicit revocations.
IMPLEMENTATION_VETOES = frozenset({"tool_safety_revoked", "tool_not_active"})
SOURCE_VETOES = frozenset({"intended_use_rejected", "precondition_rejected"})


def ordered(values):
    unique = {digest(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def derive_attribution(request, candidates, rejected, gaps):
    """Derive targets from own failures, never from a blocked consumer's input.

    Group product demands by full constraint + resolution + catalog producer set.
    Group extensions by exact capability/tool/implementation/output. Both retain
    every affected occurrence/port; the request binding separates cases and runs.
    """
    demands, extensions = {}, {}
    own_findings = {}
    for finding in gaps:
        if finding.source is not None:
            own_findings.setdefault(finding.source, []).append(finding)
    vetoed_implementations = {
        (f.catalog_ref, f.source.candidate.implementation_ref)
        for f in gaps
        if f.source is not None and f.input_port is None and f.reason in IMPLEMENTATION_VETOES
    }

    def visit(rejection, upstream):
        source = rejection.source
        vetoed = (
            request.lock.catalog_ref,
            source.candidate.implementation_ref,
        ) in vetoed_implementations or any(
            f.input_port is None and f.reason in SOURCE_VETOES for f in own_findings.get(source, ())
        )
        direct = tuple(
            f
            for f in rejection.reasons
            if not vetoed
            and f.source == source
            and f.input_port is None
            and f.category in EXTENSIBLE
        )
        if direct:
            candidate = source.candidate
            target = (
                candidate.capability_ref,
                candidate.tool_ref,
                candidate.implementation_ref,
                candidate.output,
            )
            entry = extensions.setdefault(
                target, dict(findings=set(), sources=set(), impacts=set())
            )
            entry["findings"].update(digest(f) for f in direct)
            entry["sources"].add(source)
            entry["impacts"].update(upstream)
        for missing in rejection.reasons:
            if missing.reason != "required_input_missing":
                continue
            impact = InputImpact(source=source, input_port=missing.input_port)
            # Includes incompatible existing assets only on this unsatisfied port.
            causes = tuple(
                f for f in own_findings.get(source, ()) if f.input_port == missing.input_port
            )
            children = tuple(
                r
                for r in rejected
                if r.path[:-1] == missing.path and r.source.mapping_path[:-1] == source.mapping_path
            )
            for child in children:
                visit(child, (*upstream, impact))
            discoveries = ordered(p for f in causes for p in f.catalog_producers)
            reasons = {f.reason for f in causes}
            resolution = next(
                (
                    r
                    for r in (
                        "patient_data_missing",
                        "unmapped_in_locked_view",
                        "unresolved_producer",
                    )
                    if r in reasons
                ),
                None,
            )
            if resolution is None:
                resolution = "mapped_producer_blocked" if children else "input_incompatible"
                if any(
                    f.category == "search_incomplete"
                    and f.path[: len(missing.path)] == missing.path
                    for f in gaps
                ):
                    resolution = "search_incomplete"
            key = (missing.constraint, resolution, discoveries)
            entry = demands.setdefault(key, dict(impacts=set(), findings=set(), producers=set()))
            entry["impacts"].add(impact)
            entry["producers"].update(c.source for c in children)
            entry["findings"].update(digest(f) for f in causes)
            entry["findings"].update(digest(f) for c in children for f in c.reasons)

    for rejection in rejected:
        if len(rejection.path) == 1:
            visit(rejection, ())
    product_demands = []
    for (constraint, resolution, producers), entry in demands.items():
        product_demands.append(
            ProductDemand(
                binding_checksum=digest(request),
                requirement_ref=request.requirement_ref,
                catalog_ref=request.lock.catalog_ref,
                policy_ref=request.lock.policy.reference(),
                constraint=constraint,
                resolution=resolution,
                catalog_producers=producers,
                producer_paths=ordered(entry["producers"]),
                affected_inputs=ordered(entry["impacts"]),
                finding_checksums=tuple(sorted(entry["findings"])),
            )
        )
    suggestions = []
    by_checksum = {digest(f): f for f in gaps}
    for (capability, tool, implementation, constraint), entry in extensions.items():
        checksums = tuple(sorted(entry["findings"]))
        sources, impacts = ordered(entry["sources"]), ordered(entry["impacts"])
        identity = digest(
            (
                digest(request),
                capability,
                tool,
                implementation,
                constraint,
                checksums,
                sources,
                impacts,
            )
        )
        gap = dict(
            schema_version="1.0",
            contract_type="CapabilityGap",
            gap_id="REQUIREMENT:OID_" + identity.split(":")[1][:32].upper(),
            origin_run_id=request.lock.run_id,
            target_node_code=request.target_node_code,
            capability_code=capability.code,
            gap_codes=sorted({EXTENSIBLE[by_checksum[k].category] for k in checksums}),
            observation="UNOBSERVED_TOOL_GAP",
            capability="UNAVAILABLE_CAPABILITY",
            validity="UNASSESSED",
            clinical_effect="NONE",
            alternative_paths=sorted({c.candidate.code for c in candidates}),
            blocks_required_path=not bool(candidates),
            scope=dict(
                scope_type="RUN", run_id=request.lock.run_id, case_code=request.scope.case_code
            ),
            laterality=request.scope.laterality,
            coordinate_space=constraint.coordinate_space,
            unit=None,
            provenance=dict(
                actor="RULE_ENGINE",
                created_at=request.snapshot_at,
                generator_version="05B.attribution.2.0.0",
                source_codes=["SOURCE:GROUNDING:QUERY"],
                input_checksums=[digest(request)],
            ),
        )
        suggestions.append(
            validate(
                ExtensionSuggestion,
                dict(
                    capability_gap=gap,
                    target_capability_ref=capability,
                    target_tool_ref=tool,
                    target_implementation_ref=implementation,
                    constraint=constraint,
                    source_paths=sources,
                    affected_inputs=impacts,
                    finding_checksums=checksums,
                    requirement_ref=request.requirement_ref,
                    catalog_ref=request.lock.catalog_ref,
                    policy_ref=request.lock.policy.reference(),
                ),
            )
        )
    return ordered(product_demands), ordered(suggestions)

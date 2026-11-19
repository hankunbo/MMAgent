"""Deterministic count coverage and explicitly bound port roles, no medical rules."""

from packages.grounding.models import digest

from .models_v11 import ContractMatch, PortBinding, Provenance


def match_contract(record, required):
    def answer(status, *reasons, bindings=(), omitted=()):
        return ContractMatch(
            status=status,
            requirement_contract_checksum=digest(required),
            reasons=reasons,
            bindings=bindings,
            omitted_optional=omitted,
        )

    if not isinstance(record, Provenance):
        return answer("missing", "legacy_input_contract")
    offered = record.semantics
    if offered is None or not offered.complete:
        return answer("missing", "input_contract_fields_missing")
    if any(
        getattr(offered, name) != getattr(required, name)
        for name in ("capability", "output", "task_type", "intended_use")
    ):
        return answer("incompatible", "task_contract_mismatch")
    mapping = next(
        (m for m in record.input_mappings if m.requirement_contract_checksum == digest(required)),
        None,
    )
    src = {digest(i.port): i for i in offered.inputs}
    dst = {digest(i.port): i for i in required.inputs}
    if mapping is None:
        if set(src) != set(dst):
            return answer("missing", "input_role_mapping_required")
        bindings = tuple(PortBinding(source=i.port, target=i.port) for i in required.inputs)
        omitted = ()
    else:
        bindings, omitted = mapping.bindings, mapping.omitted_optional
        mapped_src = {digest(b.source) for b in bindings}
        mapped_dst = {digest(b.target) for b in bindings}
        omitted_keys = {digest(p) for p in omitted}
        if not mapped_src <= set(src) or not mapped_dst <= set(dst) or not omitted_keys <= set(src):
            return answer("incompatible", "input_mapping_endpoint_invalid")
        if mapped_src | omitted_keys != set(src) or mapped_dst != set(dst):
            return answer("missing", "input_role_mapping_incomplete")
        if any(src[k].required or src[k].cardinality_min != 0 for k in omitted_keys):
            return answer("incompatible", "required_input_cannot_be_omitted")
    for b in bindings:
        c, q = src[digest(b.source)], dst[digest(b.target)]
        if c.constraint != q.constraint:
            return answer("incompatible", "input_constraint_mismatch", bindings=bindings)
        # Every count allowed by the owner must be accepted by the candidate.
        if c.cardinality_min > q.cardinality_min or c.cardinality_max < q.cardinality_max:
            return answer("incompatible", "input_cardinality_not_covered", bindings=bindings)
    if offered == required and all(b.source == b.target for b in bindings) and not omitted:
        return answer("exact", "input_contract_exact", bindings=bindings)
    reasons = ["input_count_interval_covers_requirement"]
    if mapping is not None:
        reasons.append("explicit_input_role_mapping")
    if omitted:
        reasons.append("explicit_optional_inputs_omitted")
    return answer("compatible", *reasons, bindings=bindings, omitted=omitted)


def match(record, query):
    rank = {"exact": 0, "compatible": 1, "missing": 2, "incompatible": 3}
    return min(
        (match_contract(record, c) for c in query.contracts),
        key=lambda m: (rank[m.status], m.requirement_contract_checksum),
    )

"""Deterministic query extraction, provenance screening and engineering ranking."""

from packages.grounding.models import Candidate, digest, validate
from packages.tool_gap_queue.models import Application

from .matching import match
from .models_v11 import CandidateObservation, DiscoveryQuery, Semantics


def contract(candidate):
    candidate = validate(Candidate, candidate).model_dump(mode="json")
    return validate(
        Semantics,
        dict(
            capability=candidate["capability_ref"],
            task_type=candidate["capability_ref"]["code"],
            inputs=tuple(candidate["inputs"]),
            output=candidate["output"],
        ),
    )


def query(application, policy):
    a = validate(Application, application)
    base = a
    if a.owner in {"08", "09"}:
        base = validate(Application, a.material["resolved"])
    allowed = {digest(f.capability) for f in base.findings if f.queueable}
    candidates = []
    if base.owner == "06_research":
        candidates.append(base.material["request"]["candidate"])
    elif base.owner in {"05", "06_formal"}:
        result = base.material["result"]
        for s in result["suggestions"]:
            if digest(s["target_capability_ref"]) in allowed:
                candidates.extend(p["candidate"] for p in s["source_paths"])
        if base.owner == "06_formal":
            candidates.extend(r["source"]["candidate"] for r in result["rejected"])
            candidates.extend(r["candidate"] for r in result["candidates"])
    contracts = {}
    for c in candidates:
        if digest(c["capability_ref"]) in allowed:
            item = contract(c)
            contracts[digest(item)] = item
    if not contracts:
        raise ValueError("discovery_resolved_contract_required")
    return DiscoveryQuery(
        application_checksum=digest(a),
        requirement=a.requirement,
        purpose=a.purpose,
        contracts=tuple(contracts[k] for k in sorted(contracts)),
        max_memory_mb=policy.max_memory_mb,
    )


def absent(value):
    return (
        value is None
        or value == ()
        or (
            isinstance(value, str)
            and value.strip().lower() in {"", "unknown", "unavailable", "not_obtained"}
        )
    )


def inspect(record, source, q, policy):
    p = record
    missing = [
        name
        for name in (
            "identity_ref",
            "repository",
            "repository_owner",
            "model_id",
            "revision",
            "revision_kind",
            "code_license",
            "code_checksum",
            "model_card",
            "framework",
            "inference_entry",
            "semantics",
            "input_shape",
            "output_shape",
            "preprocessing",
            "postprocessing",
            "memory_mb",
            "requires_gpu",
            "training_data",
            "evaluation_data",
            "applicability",
            "limitations",
        )
        if absent(getattr(p, name))
    ]
    if p.weights_required:
        missing += [
            name for name in ("weights_license", "weights_checksum") if absent(getattr(p, name))
        ]
    reasons = []
    proof = source.identity(p.identity_ref)
    if proof is None:
        missing.append("registered_identity_evidence")
        if p.claimed_official:
            reasons.append("official_identity_unverified")
    else:
        if p.source_url not in proof.source_urls:
            reasons.append("source_url_not_registered")
        for field in (
            "repository",
            "repository_owner",
            "model_id",
            "revision",
            "code_checksum",
            "weights_checksum",
        ):
            if getattr(p, field) is not None and getattr(p, field) != getattr(proof, field):
                reasons.append(field + "_changed")
        if p.claimed_official and not proof.official:
            reasons.append("official_identity_unverified")
    if p.source_kind in {"official_package", "official_model", "paper_code"}:
        if proof is None or not proof.official:
            reasons.append("official_identity_required")
    if p.source_kind == "paper_code" and not p.paper_doi:
        missing.append("paper_doi")
    if p.source_kind == "lead":
        missing.append("controlled_source_required")
    if p.revision_kind == "branch":
        reasons.append("mutable_revision")
    input_match = match(p, q)
    if input_match.status == "missing":
        missing.extend(input_match.reasons)
    elif input_match.status == "incompatible":
        reasons.extend(input_match.reasons)
    if p.uploads_patient_data:
        reasons.append("patient_upload_required")
    if p.arbitrary_remote_code:
        reasons.append("arbitrary_remote_code_required")
    if p.requires_gpu or (p.memory_mb is not None and p.memory_mb > policy.max_memory_mb):
        reasons.append("resource_limit")
    missing.extend(("frozen_local_evaluation_dataset", "approved_threshold_policy"))
    # These last two are later-stage prerequisites, not discovery rejection or fake Gate failures.
    blocking_missing = set(missing) - {
        "frozen_local_evaluation_dataset",
        "approved_threshold_policy",
    }
    disposition = "quarantined" if reasons else "incomplete" if blocking_missing else "discovered"
    dimensions = dict(
        source_trust=1.0 if proof and proof.official else 0.5 if proof else 0.0,
        contract_completeness=1.0 if not blocking_missing else 0.0,
        task_match={"exact": 1.0, "compatible": 0.75}.get(input_match.status, 0.0),
        reproducibility=1.0
        if proof and p.revision_kind in {"commit", "immutable_version"} and p.code_checksum
        else 0.0,
        dependency_simplicity=1.0 / (1 + len(p.dependencies)),
        public_evaluation=None,  # Source claims are retained as text, not local performance scores.
        resource_cost=None
        if p.memory_mb is None
        else 1.0 / (1 + p.memory_mb / policy.max_memory_mb),
    )
    score = sum((v or 0.0) * w for v, w in zip(dimensions.values(), policy.weights, strict=True))
    # Complete registered identities permit mirrors to share one identity; incomplete ones do not.
    identity = (
        digest(
            dict(
                repository=p.repository,
                owner=p.repository_owner,
                model=p.model_id,
                revision=p.revision,
                code=p.code_checksum,
                weights=p.weights_checksum,
                semantics=p.semantics,
                input_mappings=p.input_mappings,
                processing=(p.preprocessing, p.postprocessing),
                environment=(p.framework, p.dependencies, p.inference_entry),
                resources=(p.memory_mb, p.requires_gpu),
                weights_required=p.weights_required,
                shapes=(p.input_shape, p.output_shape),
            )
        )
        if proof and not reasons and not blocking_missing
        else digest(p)
    )
    priority = {
        "local_contract": 0,
        "official_package": 1,
        "official_model": 1,
        "paper_code": 2,
        "lead": 3,
    }[p.source_kind]
    return CandidateObservation(
        identity=identity,
        input_match=input_match,
        records=(p,),
        identity_evidence=() if proof is None else (proof,),
        disposition=disposition,
        missing=tuple(sorted(set(missing))),
        rejection_codes=tuple(sorted(set(reasons))),
        dimensions=dimensions,
        score=score,
        source_priority=priority,
    )


def merge(observations):
    result = {}
    for c in observations:
        old = result.get(c.identity)
        if old is None:
            result[c.identity] = c
        else:
            records = {digest(r): r for r in old.records + c.records}
            best = min((old, c), key=lambda x: (x.source_priority, -x.score))
            proofs = {digest(p): p for p in old.identity_evidence + c.identity_evidence}
            result[c.identity] = best.model_copy(
                update={
                    "records": tuple(records.values()),
                    "identity_evidence": tuple(proofs.values()),
                }
            )
    return tuple(sorted(result.values(), key=lambda c: (c.source_priority, -c.score, c.identity)))

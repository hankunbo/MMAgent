"""Versioned citation-fidelity proof bound to unchanged model candidate and source.

The same proof is rebuilt at submit and accept. It is part of the immutable
submission Artifact, not an independent write or caller authorization token.
"""

from packages.grounding.models import digest
from packages.knowledge_claims.research_candidates import source_quote_alignment
from packages.knowledge_claims.research_quote import LAYOUT_POLICY, quote_policy
from packages.llm_provider.models import SourceProposal


def source_quote_proof(submission, material, *, policy):
    if policy != LAYOUT_POLICY or not isinstance(submission.candidate, SourceProposal):
        raise ValueError("research_quote_proof_policy")
    alignment = source_quote_alignment(submission.candidate, material, quote_policy=policy)
    source = next(
        s for s in material.sources if s.reference.code == submission.candidate.source_code
    )
    return dict(
        schema_version="research-source-quote-validation/1.0.0",
        policy=policy,
        request_id=submission.request_id,
        submission_checksum=digest(submission),
        candidate_checksum=digest(submission.candidate),
        scope_checksum=digest(material.scope),
        source_reference=source.reference.model_dump(mode="json"),
        alignment=alignment,
    )


def installed_source_proof(submission, material, owner):
    if not isinstance(submission.candidate, SourceProposal):
        return None
    policy = quote_policy(owner)
    if policy != LAYOUT_POLICY:
        return None
    return source_quote_proof(submission, material, policy=policy)

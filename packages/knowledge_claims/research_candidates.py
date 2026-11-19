"""04 source/requirement gate. A research citation is never a Published release."""

from packages.grounding.models import validate
from packages.llm_provider.models import Material, SourceProposal

from .research_quote import (
    LAYOUT_POLICY,
    LITERAL_POLICY,
    align_research_quote,
    check_quote_policy,
)


def source_quote_alignment(candidate, material, *, quote_policy=LITERAL_POLICY):
    check_quote_policy(quote_policy)
    candidate = validate(SourceProposal, candidate)
    material = validate(Material, material)
    sources = {s.reference.code: s for s in material.sources}
    source = sources.get(candidate.source_code)
    proof = (
        align_research_quote(
            candidate.quote, source.text, allow_layout=quote_policy == LAYOUT_POLICY
        )
        if source is not None
        else None
    )
    if proof is None:
        raise ValueError("research_source_or_target_mismatch")
    validate_source_target(candidate.target, material)
    return proof


def validate_research_source(candidate, material, *, quote_policy=LITERAL_POLICY):
    candidate = validate(SourceProposal, candidate)
    source_quote_alignment(candidate, material, quote_policy=quote_policy)
    return candidate


def validate_source_target(target, material):
    if target != material.goal_code:
        raise ValueError("research_source_or_target_mismatch")


def submit_research_candidate(service, submission, *, key):
    """Uses the shared domain submission receipt; this does not publish Knowledge."""
    if not isinstance(submission.candidate, SourceProposal):
        raise ValueError("source_candidate_required")
    return service.submit(submission, key=key)

"""Formal v1 is unchanged: no conversion from a research proposal or source flag."""

from packages.contracts import ExpertAssessment
from packages.rule_engine import validate_expert_action


def validate_formal_assessment(
    value, *, legal_actions, published_owner, expected_scope, expected_proposal, purpose
):
    raw = value.model_dump(mode="python") if isinstance(value, ExpertAssessment) else value
    assessment = ExpertAssessment.model_validate(raw)
    if assessment.scope != expected_scope or assessment.proposal_code != expected_proposal:
        raise ValueError("formal_expert_scope")
    # Installed original 04 owner must check exact release membership, withdrawal and use.
    for reference in assessment.knowledge_refs:
        published_owner.validate_reference(reference, purpose=purpose, scope=expected_scope)
    return assessment, validate_expert_action(assessment, legal_actions)


class PublishedAssessmentSources:
    """Original 04 exact Published reads plus release membership and installed use policy.

    reference_kinds and release are installed configuration, never taken from the
    Expert output. verify_use is the original application's applicability/use gate.
    """

    def __init__(self, *, repository, release, reference_kinds, applicability, verify_use):
        self.repository, self.release = repository, release
        self.kinds, self.applicability, self.verify_use = reference_kinds, applicability, verify_use

    def validate_reference(self, reference, *, purpose, scope):
        from db.knowledge.models import KnowledgeReference

        kind = self.kinds.get(reference.source_ref.code)
        if kind is None:
            raise ValueError("formal_source_not_installed")
        exact = KnowledgeReference.from_versioned_reference(kind, reference.source_ref)
        source = self.repository.get_published(exact, applicability=self.applicability)
        release = self.repository.get_published(self.release, applicability=self.applicability)
        if source.to_published_reference() != reference or exact not in release.record.member_refs:
            raise ValueError("formal_published_membership")
        self.verify_use(source, release, purpose=purpose, scope=scope)

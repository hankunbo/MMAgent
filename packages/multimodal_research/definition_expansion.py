"""One bounded append-only requirement revision inside the original campaign.

No replacement runner/database/scoring engine. Sources and requests are supplied
by the original Knowledge and Provider owners; durable facts use RuntimeDocuments.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.evidence_reuse.definition_acceptance import (
    CONFIRMED_RECEIPTS,
    ResearchDefinitionOwner,
    requirement_rows,
)
from packages.evidence_reuse.definition_sources import (
    BalancedPassagePlan,
    BoundedMaterialPassagePlan,
    DiscoursePassagePlan,
    ExplanatoryPassagePlan,
    IntentPassagePlan,
    PassagePlan,
    WholeBlockMaterialPassagePlan,
)
from packages.evidence_reuse.definitions import (
    ConfirmedInterpretationProposal,
    DefinitionProposal,
    EvidenceDefinitionProposal,
    OwnerBoundDefinitionProposal,
    QuotedDefinitionProposal,
    ScopedDefinitionProposal,
)
from packages.grounding.models import Checksum, Value, canonical, digest, validate
from packages.llm_provider.models import Source, TaskDefinition
from packages.llm_provider.provider import ProviderError
from packages.planner_control.models import Reference


class ExpansionPolicy(Value):
    schema_version: Literal["campaign-definition-expansion/1.0.0"] = (
        "campaign-definition-expansion/1.0.0"
    )
    index_keys: tuple[str, ...] = Field(max_length=8)
    public_queries: tuple[str, ...] = Field(min_length=1, max_length=3)
    public_topics: dict[str, str] = Field(default_factory=dict, max_length=3)
    query_requirements: dict[str, tuple[str, ...]] = Field(default_factory=dict, max_length=3)
    max_candidates: int = Field(default=3, ge=1, le=5)
    max_definitions: int = Field(default=3, ge=1, le=3)
    max_rounds: Literal[1] = 1
    max_operations: int = Field(default=8, ge=1, le=12)
    reserve_review_requests: int = Field(default=12, ge=1, le=30)
    reserve_close_seconds: int = Field(default=300, ge=60, le=900)
    direction_policy: Literal["direction-assessment/1.0.0"] = "direction-assessment/1.0.0"

    @model_validator(mode="after")
    def public_scope(self):
        from packages.knowledge_search.broad import TOPICS
        from packages.knowledge_search.privacy import safe_text

        if not set(self.query_requirements) <= set(self.public_queries):
            raise ValueError("definition_query_gap_scope")
        if not set(self.public_topics) <= set(self.public_queries) or not set(
            self.public_topics.values()
        ) <= set(TOPICS):
            raise ValueError("definition_public_topic_scope")
        if any(len(q) > 240 or not safe_text(q) for q in self.public_queries):
            raise ValueError("definition_public_question_bound")
        return self


class TargetExpansionPolicy(ExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.1.0"] = (
        "campaign-definition-expansion/1.1.0"
    )
    require_target_relation: Literal[True] = True


class ResilientExpansionPolicy(TargetExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.2.0"] = (
        "campaign-definition-expansion/1.2.0"
    )
    search_timeout_ms: int = Field(default=120000, ge=30000, le=120000)
    search_failure_policy: Literal["technical_backlog_no_retry"] = "technical_backlog_no_retry"


class TargetedExpansionPolicy(ResilientExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.3.0"] = (
        "campaign-definition-expansion/1.3.0"
    )
    public_topic_sets: dict[str, tuple[str, ...]] = Field(min_length=1, max_length=2)
    max_operations: int = Field(default=24, ge=1, le=24)
    max_search_requests: int = Field(default=2, ge=1, le=2)
    max_fetch_requests: int = Field(default=2, ge=1, le=2)
    max_new_observations: Literal[1] = 1
    max_definitions: Literal[1] = 1
    material_policy: Literal["xml-scripts-whole-text/1.0.0"] = "xml-scripts-whole-text/1.0.0"

    @model_validator(mode="after")
    def exact_topics(self):
        from packages.knowledge_search.broad import TOPICS

        if (
            self.public_topics
            or set(self.public_topic_sets) != set(self.public_queries)
            or len(set(self.public_queries)) != len(self.public_queries)
        ):
            raise ValueError("definition_public_topic_sets_binding")
        if any(
            not 1 <= len(v) <= 3 or len(set(v)) != len(v) or not set(v) <= set(TOPICS)
            for v in self.public_topic_sets.values()
        ):
            raise ValueError("definition_public_topic_sets_scope")
        return self


class ContextualExpansionPolicy(TargetedExpansionPolicy):
    """Public search vocabulary remains separate from local paragraph recall."""

    schema_version: Literal["campaign-definition-expansion/1.4.0"] = (
        "campaign-definition-expansion/1.4.0"
    )
    passage_plans: dict[str, PassagePlan] = Field(min_length=1, max_length=2)
    max_candidates: int = Field(default=5, ge=1, le=5)
    prompt_version: Literal["definition-draft-2.1.0"] = "definition-draft-2.1.0"

    @model_validator(mode="after")
    def passage_scope(self):
        if set(self.passage_plans) != set(self.public_queries):
            raise ValueError("definition_passage_plan_binding")
        return self


class LocatedExpansionPolicy(ContextualExpansionPolicy):
    """Source-local coordinates and method input shapes, never patient values."""

    schema_version: Literal["campaign-definition-expansion/1.5.0"] = (
        "campaign-definition-expansion/1.5.0"
    )
    prompt_version: Literal["definition-draft-2.2.0"] = "definition-draft-2.2.0"


class OwnerLocatedExpansionPolicy(LocatedExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.6.0"] = (
        "campaign-definition-expansion/1.6.0"
    )
    prompt_version: Literal["definition-draft-2.3.0"] = "definition-draft-2.3.0"


class PublicTargetExpansionPolicy(OwnerLocatedExpansionPolicy):
    """Explicit public proposition binding; old policies keep their original queries."""

    schema_version: Literal["campaign-definition-expansion/1.7.0"] = (
        "campaign-definition-expansion/1.7.0"
    )
    public_target: str

    @model_validator(mode="after")
    def supported_target(self):
        from packages.knowledge_search.targeted import TARGETS

        if self.public_target not in TARGETS:
            raise ValueError("definition_public_target_scope")
        return self


class JatsExpansionPolicy(PublicTargetExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.8.0"] = (
        "campaign-definition-expansion/1.8.0"
    )
    material_policy: Literal["xml-jats-whole-text/1.0.0"] = "xml-jats-whole-text/1.0.0"


class DependencyExpansionPolicy(JatsExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.9.0"] = (
        "campaign-definition-expansion/1.9.0"
    )
    prompt_version: Literal["definition-draft-3.0.0"] = "definition-draft-3.0.0"


class CapacityExpansionPolicy(DependencyExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.10.0"] = (
        "campaign-definition-expansion/1.10.0"
    )
    source_output_tokens: Literal[8192] = 8192


class EvidenceExpansionPolicy(CapacityExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.11.0"] = (
        "campaign-definition-expansion/1.11.0"
    )
    prompt_version: Literal["definition-draft-4.0.0"] = "definition-draft-4.0.0"


class CitationExpansionPolicy(EvidenceExpansionPolicy):
    """Opt in to citation-preserving automatic indexing, without changing old plans."""

    schema_version: Literal["campaign-definition-expansion/1.12.0"] = (
        "campaign-definition-expansion/1.12.0"
    )
    material_policy: Literal["xml-jats-citations-whole-text/1.0.0"] = (
        "xml-jats-citations-whole-text/1.0.0"
    )


class IntentExpansionPolicy(CitationExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.13.0"] = (
        "campaign-definition-expansion/1.13.0"
    )
    passage_plans: dict[str, IntentPassagePlan] = Field(min_length=1, max_length=2)


class RequirementScopeExpansionPolicy(IntentExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.14.0"] = (
        "campaign-definition-expansion/1.14.0"
    )
    review_source_scope: Literal["requirements-and-items/1.0.0"] = "requirements-and-items/1.0.0"


class GapDiscoveryExpansionPolicy(RequirementScopeExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.15.0"] = (
        "campaign-definition-expansion/1.15.0"
    )
    unresolved_relation_retrieval: Literal["local-and-public-for-owner-gap/1.0.0"] = (
        "local-and-public-for-owner-gap/1.0.0"
    )


class UnseenFirstExpansionPolicy(GapDiscoveryExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.16.0"] = (
        "campaign-definition-expansion/1.16.0"
    )
    unresolved_relation_retrieval: Literal["unseen-first-for-owner-gap/1.0.0"] = (
        "unseen-first-for-owner-gap/1.0.0"
    )


class WindowedExpansionPolicy(UnseenFirstExpansionPolicy):
    # Extra immutable local plan/index receipts only; HTTP/LLM budgets unchanged.
    max_operations: int = Field(default=36, ge=1, le=36)
    schema_version: Literal["campaign-definition-expansion/1.17.0"] = (
        "campaign-definition-expansion/1.17.0"
    )
    material_policy: Literal["xml-jats-citations-windowed/1.0.0"] = (
        "xml-jats-citations-windowed/1.0.0"
    )


class CompletePoolExpansionPolicy(WindowedExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.18.0"] = (
        "campaign-definition-expansion/1.18.0"
    )
    passage_plans: dict[str, DiscoursePassagePlan] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def shared_pool_quotas(self):
        if (
            len(
                {
                    (p.max_per_document, p.context_radius, p.max_characters)
                    for p in self.passage_plans.values()
                }
            )
            != 1
        ):
            raise ValueError("definition_pool_quota_binding")
        return self


class RequirementContextExpansionPolicy(CompletePoolExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.19.0"] = (
        "campaign-definition-expansion/1.19.0"
    )
    prompt_version: Literal["definition-draft-4.1.0"] = "definition-draft-4.1.0"


class BalancedSourceExpansionPolicy(RequirementContextExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.20.0"] = (
        "campaign-definition-expansion/1.20.0"
    )
    passage_plans: dict[str, BalancedPassagePlan] = Field(min_length=1, max_length=2)


class ConfirmedInterpretationExpansionPolicy(BalancedSourceExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.21.0"] = (
        "campaign-definition-expansion/1.21.0"
    )
    prompt_version: Literal["definition-draft-5.0.0"] = "definition-draft-5.0.0"


class OwnerDependencyExpansionPolicy(ConfirmedInterpretationExpansionPolicy):
    schema_version: Literal["campaign-definition-expansion/1.22.0"] = (
        "campaign-definition-expansion/1.22.0"
    )
    prompt_version: Literal["definition-draft-6.0.0"] = "definition-draft-6.0.0"


class AdaptiveGapExpansionPolicy(OwnerDependencyExpansionPolicy):
    """Finite, distinct public directions, sharing the original campaign limits."""

    schema_version: Literal["campaign-definition-expansion/1.23.0"] = (
        "campaign-definition-expansion/1.23.0"
    )
    max_rounds: int = Field(default=2, ge=1, le=2)
    reserve_review_seconds: int = Field(default=2100, ge=60, le=7200)

    @model_validator(mode="after")
    def distinct_directions(self):
        queries = self.public_queries
        if len(queries) != len(set(q.strip().casefold() for q in queries)):
            raise ValueError("definition_duplicate_public_direction")
        if len(queries) > self.max_rounds:
            raise ValueError("definition_direction_round_limit")
        signatures = [
            digest(
                (self.public_topic_sets[q], self.passage_plans[q], self.query_requirements.get(q))
            )
            for q in queries
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("definition_equivalent_public_direction")
        return self


class SingleDocumentGapExpansionPolicy(AdaptiveGapExpansionPolicy):
    """New prompt identity; old prompt and policy bytes remain reproducible."""

    schema_version: Literal["campaign-definition-expansion/1.24.0"] = (
        "campaign-definition-expansion/1.24.0"
    )
    prompt_version: Literal["definition-draft-6.1.0"] = "definition-draft-6.1.0"


class ExplanatoryGapExpansionPolicy(SingleDocumentGapExpansionPolicy):
    """Explicit body-anchor retrieval; no new medical interpretation formula."""

    schema_version: Literal["campaign-definition-expansion/1.25.0"] = (
        "campaign-definition-expansion/1.25.0"
    )
    passage_plans: dict[str, ExplanatoryPassagePlan] = Field(min_length=1, max_length=2)


class RawRevisionExpansionPolicy(ExplanatoryGapExpansionPolicy):
    """Expose existing owner-checked same-method raw-input revisions in the wire."""

    schema_version: Literal["campaign-definition-expansion/1.26.0"] = (
        "campaign-definition-expansion/1.26.0"
    )
    prompt_version: Literal["definition-draft-6.2.0"] = "definition-draft-6.2.0"
    # Re-read the original current pool first. Public fallback is bounded and
    # only needed when recall found no material; old discovery policies remain.
    unresolved_relation_retrieval: None = None


class DependencyEncodingExpansionPolicy(RawRevisionExpansionPolicy):
    """Explicit prompt-only dependency representation clarification."""

    schema_version: Literal["campaign-definition-expansion/1.27.0"] = (
        "campaign-definition-expansion/1.27.0"
    )
    prompt_version: Literal["definition-draft-6.3.0"] = "definition-draft-6.3.0"


class CandidateContractExpansionPolicy(DependencyEncodingExpansionPolicy):
    """Owner-built scalar/selector choices, without repairing model answers."""

    schema_version: Literal["campaign-definition-expansion/1.28.0"] = (
        "campaign-definition-expansion/1.28.0"
    )
    prompt_version: Literal["definition-draft-6.4.0"] = "definition-draft-6.4.0"


class DefinitionFeedbackExpansionPolicy(CandidateContractExpansionPolicy):
    """One original-Provider format correction within the unchanged campaign budget."""

    schema_version: Literal["campaign-definition-expansion/1.29.0"] = (
        "campaign-definition-expansion/1.29.0"
    )
    prompt_version: Literal["definition-draft-6.5.0"] = "definition-draft-6.5.0"


class DiscoveredCandidateExpansionPolicy(CandidateContractExpansionPolicy):
    """Current owner-built candidate choices with explicitly bounded new recall.

    Existing local-first policies retain their meaning. Discovery is not a grant
    or proof of novelty: original sources, acceptance and revision owners decide.
    """

    schema_version: Literal["campaign-definition-expansion/1.30.0"] = (
        "campaign-definition-expansion/1.30.0"
    )
    unresolved_relation_retrieval: Literal["unseen-first-for-owner-gap/1.0.0"] = (
        "unseen-first-for-owner-gap/1.0.0"
    )


class ConfirmedIndexExpansionPolicy(CandidateContractExpansionPolicy):
    """Consume current original-owner indexes without any new public HTTP."""

    schema_version: Literal["campaign-definition-expansion/1.31.0"] = (
        "campaign-definition-expansion/1.31.0"
    )
    index_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    retrieval_scope: Literal["confirmed-index-only/1.0.0"] = "confirmed-index-only/1.0.0"
    unresolved_relation_retrieval: None = None


class RevisedIndexAcquisitionPolicy(ConfirmedIndexExpansionPolicy):
    """Current indexed definitions may schedule a revised, unattempted check."""

    schema_version: Literal["campaign-definition-expansion/1.32.0"] = (
        "campaign-definition-expansion/1.32.0"
    )
    acquisition_revision_scope: Literal["changed-unattempted-current-input/1.0.0"] = (
        "changed-unattempted-current-input/1.0.0"
    )


class RevisedPublicAcquisitionPolicy(DiscoveredCandidateExpansionPolicy):
    """The same scheduling contract after bounded original public discovery."""

    schema_version: Literal["campaign-definition-expansion/1.33.0"] = (
        "campaign-definition-expansion/1.33.0"
    )
    acquisition_revision_scope: Literal["changed-unattempted-current-input/1.0.0"] = (
        "changed-unattempted-current-input/1.0.0"
    )


class IdentityBoundIndexAcquisitionPolicy(RevisedIndexAcquisitionPolicy):
    """One versioned identity check/format feedback over current acquired material."""

    schema_version: Literal["campaign-definition-expansion/1.34.0"] = (
        "campaign-definition-expansion/1.34.0"
    )
    prompt_version: Literal["definition-draft-6.6.0"] = "definition-draft-6.6.0"


class TypedFeedbackIndexAcquisitionPolicy(IdentityBoundIndexAcquisitionPolicy):
    """Request-local dependency rules and candidate-variant-specific diagnostics."""

    schema_version: Literal["campaign-definition-expansion/1.35.0"] = (
        "campaign-definition-expansion/1.35.0"
    )
    prompt_version: Literal["definition-draft-6.7.0"] = "definition-draft-6.7.0"


class NumericInterpretationPublicPolicy(RevisedPublicAcquisitionPolicy):
    """Original numeric observations may receive a source-grounded interpretation.

    This version adds a confirmed-method owner, not measurements or medical QC.
    """

    schema_version: Literal["campaign-definition-expansion/1.36.0"] = (
        "campaign-definition-expansion/1.36.0"
    )
    prompt_version: Literal["definition-draft-6.7.0"] = "definition-draft-6.7.0"


class PassageSelectionExpansionPolicy(NumericInterpretationPublicPolicy):
    schema_version: Literal["campaign-definition-expansion/1.37.0"] = (
        "campaign-definition-expansion/1.37.0"
    )
    prompt_version: Literal["definition-draft-6.8.0"] = "definition-draft-6.8.0"


class BoundedIndexMaterialPolicy(PassageSelectionExpansionPolicy):
    """More original current paragraphs, no additional public calls or verdicts."""

    schema_version: Literal["campaign-definition-expansion/1.38.0"] = (
        "campaign-definition-expansion/1.38.0"
    )
    index_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    retrieval_scope: Literal["confirmed-index-only/1.0.0"] = "confirmed-index-only/1.0.0"
    unresolved_relation_retrieval: None = None
    max_candidates: int = Field(default=16, ge=1, le=16)
    passage_plans: dict[str, BoundedMaterialPassagePlan] = Field(min_length=1, max_length=2)


class WholeBlockIndexMaterialPolicy(BoundedIndexMaterialPolicy):
    """Versioned whole research material; old claim and definition profiles unchanged."""

    schema_version: Literal["campaign-definition-expansion/1.39.0"] = (
        "campaign-definition-expansion/1.39.0"
    )
    passage_plans: dict[str, WholeBlockMaterialPassagePlan] = Field(min_length=1, max_length=2)


class AuthorMethodIndexMaterialPolicy(WholeBlockIndexMaterialPolicy):
    """Explicit public author-method context from the existing numeric owner."""

    schema_version: Literal["campaign-definition-expansion/1.40.0"] = (
        "campaign-definition-expansion/1.40.0"
    )
    confirmed_method_metadata_profile: Literal["grape-author-methods/1.0.0"] = (
        "grape-author-methods/1.0.0"
    )
    confirmed_method_metadata_targets: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def method_metadata_scope(self):
        selected = set(self.confirmed_method_metadata_targets)
        requested = {item for values in self.query_requirements.values() for item in values}
        if (
            len(selected) != len(self.confirmed_method_metadata_targets)
            or not selected <= requested
        ):
            raise ValueError("definition_author_method_requested_scope")
        return self


class GenericConfirmedIndexPolicy(WholeBlockIndexMaterialPolicy):
    """One research interpretation of a confirmed Generic method, no new acquisition."""

    schema_version: Literal["campaign-definition-expansion/1.41.0"] = (
        "campaign-definition-expansion/1.41.0"
    )
    max_definitions: Literal[1] = 1
    confirmed_requirements: tuple[str, ...] = Field(min_length=1, max_length=3)
    public_proposition_checksum: Checksum

    @model_validator(mode="after")
    def supported_target(self):
        # This explicit version is bound to the original Generic root at assembly.
        # Do not force a general original question into the staged-only vocabulary.
        if not self.public_target.startswith("GOAL:") or len(self.public_target) > 160:
            raise ValueError("generic_definition_public_target_code")
        return self

    @model_validator(mode="after")
    def confirmed_scope(self):
        selected = set(self.confirmed_requirements)
        requested = {item for values in self.query_requirements.values() for item in values}
        if len(selected) != len(self.confirmed_requirements) or requested != selected:
            raise ValueError("generic_definition_requested_scope")
        return self


class GenericPublicInterpretationPolicy(GenericConfirmedIndexPolicy):
    """Bounded original public retrieval for an existing confirmed observation.

    Uses the established explanatory passage contract; the wider local-only
    material profiles retain their no-network semantics. Discovery is not proof
    of novelty, medical entailment, or an accepted interpretation.
    """

    schema_version: Literal["campaign-definition-expansion/1.43.0"] = (
        "campaign-definition-expansion/1.43.0"
    )
    retrieval_scope: Literal["generic-bounded-public/1.0.0"] = "generic-bounded-public/1.0.0"
    unresolved_relation_retrieval: Literal["unseen-first-for-owner-gap/1.0.0"] = (
        "unseen-first-for-owner-gap/1.0.0"
    )
    passage_plans: dict[str, ExplanatoryPassagePlan] = Field(min_length=1, max_length=1)
    public_queries: tuple[str, ...] = Field(min_length=1, max_length=1)
    max_candidates: int = Field(default=5, ge=1, le=5)
    max_search_requests: Literal[1] = 1
    max_fetch_requests: int = Field(default=2, ge=1, le=2)
    public_accounting: Literal["original-wire-and-agent-journals/1.0.0"] = (
        "original-wire-and-agent-journals/1.0.0"
    )
    public_query_profile: Literal["glaucoma-general-original-root/1.0.0"] = (
        "glaucoma-general-original-root/1.0.0"
    )


class GenericRegisteredMeasurementPolicy(GenericConfirmedIndexPolicy):
    """Explicit new original-Runtime method observation, preserving the old case."""

    schema_version: Literal["campaign-definition-expansion/1.42.0"] = (
        "campaign-definition-expansion/1.42.0"
    )
    confirmed_observation_scope: Literal["registered-partial-rim-observation/1.0.0"] = (
        "registered-partial-rim-observation/1.0.0"
    )

    @model_validator(mode="after")
    def registered_scope(self):
        if self.confirmed_requirements != ("criterion:RIM_RADIAL_LT_01",):
            raise ValueError("generic_definition_registered_method_scope")
        return self


def located_proposal_context(payload, base):
    materials = []
    for source in payload["source_materials"]:
        text = source["text"]
        materials.append(
            dict(
                code=source["code"],
                checksum=source["checksum"],
                text=text,
                whole_text_excerpt=dict(
                    source_code=source["code"],
                    source_checksum=source["checksum"],
                    start=0,
                    end=len(text),
                    text=text,
                ),
            )
        )
    inputs = []
    for c in base.manifest.components:
        if c.status == "withheld":
            continue
        methods = []
        if c.status in {"obtained", "partial"}:
            if c.kind in {"image", "volume", "image_series"}:
                methods = ["live_image_observation"]
            elif c.kind in {"scalar", "vector", "map"}:
                methods = ["source_numeric_read"]
        inputs.append(
            dict(
                kind=c.kind,
                modality=c.modality,
                concept=c.concept,
                status=c.status,
                installed_methods=methods,
                unit=(c.unit or "source_units_unqualified")
                if "source_numeric_read" in methods
                else None,
                current_values_and_method_still_require_owner_validation=True,
            )
        )
    return dict(
        payload,
        source_materials=materials,
        input_capabilities=inputs,
        citation_contract=dict(
            coordinate_space="source_text_unicode_codepoints",
            end_exclusive=True,
            document_offsets_not_accepted=True,
            source_text_and_original_provenance_retained_by_owner=True,
        ),
    )


class RegisteredTaskExpansionPolicy(SingleDocumentGapExpansionPolicy):
    """Explicit new public-question vocabulary, tied to the original task owner."""

    schema_version: Literal["campaign-definition-expansion/1.44.0"] = (
        "campaign-definition-expansion/1.44.0"
    )
    public_target_vocabulary: Literal["registered-task-public/1.0.0"] = (
        "registered-task-public/1.0.0"
    )
    public_task_checksum: Checksum

    @model_validator(mode="after")
    def supported_target(self):
        from packages.knowledge_search.registered_targets import TARGETS

        if self.public_target not in TARGETS:
            raise ValueError("definition_registered_public_target_scope")
        return self


def read_expansion_policy(value):
    cls = (
        RegisteredTaskExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.44.0"
        else GenericPublicInterpretationPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.43.0"
        else GenericRegisteredMeasurementPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.42.0"
        else GenericConfirmedIndexPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.41.0"
        else AuthorMethodIndexMaterialPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.40.0"
        else WholeBlockIndexMaterialPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.39.0"
        else BoundedIndexMaterialPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.38.0"
        else PassageSelectionExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.37.0"
        else NumericInterpretationPublicPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.36.0"
        else TypedFeedbackIndexAcquisitionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.35.0"
        else IdentityBoundIndexAcquisitionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.34.0"
        else RevisedPublicAcquisitionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.33.0"
        else RevisedIndexAcquisitionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.32.0"
        else ConfirmedIndexExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.31.0"
        else DiscoveredCandidateExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.30.0"
        else DefinitionFeedbackExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.29.0"
        else CandidateContractExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.28.0"
        else DependencyEncodingExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.27.0"
        else RawRevisionExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.26.0"
        else ExplanatoryGapExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.25.0"
        else SingleDocumentGapExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.24.0"
        else AdaptiveGapExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.23.0"
        else OwnerDependencyExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.22.0"
        else ConfirmedInterpretationExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.21.0"
        else BalancedSourceExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.20.0"
        else RequirementContextExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.19.0"
        else CompletePoolExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.18.0"
        else WindowedExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.17.0"
        else UnseenFirstExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.16.0"
        else GapDiscoveryExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.15.0"
        else RequirementScopeExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.14.0"
        else IntentExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.13.0"
        else CitationExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.12.0"
        else EvidenceExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.11.0"
        else CapacityExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.10.0"
        else DependencyExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.9.0"
        else JatsExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.8.0"
        else PublicTargetExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.7.0"
        else OwnerLocatedExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.6.0"
        else LocatedExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.5.0"
        else ContextualExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.4.0"
        else TargetedExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.3.0"
        else ResilientExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.2.0"
        else TargetExpansionPolicy
        if value.get("schema_version") == "campaign-definition-expansion/1.1.0"
        else ExpansionPolicy
    )
    return validate(cls, value)


def definition_config(
    config, *, prompt_version="definition-draft-2.0.0", source_output_tokens=None
):
    if prompt_version not in {
        "definition-draft-2.0.0",
        "definition-draft-2.1.0",
        "definition-draft-2.2.0",
        "definition-draft-2.3.0",
        "definition-draft-3.0.0",
        "definition-draft-4.0.0",
        "definition-draft-4.1.0",
        "definition-draft-5.0.0",
        "definition-draft-6.0.0",
        "definition-draft-6.1.0",
        "definition-draft-6.2.0",
        "definition-draft-6.3.0",
        "definition-draft-6.4.0",
        "definition-draft-6.5.0",
        "definition-draft-6.6.0",
        "definition-draft-6.7.0",
        "definition-draft-6.8.0",
    }:
        raise ValueError("definition_prompt_version")
    raw = config.model_dump(mode="json")
    if prompt_version in {
        "definition-draft-2.3.0",
        "definition-draft-3.0.0",
        "definition-draft-4.0.0",
        "definition-draft-4.1.0",
        "definition-draft-5.0.0",
        "definition-draft-6.0.0",
        "definition-draft-6.1.0",
        "definition-draft-6.2.0",
        "definition-draft-6.3.0",
        "definition-draft-6.4.0",
        "definition-draft-6.5.0",
        "definition-draft-6.6.0",
        "definition-draft-6.7.0",
        "definition-draft-6.8.0",
    }:
        raw["transport"]["max_retries"] = 0
    role = next(r for r in raw["roles"] if r["role"] == "source")
    if source_output_tokens is not None:
        if source_output_tokens != 8192 or prompt_version not in {
            "definition-draft-3.0.0",
            "definition-draft-4.0.0",
            "definition-draft-4.1.0",
            "definition-draft-5.0.0",
            "definition-draft-6.0.0",
            "definition-draft-6.1.0",
            "definition-draft-6.2.0",
            "definition-draft-6.3.0",
            "definition-draft-6.4.0",
            "definition-draft-6.5.0",
            "definition-draft-6.6.0",
            "definition-draft-6.7.0",
            "definition-draft-6.8.0",
        }:
            raise ValueError("definition_source_capacity")
        role["max_output_tokens"] = source_output_tokens
    role.update(
        prompt_version=prompt_version,
        prompt_checksum=digest(Path("configs/prompts/agents", prompt_version + ".txt").read_text()),
    )
    from packages.llm_provider.config import read_provider_config

    return read_provider_config(raw)


class DefinitionExpansion:
    def __init__(self, campaign, *, documents, sources, provider, budget, scope, policy):
        self.campaign, self.documents, self.sources = campaign, documents, sources
        self.provider, self.budget, self.scope = provider, budget, scope
        self.policy = read_expansion_policy(policy)
        if isinstance(self.policy, ContextualExpansionPolicy):
            self.sources.before_read = self.control
        self.original_rows = list(campaign.base.ledger)
        self.binding = dict(
            campaign_id=getattr(campaign, "logical_campaign_id", campaign.campaign_id),
            input_checksum=campaign.base.manifest.checksum,
            target=campaign.base.task.target_code,
            protocol_checksum=digest(campaign.protocol),
            parent_requirements_checksum=digest(self.original_rows),
            policy_checksum=digest(self.policy),
        )
        if isinstance(self.policy, PublicTargetExpansionPolicy) and (
            self.policy.public_target != self.binding["target"]
        ):
            raise ValueError("definition_public_target_binding")
        if isinstance(self.policy, RegisteredTaskExpansionPolicy) and (
            self.policy.public_task_checksum != campaign.base.task.checksum
        ):
            raise ValueError("definition_registered_task_binding")
        confirmed_owner = getattr(campaign.base, "component_observations", None)
        if isinstance(self.policy, GenericConfirmedIndexPolicy):
            from packages.evaluation_preparation.generic_definition_input import (
                GenericDefinitionInputView,
            )

            if type(campaign.base) is not GenericDefinitionInputView:
                raise ValueError("generic_definition_original_input_view_required")
            campaign.base.check()
            if (
                tuple(r["id"] for r in self.original_rows) != self.policy.confirmed_requirements
                or digest(campaign.base.public_goal) != self.policy.public_proposition_checksum
            ):
                raise ValueError("generic_definition_policy_owner_binding")
            confirmed_owner = campaign.base.facts
        elif isinstance(self.policy, NumericInterpretationPublicPolicy):
            from .confirmed_numeric_observations import ConfirmedNumericObservations

            confirmed_owner = ConfirmedNumericObservations(
                campaign.base,
                **(
                    dict(
                        method_context_profile=self.policy.confirmed_method_metadata_profile,
                        method_context_targets=self.policy.confirmed_method_metadata_targets,
                    )
                    if isinstance(self.policy, AuthorMethodIndexMaterialPolicy)
                    else {}
                ),
            )
        self.owner = ResearchDefinitionOwner(
            documents=documents,
            sources=sources,
            campaign_id=self.binding["campaign_id"],
            input_checksum=self.binding["input_checksum"],
            target=self.binding["target"],
            guard=self.control,
            input_reader=self.current_manifest,
            fact_reader=self.current_fact,
            confirmed_observations=confirmed_owner
            if isinstance(self.policy, ConfirmedInterpretationExpansionPolicy)
            else None,
            requirement_ids=tuple(r["id"] for r in self.original_rows)
            if isinstance(self.policy, ConfirmedInterpretationExpansionPolicy)
            else (),
        )

    def current_fact(self, contract):
        reader = getattr(self.campaign.base, "read_dependency_fact", None)
        if reader is None:
            raise ValueError("definition_dependency_fact_owner_missing")
        return reader(contract)

    def project(self, receipt):
        if receipt.get("schema_version") in CONFIRMED_RECEIPTS:
            if not isinstance(self.policy, ConfirmedInterpretationExpansionPolicy):
                raise ValueError("definition_confirmed_policy_required")
            return [self.owner.project_confirmed(receipt)]
        return requirement_rows(receipt, self.current_manifest(), fact_reader=self.current_fact)

    def current_manifest(self):
        # This original input port excludes definition_current to avoid recursion.
        reader = getattr(self.campaign.base, "read_dependency_input", None)
        if reader is None:
            raise ValueError("definition_input_owner_missing")
        return reader()

    def fast_control(self):
        """Local polling only; never performs a DB round trip or authorizes a send."""
        failure = self.campaign.output / "stage-failure.json"
        amendment = getattr(self.campaign, "local_material_repair", {})
        repaired = (
            bool(amendment)
            and failure.exists()
            and digest(json.loads(failure.read_bytes())) == amendment["original_failure_checksum"]
        )
        closure = self.campaign.output / "stage-closure-error.json"
        repaired_closure = (
            repaired
            and amendment.get("schema_version") == "confirmed-material-local-repair/1.6.0"
            and closure.exists()
            and digest(json.loads(closure.read_bytes())) == amendment["original_closure_checksum"]
        )
        if (
            (failure.exists() and not repaired)
            or (closure.exists() and not repaired_closure)
            or (self.campaign.output / "local-repair-failure.json").exists()
        ):
            raise ProviderError("stage_previous_failure_read_only")
        if (self.campaign.output / "cancel").exists():
            raise ProviderError("cancelled")
        if datetime.now(UTC) >= datetime.fromisoformat(self.campaign.protocol["deadline"]):
            raise ProviderError("deadline")

    def control(self):
        self.fast_control()
        current = self.campaign.sessions.read(self.scope.run_id)
        if current is not None and (current.cancelled or current.phase == "stopped"):
            raise ProviderError("cancelled")

    def guard(self, *, check_reserve=True):
        self.control()
        self.check_materials()
        if datetime.now(UTC) >= datetime.fromisoformat(
            self.campaign.protocol["deadline"]
        ) - timedelta(seconds=self.policy.reserve_close_seconds):
            raise ProviderError("definition_close_reserve")
        previous = getattr(self.campaign, "previous_attempts", lambda: [])()
        if check_reserve and isinstance(self.policy, AdaptiveGapExpansionPolicy):
            used = self.campaign.usage()
            remaining = min(
                self.campaign.protocol.get("max_new_requests", 40) - used,
                40 - used - sum(p["requests"] for p in previous),
            )
            if remaining <= self.policy.reserve_review_requests:
                raise ProviderError("definition_review_reserve")
        cap = (
            min(40, self.campaign.protocol["max_new_requests"])
            if hasattr(self.campaign, "trial")
            else 40
        )
        if check_reserve and (
            self.campaign.usage() + sum(p["requests"] for p in previous)
            >= cap - self.policy.reserve_review_requests
        ):
            raise ProviderError("definition_review_reserve")
        if check_reserve and isinstance(self.policy, AdaptiveGapExpansionPolicy):
            if datetime.now(UTC) >= datetime.fromisoformat(
                self.campaign.protocol["deadline"]
            ) - timedelta(
                seconds=self.policy.reserve_review_seconds + self.policy.reserve_close_seconds
            ):
                raise ProviderError("definition_review_time_reserve")

    def check_materials(self):
        # Real campaigns expose the original input owner's narrower dependency
        # port. Legacy assemblies still perform their original complete guard.
        check = getattr(self.campaign, "guard_definition_materials", self.campaign.guard)
        check()

    def operations(self):
        # Individual immutable slots, not a mutable list/hash.
        from packages.research_runtime.agent_records import RuntimeDocuments

        keys = tuple(f"definition:operation:{n}" for n in range(self.operation_limit))
        values = (
            self.documents.read_many(keys)
            if type(self.documents) is RuntimeDocuments
            else tuple(self.documents.read(key) for key in keys)
        )
        return [saved[0] for saved in values if saved]

    @property
    def operation_limit(self):
        return getattr(self.campaign, "definition_operation_limit", self.policy.max_operations)

    def reserve_wire(self, provider, *, kind=None):
        if isinstance(self.policy, ConfirmedIndexExpansionPolicy):
            raise ProviderError("definition_index_only_no_public_dispatch")
        """One full current-owner check at the wire boundary, not every timer poll."""
        self.guard()
        with self.documents.session():
            self.control()
            ops = self.operations()
            extra = {}
            if isinstance(self.policy, TargetedExpansionPolicy):
                if kind not in {"search", "fetch"}:
                    raise ProviderError("definition_wire_kind_required")
                limit = getattr(self.policy, "max_" + kind + "_requests")
                if sum(o.get("wire_kind") == kind for o in ops) >= limit:
                    raise ProviderError("definition_public_request_budget")
                extra["wire_kind"] = kind
            if len(ops) >= self.operation_limit:
                raise ProviderError("definition_operation_budget")
            key = f"definition:wire:{len(ops)}"
            self.documents.write(
                f"definition:operation:{len(ops)}",
                (dict(key=key, kind="wire", binding=self.binding, **extra),),
            )
            self.documents.write(key + ":result", (dict(provider=provider, reservation_only=True),))
        self.control()
        return key

    def operation(self, key, kind, call):
        # Reserve is a dispatch boundary, not a ban on reading confirmed facts.
        # Current input/control, absolute/working time and total budget still apply.
        key = self.attempt_key(key)
        if isinstance(self.policy, AdaptiveGapExpansionPolicy):
            self.control()
            saved = self.documents.read(key + ":result")
            if saved:
                # Immutable orchestration history, NOT a current-use decision.
                # New dispatches, acceptance and revision consumption still use
                # their original current-source/input/Control owners below.
                self.control()
                return saved[0]
        self.guard(check_reserve=False)
        # session is an original per-run mutex, NOT a multi-write DB transaction.
        # One atomic bundle is both the reservation and its counted slot.
        with self.documents.session():
            self.control()
            saved = self.documents.read(key + ":result")
            if saved:
                return saved[0]
            ops = self.operations()
            if any(o["key"] == key for o in ops):
                amendment = getattr(self.campaign, "local_material_repair", {})
                if amendment.get("pending_key") == key:
                    from .archive_recovery import admit

                    admit(self, key, ops)
                elif (
                    isinstance(self.policy, CompletePoolExpansionPolicy)
                    and kind == "local_pool_reconciliation"
                ) or (
                    isinstance(self.policy, WindowedExpansionPolicy)
                    and kind
                    in {
                        "local_search",
                        "local_index_plan",
                        "local_index_window",
                    }
                ):
                    # Only local orchestration / deterministic owner writes can
                    # resume. Nested HTTP operations retain their own UNKNOWN
                    # boundary, so an unconfirmed remote request is never replayed.
                    original = [o for o in ops if o["key"] == key]
                    if original != [dict(key=key, kind=kind, binding=self.binding)]:
                        raise ProviderError("definition_local_resume_binding")
                elif (self.campaign.output / "archive-repair.json").exists():
                    from .archive_recovery import admit

                    admit(self, key, ops)
                elif (self.campaign.output / "local-search-repair.json").exists():
                    from .search_recovery import admit_operation

                    admit_operation(self, key, ops)
                else:
                    raise ProviderError("definition_response_unknown_no_repeat")
                self.guard()
            else:
                self.guard()
                if len(ops) >= self.operation_limit:
                    raise ProviderError("definition_operation_budget")
                claim = dict(key=key, kind=kind, binding=self.binding)
                # Check before a new group can start. Existing unconfirmed groups
                # above remain UNKNOWN even when the public quota is exhausted.
                wire_kind = {"search_group": "search", "fetch_group": "fetch"}.get(kind)
                if (
                    isinstance(self.policy, TargetedExpansionPolicy)
                    and wire_kind
                    and sum(o.get("wire_kind") == wire_kind for o in ops)
                    >= getattr(self.policy, "max_" + wire_kind + "_requests")
                ):
                    value = dict(
                        status="public_budget_exhausted",
                        wire_kind=wire_kind,
                        dispatched=False,
                        reason="definition_public_request_budget",
                    )
                    self.documents.write(f"definition:operation:{len(ops)}", (claim,))
                    self.documents.write(key + ":result", (value,))
                    return value
                self.documents.write(f"definition:operation:{len(ops)}", (claim,))
        value = json.loads(canonical(call()))
        # Confirmed response is retained even if cancellation occurs on return.
        with self.documents.session():
            self.documents.write(key + ":result", (value,))
        self.guard(check_reserve=False)
        return value

    def current(self):
        self.control()
        revision = self.read_revision()
        self.control()
        return revision

    def read_revision(self):
        """Current source bindings of saved facts; no permission to start new work.

        Reporting may read these after the campaign deadline. Execution still
        uses current()/guard(), which independently checks cancellation/time.
        """
        saved = self.documents.read("definition:revision")
        if saved is None or saved[0]["binding"] != self.binding:
            raise ValueError("definition_revision_binding")
        revision = saved[0]
        if isinstance(self.policy, AdaptiveGapExpansionPolicy):
            from .gap_expansion import verify_summary

            verify_summary(self, revision["gap_expansion"])
        if isinstance(self.policy, RequirementScopeExpansionPolicy):
            if revision.get("review_source_scope") != self.source_scope_record(revision):
                raise ValueError("definition_review_source_scope_binding")
        self.sources.restore(revision["materials"])
        required_materials = self.required_materials(revision)
        batch = getattr(self.sources, "read_many", None)
        materials = (
            batch([s["code"] for s in required_materials])
            if batch is not None
            else [self.sources.read(s["code"]) for s in required_materials]
        )
        if materials != required_materials:
            raise ValueError("definition_revision_source_changed")
        adopted = (
            {r.get("definition_receipt_key") for r in revision["rows"]} - {None}
            if isinstance(self.policy, RequirementScopeExpansionPolicy)
            else set(revision["accepted"])
        )
        inherited, inherited_proof = self.inherited_revision()
        if revision.get("inherited_definition_origin") != inherited_proof:
            raise ValueError("definition_inherited_origin_changed")
        inherited_keys = {r.get("definition_receipt_key") for r in inherited.get("rows", [])} - {
            None
        }
        if not adopted <= revision["accepted"].keys() | inherited_keys:
            raise ValueError("definition_dependency_projection_binding")
        for key, expected in revision["accepted"].items():
            # Unused receipts remain exact original owner history, not current
            # authorization. Adopted rows below still require full reconstruction.
            if key in adopted and expected.get("schema_version") in CONFIRMED_RECEIPTS:
                if not isinstance(self.policy, ConfirmedInterpretationExpansionPolicy):
                    raise ValueError("definition_confirmed_policy_required")
                # Its current read is paired with projection at the original
                # projection boundary below, not retained across owner calls.
                current = self.owner.read(key, historical=True)
            else:
                current = (
                    self.owner.read_for_report(key)
                    if key in adopted
                    else self.owner.read(key, historical=True)
                )
            if canonical(current) != canonical(expected):
                raise ValueError("definition_revision_source_changed")
        if isinstance(self.policy, DependencyExpansionPolicy):
            original = {r["id"]: json.loads(canonical(r)) for r in self.original_rows}
            added = set(revision["added"])
            replacements = {r["id"] for r in revision["replaced"]}
            row_ids = {r["id"] for r in revision["rows"]}
            if (
                len(row_ids) != len(revision["rows"])
                or len(added) != len(revision["added"])
                or len(replacements) != len(revision["replaced"])
                or added & original.keys()
                or not replacements <= original.keys()
                or row_ids != original.keys() | added
                or revision["denominator_before"] != len(original)
                or revision["denominator_after"] != len(row_ids)
            ):
                raise ValueError("definition_dependency_projection_binding")
            projected_ids = {}
            for key, receipt in revision["accepted"].items():
                if key not in adopted:
                    continue
                if receipt.get("schema_version") in CONFIRMED_RECEIPTS:
                    current, projection = self.owner.read_confirmed_projection(key)
                    if canonical(current) != canonical(receipt):
                        raise ValueError("definition_revision_source_changed")
                    projections = [projection]
                else:
                    projections = self.project(receipt)
                for projection in projections:
                    projected_ids[projection["id"]] = (key, projection)
                    for replacement in revision["replaced"]:
                        if replacement["definition"] == key:
                            projected_ids[replacement["id"]] = (key, projection)
            for row in revision["rows"]:
                if row["id"] in original and row["id"] not in replacements:
                    if row != original[row["id"]]:
                        raise ValueError("definition_dependency_projection_binding")
                    continue
                projected = projected_ids.get(row["id"])
                if projected is None:
                    raise ValueError("definition_dependency_projection_binding")
                key, expected = projected
                if row.get("definition_receipt_key") != key:
                    raise ValueError("definition_dependency_projection_binding")
                if row["id"] in added:
                    complete = dict(expected, definition_receipt_key=key)
                    if (
                        complete.get("acquisition_method") == "source_measurement"
                        and complete["component"]
                        and self.numeric_observation(complete) is None
                    ):
                        complete["missing_reason"] = (
                            "definition_measurement_method_unit_mismatch_or_missing"
                        )
                    if row != json.loads(canonical(complete)):
                        raise ValueError("definition_dependency_projection_binding")
                else:
                    if isinstance(self.policy, RequirementContextExpansionPolicy):
                        from .definition_context import same_examination

                        if not same_examination(original[row["id"]], expected) or not (
                            self.compatible_interpretation(original[row["id"]], expected)
                        ):
                            raise ValueError("definition_dependency_projection_binding")
                    changes = {
                        k: expected[k]
                        for k in (
                            "definition",
                            "basis_kind",
                            "definition_receipt_checksum",
                            "dependency_assessment",
                            "acquisition_allowed",
                            "review_allowed",
                        )
                    }
                    complete = dict(original[row["id"]], **changes, definition_receipt_key=key)
                    if revision["accepted"][key].get("schema_version") in CONFIRMED_RECEIPTS:
                        complete.update(
                            expected_method=expected["expected_method"],
                            expected_unit=expected["expected_unit"],
                        )
                    if row != json.loads(canonical(complete)):
                        raise ValueError("definition_dependency_projection_binding")
                # Original adapter rows may bind method/unit exclusively through
                # their confirmed observation. Do not rewrite an immutable legacy
                # revision to add fields: compare the effective owner-bound values.
                compared = dict(row)
                if row["id"] in replacements:
                    method, unit = self.effective_interpretation_method(original[row["id"]])
                    compared.setdefault("expected_method", method)
                    compared.setdefault("expected_unit", unit)
                if any(
                    json.loads(canonical(compared.get(k))) != json.loads(canonical(expected[k]))
                    for k in (
                        "definition",
                        "dependency_assessment",
                        "acquisition_allowed",
                        "review_allowed",
                        "component",
                        "input_modalities",
                        "acquisition_method",
                        "definition_receipt_checksum",
                        "expected_method",
                        "expected_unit",
                    )
                ):
                    raise ValueError("definition_dependency_projection_binding")
        return revision

    @property
    def proposal_model(self):
        return (
            OwnerBoundDefinitionProposal
            if isinstance(self.policy, OwnerDependencyExpansionPolicy)
            else ConfirmedInterpretationProposal
            if isinstance(self.policy, ConfirmedInterpretationExpansionPolicy)
            else EvidenceDefinitionProposal
            if isinstance(self.policy, EvidenceExpansionPolicy)
            else ScopedDefinitionProposal
            if isinstance(self.policy, DependencyExpansionPolicy)
            else QuotedDefinitionProposal
            if isinstance(self.policy, OwnerLocatedExpansionPolicy)
            else DefinitionProposal
        )

    def checkpoint(self, stage, payload):
        self.control()
        value = json.loads(
            canonical(
                dict(
                    schema_version="definition-stage-checkpoint/1.0.0",
                    stage=stage,
                    binding=self.binding,
                    **payload,
                )
            )
        )
        with self.documents.session():
            self.control()
            self.documents.write("definition:stage:" + stage, (value,))
        return value

    def advance(self, *, until=None):
        if until not in {None, "sources", "definitions", "requirements"}:
            raise ValueError("definition_stage_unknown")
        prior = self.documents.read("definition:revision")
        if prior:
            return self.install(self.current())
        continuation = getattr(self.campaign, "definition_continuation", None)
        if continuation is not None and continuation.defer:
            return continuation.defer_expansion(self, until)
        if isinstance(self.policy, AdaptiveGapExpansionPolicy):
            from .gap_expansion import advance

            return advance(self, until=until)
        try:
            return self._advance(until=until)
        except ProviderError as exc:
            if exc.code not in {
                "definition_close_reserve",
                "definition_review_reserve",
                "definition_operation_budget",
                "definition_public_request_budget",
            }:
                raise
            if until in {"sources", "definitions"}:
                raise
            # A finite expansion limit does not discard prior patient facts or spend
            # the reserved final-review budget. Unknown transport is NOT this path.
            return self.commit_revision(
                list(self.original_rows), {}, {}, [], [], [dict(reason=exc.code)], stop=exc.code
            )

    def query_plan(self):
        """Public questions are selected by actual missing definition coverage.

        Configured requirement IDs are medical-profile data, never patient values.
        Unknown scopes without a configured public query remain explicit backlog.
        """
        result = []
        covered = {m for r in self.original_rows for m in r["input_modalities"]}
        uncovered = sorted(
            {
                c.modality
                for c in self.campaign.base.manifest.components
                if c.status in {"obtained", "partial"}
            }
            - covered
        )
        for query in self.policy.public_queries:
            ids = self.policy.query_requirements.get(query, ())
            matched = [
                r
                for r in self.original_rows
                if any(r["id"] == i or r["id"].startswith(i + ":") for i in ids)
            ]
            missing = [
                i
                for i in ids
                if not any(r["id"] == i or r["id"].startswith(i + ":") for r in matched)
            ]
            unbound = [
                r["id"]
                for r in (matched if ids else self.original_rows)
                if r["definition"].interpretation_status != "installed"
            ]
            target_gaps = (
                [r["id"] for r in matched if not self.target_relation_bound(r)]
                if isinstance(self.policy, TargetExpansionPolicy)
                else []
            )
            needed = bool(
                missing
                or unbound
                or target_gaps
                or (not ids and (uncovered or not self.original_rows))
            )
            result.append(
                dict(
                    query=query,
                    selected=needed,
                    missing_definition_ids=missing,
                    unbound_interpretation_ids=unbound,
                    uncovered_modalities=uncovered,
                    reason="target_relation_unverified"
                    if target_gaps
                    else "definition_coverage_gap"
                    if needed
                    else "current_definition_reused",
                    **(
                        dict(target_relation_gap_ids=target_gaps)
                        if isinstance(self.policy, TargetExpansionPolicy)
                        else {}
                    ),
                    **(
                        dict(
                            retrieval_mode=self.policy.unresolved_relation_retrieval
                            if target_gaps
                            else None
                        )
                        if isinstance(self.policy, GapDiscoveryExpansionPolicy)
                        else {}
                    ),
                )
            )
        return result

    def target_relation_bound(self, row):
        """An installed generic instruction is not an owner-accepted target relation.

        Only an exact current receipt from this campaign qualifies. No keyword
        matching, caller validated flag, or invented medical stage relation.
        """
        key = row.get("definition_receipt_key")
        if not key:
            return False
        accepted = self.owner.read(key)
        return (
            accepted["target"] == self.binding["target"]
            and accepted["draft"]["relation_use"] == "directional"
            and digest(accepted) == row.get("definition_receipt_checksum")
            and accepted["definition"] == json.loads(canonical(row["definition"]))
        )

    def attempt_key(self, key):
        attempt = getattr(self, "gap_attempt", None)
        if attempt is None or key.startswith("definition:gap:"):
            return key
        return f"definition:gap:{attempt}:" + key.removeprefix("definition:")

    def collect_materials(self, planned_queries):
        found, backlog = {}, []
        selected_queries = sum(p["selected"] for p in planned_queries)
        per_query_capacity = (
            (self.policy.max_candidates + selected_queries - 1) // selected_queries
            if selected_queries
            else self.policy.max_candidates
        )
        for index, planned in enumerate(planned_queries):
            if not planned["selected"]:
                continue
            query = planned["query"]
            remaining_materials = self.policy.max_candidates - len(found)
            if isinstance(self.policy, CompletePoolExpansionPolicy):
                # Provisional recall must not prevent another planned source
                # from entering the pool. The final shared quota is applied once
                # by reconciliation, before any material reaches the Provider.
                remaining_materials = per_query_capacity
            elif isinstance(self.policy, ContextualExpansionPolicy):
                remaining_materials = min(
                    remaining_materials,
                    per_query_capacity,
                )
            inherited = getattr(self.campaign, "discovery_continuation", None)
            if inherited and query in inherited.failed_queries:
                backlog.append(
                    dict(
                        reason="previous_search_technical_failure_not_replayed",
                        query=query,
                        previous=inherited.reference,
                    )
                )
                continue
            value = self.operation(
                f"definition:query:{index}",
                "local_search",
                lambda query=query, index=index, limit=remaining_materials, planned=planned: (
                    self.sources.search(
                        query=query,
                        key=self.attempt_key(
                            f"definition:search:{digest(self.binding)[7:31]}:{index}"
                        ),
                        limit=limit,
                        operation=self.operation,
                        **(
                            dict(
                                retrieval_mode=None
                                if self.campaign.protocol.get("new_public_search_allowed") is False
                                else planned["retrieval_mode"]
                            )
                            if isinstance(self.policy, GapDiscoveryExpansionPolicy)
                            else {}
                        ),
                    )
                ),
            )
            if isinstance(self.policy, IntentExpansionPolicy):
                if value.get("schema_version") != "research-definition-search/1.1.0":
                    raise ValueError("definition_index_pool_result_version")
                # Only after the original operation returned/confirmed. Cold
                # recovery reaches this same path without reissuing HTTP work.
                self.sources.restore_indexes(value["acquired_indexes"])
            for material in value["materials"]:
                found[material["code"]] = material
            backlog.extend(value["backlog"])
            if isinstance(self.policy, GapDiscoveryExpansionPolicy):
                backlog.append(
                    dict(
                        reason="owner_gap_retrieval_plan",
                        plan=planned,
                        discovery=value.get("discovery"),
                    )
                )
            if "retrieval" in value:
                backlog.append(
                    dict(reason="contextual_retrieval_trace", retrieval=value["retrieval"])
                )
            if (
                not isinstance(self.policy, CompletePoolExpansionPolicy)
                and len(found) >= self.policy.max_candidates
            ):
                break
        provisional_limit = (
            selected_queries * per_query_capacity
            if isinstance(self.policy, CompletePoolExpansionPolicy)
            else self.policy.max_candidates
        )
        if len(found) > provisional_limit:
            raise ValueError("definition_source_budget")
        if isinstance(self.policy, CompletePoolExpansionPolicy):
            queries = [p["query"] for p in planned_queries if p["selected"]]
            if queries:
                pool_binding = dict(
                    campaign=self.binding,
                    queries=queries,
                    indexes=list(self.sources.index_keys),
                    policy=digest(self.policy),
                    provisional_codes=list(found),
                )
                reconciled = self.operation(
                    "definition:pool:" + digest(pool_binding)[7:39],
                    "local_pool_reconciliation",
                    lambda: self.sources.reconcile_pool(
                        queries=queries, limit=self.policy.max_candidates
                    ),
                )
                if reconciled.get("schema_version") != "research-definition-pool/1.0.0":
                    raise ValueError("definition_pool_result_version")
                materials = reconciled["materials"]
                if len(materials) > self.policy.max_candidates or len(
                    {m["code"] for m in materials}
                ) != len(materials):
                    raise ValueError("definition_source_budget")
                backlog.extend(reconciled["backlog"])
                backlog.append(
                    dict(
                        reason="complete_source_pool_reconciliation",
                        binding=pool_binding,
                        previous_codes=list(found),
                        selected_codes=[m["code"] for m in materials],
                        retrieval=reconciled["retrieval"],
                    )
                )
                found = {m["code"]: m for m in materials}
        return found, backlog

    def _advance(self, *, until=None, planned_queries=None, candidate_only=False):
        # Cold phase recovery must be able to consume the already confirmed
        # proposal at the reserve boundary; each new operation/wire checks reserve.
        self.guard(check_reserve=False)
        from .source_checkpoint import materials as checkpoint_materials

        planned_queries = self.query_plan() if planned_queries is None else planned_queries
        inherited = checkpoint_materials(self, planned_queries)
        found, backlog = (
            inherited if inherited is not None else self.collect_materials(planned_queries)
        )
        self.sources.restore(list(found.values()))
        if until == "sources":
            return self.checkpoint("sources", dict(materials=list(found.values()), backlog=backlog))
        rows, added, merged, accepted, replaced = list(self.original_rows), [], [], {}, []
        material_fingerprint = digest(
            sorted((s["code"], s["checksum"], digest(s["text"])) for s in found.values())
        )
        # Same target, unchanged requirements and identical evidence do not justify
        # another Source call merely because a public query has a different name.
        evaluation_fingerprint = digest(
            (
                material_fingerprint,
                sorted(
                    {
                        i
                        for p in planned_queries
                        for name in (
                            "missing_definition_ids",
                            "unbound_interpretation_ids",
                            "target_relation_gap_ids",
                            "uncovered_modalities",
                        )
                        for i in p.get(name, ())
                    }
                ),
            )
        )
        duplicate_material = candidate_only and evaluation_fingerprint in getattr(
            self, "gap_seen_materials", set()
        )
        if duplicate_material:
            backlog.append(
                dict(reason="same_material_no_new_context", fingerprint=material_fingerprint)
            )
        if found and not duplicate_material:
            payload = dict(
                target=self.binding["target"],
                max_definitions=self.policy.max_definitions,
                existing_examinations=[r["definition"].examination for r in rows],
                definition_gaps=self.query_plan(),
                available_modalities=sorted(
                    {c.modality for c in self.campaign.base.manifest.components}
                ),
                source_materials=list(found.values()),
                public_queries=[p["query"] for p in planned_queries]
                if candidate_only
                else self.policy.public_queries,
            )
            if isinstance(self.policy, LocatedExpansionPolicy):
                payload = located_proposal_context(payload, self.campaign.base)
            if isinstance(self.policy, RequirementContextExpansionPolicy):
                from .definition_context import (
                    public_examination,
                    public_gap_context,
                    requirement_context,
                    same_examination,
                )

                payload["existing_examinations"] = [public_examination(r) for r in rows]
                payload["existing_requirement_context"] = requirement_context(
                    rows, payload["definition_gaps"], self.campaign.base.observations
                )
                payload["definition_gaps"] = public_gap_context(rows, payload["definition_gaps"])
            if isinstance(self.policy, OwnerDependencyExpansionPolicy):
                from .definition_context import owner_capabilities

                if isinstance(self.policy, GenericConfirmedIndexPolicy):
                    payload = self.campaign.base.draft_context(payload)
                else:
                    payload = owner_capabilities(
                        payload,
                        self.campaign.base,
                        rows,
                        confirmed_owner=self.owner.confirmed_observations
                        if isinstance(self.policy, NumericInterpretationPublicPolicy)
                        else None,
                    )
            if isinstance(self.policy, CandidateContractExpansionPolicy):
                from .definition_context import candidate_contract_choices

                payload = candidate_contract_choices(payload)
            if isinstance(self.policy, OwnerLocatedExpansionPolicy):
                for material in payload["source_materials"]:
                    material.pop("whole_text_excerpt")
                payload["citation_contract"] = dict(
                    exact_quote_only=True,
                    offsets_assigned_by_owner=True,
                    prefix_suffix_are_exact_adjacent_context=True,
                )
            if isinstance(self.policy, ContextualExpansionPolicy):
                payload["retrieval_policy"] = dict(
                    version=self.policy.schema_version,
                    local_plans={
                        q: p.model_dump(mode="json") for q, p in self.policy.passage_plans.items()
                    },
                    neighbors_are_context_not_independent_support=True,
                )

            # No patient values or IDs in the payload. Scope remains transport/journal identity.
            def propose():
                result = self.provider.generate_definitions(
                    payload=payload,
                    scope=self.scope,
                    budget=self.budget,
                    before_request=self.guard,
                    capture=lambda record, reply: self.capture(record, reply, payload=payload),
                )
                return result.value if hasattr(result, "value") else result

            raw = self.documents.read(self.attempt_key("definition:confirmed-proposal"))
            if raw:
                proposal = validate(self.proposal_model, raw[0])
            else:
                proposal = validate(
                    self.proposal_model, self.operation("definition:proposal", "llm", propose)
                )
            if len(proposal.definitions) > self.policy.max_definitions:
                raise ValueError("definition_candidate_budget")
            for draft in proposal.definitions:
                if (
                    isinstance(self.policy, TargetedExpansionPolicy)
                    and not isinstance(self.policy, DependencyExpansionPolicy)
                    and draft.relation_use != "directional"
                ):
                    backlog.append(
                        dict(reason="target_direction_not_established", definition=digest(draft))
                    )
                    continue
                if not {e.source_code for e in draft.excerpts} <= found.keys():
                    raise ValueError("definition_draft_source_scope")
                if (
                    isinstance(self.policy, GenericConfirmedIndexPolicy)
                    and draft.acquisition != "confirmed_observation"
                ):
                    backlog.append(
                        dict(
                            reason="generic_confirmed_interpretation_only", definition=digest(draft)
                        )
                    )
                    continue
                key = digest(draft)[7:47]
                try:
                    value = self.owner.accept(key, draft)
                    projected = (
                        self.project(value)
                        if isinstance(self.policy, EvidenceExpansionPolicy)
                        else requirement_rows(value, self.campaign.base.manifest)
                    )
                except ValueError as exc:
                    # Only known domain refusals are backlog. DB/object/unknown errors
                    # retain their technical failure semantics and stop this attempt.
                    if str(exc) not in {
                        "definition_draft_target",
                        "definition_draft_exact_excerpt",
                        "definition_quote_ambiguous",
                        "definition_quote_not_found",
                        "definition_joint_input_method_not_installed",
                        "definition_input_method_mismatch",
                        "definition_dependency_unresolved",
                    }:
                        raise
                    backlog.append(dict(definition=key, reason=str(exc)))
                    continue
                for row in projected:
                    equivalent = next(
                        (
                            r
                            for r in rows
                            if (
                                same_examination(r, row)
                                if isinstance(self.policy, RequirementContextExpansionPolicy)
                                else r["component"] == row["component"]
                                and r["definition"].examination.strip().casefold()
                                == row["definition"].examination.strip().casefold()
                            )
                        ),
                        None,
                    )
                    if equivalent:
                        target_gap = isinstance(
                            self.policy, TargetExpansionPolicy
                        ) and not self.target_relation_bound(equivalent)
                        if (
                            equivalent["definition"].interpretation_status == "missing"
                            or target_gap
                        ) and row["definition"].interpretation_status == "installed":
                            if not self.compatible_interpretation(equivalent, row):
                                backlog.append(
                                    dict(
                                        definition=key,
                                        existing=equivalent["id"],
                                        reason="interpretation_method_or_unit_mismatch",
                                    )
                                )
                                continue
                            # Expansion precedes Review. Never reinterpret an already
                            # started review under a different definition or reuse its score.
                            if (self.campaign.output / "review" / "request.json").exists():
                                raise ValueError("definition_replacement_after_review_started")
                            updated = dict(
                                equivalent,
                                definition=row["definition"],
                                basis_kind=row["basis_kind"],
                                definition_receipt_checksum=row["definition_receipt_checksum"],
                                **(
                                    {
                                        k: row[k]
                                        for k in (
                                            "dependency_assessment",
                                            "acquisition_allowed",
                                            "review_allowed",
                                        )
                                    }
                                    if "dependency_assessment" in row
                                    else {}
                                ),
                                **(
                                    dict(definition_receipt_key=key)
                                    if isinstance(self.policy, TargetExpansionPolicy)
                                    else {}
                                ),
                            )
                            if value.get("schema_version") in CONFIRMED_RECEIPTS:
                                updated.update(
                                    expected_method=row["expected_method"],
                                    expected_unit=row["expected_unit"],
                                )
                            rows[rows.index(equivalent)] = updated
                            replaced.append(
                                dict(
                                    id=equivalent["id"],
                                    definition=key,
                                    before=equivalent["definition"].model_dump(mode="json"),
                                    after=row["definition"].model_dump(mode="json"),
                                    reason="target_relation_rebound"
                                    if target_gap
                                    else "missing_interpretation_installed",
                                    review_required=True,
                                )
                            )
                            continue
                        merged.append(
                            dict(
                                definition=key,
                                existing=equivalent["id"],
                                reason="existing_examination_requires_no_duplicate_vote",
                                interpretation_changed=(
                                    equivalent["definition"] != row["definition"]
                                ),
                                disposition="existing_definition_kept_new_relation_not_applied",
                            )
                        )
                        continue
                    if len(rows) >= 64:
                        backlog.append(dict(definition=key, reason="requirement_capacity"))
                        continue
                    if isinstance(self.policy, TargetExpansionPolicy):
                        row = dict(row, definition_receipt_key=key)
                    rows.append(row)
                    added.append(row["id"])
                accepted[key] = value
            if proposal.no_addition_reason:
                backlog.append(
                    dict(reason="no_effective_definition", detail=proposal.no_addition_reason)
                )
        if candidate_only:
            return dict(
                rows=rows,
                found=found,
                accepted=accepted,
                added=added,
                merged=merged,
                backlog=backlog,
                replaced=replaced,
                material_fingerprint=material_fingerprint,
                evaluation_fingerprint=evaluation_fingerprint,
            )
        if until == "definitions":
            return self.checkpoint(
                "definitions",
                dict(accepted=accepted, rejected=backlog, materials=list(found.values())),
            )
        return self.commit_revision(
            rows, found, accepted, added, merged, backlog, replaced=replaced
        )

    def effective_interpretation_method(self, existing):
        """Original row/observation semantics, including legacy qualitative Vision.

        This read projection never fills or rewrites saved bytes, and never
        infers a quantitative method from a newly proposed definition.
        """
        observation = self.campaign.base.observations.get(existing["id"])
        if observation is not None:
            method, unit = observation.method, observation.unit
        elif existing["acquisition_method"] == "vision":
            method, unit = "live_image_observation", "qualitative"
        else:
            method, unit = existing.get("expected_method"), existing.get("expected_unit")
        return existing.get("expected_method", method), existing.get("expected_unit", unit)

    def compatible_interpretation(self, existing, proposed):
        """A same-name interpretation may fill a gap only for the same observation method."""
        if (
            tuple(existing["input_modalities"]) != tuple(proposed["input_modalities"])
            or existing.get("acquisition_method") != proposed["acquisition_method"]
        ):
            return False
        observation = self.campaign.base.observations.get(existing["id"])
        proposed_method = (proposed["expected_method"], proposed["expected_unit"])
        return self.effective_interpretation_method(existing) == proposed_method and (
            observation is None or (observation.method, observation.unit) == proposed_method
        )

    def commit_revision(
        self, rows, found, accepted, added, merged, backlog, *, stop=None, replaced=()
    ):
        self.control()
        inherited, inherited_proof = self.inherited_revision()
        if inherited_proof is not None:
            # Original parent owner has rechecked these receipts and materials;
            # keep that owner/identity rather than republishing its definitions.
            found = dict(found)
            for saved in inherited["materials"]:
                if saved["code"] in found and found[saved["code"]] != saved:
                    raise ValueError("definition_inherited_material_changed")
                found[saved["code"]] = saved
        # Only trusted source payloads supply numeric observations; bind method/unit
        # before freezing the rows. Do not mutate a saved plan during installation.
        for row in rows:
            if row["id"] in added and row.get("acquisition_method") == "source_measurement":
                if self.numeric_observation(row) is None and row["component"]:
                    row["missing_reason"] = "definition_measurement_method_unit_mismatch_or_missing"
        revision = dict(
            schema_version="campaign-requirement-revision/1.1.0",
            binding=self.binding,
            revision=1,
            query_plan=self.query_plan(),
            materials=list(found.values()),
            accepted=accepted,
            rows=json.loads(canonical(rows)),
            added=added,
            merged=merged,
            replaced=list(replaced),
            rejected=backlog,
            denominator_before=len(self.original_rows),
            denominator_after=len(rows),
            invalidated_interpretations=[],
            observation_reuse="same_input_method_only",
            expansion_stop=stop
            or (
                "single_bounded_revision_complete"
                if added or replaced
                else "no_new_valid_requirement"
            ),
            formal_publication=False,
        )
        if inherited_proof is not None:
            revision["inherited_definition_origin"] = inherited_proof
        if getattr(self.campaign, "local_control_amendment", None):
            revision["local_control_amendment"] = dict(
                checksum=self.campaign.local_control_amendment,
                original_local_limit=self.policy.max_operations,
                effective_local_limit=self.operation_limit,
                external_request_limit=self.campaign.protocol["max_new_requests"],
                deadline=self.campaign.protocol["deadline"],
            )
        if self.campaign.protocol.get("continuation"):
            revision["continuation"] = self.campaign.protocol["continuation"]
            revision["revision"] = 2
        if isinstance(self.policy, RequirementScopeExpansionPolicy):
            revision["review_source_scope"] = self.source_scope_record(revision)
        if isinstance(self.policy, AdaptiveGapExpansionPolicy):
            revision["gap_expansion"] = self.gap_expansion_summary
        self.control()
        for key in accepted:
            self.owner.read(key)
        with self.documents.session():
            self.control()
            self.documents.write("definition:revision", (revision,))
        return self.install(self.current())

    def numeric_observation(self, row):
        from .material import numeric, observation

        base = self.campaign.base
        if not row["component"]:
            return None
        c = next(c for c in base.manifest.components if c.id == row["component"])
        result = numeric(c, dict(base.prepared.payload_bytes))
        # The existing adapter does not attest a device measurement method. Only
        # source_numeric_read is implemented here, not arbitrary derived metrics.
        if result is None or (row["expected_method"], row["expected_unit"]) != (
            "source_numeric_read",
            result["unit"],
        ):
            return None
        return observation(row["id"], result, method="source_numeric_read", unit=result["unit"])

    def capture(self, record, reply, *, payload=None):
        from packages.llm_provider.provider import strict_json

        value = dict(
            record=record.model_dump(mode="json"),
            raw_text=reply.content.decode("utf-8", errors="replace"),
        )
        with self.documents.session():
            self.documents.write("definition:response:" + record.request_id, (value,))
            if isinstance(self.policy, OwnerLocatedExpansionPolicy):
                diagnostic = reply.diagnostic or {}
                failure = diagnostic.get("transport_failure", {})
                fields = (
                    "schema_version",
                    "category",
                    "stage",
                    "errno",
                    "tls_reason",
                    "certificate_verify_code",
                    "remote_execution",
                    "automatic_replay_authorized",
                )
                safe = {k: failure[k] for k in fields if k in failure}
                http = diagnostic.get("http_error", diagnostic)
                status = http.get("http_status")
                self.documents.write(
                    "definition:diagnostic:" + record.request_id,
                    (
                        dict(
                            request_id=record.request_id,
                            transport_failure=safe,
                            elapsed_seconds=record.latency_seconds,
                            http_status=status
                            if type(status) is int and 100 <= status <= 599
                            else None,
                        ),
                    ),
                )
            if record.error is None:
                if getattr(self.policy, "prompt_version", None) == "definition-draft-6.8.0":
                    from packages.llm_provider.definition_passages import decode

                    proposal, _ = decode(strict_json(reply.content), payload)
                    if digest(proposal) != record.output_checksum:
                        raise ValueError("definition_passage_capture_binding")
                else:
                    proposal = validate(self.proposal_model, strict_json(reply.content))
                self.documents.write(self.attempt_key("definition:confirmed-proposal"), (proposal,))

    def inherited_revision(self):
        """Explicit 1.79 successor retains parent definitions under their original owner."""
        if self.campaign.protocol.get("schema_version") not in {
            "multimodal-campaign-renewal/1.79.0",
            "multimodal-campaign-renewal/1.80.0",
        }:
            return {}, None
        parent = self.campaign.parent
        from .source_checkpoint import retained_tip

        tip = retained_tip(self.campaign.protocol)
        if str(parent.output) != tip["path"]:
            raise ValueError("definition_inherited_owner_binding")
        prior = parent.base.definition_current()
        if digest(prior) != tip["definition"]["revision"]:
            raise ValueError("definition_inherited_origin_changed")
        return prior, dict(path=tip["path"], revision=digest(prior), original_owner_retained=True)

    def source_scope_record(self, revision):
        from .material import requirement_source_refs

        refs = requirement_source_refs(revision["rows"])
        return dict(
            schema_version="definition-review-source-scope/1.0.0",
            policy="requirements-and-items/1.0.0",
            requirements_checksum=digest(revision["rows"]),
            source_refs=[ref.model_dump(mode="json") for ref in refs.values()],
            retrieval_history_checksum=digest(revision["materials"]),
            history_is_not_current_consumption_authority=True,
        )

    def required_materials(self, revision):
        if not isinstance(self.policy, RequirementScopeExpansionPolicy):
            return revision["materials"]
        from .material import requirement_source_refs

        refs = requirement_source_refs(revision["rows"])
        selected = []
        seen = set()
        for saved in revision["materials"]:
            if saved["code"] not in refs:
                continue
            source = validate(Source, saved["material"])
            if (
                source.reference != refs[saved["code"]]
                or saved["code"] in seen
                or saved["checksum"] != source.reference.checksum
            ):
                raise ValueError("definition_revision_source_changed")
            selected.append(saved)
            seen.add(saved["code"])
        return selected

    def install(self, revision):
        base = self.campaign.base
        if isinstance(self.policy, RequirementScopeExpansionPolicy):
            if revision.get("review_source_scope") != self.source_scope_record(revision):
                raise ValueError("definition_review_source_scope_binding")
        rows = [
            dict(
                r,
                definition=validate(TaskDefinition, r["definition"]),
                input_modalities=tuple(r["input_modalities"]),
                **(
                    {"dependency_paths": tuple(tuple(p) for p in r["dependency_paths"])}
                    if "dependency_paths" in r
                    else {}
                ),
                **(
                    {"quality_source": validate(Reference, r["quality_source"])}
                    if r.get("quality_source") is not None
                    else {}
                ),
            )
            for r in revision["rows"]
        ]
        component = (
            base
            if isinstance(self.policy, GenericConfirmedIndexPolicy)
            else self.owner.confirmed_observations
            if isinstance(self.policy, NumericInterpretationPublicPolicy)
            else getattr(base, "component_observations", None)
        )
        confirmed = [
            r
            for r in revision.get("replaced", ())
            if revision["accepted"][r["definition"]].get("schema_version") in CONFIRMED_RECEIPTS
        ]
        if confirmed:
            if len(confirmed) != 1 or component is None:
                raise ValueError("definition_confirmed_projection_owner_required")
            adoption = component.prepare_interpretation(self.owner, confirmed[0]["definition"])
            component.install_interpretation(self.owner, adoption, rows)
        else:
            base.ledger = rows
        base.definition_materials = {
            s["code"]: validate(Source, s["material"]) for s in self.required_materials(revision)
        }
        if isinstance(self.policy, RequirementScopeExpansionPolicy):
            base.definition_source_scope = self.policy.review_source_scope
        for row in base.ledger:
            if (
                row["id"] not in revision["added"]
                or not row["component"]
                or row["acquisition_method"] != "source_measurement"
            ):
                continue
            result = self.numeric_observation(row)
            if result is not None:
                base.observations[row["id"]] = result
        available = {v["component"] for v in base.views["views"] if v["status"] == "obtained"}
        self.campaign.targets = self.campaign.select_targets(base, available)
        if confirmed:
            retained = {r["id"] for r in confirmed}
            self.campaign.targets = tuple(t for t in self.campaign.targets if t not in retained)
        base.definition_revision = revision
        base.definition_current = self.current
        return revision

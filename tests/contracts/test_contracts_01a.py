"""Stage 01A stable identifiers, enums, transitions, and semantic negative cases."""

from __future__ import annotations

import json
from enum import StrEnum
from itertools import product

import pytest

import packages.contracts.enums as contract_enums
from packages.contracts import (
    ActorType,
    CapabilityAvailability,
    ClinicalEffect,
    CodeNamespace,
    ControlledTermCode,
    EvidenceObservationStatus,
    EvidenceSemantics,
    EvidenceValidity,
    ExecutionPolicy,
    GraphNodeStatus,
    GraphUpdateSource,
    IdentifierError,
    IncompatibleLegacyValue,
    IntendedUse,
    OfflineReviewStatus,
    OnlineTaskType,
    OpaqueEntityId,
    PlanAction,
    ProposalKind,
    QCStatus,
    RunStatus,
    SemanticConstraintError,
    StableCodeKind,
    StateMachine,
    StateTransitionError,
    StepStatus,
    ToolCatalogStatus,
    ToolExecutionStatus,
    ToolExtensionJobStatus,
    ToolImplementationIdentity,
    ToolValidationStatus,
    allowed_proposals_for,
    can_transition,
    expand_legacy_node_state,
    generate_opaque_entity_id,
    make_controlled_term_code,
    make_opaque_entity_id,
    make_stable_code,
    normalize_code_segment,
    parse_compatible,
    require_graph_node_status,
    require_transition,
    state_machine_of,
    validate_actor_proposal,
    validate_evidence_semantics,
    validate_graph_status_source,
    validate_new_implementation_after_terminal,
    validate_opaque_entity_id,
    validate_stable_code,
    validate_tool_use,
)

NON_DIRECTIONAL_OBSERVATIONS = (
    EvidenceObservationStatus.UNOBSERVED,
    EvidenceObservationStatus.UNOBSERVED_TOOL_GAP,
    EvidenceObservationStatus.INCONCLUSIVE,
    EvidenceObservationStatus.NOT_APPLICABLE,
)
NON_NONE_EFFECTS = tuple(effect for effect in ClinicalEffect if effect is not ClinicalEffect.NONE)
NON_SUCCEEDED_TOOL_STATES = tuple(
    status for status in ToolExecutionStatus if status is not ToolExecutionStatus.SUCCEEDED
)
NON_EXECUTABLE_CATALOG_STATUSES = tuple(
    status for status in ToolCatalogStatus if status is not ToolCatalogStatus.ACTIVE
)
TERMINAL_CATALOG_STATUSES = (
    ToolCatalogStatus.QUARANTINED,
    ToolCatalogStatus.FAILED,
    ToolCatalogStatus.DEPRECATED,
)


def implementation_identity(
    token: str, version: str, checksum_character: str
) -> ToolImplementationIdentity:
    return ToolImplementationIdentity(
        implementation_code=make_opaque_entity_id(CodeNamespace.IMPLEMENTATION, token),
        version=version,
        checksum=f"sha256:{checksum_character * 64}",
    )


def test_controlled_term_generation_validation_and_serialization() -> None:
    code = make_controlled_term_code(CodeNamespace.CLAIM, "poag suspect", "a1 open-angle")
    assert isinstance(code, ControlledTermCode)
    assert code == "CLAIM:POAG_SUSPECT:A1_OPEN-ANGLE"
    assert code.namespace is CodeNamespace.CLAIM
    assert code.kind is StableCodeKind.CONTROLLED_TERM
    assert code.parts == ("POAG_SUSPECT", "A1_OPEN-ANGLE")
    assert make_stable_code(CodeNamespace.CLAIM, "poag suspect") == "CLAIM:POAG_SUSPECT"
    assert str(validate_stable_code(str(code), CodeNamespace.CLAIM)) == str(code)


def test_opaque_entity_ids_require_anonymized_or_generated_tokens() -> None:
    anonymous = make_opaque_entity_id(CodeNamespace.CASE, "ANON_000001")
    generated = generate_opaque_entity_id(CodeNamespace.RUN)
    assert isinstance(anonymous, OpaqueEntityId)
    assert anonymous.kind is StableCodeKind.OPAQUE_ENTITY_ID
    assert validate_opaque_entity_id(str(anonymous)) is not None
    assert generated.startswith("RUN:OID_")
    assert generated.namespace is CodeNamespace.RUN


@pytest.mark.parametrize(
    "value",
    (
        "123",
        "claim:POAG",
        "UNKNOWN:THING",
        "CASE:/opt/data/patient",
        "CASE:file://patient",
        "CASE:患者",
        "CASE:MRN_123456",
        "CASE:JOHN_DOE_123",
    ),
)
def test_stable_code_rejects_invalid_paths_uris_and_raw_entity_identifiers(value: str) -> None:
    with pytest.raises(IdentifierError):
        validate_stable_code(value)


@pytest.mark.parametrize(
    "unsafe_label",
    (
        "/opt/data/patient",
        "../patient",
        "s3://bucket/patient",
        "urn:patient",
        "C:patient",
        "patient\\scan",
    ),
)
def test_controlled_label_normalization_rejects_paths_and_uris(unsafe_label: str) -> None:
    with pytest.raises(IdentifierError, match="path or URI"):
        normalize_code_segment(unsafe_label)


def test_code_kind_and_namespace_cannot_be_crossed() -> None:
    with pytest.raises(IdentifierError, match="opaque entity"):
        make_stable_code(CodeNamespace.CASE, "ANON_000001")
    with pytest.raises(IdentifierError, match="controlled term"):
        make_opaque_entity_id(CodeNamespace.CLAIM, "OID_123456")
    with pytest.raises(IdentifierError, match="expected namespace"):
        validate_stable_code("GRAPH:OID_123456", CodeNamespace.CASE)
    with pytest.raises(IdentifierError, match="expected kind"):
        validate_stable_code("CASE:ANON_000001", expected_kind=StableCodeKind.CONTROLLED_TERM)


def test_enum_codes_round_trip_without_arbitrary_string_fallback() -> None:
    assert GraphNodeStatus(GraphNodeStatus.SUPPORTED.value) is GraphNodeStatus.SUPPORTED
    assert ToolValidationStatus("smoke_tested") is ToolValidationStatus.SMOKE_TESTED
    serialized = json.dumps(
        {
            "node": GraphNodeStatus.SUPPORTED,
            "catalog_status": ToolCatalogStatus.ACTIVE,
            "validation": ToolValidationStatus.SMOKE_TESTED,
        }
    )
    assert json.loads(serialized) == {
        "node": "SUPPORTED",
        "catalog_status": "active",
        "validation": "smoke_tested",
    }
    with pytest.raises(ValueError):
        GraphNodeStatus("supported")
    with pytest.raises(ValueError):
        ToolValidationStatus("whatever_the_agent_said")


def test_canonical_enums_have_no_hidden_value_aliases() -> None:
    assert not hasattr(contract_enums, "ToolAdmissionLevel")
    assert not hasattr(contract_enums, "ToolOnboardingStatus")
    assert not hasattr(contract_enums, "ToolExtensionStatus")
    for candidate in vars(contract_enums).values():
        if isinstance(candidate, type) and issubclass(candidate, StrEnum):
            assert len(candidate.__members__) == len(candidate), candidate.__name__


def test_online_run_has_no_waiting_for_human_state() -> None:
    assert "WAITING_FOR_HUMAN" not in {state.value for state in RunStatus}
    assert "HUMAN_REVIEW" not in {state.value for state in RunStatus}
    assert PlanAction.EXPERT_ASSESSMENT.value == OnlineTaskType.EXPERT_ASSESSMENT.value
    assert (
        OfflineReviewStatus.OFFLINE_REVIEW_PENDING.value != OnlineTaskType.EXPERT_ASSESSMENT.value
    )


def test_tool_state_domains_have_unique_ownership() -> None:
    assert state_machine_of(ToolCatalogStatus.ACTIVE) is StateMachine.TOOL_CATALOG
    assert state_machine_of(ToolValidationStatus.SMOKE_TESTED) is StateMachine.TOOL_VALIDATION
    assert state_machine_of(ToolExtensionJobStatus.BENCHMARKING) is StateMachine.TOOL_EXTENSION_JOB
    with pytest.raises(TypeError, match="not a transition state"):
        state_machine_of(IntendedUse.RESEARCH_ONLY)


def test_each_execution_state_machine_has_unique_ownership() -> None:
    assert state_machine_of(GraphNodeStatus.PENDING) is StateMachine.GRAPH_NODE
    assert state_machine_of(RunStatus.PLANNING) is StateMachine.RUN
    assert state_machine_of(StepStatus.RUNNING) is StateMachine.STEP
    assert state_machine_of(ToolExecutionStatus.RUNNING) is StateMachine.TOOL_EXECUTION
    assert state_machine_of(QCStatus.PENDING) is StateMachine.QC
    assert (
        state_machine_of(OfflineReviewStatus.OFFLINE_REVIEW_PENDING) is StateMachine.OFFLINE_REVIEW
    )


def test_legal_illegal_cross_machine_and_idempotent_transitions() -> None:
    assert can_transition(RunStatus.CREATED, RunStatus.NORMALIZING)
    assert require_transition(RunStatus.PLANNING, RunStatus.PLANNING) is RunStatus.PLANNING
    assert not can_transition(RunStatus.COMPLETED_SUPPORTED, RunStatus.PLANNING)
    with pytest.raises(StateTransitionError, match="cannot mix"):
        can_transition(RunStatus.EXECUTING, StepStatus.SUCCEEDED)
    with pytest.raises(StateTransitionError, match="illegal"):
        require_transition(RunStatus.CREATED, RunStatus.COMPLETED_SUPPORTED)


def test_all_declared_transition_states_support_idempotent_retry() -> None:
    enum_types = (
        GraphNodeStatus,
        RunStatus,
        StepStatus,
        ToolExecutionStatus,
        QCStatus,
        ToolCatalogStatus,
        ToolValidationStatus,
        OfflineReviewStatus,
        ToolExtensionJobStatus,
    )
    for enum_type in enum_types:
        for state in enum_type:
            assert can_transition(state, state), f"missing transition ownership for {state}"


def test_tool_catalog_and_validation_gates_are_independent() -> None:
    assert can_transition(ToolCatalogStatus.CANDIDATE, ToolCatalogStatus.REGISTERED)
    assert can_transition(ToolCatalogStatus.REGISTERED, ToolCatalogStatus.ACTIVE)
    assert not can_transition(ToolCatalogStatus.CANDIDATE, ToolCatalogStatus.ACTIVE)
    assert can_transition(ToolValidationStatus.UNVALIDATED, ToolValidationStatus.SMOKE_TESTED)
    assert can_transition(ToolValidationStatus.SMOKE_TESTED, ToolValidationStatus.BENCHMARK_PASSED)
    assert can_transition(
        ToolValidationStatus.BENCHMARK_PASSED,
        ToolValidationStatus.AUTO_VALIDATED_RESEARCH,
    )
    assert not can_transition(
        ToolValidationStatus.SMOKE_TESTED,
        ToolValidationStatus.AUTO_VALIDATED_RESEARCH,
    )


@pytest.mark.parametrize(
    ("terminal_status", "target_status"),
    tuple(
        (terminal_status, target_status)
        for terminal_status, target_status in product(
            TERMINAL_CATALOG_STATUSES, tuple(ToolCatalogStatus)
        )
        if terminal_status is not target_status
    ),
)
def test_terminal_catalog_versions_cannot_transition_to_any_other_status(
    terminal_status: ToolCatalogStatus, target_status: ToolCatalogStatus
) -> None:
    assert not can_transition(terminal_status, target_status)
    with pytest.raises(StateTransitionError, match="illegal"):
        require_transition(terminal_status, target_status)


@pytest.mark.parametrize("terminal_status", TERMINAL_CATALOG_STATUSES)
def test_terminal_catalog_version_requires_a_new_candidate_implementation(
    terminal_status: ToolCatalogStatus,
) -> None:
    previous = implementation_identity("OID_IMPL0001", "1.0.0", "a")
    replacement = implementation_identity("OID_IMPL0002", "1.0.1", "b")
    accepted = validate_new_implementation_after_terminal(
        terminal_status,
        previous,
        ToolCatalogStatus.CANDIDATE,
        replacement,
    )
    assert accepted is replacement
    assert previous.implementation_code == "IMPLEMENTATION:OID_IMPL0001"


@pytest.mark.parametrize(
    ("terminal_status", "new_status"),
    tuple(
        product(
            TERMINAL_CATALOG_STATUSES,
            tuple(
                status for status in ToolCatalogStatus if status is not ToolCatalogStatus.CANDIDATE
            ),
        )
    ),
)
def test_replacement_implementation_must_start_from_candidate(
    terminal_status: ToolCatalogStatus, new_status: ToolCatalogStatus
) -> None:
    with pytest.raises(SemanticConstraintError, match="start from candidate"):
        validate_new_implementation_after_terminal(
            terminal_status,
            implementation_identity("OID_IMPL0001", "1.0.0", "a"),
            new_status,
            implementation_identity("OID_IMPL0002", "1.0.1", "b"),
        )


@pytest.mark.parametrize(
    ("terminal_status", "unchanged_field"),
    tuple(product(TERMINAL_CATALOG_STATUSES, ("implementation_code", "version", "checksum"))),
)
def test_replacement_requires_new_code_version_and_checksum(
    terminal_status: ToolCatalogStatus, unchanged_field: str
) -> None:
    previous = implementation_identity("OID_IMPL0001", "1.0.0", "a")
    replacement_values = {
        "token": "OID_IMPL0002",
        "version": "1.0.1",
        "checksum_character": "b",
    }
    if unchanged_field == "implementation_code":
        replacement_values["token"] = "OID_IMPL0001"
    elif unchanged_field == "version":
        replacement_values["version"] = "1.0.0"
    else:
        replacement_values["checksum_character"] = "a"
    replacement = implementation_identity(**replacement_values)

    with pytest.raises(SemanticConstraintError, match=f"new {unchanged_field}"):
        validate_new_implementation_after_terminal(
            terminal_status,
            previous,
            ToolCatalogStatus.CANDIDATE,
            replacement,
        )


@pytest.mark.parametrize(
    "non_terminal_status",
    (ToolCatalogStatus.CANDIDATE, ToolCatalogStatus.REGISTERED, ToolCatalogStatus.ACTIVE),
)
def test_replacement_rule_cannot_be_used_to_bypass_non_terminal_lifecycle(
    non_terminal_status: ToolCatalogStatus,
) -> None:
    with pytest.raises(SemanticConstraintError, match="terminal implementation"):
        validate_new_implementation_after_terminal(
            non_terminal_status,
            implementation_identity("OID_IMPL0001", "1.0.0", "a"),
            ToolCatalogStatus.CANDIDATE,
            implementation_identity("OID_IMPL0002", "1.0.1", "b"),
        )


def test_step_failure_and_timeout_require_explicit_retry_scheduling() -> None:
    for failure in (StepStatus.FAILED, StepStatus.TIMED_OUT):
        assert can_transition(StepStatus.RUNNING, failure)
        assert can_transition(failure, StepStatus.RETRY_SCHEDULED)
        assert not can_transition(failure, StepStatus.READY)
    assert not can_transition(StepStatus.RUNNING, StepStatus.RETRY_SCHEDULED)
    assert can_transition(StepStatus.RETRY_SCHEDULED, StepStatus.READY)


def test_success_qc_validity_and_clinical_effect_are_ordered_gates() -> None:
    accepted = validate_evidence_semantics(
        EvidenceSemantics(
            observation=EvidenceObservationStatus.POSITIVE,
            validity=EvidenceValidity.VALID,
            clinical_effect=ClinicalEffect.SUPPORTS,
            tool_execution=ToolExecutionStatus.SUCCEEDED,
            qc=QCStatus.PASS,
        )
    )
    assert accepted.clinical_effect is ClinicalEffect.SUPPORTS


@pytest.mark.parametrize(
    ("observation", "effect"), tuple(product(NON_DIRECTIONAL_OBSERVATIONS, NON_NONE_EFFECTS))
)
def test_non_directional_observations_reject_every_clinical_effect(
    observation: EvidenceObservationStatus, effect: ClinicalEffect
) -> None:
    capability = (
        CapabilityAvailability.UNAVAILABLE_CAPABILITY
        if observation is EvidenceObservationStatus.UNOBSERVED_TOOL_GAP
        else CapabilityAvailability.AVAILABLE
    )
    with pytest.raises(SemanticConstraintError):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=observation,
                validity=EvidenceValidity.VALID,
                clinical_effect=effect,
                capability=capability,
                tool_execution=ToolExecutionStatus.SUCCEEDED,
                qc=QCStatus.PASS,
            )
        )


@pytest.mark.parametrize(
    ("observation", "validity", "effect"),
    tuple(
        product(
            (EvidenceObservationStatus.POSITIVE, EvidenceObservationStatus.NEGATIVE),
            (
                EvidenceValidity.UNASSESSED,
                EvidenceValidity.INVALID,
                EvidenceValidity.INCONCLUSIVE,
            ),
            NON_NONE_EFFECTS,
        )
    ),
)
def test_non_valid_directional_observations_reject_every_clinical_effect(
    observation: EvidenceObservationStatus,
    validity: EvidenceValidity,
    effect: ClinicalEffect,
) -> None:
    with pytest.raises(SemanticConstraintError, match="only VALID"):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=observation,
                validity=validity,
                clinical_effect=effect,
                tool_execution=ToolExecutionStatus.SUCCEEDED,
                qc=QCStatus.PASS,
            )
        )


@pytest.mark.parametrize("tool_status", NON_SUCCEEDED_TOOL_STATES)
def test_every_non_succeeded_tool_state_rejects_qc_pass(
    tool_status: ToolExecutionStatus,
) -> None:
    with pytest.raises(SemanticConstraintError, match="cannot have QC PASS"):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=EvidenceObservationStatus.UNOBSERVED,
                validity=EvidenceValidity.UNASSESSED,
                clinical_effect=ClinicalEffect.NONE,
                tool_execution=tool_status,
                qc=QCStatus.PASS,
            )
        )


@pytest.mark.parametrize(
    ("tool_status", "observation"),
    tuple(
        product(
            NON_SUCCEEDED_TOOL_STATES,
            (EvidenceObservationStatus.POSITIVE, EvidenceObservationStatus.NEGATIVE),
        )
    ),
)
def test_every_non_succeeded_tool_state_rejects_directional_observations(
    tool_status: ToolExecutionStatus, observation: EvidenceObservationStatus
) -> None:
    with pytest.raises(SemanticConstraintError, match="has no observation"):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=observation,
                validity=EvidenceValidity.UNASSESSED,
                clinical_effect=ClinicalEffect.NONE,
                tool_execution=tool_status,
                qc=QCStatus.UNASSESSED,
            )
        )


@pytest.mark.parametrize("effect", NON_NONE_EFFECTS)
def test_qc_failure_rejects_every_clinical_effect(effect: ClinicalEffect) -> None:
    with pytest.raises(SemanticConstraintError, match="QC FAIL"):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=EvidenceObservationStatus.POSITIVE,
                validity=EvidenceValidity.INVALID,
                clinical_effect=effect,
                tool_execution=ToolExecutionStatus.SUCCEEDED,
                qc=QCStatus.FAIL,
            )
        )


def test_unavailable_capability_requires_tool_gap_observation() -> None:
    with pytest.raises(SemanticConstraintError, match="requires UNOBSERVED_TOOL_GAP"):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=EvidenceObservationStatus.UNOBSERVED,
                validity=EvidenceValidity.UNASSESSED,
                clinical_effect=ClinicalEffect.NONE,
                capability=CapabilityAvailability.UNAVAILABLE_CAPABILITY,
            )
        )

    with pytest.raises(SemanticConstraintError, match="requires UNAVAILABLE_CAPABILITY"):
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=EvidenceObservationStatus.UNOBSERVED_TOOL_GAP,
                validity=EvidenceValidity.UNASSESSED,
                clinical_effect=ClinicalEffect.NONE,
                capability=CapabilityAvailability.AVAILABLE,
            )
        )


@pytest.mark.parametrize(
    ("catalog_status", "validation", "intended_use"),
    tuple(
        product(
            NON_EXECUTABLE_CATALOG_STATUSES,
            tuple(ToolValidationStatus),
            tuple(IntendedUse),
        )
    ),
)
def test_every_non_active_catalog_status_is_non_executable(
    catalog_status: ToolCatalogStatus,
    validation: ToolValidationStatus,
    intended_use: IntendedUse,
) -> None:
    with pytest.raises(SemanticConstraintError, match="cannot execute"):
        validate_tool_use(
            catalog_status,
            validation,
            intended_use,
            ExecutionPolicy.RESEARCH,
            decisive_evidence=False,
        )


@pytest.mark.parametrize(
    ("catalog_status", "validation", "intended_use", "decisive"),
    tuple(
        product(
            tuple(ToolCatalogStatus),
            tuple(ToolValidationStatus),
            tuple(IntendedUse),
            (False, True),
        )
    ),
)
def test_every_clinical_deployment_combination_is_rejected(
    catalog_status: ToolCatalogStatus,
    validation: ToolValidationStatus,
    intended_use: IntendedUse,
    decisive: bool,
) -> None:
    with pytest.raises(SemanticConstraintError, match="no clinical approval"):
        validate_tool_use(
            catalog_status,
            validation,
            intended_use,
            ExecutionPolicy.CLINICAL_DEPLOYMENT,
            decisive_evidence=decisive,
        )


@pytest.mark.parametrize(
    ("validation", "intended_use"),
    tuple(
        product(
            (
                ToolValidationStatus.SMOKE_TESTED,
                ToolValidationStatus.BENCHMARK_PASSED,
            ),
            tuple(IntendedUse),
        )
    ),
)
def test_smoke_and_benchmark_statuses_cannot_provide_decisive_evidence(
    validation: ToolValidationStatus, intended_use: IntendedUse
) -> None:
    with pytest.raises(SemanticConstraintError, match="auto_validated_research"):
        validate_tool_use(
            ToolCatalogStatus.ACTIVE,
            validation,
            intended_use,
            ExecutionPolicy.RESEARCH,
            decisive_evidence=True,
        )


def test_active_research_tool_execution_rules_have_positive_cases() -> None:
    for validation in (
        ToolValidationStatus.SMOKE_TESTED,
        ToolValidationStatus.BENCHMARK_PASSED,
    ):
        validate_tool_use(
            ToolCatalogStatus.ACTIVE,
            validation,
            IntendedUse.RESEARCH_ONLY,
            ExecutionPolicy.RESEARCH,
            decisive_evidence=False,
        )
    validate_tool_use(
        ToolCatalogStatus.ACTIVE,
        ToolValidationStatus.AUTO_VALIDATED_RESEARCH,
        IntendedUse.RESEARCH_ONLY,
        ExecutionPolicy.RESEARCH,
        decisive_evidence=True,
    )


def test_active_but_unvalidated_tool_cannot_execute() -> None:
    with pytest.raises(SemanticConstraintError, match="unvalidated"):
        validate_tool_use(
            ToolCatalogStatus.ACTIVE,
            ToolValidationStatus.UNVALIDATED,
            IntendedUse.RESEARCH_ONLY,
            ExecutionPolicy.RESEARCH,
            decisive_evidence=False,
        )


@pytest.mark.parametrize(
    ("actor", "proposal"), tuple(product(tuple(ActorType), tuple(ProposalKind)))
)
def test_actor_proposal_cartesian_product_matches_allow_matrix(
    actor: ActorType, proposal: ProposalKind
) -> None:
    if proposal in allowed_proposals_for(actor):
        validate_actor_proposal(actor, proposal)
    else:
        with pytest.raises(SemanticConstraintError, match="cannot submit"):
            validate_actor_proposal(actor, proposal)


def test_graph_delta_candidate_validation_and_application_have_distinct_owners() -> None:
    assert ProposalKind.GRAPH_DELTA_CANDIDATE in allowed_proposals_for(ActorType.VERIFICATION_AGENT)
    for actor in (
        ActorType.PLANNER,
        ActorType.TRACE_AGENT,
        ActorType.TOOL_AGENT,
        ActorType.EXPERT_AGENT,
    ):
        assert ProposalKind.GRAPH_DELTA_CANDIDATE not in allowed_proposals_for(actor)
    assert ProposalKind.GRAPH_DELTA_VALIDATION in allowed_proposals_for(ActorType.RULE_ENGINE)
    assert ProposalKind.GRAPH_DELTA_APPLICATION in allowed_proposals_for(ActorType.GRAPH_ENGINE)


def test_graph_node_status_cannot_be_derived_from_run_or_extension_status() -> None:
    assert require_graph_node_status(GraphNodeStatus.UNKNOWN) is GraphNodeStatus.UNKNOWN
    for foreign_state in (RunStatus.EXECUTING, ToolExtensionJobStatus.BENCHMARKING):
        with pytest.raises(StateTransitionError, match="GraphNodeStatus"):
            require_graph_node_status(foreign_state)

    for source in (
        GraphUpdateSource.RUN_STATUS,
        GraphUpdateSource.TOOL_EXTENSION_JOB,
        GraphUpdateSource.TOOL_GAP,
        GraphUpdateSource.TOOL_EXECUTION,
        GraphUpdateSource.QC_RESULT,
        GraphUpdateSource.EXPERT_ASSESSMENT,
    ):
        with pytest.raises(SemanticConstraintError, match="cannot directly derive"):
            validate_graph_status_source(GraphNodeStatus.REFUTED, source)

    validate_graph_status_source(GraphNodeStatus.REFUTED, GraphUpdateSource.VALIDATED_EVIDENCE)


def test_legacy_mapping_is_conservative_and_cannot_revive_online_human_wait() -> None:
    assert parse_compatible(GraphNodeStatus, "Supported") is GraphNodeStatus.SUPPORTED
    assert parse_compatible(ToolValidationStatus, "Validated") is ToolValidationStatus.SMOKE_TESTED
    assert parse_compatible(ToolValidationStatus, "validated") is ToolValidationStatus.SMOKE_TESTED
    assert parse_compatible(ToolCatalogStatus, "Candidate") is ToolCatalogStatus.CANDIDATE
    assert parse_compatible(PlanAction, "HUMAN_REVIEW") is PlanAction.EXPERT_ASSESSMENT
    with pytest.raises(IncompatibleLegacyValue, match="cannot be migrated"):
        parse_compatible(RunStatus, "WAITING_FOR_HUMAN")
    with pytest.raises(IncompatibleLegacyValue, match="no safe"):
        parse_compatible(ProposalKind, "GRAPH_DELTA")


def test_legacy_invalid_node_state_splits_validity_from_graph_outcome() -> None:
    expanded = expand_legacy_node_state("Invalid")
    assert expanded.graph_node_status is GraphNodeStatus.UNKNOWN
    assert expanded.evidence_validity is EvidenceValidity.INVALID

"""Stage 01B Pydantic v2 contracts, schemas, examples, and boundary tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pydantic
import pytest
from pydantic import ValidationError

import packages.contracts as contracts
from packages.contracts import (
    CONTRACT_MODEL_BY_TYPE,
    CONTRACT_MODELS,
    ActorType,
    BaseContract,
    GraphDelta,
    IntendedUse,
    MessageEnvelope,
    ProposalKind,
    ScopeReference,
    ToolCatalogStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExtensionJobStatus,
    ToolOnboardingDecision,
    ToolOnboardingJob,
    ToolValidationStatus,
    ValidationResult,
)
from packages.contracts.example_data import (
    CHECKSUM_B,
    VALID_EXAMPLES,
    invalid_state_example,
    policy,
    scope,
    version_mismatch_example,
)
from packages.contracts.schema_tools import EXAMPLE_ROOT, SCHEMA_ROOT, rendered_files

EXPECTED_CONTRACT_TYPES = frozenset(
    {
        "Artifact",
        "CapabilityGap",
        "CaseBundle",
        "CounterfactualPair",
        "Episode",
        "ExpertAssessment",
        "GECEGInstance",
        "GECEGTemplate",
        "GraphDelta",
        "MessageEnvelope",
        "PlannerAction",
        "PlannerFeedback",
        "PlannerObservation",
        "PlannerState",
        "ToolExecutionRequest",
        "ToolExecutionResult",
        "ToolOnboardingDecision",
        "ToolOnboardingJob",
        "ToolRequirementDocument",
        "TrajectoryStep",
        "ValidationResult",
    }
)
CONTRACT_TYPES = tuple(sorted(EXPECTED_CONTRACT_TYPES))
BASE_REQUIRED_FIELDS = {
    "schema_version",
    "contract_type",
    "provenance",
    "scope",
    "laterality",
    "coordinate_space",
    "unit",
}


def example(contract_type: str) -> dict[str, object]:
    return deepcopy(VALID_EXAMPLES[contract_type])


def test_pydantic_v2_is_the_locked_runtime_contract_dependency() -> None:
    major = int(pydantic.__version__.partition(".")[0])
    assert major == 2


def test_public_registry_contains_every_required_01b_contract_once() -> None:
    assert frozenset(CONTRACT_MODEL_BY_TYPE) == EXPECTED_CONTRACT_TYPES
    assert len(CONTRACT_MODELS) == len(set(CONTRACT_MODELS)) == 21
    for contract_type, model in CONTRACT_MODEL_BY_TYPE.items():
        assert getattr(contracts, contract_type) is model
        assert issubclass(model, BaseContract)


def test_tool_contracts_reuse_the_four_frozen_01a_domains_exactly() -> None:
    for model in (
        ToolExecutionRequest,
        ToolExecutionResult,
        ToolOnboardingJob,
        ToolOnboardingDecision,
    ):
        assert model.model_fields["catalog_status"].annotation is ToolCatalogStatus
        assert model.model_fields["validation_status"].annotation is ToolValidationStatus
        assert model.model_fields["intended_use"].annotation is IntendedUse
    for model in (ToolOnboardingJob, ToolOnboardingDecision):
        assert model.model_fields["extension_job_status"].annotation is ToolExtensionJobStatus
        assert "onboarding_status" not in model.model_fields

    for forbidden in ("ToolAdmissionLevel", "ToolOnboardingStatus", "ToolExtensionStatus"):
        assert not hasattr(contracts, forbidden)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_every_valid_example_file_is_parsed_by_its_real_model_and_round_trips(
    contract_type: str,
) -> None:
    path = EXAMPLE_ROOT / f"{contract_type}.valid.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    model_type = CONTRACT_MODEL_BY_TYPE[contract_type]
    parsed = model_type.model_validate(document)
    reparsed = model_type.model_validate_json(parsed.model_dump_json())
    assert reparsed == parsed
    assert parsed.schema_version == "1.0"
    assert parsed.contract_type == contract_type


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_every_invalid_state_example_file_is_rejected_by_its_real_model(
    contract_type: str,
) -> None:
    path = EXAMPLE_ROOT / f"{contract_type}.invalid-state.json"
    with pytest.raises(ValidationError):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_every_version_mismatch_example_file_is_rejected_by_its_real_model(
    contract_type: str,
) -> None:
    path = EXAMPLE_ROOT / f"{contract_type}.version-mismatch.json"
    with pytest.raises(ValidationError, match="schema_version"):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_every_contract_rejects_missing_schema_version_and_unknown_fields(
    contract_type: str,
) -> None:
    model_type = CONTRACT_MODEL_BY_TYPE[contract_type]
    missing_version = example(contract_type)
    missing_version.pop("schema_version")
    with pytest.raises(ValidationError, match="schema_version"):
        model_type.model_validate(missing_version)

    unknown_field = example(contract_type)
    unknown_field["uncontrolled_state"] = "SUPPORTED"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model_type.model_validate(unknown_field)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_every_contract_rejects_a_wrong_contract_type(contract_type: str) -> None:
    document = example(contract_type)
    document["contract_type"] = "WrongContract"
    with pytest.raises(ValidationError, match="contract_type"):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate(document)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
@pytest.mark.parametrize(
    "extensions",
    (
        {"status": "SUPPORTED"},
        {"vendor": {"qc": "PASS"}},
        {"graph-delta": {"op": "ADD_NODE"}},
        {"audit": {"chain_of_thought": "hidden"}},
    ),
)
def test_extensions_cannot_control_state_or_hide_reasoning(
    contract_type: str, extensions: dict[str, object]
) -> None:
    document = example(contract_type)
    document["extensions"] = extensions
    with pytest.raises(ValidationError, match="may not control state or hidden reasoning"):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate(document)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_explicit_non_controlling_extensions_round_trip(contract_type: str) -> None:
    document = example(contract_type)
    document["extensions"] = {"vendor_metadata": {"score": 0.25, "label": "synthetic"}}
    parsed = CONTRACT_MODEL_BY_TYPE[contract_type].model_validate(document)
    assert parsed.extensions["vendor_metadata"] == {"score": 0.25, "label": "synthetic"}


@pytest.mark.parametrize(
    ("scope_document", "message"),
    (
        (scope("CASE"), "requires case_code"),
        (scope("GLOBAL", case_code="CASE:ANON_CASE0001"), "forbids references"),
        (scope("STEP", run_id="RUN:OID_RUN00001"), "requires step_id"),
        (
            scope("TOOL_EXTENSION", task_id="TASK:OID_TASK0001", step_id="STEP:OID_STEP0001"),
            "forbids references",
        ),
    ),
)
def test_scope_reference_requires_only_its_owned_references(
    scope_document: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ScopeReference.model_validate(scope_document)


@pytest.mark.parametrize(
    "contract_type",
    tuple(name for name in CONTRACT_TYPES if name != "GECEGTemplate"),
)
def test_contracts_reject_global_scope_when_their_domain_is_not_global(
    contract_type: str,
) -> None:
    document = example(contract_type)
    document["scope"] = scope("GLOBAL")
    with pytest.raises(ValidationError, match="does not allow GLOBAL scope"):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate(document)


def test_global_template_rejects_case_scope() -> None:
    document = example("GECEGTemplate")
    document["scope"] = scope("CASE", case_code="CASE:ANON_CASE0001")
    with pytest.raises(ValidationError, match="does not allow CASE scope"):
        CONTRACT_MODEL_BY_TYPE["GECEGTemplate"].model_validate(document)


def test_message_envelope_parses_the_declared_real_payload_model() -> None:
    parsed = MessageEnvelope.model_validate(example("MessageEnvelope"))
    assert parsed.payload_contract_type == "PlannerAction"

    mismatch = example("MessageEnvelope")
    mismatch["payload_contract_type"] = "PlannerState"
    with pytest.raises(ValidationError, match="must match"):
        MessageEnvelope.model_validate(mismatch)

    invalid_payload = example("MessageEnvelope")
    invalid_payload["payload"].pop("schema_version")  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="schema_version"):
        MessageEnvelope.model_validate(invalid_payload)

    wrong_payload_graph = example("MessageEnvelope")
    wrong_payload_graph["payload"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    wrong_payload_graph["payload"]["scope"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="message scope.graph_code"):
        MessageEnvelope.model_validate(wrong_payload_graph)


def test_expert_assessment_cannot_carry_or_disguise_graph_delta_or_hidden_thought() -> None:
    for field_name, value in (
        ("graph_delta", example("GraphDelta")),
        ("chain_of_thought", "hidden reasoning"),
    ):
        document = example("ExpertAssessment")
        document[field_name] = value
        with pytest.raises(ValidationError, match="extra_forbidden"):
            CONTRACT_MODEL_BY_TYPE["ExpertAssessment"].model_validate(document)

    disguised = example("ExpertAssessment")
    disguised["extensions"] = {"vendor": {"graph_delta_candidate": example("GraphDelta")}}
    with pytest.raises(ValidationError, match="may not control state"):
        CONTRACT_MODEL_BY_TYPE["ExpertAssessment"].model_validate(disguised)


@pytest.mark.parametrize(
    "catalog_status",
    tuple(status.value for status in ToolCatalogStatus if status is not ToolCatalogStatus.ACTIVE),
)
def test_tool_request_binds_non_active_catalog_rejection(catalog_status: str) -> None:
    document = example("ToolExecutionRequest")
    document["catalog_status"] = catalog_status
    with pytest.raises(ValidationError, match="cannot execute"):
        ToolExecutionRequest.model_validate(document)


@pytest.mark.parametrize("validation_status", ("smoke_tested", "benchmark_passed"))
def test_tool_request_binds_decisive_evidence_gate(validation_status: str) -> None:
    document = example("ToolExecutionRequest")
    document["validation_status"] = validation_status
    with pytest.raises(ValidationError, match="auto_validated_research"):
        ToolExecutionRequest.model_validate(document)


@pytest.mark.parametrize(
    "execution_status",
    (
        "REQUESTED",
        "DRY_RUN_VALIDATED",
        "QUEUED",
        "RUNNING",
        "FAILED",
        "TIMED_OUT",
        "CANCELLED",
        "BLOCKED",
    ),
)
def test_tool_result_binds_non_succeeded_qc_pass_rejection(execution_status: str) -> None:
    document = example("ToolExecutionResult")
    document["status"] = execution_status
    with pytest.raises(ValidationError, match="cannot have QC PASS"):
        ToolExecutionResult.model_validate(document)


def test_tool_result_overall_qc_is_conservative_artifact_aggregate() -> None:
    failed = example("ToolExecutionResult")
    failed["qc_status"] = "FAIL"
    failed["artifacts"][0]["qc_status"] = "FAIL"  # type: ignore[index]
    assert ToolExecutionResult.model_validate(failed).qc_status.value == "FAIL"

    mismatch = example("ToolExecutionResult")
    mismatch["artifacts"][0]["qc_status"] = "FAIL"  # type: ignore[index]
    with pytest.raises(ValidationError, match="conservative aggregate"):
        ToolExecutionResult.model_validate(mismatch)

    mixed = example("ToolExecutionResult")
    second_artifact = deepcopy(mixed["artifacts"][0])  # type: ignore[index]
    second_artifact["asset_code"] = "ASSET:OID_ASSET003"
    second_artifact["qc_status"] = "UNASSESSED"
    mixed["artifacts"].append(second_artifact)  # type: ignore[union-attr]
    mixed["qc_status"] = "PENDING"
    assert ToolExecutionResult.model_validate(mixed).qc_status.value == "PENDING"


def test_validation_and_capability_gap_bind_01a_evidence_semantics() -> None:
    invalid_validation = example("ValidationResult")
    invalid_validation["observation"] = "UNOBSERVED_TOOL_GAP"
    invalid_validation["capability"] = "UNAVAILABLE_CAPABILITY"
    with pytest.raises(ValidationError, match="tool gap cannot"):
        ValidationResult.model_validate(invalid_validation)

    invalid_gap = example("CapabilityGap")
    invalid_gap["validity"] = "VALID"
    with pytest.raises(ValidationError, match="tool gap cannot be VALID"):
        CONTRACT_MODEL_BY_TYPE["CapabilityGap"].model_validate(invalid_gap)


def test_graph_delta_actor_and_provenance_cannot_bypass_01a_matrix() -> None:
    document = example("GraphDelta")
    document["proposed_by"] = ActorType.EXPERT_AGENT.value
    document["provenance"]["actor"] = ActorType.EXPERT_AGENT.value  # type: ignore[index]
    with pytest.raises(ValidationError, match="cannot submit"):
        GraphDelta.model_validate(document)

    mismatched = example("GraphDelta")
    mismatched["provenance"]["actor"] = ActorType.GRAPH_ENGINE.value  # type: ignore[index]
    with pytest.raises(ValidationError, match="must match provenance.actor"):
        GraphDelta.model_validate(mismatched)

    accepted = GraphDelta.model_validate(example("GraphDelta"))
    assert accepted.proposal_kind is ProposalKind.GRAPH_DELTA_CANDIDATE
    assert accepted.proposed_by is ActorType.VERIFICATION_AGENT


def test_tool_onboarding_decision_requires_matching_immutable_policy_and_machine_gate() -> None:
    mismatched_policy = example("ToolOnboardingDecision")
    other_policy = policy()
    other_policy["checksum"] = CHECKSUM_B
    mismatched_policy["machine_evaluation"]["policy_ref"] = other_policy  # type: ignore[index]
    with pytest.raises(ValidationError, match="must use the referenced immutable policy"):
        ToolOnboardingDecision.model_validate(mismatched_policy)

    rejected_gate = example("ToolOnboardingDecision")
    rejected_gate["machine_evaluation"]["passed"] = False  # type: ignore[index]
    with pytest.raises(ValidationError, match="passing machine evaluation"):
        ToolOnboardingDecision.model_validate(rejected_gate)

    wrong_actor = example("ToolOnboardingDecision")
    wrong_actor["provenance"]["actor"] = "TOOL_AGENT"  # type: ignore[index]
    with pytest.raises(ValidationError, match="must be RULE_ENGINE"):
        ToolOnboardingDecision.model_validate(wrong_actor)


def test_every_top_level_contract_explicitly_declares_scope_identity_ownership() -> None:
    for model in CONTRACT_MODELS:
        assert "scope_identity_fields" in model.__dict__, model.__name__


_ALTERNATIVE_SCOPE_CODES = {
    "case_code": "CASE:ANON_CASE9999",
    "run_id": "RUN:OID_RUN99999",
    "graph_code": "GRAPH:OID_GRAPH999",
    "step_id": "STEP:OID_STEP9999",
    "task_id": "TASK:OID_TASK9999",
    "episode_code": "EPISODE:OID_EPISODE9",
}
_SCOPE_IDENTITY_CASES = tuple(
    (contract_type, model_field, scope_field)
    for contract_type, model in sorted(CONTRACT_MODEL_BY_TYPE.items())
    for model_field, scope_field in model.scope_identity_fields
)


@pytest.mark.parametrize(("contract_type", "model_field", "scope_field"), _SCOPE_IDENTITY_CASES)
def test_top_level_scope_identity_mismatch_is_rejected(
    contract_type: str, model_field: str, scope_field: str
) -> None:
    document = example(contract_type)
    document["scope"][scope_field] = _ALTERNATIVE_SCOPE_CODES[scope_field]  # type: ignore[index]
    with pytest.raises(ValidationError, match=scope_field):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate(document)


@pytest.mark.parametrize(
    "contract_type",
    ("ToolExecutionRequest", "ToolExecutionResult", "ToolOnboardingJob", "ToolOnboardingDecision"),
)
def test_current_research_strategy_rejects_clinical_intended_use(contract_type: str) -> None:
    document = example(contract_type)
    document["intended_use"] = "clinical_deployment"
    with pytest.raises(ValidationError, match="research_only"):
        CONTRACT_MODEL_BY_TYPE[contract_type].model_validate(document)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("catalog_status", "registered"),
        ("validation_status", "benchmark_passed"),
        ("intended_use", "clinical_deployment"),
    ),
)
def test_activated_onboarding_job_requires_exact_research_activation_tuple(
    field_name: str, invalid_value: str
) -> None:
    document = example("ToolOnboardingJob")
    document.update(
        extension_job_status="ACTIVATED_NEXT_RUN",
        catalog_status="active",
        validation_status="auto_validated_research",
        intended_use="research_only",
    )
    document[field_name] = invalid_value
    with pytest.raises(ValidationError, match="research_only|ACTIVATED_NEXT_RUN"):
        ToolOnboardingJob.model_validate(document)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("graph_delta_candidate", "graph_code"), "GRAPH:OID_GRAPH999", "same graph"),
        (
            ("graph_delta_candidate", "scope", "case_code"),
            "CASE:ANON_CASE9999",
            "scope case_code",
        ),
        (
            ("graph_delta_candidate", "scope", "run_id"),
            "RUN:OID_RUN99999",
            "scope run_id",
        ),
        (
            ("graph_delta_candidate", "operations", 0, "node_code"),
            "EVIDENCE:OTHER_TARGET",
            "target",
        ),
        (
            ("graph_delta_candidate", "operations", 0, "evidence_binding"),
            "BINDING:OID_OTHER001",
            "evidence binding",
        ),
        (
            ("graph_delta_candidate", "operations", 0, "source"),
            "RULE_ENGINE",
            "source",
        ),
    ),
)
def test_validation_graph_delta_binding_bypasses_are_rejected(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    document = example("ValidationResult")
    target: object = document
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    if path == ("graph_delta_candidate", "graph_code"):
        document["graph_delta_candidate"]["scope"]["graph_code"] = value  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        ValidationResult.model_validate(document)


def test_validation_graph_delta_requires_declared_binding_and_single_update() -> None:
    no_binding = example("ValidationResult")
    no_binding["evidence_bindings"] = []
    with pytest.raises(ValidationError, match="requires evidence_bindings"):
        ValidationResult.model_validate(no_binding)

    multiple = example("ValidationResult")
    operations = multiple["graph_delta_candidate"]["operations"]  # type: ignore[index]
    operations.append(deepcopy(operations[0]))  # type: ignore[union-attr,index]
    with pytest.raises(ValidationError, match="exactly one"):
        ValidationResult.model_validate(multiple)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("job_id", "TASK:OID_EXTJOB99"),
        ("implementation_code", "IMPLEMENTATION:OID_IMPL9999"),
        ("implementation_version", "2.0.0"),
        ("implementation_checksum", CHECKSUM_B),
    ),
)
def test_onboarding_decision_rejects_each_machine_evaluation_identity_mismatch(
    field_name: str, invalid_value: str
) -> None:
    document = example("ToolOnboardingDecision")
    document["machine_evaluation"][field_name] = invalid_value  # type: ignore[index]
    with pytest.raises(ValidationError, match=field_name):
        ToolOnboardingDecision.model_validate(document)


def test_encounter_namespace_is_distinct_from_memory_episode_namespace() -> None:
    assert contracts.CodeNamespace.ENCOUNTER.value == "ENCOUNTER"
    assert hasattr(contracts, "EncounterCode")
    assert hasattr(contracts, "GraphSnapshotReference")
    valid = CONTRACT_MODEL_BY_TYPE["CaseBundle"].model_validate(example("CaseBundle"))
    assert valid.encounter_code.startswith("ENCOUNTER:")

    document = example("CaseBundle")
    document["encounter_code"] = "EPISODE:OID_VISIT001"
    with pytest.raises(ValidationError, match="ENCOUNTER"):
        CONTRACT_MODEL_BY_TYPE["CaseBundle"].model_validate(document)


def test_nested_case_and_tool_artifact_scope_bypasses_are_rejected() -> None:
    case_document = example("CaseBundle")
    case_document["artifacts"][0]["scope"]["case_code"] = "CASE:ANON_CASE9999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="artifact scope.case_code"):
        CONTRACT_MODEL_BY_TYPE["CaseBundle"].model_validate(case_document)

    result = example("ToolExecutionResult")
    result["artifacts"][0]["scope"]["run_id"] = "RUN:OID_RUN99999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="artifact run_id"):
        ToolExecutionResult.model_validate(result)

    wrong_case = example("ToolExecutionResult")
    wrong_case["artifacts"][0]["scope"]["case_code"] = "CASE:ANON_CASE9999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="artifact case_code"):
        ToolExecutionResult.model_validate(wrong_case)

    nonresearch_output = example("ToolExecutionResult")
    nonresearch_output["artifacts"][0]["research_only"] = False  # type: ignore[index]
    with pytest.raises(ValidationError, match="marked research_only"):
        ToolExecutionResult.model_validate(nonresearch_output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("state_case", "scopes must share case_code"),
        ("state_run", "share run_id"),
        ("state_graph", "share graph_code"),
        ("observation_step", "observation step_id"),
        ("feedback_run", "feedback run_id"),
        ("selected_version", "selected action graph_version"),
        ("next_version", "advance by one"),
        ("legal_actions", "equal state legal_actions"),
        ("selected_action", "member of legal_actions"),
        ("observation_action", "match selected action"),
        ("feedback_version", "feedback expected graph version"),
        ("feedback_actions", "feedback allowed_actions"),
    ),
)
def test_trajectory_internal_consistency_bypasses_are_rejected(mutation: str, message: str) -> None:
    document = example("TrajectoryStep")
    if mutation == "state_case":
        document["state"]["scope"]["case_code"] = "CASE:ANON_CASE9999"  # type: ignore[index]
    elif mutation == "state_run":
        document["state"]["run_id"] = "RUN:OID_RUN99999"  # type: ignore[index]
        document["state"]["scope"]["run_id"] = "RUN:OID_RUN99999"  # type: ignore[index]
    elif mutation == "state_graph":
        document["state"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
        document["state"]["scope"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    elif mutation == "observation_step":
        document["observation"]["step_id"] = "STEP:OID_STEP9999"  # type: ignore[index]
        document["observation"]["scope"]["step_id"] = "STEP:OID_STEP9999"  # type: ignore[index]
    elif mutation == "feedback_run":
        document["feedback"]["run_id"] = "RUN:OID_RUN99999"  # type: ignore[index]
        document["feedback"]["scope"]["run_id"] = "RUN:OID_RUN99999"  # type: ignore[index]
    elif mutation == "selected_version":
        document["selected_action"]["graph_version"] = 2  # type: ignore[index]
    elif mutation == "next_version":
        document["next_state"]["graph_version"] = 3  # type: ignore[index]
    elif mutation == "legal_actions":
        document["legal_actions"] = ["VERIFY"]
    elif mutation == "selected_action":
        document["selected_action"]["selected_action"] = "EXPAND_CLAIM"  # type: ignore[index]
    elif mutation == "observation_action":
        document["observation"]["action"] = "STOP"  # type: ignore[index]
    elif mutation == "feedback_version":
        document["feedback"]["expected_graph_version"] = 2  # type: ignore[index]
    else:
        document["feedback"]["allowed_actions"] = ["STOP"]  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        CONTRACT_MODEL_BY_TYPE["TrajectoryStep"].model_validate(document)


def test_rejected_planner_feedback_can_return_updated_legal_actions() -> None:
    document = example("TrajectoryStep")
    document["feedback"]["accepted"] = False  # type: ignore[index]
    document["feedback"]["allowed_actions"] = []  # type: ignore[index]
    parsed = CONTRACT_MODEL_BY_TYPE["TrajectoryStep"].model_validate(document)
    assert parsed.feedback is not None
    assert parsed.feedback.allowed_actions == parsed.next_state.legal_actions == ()


@pytest.mark.parametrize(
    "run_status",
    (
        "COMPLETED_SUPPORTED",
        "COMPLETED_REFUTED",
        "COMPLETED_UNCERTAIN",
        "COMPLETED_INSUFFICIENT_EVIDENCE",
        "FAILED_TECHNICAL",
        "STOPPED_UNSAFE",
    ),
)
def test_terminal_planner_state_requires_empty_legal_actions(run_status: str) -> None:
    document = example("PlannerState")
    document["run_status"] = run_status
    document["legal_actions"] = []
    assert CONTRACT_MODEL_BY_TYPE["PlannerState"].model_validate(document).legal_actions == ()

    document["legal_actions"] = ["STOP"]
    with pytest.raises(ValidationError, match="terminal PlannerState requires empty"):
        CONTRACT_MODEL_BY_TYPE["PlannerState"].model_validate(document)


def test_nonterminal_planner_state_requires_a_legal_action() -> None:
    document = example("PlannerState")
    document["legal_actions"] = []
    with pytest.raises(ValidationError, match="non-terminal PlannerState"):
        CONTRACT_MODEL_BY_TYPE["PlannerState"].model_validate(document)


@pytest.mark.parametrize(
    ("top_field", "scope_field", "value", "message"),
    (
        ("episode_code", "episode_code", "EPISODE:OID_EPISODE9", "trajectory episode_code"),
        ("case_code", "case_code", "CASE:ANON_CASE9999", "trajectory case_code"),
        ("run_id", "run_id", "RUN:OID_RUN99999", "trajectory run_id"),
    ),
)
def test_episode_rejects_contained_trajectory_identity_mismatch(
    top_field: str, scope_field: str, value: str, message: str
) -> None:
    document = example("Episode")
    document[top_field] = value
    document["scope"][scope_field] = value  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(document)


def test_episode_rejects_graph_and_snapshot_version_bypasses() -> None:
    scope_snapshot_mismatch = example("Episode")
    scope_snapshot_mismatch["final_graph_snapshot_ref"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="initial and final graph"):
        CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(scope_snapshot_mismatch)

    wrong_graph = example("Episode")
    wrong_graph["scope"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    wrong_graph["initial_graph_snapshot_ref"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    wrong_graph["final_graph_snapshot_ref"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="trajectory graph_code"):
        CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(wrong_graph)

    wrong_version = example("Episode")
    wrong_version["final_graph_snapshot_ref"]["graph_version"] = 3  # type: ignore[index]
    with pytest.raises(ValidationError, match="final trajectory graph version"):
        CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(wrong_version)


@pytest.mark.parametrize(
    ("conclusion_status", "run_status"),
    (
        ("SUPPORTED", "COMPLETED_SUPPORTED"),
        ("REFUTED", "COMPLETED_REFUTED"),
        ("UNCERTAIN_MISSING_EVIDENCE", "COMPLETED_INSUFFICIENT_EVIDENCE"),
        ("UNCERTAIN_CONFLICTING_EVIDENCE", "COMPLETED_UNCERTAIN"),
        ("UNABLE_TO_EVALUATE_TECHNICAL", "FAILED_TECHNICAL"),
        ("CAPABILITY_UNAVAILABLE", "COMPLETED_INSUFFICIENT_EVIDENCE"),
        ("STOPPED_UNSAFE", "STOPPED_UNSAFE"),
    ),
)
def test_episode_conclusion_matches_final_next_state(
    conclusion_status: str, run_status: str
) -> None:
    document = example("Episode")
    document["conclusion_status"] = conclusion_status
    document["trajectory"][-1]["next_state"]["run_status"] = run_status  # type: ignore[index]
    CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(document)

    document["trajectory"][-1]["next_state"]["run_status"] = "COMPLETED_REFUTED"  # type: ignore[index]
    if conclusion_status == "REFUTED":
        document["trajectory"][-1]["next_state"]["run_status"] = "COMPLETED_SUPPORTED"  # type: ignore[index]
    with pytest.raises(ValidationError, match="conclusion_status"):
        CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(document)


@pytest.mark.parametrize(
    "field_name",
    (
        "initial_graph_snapshot_ref",
        "final_graph_snapshot_ref",
        "tool_catalog_ref",
        "policy_refs",
        "tool_refs",
        "rule_refs",
        "expert_assessment_refs",
    ),
)
def test_episode_requires_replay_version_and_provenance_ownership(field_name: str) -> None:
    document = example("Episode")
    document.pop(field_name)
    with pytest.raises(ValidationError, match=field_name):
        CONTRACT_MODEL_BY_TYPE["Episode"].model_validate(document)


_COUNTERFACTUAL_OPERATION = {
    "REMOVE_EVIDENCE": "REMOVE",
    "REVERSE_EVIDENCE": "REVERSE",
    "SELECT_ALTERNATIVE_CHECK": "REPLACE",
    "REPLACE_TOOL": "REPLACE",
    "FAILURE_RECOVERY": "RETRY_WITH_ALTERNATIVE",
}
_COUNTERFACTUAL_MISMATCHES = tuple(
    (pair_type, operation)
    for pair_type, expected in _COUNTERFACTUAL_OPERATION.items()
    for operation in ("REMOVE", "REVERSE", "REPLACE", "RETRY_WITH_ALTERNATIVE")
    if operation != expected
)


@pytest.mark.parametrize(("pair_type", "operation"), _COUNTERFACTUAL_MISMATCHES)
def test_counterfactual_pair_type_operation_cartesian_mismatches_are_rejected(
    pair_type: str, operation: str
) -> None:
    document = example("CounterfactualPair")
    document["pair_type"] = pair_type
    document["intervention"]["operation"] = operation  # type: ignore[index]
    document["intervention"]["replacement_code"] = (  # type: ignore[index]
        "TOOL:SYNTHETIC_ALT" if operation in {"REPLACE", "RETRY_WITH_ALTERNATIVE"} else None
    )
    with pytest.raises(ValidationError, match="pair_type"):
        CONTRACT_MODEL_BY_TYPE["CounterfactualPair"].model_validate(document)


def test_counterfactual_factual_graph_must_match_memory_scope() -> None:
    document = example("CounterfactualPair")
    document["factual_graph_ref"]["graph_code"] = "GRAPH:OID_GRAPH999"  # type: ignore[index]
    with pytest.raises(ValidationError, match="scope graph"):
        CONTRACT_MODEL_BY_TYPE["CounterfactualPair"].model_validate(document)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_committed_schema_is_strict_versioned_and_has_explicit_context(
    contract_type: str,
) -> None:
    schema_document = json.loads(
        (SCHEMA_ROOT / f"{contract_type}.schema.json").read_text(encoding="utf-8")
    )
    assert schema_document["additionalProperties"] is False
    assert BASE_REQUIRED_FIELDS <= set(schema_document["required"])
    assert schema_document["properties"]["schema_version"]["const"] == "1.0"
    assert schema_document["properties"]["contract_type"]["const"] == contract_type
    assert set(
        ("extensions", "laterality", "coordinate_space", "unit", "provenance", "scope")
    ) <= set(schema_document["properties"])


def test_schema_and_example_snapshots_equal_deterministic_rendering() -> None:
    expected = rendered_files()
    assert len(expected) == 84
    assert len(tuple(SCHEMA_ROOT.glob("*.json"))) == 21
    assert len(tuple(EXAMPLE_ROOT.glob("*.json"))) == 63
    for path, expected_text in expected.items():
        assert path.read_text(encoding="utf-8") == expected_text


def test_generated_artifacts_contain_no_forbidden_legacy_tool_domain_names() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (SCHEMA_ROOT, EXAMPLE_ROOT)
        for path in sorted(root.glob("*.json"))
    )
    for forbidden in (
        "ToolAdmissionLevel",
        "ToolOnboardingStatus",
        "ToolExtensionStatus",
        '"onboarding_status"',
    ):
        assert forbidden not in combined


def test_in_memory_negative_example_builders_are_real_rejections() -> None:
    for contract_type, model_type in CONTRACT_MODEL_BY_TYPE.items():
        with pytest.raises(ValidationError):
            model_type.model_validate(invalid_state_example(contract_type))
        with pytest.raises(ValidationError):
            model_type.model_validate(version_mismatch_example(contract_type))


def test_contract_models_are_immutable() -> None:
    parsed = CONTRACT_MODEL_BY_TYPE["CaseBundle"].model_validate(example("CaseBundle"))
    with pytest.raises(ValidationError, match="frozen_instance"):
        parsed.schema_version = "1.0"


def test_schema_directories_are_inside_the_contract_package() -> None:
    package_root = Path(contracts.__file__).resolve().parent
    assert SCHEMA_ROOT == package_root / "schemas" / "v1"
    assert EXAMPLE_ROOT == package_root / "examples" / "v1"

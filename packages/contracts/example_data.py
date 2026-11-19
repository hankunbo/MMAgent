"""Deterministic synthetic examples for every v1 top-level contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import CONTRACT_MODEL_BY_TYPE

CHECKSUM_A = "sha256:" + "a" * 64
CHECKSUM_B = "sha256:" + "b" * 64
CREATED_AT = "2026-08-29T00:00:00Z"


def provenance(actor: str = "RULE_ENGINE") -> dict[str, Any]:
    return {
        "actor": actor,
        "created_at": CREATED_AT,
        "generator_version": "1.0.0",
        "source_codes": ["SOURCE:SYNTHETIC_01B"],
        "rule_versions": [],
        "knowledge_versions": [],
        "tool_versions": [],
        "input_checksums": [CHECKSUM_A],
    }


def scope(scope_type: str = "GLOBAL", **references: str) -> dict[str, Any]:
    return {
        "scope_type": scope_type,
        "case_code": None,
        "run_id": None,
        "graph_code": None,
        "step_id": None,
        "task_id": None,
        "episode_code": None,
        **references,
    }


def base(
    contract_type: str,
    *,
    actor: str = "RULE_ENGINE",
    contract_scope: dict[str, Any] | None = None,
    laterality: str = "NOT_APPLICABLE",
    coordinate_space: str = "NOT_APPLICABLE",
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "contract_type": contract_type,
        "provenance": provenance(actor),
        "scope": contract_scope or scope(),
        "laterality": laterality,
        "coordinate_space": coordinate_space,
        "unit": unit,
        "extensions": {},
    }


def versioned(code: str, version: str = "1.0.0") -> dict[str, Any]:
    return {"code": code, "version": version, "checksum": CHECKSUM_A}


def policy() -> dict[str, Any]:
    return {
        "policy_code": "POLICY:TOOL_GATE_01B",
        "version": "1.0.0",
        "checksum": CHECKSUM_A,
    }


def artifact() -> dict[str, Any]:
    return {
        **base(
            "Artifact",
            actor="TOOL_AGENT",
            contract_scope=scope("CASE", case_code="CASE:ANON_CASE0001"),
            laterality="RIGHT",
            coordinate_space="PIXEL_2D",
            unit="1",
        ),
        "asset_code": "ASSET:OID_ASSET001",
        "artifact_type_code": "ARTIFACT:FUNDUS_IMAGE",
        "uri": "object://synthetic/case0001/fundus.png",
        "checksum": CHECKSUM_A,
        "media_type": "image/png",
        "qc_status": "PASS",
        "dimensions": [512, 512],
        "measurements": [{"name": "width", "value": 512.0, "unit": "px"}],
        "derived_from": [],
        "research_only": True,
    }


def case_bundle() -> dict[str, Any]:
    return {
        **base(
            "CaseBundle",
            actor="PLANNER",
            contract_scope=scope("CASE", case_code="CASE:ANON_CASE0001"),
            laterality="RIGHT",
        ),
        "case_code": "CASE:ANON_CASE0001",
        "patient_hash": CHECKSUM_B,
        "encounter_code": "ENCOUNTER:OID_VISIT001",
        "goals": [{"goal_code": "GOAL:SYNTHETIC_SCREEN", "goal_level": "screening"}],
        "artifacts": [artifact()],
        "observations": [
            {
                "concept_code": "CONCEPT:IMAGE_AVAILABLE",
                "observation": "POSITIVE",
                "value": True,
                "unit": None,
                "observed_at": CREATED_AT,
                "source_code": "SOURCE:SYNTHETIC_01B",
            }
        ],
        "report_uri": "object://synthetic/case0001/report.txt",
    }


def template_nodes() -> list[dict[str, Any]]:
    return [
        {
            "node_code": "GOAL:SYNTHETIC_SCREEN",
            "node_type": "GOAL",
            "concept_code": "CONCEPT:SYNTHETIC_GOAL",
            "required": True,
        },
        {
            "node_code": "EVIDENCE:IMAGE_USABLE",
            "node_type": "EVIDENCE",
            "concept_code": "CONCEPT:IMAGE_USABLE",
            "required": True,
        },
    ]


def graph_edges() -> list[dict[str, Any]]:
    return [
        {
            "source_node_code": "EVIDENCE:IMAGE_USABLE",
            "target_node_code": "GOAL:SYNTHETIC_SCREEN",
            "edge_type": "SUPPORTS",
            "rule_ref": versioned("RULE:SYNTHETIC_RULE"),
        }
    ]


def geceg_template() -> dict[str, Any]:
    return {
        **base("GECEGTemplate"),
        "template_code": "TEMPLATE:SYNTHETIC_SCREEN",
        "template_version": "1.0.0",
        "goal_node_code": "GOAL:SYNTHETIC_SCREEN",
        "nodes": template_nodes(),
        "edges": graph_edges(),
        "citations": [versioned("SOURCE:SYNTHETIC_01B")],
    }


def graph_nodes() -> list[dict[str, Any]]:
    return [
        {
            "node_code": "GOAL:SYNTHETIC_SCREEN",
            "node_type": "GOAL",
            "concept_code": "CONCEPT:SYNTHETIC_GOAL",
            "status": "PENDING",
            "availability": "DERIVABLE",
            "evidence_bindings": [],
        },
        {
            "node_code": "EVIDENCE:IMAGE_USABLE",
            "node_type": "EVIDENCE",
            "concept_code": "CONCEPT:IMAGE_USABLE",
            "status": "PENDING",
            "availability": "AVAILABLE",
            "evidence_bindings": ["BINDING:OID_BIND0001"],
        },
    ]


def geceg_instance() -> dict[str, Any]:
    return {
        **base(
            "GECEGInstance",
            actor="GRAPH_ENGINE",
            contract_scope=scope(
                "GRAPH",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
            ),
            laterality="RIGHT",
        ),
        "graph_code": "GRAPH:OID_GRAPH001",
        "case_code": "CASE:ANON_CASE0001",
        "template_ref": versioned("TEMPLATE:SYNTHETIC_SCREEN"),
        "graph_version": 1,
        "goal_node_code": "GOAL:SYNTHETIC_SCREEN",
        "nodes": graph_nodes(),
        "edges": graph_edges(),
    }


def graph_delta() -> dict[str, Any]:
    return {
        **base(
            "GraphDelta",
            actor="VERIFICATION_AGENT",
            contract_scope=scope(
                "GRAPH",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
            ),
            laterality="RIGHT",
        ),
        "graph_code": "GRAPH:OID_GRAPH001",
        "base_version": 1,
        "idempotency_key": "TASK:OID_DELTA001",
        "proposed_by": "VERIFICATION_AGENT",
        "proposal_kind": "GRAPH_DELTA_CANDIDATE",
        "reason_codes": ["RULE:SYNTHETIC_RULE"],
        "operations": [
            {
                "op": "UPDATE_NODE_STATUS",
                "node_code": "EVIDENCE:IMAGE_USABLE",
                "from_status": "PENDING",
                "to_status": "SUPPORTED",
                "source": "VALIDATED_EVIDENCE",
                "evidence_binding": "BINDING:OID_BIND0001",
            }
        ],
    }


def tool_request() -> dict[str, Any]:
    return {
        **base(
            "ToolExecutionRequest",
            actor="TOOL_AGENT",
            contract_scope=scope("RUN", case_code="CASE:ANON_CASE0001", run_id="RUN:OID_RUN00001"),
            laterality="RIGHT",
        ),
        "request_id": "TASK:OID_TOOLREQ1",
        "case_code": "CASE:ANON_CASE0001",
        "run_id": "RUN:OID_RUN00001",
        "target_node_code": "EVIDENCE:IMAGE_USABLE",
        "workflow_ref": versioned("WORKFLOW:SYNTHETIC_QC"),
        "implementation_code": "IMPLEMENTATION:OID_IMPL0001",
        "implementation_version": "1.0.0",
        "implementation_checksum": CHECKSUM_A,
        "inputs": [{"port_name": "image", "asset_code": "ASSET:OID_ASSET001"}],
        "execution_policy": "research",
        "catalog_status": "active",
        "validation_status": "auto_validated_research",
        "intended_use": "research_only",
        "decisive_evidence": True,
        "dry_run": False,
        "idempotency_key": "TASK:OID_IDEMP001",
    }


def tool_result() -> dict[str, Any]:
    output = deepcopy(artifact())
    output["asset_code"] = "ASSET:OID_ASSET002"
    output["artifact_type_code"] = "ARTIFACT:QC_REPORT"
    output["uri"] = "object://synthetic/run0001/qc.json"
    output["media_type"] = "application/json"
    output["scope"] = scope("RUN", case_code="CASE:ANON_CASE0001", run_id="RUN:OID_RUN00001")
    return {
        **base(
            "ToolExecutionResult",
            actor="TOOL_AGENT",
            contract_scope=scope("RUN", case_code="CASE:ANON_CASE0001", run_id="RUN:OID_RUN00001"),
            laterality="RIGHT",
        ),
        "request_id": "TASK:OID_TOOLREQ1",
        "workflow_run_id": "RUN:OID_RUN00001",
        "status": "SUCCEEDED",
        "execution_policy": "research",
        "implementation_code": "IMPLEMENTATION:OID_IMPL0001",
        "implementation_version": "1.0.0",
        "implementation_checksum": CHECKSUM_A,
        "catalog_status": "active",
        "validation_status": "auto_validated_research",
        "intended_use": "research_only",
        "decisive_evidence": True,
        "qc_status": "PASS",
        "artifacts": [output],
        "error": None,
    }


def validation_result() -> dict[str, Any]:
    return {
        **base(
            "ValidationResult",
            actor="VERIFICATION_AGENT",
            contract_scope=scope(
                "GRAPH",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
            ),
            laterality="RIGHT",
        ),
        "validation_code": "VALIDATION:OID_VALID001",
        "validation_scope": "EVIDENCE",
        "target_node_code": "EVIDENCE:IMAGE_USABLE",
        "observation": "POSITIVE",
        "capability": "AVAILABLE",
        "tool_execution_status": "SUCCEEDED",
        "qc_status": "PASS",
        "validity": "VALID",
        "clinical_effect": "SUPPORTS",
        "support_score": 0.9,
        "refute_score": 0.0,
        "rule_refs": [versioned("RULE:SYNTHETIC_RULE")],
        "evidence_bindings": ["BINDING:OID_BIND0001"],
        "graph_delta_candidate": graph_delta(),
        "reason_codes": ["RULE:SYNTHETIC_RULE"],
    }


def planner_state(
    graph_version: int = 1,
    *,
    run_status: str = "PLANNING",
    legal_actions: list[str] | None = None,
    unresolved_nodes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        **base(
            "PlannerState",
            actor="PLANNER",
            contract_scope=scope(
                "RUN",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
            ),
            laterality="RIGHT",
        ),
        "run_id": "RUN:OID_RUN00001",
        "run_status": run_status,
        "graph_code": "GRAPH:OID_GRAPH001",
        "graph_version": graph_version,
        "unresolved_nodes": (
            ["EVIDENCE:IMAGE_USABLE"] if unresolved_nodes is None else unresolved_nodes
        ),
        "available_artifacts": ["ASSET:OID_ASSET001"],
        "legal_actions": ["VERIFY", "STOP"] if legal_actions is None else legal_actions,
        "remaining_budget": {"steps": 3, "tool_calls": 1},
    }


def planner_action() -> dict[str, Any]:
    return {
        **base(
            "PlannerAction",
            actor="PLANNER",
            contract_scope=scope(
                "RUN",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
            ),
            laterality="RIGHT",
        ),
        "run_id": "RUN:OID_RUN00001",
        "graph_code": "GRAPH:OID_GRAPH001",
        "graph_version": 1,
        "proposed_by": "PLANNER",
        "proposal_kind": "PLAN_DECISION",
        "selected_action": "VERIFY",
        "target_code": "EVIDENCE:IMAGE_USABLE",
        "reason_codes": ["RULE:SYNTHETIC_RULE"],
        "idempotency_key": "TASK:OID_ACTION001",
    }


def planner_observation() -> dict[str, Any]:
    return {
        **base(
            "PlannerObservation",
            actor="PLANNER",
            contract_scope=scope(
                "STEP",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
                step_id="STEP:OID_STEP0001",
            ),
            laterality="RIGHT",
        ),
        "run_id": "RUN:OID_RUN00001",
        "graph_code": "GRAPH:OID_GRAPH001",
        "step_id": "STEP:OID_STEP0001",
        "step_status": "SUCCEEDED",
        "action": "VERIFY",
        "observed_codes": ["VALIDATION:OID_VALID001"],
        "tool_result_ref": "TASK:OID_TOOLREQ1",
        "capability_gap_refs": [],
    }


def planner_feedback() -> dict[str, Any]:
    return {
        **base(
            "PlannerFeedback",
            actor="RULE_ENGINE",
            contract_scope=scope(
                "STEP",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
                step_id="STEP:OID_STEP0001",
            ),
            laterality="RIGHT",
        ),
        "run_id": "RUN:OID_RUN00001",
        "graph_code": "GRAPH:OID_GRAPH001",
        "step_id": "STEP:OID_STEP0001",
        "accepted": True,
        "reason_codes": ["RULE:SYNTHETIC_RULE"],
        "allowed_actions": ["VERIFY", "STOP"],
        "expected_graph_version": 1,
    }


def expert_assessment() -> dict[str, Any]:
    return {
        **base(
            "ExpertAssessment",
            actor="EXPERT_AGENT",
            contract_scope=scope("STEP", run_id="RUN:OID_RUN00001", step_id="STEP:OID_STEP0001"),
            laterality="RIGHT",
        ),
        "expert_assessment_id": "EXPERT_ASSESSMENT:OID_EXPERT01",
        "run_id": "RUN:OID_RUN00001",
        "step_id": "STEP:OID_STEP0001",
        "status": "COMPLETED",
        "task_type": "ACTION_REVIEW",
        "input_versions": [versioned("TEMPLATE:SYNTHETIC_SCREEN")],
        "proposal_code": "RULE:SYNTHETIC_RULE",
        "decision": "ACCEPT_PROPOSAL",
        "allowed_action_codes": ["VERIFY", "STOP"],
        "selected_action_code": None,
        "finding_codes": ["RULE:SYNTHETIC_RULE"],
        "knowledge_refs": [
            {
                "source_ref": versioned("SOURCE:SYNTHETIC_01B"),
                "release_status": "PUBLISHED",
            }
        ],
        "unresolved_nodes": ["EVIDENCE:IMAGE_USABLE"],
        "scope_limitations": ["Synthetic research example only"],
        "summary": "The proposed verification remains within the declared evidence scope.",
    }


def capability_gap() -> dict[str, Any]:
    return {
        **base(
            "CapabilityGap",
            actor="TOOL_AGENT",
            contract_scope=scope("RUN", case_code="CASE:ANON_CASE0001", run_id="RUN:OID_RUN00001"),
            laterality="RIGHT",
        ),
        "gap_id": "REQUIREMENT:OID_REQ00001",
        "origin_run_id": "RUN:OID_RUN00001",
        "target_node_code": "EVIDENCE:IMAGE_USABLE",
        "capability_code": "CAPABILITY:FUNDUS_QC",
        "gap_codes": ["NO_IMPLEMENTATION"],
        "observation": "UNOBSERVED_TOOL_GAP",
        "capability": "UNAVAILABLE_CAPABILITY",
        "validity": "UNASSESSED",
        "clinical_effect": "NONE",
        "alternative_paths": [],
        "blocks_required_path": True,
    }


def artifact_requirement() -> dict[str, Any]:
    return {
        "artifact_type_code": "ARTIFACT:FUNDUS_IMAGE",
        "required": True,
        "laterality": "RIGHT",
        "coordinate_space": "PIXEL_2D",
        "unit": None,
    }


def tool_requirement_document() -> dict[str, Any]:
    return {
        **base(
            "ToolRequirementDocument",
            actor="TOOL_AGENT",
            contract_scope=scope(
                "TOOL_EXTENSION",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                task_id="TASK:OID_EXTJOB01",
            ),
            laterality="RIGHT",
        ),
        "requirement_id": "REQUIREMENT:OID_REQ00001",
        "origin_run_id": "RUN:OID_RUN00001",
        "extension_job_id": "TASK:OID_EXTJOB01",
        "target_node_code": "EVIDENCE:IMAGE_USABLE",
        "capability_code": "CAPABILITY:FUNDUS_QC",
        "clinical_purpose": "Assess whether a synthetic fundus artifact is usable.",
        "available_inputs": ["ASSET:OID_ASSET001"],
        "required_input_contract": [artifact_requirement()],
        "required_output_contract": [
            {
                **artifact_requirement(),
                "artifact_type_code": "ARTIFACT:QC_REPORT",
                "coordinate_space": "NOT_APPLICABLE",
            }
        ],
        "attempted_candidates": [
            {
                "candidate_code": "TOOL:SYNTHETIC_QC",
                "failed_stage": "BENCHMARK",
                "reason_codes": ["RULE:SYNTHETIC_RULE"],
            }
        ],
        "validation_plan": {
            "frozen_datasets": [versioned("SOURCE:SYNTHETIC_01B")],
            "metrics": [{"metric_code": "RULE:SYNTHETIC_METRIC", "unit": "1"}],
            "threshold_policy": policy(),
        },
        "blocked_nodes": ["EVIDENCE:IMAGE_USABLE"],
        "conclusion_effect": "INSUFFICIENT_EVIDENCE",
        "available_alternatives": [],
        "replay_when_available": True,
    }


def onboarding_job() -> dict[str, Any]:
    return {
        **base(
            "ToolOnboardingJob",
            actor="TOOL_AGENT",
            contract_scope=scope("TOOL_EXTENSION", task_id="TASK:OID_EXTJOB01"),
        ),
        "job_id": "TASK:OID_EXTJOB01",
        "extension_job_status": "POLICY_GATE",
        "catalog_status": "registered",
        "validation_status": "benchmark_passed",
        "intended_use": "research_only",
        "tool_code": "TOOL:SYNTHETIC_QC",
        "capability_code": "CAPABILITY:FUNDUS_QC",
        "implementation_code": "IMPLEMENTATION:OID_IMPL0001",
        "implementation_version": "1.0.0",
        "implementation_checksum": CHECKSUM_A,
        "policy_ref": policy(),
        "source_catalog_version": "1.0.0",
        "target_catalog_version": "1.1.0",
        "affects_current_run": False,
    }


def onboarding_decision() -> dict[str, Any]:
    return {
        **base(
            "ToolOnboardingDecision",
            actor="RULE_ENGINE",
            contract_scope=scope("TOOL_EXTENSION", task_id="TASK:OID_EXTJOB01"),
        ),
        "decision_id": "VALIDATION:OID_DECIDE01",
        "job_id": "TASK:OID_EXTJOB01",
        "extension_job_status": "ACTIVATED_NEXT_RUN",
        "catalog_status": "active",
        "validation_status": "auto_validated_research",
        "intended_use": "research_only",
        "implementation_code": "IMPLEMENTATION:OID_IMPL0001",
        "implementation_version": "1.0.0",
        "implementation_checksum": CHECKSUM_A,
        "policy_ref": policy(),
        "machine_evaluation": {
            "evaluation_code": "VALIDATION:OID_GATE0001",
            "job_id": "TASK:OID_EXTJOB01",
            "implementation_code": "IMPLEMENTATION:OID_IMPL0001",
            "implementation_version": "1.0.0",
            "implementation_checksum": CHECKSUM_A,
            "evaluator_version": "1.0.0",
            "result_checksum": CHECKSUM_B,
            "policy_ref": policy(),
            "passed": True,
            "gate_evidence": [versioned("SOURCE:SYNTHETIC_01B")],
            "reason_codes": ["RULE:SYNTHETIC_RULE"],
        },
        "published_catalog_version": "1.1.0",
        "visible_to_current_run": False,
    }


def trajectory_step() -> dict[str, Any]:
    return {
        **base(
            "TrajectoryStep",
            actor="PLANNER",
            contract_scope=scope(
                "STEP",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
                step_id="STEP:OID_STEP0001",
                episode_code="EPISODE:OID_EPISODE1",
            ),
            laterality="RIGHT",
        ),
        "case_code": "CASE:ANON_CASE0001",
        "run_id": "RUN:OID_RUN00001",
        "graph_code": "GRAPH:OID_GRAPH001",
        "step_id": "STEP:OID_STEP0001",
        "episode_code": "EPISODE:OID_EPISODE1",
        "step_index": 0,
        "state": planner_state(run_status="VERIFYING"),
        "legal_actions": ["VERIFY", "STOP"],
        "selected_action": planner_action(),
        "observation": planner_observation(),
        "next_state": planner_state(
            2,
            run_status="COMPLETED_SUPPORTED",
            legal_actions=[],
            unresolved_nodes=[],
        ),
        "feedback": planner_feedback(),
    }


def episode() -> dict[str, Any]:
    return {
        **base(
            "Episode",
            actor="RULE_ENGINE",
            contract_scope=scope(
                "MEMORY",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
                episode_code="EPISODE:OID_EPISODE1",
            ),
            laterality="RIGHT",
        ),
        "episode_code": "EPISODE:OID_EPISODE1",
        "case_code": "CASE:ANON_CASE0001",
        "run_id": "RUN:OID_RUN00001",
        "initial_graph_snapshot_ref": {
            "graph_code": "GRAPH:OID_GRAPH001",
            "graph_version": 1,
            "checksum": CHECKSUM_B,
        },
        "final_graph_snapshot_ref": {
            "graph_code": "GRAPH:OID_GRAPH001",
            "graph_version": 2,
            "checksum": CHECKSUM_A,
        },
        "tool_catalog_ref": versioned("TOOL:SYNTHETIC_CATALOG"),
        "policy_refs": [policy()],
        "tool_refs": [versioned("TOOL:SYNTHETIC_QC")],
        "rule_refs": [versioned("RULE:SYNTHETIC_RULE")],
        "expert_assessment_refs": ["EXPERT_ASSESSMENT:OID_EXPERT01"],
        "conclusion_status": "SUPPORTED",
        "review_status": "APPROVED",
        "trajectory": [trajectory_step()],
        "eligible_for_retrieval": True,
    }


def counterfactual_pair() -> dict[str, Any]:
    return {
        **base(
            "CounterfactualPair",
            actor="RULE_ENGINE",
            contract_scope=scope(
                "MEMORY",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
                episode_code="EPISODE:OID_EPISODE1",
            ),
            laterality="RIGHT",
        ),
        "pair_code": "CFPAIR:OID_CFPAIR01",
        "episode_code": "EPISODE:OID_EPISODE1",
        "pair_type": "REMOVE_EVIDENCE",
        "factual_graph_ref": {
            "graph_code": "GRAPH:OID_GRAPH001",
            "graph_version": 2,
            "checksum": CHECKSUM_A,
        },
        "intervention": {
            "target_code": "EVIDENCE:IMAGE_USABLE",
            "operation": "REMOVE",
            "replacement_code": None,
        },
        "outcome": {
            "root_status_before": "SUPPORTED",
            "root_status_after": "INSUFFICIENT_EVIDENCE",
            "support_delta": -0.5,
            "refute_delta": 0.0,
            "completeness_delta": -0.5,
        },
        "applicability_codes": ["GOAL:SYNTHETIC_SCREEN"],
        "validation_refs": ["VALIDATION:OID_VALID001"],
        "review_status": "APPROVED",
        "eligible_for_planning": True,
        "lesson_codes": ["RULE:SYNTHETIC_RULE"],
    }


def message_envelope() -> dict[str, Any]:
    return {
        **base(
            "MessageEnvelope",
            actor="PLANNER",
            contract_scope=scope(
                "RUN",
                case_code="CASE:ANON_CASE0001",
                run_id="RUN:OID_RUN00001",
                graph_code="GRAPH:OID_GRAPH001",
            ),
            laterality="RIGHT",
        ),
        "message_id": "MESSAGE:OID_MSG00001",
        "correlation_id": "RUN:OID_RUN00001",
        "causation_id": None,
        "actor": "PLANNER",
        "proposal_kind": "PLAN_DECISION",
        "payload_contract_type": "PlannerAction",
        "payload": planner_action(),
    }


VALID_EXAMPLES: dict[str, dict[str, Any]] = {
    "Artifact": artifact(),
    "CapabilityGap": capability_gap(),
    "CaseBundle": case_bundle(),
    "CounterfactualPair": counterfactual_pair(),
    "Episode": episode(),
    "ExpertAssessment": expert_assessment(),
    "GECEGInstance": geceg_instance(),
    "GECEGTemplate": geceg_template(),
    "GraphDelta": graph_delta(),
    "MessageEnvelope": message_envelope(),
    "PlannerAction": planner_action(),
    "PlannerFeedback": planner_feedback(),
    "PlannerObservation": planner_observation(),
    "PlannerState": planner_state(),
    "ToolExecutionRequest": tool_request(),
    "ToolExecutionResult": tool_result(),
    "ToolOnboardingDecision": onboarding_decision(),
    "ToolOnboardingJob": onboarding_job(),
    "ToolRequirementDocument": tool_requirement_document(),
    "TrajectoryStep": trajectory_step(),
    "ValidationResult": validation_result(),
}

if set(VALID_EXAMPLES) != set(CONTRACT_MODEL_BY_TYPE):
    raise RuntimeError("every registered contract model must have exactly one valid example")


INVALID_STATE_PATHS: dict[str, tuple[str | int, ...]] = {
    "Artifact": ("qc_status",),
    "CapabilityGap": ("observation",),
    "CaseBundle": ("artifacts", 0, "qc_status"),
    "CounterfactualPair": ("review_status",),
    "Episode": ("review_status",),
    "ExpertAssessment": ("status",),
    "GECEGInstance": ("nodes", 0, "status"),
    "GECEGTemplate": ("nodes", 0, "node_type"),
    "GraphDelta": ("operations", 0, "to_status"),
    "MessageEnvelope": ("proposal_kind",),
    "PlannerAction": ("selected_action",),
    "PlannerFeedback": ("allowed_actions", 0),
    "PlannerObservation": ("step_status",),
    "PlannerState": ("run_status",),
    "ToolExecutionRequest": ("catalog_status",),
    "ToolExecutionResult": ("status",),
    "ToolOnboardingDecision": ("extension_job_status",),
    "ToolOnboardingJob": ("extension_job_status",),
    "ToolRequirementDocument": ("attempted_candidates", 0, "failed_stage"),
    "TrajectoryStep": ("observation", "step_status"),
    "ValidationResult": ("validity",),
}


def _set_path(document: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def invalid_state_example(contract_type: str) -> dict[str, Any]:
    document = deepcopy(VALID_EXAMPLES[contract_type])
    _set_path(document, INVALID_STATE_PATHS[contract_type], "ILLEGAL_STATE")
    return document


def version_mismatch_example(contract_type: str) -> dict[str, Any]:
    document = deepcopy(VALID_EXAMPLES[contract_type])
    document["schema_version"] = "2.0"
    return document

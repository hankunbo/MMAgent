"""Finite metadata-only Fake checks. No callable, runtime, subprocess or medical benchmark."""

from typing import Literal

from packages.grounding.models import Value, digest, validate

from .models import AdapterPlan, AdapterResult, GateEvidence


class FakeProtocol(Value):
    schema_version: Literal["synthetic-adapter-fake/1.0.0"] = "synthetic-adapter-fake/1.0.0"
    case: Literal[
        "ok",
        "safety",
        "interface",
        "empty_mask",
        "performance",
        "repeatability",
        "resource",
        "technical_failure",
        "unknown",
        "not_installed",
    ] = "ok"

    def evaluate(self, plan):
        plan = validate(AdapterPlan, plan)
        if digest(self) != plan.fake_protocol_checksum:
            raise ValueError("adapter_fake_changed")
        gates = []

        def add(stage, outcome, reasons, evidence):
            if isinstance(evidence, Value):
                evidence = evidence.model_dump(mode="json")
            gates.append(
                GateEvidence(
                    stage=stage,
                    outcome=outcome,
                    reasons=tuple(reasons),
                    policy_checksum=digest(plan.policy),
                    input_checksum=digest(plan),
                    evidence_checksum=digest(evidence),
                    evidence=evidence,
                )
            )

        observation = plan.candidate
        contract_outcome = (
            "incompatible"
            if observation.rejection_codes
            else "missing"
            if observation.disposition == "incomplete"
            else "passed"
        )
        reasons = observation.rejection_codes or tuple(
            m
            for m in observation.missing
            if m not in {"frozen_local_evaluation_dataset", "approved_threshold_policy"}
        )
        if plan.manifest.memory_mb > plan.policy.max_memory_mb:
            contract_outcome, reasons = "incompatible", ("adapter_resource_limit",)
        add("contract", contract_outcome, reasons, observation)
        # Generated fixture observations are metadata, never measurements of the candidate.
        observation_data = dict(
            synthetic=True,
            executed=False,
            output_shape=list(plan.manifest.candidate.output_shape or ()),
            output_contract=plan.manifest.candidate.semantics.output
            if plan.manifest.candidate.semantics
            else None,
            nonempty=True,
            sample_checksums=[digest("fixture-output")] * 2,
            elapsed_seconds=0.0,
            memory_mb=1,
            safe=True,
        )
        if self.case == "safety":
            observation_data["safe"] = False
        if self.case == "interface":
            observation_data["output_shape"] = ["wrong"]
        if self.case == "empty_mask":
            observation_data["nonempty"] = False
        if self.case == "repeatability":
            observation_data["sample_checksums"][1] = digest("different")
        if self.case == "performance":
            observation_data["elapsed_seconds"] = plan.manifest.timeout_seconds + 1
        if self.case == "resource":
            observation_data["memory_mb"] = plan.manifest.memory_mb + 1
        checks = dict(
            safety=observation_data["safe"],
            interface=observation_data["output_shape"]
            == list(plan.manifest.candidate.output_shape or ()),
            empty_mask=observation_data["nonempty"],
            repeatability=len(set(observation_data["sample_checksums"])) == 1,
            performance=observation_data["elapsed_seconds"] <= plan.manifest.timeout_seconds,
            resource=observation_data["memory_mb"] <= plan.manifest.memory_mb,
        )
        observation_data["checks"] = checks
        fake_outcome = "not_executed"
        if contract_outcome == "passed":
            fake_outcome = {
                "ok": "passed",
                "technical_failure": "technical_failure",
                "unknown": "unknown",
                "not_installed": "not_executed",
            }.get(self.case, "passed" if all(checks.values()) else "failed")
        add("fake_smoke", fake_outcome, ("synthetic_" + self.case,), observation_data)
        missing = (
            "real_candidate_identity",
            "formal_extension_scope_task",
            "source_target_catalog",
            "original_06_execution_not_installed",
            "real_sandbox_not_installed",
            "frozen_local_evaluation_dataset",
            "preregistered_metrics",
            "approved_threshold_policy",
        )
        for stage, reason in (
            ("sandbox", "real_sandbox_not_installed"),
            ("benchmark", "frozen_local_evaluation_dataset"),
            ("policy", "approved_threshold_policy"),
        ):
            add(stage, "not_executed", (reason,), dict(installed=False, reason=reason))
        outcome = (
            contract_outcome
            if contract_outcome != "passed"
            else ("missing" if fake_outcome == "passed" else fake_outcome)
        )
        failed = outcome in {"failed", "incompatible"}
        return AdapterResult(
            task_id=plan.task_id,
            plan_checksum=digest(plan),
            outcome=outcome,
            candidate_disposition=observation.disposition,
            catalog_status="quarantined" if failed else "candidate",
            extension_job_status="FAILED" if failed else "REQUIREMENT_DOCUMENTED",
            gates=tuple(gates),
            missing=missing,
        )

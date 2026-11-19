"""Explicit synthetic 11D values. No formal activation or execution authority."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Policy, PortRef, Ref, Token, Value, digest
from packages.tool_adapter.models import AdapterPlan, AdapterResult


class ExtensionPolicy(Value):
    schema_version: Literal["synthetic-extension-policy/1.0.0"] = "synthetic-extension-policy/1.0.0"
    baseline_only: Literal[True] = True
    code: Literal["POLICY:SYNTHETIC_EXTENSION"] = "POLICY:SYNTHETIC_EXTENSION"
    version: Literal["1.0.0"] = "1.0.0"
    max_seconds: int = Field(default=10, ge=1, le=30)
    lease_seconds: int = Field(default=30, ge=2, le=60)
    max_attempts: Literal[1] = 1
    real_benchmark: None = None


class CatalogFixture(Value):
    schema_version: Literal["synthetic-extension-catalog/1.0.0"] = (
        "synthetic-extension-catalog/1.0.0"
    )
    synthetic: Literal[True] = True
    purpose: Literal["offline_interface_test_only"] = "offline_interface_test_only"
    source: Literal["independently_installed_test_policy"] = "independently_installed_test_policy"
    adapter_checksum: Checksum
    candidate_identity: Checksum
    catalog_ref: Ref
    implementation_ref: Ref
    target_output_port: PortRef
    fixture_policy: Policy
    fixture_policy_checksum: Checksum
    expires_at: datetime

    @model_validator(mode="after")
    def binding(self):
        if self.catalog_ref.kind != "catalog" or self.implementation_ref.kind != "implementation":
            raise ValueError("extension_fixture_reference_kind")
        if self.expires_at.tzinfo is None:
            raise ValueError("extension_fixture_timezone")
        if (
            not self.fixture_policy.synthetic_only
            or digest(self.fixture_policy) != self.fixture_policy_checksum
        ):
            raise ValueError("extension_fixture_policy_binding")
        return self


class ExtensionGrant(Value):
    schema_version: Literal["synthetic-extension-grant/1.0.0"] = "synthetic-extension-grant/1.0.0"
    synthetic: Literal[True] = True
    purpose: Literal["offline_extension_interfaces_only"] = "offline_extension_interfaces_only"
    run_id: Token
    task_id: Token
    stage_key: Token
    worker: Token
    adapter_checksum: Checksum
    policy_checksum: Checksum
    catalog_fixture_checksum: Checksum
    future_binding_checksum: Checksum
    expires_at: datetime
    budget: Literal[1] = 1

    @model_validator(mode="after")
    def timezone(self):
        if self.expires_at.tzinfo is None:
            raise ValueError("extension_grant_timezone")
        return self


class ExtensionPlan(Value):
    schema_version: Literal["synthetic-extension-plan/1.0.0"] = "synthetic-extension-plan/1.0.0"
    task_id: Token
    adapter_record: dict
    adapter_plan: AdapterPlan
    adapter_result: AdapterResult
    policy: ExtensionPolicy
    grant: ExtensionGrant
    catalog_fixture: CatalogFixture | None
    catalog_observation: dict | None
    future_binding: dict | None
    runtime_context: dict
    lease_token: Token

    @model_validator(mode="after")
    def bindings(self):
        g, a = self.grant, self.adapter_plan
        if (
            self.task_id != g.task_id
            or a.task_id != g.task_id
            or a.application.run_id != g.run_id
            or digest(self.adapter_record) != g.adapter_checksum
            or self.adapter_record.get("status") != "saved"
            or self.adapter_record.get("plan") != a.model_dump(mode="json")
            or self.adapter_record.get("result") != self.adapter_result.model_dump(mode="json")
            or self.adapter_result.plan_checksum != digest(a)
            or digest(self.policy) != g.policy_checksum
            or digest(self.catalog_fixture) != g.catalog_fixture_checksum
            or digest(self.future_binding) != g.future_binding_checksum
        ):
            raise ValueError("extension_plan_binding")
        f = self.catalog_fixture
        if f is not None and (
            f.adapter_checksum != g.adapter_checksum or f.candidate_identity != a.candidate.identity
        ):
            raise ValueError("extension_fixture_adapter_binding")
        return self


class ExtensionResult(Value):
    schema_version: Literal["synthetic-extension-result/1.0.0"] = "synthetic-extension-result/1.0.0"
    synthetic: Literal[True] = True
    task_id: Token
    plan_checksum: Checksum
    adapter_outcome: Literal[
        "missing", "failed", "incompatible", "technical_failure", "unknown", "not_executed"
    ]
    catalog_status: Literal["candidate", "quarantined", "failed"]
    extension_job_status: Literal["REQUIREMENT_DOCUMENTED", "FAILED"]
    validation_status: Literal["unvalidated"] = "unvalidated"
    intended_use: Literal["research_only"] = "research_only"
    fake_catalog_ready: bool
    future_check: dict | None
    missing: tuple[Token, ...]
    catalog_written: Literal[False] = False
    formal_job_created: Literal[False] = False
    completed: Literal[False] = False
    available_next_run: Literal[False] = False
    real_evaluation: Literal["not_executed"] = "not_executed"
    execution: Literal["original_06_not_installed"] = "original_06_not_installed"
    memory_published: Literal[False] = False
    live_domain_acceptance: Literal["live_domain_acceptance_not_enabled"] = (
        "live_domain_acceptance_not_enabled"
    )

    @model_validator(mode="after")
    def domains(self):
        failed = self.adapter_outcome in {"failed", "incompatible"}
        if (self.extension_job_status == "FAILED") != failed:
            raise ValueError("extension_job_domain")
        if (self.catalog_status in {"quarantined", "failed"}) != failed:
            raise ValueError("extension_catalog_domain")
        if self.fake_catalog_ready and self.adapter_outcome != "missing":
            raise ValueError("extension_unconfirmed_cannot_preview")
        if self.future_check is not None:
            from packages.grounding.models import validate
            from packages.grounding.query_models import QueryResult
            from packages.tool_admission.models import AdmissionResult

            if not self.fake_catalog_ready:
                raise ValueError("extension_future_requires_fixture")
            f = self.future_check
            if digest({k: v for k, v in f.items() if k != "evidence_checksum"}) != f.get(
                "evidence_checksum"
            ):
                raise ValueError("extension_future_evidence_binding")
            admission = validate(AdmissionResult, f["admission"])
            if (
                admission.run_id != f["new_run_id"]
                or admission.catalog_ref.model_dump(mode="json") != f["new_lock"]
            ):
                raise ValueError("extension_future_admission_binding")
            if f["grounding"] is not None:
                validate(QueryResult, f["grounding"])
            if (
                f["original_run_id"] == f["new_run_id"]
                or f["original_lock"] == f["new_lock"]
                or f["available_next_run"] is not False
                or f["memory_published"] is not False
                or f["execution"] != "not_executed"
                or f["synthetic"] is not True
            ):
                raise ValueError("extension_future_synthetic_boundary")
        return self

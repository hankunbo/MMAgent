"""One-item then small-batch PUBLIC qualification on the original request ledger."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.review_batches import (
    BatchedSubmission,
    ReviewBatchProposal,
    batch_payload,
    make_plan,
    validate_batch_receipts,
)
from packages.grounding.models import Checksum, Code, Value, digest, validate

from .config import prompt
from .http_recovery import observation_for
from .models import RolePayload, SourceProposal, VisionProposal
from .provider import StructuredProvider
from .public_shape import PublicReviewMaterials
from .transport import wire_request


class PublicRoleReuse(Value):
    role: Literal["source", "vision"]
    request_id: str
    wire_checksum: Checksum
    audit_checksum: Checksum
    result_checksum: Checksum
    reused_not_sent: Literal[True]


class PublicBatchQualification(Value):
    schema_version: Literal["bounded-public-qualification/3.0.0"] = (
        "bounded-public-qualification/3.0.0"
    )
    execution: Literal["synthetic_test", "live_http"]
    round_checksum: Checksum
    config_checksum: Checksum
    run_id: Code
    plan_checksum: Checksum
    submissions: tuple[BatchedSubmission, ...] = Field(min_length=2, max_length=2)
    source_vision_reuse: tuple[PublicRoleReuse, ...] = Field(min_length=2, max_length=2)
    remote_constraint: Literal["json_object"] = "json_object"
    patient_requests: Literal[0] = 0


def batch_wire(config, state, plan, batch):
    setting, text = prompt(config, "review")
    return wire_request(
        config.transport,
        setting,
        text,
        batch_payload(state, plan, batch),
        ReviewBatchProposal.model_json_schema(),
        (),
        scope_binding=digest(state.material.scope),
        wire_policy=config.wire_policy,
        wire_mode=config.wire_mode,
    )


def public_plan(run_id, config):
    counts = (1, config.review_batching.batch_size)
    if counts[1] < 2:
        raise ValueError("public_small_batch_required")
    steps = []
    for count in counts:
        owner = PublicReviewMaterials(run_id, config, task_count=count)
        plan = make_plan(owner.current(), config.review_batching, key="public-batch:" + str(count))
        steps.append(
            dict(
                task_count=count,
                plan=plan.model_dump(mode="json"),
                wires=[digest(batch_wire(config, owner.current(), plan, b)) for b in plan.batches],
            )
        )
    return dict(
        schema_version="public-batch-plan/1.0.0",
        run_id=run_id,
        config_checksum=digest(config),
        steps=steps,
        source_vision_reuse=source_vision_reuse(config),
        patient_material=False,
    )


def source_vision_reuse(config):
    """Prove unchanged role wire against saved REAL successes; never claim a new send."""
    from packages.object_store import checksum

    root = Path("docs/acceptance/12-live-schema-compat-v1")
    audit_path = root / "actual/public-request-audit.json"
    audit = json.loads(audit_path.read_text())
    results_path = root / "actual/public-results.json"
    results = json.loads(results_path.read_text())
    proofs = []
    for role, cls in (("source", SourceProposal), ("vision", VisionProposal)):
        (row,) = [
            r for r in audit["table"] if r["candidate_id"] == "json_object" and r["role"] == role
        ]
        if row["http_status"] != 200 or row["error"] is not None:
            raise ValueError("old_public_role_not_passed")
        path = Path(row["wire_file"])
        if checksum(path.read_bytes()) != row["wire_file_checksum"]:
            raise ValueError("old_public_wire_changed")
        body = json.loads(path.read_text())
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        projected = validate(RolePayload, payload)
        images = ()
        if role == "vision":
            from .live_probes import red_png

            (image,) = [o.image for o in projected.material.observations if o.image]
            images = ((image, red_png()),)
        setting, text = prompt(config, role)
        new = wire_request(
            config.transport,
            setting,
            text,
            payload,
            cls.model_json_schema(),
            images,
            scope_binding=digest(projected.material.scope),
            wire_policy=config.wire_policy,
            wire_mode=config.wire_mode,
        )
        (result,) = results["compat2:json_object:" + role + ":result"]["values"]
        if (
            new != body
            or digest(body) != row["wire_checksum"]
            or result["status"] != "passed"
            or result["request_id"] != row["request_id"]
        ):
            raise ValueError("public_role_wire_revalidation_required")
        proofs.append(
            dict(
                role=role,
                request_id=row["request_id"],
                wire_checksum=digest(body),
                audit_checksum=checksum(audit_path.read_bytes()),
                result_checksum=checksum(results_path.read_bytes()),
                reused_not_sent=True,
            )
        )
    return proofs


def run_public_batches(config, budget, documents, round_checksum, *, provider=None):
    plan = public_plan(budget.run_id, config)
    prior = documents.read("public-qualification")
    if prior:
        check_public_batches(config, budget, documents, prior[0], round_checksum)
        return prior[0]
    budget.check_control()
    with documents.session():
        documents.write("public-batch-plan", (plan,))
    provider = provider or StructuredProvider(config, live_opt_in=True)
    batches = []
    for step in plan["steps"]:
        owner = PublicReviewMaterials(budget.run_id, config, task_count=step["task_count"])
        agent = StructuredRoleAgent("review", provider=provider, materials=owner, budget=budget)
        sub = run_batches(
            agent, state=owner.current(), documents=documents, key=step["plan"]["key"], round_id="A"
        )
        batches.append(sub.model_dump(mode="json"))
    synthetic = provider.transport.simulated
    contexts = getattr(documents, "contexts", None)
    if contexts is not None:
        synthetic = synthetic or contexts.read_research_run(budget.run_id).synthetic
    qualification = dict(
        schema_version="bounded-public-qualification/3.0.0",
        execution="synthetic_test" if synthetic else "live_http",
        round_checksum=round_checksum,
        config_checksum=digest(config),
        run_id=budget.run_id,
        plan_checksum=digest(plan),
        submissions=batches,
        source_vision_reuse=plan["source_vision_reuse"],
        remote_constraint="json_object",
        patient_requests=0,
    )
    check_public_batches(config, budget, documents, qualification, round_checksum)
    budget.check_control()
    with documents.session():
        documents.write("public-qualification", (qualification,))
    return qualification


def check_public_batches(
    config, budget, documents, qualification, round_checksum, *, require_live=False
):
    qualification = validate(PublicBatchQualification, qualification).model_dump(mode="json")
    plan = public_plan(budget.run_id, config)
    if (
        qualification["schema_version"] != "bounded-public-qualification/3.0.0"
        or qualification["config_checksum"] != digest(config)
        or qualification["round_checksum"] != round_checksum
        or qualification["run_id"] != budget.run_id
        or qualification["plan_checksum"] != digest(plan)
        or qualification["source_vision_reuse"] != plan["source_vision_reuse"]
        or qualification["remote_constraint"] != "json_object"
        or len(qualification["submissions"]) != len(plan["steps"])
    ):
        raise ValueError("public_batch_qualification_binding")
    if require_live and qualification["execution"] != "live_http":
        raise ValueError("real_public_batch_qualification_required")
    contexts = getattr(documents, "contexts", None)
    if qualification["execution"] == "live_http" and (
        contexts is None or contexts.read_research_run(budget.run_id).synthetic
    ):
        raise ValueError("real_public_context_required")
    durations = []
    for step, raw in zip(plan["steps"], qualification["submissions"], strict=True):
        sub = validate(BatchedSubmission, raw)
        if sub.plan.model_dump(mode="json") != step["plan"]:
            raise ValueError("public_batch_plan_binding")
        state = PublicReviewMaterials(
            budget.run_id, config, task_count=step["task_count"]
        ).current()
        records = validate_batch_receipts(state, sub, documents, budget)
        for record, wire in zip(records, step["wires"], strict=True):
            event = observation_for(budget.journal, record.request_id)
            if qualification["execution"] == "live_http" and (
                record.simulated
                or event is None
                or event.get("origin") != "http_worker"
                or event.get("http_status") != 200
                or event.get("wire_checksum") != wire
                or event.get("config_checksum") != digest(config)
            ):
                raise ValueError("real_batch_http_receipt_required")
            durations.append(record.latency_seconds)
    return durations


def case_estimate(task_count, config, public_used, durations):
    count = (
        task_count + config.review_batching.batch_size - 1
    ) // config.review_batching.batch_size
    normal = count + 2  # Source + Vision; Expert triggers, if any, consume the same ledger.
    maximum = (
        normal + config.max_extra_requests if hasattr(config, "max_extra_requests") else 2 * normal
    )
    estimate = max(durations, default=0) * 1.5 * count + 180
    return dict(
        task_count=task_count,
        review_batches=count,
        planner_steps=14,
        normal_case_requests=normal,
        maximum_with_one_retry=maximum,
        public_used=public_used,
        remaining_requests=40 - public_used,
        estimated_seconds=estimate,
        estimate_is_not_latency_guarantee=True,
        feasible=bool(durations)
        and count <= config.review_batching.max_batches
        and maximum <= 40 - public_used
        and estimate < 2700,
    )

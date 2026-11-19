"""Bounded successor of a locally expired, never-dispatched capability plan.

Original Runtime claim, task/Control/document owners prove the absence of work.
The old context, deadline and failure remain immutable. No new execution owner.
"""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest

from . import capability_window as window

VERSION = window.RECOVERY_VERSION
PROOF_VERSION = "capability-pre-dispatch-deadline/1.0.0"


def origin(parent, previous):
    from .capability_review_origin import resolve
    from .input_scope import configuration

    out = Path(previous).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("capability_recovery_origin_symlink")
    original = json.loads((out / "protocol.json").read_bytes())
    if original.get("schema_version") != window.VERSION or original.get("output") != str(out):
        raise ValueError("capability_recovery_origin_version")
    old = resolve(parent, original)  # Original Runtime claim and amendment proof.
    failure = json.loads((out / "stage-failure.json").read_bytes())
    if (
        datetime.fromisoformat(old["deadline"]) > datetime.now(UTC)
        or any((out / p).exists() for p in ("cancel", "result.json", "stage-closure-error.json"))
        or list(out.glob("**/request.json"))
        or failure.get("schema_version") != "multimodal-stage-failure/1.1.0"
        or failure.get("campaign_id") != out.name
        or failure.get("protocol_checksum") != digest(old)
        or failure.get("error_code") != "deadline"
        or failure.get("failure_kind") != "local_stage_failure"
        or failure.get("failure_stage") != "acquisition"
        or failure.get("input_checksum") != old["input"]
        or failure.get("target") != parent.base.task.target_code
        or failure.get("previous_attempts") != old["previous_attempts"]
        or failure.get("new_requests") != 0
        or failure.get("phase_reservations") != 0
        or failure.get("confirmed_phases") != []
        or failure.get("cancel_marker") is not False
        or failure.get("score_status") != "not_scored"
        or failure.get("control", {}).get("exists") is not False
    ):
        raise ValueError("capability_recovery_not_local_unstarted_deadline")
    states = []
    targets = (
        None,
        *old["input_scope"]["targets"],
        *old["acquisition_targets"],
        "definition-expansion",
    )
    for target in targets:
        key = "mm-live:" + digest((out.name, target or "review"))[7:55]
        run = identity("RUN", "task-research:" + key)
        context = parent.contexts.read_research_run(run, missing_ok=True)
        if parent.sessions.read(run) is not None:
            raise ValueError("capability_recovery_control_exists")
        keys = (
            "journal:0",
            "dynamic-phase-input",
            "dynamic-phase-result",
            "multimodal-case-result",
        )
        bundles = parent.base.runtime.read_observation_bundles(
            tuple(identity("TASK", run + ":agent:" + k) for k in keys)
        )
        if any(v is not None for v in bundles):
            raise ValueError("capability_recovery_request_or_result_exists")
        if context is not None:
            scope_plan = old["capability_transition"]["scope_plan"]
            expected_policy = digest((scope_plan, configuration(parent.config, scope_plan), target))
            if (
                target not in old["input_scope"]["targets"]
                or context.run_id != run
                or context.request.scope.case_code != parent.base.binding["case_code"]
                or context.request.input_checksum != old["input"]
                or context.request.task != parent.base.task
                or context.request.policy_ref.checksum != expected_policy
            ):
                raise ValueError("capability_recovery_registered_context_binding")
        states.append(dict(target=target, run_id=run, context=digest(context) if context else None))
    return old, dict(
        schema_version=PROOF_VERSION,
        previous=str(out),
        original_protocol=digest(original),
        effective_protocol=digest(old),
        failure=digest(failure),
        phases=states,
        predecessor_requests=0,
        old_deadline_unchanged=True,
    )


def plan_for(old, proof, *, output, authorization, now, implementation):
    out = Path(output).absolute()
    if (
        old.get("schema_version") != window.VERSION
        or proof.get("schema_version") != PROOF_VERSION
        or proof.get("effective_protocol") != digest(old)
        or proof.get("previous") != old["output"]
        or proof.get("predecessor_requests") != 0
        or not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or now < datetime.fromisoformat(old["deadline"])
        or not 600 <= old["seconds"] <= 5400
        or not 2 <= old["max_new_requests"] <= 12
        or old["max_model_dispatches"] != old["max_new_requests"] - 1
        or any(
            type(p.get("requests")) is not int or p["requests"] < 0
            for p in old["previous_attempts"]
        )
        or sum(p["requests"] for p in old["previous_attempts"]) + old["max_new_requests"]
        > old["max_cumulative_requests"]
        or out == Path(old["output"])
        or out.is_relative_to(old["output"])
        or Path(old["output"]).is_relative_to(out)
    ):
        raise ValueError("capability_recovery_bounds")
    plan = deepcopy(old)
    plan.update(
        schema_version=VERSION,
        output=str(out),
        authorization=authorization,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=old["seconds"])).isoformat(),
        prior_deadline=old["deadline"],
        implementation=implementation,
        pre_dispatch_recovery=proof,
    )
    return plan


def claim_key(old):
    return "capability:pre-dispatch-successor:" + digest(old)[7:]


def check(parent, plan, *, found=None, tip=None, require_claim=True):
    if plan.get("schema_version") != VERSION:
        raise ValueError("capability_recovery_version")
    old, proof = origin(parent, plan["pre_dispatch_recovery"]["previous"])
    if found is not None:
        expected_old = window.plan_for(
            parent,
            tip,
            found,
            output=old["output"],
            authorization=old["authorization"],
            now=datetime.fromisoformat(old["started_at"]),
            implementation=old["implementation"],
            seconds=old["seconds"],
        )
        if expected_old != old:
            raise ValueError("capability_recovery_current_inventory_changed")
    expected = plan_for(
        old,
        proof,
        output=plan["output"],
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
    )
    if expected != plan:
        raise ValueError("capability_recovery_plan_changed")
    docs = parent.documents(old["tip"]["run_id"])
    if require_claim and docs.read(claim_key(old)) != (window.claim(plan, plan["output"]),):
        raise ValueError("capability_recovery_claim_binding")
    return old, proof


def prepare(*, previous, output, authorization):
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .campaign_renewal import completed_parent
    from .execution_lease import session
    from .implementation import freeze
    from .unknown_acquisition import check_network

    previous, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("capability_recovery_output_symlink")
    raw = json.loads((previous / "protocol.json").read_bytes())
    with historical_fact_read_scope(), _projection_visible_reads():
        parent, _ = completed_parent(raw["facts"])
        lease = SimpleNamespace(output=previous, target=parent.target, base=parent.base)
        with session(lease):
            old, proof = origin(parent, previous)
            check_network(parent.config)
            tip, found, _ = window.inventory(parent, out, frozen_methods=old["procedure_reuse"])
            docs = parent.documents(old["tip"]["run_id"])
            saved = docs.read(claim_key(old))
            if saved:
                if len(saved) != 1 or saved[0].get("output") != str(out):
                    raise ValueError("capability_recovery_already_claimed")
                plan = saved[0]["plan"]
                if plan["implementation"] != freeze():
                    raise ValueError("capability_recovery_code_changed")
            else:
                if out.exists():
                    raise ValueError("capability_recovery_new_directory_required")
                plan = plan_for(
                    old,
                    proof,
                    output=out,
                    authorization=authorization,
                    now=datetime.now(UTC),
                    implementation=freeze(),
                )
            # Re-read old context/journals after potentially long inventory work.
            check(parent, plan, found=found, tip=tip, require_claim=bool(saved))
            if not saved:
                with docs.session():
                    if docs.read(claim_key(old)) is not None:
                        raise ValueError("capability_recovery_already_claimed")
                    docs.write(claim_key(old), (window.claim(plan, out),))
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        status="already_recorded" if saved else "prepared",
        deadline=plan["deadline"],
        previous_requests=sum(p["requests"] for p in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        scope_targets=plan["input_scope"]["targets"],
        pending_review_ids=sorted(
            set(plan["review_target_ids"]) - set(plan["reuse_assessment_ids"])
        ),
        retained_assessment_ids=plan["reuse_assessment_ids"],
        api_requests=0,
        database_writes=0 if saved else 1,
        old_failure_preserved=True,
    )

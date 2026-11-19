"""Explicit new Scope attempt after an exact original UNKNOWN analytical request.

The old run stays stopped. The original campaign executes the new plan, and the
original request/acceptance/budget owners remain authoritative on cold restore.
"""

import json
from copy import copy, deepcopy
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate

from . import capability_window as window

VERSION = window.SCOPE_UNKNOWN_VERSION
KIND = "unknown-scope-new-attempt/1.0.0"
CONFIRMED_KIND = "confirmed-invalid-scope-new-attempt/1.0.0"


def confirmed(proof):
    return proof.get("schema_version") == "original-confirmed-invalid-scope-proof/1.0.0"


def saved_attempt(plan):
    mode = plan.get("schema_version") == window.SCOPE_CONFIRMED_VERSION
    field = "confirmed_scope_attempt" if mode else "unknown_scope_attempt"
    other = "unknown_scope_attempt" if mode else "confirmed_scope_attempt"
    proof = plan.get(field)
    if other in plan or not isinstance(proof, dict) or confirmed(proof) != mode:
        raise ValueError("scope_attempt_explicit_mode_binding")
    return proof


def source(parent, previous):
    from .campaign_renewal import assemble_bound_parent

    path = Path(previous).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("scope_unknown_original_symlink")
    protocol = json.loads((path / "protocol.json").read_bytes())
    if protocol.get("schema_version") not in {window.CURRENT_VERSION, window.ROUTED_VERSION}:
        raise ValueError("scope_unknown_original_version")
    return assemble_bound_parent(parent, path, protocol)


def inspect(predecessor, acknowledged, *, confirmed_invalid=False):
    from .campaign import Phase
    from .input_scope import ScopeTargetReader, configuration, current_plan
    from .unknown_acquisition import original_attempt

    path, old = predecessor.output, predecessor.protocol
    failure = json.loads((path / "stage-failure.json").read_bytes())
    if (
        (path / "cancel").exists()
        or (path / "result.json").exists()
        or failure.get("protocol_checksum") != digest(old)
        or failure.get("input_checksum") != old["input"]
        or failure.get("campaign_id") != path.name
        or failure.get("previous_attempts") != old["previous_attempts"]
        or failure.get("cancel_marker") is not False
        or failure.get("complete") is not False
        or failure.get("failure_stage") != "acquisition"
        or failure.get("new_requests") != 1
        or len(failure.get("confirmed_phases", [])) != 1
        or not isinstance(acknowledged, str)
        or not acknowledged
    ):
        raise ValueError("scope_unknown_original_failure_binding")
    failed = failure["confirmed_phases"][0]
    target = failed.get("target")
    if target not in predecessor.scope_targets:
        raise ValueError("scope_unknown_survey_target_required")
    expected_run = identity("RUN", "task-research:mm-live:" + digest((path.name, target))[7:55])
    if failed.get("run_id") != expected_run or failure.get("run_id") != expected_run:
        raise ValueError("scope_unknown_original_run_binding")
    docs = predecessor.documents(expected_run)
    if (
        docs.read("dynamic-campaign") != (old,)
        or docs.read("multimodal-stage-failure") != (failure,)
        or json.loads((predecessor.phase_path(target) / "result.json").read_bytes()) != failed
    ):
        raise ValueError("scope_unknown_original_persisted_binding")
    context = predecessor.contexts.read_research_run(docs.run_id)
    if context.request.task != predecessor.base.task:
        raise ValueError("scope_unknown_task_binding")
    base = copy(predecessor.base)
    base.contexts = dict(base.contexts)
    base.contexts[docs.run_id] = context
    projected = copy(predecessor)
    projected.base = base
    projected.config = configuration(predecessor.config, current_plan(base))
    phase = Phase(base, target=target)
    current, _ = phase.project(run_id=docs.run_id, config=projected.config)
    phase.verify(current)

    class Reader:
        def current(self):
            phase.verify(current)
            return current

    reader = ScopeTargetReader(Reader(), phase)
    proof = original_attempt(
        projected,
        failed,
        acknowledged,
        scope_owner=reader,
        **({"confirmed_invalid": True} if confirmed_invalid else {}),
    )
    initial = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if initial.material != current.material or initial.policy != current.policy:
        raise ValueError("scope_unknown_current_material_changed")
    states = []
    targets = (
        None,
        *predecessor.scope_targets,
        *old["acquisition_targets"],
        "definition-expansion",
    )
    for other in dict.fromkeys(targets):
        if other == target:
            continue
        run = identity(
            "RUN", "task-research:mm-live:" + digest((path.name, other or "review"))[7:55]
        )
        context = predecessor.contexts.read_research_run(run, missing_ok=True)
        if context is not None or predecessor.sessions.read(run) is not None:
            raise ValueError("scope_unknown_other_phase_registered")
        states.append(dict(target=other, run_id=run, state="never_registered"))
    return dict(
        schema_version="original-confirmed-invalid-scope-proof/1.0.0"
        if confirmed_invalid
        else "original-unknown-scope-proof/1.0.0",
        previous=str(path.absolute()),
        protocol_checksum=digest(old),
        failure_checksum=digest(failure),
        original_attempt=proof,
        task=reader.input_scope_task(current.material.scope).model_dump(mode="json"),
        states=states,
        requests=1,
        acknowledged_request_id=acknowledged,
        new_attempt_not_old_replay=True,
        duplicate_billing_possible=True,
    )


def history_row(proof):
    return dict(
        directory=proof["previous"],
        checksum=proof["failure_checksum"],
        requests=1,
        kind=CONFIRMED_KIND if confirmed(proof) else KIND,
    )


def corrected_inventory_baseline(before, found):
    from .active_requirements import load
    from .input_scope import read_policy

    old_ref, new_ref = before["profile"], found["profile"]
    if (
        old_ref["path"] != window.NEW_CURRENT_PROFILE
        or new_ref["path"] != window.CORRECTED_SCOPE_PROFILE
    ):
        raise ValueError("scope_confirmed_profile_transition")
    old, new = load(old_ref["path"]), load(new_ref["path"])
    if digest(old) != old_ref["checksum"] or digest(new) != new_ref["checksum"]:
        raise ValueError("scope_confirmed_profile_binding")
    if {k: v for k, v in old.items() if k not in {"id", "input_scope_policy"}} != {
        k: v for k, v in new.items() if k not in {"id", "input_scope_policy"}
    }:
        raise ValueError("scope_confirmed_only_survey_instructions")
    old_policy, new_policy = (
        read_policy(old["input_scope_policy"]),
        read_policy(new["input_scope_policy"]),
    )
    if old_policy.model_dump(exclude={"procedure"}) != new_policy.model_dump(exclude={"procedure"}):
        raise ValueError("scope_confirmed_questions_changed")
    value = deepcopy(before)
    # These exact metadata locations change when only survey instructions change.
    for holder in (value["transition"], value["transition"]["catalog_plan"]):
        if holder["profile_checksum"] != old_ref["checksum"]:
            raise ValueError("scope_confirmed_original_profile_projection")
        holder["profile_checksum"] = new_ref["checksum"]
    if (
        value["scope_plan"]["policy"] != old["input_scope_policy"]
        or found["scope_plan"]["policy"] != new["input_scope_policy"]
    ):
        raise ValueError("scope_confirmed_policy_projection")
    value["scope_plan"]["policy"] = new["input_scope_policy"]
    return value


def inventory(
    parent,
    predecessor,
    output,
    acknowledged,
    route,
    *,
    frozen_methods=None,
    confirmed_invalid=False,
):
    proof = inspect(
        predecessor, acknowledged, **({"confirmed_invalid": True} if confirmed_invalid else {})
    )
    tip, found, reader = window.inventory(
        parent,
        output,
        frozen_methods=frozen_methods,
        current=True,
        route_transition=route,
        profile_path=window.CORRECTED_SCOPE_PROFILE
        if confirmed_invalid
        else window.frozen_current_profile(predecessor.protocol),
    )
    before = predecessor.protocol["capability_transition"]
    if confirmed_invalid:
        before = corrected_inventory_baseline(before, found)
    # Scope retry changes transport and attempt identity, never the task, views,
    # interpretation, already selected methods, or the medically visible plan.
    for field in (
        "transition",
        "scope_plan",
        "review_target_ids",
        "acquisition_targets",
        "native_checksum",
        "bound_observations_checksum",
        "confirmed_observations_checksum",
    ):
        if found[field] != before[field]:
            raise ValueError("scope_unknown_current_inventory_changed:" + field)
    target = proof["original_attempt"]["target"]
    entries = [e for e in found["scope_plan"]["entries"] if e["target"] == target]
    if (
        len(entries) != 1
        or entries[0]["task"] != proof["task"]
        or target not in found["input_scope"]["dispatch_targets"]
        or tip["previous_attempts"] != predecessor.protocol["previous_attempts"]
        or parent.base.task != predecessor.base.task
        or parent.base.manifest.checksum != predecessor.base.manifest.checksum
    ):
        raise ValueError("scope_unknown_current_task_or_budget_changed")
    return tip, found, reader, proof


def plan_for(
    parent,
    predecessor,
    tip,
    found,
    proof,
    *,
    output,
    authorization,
    now,
    implementation,
    route_transition,
    seconds=5400,
):
    adjusted = deepcopy(tip)
    adjusted["previous_attempts"] = [*tip["previous_attempts"], history_row(proof)]
    plan = window.plan_for(
        parent,
        adjusted,
        found,
        output=output,
        authorization=authorization,
        now=now,
        implementation=implementation,
        seconds=seconds,
        current=True,
        route_transition=route_transition,
    )
    plan.update(
        schema_version=window.SCOPE_CONFIRMED_VERSION if confirmed(proof) else VERSION,
        tip=tip,
        **(
            {"confirmed_scope_attempt": proof}
            if confirmed(proof)
            else {"unknown_scope_attempt": proof}
        ),
        prior_deadline=predecessor.protocol["deadline"],
    )
    return plan


def claim_key(proof):
    return "capability:unknown-scope-successor:" + proof["protocol_checksum"][7:]
    # Both modes share one original attempt claim: switching modes cannot mint a second successor.


def claim_documents(predecessor, proof):
    return predecessor.documents(proof["original_attempt"]["run_id"])


def check(parent, plan, *, require_claim=True):
    if plan.get("schema_version") not in window.SCOPE_ATTEMPT_VERSIONS:
        raise ValueError("scope_unknown_successor_version")
    saved = saved_attempt(plan)
    predecessor = source(parent, saved["previous"])
    tip, found, reader, proof = inventory(
        parent,
        predecessor,
        plan["output"],
        saved["acknowledged_request_id"],
        plan["route_transition"],
        frozen_methods=plan["procedure_reuse"],
        **({"confirmed_invalid": True} if confirmed(saved) else {}),
    )
    expected = plan_for(
        parent,
        predecessor,
        tip,
        found,
        proof,
        output=plan["output"],
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
        seconds=plan["seconds"],
        route_transition=plan["route_transition"],
    )
    if expected != plan:
        raise ValueError("scope_unknown_successor_binding_changed")
    if require_claim and claim_documents(predecessor, proof).read(claim_key(proof)) != (
        window.claim(plan, plan["output"]),
    ):
        raise ValueError("scope_unknown_successor_claim_binding")
    return reader


def check_history(plan):
    proof = saved_attempt(plan)
    path = Path(proof["previous"])
    if (
        digest(json.loads((path / "protocol.json").read_bytes())) != proof["protocol_checksum"]
        or digest(json.loads((path / "stage-failure.json").read_bytes()))
        != proof["failure_checksum"]
        or plan["previous_attempts"] != [*plan["tip"]["previous_attempts"], history_row(proof)]
    ):
        raise ValueError("scope_unknown_history_changed")


def verify_budget_history(parent, attempt):
    from .campaign_renewal import MAX_HISTORY_DEPTH

    current, seen = parent, set()
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if current.protocol.get("schema_version") in window.SCOPE_ATTEMPT_VERSIONS:
            proof = saved_attempt(current.protocol)
            if history_row(proof) == attempt:
                check_history(current.protocol)
                prior = source(current.parent, proof["previous"])
                if (
                    inspect(
                        prior,
                        proof["acknowledged_request_id"],
                        **({"confirmed_invalid": True} if confirmed(proof) else {}),
                    )
                    != proof
                ):
                    raise ValueError("scope_unknown_history_original_changed")
                if claim_documents(prior, proof).read(claim_key(proof)) != (
                    window.claim(current.protocol, current.output),
                ):
                    raise ValueError("scope_unknown_history_claim_binding")
                return
        if current.protocol.get("schema_version") == window.CURRENT_SCOPE_RECOVERY_VERSION:
            from .scope_successor import budget_predecessor

            current = budget_predecessor(current)
        else:
            current = getattr(current, "parent", None)
    raise ValueError("scope_unknown_history_owner_missing")


def prepare(
    *,
    previous,
    output,
    authorization,
    acknowledge_request,
    network_route,
    check_only=False,
    confirmed_invalid=False,
):
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .campaign_renewal import completed_parent
    from .execution_lease import session
    from .implementation import freeze
    from .unknown_acquisition import check_network

    previous, out = Path(previous).absolute(), Path(output).absolute()
    if (
        any(p.is_symlink() for p in (out, *out.parents))
        or out == previous
        or out.is_relative_to(previous)
        or previous.is_relative_to(out)
    ):
        raise ValueError("scope_unknown_output_must_preserve_history")
    old = json.loads((previous / "protocol.json").read_bytes())
    with historical_fact_read_scope(), _projection_visible_reads():
        parent, _ = completed_parent(old["facts"])
        predecessor = source(parent, previous)
        with session(predecessor):
            route = window.route_binding(parent, network_route)
            check_network(window.route_configuration(parent, route))
            tip, found, _, proof = inventory(
                parent,
                predecessor,
                out,
                acknowledge_request,
                route,
                **({"confirmed_invalid": True} if confirmed_invalid else {}),
            )
    with session(predecessor):
        if (
            inspect(
                predecessor,
                acknowledge_request,
                **({"confirmed_invalid": True} if confirmed_invalid else {}),
            )
            != proof
        ):
            raise ValueError("scope_unknown_original_changed_before_claim")
        docs = claim_documents(predecessor, proof)
        key = claim_key(proof)
        saved = docs.read(key)
        if saved and (
            len(saved) != 1 or saved[0].get("output") != str(out) or "plan" not in saved[0]
        ):
            raise ValueError("scope_unknown_successor_already_claimed")
        if check_only and saved:
            raise ValueError("scope_unknown_successor_already_claimed")
        plan = plan_for(
            parent,
            predecessor,
            tip,
            found,
            proof,
            output=out,
            authorization=authorization,
            now=datetime.fromisoformat(saved[0]["plan"]["started_at"])
            if saved
            else datetime.now(UTC),
            implementation=freeze(),
            route_transition=route,
        )
        if saved and saved != (window.claim(plan, out),):
            raise ValueError("scope_unknown_saved_claim_changed")
        if not check_only and not saved:
            if out.exists():
                raise ValueError("scope_unknown_new_directory_required")
            with docs.session():
                if docs.read(key) is not None:
                    raise ValueError("scope_unknown_successor_already_claimed")
                docs.write(key, (window.claim(plan, out),))
    if not check_only:
        out.mkdir(parents=True, exist_ok=True)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        check_only=check_only,
        protocol_checksum=digest(plan),
        schema_version=plan["schema_version"],
        previous_requests=sum(r["requests"] for r in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        deadline=plan["deadline"],
        retained_assessment_ids=plan["reuse_assessment_ids"],
        scope_targets=plan["input_scope"]["dispatch_targets"],
        **(
            {"acknowledged_confirmed_invalid_request": acknowledge_request}
            if confirmed_invalid
            else {"acknowledged_unknown_request": acknowledge_request}
        ),
        duplicate_billing_possible=True,
        api_requests=0,
        database_writes=0 if check_only or saved else 1,
    )

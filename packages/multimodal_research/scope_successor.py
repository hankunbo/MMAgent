"""Qualified survey reuse and a bounded successor through RenewedCampaign.

The predecessor remains stopped. Its confirmed request, failed phase and local
recovery are separate immutable records. Only never-attempted survey targets
are dispatched; observations, Review, claims and budget keep their old owners.
"""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

from . import capability_window as window

VERSION = window.SCOPE_SUCCESSOR_VERSION
CURRENT_VERSION = window.CURRENT_SCOPE_RECOVERY_VERSION
VERSIONS = window.SCOPE_RECOVERY_VERSIONS
CURRENT_PROFILE = "configs/research/multimodal_catalog_acquisition_v9.json"
PROFILE = "configs/research/multimodal_catalog_acquisition_v7.json"


def progress(stage):
    print(json.dumps(dict(scope_successor=stage, at=datetime.now(UTC).isoformat())), flush=True)


def source(parent, previous):
    from .campaign_renewal import assemble_bound_parent

    path = Path(previous).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("scope_successor_original_symlink")
    old = json.loads((path / "protocol.json").read_bytes())
    if old.get("schema_version") not in {
        window.VERSION,
        window.RECOVERY_VERSION,
        window.CURRENT_VERSION,
        window.ROUTED_VERSION,
        window.SCOPE_UNKNOWN_VERSION,
    }:
        raise ValueError("scope_successor_original_version")
    return assemble_bound_parent(parent, path, old)


def inspect(predecessor):
    """Read actual original owners, including untouched never-sent phases."""
    from .finalization import read_finalized_scope

    old, path = predecessor.protocol, predecessor.output
    failure = json.loads((path / "stage-failure.json").read_bytes())
    if (
        (path / "cancel").exists()
        or (path / "result.json").exists()
        or failure.get("protocol_checksum") != digest(old)
        or failure.get("input_checksum") != old["input"]
        or failure.get("cancel_marker") is not False
        or failure.get("complete") is not False
        or failure.get("failure_stage") != "acquisition"
    ):
        raise ValueError("scope_successor_original_failure_binding")
    rows, states, requests = [], [], 0
    targets = (
        None,
        *predecessor.scope_targets,
        *old["acquisition_targets"],
        "definition-expansion",
    )
    for target in targets:
        key = "mm-live:" + digest((path.name, target or "review"))[7:55]
        run = identity("RUN", "task-research:" + key)
        context = predecessor.contexts.read_research_run(run, missing_ok=True)
        control = predecessor.sessions.read(run)
        docs = predecessor.documents(run) if context is not None else None
        events = RuntimeAgentJournal(docs).read() if context is not None else ()
        phase = docs.read("dynamic-phase-result") if context is not None else None
        if phase is None:
            # No inferred retry from a missing result: any intent/control/request
            # requires separate original-owner confirmation before a new attempt.
            if context is not None or control is not None or events:
                raise ValueError("scope_successor_unresolved_phase")
            states.append(dict(target=target, run_id=run, state="never_registered"))
            continue
        if (
            target not in predecessor.scope_targets
            or len(phase) != 1
            or phase[0].get("target") != target
            or phase[0].get("run_id") != run
            or phase[0].get("accepted") is not False
            or control is None
            or control.phase != "stopped"
            or control.pending is not None
            or control.cancelled
            or docs.read("dynamic-campaign") != (old,)
            or json.loads((predecessor.phase_path(target) / "result.json").read_bytes()) != phase[0]
        ):
            raise ValueError("scope_successor_original_phase_binding")
        fact = read_finalized_scope(predecessor, target)
        reserved = [e for e in events if e.get("phase") == "reserved"]
        completed = [e for e in events if e.get("phase") == "completed"]
        # The recovery reader already proves the exact failed, confirmed request.
        if len(reserved) != 1 or len(completed) != 1:
            raise ValueError("scope_successor_original_request_inventory")
        requests += len(reserved)
        rows.append(dict(target=target, fact_checksum=digest(fact), fact=fact))
        states.append(
            dict(
                target=target,
                run_id=run,
                phase=digest(phase),
                journal=digest(events),
                control=digest(control),
                recovery=digest(docs.read("recorded-input-scope:result:1")),
            )
        )
    if not rows or requests != failure.get("new_requests"):
        raise ValueError("scope_successor_original_request_count")
    return dict(
        schema_version="accepted-scope-successor-proof/1.0.0",
        previous=str(path.absolute()),
        protocol_checksum=digest(old),
        failure_checksum=digest(failure),
        states=states,
        reused=rows,
        requests=requests,
    )


def inventory(parent, predecessor, output, *, frozen_methods=None):
    current = predecessor.protocol.get("schema_version") in window.CURRENT_VERSIONS
    route = predecessor.protocol.get("route_transition") if current else None
    tip, found, reader = window.inventory(
        parent,
        output,
        frozen_methods=frozen_methods,
        profile_path=CURRENT_PROFILE if current else PROFILE,
        current=current,
        route_transition=route,
    )
    proof = inspect(predecessor)
    before = predecessor.protocol["capability_transition"]
    # Only the survey representation/Prompt changes, never medical definitions,
    # original facts, method hints or the prior accepted Review selection.
    expected = deepcopy(before)
    for name in ("profile", "scope_plan", "input_scope"):
        expected[name] = found[name]
    for value in (expected["transition"], expected["transition"]["catalog_plan"]):
        value["profile_checksum"] = found["profile"]["checksum"]
    if current:
        # Original Phase.review_reuse_preview binds its state to the NEW
        # deterministic preview run and projected profile. The original owner
        # has just checked the accepted subset. Keep policy/source/assessment
        # checks exact; only this per-window state checksum is regenerated.
        old_preview = expected["confirmed_review"]["compatibility_preview"]
        new_preview = found["confirmed_review"]["compatibility_preview"]
        if any(
            value.get("schema_version") != "review-compatibility-preview/1.0.0"
            or value.get("execution_authorized") is not False
            or value.get("current_scope_required_before_dispatch") is not True
            for value in (old_preview, new_preview)
        ):
            raise ValueError("scope_successor_preview_contract")
        old_preview["state_checksum"] = new_preview["state_checksum"]
    if found != expected:
        raise ValueError("scope_successor_medical_inventory_changed")
    if found["scope_plan"]["entries"] != before["scope_plan"]["entries"]:
        raise ValueError("scope_successor_task_or_view_changed")
    reuse = {r["target"] for r in proof["reused"]}
    retained = found.get("current_scope_reuse")
    if retained:
        reuse.update(r["target"] for r in retained["reused"])
    found["input_scope"]["dispatch_targets"] = [
        t for t in found["input_scope"]["targets"] if t not in reuse
    ]
    found["scope_reuse"] = proof
    return tip, found, reader


def plan_for(parent, predecessor, tip, found, **kwargs):
    proof = found["scope_reuse"]
    adjusted_tip = deepcopy(tip)
    attempts = deepcopy(predecessor.protocol["previous_attempts"])
    attempts.append(
        dict(
            directory=proof["previous"],
            checksum=proof["failure_checksum"],
            requests=proof["requests"],
            kind="accepted-scope-local-recovery/1.0.0",
        )
    )
    adjusted_tip["previous_attempts"] = attempts
    pending = deepcopy(found)
    pending["input_scope"]["targets"] = found["input_scope"]["dispatch_targets"]
    current = predecessor.protocol.get("schema_version") in window.CURRENT_VERSIONS
    plan = window.plan_for(
        parent,
        adjusted_tip,
        pending,
        current=current,
        route_transition=predecessor.protocol.get("route_transition") if current else None,
        **kwargs,
    )
    plan.update(
        schema_version=CURRENT_VERSION if current else VERSION,
        tip=tip,
        visible=parent.base.visible_value.checksum,
        campaign_id=Path(kwargs["output"]).name,
        capability_transition=found,
        input_scope=found["input_scope"],
        scope_reuse=proof,
        prior_deadline=predecessor.protocol["deadline"],
    )
    return plan


def claim_key(proof):
    return "capability:accepted-scope-successor:" + proof["protocol_checksum"][7:]


def claim_documents(predecessor, proof):
    return predecessor.documents(proof["reused"][0]["fact"]["phase"]["run_id"])


def check(parent, plan, *, require_claim=True, with_predecessor=False):
    if plan.get("schema_version") not in VERSIONS:
        raise ValueError("scope_successor_version")
    predecessor = source(parent, plan["scope_reuse"]["previous"])
    tip, found, reader = inventory(
        parent, predecessor, plan["output"], frozen_methods=plan["procedure_reuse"]
    )
    expected = plan_for(
        parent,
        predecessor,
        tip,
        found,
        output=plan["output"],
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
        seconds=plan["seconds"],
    )
    if plan != expected:
        raise ValueError("scope_successor_current_binding_changed")
    docs = claim_documents(predecessor, found["scope_reuse"])
    if require_claim and docs.read(claim_key(found["scope_reuse"])) != (
        window.claim(plan, plan["output"]),
    ):
        raise ValueError("scope_successor_claim_binding")
    return (reader, predecessor) if with_predecessor else reader


def inherited(campaign):
    """Current original-owner survey read; saved proof alone never authorizes use."""
    plan = getattr(campaign, "protocol", {})
    if plan.get("schema_version") in window.CURRENT_VERSIONS - {CURRENT_VERSION}:
        from .input_scope import current_plan

        proof = current_proof(campaign.parent, current_plan(campaign.base))
        if proof != plan.get("current_scope_reuse"):
            raise ValueError("scope_current_reuse_owner_changed")
        return {r["target"]: r["fact"] for r in proof["reused"]} if proof else {}
    if plan.get("schema_version") not in VERSIONS:
        return {}
    from .finalization import read_finalized_scope
    from .input_scope import current_plan

    predecessor = getattr(campaign.base, "scope_reuse_predecessor", None)
    if predecessor is None:
        predecessor = source(campaign.parent, plan["scope_reuse"]["previous"])
        campaign.base.scope_reuse_predecessor = predecessor
    # Reuse an assembled owner, never a cached authorization or fact. The
    # original domain/input/source/Control/receipt readers run again below.
    check_history(plan)
    if str(predecessor.output.absolute()) != plan["scope_reuse"]["previous"]:
        raise ValueError("scope_successor_predecessor_binding")
    entries = {e["target"]: e for e in current_plan(campaign.base)["entries"]}
    rows = {}
    if plan.get("schema_version") == CURRENT_VERSION:
        proof = current_proof(campaign.parent, current_plan(campaign.base))
        if proof != plan.get("current_scope_reuse"):
            raise ValueError("scope_current_reuse_owner_changed")
        rows = {r["target"]: r["fact"] for r in proof["reused"]} if proof else {}
    for bound in plan["scope_reuse"]["reused"]:
        target = bound["target"]
        fact = read_finalized_scope(predecessor, target)
        if (
            target not in entries
            or entries[target]["task"] != fact["task"]
            or fact != bound["fact"]
            or digest(fact) != bound["fact_checksum"]
        ):
            raise ValueError("scope_successor_reused_fact_changed")
        if target in rows and rows[target] != fact:
            raise ValueError("scope_successor_conflicting_scope_origin")
        rows[target] = fact
    if campaign.documents(plan["scope_reuse"]["reused"][0]["fact"]["phase"]["run_id"]).read(
        claim_key(plan["scope_reuse"])
    ) != (window.claim(plan, campaign.output),):
        raise ValueError("scope_successor_claim_binding")
    return rows


def current_proof(parent, scope_plan):
    """Requalify exact surveys from the bound parent, never from caller records."""
    from .input_scope import read_results

    if not getattr(parent, "scope_targets", ()):
        return None
    result = json.loads((parent.output / "result.json").read_bytes())
    final = result["phases"][-1]
    docs = parent.documents(final["run_id"])
    control = parent.sessions.read(final["run_id"])
    if (
        docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-campaign") != (parent.protocol,)
        or control is None
        or control.phase != "stopped"
        or control.pending
        or (parent.output / "cancel").exists()
    ):
        raise ValueError("scope_current_reuse_parent_not_terminal")
    closure = None
    if control.cancelled:
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        from .acquired_review import projection_terminal_control

        # Original failure cleanup can cancel an unlaunched Review while its
        # earlier accepted Scope facts stay intact. Prove that exact closure;
        # a user cancel or an attempted/UNKNOWN Review never takes this path.
        if (
            final.get("target") is not None
            or final.get("accepted")
            or final.get("acceptance_key") is not None
            or docs.read("dynamic-phase-result") != (final,)
            or any(
                e.get("phase") in {"reserved", "completed"}
                for e in RuntimeAgentJournal(docs).read()
            )
        ):
            raise ValueError("scope_current_reuse_unproven_cancellation")
        try:
            closure = projection_terminal_control(docs, control, final, parent.output)
        except ValueError as error:
            raise ValueError("scope_current_reuse_unproven_cancellation") from error
    facts = read_results(parent, require_all=True)
    old = {e["target"]: e for e in parent.base.input_scope_plan["entries"]}
    new = {e["target"]: e for e in scope_plan["entries"]}
    rows = []
    for fact in facts:
        target = fact["phase"]["target"]
        if target not in new or old.get(target) != new[target]:
            continue
        if not fact.get("observation") or fact["task"] != new[target]["task"]:
            raise ValueError("scope_current_reuse_task_binding")
        rows.append(dict(target=target, fact=fact, fact_checksum=digest(fact)))
    return dict(
        schema_version="current-owner-scope-reuse/1.1.0"
        if closure is not None
        else "current-owner-scope-reuse/1.0.0",
        **(
            {"terminal_closure_checksum": closure, "terminal_control_checksum": digest(control)}
            if closure is not None
            else {}
        ),
        parent_run_id=final["run_id"],
        parent_protocol_checksum=digest(parent.protocol),
        parent_result_checksum=digest(result),
        reused=rows,
    )


def check_history(plan):
    proof = plan["scope_reuse"]
    path = Path(proof["previous"])
    if (
        digest(json.loads((path / "protocol.json").read_bytes())) != proof["protocol_checksum"]
        or digest(json.loads((path / "stage-failure.json").read_bytes()))
        != proof["failure_checksum"]
    ):
        raise ValueError("scope_successor_original_history_changed")


def verify_budget_history(parent, attempt):
    """Qualify the original recovered survey entry without making it a disease check.

    A saved kind string is not authority. Find its bound successor in the actual
    ancestry, then reread the original failed request, recovered fact and claim.
    """
    from .campaign_renewal import MAX_HISTORY_DEPTH

    path = str(Path(attempt["directory"]).absolute())
    current, seen = parent, set()
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if current is None:
            break
        if id(current) in seen:
            raise ValueError("scope_budget_history_cycle")
        seen.add(id(current))
        plan = current.protocol
        proof = plan.get("scope_reuse")
        if plan.get("schema_version") in VERSIONS and proof and proof["previous"] == path:
            expected = dict(
                directory=proof["previous"],
                checksum=proof["failure_checksum"],
                requests=proof["requests"],
                kind="accepted-scope-local-recovery/1.0.0",
            )
            if attempt != expected or any(
                plan.get(k) != parent.protocol.get(k) for k in ("input", "task")
            ):
                raise ValueError("scope_budget_history_binding")
            predecessor = getattr(current.base, "scope_reuse_predecessor", None)
            if predecessor is None:
                predecessor = source(current.parent, path)
            if str(predecessor.output.absolute()) != path or inspect(predecessor) != proof:
                raise ValueError("scope_budget_history_original_owner_changed")
            # This also checks the exact saved Runtime successor claim and the
            # current input/source/acceptance owner of each recovered survey.
            inherited(current)
            return
        current = getattr(current, "parent", None)
    raise ValueError("scope_budget_history_owner_missing")


def prepare(*, previous, output, authorization, check_only=False):
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
        raise ValueError("scope_successor_output_must_not_change_history")
    old = json.loads((previous / "protocol.json").read_bytes())
    with historical_fact_read_scope(), _projection_visible_reads():
        progress("restore_original_fact_owner")
        parent, _ = completed_parent(old["facts"])
        progress("restore_stopped_survey_owner")
        predecessor = source(parent, previous)
        with session(predecessor):
            check_network(window.route_configuration(parent, old.get("route_transition")))
            progress("qualify_current_inventory_and_accepted_survey")
            tip, found, _ = inventory(
                parent, predecessor, out, frozen_methods=old["procedure_reuse"]
            )
            progress("current_inventory_verified")
    # Read scopes finish their fresh return checks before any Runtime write.
    # Start a new window only after that potentially expensive preparation.
    with session(predecessor):
        docs = claim_documents(predecessor, found["scope_reuse"])
        saved = docs.read(claim_key(found["scope_reuse"]))
        if saved and (len(saved) != 1 or saved[0].get("output") != str(out)):
            raise ValueError("scope_successor_already_claimed")
        plan = plan_for(
            parent,
            predecessor,
            tip,
            found,
            output=out,
            authorization=authorization,
            now=datetime.fromisoformat(saved[0]["plan"]["started_at"])
            if saved
            else datetime.now(UTC),
            implementation=freeze(),
        )
        check_history(plan)
        for state in found["scope_reuse"]["states"]:
            control = predecessor.sessions.read(state["run_id"])
            if state.get("state") == "never_registered":
                if (
                    control is not None
                    or predecessor.contexts.read_research_run(state["run_id"], missing_ok=True)
                    is not None
                ):
                    raise ValueError("scope_successor_phase_changed_before_claim")
            elif digest(control) != state["control"]:
                raise ValueError("scope_successor_control_changed_before_claim")
        if (previous / "cancel").exists():
            raise ValueError("scope_successor_cancelled_before_claim")
        if saved and saved != (window.claim(plan, out),):
            raise ValueError("scope_successor_saved_claim_changed")
        if not check_only and not saved:
            if out.exists():
                raise ValueError("scope_successor_new_directory_required")
            with docs.session():
                if docs.read(claim_key(found["scope_reuse"])) is not None:
                    raise ValueError("scope_successor_already_claimed")
                docs.write(claim_key(found["scope_reuse"]), (window.claim(plan, out),))
    if not check_only:
        out.mkdir(parents=True, exist_ok=True)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        check_only=check_only,
        protocol_checksum=digest(plan),
        previous_requests=sum(r["requests"] for r in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        deadline=plan["deadline"],
        reused_scope_targets=[r["target"] for r in found["scope_reuse"]["reused"]],
        dispatch_scope_targets=plan["input_scope"]["dispatch_targets"],
        pending_review_ids=sorted(
            set(plan["review_target_ids"]) - set(plan["reuse_assessment_ids"])
        ),
        api_requests=0,
        database_writes=0 if check_only or saved else 1,
    )


def budget_predecessor(campaign):
    """Follow a qualified recovery branch for historical budget, never new facts."""
    plan = campaign.protocol
    if plan.get("schema_version") != CURRENT_VERSION:
        raise ValueError("scope_budget_predecessor_version")
    check_history(plan)
    proof = plan["scope_reuse"]
    prior = getattr(campaign.base, "scope_reuse_predecessor", None)
    if prior is None:
        prior = source(campaign.parent, proof["previous"])
    if (
        str(prior.output.absolute()) != proof["previous"]
        or digest(prior.protocol) != proof["protocol_checksum"]
        or any(prior.protocol.get(k) != plan.get(k) for k in ("input", "task"))
        or claim_documents(prior, proof).read(claim_key(proof))
        != (window.claim(plan, campaign.output),)
    ):
        raise ValueError("scope_budget_predecessor_binding")
    return prior

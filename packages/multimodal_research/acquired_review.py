"""Read-only proof for text closure of confirmed observations after acquisition stops.

No acquisition retry, new window, medical inference or parallel persistence owner.
"""

from datetime import UTC, datetime

from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal


def projection_terminal_control(documents, control, phase, path):
    """Explain original automatic pending-review closure, never undo cancel."""
    from pathlib import Path

    if (
        control is None
        or control.phase != "stopped"
        or control.pending
        or (Path(path) / "cancel").exists()
    ):
        raise ValueError("acquisition_review_projection_terminal_control")
    closed = documents.read("multimodal-failed-review-cancellation")
    if control.cancelled and (
        not closed
        or len(closed) != 1
        or closed[0].get("schema_version") != "multimodal-review-cancellation/1.0.0"
        or closed[0].get("before") != phase.get("planner", {}).get("control_checksum")
        or closed[0].get("after") != digest(control)
        or not control.events
        or closed[0].get("event") != control.events[-1].model_dump(mode="json")
        or control.events[-1].command.key != "live-review:review"
        or control.events[-1].command.action != "review"
        or control.events[-1].after != "stopped"
        or control.events[-1].outcome.reason != "cancelled"
        or control.events[-1].outcome.started != 0
        or control.events[-1].outcome.confirmed != 0
        or closed[0].get("cancelled") is not True
        or closed[0].get("new_requests") != 0
        or closed[0].get("remote_response_unknown_preserved") is not True
    ):
        raise ValueError("acquisition_review_projection_unproven_cancellation")
    return digest(closed)


def projection_repair_proof(parent, result, tip):
    """Prove accepted acquisitions omitted from a zero-wire inherited Review.

    This does not accept a model answer or supply a direction. The old accepted
    reader remains authoritative for historical judgments; new observations are
    read through their actual acquisition acceptance owner.
    """
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.grounding.models import validate

    from .campaign_renewal import (
        EXACT_ACQUISITION_VERSIONS,
        accepted_reader,
        acquisition_review_ids,
    )

    if parent.protocol.get("schema_version") in EXACT_ACQUISITION_VERSIONS:
        return retained_projection_repair_proof(parent, result, tip)
    old = parent.protocol
    phases = result.get("phases", [])
    if (
        old.get("schema_version") != "multimodal-campaign-renewal/1.26.0"
        or old.get("mode") != "acquisition"
        or old.get("review_partition") != "owner_bound_reused_and_new/1.0.0"
        or old.get("acquisition_increment_only")
        or result.get("complete") is not False
        or len(phases) != len(old.get("acquisition_targets", [])) + 1
        or [p.get("target") for p in phases[:-1]] != old["acquisition_targets"]
        or any(not p.get("accepted") for p in phases[:-1])
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted")
        or phases[-1].get("acceptance_key") is not None
        or phases[-1].get("error") != "batch_incomplete_or_reused_request"
        or phases[-1].get("run_id") != tip["run_id"]
        or result.get("previous_attempts") != old["previous_attempts"]
        or parent.config.transport.max_retries != 0
    ):
        raise ValueError("acquisition_review_projection_origin")
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(docs.run_id)
    events = RuntimeAgentJournal(docs).read()
    closure = projection_terminal_control(docs, control, phases[-1], tip["path"])
    if (
        docs.read("dynamic-phase-result") != (phases[-1],)
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-campaign") != (old,)
        or any(e.get("phase") in {"reserved", "completed"} for e in events)
    ):
        raise ValueError("acquisition_review_projection_not_zero_wire_terminal")
    old_state = validate(DomainState, docs.read("dynamic-phase-input")[0])
    tasks = project_review_tasks(old_state)
    reader = accepted_reader(
        parent.parent, old["tip"]["run_id"], recorded_origin=old.get("recorded_review_origin")
    )
    origin = reader.current(old_state)
    selected = acquisition_review_ids(old)
    if set(tasks.assessment_ids) != set(old["reuse_assessment_ids"]) or set(
        origin.assessment_ids
    ) != set(old["reuse_assessment_ids"]):
        raise ValueError("acquisition_review_projection_not_inherited_subset")
    observations = parent.accepted_observations()
    if any(
        t not in observations or observations[t].status != "obtained"
        for t in old["acquisition_targets"]
    ):
        raise ValueError("acquisition_review_projection_observation_missing")
    return dict(
        old_state_checksum=digest(old_state),
        origin_checksum=digest(origin),
        control_checksum=digest(control),
        journal_checksum=digest(events),
        phase_checksum=digest(phases[-1]),
        terminal_closure_checksum=closure,
        review_target_ids=list(selected),
        new_assessment_ids=old["acquisition_targets"],
        reuse_assessment_ids=old["reuse_assessment_ids"],
        observation_checksums={t: digest(observations[t]) for t in old["acquisition_targets"]},
        original_failure_preserved=True,
        old_review_requests=0,
    )


def validate_projection_repair(parent, plan):
    import json
    from datetime import timedelta
    from pathlib import Path

    from .campaign_renewal import ACQUISITION_REVIEW_VERSION, RETAINED_REVIEW_VERSION, stopped_tip

    if plan.get("schema_version") == RETAINED_REVIEW_VERSION:
        return validate_retained_projection_repair(parent, plan)
    result = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    tip = stopped_tip(parent, plan["facts"])
    proof = projection_repair_proof(parent, result, tip)
    batches = (
        len(proof["new_assessment_ids"]) + parent.config.review_batching.batch_size - 1
    ) // parent.config.review_batching.batch_size
    if (
        plan.get("schema_version") != ACQUISITION_REVIEW_VERSION
        or proof.get("retained_context_scope") is not None
        or plan.get("mode") != "review"
        or plan["tip"] != tip
        or plan.get("review_acquisition_gap") != proof
        or plan["facts"] != tip["path"]
        or plan["original_fact_result"] != digest(result)
        or plan["previous_attempts"] != tip["previous_attempts"]
        or plan["provider"] != digest(parent.config)
        or plan["input"] != parent.base.manifest.checksum
        or plan["task"] != parent.base.task.checksum
        or plan["review_target_ids"] != proof["review_target_ids"]
        or plan["reuse_assessment_ids"] != proof["reuse_assessment_ids"]
        or plan.get("acquisition_targets") != []
        or plan.get("expansion_policy") is not None
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan["seconds"] != 5400
        or plan["max_model_dispatches"] != batches
        or plan["max_new_requests"] != batches + 1
        or plan["max_cumulative_requests"] != 40
        or sum(p["requests"] for p in tip["previous_attempts"]) + batches + 1 > 40
        or plan["deadline"]
        != (datetime.fromisoformat(plan["started_at"]) + timedelta(seconds=5400)).isoformat()
        or plan["prior_deadline"] != parent.protocol["deadline"]
        or plan["confirmed_observations_checksum"] != digest(parent.accepted_observations())
    ):
        raise ValueError("acquisition_review_projection_plan_binding")


def prepare_projection_repair(
    *,
    facts,
    previous,
    output,
    mode,
    authorization,
    request_limit=None,
    review_acquisition_gap=False,
    recovery_network_route=None,
    **options,
):
    from datetime import timedelta
    from pathlib import Path

    from apps.vectorengine_single_case import write_once

    from .campaign_renewal import (
        ACQUISITION_REVIEW_VERSION,
        RETAINED_REVIEW_VERSION,
        completed_parent,
        recovery_configuration,
        stopped_tip,
    )
    from .implementation import freeze

    if (
        not review_acquisition_gap
        or mode != "review"
        or any(options.values())
        or Path(facts).absolute() != Path(previous).absolute()
        or not authorization.strip()
        or len(authorization) > 200
    ):
        raise ValueError("acquisition_review_projection_explicit_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    parent, result = completed_parent(facts)
    parent.base.check()
    tip = stopped_tip(parent, previous)
    proof = projection_repair_proof(parent, result, tip)
    size = parent.config.review_batching.batch_size
    batches = (len(proof["new_assessment_ids"]) + size - 1) // size
    retained = proof.get("retained_context_scope") is not None
    config, route = parent.config, None
    if recovery_network_route is not None:
        from packages.llm_provider.network_route import load_route

        if not retained:
            raise ValueError("recovery_network_route_scope")
        route = load_route(recovery_network_route).model_dump(mode="json")
        config = recovery_configuration(
            config,
            dict(
                schema_version=RETAINED_REVIEW_VERSION,
                mode="review",
                recovery_network_route=route,
            ),
        )
    if retained:
        cap = retained_projection_bounds(proof, parent.config, request_limit)
    else:
        if request_limit not in (None, batches + 1):
            raise ValueError("acquisition_review_projection_request_limit")
        cap = batches + 1
    seconds = 2700 if retained else 5400
    now = datetime.now(UTC)
    plan = dict(
        schema_version=RETAINED_REVIEW_VERSION if retained else ACQUISITION_REVIEW_VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        output=str(out),
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        provider=digest(config),
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        implementation=freeze(),
        max_new_requests=cap,
        max_model_dispatches=cap if retained else batches,
        max_cumulative_requests=40,
        seconds=seconds,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=seconds)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        review_acquisition_gap=proof,
        review_target_ids=proof["review_target_ids"],
        reuse_assessment_ids=proof["reuse_assessment_ids"],
        acquisition_targets=[],
        expansion_policy=None,
        new_vision_requests=0,
        new_public_search_allowed=False,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
    )
    if retained:
        plan["review_context_scope"] = proof["retained_context_scope"]
    if route is not None:
        plan["recovery_network_route"] = route
    validate_projection_repair(parent, plan)
    docs = parent.documents(tip["run_id"])
    with docs.session():
        if docs.read("authorized-campaign-renewal") is not None:
            raise ValueError("renewal_tip_already_claimed")
        docs.write(
            "authorized-campaign-renewal", (dict(output=str(out), plan_checksum=digest(plan)),)
        )
    out.mkdir(mode=0o700)
    write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        mode=mode,
        api_requests=0,
        previous_requests=sum(p["requests"] for p in tip["previous_attempts"]),
        max_new_requests=cap,
        max_model_dispatches=cap if retained else batches,
        new_assessment_ids=proof["new_assessment_ids"],
        reused_assessment_ids=proof["reuse_assessment_ids"],
    )


def terminal_inventory(parent, path, protocol, result):
    phases = result.get("phases", [])
    if (
        protocol.get("schema_version")
        not in {"multimodal-campaign-renewal/1.17.0", "multimodal-campaign-renewal/1.18.0"}
        or protocol.get("mode") != "acquisition"
        or protocol.get("acquisition_increment_only") is not True
        or result.get("complete") is not False
        or result.get("previous_attempts") != protocol["previous_attempts"]
        or result.get("case_code") != parent.base.binding["case_code"]
        or result.get("task") != parent.base.task.model_dump(mode="json")
        or protocol["input"] != parent.base.manifest.checksum
        or protocol["task"] != parent.base.task.checksum
        or not phases
        or [p.get("target") for p in phases] != protocol["acquisition_targets"][: len(phases)]
        or any(p.get("target") is None for p in phases)
        or not any(p.get("accepted") for p in phases)
        or phases[-1].get("accepted")
        or (path / "cancel").exists()
    ):
        raise ValueError("acquired_review_identity")
    journals, count = [], 0
    for index, phase in enumerate(phases):
        docs = parent.documents(phase["run_id"])
        control = parent.sessions.read(docs.run_id)
        if (
            control is None
            or control.phase != "stopped"
            or control.cancelled
            or control.pending is not None
            or docs.read("dynamic-campaign") != (protocol,)
            or docs.read("dynamic-phase-result") != (phase,)
            or (not phase.get("accepted") and index != len(phases) - 1)
        ):
            raise ValueError("acquired_review_not_terminal")
        events = RuntimeAgentJournal(docs).read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        records = [e["record"] for e in events if e.get("phase") == "completed"]
        confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
        if (
            len(reserved) != 1
            or len(records) != 1
            or len(confirmations) != 1
            or confirmations[0]
            != dict(phase="response_confirmation", request_id=reserved[0], confirmed=True)
        ):
            raise ValueError("acquired_review_unknown")
        record = records[0]
        if (
            record.get("request_id") != reserved[0]
            or record.get("role") != "vision"
            or record.get("simulated") is not False
            or record.get("api_requests") != 1
            or record.get("retry_index") != 0
            or record["scope"]["run_id"] != docs.run_id
            or record["scope"]["case_code"] != result["case_code"]
            or record["scope"]["input_checksum"] != protocol["input"]
        ):
            raise ValueError("acquired_review_request_binding")
        if phase["accepted"]:
            if not phase.get("acceptance_key") or record.get("error") is not None:
                raise ValueError("acquired_review_acceptance_binding")
        else:
            diagnostics = [
                e["value"]
                for e in events
                if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
                and e.get("request_id") == reserved[0]
            ]
            if (
                phase.get("acceptance_key")
                or record.get("error") != "rate_limit"
                or record.get("output_checksum") is not None
                or len(diagnostics) != 1
                or diagnostics[0].get("http_status") != 429
                or diagnostics[0].get("category") not in {"rate_limit", "unclassified_429"}
            ):
                raise ValueError("acquired_review_unconfirmed_failure")
        count += len(reserved)
        journals.append(digest(events))
    if count != result.get("new_requests") or docs.read("multimodal-case-result") != (result,):
        raise ValueError("acquired_review_accounting")
    # Original observation owner checks receipts, scope, exact image and records.
    observations = parent.accepted_observations()
    selected = [p["target"] for p in phases if p["accepted"]]
    if any(t not in observations or observations[t].status != "obtained" for t in selected) or set(
        selected
    ) & set(protocol["reuse_assessment_ids"]):
        raise ValueError("acquired_review_observation_binding")
    return dict(
        path=str(path),
        result=digest(result),
        protocol=digest(protocol),
        journals=journals,
        previous_attempts=protocol["previous_attempts"]
        + [dict(directory=str(path), requests=count, checksum=digest(result))],
        run_id=docs.run_id,
        deadline=protocol["deadline"],
        acquired_review=dict(
            selected_ids=selected,
            observations={t: digest(observations[t]) for t in selected},
            historical_review_ids=protocol["reuse_assessment_ids"],
            remaining_requests=protocol["max_new_requests"] - count,
            deadline=protocol["deadline"],
            new_acquisition_allowed=False,
        ),
    )


def bounds(tip, config, requested_cap, now=None):
    proof = tip["acquired_review"]
    deadline = datetime.fromisoformat(proof["deadline"])
    seconds = int((deadline - (now or datetime.now(UTC))).total_seconds())
    count = len(proof["selected_ids"])
    dispatches = (
        count + config.review_batching.batch_size - 1
    ) // config.review_batching.batch_size
    # No retry route: a failed text request is kept, never loop until a score.
    if config.transport.max_retries != 0:
        raise ValueError("acquired_review_no_retry_required")
    cap = min(requested_cap, proof["remaining_requests"])
    if seconds <= 300 or dispatches < 1 or cap < dispatches + 1:
        raise ValueError("acquired_review_window_exhausted")
    return deadline, seconds, cap, dispatches


def contract_repair_tip(parent, path, protocol, result):
    """One explicit contract repair after confirmed HTTP400; not recursive retry."""
    from packages.llm_provider.config import qualified_citation_upgrade

    from .campaign_renewal import stopped_tip

    phases = result.get("phases", [])
    if (
        protocol.get("schema_version") != "multimodal-campaign-renewal/1.19.0"
        or protocol.get("review_acquired_only") is not True
        or result.get("complete") is not False
        or len(phases) != 1
        or phases[0].get("target") is not None
        or phases[0].get("accepted")
        or phases[0].get("acceptance_key")
        or result.get("new_requests") != 1
        or result.get("previous_attempts") != protocol["previous_attempts"]
        or protocol["input"] != parent.base.manifest.checksum
        or protocol["task"] != parent.base.task.checksum
        or result.get("case_code") != parent.base.binding["case_code"]
        or result.get("task") != parent.base.task.model_dump(mode="json")
        or (path / "cancel").exists()
    ):
        raise ValueError("acquired_contract_repair_identity")
    phase = phases[0]
    docs = parent.documents(phase["run_id"])
    control = parent.sessions.read(docs.run_id)
    if (
        control is None
        or control.phase != "stopped"
        or control.cancelled
        or control.pending
        or docs.read("dynamic-campaign") != (protocol,)
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("multimodal-case-result") != (result,)
    ):
        raise ValueError("acquired_contract_repair_not_terminal")
    events = RuntimeAgentJournal(docs).read()
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    if len(reserved) != 1 or len(records) != 1 or len(confirmations) != 1:
        raise ValueError("acquired_contract_repair_unknown")
    record = records[0]
    diagnostics = [
        e["value"]
        for e in events
        if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
        and e.get("request_id") == reserved[0]
    ]
    if (
        confirmations[0]
        != dict(phase="response_confirmation", request_id=reserved[0], confirmed=True)
        or record.get("request_id") != reserved[0]
        or record.get("role") != "review"
        or record.get("error") != "configuration"
        or record.get("simulated") is not False
        or record.get("api_requests") != 1
        or record.get("retry_index") != 0
        or record.get("output_checksum") is not None
        or record["scope"]["run_id"] != docs.run_id
        or record["scope"]["case_code"] != result["case_code"]
        or record["scope"]["input_checksum"] != protocol["input"]
        or len(diagnostics) != 1
        or diagnostics[0].get("http_status") != 400
    ):
        raise ValueError("acquired_contract_repair_unknown")
    original = stopped_tip(parent.parent, protocol["tip"]["path"])
    if original != protocol["tip"]:
        raise ValueError("acquired_contract_repair_origin_changed")
    proof = original["acquired_review"]
    if (
        protocol["review_target_ids"] != proof["selected_ids"]
        or protocol["deadline"] != proof["deadline"]
        or protocol["confirmed_observations_checksum"] != digest(parent.accepted_observations())
    ):
        raise ValueError("acquired_contract_repair_observations_changed")
    config = qualified_citation_upgrade(parent.config)
    # No immutable output/body is edited. Only the later window uses the new contract.
    return dict(
        path=str(path),
        result=digest(result),
        protocol=digest(protocol),
        journals=[digest(events)],
        previous_attempts=protocol["previous_attempts"]
        + [dict(directory=str(path), requests=1, checksum=digest(result))],
        run_id=docs.run_id,
        deadline=protocol["deadline"],
        acquired_review=dict(proof, remaining_requests=proof["remaining_requests"] - 1),
        contract_repair=dict(
            old_provider=digest(parent.config),
            new_provider=digest(config),
            request_id=record["request_id"],
            request_checksum=digest(record),
            original_acquisition=original,
            further_contract_successors_allowed=False,
            reason="qualified_2_2_prompt_promised_owner_table_but_wire_omitted_it",
            http_400_cause_not_proven=True,
        ),
    )


def retained_projection_repair_proof(parent, result, tip):
    from copy import copy

    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.grounding.models import validate

    from .campaign_renewal import (
        EXACT_ACQUISITION_VERSIONS,
        RETAINED_CONTEXT_SCOPE,
        accepted_reader,
        acquisition_review_ids,
    )
    from .review_context import retain_observations

    old, phases = parent.protocol, result.get("phases", [])
    if (
        old.get("schema_version") not in EXACT_ACQUISITION_VERSIONS
        or old.get("mode") != "acquisition"
        or old.get("review_partition") != "owner_bound_reused_and_new/1.0.0"
        or old.get("acquisition_increment_only")
        or result.get("complete") is not False
        or len(phases) != len(old.get("acquisition_targets", [])) + 1
        or [p.get("target") for p in phases[:-1]] != old["acquisition_targets"]
        or any(not p.get("accepted") for p in phases[:-1])
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted")
        or phases[-1].get("acceptance_key") is not None
        or phases[-1].get("error") != "accepted_review_reuse_visible_context_changed"
        or phases[-1].get("run_id") != tip["run_id"]
        or result.get("previous_attempts") != old["previous_attempts"]
        or parent.config.transport.max_retries != 0
    ):
        raise ValueError("retained_review_projection_origin")
    parent.base.check()
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(tip["run_id"])
    events = RuntimeAgentJournal(docs).read()
    closure = projection_terminal_control(docs, control, phases[-1], tip["path"])
    if (
        docs.read("dynamic-phase-result") != (phases[-1],)
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-campaign") != (old,)
        or any(e.get("phase") in {"reserved", "completed"} for e in events)
    ):
        raise ValueError("retained_review_projection_not_zero_wire_terminal")
    failed = validate(DomainState, docs.read("dynamic-phase-input")[0])
    selected = acquisition_review_ids(old)
    if set(project_review_tasks(failed).assessment_ids) != set(selected):
        raise ValueError("retained_review_projection_selection_changed")
    reader = accepted_reader(
        parent.parent, old["tip"]["run_id"], recorded_origin=old.get("recorded_review_origin")
    )
    if digest(reader.state_reader()) != old["accepted_state_checksum"]:
        raise ValueError("retained_review_projection_accepted_state_changed")
    observations = parent.accepted_observations()
    if any(
        t not in observations or observations[t].status != "obtained"
        for t in old["acquisition_targets"]
    ):
        raise ValueError("retained_review_projection_observation_missing")
    base = copy(parent.base)
    base.review_context_scope = RETAINED_CONTEXT_SCOPE
    base.confirmed_review_reader = reader
    retained = retain_observations(
        base,
        {**base.observations, **observations},
        {o.reference.code: o for o in failed.material.observations},
        (),
    )
    if not retained:
        raise ValueError("retained_review_projection_no_missing_context")
    raw = failed.model_dump(mode="json")
    raw["material"]["observations"].extend(o.model_dump(mode="json") for o in retained)
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    candidate = validate(DomainState, raw)
    origin = reader.current(candidate)
    if set(origin.assessment_ids) != set(old["reuse_assessment_ids"]):
        raise ValueError("retained_review_projection_reuse_changed")
    return dict(
        old_state_checksum=digest(failed),
        origin_checksum=digest(origin),
        control_checksum=digest(control),
        journal_checksum=digest(events),
        phase_checksum=digest(phases[-1]),
        terminal_closure_checksum=closure,
        review_target_ids=list(selected),
        new_assessment_ids=sorted(set(selected) - set(origin.assessment_ids)),
        reuse_assessment_ids=list(origin.assessment_ids),
        observation_checksums={t: digest(observations[t]) for t in old["acquisition_targets"]},
        retained_context_scope=RETAINED_CONTEXT_SCOPE,
        retained_context_checksums={o.reference.code: digest(o) for o in retained},
        unused_original_request_allowance=old["max_new_requests"] - result["new_requests"],
        original_failure_preserved=True,
        old_review_requests=0,
    )


def retained_projection_bounds(proof, config, requested=None):
    size = config.review_batching.batch_size
    needed = (len(proof["new_assessment_ids"]) + size - 1) // size
    remaining = proof["unused_original_request_allowance"]
    cap = min(remaining, needed + 1) if requested is None else requested
    if (
        not needed
        or type(remaining) is not int
        or type(cap) is not int
        or not needed <= cap <= min(remaining, 8)
        or config.transport.max_retries != 0
    ):
        raise ValueError("retained_review_projection_request_limit")
    return cap


def validate_retained_projection_repair(parent, plan):
    import json
    from datetime import timedelta
    from pathlib import Path

    from .campaign_renewal import (
        RETAINED_CONTEXT_SCOPE,
        RETAINED_REVIEW_VERSION,
        recovery_configuration,
        stopped_tip,
    )

    result = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    tip = stopped_tip(parent, plan["facts"])
    proof = retained_projection_repair_proof(parent, result, tip)
    cap = retained_projection_bounds(proof, parent.config, plan.get("max_new_requests"))
    if (
        plan.get("schema_version") != RETAINED_REVIEW_VERSION
        or plan.get("mode") != "review"
        or plan.get("review_context_scope") != RETAINED_CONTEXT_SCOPE
        or plan["tip"] != tip
        or plan.get("review_acquisition_gap") != proof
        or plan["facts"] != tip["path"]
        or plan["original_fact_result"] != digest(result)
        or plan["previous_attempts"] != tip["previous_attempts"]
        or plan["provider"] != digest(recovery_configuration(parent.config, plan))
        or plan["input"] != parent.base.manifest.checksum
        or plan["task"] != parent.base.task.checksum
        or plan["review_target_ids"] != proof["review_target_ids"]
        or plan["reuse_assessment_ids"] != proof["reuse_assessment_ids"]
        or plan.get("acquisition_targets") != []
        or plan.get("expansion_policy") is not None
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan["seconds"] != 2700
        or plan["max_model_dispatches"] != cap
        or plan["max_cumulative_requests"] != 40
        or sum(p["requests"] for p in tip["previous_attempts"]) + cap > 40
        or plan["deadline"]
        != (datetime.fromisoformat(plan["started_at"]) + timedelta(seconds=2700)).isoformat()
        or plan["prior_deadline"] != parent.protocol["deadline"]
        or plan["confirmed_observations_checksum"] != digest(parent.accepted_observations())
    ):
        raise ValueError("retained_review_projection_plan_binding")

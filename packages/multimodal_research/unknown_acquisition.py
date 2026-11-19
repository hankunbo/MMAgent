"""Explicit new analytical attempts, never replay an unresolved tool or old run.

Original Runtime documents and Provider receipts establish the failed attempt.
The ordinary campaign owners still establish current input/source admission,
budget, single successor, execution lease, acceptance and scoring.
"""

import json
from copy import copy
from datetime import datetime
from pathlib import Path

from agents.expert_agent.models import ReviewFrame
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.live_permission import ResearchLiveGrant
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider.models import read_request_record
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal

VERSION = "multimodal-campaign-renewal/1.47.0"
LOCAL_VERSION = "multimodal-campaign-renewal/1.48.0"
CONFIGURATION_VERSION = "multimodal-campaign-renewal/1.51.0"
CONFIRMED_INVALID_VERSION = "multimodal-campaign-renewal/1.66.0"
HISTORICAL_CLIENT_VERSION = "multimodal-campaign-renewal/1.67.0"
POST_DEFINITION_VERSION = "multimodal-campaign-renewal/1.70.0"
VERSIONS = {
    POST_DEFINITION_VERSION,
    VERSION,
    LOCAL_VERSION,
    CONFIGURATION_VERSION,
    CONFIRMED_INVALID_VERSION,
    HISTORICAL_CLIENT_VERSION,
}


def historical_client_attempt(parent, path, phase, request_id):
    """Prove the old owner under its exact client, then recheck current material.

    This permits only the already bound HTTP-client upgrade, not another model,
    prompt, transport route, definition, image projection or source change.
    """
    from packages.llm_provider.client_selection import apply_selection
    from packages.llm_provider.config import read_provider_config

    from .campaign import Phase

    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("historical_acquisition_origin_symlink")
    protocol = json.loads((path / "protocol.json").read_bytes())
    entry = json.loads((path / "entry.json").read_bytes())
    old_config = read_provider_config(entry["provider"])
    binding = parent.protocol.get("recovery_http_client")
    if (
        not binding
        or digest(old_config) != protocol.get("provider")
        or old_config == parent.config
        or apply_selection(old_config, binding) != parent.config
    ):
        raise ValueError("historical_acquisition_client_transition_binding")
    ancestor = copy(parent)
    ancestor.config = old_config
    original = original_attempt(ancestor, phase, request_id)
    docs = parent.documents(phase["run_id"])
    old = validate(DomainState, docs.read("dynamic-phase-input")[0])
    # The original context is read, never created or rewritten for this check.
    base = copy(parent.base)
    base.contexts = dict(parent.base.contexts)
    base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    projection = Phase(base, target=phase["target"])
    current, _ = projection.project(run_id=docs.run_id, config=parent.config)
    projection.verify(current)
    if (
        old.material != current.material
        or old.policy != current.policy
        or old.prompt_checksum != current.prompt_checksum
    ):
        raise ValueError("historical_acquisition_material_changed")
    return dict(
        original,
        historical_client_transition=dict(
            origin=str(path),
            original_entry_checksum=digest(entry),
            original_protocol_checksum=digest(protocol),
            original_provider=digest(old_config),
            current_provider=digest(parent.config),
            client_binding=binding,
            current_state_checksum=digest(current),
            exact_material_policy_and_vision_prompt=True,
        ),
    )


def original_attempt(
    parent,
    phase,
    request_id,
    *,
    local_configuration=False,
    confirmed_invalid=False,
    scope_owner=None,
):
    """No heuristic error-text authorization; require the exact original binding."""
    if local_configuration and confirmed_invalid:
        raise ValueError("confirmed_invalid_acquisition_mode_conflict")
    docs = parent.documents(phase["run_id"])
    control = parent.sessions.read(docs.run_id)
    if (
        docs.read("dynamic-phase-result") != (phase,)
        or phase.get("accepted") is not False
        or phase.get("acceptance_key") is not None
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or control.cancelled
        or control.started != 0
        or control.reserved != 0
        or phase.get("planner", {}).get("started_processes") != 0
        or phase.get("planner", {}).get("reserved_actions") != 0
    ):
        raise ValueError("unknown_acquisition_terminal_analysis_required")
    saved = docs.read("dynamic-phase-input")
    intents = docs.read("live-stage-intent:vision")
    if not saved or len(saved) != 1 or not intents or len(intents) != 1:
        raise ValueError("unknown_acquisition_original_intent_missing")
    initial_state = validate(DomainState, saved[0])
    # Generic phases historically save the initial Review prompt before the
    # original live assembly selects its Vision role. Validate exactly that
    # configured transition; never ignore a different material/policy/config.
    from packages.evidence_details.successor_owner import ResearchSuccessorOwner

    allowed_prompts = {
        r.prompt_checksum for r in parent.config.roles if r.role in {"vision", "review"}
    }
    if initial_state.prompt_checksum not in allowed_prompts:
        raise ValueError("unknown_acquisition_initial_prompt_binding")
    configured_owner = ResearchSuccessorOwner(
        initial=initial_state,
        documents=docs,
        verify_dependencies=lambda value: docs.validate_scope(value.material.scope),
        read_image=None,
        config=parent.config,
        role="vision",
    )
    # Pure historical role projection, not current permission or a new grant.
    state = configured_owner._configured(initial_state)
    scope, intent = state.material.scope, intents[0]
    docs.validate_scope(scope)
    event = intent["event"]
    prefix = "review:" + event["code"]
    original = docs.read(prefix + ":intent")
    if not original or len(original) != 1:
        raise ValueError("unknown_acquisition_original_intent_missing")
    frame = validate(ReviewFrame, original[0]["frame"])
    if (
        validate(DomainState, original[0]["state"]) != state
        or frame.scope != scope
        or frame.event.model_dump(mode="json") != event
        or intent["role"] != "vision"
        or intent["config_checksum"] != state.config_checksum
        or state.config_checksum != digest(parent.config)
        or scope.case_code != parent.base.binding["case_code"]
        or scope.input_checksum != parent.base.manifest.checksum
        or state.material.goal_code != parent.base.task.target_code
        or not any(
            a.id == intent["proposed_action_id"]
            and a.kind == "request_review"
            and a.branch == "research_vision"
            for a in frame.actions.actions
        )
        or docs.read(prefix + ":command")
        != (
            dict(
                event=event,
                proposed_action_id=intent["proposed_action_id"],
                role="vision",
                abc=False,
            ),
        )
    ):
        raise ValueError("unknown_acquisition_original_owner_binding")
    if docs.read("submission:" + event["code"] + ":domain") or docs.read(
        "accepted:" + event["code"] + ":domain"
    ):
        raise ValueError("unknown_acquisition_local_recovery_required")
    journal = RuntimeAgentJournal(docs)
    events = journal.read()
    records = [read_request_record(e["record"]) for e in events if e.get("phase") == "completed"]
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    if len(records) != 1 or reserved != [request_id] or records[0].request_id != request_id:
        raise ValueError("unknown_acquisition_exact_request_required")
    record = records[0]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    if not confirmed_invalid and (
        record.output_checksum is not None
        or docs.read("live-response:" + request_id) is not None
        or any(e.get("confirmed") is True for e in confirmations)
    ):
        raise ValueError("unknown_acquisition_local_recovery_required")
    response_proof = (
        confirmed_invalid_response(docs, record, confirmations) if confirmed_invalid else None
    )
    if (
        record.role != "vision"
        or record.scope != scope
        or record.config_checksum != state.config_checksum
        or record.prompt_checksum != state.prompt_checksum
        or record.simulated
        or record.api_requests != (0 if local_configuration else 1)
        or record.retry_index != 0
        or record.error
        not in (
            {"schema"}
            if confirmed_invalid
            else {"configuration"}
            if local_configuration
            else {"timeout", "transport"}
        )
        or record.endpoint != "chat_completions"
        or len(confirmations) != 1
        or confirmations[0].get("request_id") != request_id
        or confirmations[0].get("confirmed")
        is not (True if confirmed_invalid else False if local_configuration else None)
    ):
        raise ValueError("unknown_acquisition_request_not_eligible")
    grant = docs.read("research-live-grant")
    g = validate(ResearchLiveGrant, grant[0]) if grant and len(grant) == 1 else None
    if (
        g is None
        or not journal.durable
        or (
            g.run_id,
            g.case_code,
            g.input_checksum,
            g.visible_checksum,
            g.policy_checksum,
            g.config_checksum,
            g.root_checksum,
        )
        != (
            scope.run_id,
            scope.case_code,
            scope.input_checksum,
            scope.visible_checksum,
            scope.policy_checksum,
            state.config_checksum,
            docs.root.checksum,
        )
        or "vision" not in g.roles
        or record.deadline > g.deadline
    ):
        raise ValueError("unknown_acquisition_original_grant")
    tasks = project_review_tasks(state).tasks
    if len(tasks) != 1 or tasks[0].assessment_id != phase["target"]:
        raise ValueError("unknown_acquisition_original_target")
    payload = dict(
        role="vision",
        material=state.material.model_dump(mode="json"),
        schema_version="targeted-vision-input/1.0.0",
        vision_task=tasks[0].model_dump(mode="json"),
    )
    if scope_owner is not None:
        from packages.evidence_details.input_scope import payload as scope_payload

        from .input_scope import is_target

        if local_configuration or not is_target(phase["target"]):
            raise ValueError("unknown_scope_original_mode")
        payload = scope_payload(scope_owner, state)
        if confirmed_invalid:
            saved_response = docs.read("live-response:" + request_id)[0]
            if (
                digest(saved_response["raw_model_text"].encode("utf-8").hex())
                != record.output_checksum
            ):
                raise ValueError("confirmed_scope_original_output_binding")
    if digest(payload) != record.input_checksum:
        raise ValueError("unknown_acquisition_original_payload")
    value = dict(
        target=phase["target"],
        run_id=docs.run_id,
        request_id=request_id,
        record_checksum=digest(record),
        input_checksum=record.input_checksum,
        state_checksum=digest(state),
        journal_checksum=digest(events),
        grant_checksum=digest(g),
        old_response_state="not_sent" if local_configuration else "unknown",
        saved_response_present=False,
        remote_lookup="chat_receipt_has_no_retrieval_id",
        duplicate_billing_possible=not local_configuration,
        new_attempt_not_old_replay=True,
    )
    if confirmed_invalid:
        value.update(
            old_response_state="confirmed_invalid",
            saved_response_present=True,
            duplicate_billing_possible=True,
            response_proof=response_proof,
        )
    return value


def confirmed_invalid_response(docs, record, confirmations):
    """A saved, invalid response is neither UNKNOWN nor a medical observation."""
    saved = docs.read("live-response:" + record.request_id)
    if (
        record.error != "schema"
        or record.output_checksum is None
        or len(confirmations) != 1
        or confirmations[0].get("request_id") != record.request_id
        or confirmations[0].get("confirmed") is not True
        or not saved
        or len(saved) != 1
    ):
        raise ValueError("confirmed_invalid_acquisition_response_required")
    response = saved[0]
    raw = response.get("raw_model_text")
    if (
        response.get("schema_version") != "bounded-live-response/1.0.0"
        or response.get("request_id") != record.request_id
        or response.get("validation_error") != "schema"
        or response.get("accepted_medical_evidence") is not False
        or response.get("untrusted_model_output") is not True
        or not isinstance(raw, str)
        or not raw
        or response.get("raw_checksum") != checksum(raw.encode("utf-8"))
    ):
        raise ValueError("confirmed_invalid_acquisition_response_binding")
    return dict(response_checksum=digest(response), output_checksum=record.output_checksum)


def check_network(config):
    """Original frozen route/client, before a claim, credential read or reservation."""
    from packages.llm_provider.network_route import route_opener

    route_opener(getattr(config, "network_route", None), config.transport.base_url)
    client = getattr(config, "http_client", None)
    if client is not None:
        from packages.llm_provider.curl_client import verify_binary

        verify_binary(client)


def fact_review_run(parent, plan):
    """Separate the stopped predecessor from the original accepted fact owner."""
    if plan.get("schema_version") not in {
        LOCAL_VERSION,
        CONFIGURATION_VERSION,
        POST_DEFINITION_VERSION,
    }:
        if plan.get("accepted_fact_run_id"):
            raise ValueError("unknown_fact_origin_version")
        return plan["tip"]["run_id"]
    original = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    phases = original.get("phases", [])
    if plan.get("schema_version") == CONFIGURATION_VERSION and plan.get("recorded_review_origin"):
        from .campaign_renewal import recorded_acquisition_origin

        if (
            not phases
            or digest(original) != plan["original_fact_result"]
            or phases[-1].get("target") is not None
            or plan.get("accepted_fact_run_id") != phases[-1]["run_id"]
            or recorded_acquisition_origin(parent, phases[-1]["run_id"])
            != plan["recorded_review_origin"]
        ):
            raise ValueError("unknown_accepted_fact_origin_changed")
        return phases[-1]["run_id"]
    if (
        digest(original) != plan["original_fact_result"]
        or original.get("complete") is not True
        or not phases
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted") is not True
        or plan.get("accepted_fact_run_id") != phases[-1]["run_id"]
        or parent.documents(phases[-1]["run_id"]).read("dynamic-phase-result") != (phases[-1],)
    ):
        raise ValueError("unknown_accepted_fact_origin_changed")
    return phases[-1]["run_id"]


def local_configuration_tip(parent, path, protocol, result, *, ordinary=False):
    """One narrow predecessor: zero-dispatch first phase, not a stopped-run resume."""
    path = Path(path).absolute()
    phases = result.get("phases", [])
    if (
        protocol.get("schema_version")
        != ("multimodal-campaign-renewal/1.37.0" if ordinary else VERSION)
        or (bool(protocol.get("unknown_acquisition")) == ordinary)
        or (ordinary and len(protocol.get("acquisition_targets", [])) != 1)
        or (path / "cancel").exists()
        or result.get("complete") is not False
        or result.get("new_requests") != 1
        or len(phases) != 1
        or phases[0].get("target") != protocol["acquisition_targets"][0]
        or result.get("case_code") != parent.base.binding["case_code"]
        or result.get("task") != parent.base.task.model_dump(mode="json")
        or protocol["input"] != parent.base.manifest.checksum
        or protocol["task"] != parent.base.task.checksum
        or result.get("previous_attempts") != protocol.get("previous_attempts")
    ):
        raise ValueError("unknown_local_configuration_scope")
    phase = phases[0]
    docs = parent.documents(phase["run_id"])
    events = RuntimeAgentJournal(docs).read()
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    if (
        len(reserved) != 1
        or docs.read("dynamic-campaign") != (protocol,)
        or docs.read("multimodal-case-result") != (result,)
    ):
        raise ValueError("unknown_local_configuration_owner")
    proof = original_attempt(parent, phase, reserved[0], local_configuration=True)
    value = dict(
        path=str(path),
        result=digest(result),
        protocol=digest(protocol),
        journals=[digest(events)],
        previous_attempts=[
            *result["previous_attempts"],
            dict(directory=str(path), requests=1, checksum=digest(result)),
        ],
        run_id=docs.run_id,
        local_configuration_stop=proof,
    )
    if ordinary:
        value["configuration_attempt_stop"] = True
    return value


def selection(
    parent,
    attempts,
    request_ids,
    confirmed,
    *,
    local_request_ids=(),
    confirmed_invalid=False,
    historical_origin=None,
):
    """Reconstruct each specific request, not a caller-supplied authorized flag."""
    from .campaign import MultimodalCampaign

    request_ids = tuple(request_ids)
    local_request_ids = tuple(local_request_ids)
    if historical_origin and (len(request_ids) != 1 or local_request_ids or confirmed_invalid):
        raise ValueError("historical_acquisition_single_unknown_required")
    if confirmed_invalid and (len(request_ids) != 1 or local_request_ids):
        raise ValueError("confirmed_invalid_acquisition_single_request")
    if local_request_ids and (len(local_request_ids) != 1 or local_request_ids != request_ids):
        raise ValueError("configuration_attempt_single_request_required")
    if not 1 <= len(request_ids) <= 3 or len(set(request_ids)) != len(request_ids):
        raise ValueError("unknown_acquisition_bounded_request_ids")
    parent.base.check()
    eligible = set(MultimodalCampaign.available_targets(parent.base)) - set(confirmed)
    rows = {r["id"]: r for r in parent.base.ledger}
    found, seen_targets = {}, set()
    for attempt in attempts:
        if attempt.get("kind"):
            continue  # Source histories are checked by the original renewal owner.
        path = Path(attempt["directory"])
        original = json.loads((path / "result.json").read_bytes())
        protocol = json.loads((path / "protocol.json").read_bytes())
        if digest(original) != attempt["checksum"]:
            raise ValueError("unknown_acquisition_history_changed")
        if protocol.get("confirmed_invalid_acquisition"):
            seen_targets.update(protocol["confirmed_invalid_acquisition"]["selected"])
        if protocol.get("unknown_acquisition"):
            if (
                protocol.get("schema_version") == VERSION
                and original.get("complete") is False
                and len(original.get("phases", [])) == 1
                and original["phases"][0].get("requests", {}).get("api_requests_observed") == 0
                and original["phases"][0].get("requests", {}).get("api_requests_unknown") == 0
            ):
                # The original owner, not these mirror counts, proves no dispatch.
                local_configuration_tip(parent, path, protocol, original)
            else:
                seen_targets.update(
                    r["target"] for r in protocol["unknown_acquisition"]["attempts"]
                )
        for phase in original.get("phases", []):
            target = phase.get("target")
            if not target or phase.get("accepted"):
                continue
            docs = parent.documents(phase["run_id"])
            records = [
                e["record"]
                for e in RuntimeAgentJournal(docs).read()
                if e.get("phase") == "completed"
            ]
            selected = [r["request_id"] for r in records if r["request_id"] in request_ids]
            for request_id in selected:
                if (
                    request_id in found
                    or target not in eligible
                    or target not in rows
                    or rows[target].get("acquisition_method") != "vision"
                ):
                    raise ValueError("unknown_acquisition_current_eligibility")
                local = request_id in local_request_ids
                if local:
                    tip = local_configuration_tip(parent, path, protocol, original, ordinary=True)
                    if tip["local_configuration_stop"]["request_id"] != request_id:
                        raise ValueError("configuration_attempt_original_request_changed")
                found[request_id] = dict(
                    historical_client_attempt(parent, path, phase, request_id)
                    if historical_origin and path == Path(historical_origin).absolute()
                    else original_attempt(
                        parent,
                        phase,
                        request_id,
                        local_configuration=local,
                        **({"confirmed_invalid": True} if confirmed_invalid else {}),
                    ),
                    directory=str(path),
                    result_checksum=digest(original),
                )
                if historical_origin and (
                    path != Path(historical_origin).absolute()
                    or "historical_client_transition" not in found[request_id]
                ):
                    raise ValueError("historical_acquisition_exact_origin_required")
    if set(found) != set(request_ids) or len({r["target"] for r in found.values()}) != len(found):
        raise ValueError("unknown_acquisition_exact_request_required")
    if seen_targets & {r["target"] for r in found.values()}:
        raise ValueError("unknown_acquisition_single_new_attempt_per_target")
    ordered = sorted(found.values(), key=lambda r: r["target"])
    return dict(
        schema_version="confirmed-invalid-analytical-acquisition/1.0.0"
        if confirmed_invalid
        else "unknown-analytical-acquisition/1.1.0"
        if historical_origin
        else "unknown-analytical-acquisition/1.0.0",
        attempts=ordered,
        selected=[r["target"] for r in ordered],
        cumulative_history_checksum=digest(attempts),
        old_failures_preserved=True,
        side_effect_tool_replay_allowed=False,
    )


def check(parent, plan):
    post_definition = plan.get("schema_version") == POST_DEFINITION_VERSION
    if bool(plan.get("post_definition_acquisition")) != post_definition:
        raise ValueError("post_definition_acquisition_explicit_version")
    if post_definition:
        tip, proof = post_definition_tip(parent, plan["tip"]["path"], plan["facts"])
        if (
            tip != plan["tip"]
            or proof != plan["post_definition_acquisition"]
            or plan.get("max_new_requests") != 2
            or plan.get("max_model_dispatches") != 2
            or len(plan.get("acquisition_targets", [])) != 1
        ):
            raise ValueError("post_definition_acquisition_binding")
        fact_review_run(parent, plan)
    confirmed_invalid = plan.get("schema_version") == CONFIRMED_INVALID_VERSION
    proof = plan.get(
        "confirmed_invalid_acquisition" if confirmed_invalid else "unknown_acquisition"
    )
    if (
        plan.get("schema_version") not in VERSIONS
        or (
            bool(plan.get("historical_acquisition_origin"))
            != (plan.get("schema_version") == HISTORICAL_CLIENT_VERSION)
        )
        or (
            plan.get("schema_version") == HISTORICAL_CLIENT_VERSION
            and (
                plan.get("max_new_requests") != 2
                or plan.get("max_model_dispatches") != 2
                or len(plan.get("acquisition_targets", [])) != 1
            )
        )
        or (
            confirmed_invalid
            and (
                plan.get("max_new_requests") != 2
                or plan.get("max_model_dispatches") != 2
                or len(plan.get("acquisition_targets", [])) != 1
            )
        )
        or (confirmed_invalid and plan.get("unknown_acquisition"))
        or (not confirmed_invalid and plan.get("confirmed_invalid_acquisition"))
        or not proof
        or plan.get("mode") != "acquisition"
        or plan.get("seconds") != 2700
        or (
            datetime.fromisoformat(plan["deadline"]) - datetime.fromisoformat(plan["started_at"])
        ).total_seconds()
        != 2700
        or parent.config.transport.max_retries != 0
        or plan.get("review_clarification")
        or plan.get("new_public_search_allowed") is not False
        or plan.get("acquisition_targets") != proof["selected"]
        or proof
        != selection(
            parent,
            plan["previous_attempts"],
            [r["request_id"] for r in proof["attempts"]],
            parent.accepted_observations(),
            **(
                {"local_request_ids": [r["request_id"] for r in proof["attempts"]]}
                if plan.get("schema_version") == CONFIGURATION_VERSION
                else {"confirmed_invalid": True}
                if confirmed_invalid
                else {"historical_origin": plan["historical_acquisition_origin"]}
                if plan.get("schema_version") == HISTORICAL_CLIENT_VERSION
                else {}
            ),
        )
    ):
        raise ValueError("unknown_acquisition_plan_binding")
    if plan["schema_version"] in {LOCAL_VERSION, CONFIGURATION_VERSION}:
        fact_review_run(parent, plan)
        tip = plan["tip"]
        path = Path(tip["path"])
        if tip != local_configuration_tip(
            parent,
            path,
            json.loads((path / "protocol.json").read_bytes()),
            json.loads((path / "result.json").read_bytes()),
            ordinary=plan["schema_version"] == CONFIGURATION_VERSION,
        ):
            raise ValueError("unknown_local_configuration_tip_changed")
    elif plan.get("tip", {}).get("local_configuration_stop"):
        raise ValueError("unknown_local_configuration_explicit_version")
    # Historical proof reconstruction is not a new network dispatch.
    # prepare/run and the current campaign guard validate the live route.
    return proof["selected"]


def post_definition_tip(parent, path, facts):
    """Original terminal owner plus exact old claim, not a new fact authorization.

    A zero-effective-definition window advances the chain and its budget while
    retaining the explicitly bound accepted fact run. This port does not replay
    Source, revive a stopped window or adopt any changed medical definition.
    """
    from .campaign_renewal import RUNTIME_DEFINITION_VERSION, stopped_tip

    path, facts = Path(path).absolute(), Path(facts).absolute()
    protocol = json.loads((path / "protocol.json").read_bytes())
    result = json.loads((path / "result.json").read_bytes())
    original = json.loads((facts / "result.json").read_bytes())
    phases = original.get("phases", [])
    if (
        path == facts
        or (path / "cancel").exists()
        or (facts / "cancel").exists()
        or protocol.get("schema_version") != RUNTIME_DEFINITION_VERSION
        or protocol.get("mode") != "definitions"
        or protocol.get("definition_increment_only") is not True
        or Path(protocol.get("facts", "")).absolute() != facts
        or result.get("complete") is not True
        or original.get("complete") is not True
        or not phases
        or phases[-1].get("accepted") is not True
        or phases[-1].get("target") is not None
    ):
        raise ValueError("post_definition_acquisition_scope")
    tip = stopped_tip(parent, path, completed_source=True, zero_increment_acquisition=True)
    definition = tip.get("definition")
    if not definition:
        raise ValueError("post_definition_acquisition_revision_missing")
    revision = parent.documents(definition["run_id"]).read("definition:revision")
    if (
        not revision
        or len(revision) != 1
        or revision[0].get("added") != []
        or revision[0].get("replaced") != []
        or digest(revision[0]) != definition["revision"]
    ):
        raise ValueError("post_definition_acquisition_requires_zero_effective_increment")
    fact_run = phases[-1]["run_id"]
    docs = parent.documents(fact_run)
    expected_claim = dict(output=str(path), plan_checksum=digest(protocol))
    if (
        docs.read("dynamic-phase-result") != (phases[-1],)
        or docs.read("multimodal-case-result") != (original,)
        or docs.read("authorized-campaign-renewal") != (expected_claim,)
        or not any(
            a["directory"] == str(facts) and a["checksum"] == digest(original)
            for a in tip["previous_attempts"]
        )
    ):
        raise ValueError("post_definition_acquisition_prior_claim_binding")
    for run_id in {fact_run, tip["run_id"]}:
        control = parent.sessions.read(run_id)
        if (
            control is None
            or control.phase != "stopped"
            or control.pending is not None
            or control.cancelled
        ):
            raise ValueError("post_definition_acquisition_current_control")
    source_control = parent.sessions.read(definition["run_id"])
    if source_control is not None and (
        source_control.phase != "stopped"
        or source_control.pending is not None
        or source_control.cancelled
    ):
        raise ValueError("post_definition_acquisition_source_control")
    return tip, dict(
        schema_version="post-definition-acquisition-proof/1.0.0",
        accepted_fact_run_id=fact_run,
        fact_result=digest(original),
        definition_tip=digest(tip),
        prior_claim=expected_claim,
        no_effective_definition_increment=True,
        source_replay_allowed=False,
    )

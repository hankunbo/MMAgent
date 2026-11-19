"""Opt-in closure of confirmed observations, never recovery/replay of an unknown request."""

from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal

POLICY = "confirmed-only-closure/1.0.0"


def saved_closure(campaign, phases):
    """Read an immutable historical decision, not permission for another request."""
    if getattr(campaign.base, "observation_failure_policy", None) != POLICY:
        return None
    found = []
    for phase in phases:
        if phase.get("target") is None:
            continue
        docs = campaign.documents(phase["run_id"])
        saved = docs.read("observation-closure")
        if not saved:
            continue
        value = saved[0]
        expected = dict(
            schema_version=POLICY,
            campaign_id=campaign.campaign_id,
            input_checksum=campaign.base.manifest.checksum,
            protocol_checksum=digest(campaign.protocol),
            failed_phase_checksum=digest(phase),
            failed_target=phase["target"],
            stop_new_observations=True,
            next_action="review_confirmed_only",
            original_failure_preserved=True,
        )
        if (
            len(saved) != 1
            or docs.read("dynamic-phase-result") != (phase,)
            or any(value.get(k) != v for k, v in expected.items())
        ):
            raise ValueError("observation_closure_report_binding")
        found.append(value)
    if len(found) > 1:
        raise ValueError("observation_closure_report_binding")
    return found[0] if found else None


def saved_independent_failures(campaign, phases):
    """Read-only decisions for independent targets; never authorizes replay."""
    policy = "bounded-independent-observations/1.0.0"
    if campaign.protocol.get("independent_observation_policy") != policy:
        return []
    decisions = []
    for phase in phases:
        if phase.get("target") is None:
            continue
        docs = campaign.documents(phase["run_id"])
        saved = docs.read("independent-observation-failure")
        if not saved:
            continue
        value = saved[0]
        if (
            len(saved) != 1
            or docs.read("dynamic-phase-result") != (phase,)
            or value.get("schema_version") != policy
            or value.get("protocol_checksum") != digest(campaign.protocol)
            or value.get("failed_phase_checksum") != digest(phase)
            or value.get("campaign_id") != campaign.campaign_id
            or value.get("input_checksum") != campaign.base.manifest.checksum
            or value.get("failed_target") != phase["target"]
            or value.get("original_target_replay_authorized") is not False
        ):
            raise ValueError("independent_observation_report_binding")
        decisions.append(value)
    return decisions


def failure_summary(campaign, phase):
    events = RuntimeAgentJournal(campaign.documents(phase["run_id"])).read()
    completed = [e["record"] for e in events if e.get("phase") == "completed"]
    diagnostics = [
        e["value"] for e in events if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
    ]
    return dict(
        error=completed[-1]["error"] if completed else phase.get("error"),
        requests=[
            dict(request_id=r["request_id"], error=r["error"], latency_seconds=r["latency_seconds"])
            for r in completed
        ],
        diagnostics=diagnostics,
    )


def transport_claims(events):
    """Local completion is NOT remote response confirmation. Neither is erased."""
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    if (
        not reserved
        or len(set(reserved)) != len(reserved)
        or len(records) != len(reserved)
        or {r["request_id"] for r in records} != set(reserved)
        or any(r.get("role") != "vision" or r.get("error") != "transport" for r in records)
    ):
        return None
    confirmed = {
        e["request_id"]: e["confirmed"] for e in events if e.get("phase") == "response_confirmation"
    }
    return [
        dict(
            request_id=r["request_id"],
            error=r["error"],
            response_confirmed=confirmed.get(r["request_id"]),
            api_requests=r.get("api_requests"),
            replay_authorized=False,
        )
        for r in records
    ]


def close_observation(campaign, phase):
    """Consume original owners; old closure or explicit bounded independent policy."""
    if (
        getattr(campaign.base, "observation_failure_policy", None) != POLICY
        or phase.get("target") is None
        or phase.get("accepted")
        or phase.get("error") != "candidate_failed_or_context_changed"
    ):
        return False
    docs = campaign.documents(phase["run_id"])
    claims = transport_claims(RuntimeAgentJournal(docs).read())
    if claims is None:
        return False
    campaign.guard()
    current = campaign.sessions.read(phase["run_id"])
    if (
        current is None
        or current.phase != "stopped"
        or current.cancelled
        or current.pending is not None
    ):
        return False
    if docs.read("dynamic-phase-result") != (phase,):
        raise ValueError("observation_closure_phase_binding")
    if digest(current) != phase.get("planner", {}).get("control_checksum"):
        raise ValueError("observation_closure_control_changed")
    initial = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if any(
        getattr(current.graph, k) != getattr(initial.material.scope.graph, k)
        for k in ("code", "version", "checksum")
    ):
        raise ValueError("observation_closure_graph_binding")
    evidence = campaign.phase_class(
        campaign.base,
        target=phase["target"],
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    evidence.project(run_id=phase["run_id"], config=campaign.config)
    evidence.verify(initial)
    # base.check verifies registered native facts; accepted_observations reloads original receipts.
    facts = {**campaign.base.observations, **campaign.accepted_observations()}
    if not facts:
        return False
    independent = (
        campaign.protocol.get("independent_observation_policy")
        == "bounded-independent-observations/1.0.0"
    )
    if independent and campaign.protocol.get("independent_observation_failure_limit") != 2:
        raise ValueError("independent_observation_failure_limit")
    value = dict(
        schema_version=POLICY,
        campaign_id=campaign.campaign_id,
        input_checksum=campaign.base.manifest.checksum,
        protocol_checksum=digest(campaign.protocol),
        failed_phase_checksum=digest(phase),
        failed_target=phase["target"],
        requests=claims,
        confirmed_observations_checksum=digest(facts),
        confirmed_observation_ids=sorted(facts),
        stop_new_observations=True,
        next_action="review_confirmed_only",
        original_failure_preserved=True,
    )
    if independent:
        value.update(
            schema_version="bounded-independent-observations/1.0.0",
            stop_new_observations=campaign.acquisition_exhausted(),
            next_action="review_confirmed_only"
            if campaign.acquisition_exhausted()
            else "continue_distinct_unattempted_target",
            original_target_replay_authorized=False,
            max_failed_observations=2,
        )
    # This does not resume the stopped observation run. Review is the original
    # next phase of this campaign, under the same cumulative budget/deadline.
    campaign.guard()
    with docs.session():
        if campaign.sessions.read(phase["run_id"]) != current:
            raise ValueError("observation_closure_control_changed")
        docs.write(
            "independent-observation-failure" if independent else "observation-closure", (value,)
        )
    campaign.guard()
    if not independent or value["stop_new_observations"]:
        campaign.observation_closure = value
    return True

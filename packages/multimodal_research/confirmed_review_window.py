"""Same-profile bounded successor for an expired, partially confirmed Review.

Original continuation and acceptance owners consume the saved responses. This
module only selects and binds a new window; it never dispatches or assigns scores.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest, validate

VERSION = "multimodal-campaign-renewal/1.27.0"
BUDGET_VERSION = "multimodal-campaign-renewal/1.31.0"
METHOD_VERSION = "multimodal-campaign-renewal/1.36.0"
SECONDS = 5400


def origin_proof(parent, result, tip, *, now=None, budget_stop=False):
    from .campaign import Phase
    from .campaign_renewal import historical_reader

    now = now or datetime.now(UTC)
    phase = result["phases"][-1]
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(tip["run_id"])
    accepted_method_parent = parent.protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.35.0",
        "multimodal-campaign-renewal/1.40.0",
    }
    if (
        result["complete"]
        or phase.get("target") is not None
        or phase.get("accepted")
        or phase.get("acceptance_key") is not None
        or phase.get("error")
        != ("candidate_failed_or_context_changed" if budget_stop else "deadline")
        or phase["run_id"] != tip["run_id"]
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        # As with root_review_window, a verified accepted local closure may
        # explain an already-cancelled historical parent. It never revives that
        # run: a new explicit claim/Planner/grant is still mandatory. The method
        # reader below requires the original accepted partial and Control proof.
        or (control.cancelled and not accepted_method_parent)
        or now < control.binding.deadline
        or now < datetime.fromisoformat(parent.protocol["deadline"])
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("multimodal-case-result") != (result,)
        or (Path(tip["path"]) / "cancel").exists()
    ):
        raise ValueError("confirmed_review_window_origin_not_expired_partial")
    reader = historical_reader(parent, tip["run_id"], preserve_prompt=True)
    old = reader.state_reader()
    # completed_parent explains saved phases without installing every old run in
    # the local projector. Rebuild this exact context from its original owner.
    parent.base.contexts[docs.run_id] = docs.contexts.read_research_run(docs.run_id)
    projector = Phase(parent.base, observation_reader=parent.accepted_observations)
    current, _ = projector.project(run_id=tip["run_id"], config=parent.config)
    if current != old or old.config_checksum != digest(parent.config):
        raise ValueError("confirmed_review_window_current_material_changed")
    projector.verify(old)
    raw = old.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = identity(
        "RUN", "confirmed-review-preflight:" + tip["run_id"]
    )
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    # Original reader checks every saved response, request, live grant and pending
    # claim. A remote UNKNOWN or unaccounted format failure cannot pass this gate.
    origin = reader.current(validate(type(old), raw))
    if not origin.submission.batches or not origin.submission.pending_batch_ids:
        raise ValueError("confirmed_review_window_requires_confirmed_and_pending")
    proof = dict(
        origin_checksum=digest(origin),
        state_checksum=digest(old),
        control_checksum=digest(control),
        phase_checksum=digest(phase),
        confirmed_assessment_ids=sorted(i.assessment_id for i in origin.submission.candidate.items),
        pending_batch_ids=list(origin.submission.pending_batch_ids),
        old_deadline=parent.protocol["deadline"],
        configuration_checksum=digest(parent.config),
    )
    if parent.protocol.get("schema_version") == METHOD_VERSION:
        raise ValueError("confirmed_review_method_successor_already_used")
    if parent.protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.35.0",
        "multimodal-campaign-renewal/1.40.0",
    }:
        from .review_method_window import cumulative_limit

        records = reader.requests.records()
        ceiling = cumulative_limit(parent.protocol)
        method = parent.protocol["text_review_method"]
        used = sum(a["requests"] for a in tip["previous_attempts"])
        if (
            budget_stop
            or used
            != sum(a["requests"] for a in parent.protocol["previous_attempts"])
            + sum(r.api_requests for r in records)
            or any(r.error or r.retry_index or r.simulated for r in records)
            or sum(d["requests"] for d in method["diagnostic_records"])
            != method["diagnostic_requests"]
            or method["case_related_request_ceiling"] != ceiling + method["diagnostic_requests"]
        ):
            raise ValueError("confirmed_review_method_original_accounting")
        proof["method_budget_origin"] = dict(
            protocol_checksum=digest(parent.protocol),
            original_owner_ceiling=ceiling,
            previous_owner_requests=used,
            diagnostic_records=method["diagnostic_records"],
            diagnostic_requests=method["diagnostic_requests"],
            case_related_request_ceiling=method["case_related_request_ceiling"],
            request_records_checksum=digest(records),
            confirmed_requests=len(records),
        )
    if budget_stop:
        intent = docs.read("live-stage-intent:review")
        failure = docs.read("review:" + intent[0]["event"]["code"] + ":candidate-failure")
        records = reader.requests.records()
        cap = parent.protocol.get("max_model_dispatches")
        if (
            not failure
            or len(failure) != 1
            or failure[0]
            != dict(
                schema_version="research-candidate-failure/1.0.0",
                category="provider",
                code="budget",
                fields=[],
            )
            or type(cap) is not int
            or cap <= 0
            or len(records) != cap
        ):
            raise ValueError("confirmed_review_window_budget_stop_unproven")
        proof["budget_stop"] = dict(
            failure_checksum=digest(failure),
            original_dispatch_cap=cap,
            confirmed_requests=len(records),
            request_records_checksum=digest(records),
        )
    return proof


def plan_for(parent, result, tip, proof, *, output, authorization, now, request_limit=None):
    pending = len(proof["pending_batch_ids"])
    method = proof.get("method_budget_origin")
    if bool(method) != (
        parent.protocol.get("schema_version")
        in {"multimodal-campaign-renewal/1.35.0", "multimodal-campaign-renewal/1.40.0"}
    ):
        raise ValueError("confirmed_review_method_budget_origin")
    ceiling = method["original_owner_ceiling"] if method else 40
    # One optional original corrective dispatch, plus local-close headroom. The
    # number of successful/nonzero scores never determines these bounds.
    cap = pending + 2 if request_limit is None else request_limit
    if (
        not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or not pending + 1 <= cap <= min(pending + 2, 12)
        or sum(a["requests"] for a in tip["previous_attempts"]) + cap > ceiling
        or now < datetime.fromisoformat(proof["old_deadline"])
    ):
        raise ValueError("confirmed_review_window_scope_or_budget")
    from .implementation import freeze

    return dict(
        schema_version=METHOD_VERSION
        if method
        else BUDGET_VERSION
        if proof.get("budget_stop")
        else VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(parent.config),
        implementation=freeze(),
        max_new_requests=cap,
        max_model_dispatches=cap - 1,
        max_cumulative_requests=ceiling,
        seconds=SECONDS,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=SECONDS)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        expansion_policy=None,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
        reuse_confirmed_review=True,
        same_profile_review=proof,
        acquisition_targets=[],
        new_vision_requests=0,
        new_public_search_allowed=False,
        **(dict(method_budget=method) if method else {}),
    )


def method_cumulative_limit(plan):
    """Retain the original claimed method ceiling, never grant another full budget."""
    method = plan.get("method_budget", {})
    used = sum(a["requests"] for a in plan.get("previous_attempts", []))
    if (
        plan.get("schema_version") != METHOD_VERSION
        or plan.get("mode") != "review"
        or not method
        or plan.get("same_profile_review", {}).get("method_budget_origin") != method
        or plan.get("reuse_confirmed_review") is not True
        or plan.get("acquisition_targets") != []
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or method["previous_owner_requests"] != used
        or plan.get("max_cumulative_requests") != method["original_owner_ceiling"]
        or used + plan["max_new_requests"] > method["original_owner_ceiling"]
        or plan["max_model_dispatches"] != plan["max_new_requests"] - 1
        or method["case_related_request_ceiling"]
        != method["original_owner_ceiling"] + method["diagnostic_requests"]
    ):
        raise ValueError("confirmed_review_method_budget_binding")
    return method["original_owner_ceiling"]


def check(plan, parent, result):
    version = plan.get("schema_version")
    if version not in {VERSION, BUDGET_VERSION, METHOD_VERSION} or not plan.get(
        "same_profile_review"
    ):
        raise ValueError("confirmed_review_window_version")
    proof = (
        origin_proof(parent, result, plan["tip"], budget_stop=True)
        if version == BUDGET_VERSION
        else origin_proof(parent, result, plan["tip"])
    )
    expected = plan_for(
        parent,
        result,
        plan["tip"],
        proof,
        output=plan["facts"],
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        request_limit=plan["max_new_requests"],
    )
    # A historical protocol keeps its original implementation identity. Current
    # execution separately checks it via RenewedCampaign.freeze; do not require
    # current code merely to explain a completed historical window.
    expected["implementation"] = plan["implementation"]
    if plan != expected:
        raise ValueError("confirmed_review_window_binding")
    if version == METHOD_VERSION:
        method_cumulative_limit(plan)


def prepare(
    *,
    facts,
    previous,
    output,
    mode,
    authorization,
    request_limit=None,
    budget_stop_review=False,
    **options,
):
    from .campaign_renewal import completed_parent, stopped_tip

    if (
        mode != "review"
        or not authorization.strip()
        or len(authorization) > 200
        or Path(facts).absolute() != Path(previous).absolute()
        or any(options.values())
    ):
        raise ValueError("confirmed_review_window_explicit_text_only_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    parent, result = completed_parent(facts)
    tip = stopped_tip(parent, previous, completed_source=True)
    proof = (
        origin_proof(parent, result, tip, budget_stop=True)
        if budget_stop_review
        else origin_proof(parent, result, tip)
    )
    plan = plan_for(
        parent,
        result,
        tip,
        proof,
        output=out,
        authorization=authorization,
        now=datetime.now(UTC),
        request_limit=request_limit,
    )
    docs = parent.documents(tip["run_id"])
    claim = dict(output=str(out), plan_checksum=digest(plan))
    with docs.session():
        if docs.read("authorized-campaign-renewal") is not None:
            raise ValueError("renewal_tip_already_claimed")
        docs.write("authorized-campaign-renewal", (claim,))
    out.mkdir(mode=0o700)
    write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        mode="review",
        previous_requests=sum(a["requests"] for a in tip["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        reused_items=len(proof["confirmed_assessment_ids"]),
        pending_batches=len(proof["pending_batch_ids"]),
        api_requests=0,
        old_run_restarted=False,
    )

"""Bounded, different-method text reassessment of retained accepted facts.

Reuses the original renewal/Phase/Planner/acceptance owners. It is not retry,
acquisition, historical answer reuse, or a change to the old window's budget.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest

VERSION = "multimodal-campaign-renewal/1.35.0"
FORMAT_VERSION = "multimodal-campaign-renewal/1.40.0"
VERSIONS = {VERSION, FORMAT_VERSION}
PROFILE = "relay-retained-direction-v1"
SECONDS = 5400


def confirmed_format_origin(parent, tip, requests, acknowledged):
    """Original pending owner proves every failed response; never waive UNKNOWN."""
    from .pending_review import inventory

    if not acknowledged:
        raise ValueError("review_method_format_ack_required")
    value = inventory(parent, tip, format_requests=tuple(acknowledged))
    batches = value["quarantined_batches"]
    proved = []
    for batch in batches:
        proof = batch.get("known_format_failure")
        if (
            not proof
            or proof["last_request_id"] not in acknowledged
            or batch.get("new_reassessment_authorized") is not True
            or proof.get("original_answer_not_accepted") is not True
            or set(batch["request_ids"]) != {a["request_id"] for a in proof["attempts"]}
        ):
            raise ValueError("review_method_unresolved_quarantine")
        proved.extend(a["request_id"] for a in proof["attempts"])
    failed = [r.request_id for r in requests.records() if r.error is not None]
    if not failed or len(set(proved)) != len(proved) or set(proved) != set(failed):
        raise ValueError("review_method_format_history_mismatch")
    return dict(
        inventory_checksum=digest(value),
        acknowledged_format_requests=sorted(acknowledged),
        confirmed_failed_request_ids=sorted(proved),
        accepted_ids=value["accepted_ids"],
        pending_ids=value["selected_ids"],
        old_responses_reused=False,
    )


def accepted_origin(parent, documents, state, acceptance_key):
    """Read a committed stage, including original accepted-reuse ancestors."""
    from agents.expert_agent.committed_review import read_committed_review
    from packages.evidence_details.review_batches import read_submission

    proof = read_committed_review(
        documents,
        parent=state,
        config=parent.config,
        confirmed_review_reader=getattr(parent.base, "confirmed_review_reader", None),
    )
    if proof is None or proof["acceptance_key"] != acceptance_key:
        raise ValueError("review_method_committed_origin_binding")
    submission = read_submission(documents.read("submission:" + acceptance_key)[0]["submission"])
    return proof, sorted(row.assessment_id for row in submission.candidate.items)


def configuration(origin, profile, *, format_recovery=False):
    from packages.evidence_details.live_permission import validate_live_config
    from packages.llm_provider.config import prompt, read_provider_config

    if profile != PROFILE:
        raise ValueError("review_method_profile")
    old = read_provider_config(origin)
    role, _ = prompt(old, "review")
    target = read_provider_config(
        json.loads(Path("configs/agents/relayrouter_generic_evidence_v1.json").read_bytes())
    )
    if (
        old.schema_version
        not in (
            {"agent-provider-config/9.0.0", "agent-provider-config/9.1.0"}
            if format_recovery
            else {"agent-provider-config/9.0.0"}
        )
        or old.mode != "live"
        or role.prompt_version != "review-direction-2.7.0"
        or role.images
        or old.transport.max_retries != 0
        or old.transport.base_url != target.transport.base_url
        or old.transport.endpoint != target.transport.endpoint
        or old.transport.api_key_env != target.transport.api_key_env
        or old.transport.model == target.transport.model
    ):
        raise ValueError("review_method_original_configuration")
    raw = old.model_dump(mode="json")
    raw["transport"].update(model=target.transport.model, timeout_seconds=180, max_retries=0)
    raw["review_batching"]["batch_size"] = 3
    return validate_live_config(read_provider_config(raw))


def diagnostics(paths, result):
    """Account separately for exact confirmed diagnostics; never authorize them."""
    if len(paths) > 2 or len(set(paths)) != len(paths):
        raise ValueError("review_method_diagnostic_scope")
    values = []
    for name in paths:
        path = Path(name).absolute()
        raw = json.loads(path.read_bytes())
        if (
            raw.get("schema_version") != "retained-direction-diagnostic-result/1.0.0"
            or raw.get("old_result_checksum") != digest(result)
            or raw.get("response_confirmed") is not True
            or raw.get("api_requests") != 1
            or raw.get("database_writes") != 0
            or raw.get("new_acceptance") is not False
            or raw.get("images_uploaded") != 0
        ):
            raise ValueError("review_method_diagnostic_binding")
        values.append(dict(path=str(path), checksum=digest(raw), requests=raw["api_requests"]))
    return values


def plan_for(parent, result, tip, proof, *, authorization, profile, now, diagnostic_paths=()):
    from .implementation import freeze

    ancestor = parent
    while ancestor is not None:
        if ancestor.protocol.get("schema_version") in VERSIONS:
            raise ValueError("review_method_already_used_no_same_method_loop")
        ancestor = getattr(ancestor, "parent", None)
    count = len(proof["model_review_ids"])
    if (
        not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or not 1 <= count <= 15
        or proof["old_answers_reused"] is not False
        or len(set(proof["review_target_ids"])) != len(proof["review_target_ids"])
    ):
        raise ValueError("review_method_scope")
    format_recovery = proof.get("confirmed_format_origin")
    selected = configuration(parent.config, profile, format_recovery=bool(format_recovery))
    batches = (count + 2) // 3
    # One specifically bounded correction plus one local finalization headroom.
    cap = batches + 2
    history_used = sum(r["requests"] for r in tip["previous_attempts"])
    diagnostic_records = diagnostics(diagnostic_paths, result)
    diagnostic_used = sum(r["requests"] for r in diagnostic_records)
    old_cap = (
        parent.cumulative_request_limit()
        if format_recovery
        else parent.protocol["max_cumulative_requests"]
    )
    if format_recovery and history_used + diagnostic_used + cap > min(40, old_cap):
        raise ValueError("review_method_format_cumulative_limit")
    return dict(
        schema_version=FORMAT_VERSION if format_recovery else VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(selected),
        implementation=freeze(),
        max_new_requests=cap,
        max_model_dispatches=batches + 1,
        max_cumulative_requests=history_used + cap,
        seconds=SECONDS,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=SECONDS)).isoformat(),
        original_planner_phase_max_seconds=2700,
        prior_deadline=parent.protocol["deadline"],
        deadline_semantics="new_explicit_window_not_old_deadline_extension",
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        expansion_policy=None,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
        root_direction_review=proof,
        review_target_ids=proof["review_target_ids"],
        acquisition_targets=[],
        new_vision_requests=0,
        new_public_search_allowed=False,
        text_review_method=dict(
            profile=profile,
            previous_provider=digest(parent.config),
            old_window_cap=old_cap,
            previous_owner_requests=history_used,
            diagnostic_records=diagnostic_records,
            diagnostic_requests=diagnostic_used,
            case_related_request_ceiling=history_used + diagnostic_used + cap,
            planned_batches=batches,
            old_scores_reused=False,
            reason="different_text_method_uniform_reassessment_not_sign_targeting",
        ),
    )


def check(plan, parent, result):
    from .root_review_window import origin_proof

    if plan.get("schema_version") not in VERSIONS or not plan.get("text_review_method"):
        raise ValueError("review_method_version")
    method = plan["text_review_method"]
    format_proof = plan.get("root_direction_review", {}).get("confirmed_format_origin")
    if bool(format_proof) != (plan["schema_version"] == FORMAT_VERSION):
        raise ValueError("review_method_format_version_binding")
    proof = origin_proof(
        parent,
        result,
        plan["tip"],
        method_profile=method["profile"],
        **(
            dict(format_requests=format_proof["acknowledged_format_requests"])
            if format_proof
            else {}
        ),
    )
    expected = plan_for(
        parent,
        result,
        plan["tip"],
        proof,
        authorization=plan["authorization"],
        profile=method["profile"],
        now=datetime.fromisoformat(plan["started_at"]),
        diagnostic_paths=[r["path"] for r in method["diagnostic_records"]],
    )
    expected["implementation"] = plan["implementation"]
    if expected != plan:
        raise ValueError("review_method_binding")


def cumulative_limit(plan):
    """Only a constructed, owner-claimed method window has this new bound."""
    if (
        plan.get("schema_version") not in VERSIONS
        or plan.get("mode") != "review"
        or plan.get("acquisition_targets") != []
        or plan.get("new_vision_requests") != 0
        or not plan.get("root_direction_review")
        or plan.get("text_review_method", {}).get("old_scores_reused") is not False
        or not 3 <= plan["max_new_requests"] <= 7
        or plan["max_cumulative_requests"]
        != sum(r["requests"] for r in plan["previous_attempts"]) + plan["max_new_requests"]
    ):
        raise ValueError("review_method_budget_binding")
    if plan["schema_version"] == FORMAT_VERSION:
        method = plan.get("text_review_method", {})
        if (
            not plan["root_direction_review"].get("confirmed_format_origin")
            or plan["max_cumulative_requests"] + method.get("diagnostic_requests", 0)
            > min(40, method.get("old_window_cap", 0))
            or method.get("case_related_request_ceiling")
            != plan["max_cumulative_requests"] + method.get("diagnostic_requests", 0)
        ):
            raise ValueError("review_method_format_cumulative_limit")
    return plan["max_cumulative_requests"]


def prepare(
    *,
    facts,
    previous,
    output,
    mode,
    authorization,
    review_method_profile,
    method_diagnostic=(),
    acknowledge_format_request=(),
    **options,
):
    from .accepted_aggregation import checked_path
    from .campaign_renewal import completed_parent, stopped_tip
    from .root_review_window import origin_proof

    if mode != "review" or checked_path(facts) != checked_path(previous) or any(options.values()):
        raise ValueError("review_method_explicit_text_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    parent, result = completed_parent(facts)
    tip = stopped_tip(parent, previous, completed_source=True)
    proof = origin_proof(
        parent,
        result,
        tip,
        method_profile=review_method_profile,
        **(dict(format_requests=acknowledge_format_request) if acknowledge_format_request else {}),
    )
    plan = plan_for(
        parent,
        result,
        tip,
        proof,
        authorization=authorization,
        profile=review_method_profile,
        now=datetime.now(UTC),
        diagnostic_paths=method_diagnostic,
    )
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
        mode="review",
        max_new_requests=plan["max_new_requests"],
        case_related_request_ceiling=plan["text_review_method"]["case_related_request_ceiling"],
        review_items=len(plan["review_target_ids"]),
        old_scores_reused=False,
        observations_reused=True,
        api_requests=0,
        old_run_restarted=False,
    )

"""Explicit bounded root-question reassessment through the original campaign.

Only retained observations are reused. An original accepted deadline closure can
prove history, but neither that grant nor its cancelled Control authorizes work.
New work requires the existing successor claim, new Planner run and live grant.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate

from .review_shape_window import PROMPTS, SECONDS
from .single_proposition import root_direction_spec, root_direction_state

VERSION = "multimodal-campaign-renewal/1.29.0"
CITATION_VERSION = "multimodal-campaign-renewal/1.30.0"
PROMPT = "review-direction-2.6.0"
QUALIFIED_PROMPT = "review-direction-2.7.0"


def configuration(origin, *, citation_projection=False, batch_size=1, citation_reassessment=False):
    from packages.evidence_details.live_permission import validate_live_config
    from packages.llm_provider.config import read_provider_config

    old = read_provider_config(origin)
    if (
        type(batch_size) is not int
        or batch_size not in (1, 2, 3)
        or (batch_size != 1 and not citation_projection)
    ):
        raise ValueError("root_review_batch_size")
    raw = old.model_dump(mode="json")
    review = next(r for r in raw["roles"] if r["role"] == "review")
    if (
        old.schema_version != "agent-provider-config/9.0.0"
        or review["prompt_version"]
        not in (
            {PROMPT}
            if citation_reassessment
            else {"review-direction-2.4.0", "review-direction-2.5.0"}
        )
        or (citation_reassessment and not citation_projection)
        or review["prompt_checksum"]
        != digest((PROMPTS / (review["prompt_version"] + ".txt")).read_text())
        or old.review_batching.batch_size != 1
        or old.transport.max_retries != 0
    ):
        raise ValueError("root_review_original_configuration")
    selected_prompt = QUALIFIED_PROMPT if citation_projection else PROMPT
    review.update(
        prompt_version=selected_prompt,
        prompt_checksum=digest((PROMPTS / (selected_prompt + ".txt")).read_text()),
    )
    raw["review_batching"]["batch_size"] = batch_size
    return validate_live_config(read_provider_config(raw))


def configuration_for(origin, plan):
    if plan.get("schema_version") in {
        "multimodal-campaign-renewal/1.35.0",
        "multimodal-campaign-renewal/1.40.0",
    }:
        from .review_method_window import configuration as method_configuration

        return method_configuration(
            origin,
            plan["text_review_method"]["profile"],
            format_recovery=plan["schema_version"] == "multimodal-campaign-renewal/1.40.0",
        )
    if plan.get("schema_version") == CITATION_VERSION:
        return configuration(
            origin,
            citation_projection=True,
            batch_size=plan["review_batch_size"],
            citation_reassessment=True,
        )
    if plan.get("schema_version") != VERSION:
        raise ValueError("root_review_protocol_version")
    return configuration(origin)


def confirmed_request_history(state, documents, requests, *, citation_reassessment=False):
    if citation_reassessment:
        from packages.evidence_details.partial_finalization import accepted_citation_permission

        # A local citation refusal has a confirmed raw response, not an UNKNOWN.
        # Reconstruct the canonical grant and exact rejected raw bytes through
        # the original owner; merely having error="scope" grants nothing.
        accepted_citation_permission(state, documents, requests)
    elif any(r.error is not None for r in requests.records()):
        raise ValueError("root_review_unresolved_request_history")


def origin_proof(
    parent,
    result,
    tip,
    local_finalization=None,
    *,
    citation_reassessment=False,
    method_profile=None,
    format_requests=(),
):
    from packages.evidence_details.accepted_aggregation import (
        AcceptedOriginReader,
        HistoricalRequests,
    )
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .accepted_aggregation import checked_path, local_origin_phase
    from .campaign import Phase

    if format_requests and method_profile is None:
        raise ValueError("root_review_format_requires_different_method")
    phase = result["phases"][-1]
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(docs.run_id)
    if (
        phase["target"] is not None
        or phase["run_id"] != docs.run_id
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("multimodal-case-result") != (result,)
    ):
        raise ValueError("root_review_terminal_origin")
    local = None
    if local_finalization:
        local = json.loads((checked_path(local_finalization) / "result.json").read_bytes())
        expected_version = (
            "multimodal-partial-finalization-result/1.2.0"
            if citation_reassessment
            else "multimodal-partial-finalization-result/1.1.0"
        )
        if local.get("schema_version") != expected_version:
            raise ValueError("root_review_local_closure_version")
        phase = local_origin_phase(docs, result, local)
    elif citation_reassessment:
        raise ValueError("root_review_citation_closure_required")
    elif control.cancelled or (Path(tip["path"]) / "cancel").exists():
        raise ValueError("root_review_cancelled_without_accepted_closure")
    if not phase.get("accepted") or not phase.get("acceptance_key"):
        raise ValueError("root_review_requires_accepted_origin")
    if method_profile is not None:
        from .review_method_window import configuration as method_configuration

        if local_finalization is not None or citation_reassessment or not result["complete"]:
            raise ValueError("review_method_requires_complete_accepted_origin")
        method_configuration(parent.config, method_profile, format_recovery=bool(format_requests))
    else:
        configuration(
            parent.config,
            citation_projection=citation_reassessment,
            citation_reassessment=citation_reassessment,
        )
    saved = docs.read("dynamic-phase-input")
    if not saved or len(saved) != 1:
        raise ValueError("root_review_original_state")
    old = validate(DomainState, saved[0])
    parent.base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    projector = Phase(parent.base, observation_reader=parent.accepted_observations)
    current, _ = projector.project(run_id=docs.run_id, config=parent.config)
    if current != old:
        raise ValueError("root_review_current_material_changed")
    projector.verify(old)
    requests = HistoricalRequests(RuntimeAgentJournal(docs))
    format_proof = None
    if format_requests:
        from .review_method_window import confirmed_format_origin

        format_proof = confirmed_format_origin(parent, tip, requests, format_requests)
    else:
        confirmed_request_history(old, docs, requests, citation_reassessment=citation_reassessment)
    # Only a local structural comparison for the original accepted-owner reader.
    # No run, permission, new acceptance or old-score reuse is created here.
    raw = old.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:ROOT_DIRECTION_PROOF"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    probe = validate(DomainState, raw)
    if method_profile is not None:
        from .review_method_window import accepted_origin

        accepted_body, prior_ids = accepted_origin(parent, docs, old, phase["acceptance_key"])
    else:
        reader = AcceptedOriginReader(
            documents=docs,
            requests=requests,
            state_reader=lambda: old,
            acceptance_key=phase["acceptance_key"],
        )
        accepted, _, _ = reader.read(probe)
        accepted_body, prior_ids = accepted.model_dump(mode="json"), sorted(accepted.item_requests)
    projection = project_review_tasks(old)
    selected, fixed = sorted(projection.assessment_ids), sorted(projection.fixed_assessment_ids)
    if format_proof and (
        set(format_proof["accepted_ids"]) != set(prior_ids)
        or set(format_proof["accepted_ids"]) & set(format_proof["pending_ids"])
        or set(format_proof["accepted_ids"]) | set(format_proof["pending_ids"])
        != set(selected + fixed)
    ):
        raise ValueError("review_method_format_projection_binding")
    question = root_direction_spec(parent.base.task)
    projected = root_direction_state(old, parent.base.task, question)
    return dict(
        state_checksum=digest(old),
        accepted_origin=accepted_body,
        control_checksum=digest(control),
        local_finalization_checksum=digest(local) if local is not None else None,
        question=question,
        question_material_checksum=digest(projected.material),
        model_review_ids=selected,
        fixed_owner_ids=fixed,
        review_target_ids=sorted(selected + fixed),
        prior_accepted_ids=prior_ids,
        old_provider_checksum=digest(parent.config),
        uniform_reassessment_reason="explicit_root_question_requires_new_judgments_not_sign_flip",
        old_answers_reused=False,
        **(dict(confirmed_format_origin=format_proof) if format_proof else {}),
    )


def plan_for(
    parent,
    result,
    tip,
    proof,
    *,
    authorization,
    now,
    request_limit=None,
    local_finalization=None,
    review_batch_size=None,
):
    from .implementation import freeze

    required = len(proof["model_review_ids"])
    upgraded = review_batch_size is not None
    if upgraded and (type(review_batch_size) is not int or review_batch_size not in (1, 2, 3)):
        raise ValueError("root_review_batch_size")
    if upgraded and not local_finalization:
        raise ValueError("root_review_citation_closure_required")
    batches = (required + review_batch_size - 1) // review_batch_size if upgraded else required
    cap = batches + 2 if request_limit is None else request_limit
    if (
        not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or not 1 <= required <= 10
        or type(cap) is not int
        or not batches + 1 <= cap <= min(batches + 2, 12)
        or sum(r["requests"] for r in tip["previous_attempts"]) + cap > 40
    ):
        raise ValueError("root_review_scope_or_budget")
    return dict(
        schema_version=CITATION_VERSION if upgraded else VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(
            configuration(
                parent.config,
                citation_projection=upgraded,
                batch_size=review_batch_size or 1,
                citation_reassessment=upgraded,
            )
        ),
        implementation=freeze(),
        max_new_requests=cap,
        max_model_dispatches=cap - 1,
        max_cumulative_requests=40,
        seconds=SECONDS,
        original_planner_phase_max_seconds=2700,
        deadline_semantics="campaign_ceiling_does_not_extend_original_planner_phase",
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=SECONDS)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        expansion_policy=None,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
        root_direction_review=proof,
        local_finalization=local_finalization,
        review_target_ids=proof["review_target_ids"],
        acquisition_targets=[],
        new_vision_requests=0,
        new_public_search_allowed=False,
        **(
            dict(
                review_batch_size=review_batch_size,
                planned_review_batches=batches,
                reassessment_scope="retained_observations_new_citation_wire_not_old_answer_repair",
            )
            if upgraded
            else {}
        ),
    )


def check(plan, parent, result):
    if plan.get("schema_version") in {
        "multimodal-campaign-renewal/1.35.0",
        "multimodal-campaign-renewal/1.40.0",
    }:
        from .review_method_window import check as method_check

        return method_check(plan, parent, result)
    if plan.get("schema_version") not in {VERSION, CITATION_VERSION} or not plan.get(
        "root_direction_review"
    ):
        raise ValueError("root_review_protocol_version")
    upgraded = plan["schema_version"] == CITATION_VERSION
    proof = origin_proof(
        parent, result, plan["tip"], plan.get("local_finalization"), citation_reassessment=upgraded
    )
    expected = plan_for(
        parent,
        result,
        plan["tip"],
        proof,
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        request_limit=plan["max_new_requests"],
        local_finalization=plan.get("local_finalization"),
        review_batch_size=plan.get("review_batch_size") if upgraded else None,
    )
    expected["implementation"] = plan["implementation"]
    if plan != expected:
        raise ValueError("root_review_protocol_binding")


def prepare(
    *,
    facts,
    previous,
    output,
    mode,
    authorization,
    request_limit=None,
    local_finalization=None,
    review_batch_size=None,
    **options,
):
    from .accepted_aggregation import checked_path
    from .campaign_renewal import completed_parent, stopped_tip

    if (
        mode != "review"
        or checked_path(facts) != checked_path(previous)
        or any(options.values())
        or not authorization.strip()
        or len(authorization) > 200
    ):
        raise ValueError("root_review_explicit_text_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    local = str(checked_path(local_finalization)) if local_finalization else None
    parent, result = completed_parent(facts)
    tip = stopped_tip(parent, previous, completed_source=True)
    proof = origin_proof(
        parent, result, tip, local, citation_reassessment=review_batch_size is not None
    )
    plan = plan_for(
        parent,
        result,
        tip,
        proof,
        authorization=authorization,
        now=datetime.now(UTC),
        request_limit=request_limit,
        local_finalization=local,
        review_batch_size=review_batch_size,
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
        previous_requests=sum(r["requests"] for r in tip["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        review_items=len(plan["review_target_ids"]),
        old_accepted_items_preserved=len(proof["prior_accepted_ids"]),
        old_scores_reused=False,
        observations_reused=True,
        new_prompt=QUALIFIED_PROMPT if review_batch_size is not None else PROMPT,
        api_requests=0,
        old_run_restarted=False,
    )

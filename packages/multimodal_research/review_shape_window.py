"""Explicit uniform text reassessment after confirmed Review contract failures.

No new observations, answer normalization, historical score merging or Provider
change. Original Phase/Runtime/acceptance owners prove the retained facts.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest, validate

VERSION = "multimodal-campaign-renewal/1.28.0"
PROMPT = "review-direction-2.5.0"
SECONDS = 5400
PROMPTS = Path(__file__).resolve().parents[2] / "configs/prompts/agents"


def configuration(origin):
    from packages.evidence_details.live_permission import validate_live_config
    from packages.llm_provider.config import read_provider_config

    old = read_provider_config(origin)
    raw = old.model_dump(mode="json")
    review = next(r for r in raw["roles"] if r["role"] == "review")
    if (
        old.schema_version != "agent-provider-config/9.0.0"
        or review["prompt_version"] != "review-direction-2.4.0"
        or review["prompt_checksum"] != digest((PROMPTS / "review-direction-2.4.0.txt").read_text())
        or old.review_batching.batch_size != 1
        or old.transport.max_retries != 0
    ):
        raise ValueError("shape_review_original_configuration")
    review.update(
        prompt_version=PROMPT,
        prompt_checksum=digest((PROMPTS / (PROMPT + ".txt")).read_text()),
    )
    return validate_live_config(read_provider_config(raw))


def origin_proof(parent, result, tip):
    from agents.expert_agent.committed_review import read_committed_review
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.pending_review import read_pending_review
    from packages.evidence_details.review_tasks import project_review_tasks

    from .campaign import Phase

    phase = result["phases"][-1]
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(docs.run_id)
    if (
        not result["complete"]
        or not phase["accepted"]
        or phase["target"] is not None
        or phase["run_id"] != docs.run_id
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or control.cancelled
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("multimodal-case-result") != (result,)
        or (Path(tip["path"]) / "cancel").exists()
    ):
        raise ValueError("shape_review_terminal_origin")
    configuration(parent.config)
    saved = docs.read("dynamic-phase-input")
    if not saved or len(saved) != 1:
        raise ValueError("shape_review_original_state")
    old = validate(DomainState, saved[0])
    parent.base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    projector = Phase(parent.base, observation_reader=parent.accepted_observations)
    current, _ = projector.project(run_id=docs.run_id, config=parent.config)
    if current != old:
        raise ValueError("shape_review_current_material_changed")
    projector.verify(old)
    receipt = read_committed_review(
        docs,
        parent=old,
        config=parent.config,
        confirmed_review_reader=getattr(parent.base, "confirmed_review_reader", None),
    )
    if not receipt or receipt["acceptance_key"] != phase["acceptance_key"]:
        raise ValueError("shape_review_original_acceptance")
    pending = read_pending_review(
        documents=docs,
        state=old,
        acceptance_key=phase["acceptance_key"],
        inspect_format_failures=True,
    )
    failed = [b.get("known_format_failure") for b in pending["quarantined_batches"]]
    if not failed or any(p is None for p in failed):
        raise ValueError("shape_review_requires_confirmed_format_not_unknown")
    projection = project_review_tasks(old)
    selected = sorted(projection.assessment_ids)
    fixed = sorted(projection.fixed_assessment_ids)
    if not selected or set(selected) != (
        set(pending["accepted_ids"])
        | set(pending["selected_ids"])
        | {i for b in pending["quarantined_batches"] for i in b["assessment_ids"]}
    ):
        raise ValueError("shape_review_full_task_coverage")
    return dict(
        state_checksum=digest(old),
        receipt=receipt,
        control_checksum=digest(control),
        inventory_checksum=digest(pending),
        review_target_ids=sorted(selected + fixed),
        model_review_ids=selected,
        fixed_owner_ids=fixed,
        prior_accepted_ids=pending["accepted_ids"],
        confirmed_format_requests=sorted(p["last_request_id"] for p in failed),
        old_provider_checksum=digest(parent.config),
        uniform_reassessment_reason="new_format_prompt_cannot_silently_merge_old_prompt_scores",
        old_answers_reused=False,
    )


def plan_for(parent, result, tip, proof, *, authorization, now, request_limit=None):
    required = len(proof["model_review_ids"])
    cap = required + 2 if request_limit is None else request_limit
    config = configuration(parent.config)
    if (
        not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or not 1 <= required <= 10
        or not required + 1 <= cap <= min(required + 2, 12)
        or sum(r["requests"] for r in tip["previous_attempts"]) + cap > 40
    ):
        raise ValueError("shape_review_scope_or_budget")
    from .implementation import freeze

    return dict(
        schema_version=VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(config),
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
        shape_contract_review=proof,
        review_target_ids=proof["review_target_ids"],
        acquisition_targets=[],
        new_vision_requests=0,
        new_public_search_allowed=False,
    )


def check(plan, parent, result):
    if plan.get("schema_version") != VERSION or not plan.get("shape_contract_review"):
        raise ValueError("shape_review_protocol_version")
    proof = origin_proof(parent, result, plan["tip"])
    expected = plan_for(
        parent,
        result,
        plan["tip"],
        proof,
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        request_limit=plan["max_new_requests"],
    )
    expected["implementation"] = plan["implementation"]
    if plan != expected:
        raise ValueError("shape_review_protocol_binding")


def prepare(*, facts, previous, output, mode, authorization, request_limit=None, **options):
    from .campaign_renewal import completed_parent, stopped_tip

    if (
        mode != "review"
        or Path(facts).absolute() != Path(previous).absolute()
        or any(options.values())
        or not authorization.strip()
        or len(authorization) > 200
    ):
        raise ValueError("shape_review_explicit_text_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    parent, result = completed_parent(facts)
    tip = stopped_tip(parent, previous, completed_source=True)
    proof = origin_proof(parent, result, tip)
    plan = plan_for(
        parent,
        result,
        tip,
        proof,
        authorization=authorization,
        now=datetime.now(UTC),
        request_limit=request_limit,
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
        new_prompt=PROMPT,
        original_planner_phase_max_seconds=plan["original_planner_phase_max_seconds"],
        api_requests=0,
        old_run_restarted=False,
    )

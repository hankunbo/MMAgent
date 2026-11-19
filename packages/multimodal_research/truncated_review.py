"""One explicit, bounded Review successor for an owner-confirmed truncation.

The original campaign, accepted judgments and failed response stay immutable.
Only the exact failed batch is selected. Original Runtime executes the successor;
the accepted-aggregation owner, not this module, combines its later judgments.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest, validate

VERSION = "multimodal-campaign-renewal/1.53.0"


def configuration(origin):
    from packages.evidence_details.live_permission import validate_live_config
    from packages.llm_provider.config import read_provider_config

    old = read_provider_config(origin)
    if (
        old.schema_version != "agent-provider-config/9.2.0"
        or old.review_batching.batch_size != 3
        or old.transport.max_retries != 0
    ):
        raise ValueError("truncated_review_original_configuration")
    raw = old.model_dump(mode="json")
    raw["review_batching"]["batch_size"] = 1
    return validate_live_config(read_provider_config(raw))


def origin_proof(parent, result, tip, request_id):
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.pending_review import read_pending_review

    from .campaign import Phase

    phase = result["phases"][-1]
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(docs.run_id)
    if (
        not result.get("complete")
        or not phase.get("accepted")
        or phase.get("target") is not None
        or phase["run_id"] != docs.run_id
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or control.cancelled
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("multimodal-case-result") != (result,)
        or (Path(tip["path"]) / "cancel").exists()
        or parent.protocol.get("truncated_review")
        or parent.protocol.get("pending_review")
    ):
        raise ValueError("truncated_review_terminal_origin")
    configuration(parent.config)
    parent.base.check()
    old = validate(DomainState, docs.read("dynamic-phase-input")[0])
    parent.base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    projector = Phase(parent.base, observation_reader=parent.accepted_observations)
    current, _ = projector.project(run_id=docs.run_id, config=parent.config)
    if current != old:
        raise ValueError("truncated_review_current_material_changed")
    projector.verify(old)
    pending = read_pending_review(
        documents=docs,
        state=current,
        acceptance_key=phase["acceptance_key"],
        allow_empty=True,
        inspect_truncated_failures=True,
        inspect_empty_truncated_failures=True,
    )
    matches = [
        b
        for b in pending["quarantined_batches"]
        if (b.get("known_truncated_failure") or {}).get("last_request_id") == request_id
    ]
    if len(matches) != 1:
        raise ValueError("truncated_review_exact_request_required")
    batch = matches[0]
    selected = sorted(batch["assessment_ids"])
    if (
        not 1 <= len(selected) <= 3
        or len(set(selected)) != len(selected)
        or set(selected) & set(pending["accepted_ids"])
        or batch["new_reassessment_authorized"]
    ):
        raise ValueError("truncated_review_selection_binding")
    return dict(
        schema_version="selected-truncated-review/1.0.0",
        request_id=request_id,
        inventory=pending,
        review_target_ids=selected,
        accepted_ids=pending["accepted_ids"],
        original_response_not_replayed=True,
        original_accepted_scores_not_reevaluated=True,
    )


def plan_for(parent, result, tip, proof, *, output, authorization, started):
    from .execution_lease import PROFILE
    from .implementation import freeze

    count = len(proof["review_target_ids"])
    if (
        not authorization.strip()
        or len(authorization) > 200
        or started.tzinfo is None
        or not 1 <= count <= 3
        or sum(a["requests"] for a in tip["previous_attempts"]) + count + 1 > 40
    ):
        raise ValueError("truncated_review_authorization_or_budget")
    return dict(
        schema_version=VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        output=str(Path(output).absolute()),
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        provider=digest(configuration(parent.config)),
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        implementation=freeze(),
        max_new_requests=count + 1,
        max_model_dispatches=count,
        max_cumulative_requests=40,
        seconds=2700,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=2700)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        deadline_semantics="explicit_new_text_window_not_original_run_resume",
        execution_lease_profile=PROFILE,
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        truncated_review=proof,
        review_target_ids=proof["review_target_ids"],
        acquisition_targets=[],
        expansion_policy=None,
        new_vision_requests=0,
        new_public_search_allowed=False,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
    )


def check(parent, plan):
    from .campaign_renewal import stopped_tip

    if plan.get("schema_version") != VERSION or not plan.get("truncated_review"):
        raise ValueError("truncated_review_protocol_version")
    result = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    tip = stopped_tip(parent, plan["facts"])
    proof = origin_proof(parent, result, tip, plan["truncated_review"]["request_id"])
    expected = plan_for(
        parent,
        result,
        tip,
        proof,
        output=plan["output"],
        authorization=plan["authorization"],
        started=datetime.fromisoformat(plan["started_at"]),
    )
    expected["implementation"] = plan["implementation"]
    if expected != plan:
        raise ValueError("truncated_review_plan_binding")
    return configuration(parent.config)


def prepare(
    *, facts, previous, output, mode, authorization, acknowledge_truncated_request, **options
):
    from .campaign_renewal import completed_parent, stopped_tip

    if (
        not acknowledge_truncated_request
        or mode != "review"
        or any(options.values())
        or Path(facts).absolute() != Path(previous).absolute()
    ):
        raise ValueError("truncated_review_explicit_text_only_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    parent, result = completed_parent(facts)
    tip = stopped_tip(parent, previous)
    proof = origin_proof(parent, result, tip, acknowledge_truncated_request)
    plan = plan_for(
        parent,
        result,
        tip,
        proof,
        output=out,
        authorization=authorization,
        started=datetime.now(UTC),
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
        api_requests=0,
        max_new_requests=plan["max_new_requests"],
        max_model_dispatches=plan["max_model_dispatches"],
        previous_requests=sum(a["requests"] for a in tip["previous_attempts"]),
        review_target_ids=proof["review_target_ids"],
        preserved_accepted=len(proof["accepted_ids"]),
    )


def inspect(input_path, request_id):
    """Current-owner preflight only: no claim, grant, run, or dispatch."""
    from .campaign_renewal import completed_parent, stopped_tip

    parent, result = completed_parent(input_path)
    tip = stopped_tip(parent, input_path)
    proof = origin_proof(parent, result, tip, request_id)
    return dict(
        proof=proof,
        parent_result=digest(result),
        next_provider=digest(configuration(parent.config)),
        previous_requests=sum(a["requests"] for a in tip["previous_attempts"]),
        new_api_requests=0,
        new_claim=False,
        successor_not_prepared=True,
    )

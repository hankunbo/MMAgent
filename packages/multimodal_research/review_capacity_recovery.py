"""Explicit zero-wire capacity recovery over original accepted acquisition facts.

Only the batch-plan inventory ceiling changes, within its existing twenty-batch
contract. Actual new requests, inherited deadline and medical context do not.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal


def configuration(origin, required):
    from packages.llm_provider.config import read_provider_config

    if type(required) is not int or not origin.review_batching.max_batches < required <= 20:
        raise ValueError("review_capacity_configuration_bound")
    raw = origin.model_dump(mode="json")
    raw["review_batching"]["max_batches"] = required
    return read_provider_config(raw)


def origin_proof(parent, result, tip):
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_batches import make_plan, partition_plan
    from packages.evidence_details.review_tasks import project_review_tasks

    from .acquired_review import projection_terminal_control
    from .campaign import Phase
    from .campaign_renewal import (
        RETAINED_ACQUISITION_VERSION,
        accepted_reader,
        acquisition_review_ids,
    )

    old, phases = parent.protocol, result.get("phases", [])
    if (
        old.get("schema_version") != RETAINED_ACQUISITION_VERSION
        or old.get("mode") != "acquisition"
        or result.get("complete") is not False
        or len(phases) != len(old.get("acquisition_targets", [])) + 1
        or [p.get("target") for p in phases[:-1]] != old["acquisition_targets"]
        or any(not p.get("accepted") for p in phases[:-1])
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted")
        or phases[-1].get("acceptance_key") is not None
        or "batch_partition" not in (phases[-1].get("error") or "")
        or phases[-1].get("run_id") != tip["run_id"]
        or result.get("previous_attempts") != old["previous_attempts"]
        or parent.config.transport.max_retries != 0
    ):
        raise ValueError("review_capacity_original_failure")
    parent.base.check()
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(docs.run_id)
    events = RuntimeAgentJournal(docs).read()
    closure = projection_terminal_control(docs, control, phases[-1], tip["path"])
    intent = docs.read("live-stage-intent:review")
    if (
        docs.read("dynamic-phase-result") != (phases[-1],)
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-campaign") != (old,)
        or any(e.get("phase") in {"reserved", "completed"} for e in events)
        or not intent
        or docs.read("review-batches:" + intent[0]["event"]["code"] + ":A:plan") is not None
    ):
        raise ValueError("review_capacity_not_zero_wire_terminal")
    failed = validate(DomainState, docs.read("dynamic-phase-input")[0])
    parent.base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    projector = Phase(parent.base, observation_reader=parent.accepted_observations)
    current, _ = projector.project(run_id=docs.run_id, config=parent.config)
    if current != failed:
        raise ValueError("review_capacity_current_context_changed")
    projector.verify(failed)
    ids = project_review_tasks(failed).assessment_ids
    if tuple(ids) != acquisition_review_ids(old):
        raise ValueError("review_capacity_inventory_changed")
    size = parent.config.review_batching.batch_size
    reused, new = set(old["reuse_assessment_ids"]), set(ids) - set(old["reuse_assessment_ids"])
    required = max(
        (len(ids) + size - 1) // size,
        (len(reused) + size - 1) // size + (len(new) + size - 1) // size,
    )
    config = configuration(parent.config, required)
    observations = parent.accepted_observations()
    if new != set(old["acquisition_targets"]) or any(
        t not in observations or observations[t].status != "obtained" for t in new
    ):
        raise ValueError("review_capacity_observation_binding")
    reader = accepted_reader(
        parent.parent,
        old["tip"]["run_id"],
        recorded_origin=old.get("recorded_review_origin"),
        capacity_config=config,
    )
    following = failed.model_copy(update={"config_checksum": digest(config)})
    origin = reader.current(following)
    if set(origin.assessment_ids) != reused:
        raise ValueError("review_capacity_reused_inventory_changed")
    plan = make_plan(following, config.review_batching, key="capacity-preflight")
    if any(
        set(b.assessment_ids) & reused and not set(b.assessment_ids) <= reused for b in plan.batches
    ):
        plan = partition_plan(plan, reused, digest(origin))
    dispatches = sum(not set(b.assessment_ids) <= reused for b in plan.batches)
    unused = old["max_new_requests"] - result["new_requests"]
    if (
        not 1 <= dispatches <= unused
        or sum(p["requests"] for p in tip["previous_attempts"]) + dispatches > 40
    ):
        raise ValueError("review_capacity_remaining_request_budget")
    return dict(
        state_checksum=digest(failed),
        origin_checksum=digest(origin),
        control_checksum=digest(control),
        journal_checksum=digest(events),
        phase_checksum=digest(phases[-1]),
        terminal_closure_checksum=closure,
        required_batches=required,
        previous_max_batches=parent.config.review_batching.max_batches,
        review_target_ids=list(ids),
        reuse_assessment_ids=sorted(reused),
        new_assessment_ids=sorted(new),
        observation_checksums={t: digest(observations[t]) for t in sorted(new)},
        new_dispatches=dispatches,
        unused_original_allowance=unused,
        old_review_requests=0,
        original_failure_preserved=True,
    )


def plan_for(parent, result, tip, proof, *, output, authorization, started):
    from .campaign_renewal import CAPACITY_REVIEW_VERSION, RETAINED_CONTEXT_SCOPE
    from .implementation import freeze

    deadline = datetime.fromisoformat(parent.protocol["deadline"])
    seconds = int((deadline - started).total_seconds())
    if not authorization.strip() or len(authorization) > 200 or seconds <= 300:
        raise ValueError("review_capacity_authorization_or_remaining_time")
    config = configuration(parent.config, proof["required_batches"])
    return dict(
        schema_version=CAPACITY_REVIEW_VERSION,
        mode="review",
        authorization=authorization,
        facts=tip["path"],
        tip=tip,
        output=str(Path(output).absolute()),
        original_fact_result=digest(result),
        previous_attempts=tip["previous_attempts"],
        provider=digest(config),
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        implementation=freeze(),
        max_new_requests=proof["new_dispatches"],
        max_model_dispatches=proof["new_dispatches"],
        max_cumulative_requests=40,
        seconds=seconds,
        started_at=started.isoformat(),
        deadline=parent.protocol["deadline"],
        prior_deadline=parent.protocol["deadline"],
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        review_capacity=proof,
        review_context_scope=RETAINED_CONTEXT_SCOPE,
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


def check(parent, plan):
    from .campaign_renewal import stopped_tip

    result = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    tip = stopped_tip(parent, plan["facts"])
    proof = origin_proof(parent, result, tip)
    expected = plan_for(
        parent,
        result,
        tip,
        proof,
        output=plan["output"],
        authorization=plan["authorization"],
        started=datetime.fromisoformat(plan["started_at"]),
    )
    # Historical plan identity remains immutable; live execution freeze separately
    # requires the current code. This check also supports old accepted ancestry.
    expected["implementation"] = plan["implementation"]
    if expected != plan:
        raise ValueError("review_capacity_plan_binding")


def prepare(
    *, facts, previous, output, mode, authorization, review_capacity_recovery=False, **options
):
    from .campaign_renewal import completed_parent, stopped_tip

    if (
        not review_capacity_recovery
        or mode != "review"
        or any(options.values())
        or Path(facts).absolute() != Path(previous).absolute()
    ):
        raise ValueError("review_capacity_explicit_text_only_scope")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    parent, result = completed_parent(facts)
    tip = stopped_tip(parent, previous)
    proof = origin_proof(parent, result, tip)
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
        previous_requests=sum(p["requests"] for p in tip["previous_attempts"]),
        new_assessment_ids=proof["new_assessment_ids"],
        reused=len(proof["reuse_assessment_ids"]),
    )

"""One bounded successor for a proven local Review failure; original owners execute."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.pending_review import read_local_unsent_review
from packages.grounding.models import digest, validate

VERSION = "multimodal-campaign-renewal/1.74.0"
ORIGIN_VERSION = "multimodal-campaign-renewal/1.71.0"


def failure(origin):
    from .campaign import Phase

    old, path = origin.protocol, origin.output
    if old.get("schema_version") != ORIGIN_VERSION or old.get("mode") != "review":
        raise ValueError("local_review_origin_version")
    if (path / "cancel").exists():
        raise ValueError("local_review_cancelled")
    if json.loads((path / "protocol.json").read_bytes()) != old:
        raise ValueError("local_review_original_protocol_changed")
    result = json.loads((path / "result.json").read_bytes())
    phases = result.get("phases", [])
    rid = identity("RUN", "task-research:" + origin.request_key(None))
    if (
        result.get("complete") is not False
        or len(phases) != 1
        or phases[0].get("target") is not None
        or phases[0].get("run_id") != rid
        or phases[0].get("accepted") is not False
        or phases[0].get("acceptance_key") is not None
    ):
        raise ValueError("local_review_terminal_binding")
    docs = origin.documents(rid)
    if docs.read("multimodal-case-result") != (result,) or docs.read(
        "dynamic-phase-result"
    ) != tuple(phases):
        raise ValueError("local_review_persisted_result_changed")
    origin.base.check()
    origin.base.contexts[rid] = origin.contexts.read_research_run(rid)
    current, _ = Phase(origin.base, observation_reader=origin.accepted_observations).project(
        run_id=rid, config=origin.config
    )
    if current != validate(DomainState, docs.read("dynamic-phase-input")[0]):
        raise ValueError("local_review_current_material_changed")
    proof = read_local_unsent_review(documents=docs, state=current, config=origin.config)
    if (
        proof["selected_ids"] != sorted(old["review_target_ids"])
        or result.get("new_requests") != proof["charged_attempts"]
        or result.get("previous_attempts") != old["previous_attempts"]
        or digest(origin.config) != old["provider"]
    ):
        raise ValueError("local_review_failure_accounting")
    return docs, dict(proof=proof, result_checksum=digest(result), phase_checksum=digest(phases))


def plan_for(old, proof, *, previous, output, authorization, started, implementation):
    if (
        old.get("schema_version") != ORIGIN_VERSION
        or old.get("mode") != "review"
        or not old.get("aggregate_context_review")
        or old.get("acquisition_targets") != []
        or old.get("reuse_assessment_ids") != []
        or old.get("new_vision_requests") != 0
        or old.get("new_public_search_allowed") is not False
        or old.get("seconds") != 5400
        or old.get("max_cumulative_requests") != 40
        or old.get("execution_lease_profile") != "original-runtime-case-pipeline-session/1.0.0"
        or old.get("max_new_requests") != old.get("max_model_dispatches", 0) + 1
        or not authorization.strip()
        or len(authorization) > 200
        or started.tzinfo is None
        or str(Path(previous).absolute()) != old.get("output")
        or Path(previous).absolute() == Path(output).absolute()
    ):
        raise ValueError("local_review_successor_scope")
    count = proof["proof"]["charged_attempts"]
    if type(count) is not int or count < 1:
        raise ValueError("local_review_attempt_accounting")
    history = copy.deepcopy(old["previous_attempts"])
    if str(Path(previous).absolute()) in {p["directory"] for p in history}:
        raise ValueError("local_review_duplicate_history")
    history.append(
        dict(
            directory=str(Path(previous).absolute()),
            checksum=proof["result_checksum"],
            requests=count,
        )
    )
    if sum(p["requests"] for p in history) + old["max_new_requests"] > 40:
        raise ValueError("local_review_cumulative_budget")
    plan = copy.deepcopy(old)
    plan.pop("unstarted_window")
    plan.update(
        schema_version=VERSION,
        output=str(Path(output).absolute()),
        authorization=authorization,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=old["seconds"])).isoformat(),
        implementation=implementation,
        previous_attempts=history,
        local_review_recovery=dict(
            previous=str(Path(previous).absolute()),
            previous_protocol=digest(old),
            proof=proof,
            original_deadline_unchanged=True,
            provider_unchanged=True,
        ),
    )
    return plan


def key(old):
    return "authorized-local-unsent-review:" + digest(old)[7:39]


def validate_plan(origin, plan, output, proof):
    expected = plan_for(
        origin.protocol,
        proof,
        previous=origin.output,
        output=output,
        authorization=plan["authorization"],
        started=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
    )
    if expected != plan:
        raise ValueError("local_review_successor_binding")


def build(origin, output, plan):
    from .context_review import from_verified_parent

    campaign = from_verified_parent(
        origin.parent, output, plan, plan["material_context_review"]["contexts"], origin.config
    )
    campaign.local_review_origin = origin
    return campaign


def recheck(campaign):
    origin = campaign.local_review_origin
    docs, proof = failure(origin)
    validate_plan(origin, campaign.protocol, campaign.output, proof)
    if docs.read(key(origin.protocol)) != (
        dict(output=str(campaign.output), plan=campaign.protocol),
    ):
        raise ValueError("local_review_successor_claim_binding")


def prepare_and_run(*, previous, output, authorization, run=True):
    from packages.llm_provider.network_route import route_opener

    from .campaign_renewal import assemble as original_assemble
    from .execution_lease import session
    from .implementation import freeze

    previous, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (previous, *previous.parents, out, *out.parents)):
        raise ValueError("local_review_symlink")
    origin = original_assemble(previous)
    # Exact frozen route, no connection. No new key/configuration chosen here.
    route_opener(getattr(origin.config, "network_route", None), origin.config.transport.base_url)
    with session(origin):
        docs, proof = failure(origin)
        current = freeze()
        with docs.session():
            saved = docs.read(key(origin.protocol))
            if saved is not None:
                if len(saved) != 1 or saved[0].get("output") != str(out):
                    raise ValueError("local_review_already_claimed")
                plan = saved[0]["plan"]
                validate_plan(origin, plan, out, proof)
                if plan["implementation"] != current:
                    raise ValueError("local_review_implementation_changed")
            else:
                if out.exists():
                    raise ValueError("local_review_new_directory_required")
                plan = plan_for(
                    origin.protocol,
                    proof,
                    previous=previous,
                    output=out,
                    authorization=authorization,
                    started=datetime.now(UTC),
                    implementation=current,
                )
                docs.write(key(origin.protocol), (dict(output=str(out), plan=plan),))
        out.mkdir(mode=0o700, exist_ok=True)
        write_once(out / "protocol.json", plan)
        campaign = build(origin, out, plan) if run else None
    prepared = dict(
        output=str(out),
        deadline=plan["deadline"],
        api_requests=0,
        cumulative_before=sum(p["requests"] for p in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        old_failure_unchanged=True,
        status="already_recorded" if saved else "prepared",
    )
    if not run:
        return prepared
    print(json.dumps(dict(preparation=prepared)), flush=True)
    return campaign.execute()


def assemble(output, *, depth=0, ancestry=()):
    from .campaign_renewal import assemble as original_assemble
    from .provider_transition import read_ancestry

    out = Path(output).absolute()
    plan = json.loads((out / "protocol.json").read_bytes())
    if plan.get("schema_version") != VERSION:
        raise ValueError("local_review_version")
    ancestry = read_ancestry(out, depth, ancestry)
    origin = original_assemble(
        plan["local_review_recovery"]["previous"], depth=depth + 1, ancestry=ancestry
    )
    campaign = build(origin, out, plan)
    recheck(campaign)
    return campaign

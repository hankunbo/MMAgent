"""Explicit bounded successor for an expired, never-dispatched acquisition plan.

The original Runtime claim remains immutable. This is an authorization seam,
not a runner: execution, acceptance, scores and reports remain RenewedCampaign's.
"""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest

VERSION = "multimodal-campaign-renewal/1.49.0"
ACTIVATION_VERSION = "multimodal-campaign-renewal/1.50.0"
ORIGIN_VERSION = "multimodal-campaign-renewal/1.37.0"
REVIEW_ORIGIN_VERSION = "multimodal-campaign-renewal/1.69.0"
REVIEW_VERSION = "multimodal-campaign-renewal/1.71.0"
AGGREGATE_ORIGIN_VERSION = "multimodal-campaign-renewal/1.68.0"
AGGREGATE_VERSION = "multimodal-campaign-renewal/1.73.0"


def plan_for(old, *, previous, output, authorization, started, implementation, activate=False):
    if old.get("schema_version") == AGGREGATE_ORIGIN_VERSION:
        if activate:
            raise ValueError("unstarted_aggregate_recursive_activation")
        return aggregate_plan_for(
            old,
            previous=previous,
            output=output,
            authorization=authorization,
            started=started,
            implementation=implementation,
        )
    if old.get("schema_version") == REVIEW_ORIGIN_VERSION:
        if activate:
            raise ValueError("unstarted_review_recursive_activation")
        return review_plan_for(
            old,
            previous=previous,
            output=output,
            authorization=authorization,
            started=started,
            implementation=implementation,
        )
    targets = old.get("acquisition_targets", [])
    counts = [r.get("requests") for r in old.get("previous_attempts", [])]
    cap, seconds = old.get("max_new_requests"), old.get("seconds")
    if (
        old.get("schema_version") != (VERSION if activate else ORIGIN_VERSION)
        or (activate and not old.get("unstarted_window"))
        or (not activate and old.get("unstarted_window"))
        or old.get("mode") != "acquisition"
        or not 1 <= len(targets) <= 3
        or len(set(targets)) != len(targets)
        or old.get("acquisition_selection", {}).get("selected") != targets
        or not counts
        or any(type(c) is not int or c < 0 for c in counts)
        or type(cap) is not int
        or not 1 <= cap <= 12
        or old.get("max_model_dispatches") != cap
        or old.get("max_cumulative_requests") != 40
        or sum(counts) + cap > 40
        or type(seconds) is not int
        or not 300 < seconds <= 5400
        or started.tzinfo is None
        or datetime.fromisoformat(old["deadline"]) > started
        or not authorization.strip()
        or len(authorization) > 200
        or old.get("original_unknown_not_replayed") is not True
        or old.get("new_public_search_allowed") is not False
        or old.get("execution_lease_profile") != "original-runtime-case-pipeline-session/1.0.0"
        or old.get("exact_request_budget") != "dispatch_only_local_completion_zero_request/1.0"
        or any(
            old.get(k)
            for k in (
                "review_clarification",
                "recovery_http_client",
                "recovery_network_route",
                "unknown_acquisition",
                "expansion_policy",
                "acquisition_increment_only",
                "continue_acquisition",
            )
        )
    ):
        raise ValueError("unstarted_window_scope")
    plan = copy.deepcopy(old)
    plan.update(
        schema_version=ACTIVATION_VERSION if activate else VERSION,
        authorization=authorization,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=seconds)).isoformat(),
        implementation=implementation,
        unstarted_window=dict(
            schema_version="unstarted-acquisition-window/1.0.0",
            previous=str(Path(previous).absolute()),
            output=str(Path(output).absolute()),
            previous_protocol=digest(old),
            previous_deadline=old["deadline"],
            predecessor_requests=0,
            original_deadline_unchanged=True,
            provider_unchanged=True,
        ),
    )
    return plan


def validate_plan(old, plan, *, previous, output):
    expected = plan_for(
        old,
        previous=previous,
        output=output,
        authorization=plan["authorization"],
        started=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
        activate=plan.get("schema_version") == ACTIVATION_VERSION,
    )
    if plan != expected:
        raise ValueError("unstarted_window_plan_binding")


def verify_unused(campaign, old):
    from .execution_lease import unstarted

    path = campaign.output
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unstarted_window_symlink")
    allowed = {"protocol.json", "lock"}
    if old.get("schema_version") == AGGREGATE_ORIGIN_VERSION:
        allowed |= {"alternative-view", "alternative-preparation"}
    if {p.name for p in path.iterdir()} - allowed:
        raise ValueError("unstarted_window_activity")
    if json.loads((path / "protocol.json").read_bytes()) != old:
        raise ValueError("unstarted_window_original_protocol_changed")
    if old.get("schema_version") == REVIEW_ORIGIN_VERSION:
        if old.get("mode") != "review" or campaign.initial_targets:
            raise ValueError("unstarted_review_scope")
        run_id = identity("RUN", "task-research:" + campaign.request_key(None))
        if campaign.contexts.read_research_run(run_id, missing_ok=True) is not None:
            raise ValueError("unstarted_review_phase_already_registered")
    else:
        unstarted(campaign, allowed_versions={ORIGIN_VERSION, VERSION, AGGREGATE_ORIGIN_VERSION})
    for target in (None, *campaign.initial_targets):
        run_id = identity("RUN", "task-research:" + campaign.request_key(target))
        if campaign.sessions.read(run_id) is not None:
            raise ValueError("unstarted_window_control")
    if old.get("schema_version") == AGGREGATE_ORIGIN_VERSION:
        from .alternative_acquisition import AlternativeView, check
        from .execution_lease import validate_aggregate

        validate_aggregate(campaign)
        check(campaign.parent, old)
        owner = AlternativeView(campaign.parent, old, campaign.output)
        owner.read_material()
        if campaign.base.runtime.read_observation_bundle(owner.key) is not None:
            raise ValueError("unstarted_aggregate_view_already_registered")


def claim_key(old):
    if old.get("schema_version") == AGGREGATE_ORIGIN_VERSION:
        return "authorized-unstarted-aggregate-window:" + digest(old)[7:39]
    prefix = (
        "authorized-unstarted-review-window:"
        if old.get("schema_version") == REVIEW_ORIGIN_VERSION
        else "authorized-unstarted-acquisition-window:"
    )
    return prefix + digest(old)[7:39]


def origin_documents(origin):
    from .execution_lease import parent_documents

    if origin.protocol.get("schema_version") == REVIEW_ORIGIN_VERSION:
        from .context_review import claim_key as context_claim_key

        docs = origin.documents(origin.protocol["tip"]["run_id"])
        if docs.read(context_claim_key(origin.protocol)) != (
            dict(output=str(origin.output), plan_checksum=digest(origin.protocol)),
        ):
            raise ValueError("unstarted_review_original_claim_binding")
        return docs
    if origin.protocol.get("schema_version") == VERSION:
        # Verify the entire previous claim chain, not just a local protocol.
        recheck(origin)
        return origin.documents(origin.protocol["tip"]["run_id"])
    return parent_documents(origin)


def prepare(*, previous, output, authorization, activate=False):
    from .campaign_renewal import assemble as assemble_original
    from .execution_lease import session
    from .implementation import freeze

    if aggregate_origin(previous):
        if activate:
            raise ValueError("unstarted_aggregate_recursive_activation")
        return prepare_aggregate(previous=previous, output=output, authorization=authorization)
    if review_origin(previous):
        if activate:
            raise ValueError("unstarted_review_recursive_activation")
        return prepare_review(previous=previous, output=output, authorization=authorization)
    previous, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("unstarted_window_symlink")
    origin = assemble_original(previous)
    old = origin.protocol
    if activate is None:
        activate = old.get("schema_version") == VERSION
    # Same exclusion as the original CLI, including NFS environments.
    with session(origin):
        verify_unused(origin, old)
        docs = origin_documents(origin)
        with docs.session():
            saved = docs.read(claim_key(old))
            if saved is not None:
                if len(saved) != 1 or saved[0].get("output") != str(out):
                    raise ValueError("unstarted_window_already_claimed")
                plan = saved[0]["plan"]
                validate_plan(old, plan, previous=previous, output=out)
                if plan["implementation"] != freeze():
                    raise ValueError("unstarted_window_implementation_changed")
            else:
                if out.exists():
                    raise ValueError("unstarted_window_new_directory_required")
                plan = plan_for(
                    old,
                    previous=previous,
                    output=out,
                    authorization=authorization,
                    started=datetime.now(UTC),
                    implementation=freeze(),
                    activate=activate,
                )
                # Store the complete immutable plan for lost-confirmation recovery.
                docs.write(claim_key(old), (dict(output=str(out), plan=plan),))
        out.mkdir(mode=0o700, exist_ok=True)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        api_requests=0,
        previous_requests=sum(r["requests"] for r in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        predecessor_requests=0,
        provider_unchanged=True,
        old_protocol_unchanged=True,
        status="already_recorded" if saved else "prepared",
    )


def prepare_and_run(*, previous, output, authorization):
    """Prepare and immediately execute via the existing owner in one CLI process.

    The last activation may consume an unused 1.49 window once. It cannot
    activate 1.50 again, and no execution failure causes an automatic retry.
    """
    from .campaign_renewal import assemble as assemble_original

    if aggregate_origin(previous):
        return prepare_aggregate(
            previous=previous, output=output, authorization=authorization, run=True
        )
    if review_origin(previous):
        return prepare_review(
            previous=previous, output=output, authorization=authorization, run=True
        )
    prepared = prepare(previous=previous, output=output, authorization=authorization, activate=None)
    print(json.dumps(dict(preparation=prepared), ensure_ascii=False), flush=True)
    return assemble_original(output).execute()


def assemble(output, *, depth=0, ancestry=()):
    from .campaign_renewal import RenewedCampaign
    from .campaign_renewal import assemble as assemble_original
    from .provider_transition import read_ancestry

    out = Path(output).absolute()
    plan = json.loads((out / "protocol.json").read_bytes())
    if plan.get("schema_version") not in {
        VERSION,
        ACTIVATION_VERSION,
        REVIEW_VERSION,
        AGGREGATE_VERSION,
    }:
        raise ValueError("unstarted_window_version")
    ancestry = read_ancestry(out, depth, ancestry)
    previous = Path(plan["unstarted_window"]["previous"])
    origin = assemble_original(previous, depth=depth + 1, ancestry=ancestry)
    old = origin.protocol
    verify_unused(origin, old)
    validate_plan(old, plan, previous=previous, output=out)
    docs = origin_documents(origin)
    if docs.read(claim_key(old)) != (dict(output=str(out), plan=plan),):
        raise ValueError("unstarted_window_claim_binding")
    campaign = (
        build_review(origin, out, plan)
        if plan["schema_version"] == REVIEW_VERSION
        else RenewedCampaign(origin.parent, out, plan)
    )
    campaign.unstarted_window_origin = origin
    return campaign


def recheck(campaign):
    origin = campaign.unstarted_window_origin
    verify_unused(origin, origin.protocol)
    validate_plan(
        origin.protocol, campaign.protocol, previous=origin.output, output=campaign.output
    )
    if origin_documents(origin).read(claim_key(origin.protocol)) != (
        dict(output=str(campaign.output), plan=campaign.protocol),
    ):
        raise ValueError("unstarted_window_claim_binding")


def review_origin(previous):
    """Only select an owner; the local version alone never authorizes recovery."""
    path = Path(previous) / "protocol.json"
    return (
        path.is_file()
        and json.loads(path.read_bytes()).get("schema_version") == REVIEW_ORIGIN_VERSION
    )


def review_plan_for(old, *, previous, output, authorization, started, implementation):
    targets = old.get("reassessment_ids", [])
    counts = [item.get("requests") for item in old.get("previous_attempts", [])]
    cap, dispatches = old.get("max_new_requests"), old.get("max_model_dispatches")
    if (
        old.get("schema_version") != REVIEW_ORIGIN_VERSION
        or old.get("mode") != "review"
        or old.get("metadata_profile") != "grape-rnfl-device-unit/1.0.0"
        or old.get("aggregate_context_review") is not True
        or old.get("acquisition_targets") != []
        or old.get("reuse_assessment_ids") != []
        or old.get("review_target_ids") != targets
        or not 1 <= len(targets) <= 6
        or len(set(targets)) != len(targets)
        or old.get("unstarted_window")
        or old.get("new_vision_requests") != 0
        or old.get("new_public_search_allowed") is not False
        or old.get("original_unknown_not_replayed") is not True
        or old.get("execution_lease_profile") != "original-runtime-case-pipeline-session/1.0.0"
        or not counts
        or any(type(n) is not int or n < 0 for n in counts)
        or type(cap) is not int
        or type(dispatches) is not int
        or not 1 <= dispatches <= len(targets)
        or cap != dispatches + 1
        or old.get("max_cumulative_requests") != 40
        or sum(counts) + cap > 40
        or old.get("seconds") != 5400
        or started.tzinfo is None
        or datetime.fromisoformat(old["deadline"]) > started
        or not authorization.strip()
        or len(authorization) > 200
        or old.get("output") != str(Path(previous).absolute())
        or Path(previous).absolute() == Path(output).absolute()
    ):
        raise ValueError("unstarted_review_scope")
    plan = copy.deepcopy(old)
    plan.update(
        schema_version=REVIEW_VERSION,
        output=str(Path(output).absolute()),
        authorization=authorization,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=old["seconds"])).isoformat(),
        implementation=implementation,
        unstarted_window=dict(
            schema_version="unstarted-review-window/1.0.0",
            previous=str(Path(previous).absolute()),
            output=str(Path(output).absolute()),
            previous_protocol=digest(old),
            previous_deadline=old["deadline"],
            predecessor_requests=0,
            original_deadline_unchanged=True,
            provider_unchanged=True,
        ),
    )
    return plan


def build_review(origin, output, plan):
    from .context_review import from_verified_parent

    campaign = from_verified_parent(
        origin.parent, output, plan, plan["material_context_review"]["contexts"], origin.config
    )
    campaign.unstarted_window_origin = origin
    return campaign


def prepare_review(*, previous, output, authorization, run=False):
    """Restore history before starting the new clock, then use the original executor.

    There is one immutable successor claim. Lost local acknowledgement restores
    that same plan and deadline; an expired successor is never silently renewed.
    """
    from .campaign_renewal import assemble as assemble_original
    from .execution_lease import session
    from .implementation import freeze

    previous, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (previous, *previous.parents, out, *out.parents)):
        raise ValueError("unstarted_window_symlink")
    origin = assemble_original(previous)
    old = origin.protocol
    if old.get("schema_version") != REVIEW_ORIGIN_VERSION:
        raise ValueError("unstarted_review_scope")
    implementation = freeze()
    with session(origin):
        verify_unused(origin, old)
        docs = origin_documents(origin)
        with docs.session():
            saved = docs.read(claim_key(old))
            if saved is not None:
                if len(saved) != 1 or saved[0].get("output") != str(out):
                    raise ValueError("unstarted_window_already_claimed")
                plan = saved[0]["plan"]
                validate_plan(old, plan, previous=previous, output=out)
                if plan["implementation"] != implementation:
                    raise ValueError("unstarted_window_implementation_changed")
            else:
                if out.exists():
                    raise ValueError("unstarted_window_new_directory_required")
                plan = review_plan_for(
                    old,
                    previous=previous,
                    output=out,
                    authorization=authorization,
                    started=datetime.now(UTC),
                    implementation=implementation,
                )
                docs.write(claim_key(old), (dict(output=str(out), plan=plan),))
        out.mkdir(mode=0o700, exist_ok=True)
        write_once(out / "protocol.json", plan)
        campaign = build_review(origin, out, plan) if run else None
    prepared = dict(
        output=str(out),
        deadline=plan["deadline"],
        api_requests=0,
        previous_requests=sum(x["requests"] for x in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        predecessor_requests=0,
        provider_unchanged=True,
        old_protocol_unchanged=True,
        status="already_recorded" if saved else "prepared",
    )
    if not run:
        return prepared
    print(json.dumps(dict(preparation=prepared), ensure_ascii=False), flush=True)
    # execute() reacquires the same original case lease and rechecks before
    # dispatch; the verified parent is not reconstructed a second time here.
    return campaign.execute()


def aggregate_origin(previous):
    path = Path(previous) / "protocol.json"
    return (
        path.is_file()
        and json.loads(path.read_bytes()).get("schema_version") == AGGREGATE_ORIGIN_VERSION
    )


def aggregate_plan_for(old, *, previous, output, authorization, started, implementation):
    """One successor for a never-executed accepted-aggregate alternative plan."""
    from .alternative_acquisition import METHOD_VERSIONS

    targets = old.get("acquisition_targets", [])
    counts = [row.get("requests") for row in old.get("previous_attempts", [])]
    if (
        old.get("schema_version") != AGGREGATE_ORIGIN_VERSION
        or old.get("mode") != "acquisition"
        or old.get("acquisition_increment_only") is not True
        or not isinstance(targets, list)
        or len(targets) != 1
        or not isinstance(targets[0], str)
        or not targets[0]
        or old.get("acquisition_selection", {}).get("selected") != targets
        or old.get("review_target_ids") != targets
        or old.get("alternative_view", {}).get("target") != targets[0]
        or old.get("alternative_view", {}).get("method") not in METHOD_VERSIONS
        or not old.get("alternative_view_metadata_checksum")
        or not old.get("accepted_state_checksum")
        or not old.get("accepted_origin_checksum")
        or not counts
        or any(type(n) is not int or n < 0 for n in counts)
        or old.get("max_new_requests") != 3
        or old.get("max_cumulative_requests") != 40
        or sum(counts) + 3 > 40
        or old.get("seconds") != 5400
        or started.tzinfo is None
        or datetime.fromisoformat(old["deadline"]) > started
        or old.get("new_public_search_allowed") is not False
        or old.get("original_unknown_not_replayed") is not True
        or old.get("execution_lease_profile") != "original-runtime-case-pipeline-session/1.0.0"
        or any(
            old.get(k)
            for k in (
                "unstarted_window",
                "unknown_acquisition",
                "expansion_policy",
                "review_clarification",
                "recovery_http_client",
                "recovery_network_route",
                "exact_request_budget",
            )
        )
        or not authorization.strip()
        or len(authorization) > 200
        or Path(previous).absolute() == Path(output).absolute()
    ):
        raise ValueError("unstarted_aggregate_scope")
    plan = copy.deepcopy(old)
    plan.update(
        schema_version=AGGREGATE_VERSION,
        authorization=authorization,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=5400)).isoformat(),
        implementation=implementation,
        unstarted_window=dict(
            schema_version="unstarted-aggregate-alternative-window/1.0.0",
            previous=str(Path(previous).absolute()),
            output=str(Path(output).absolute()),
            previous_protocol=digest(old),
            previous_deadline=old["deadline"],
            predecessor_requests=0,
            original_deadline_unchanged=True,
            provider_unchanged=True,
            original_view_metadata_checksum=old["alternative_view_metadata_checksum"],
        ),
    )
    return plan


def copy_aggregate_material(origin, output):
    """Reuse exact prepared pixels, never re-run the worker or overwrite history."""
    from .alternative_acquisition import AlternativeView

    owner = AlternativeView(origin.parent, origin.protocol, origin.output)
    raw, _ = owner.read_material()
    destination = Path(output) / "alternative-view"
    if destination.is_symlink():
        raise ValueError("unstarted_aggregate_material_symlink")
    destination.mkdir(exist_ok=True)
    for name, body in (("view.png", raw), ("view.json", (owner.output / "view.json").read_bytes())):
        target = destination / name
        if target.is_symlink():
            raise ValueError("unstarted_aggregate_material_symlink")
        if target.exists():
            if target.read_bytes() != body:
                raise ValueError("unstarted_aggregate_material_changed")
        else:
            with target.open("xb") as stream:
                stream.write(body)


def prepare_aggregate(*, previous, output, authorization, run=False):
    from .campaign_renewal import RenewedCampaign
    from .campaign_renewal import assemble as assemble_original
    from .execution_lease import session
    from .implementation import freeze

    previous, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (previous, *previous.parents, out, *out.parents)):
        raise ValueError("unstarted_window_symlink")
    origin = assemble_original(previous)
    old = origin.protocol
    if old.get("schema_version") != AGGREGATE_ORIGIN_VERSION:
        raise ValueError("unstarted_aggregate_scope")
    implementation = freeze()
    with session(origin):
        verify_unused(origin, old)
        docs = origin_documents(origin)
        with docs.session():
            saved = docs.read(claim_key(old))
            if saved is not None:
                if len(saved) != 1 or saved[0].get("output") != str(out):
                    raise ValueError("unstarted_window_already_claimed")
                plan = saved[0]["plan"]
                validate_plan(old, plan, previous=previous, output=out)
                if plan["implementation"] != implementation:
                    raise ValueError("unstarted_window_implementation_changed")
            else:
                if out.exists():
                    raise ValueError("unstarted_window_new_directory_required")
                plan = aggregate_plan_for(
                    old,
                    previous=previous,
                    output=out,
                    authorization=authorization,
                    started=datetime.now(UTC),
                    implementation=implementation,
                )
                docs.write(claim_key(old), (dict(output=str(out), plan=plan),))
        out.mkdir(mode=0o700, exist_ok=True)
        copy_aggregate_material(origin, out)
        write_once(out / "protocol.json", plan)
        campaign = RenewedCampaign(origin.parent, out, plan) if run else None
        if campaign is not None:
            campaign.unstarted_window_origin = origin
    prepared = dict(
        output=str(out),
        deadline=plan["deadline"],
        api_requests=0,
        previous_requests=sum(row["requests"] for row in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        predecessor_requests=0,
        provider_unchanged=True,
        old_protocol_unchanged=True,
        alternate_pixels_reused=True,
        status="already_recorded" if saved else "prepared",
    )
    if not run:
        return prepared
    print(json.dumps(dict(preparation=prepared), ensure_ascii=False), flush=True)
    return campaign.execute()

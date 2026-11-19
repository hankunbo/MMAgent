"""Versioned automatic-demand successor through the original campaign owners.

This module prepares and qualifies a plan. Execution, budget reservations,
acceptance, scoring and reports remain in RenewedCampaign and its original owners.
"""

import json
from contextlib import nullcontext
from copy import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import canonical, digest, validate

VERSION = "multimodal-campaign-renewal/1.81.0"
RECOVERY_VERSION = "multimodal-campaign-renewal/1.82.0"
SCOPE_SUCCESSOR_VERSION = "multimodal-campaign-renewal/1.83.0"
CURRENT_VERSION = "multimodal-campaign-renewal/1.84.0"
CURRENT_PROFILE = "configs/research/multimodal_catalog_acquisition_v8.json"
NEW_CURRENT_PROFILE = "configs/research/multimodal_catalog_acquisition_v9.json"
ROUTED_VERSION = "multimodal-campaign-renewal/1.85.0"
SCOPE_UNKNOWN_VERSION = "multimodal-campaign-renewal/1.86.0"
SCOPE_CONFIRMED_VERSION = "multimodal-campaign-renewal/1.88.0"
SCOPE_ATTEMPT_VERSIONS = {SCOPE_UNKNOWN_VERSION, SCOPE_CONFIRMED_VERSION}
CORRECTED_SCOPE_PROFILE = "configs/research/multimodal_catalog_acquisition_v10.json"
CURRENT_SCOPE_RECOVERY_VERSION = "multimodal-campaign-renewal/1.87.0"
SCOPE_RECOVERY_VERSIONS = {SCOPE_SUCCESSOR_VERSION, CURRENT_SCOPE_RECOVERY_VERSION}
CURRENT_VERSIONS = {
    CURRENT_VERSION,
    ROUTED_VERSION,
    *SCOPE_ATTEMPT_VERSIONS,
    CURRENT_SCOPE_RECOVERY_VERSION,
}
VERSIONS = {VERSION, RECOVERY_VERSION, SCOPE_SUCCESSOR_VERSION, *CURRENT_VERSIONS}


def frozen_current_profile(plan):
    """Resolve the explicit public profile of a current plan without upgrading it."""
    from .active_requirements import load

    profile = plan.get("capability_transition", {}).get("profile", {})
    path = profile.get("path")
    if (
        path not in (CURRENT_PROFILE, NEW_CURRENT_PROFILE, CORRECTED_SCOPE_PROFILE)
        or set(profile) != {"path", "checksum"}
        or profile["checksum"] != digest(load(path))
    ):
        raise ValueError("capability_current_profile_binding")
    return path


def preparation_profile(parent, output):
    """New windows use the current contract; claim recovery keeps the original one."""
    from .campaign_renewal import stopped_tip

    tip = stopped_tip(
        parent, parent.output, completed_source=parent.protocol.get("mode") == "definitions"
    )
    saved = parent.documents(tip["run_id"]).read("authorized-campaign-renewal")
    if not saved:
        return NEW_CURRENT_PROFILE
    if (
        len(saved) != 1
        or saved[0].get("output") != str(Path(output).absolute())
        or saved[0].get("plan", {}).get("schema_version") not in (CURRENT_VERSION, ROUTED_VERSION)
    ):
        raise ValueError("renewal_tip_already_claimed")
    return frozen_current_profile(saved[0]["plan"])


def route_binding(parent, path):
    from packages.llm_provider.config import with_replacement_network_route
    from packages.llm_provider.network_route import load_route

    route = load_route(path)
    config = with_replacement_network_route(parent.config, route)
    return dict(
        schema_version="route-only-provider-transition/1.1.0",
        path=str(path),
        origin_checksum=digest(parent.config),
        route_checksum=digest(route),
        provider_checksum=digest(config),
    )


def route_configuration(parent, binding):
    if binding is None:
        return parent.config
    from packages.llm_provider.config import with_replacement_network_route
    from packages.llm_provider.network_route import load_route

    if not isinstance(binding, dict) or set(binding) != {
        "schema_version",
        "path",
        "origin_checksum",
        "route_checksum",
        "provider_checksum",
    }:
        raise ValueError("capability_route_binding_fields")
    route = load_route(binding["path"])
    config = with_replacement_network_route(parent.config, route)
    if binding != route_binding(parent, binding["path"]):
        raise ValueError("capability_route_original_binding_changed")
    return config


def routed_reader(reader, origin, selected):
    """Retain the original reader/ancestry; only its explicit transport proof changes."""
    from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
    from packages.llm_provider.config import with_replacement_network_route

    if (
        not isinstance(reader, AcceptedReviewReader)
        or reader.state_reader().config_checksum != digest(origin)
        or with_replacement_network_route(origin, selected.network_route) != selected
        or reader.capacity_configs is not None
        or reader.client_configs is not None
    ):
        raise ValueError("capability_route_accepted_owner_binding")
    result = copy(reader)
    result.network_configs = (origin, selected)
    return result


def review_origin(parent, *, include_completed=False):
    """Include original recorded finalizations, never just an inherited reader."""
    from .campaign_renewal import MAX_HISTORY_DEPTH, accepted_reader, recorded_definition_check

    current, seen, selected = parent, set(), None
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if current is None:
            break
        if id(current) in seen:
            raise ValueError("capability_review_ancestry_cycle")
        seen.add(id(current))
        if include_completed and selected is None:
            # A successor's inherited reader may intentionally predate its own
            # completed Review. Read only that campaign's deterministic run,
            # never scan arbitrary accepted objects from the shared Runtime.
            run_id = identity(
                "RUN", "task-research:mm-live:" + digest((current.output.name, "review"))[7:55]
            )
            result = current.documents(run_id).read("dynamic-phase-result")
            if result and result[0].get("accepted") and result[0].get("target") is None:
                candidate = qualified_review(parent, current, accepted_reader(current, run_id))
                selected = extend_review(selected, candidate)
        recorded = current.protocol.get("recorded_definition_origin")
        if recorded:
            recorded_definition_check(current.parent, current.protocol)
            proof = recorded["proof"]
            reader = accepted_reader(
                current.parent, proof["confirmation"]["run_id"], recorded_origin=proof
            )
            candidate = qualified_review(parent, current, reader)
            selected = extend_review(selected, candidate)
        elif selected is None:
            reader = getattr(current.base, "confirmed_review_reader", None)
            if reader is not None:
                selected = qualified_review(parent, current, reader)
        current = getattr(current, "parent", None)
    else:
        if current is not None:
            raise ValueError("capability_review_ancestry_depth")
    if selected is None:
        raise ValueError("capability_original_accepted_review_owner_required")
    return selected[:2]


def extend_review(selected, candidate):
    """A different owner may extend, but cannot silently replace accepted answers."""
    if selected is None:
        return candidate
    old, new = selected[2], candidate[2]
    if any(old[k] != new[k] for k in old.keys() & new.keys()):
        raise ValueError("capability_accepted_review_conflict")
    if old.keys() <= new.keys():
        return candidate if len(new) > len(old) else selected
    if new.keys() <= old.keys():
        return selected
    raise ValueError("capability_accepted_review_explicit_merge_required")


def qualified_review(parent, current, reader):
    from packages.evidence_details.acceptance_models import DomainState

    state = reader.state_reader()
    if state.config_checksum != digest(parent.config):
        raise ValueError("capability_review_provider_transition_required")
    # This read is not authorization for a future Phase: its entire new state
    # must still qualify via the original accepted reader at actual dispatch.
    raw = state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = identity(
        "RUN", "capability-review-preflight:" + digest(parent.protocol)
    )
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    origin = reader.current(validate(DomainState, raw))
    rows = {
        item.assessment_id: item.model_dump(mode="json")
        for item in origin.submission.candidate.items
        if item.assessment_id in origin.assessment_ids
    }
    return (
        reader,
        dict(
            ancestor=str(current.output.absolute()),
            protocol_checksum=digest(current.protocol),
            run_id=origin.run_id,
            acceptance_key=origin.acceptance_key,
            acceptance_checksum=origin.acceptance_checksum,
            state_checksum=origin.state_checksum,
            assessment_ids=list(origin.assessment_ids),
        ),
        rows,
    )


def typed_rows(rows):
    from packages.llm_provider.models import TaskDefinition
    from packages.planner_control.models import Reference

    return [
        dict(
            r,
            definition=validate(TaskDefinition, r["definition"]),
            input_modalities=tuple(r["input_modalities"]),
            **(
                {"dependency_paths": tuple(tuple(v) for v in r["dependency_paths"])}
                if "dependency_paths" in r
                else {}
            ),
            **(
                {"quality_source": validate(Reference, r["quality_source"])}
                if r.get("quality_source")
                else {}
            ),
        )
        for r in rows
    ]


def projected_base(parent, transition, profile, output):
    from .active_acquisition import ActiveOwner

    base = copy(parent.base)
    base.output = Path(output).absolute()
    base.contexts = dict(parent.base.contexts)
    base.observations = dict(parent.base.observations)
    base.ledger = typed_rows(transition["requirements"])
    discovery = dict(
        schema_version="active-knowledge-discovery/2.0.0",
        queries=[],
        automatic_medical_acceptance=False,
        patient_data_sent=False,
        status="retained_current_definition_revision_no_new_search",
    )
    base.active_owner = ActiveOwner(
        base,
        profile,
        transition["catalog_plan"],
        discovery,
        getattr(parent.base.active_owner, "native", None),
    )
    # Keep the original accepted definition reader and immutable revision.
    # The separate transition proves how those definitions enter this new plan.
    base.capability_transition = transition
    base.capability_parent = parent.base
    base.review_clarification = None
    base.scope_native_completed = None
    base.scope_selection_owner = None
    base.known_only = False
    return base


def inventory(
    parent, output, *, frozen_methods=None, profile_path=None, current=False, route_transition=None
):
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    # This function is read-only, including on cold assembly. Reuse proofs only
    # within this synchronous projection; both owners freshly recheck on return.
    with historical_fact_read_scope(), _projection_visible_reads():
        return _inventory(
            parent,
            output,
            frozen_methods=frozen_methods,
            profile_path=profile_path,
            current=current,
            route_transition=route_transition,
        )


def _inventory(
    parent, output, *, frozen_methods=None, profile_path=None, current=False, route_transition=None
):
    from .active_acquisition import successor_projection
    from .active_requirements import load
    from .campaign import MultimodalCampaign, Phase
    from .campaign_renewal import attempted_acquisition_targets, stopped_tip
    from .input_scope import configuration, project_plan
    from .reuse import CAPABILITY_PROFILE, ProcedureReuse, repository

    if route_transition is not None and not current:
        raise ValueError("capability_route_current_version_required")
    selected_config = route_configuration(parent, route_transition)
    tip = stopped_tip(
        parent, parent.output, completed_source=parent.protocol.get("mode") == "definitions"
    )
    profile_path = profile_path or (CURRENT_PROFILE if current else CAPABILITY_PROFILE)
    profile = load(profile_path)
    transition = successor_projection(
        parent.base, profile, **({"explicit_measurement_transition": True} if current else {})
    )
    if not current and any(c["action"] == "requires_reassessment" for c in transition["changes"]):
        raise ValueError("capability_changed_interpretation_requires_explicit_review_plan")
    facts = parent.accepted_observations()
    if any(
        digest(facts[k]) != digest(parent.base.observations[k])
        for k in facts.keys() & parent.base.observations.keys()
    ):
        raise ValueError("capability_observation_owner_conflict")
    reader, reviewed = review_origin(parent, **({"include_completed": True} if current else {}))
    if route_transition is not None:
        reader = routed_reader(reader, parent.config, selected_config)
    base = projected_base(parent, transition, profile, output)
    attempted = attempted_acquisition_targets(parent, tip["previous_attempts"])
    acquired = {**base.observations, **facts}
    targets = [
        t
        for t in MultimodalCampaign.available_targets(base)
        if t not in acquired and t not in attempted
    ]
    review_targets = sorted(set(acquired) | set(targets))
    if current:
        # Retain facts, but never treat a retired/merged question as a new vote.
        selected = {r["id"] for r in base.ledger}
        review_targets = sorted(set(review_targets) & selected)
        base.confirmed_review_reader = reader
        base.review_alias_policy = "retain-current-owner-source-aliases/1.0.0"
        base.review_context_scope = "current-owner-prior-visible/1.0.0"
        base.review_target_ids = tuple(review_targets)
        run_id = reviewed["run_id"]
        base.contexts[run_id] = parent.contexts.read_research_run(run_id)
        # The successor directory is deliberately absent during a read-only
        # preflight. Read the original verified views; installation later copies
        # those exact bytes into the new output with checksum checks.
        preview_base = copy(base)
        preview_base.output = parent.base.output
        phase = Phase(preview_base, observation_reader=lambda: facts)
        origin, preview = phase.review_reuse_preview(
            run_id=run_id,
            config=selected_config,
            preview_run_id=identity(
                "RUN", "capability-current-preview:" + digest((str(output), transition))
            ),
        )
        reviewed = dict(
            reviewed,
            original_assessment_ids=reviewed["assessment_ids"],
            assessment_ids=list(origin.assessment_ids),
            compatibility_preview=preview,
        )
    scope = project_plan(base, profile["input_scope_policy"])
    scope_reuse = None
    if current:
        from .scope_successor import current_proof

        scope_reuse = current_proof(parent, scope)
    method_campaign = SimpleNamespace(base=base, target=parent.target, contexts=parent.contexts)
    methods = ProcedureReuse(method_campaign, repository(method_campaign))
    if frozen_methods is None:
        methods.prepare()
    else:
        methods.restore(frozen_methods)
    result = dict(
        transition=transition,
        profile=dict(path=profile_path, checksum=digest(profile)),
        native_checksum=digest(base.active_owner.native),
        bound_observations_checksum=digest(base.observations),
        confirmed_observations_checksum=digest(facts),
        confirmed_review=reviewed,
        review_target_ids=review_targets,
        acquisition_targets=targets,
        excluded_attempted_targets=sorted(attempted - acquired.keys()),
        scope_plan=scope,
        input_scope=dict(
            plan_checksum=digest(scope),
            provider_checksum=digest(configuration(selected_config, scope)),
            targets=[e["target"] for e in scope["entries"]],
            budget_owner="original_campaign_and_RuntimeAgentJournal",
            disease_votes=0,
        ),
        procedure_reuse=methods.plan,
    )
    if current:
        result["current_scope_reuse"] = scope_reuse
        reused = {r["target"] for r in scope_reuse["reused"]} if scope_reuse else set()
        result["input_scope"]["dispatch_targets"] = [
            t for t in result["input_scope"]["targets"] if t not in reused
        ]
    # Original Tool references are typed values. The durable claim uses JSON;
    # compare that exact representation on cold assembly, not Python object types.
    return tip, json.loads(canonical(result)), reader


def plan_for(
    parent,
    tip,
    inventory_,
    *,
    output,
    authorization,
    now,
    implementation,
    seconds=5400,
    current=False,
    route_transition=None,
):
    from .execution_lease import PROFILE

    if route_transition is not None and not current:
        raise ValueError("capability_route_current_version_required")
    selected_config = route_configuration(parent, route_transition)
    if any(
        type(r.get("requests")) is not int or r["requests"] < 0 for r in tip["previous_attempts"]
    ):
        raise ValueError("capability_window_bounds_or_review_reserve")
    used = sum(r["requests"] for r in tip["previous_attempts"])
    cap = min(12, parent.cumulative_request_limit() - used)
    pending = set(inventory_["review_target_ids"]) - set(
        inventory_["confirmed_review"]["assessment_ids"]
    )
    batch_size = parent.config.review_batching.batch_size
    batches = (len(pending) + batch_size - 1) // batch_size
    minimum = (
        len(inventory_["input_scope"].get("dispatch_targets", inventory_["input_scope"]["targets"]))
        + len(inventory_["acquisition_targets"])
        + batches
        + 1
    )
    if (
        not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or not 600 <= seconds <= 5400
        or cap < minimum
        or cap < 2
        or any(
            type(a["requests"]) is not int or a["requests"] < 0 for a in tip["previous_attempts"]
        )
    ):
        raise ValueError("capability_window_bounds_or_review_reserve")
    return dict(
        schema_version=ROUTED_VERSION
        if route_transition is not None
        else CURRENT_VERSION
        if current
        else VERSION,
        **({"route_transition": route_transition} if route_transition is not None else {}),
        **(
            dict(
                visible=parent.base.visible_value.checksum,
                campaign_id=Path(output).name,
                review_alias_policy="retain-current-owner-source-aliases/1.0.0",
                current_scope_reuse=inventory_.get("current_scope_reuse"),
            )
            if current
            else {}
        ),
        mode="acquisition",
        authorization=authorization,
        facts=str(parent.output.absolute()),
        tip=tip,
        original_fact_result=tip["result"],
        previous_attempts=tip["previous_attempts"],
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(selected_config),
        implementation=implementation,
        max_new_requests=cap,
        max_model_dispatches=cap - 1,
        max_cumulative_requests=parent.cumulative_request_limit(),
        seconds=seconds,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=seconds)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        execution_lease_profile=PROFILE,
        confirmed_observations_checksum=inventory_["confirmed_observations_checksum"],
        acquisition_targets=inventory_["acquisition_targets"],
        reuse_assessment_ids=inventory_["confirmed_review"]["assessment_ids"],
        review_target_ids=inventory_["review_target_ids"],
        input_scope=inventory_["input_scope"],
        procedure_reuse=inventory_["procedure_reuse"],
        capability_transition=inventory_,
        output=str(Path(output).absolute()),
        logical_campaign_id=getattr(parent, "logical_campaign_id", parent.campaign_id),
        expansion_policy=None,
        new_public_search_allowed=False,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
    )


def claim(plan, output):
    return dict(output=str(Path(output).absolute()), plan_checksum=digest(plan), plan=plan)


def check(parent, plan):
    """Requalify against the original owners; stored preview is not authority."""
    from .execution_lease import PROFILE

    if plan.get("schema_version") not in VERSIONS or plan.get("execution_lease_profile") != PROFILE:
        raise ValueError("capability_window_version")
    if plan["schema_version"] == CURRENT_SCOPE_RECOVERY_VERSION:
        from .scope_successor import check as check_successor

        return check_successor(parent, plan)
    if plan["schema_version"] in SCOPE_ATTEMPT_VERSIONS:
        from .scope_unknown_successor import check as check_unknown

        return check_unknown(parent, plan)
    if ("route_transition" in plan) != (plan["schema_version"] == ROUTED_VERSION):
        raise ValueError("capability_route_explicit_version_required")
    if plan["schema_version"] == SCOPE_SUCCESSOR_VERSION:
        from .scope_successor import check as check_successor

        return check_successor(parent, plan)
    tip, found, reader = inventory(
        parent,
        plan["output"],
        frozen_methods=plan["procedure_reuse"],
        profile_path=frozen_current_profile(plan)
        if plan["schema_version"] in CURRENT_VERSIONS
        else None,
        current=plan["schema_version"] in CURRENT_VERSIONS,
        route_transition=plan.get("route_transition"),
    )
    if plan["schema_version"] == RECOVERY_VERSION:
        from .capability_recovery import check as check_recovery

        check_recovery(parent, plan, found=found, tip=tip)
        return reader
    expected = plan_for(
        parent,
        tip,
        found,
        output=plan["output"],
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
        seconds=plan["seconds"],
        current=plan["schema_version"] in CURRENT_VERSIONS,
        route_transition=plan.get("route_transition"),
    )
    if expected != plan:
        raise ValueError("capability_window_original_binding_changed")
    return reader


def install_base(parent, output, plan):
    """Materialize only exact local copies after the Runtime claim is verified."""
    from .active_requirements import load

    predecessor = None
    if plan.get("schema_version") in SCOPE_RECOVERY_VERSIONS:
        from .scope_successor import check as check_successor

        reader, predecessor = check_successor(parent, plan, with_predecessor=True)
    else:
        reader = check(parent, plan)
    out = Path(output).absolute()
    if str(out) != plan["output"]:
        raise ValueError("capability_window_output_binding")
    profile = load(plan["capability_transition"]["profile"]["path"])
    inv = plan["capability_transition"]
    base = projected_base(parent, inv["transition"], profile, out)
    if predecessor is not None:
        base.scope_reuse_predecessor = predecessor
    base.confirmed_review_reader = reader
    if plan.get("schema_version") in CURRENT_VERSIONS:
        base.review_alias_policy = plan["review_alias_policy"]
    base.review_target_ids = tuple(plan["review_target_ids"])
    base.review_context_scope = "current-owner-prior-visible/1.0.0"
    out.mkdir(parents=True, exist_ok=True)
    views = out / "views"
    views.mkdir(exist_ok=True)
    for entry in parent.base.views["views"]:
        if entry["status"] != "obtained":
            continue
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("capability_view_path")
        raw = (parent.base.output / "views" / name).read_bytes()
        from packages.object_store import checksum

        if checksum(raw) != entry["checksum"]:
            raise ValueError("capability_original_view_changed")
        destination = views / name
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError("capability_copied_view_changed")
        else:
            with destination.open("xb") as f:
                f.write(raw)
    raw = parent.base.views_path.read_bytes()
    path = views / "views.json"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("capability_view_manifest_changed")
    else:
        with path.open("xb") as f:
            f.write(raw)
    base.views_path = path
    write_once(out / "input-capabilities.json", inv["transition"]["capabilities"])
    write_once(out / "active-plan.json", base.active_owner.plan)
    write_once(out / "active-discovery.json", base.active_owner.discovery)
    write_once(out / "capability-transition.json", inv["transition"])
    base.initial_active_acquisition = json.loads(canonical(base.active_owner.report()))
    base.active_owner.check()
    return base


def install_methods(campaign):
    from .reuse import ProcedureReuse, repository

    campaign.procedure_reuse = ProcedureReuse(campaign, repository(campaign))
    campaign.procedure_reuse.restore(campaign.protocol["procedure_reuse"])


def prepare(*, previous, output, authorization, check_only=False):
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .campaign_renewal import completed_parent
    from .implementation import freeze
    from .unknown_acquisition import check_network

    out = Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)) or out.is_relative_to(
        Path(previous).absolute()
    ):
        raise ValueError("capability_output_must_not_change_history")
    with historical_fact_read_scope(), _projection_visible_reads():
        parent, _ = completed_parent(previous)
        from .execution_lease import session

        with nullcontext() if check_only else session(parent):
            check_network(parent.config)
            tip, found, _ = inventory(parent, out)
            plan = plan_for(
                parent,
                tip,
                found,
                output=out,
                authorization=authorization,
                now=datetime.now(UTC),
                implementation=freeze(),
            )
            docs = parent.documents(tip["run_id"])
            if not check_only:
                with docs.session():
                    saved = docs.read("authorized-campaign-renewal")
                    if saved is not None:
                        if (
                            len(saved) != 1
                            or saved[0].get("output") != str(out)
                            or "plan" not in saved[0]
                        ):
                            raise ValueError("renewal_tip_already_claimed")
                        original = saved[0]["plan"]
                        plan = plan_for(
                            parent,
                            tip,
                            found,
                            output=out,
                            authorization=authorization,
                            now=datetime.fromisoformat(original["started_at"]),
                            implementation=freeze(),
                            seconds=original["seconds"],
                        )
                        if saved != (claim(plan, out),):
                            raise ValueError("capability_saved_claim_changed")
                    else:
                        if out.exists():
                            raise ValueError("capability_new_directory_required")
                        docs.write("authorized-campaign-renewal", (claim(plan, out),))
            elif docs.read("authorized-campaign-renewal") is not None:
                raise ValueError("renewal_tip_already_claimed")
    if not check_only:
        out.mkdir(parents=True, exist_ok=True)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        check_only=check_only,
        actual_output_created=not check_only,
        protocol_checksum=digest(plan),
        requirements=found["transition"]["coverage"],
        acquisition_targets=plan["acquisition_targets"],
        review_target_ids=plan["review_target_ids"],
        retained_assessment_ids=plan["reuse_assessment_ids"],
        scope_targets=plan["input_scope"]["targets"],
        max_new_requests=plan["max_new_requests"],
        previous_requests=sum(r["requests"] for r in tip["previous_attempts"]),
        deadline=plan["deadline"],
        api_requests=0,
        database_writes=0 if check_only else 1,
    )


def prepare_current(*, previous, output, authorization, check_only=False, network_route=None):
    """Prepare a current-method transition without modifying the prior campaign.

    All material reconstruction scopes close before the new Runtime claim and
    clock are created. Normal execution remains the existing renewal-run owner.
    """
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .campaign_renewal import completed_parent, stopped_tip
    from .execution_lease import session
    from .implementation import freeze
    from .unknown_acquisition import check_network

    previous, out = Path(previous).absolute(), Path(output).absolute()
    if (
        any(p.is_symlink() for p in (out, *out.parents))
        or out == previous
        or out.is_relative_to(previous)
        or previous.is_relative_to(out)
    ):
        raise ValueError("capability_output_must_not_change_history")
    with historical_fact_read_scope(), _projection_visible_reads():
        parent, _ = completed_parent(previous)
        with session(parent):
            binding = route_binding(parent, network_route) if network_route is not None else None
            check_network(route_configuration(parent, binding))
            tip, found, _ = inventory(
                parent,
                out,
                current=True,
                route_transition=binding,
                profile_path=preparation_profile(parent, out),
            )
    with session(parent):
        # A read-only preview is never the claim. Re-read the original tip after
        # all read-scope return checks and before the short atomic claim write.
        if tip != stopped_tip(
            parent, previous, completed_source=parent.protocol.get("mode") == "definitions"
        ):
            raise ValueError("capability_current_tip_changed_before_claim")
        docs = parent.documents(tip["run_id"])
        saved = docs.read("authorized-campaign-renewal")
        if saved and (
            len(saved) != 1 or saved[0].get("output") != str(out) or "plan" not in saved[0]
        ):
            raise ValueError("renewal_tip_already_claimed")
        if check_only and saved:
            raise ValueError("renewal_tip_already_claimed")
        plan = plan_for(
            parent,
            tip,
            found,
            output=out,
            authorization=authorization,
            now=datetime.fromisoformat(saved[0]["plan"]["started_at"])
            if saved
            else datetime.now(UTC),
            implementation=freeze(),
            current=True,
            route_transition=binding,
        )
        if saved and saved != (claim(plan, out),):
            raise ValueError("capability_saved_claim_changed")
        if not check_only and not saved:
            if out.exists():
                raise ValueError("capability_new_directory_required")
            with docs.session():
                if docs.read("authorized-campaign-renewal") is not None:
                    raise ValueError("renewal_tip_already_claimed")
                docs.write("authorized-campaign-renewal", (claim(plan, out),))
    if not check_only:
        out.mkdir(parents=True, exist_ok=True)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        check_only=check_only,
        protocol_checksum=digest(plan),
        requirements=found["transition"]["coverage"],
        acquisition_targets=plan["acquisition_targets"],
        retained_assessment_ids=plan["reuse_assessment_ids"],
        pending_review_ids=sorted(
            set(plan["review_target_ids"]) - set(plan["reuse_assessment_ids"])
        ),
        scope_targets=plan["input_scope"]["targets"],
        max_new_requests=plan["max_new_requests"],
        previous_requests=sum(r["requests"] for r in plan["previous_attempts"]),
        deadline=plan["deadline"],
        api_requests=0,
        database_writes=0 if check_only or saved else 1,
    )

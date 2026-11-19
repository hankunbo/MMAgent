"""One explicit successor over a stopped, unscored case's unattempted work.

This prepares the original RenewedCampaign, not another case executor. Historical
UNKNOWN observations and Review batches remain excluded; a route-only provider
change neither alters those records nor grants clinical qualification.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config

VERSION = "multimodal-campaign-renewal/1.52.0"
SEMANTICS = "unscored-unattempted-route-successor/1.0.0"
PLAN_FIELDS = frozenset(
    "schema_version unscored_acquisition mode facts tip original_fact_result previous_attempts "
    "parent_inventory input task provider route_transition implementation authorization "
    "max_new_requests max_model_dispatches max_cumulative_requests seconds started_at deadline "
    "prior_deadline acquisition_targets review_target_ids reuse_assessment_ids "
    "confirmed_observations_checksum expansion_policy new_public_search_allowed "
    "original_unknown_not_replayed clinical_acceptance execution_lease_profile".split()
)


def configuration(origin, binding):
    root = Path(__file__).resolve().parents[2]
    path = (root / binding["path"]).resolve(strict=True)
    if not path.is_relative_to(root / "configs/agents") or path.suffix != ".json":
        raise ValueError("unscored_public_provider_required")
    selected = read_provider_config(json.loads(path.read_bytes()))
    before, after = origin.model_dump(mode="json"), selected.model_dump(mode="json")
    prior, route = before.pop("network_route", None), after.pop("network_route", None)
    if (
        before != after
        or prior is None
        or route is None
        or prior == route
        or selected.mode != "live"
        or selected.transport.max_retries != 0
        or selected.schema_version != "agent-provider-config/9.2.0"
        or binding.get("origin_checksum") != digest(origin)
        or binding.get("provider_checksum") != digest(selected)
        or binding.get("schema_version") != "route-only-provider-transition/1.0.0"
    ):
        raise ValueError("unscored_route_only_provider_binding")
    return selected


def inventory(parent, result, previous):
    from .campaign_renewal import (
        attempted_acquisition_targets,
        stopped_tip,
        unattempted_targets,
    )
    from .pending_review import _inventory

    previous = Path(previous).absolute()
    # No chains of this recovery, or partially accepted scores treated as absent.
    if (
        parent.protocol.get("schema_version", "").startswith("multimodal-campaign-renewal/")
        or result.get("complete") is not False
        or not result.get("phases")
        or result["phases"][-1].get("target") is not None
        or result["phases"][-1].get("accepted") is not False
        or result["phases"][-1].get("acceptance_key") is not None
        or (previous / "cancel").exists()
    ):
        raise ValueError("unscored_initial_stopped_parent_required")
    tip = stopped_tip(parent, previous)
    for phase in result["phases"]:
        control = parent.sessions.read(phase["run_id"])
        if control is None or control.cancelled or control.pending or control.phase != "stopped":
            raise ValueError("unscored_parent_control_changed")
    pending = _inventory(parent, tip)
    if pending.get("accepted_ids"):
        raise ValueError("unscored_use_accepted_recovery")
    confirmed = parent.accepted_observations()
    attempted = attempted_acquisition_targets(parent, tip["previous_attempts"])
    unattempted = unattempted_targets(
        parent, attempted, confirmed, full_inventory=True, owner_methods=True
    )
    quarantined = sorted({i for b in pending["quarantined_batches"] for i in b["assessment_ids"]})
    return dict(
        tip=tip,
        pending=pending,
        attempted=sorted(attempted),
        unattempted=unattempted,
        quarantined=quarantined,
        confirmed_observations_checksum=digest(confirmed),
    )


def validate_plan(plan, *, parent, result, current):
    from .execution_lease import PROFILE

    if set(plan) != PLAN_FIELDS:
        raise ValueError("unscored_plan_fields")
    config = configuration(parent.config, plan["route_transition"])
    targets = plan.get("acquisition_targets", [])
    review_ids = sorted(set(targets) | set(current["pending"]["selected_ids"]))
    needed = len(targets) + (len(review_ids) + config.review_batching.batch_size - 1) // (
        config.review_batching.batch_size
    )
    start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
    previous = current["tip"]["previous_attempts"]
    if (
        plan.get("schema_version") != VERSION
        or plan.get("unscored_acquisition") != SEMANTICS
        or plan.get("mode") != "acquisition"
        or plan.get("facts") != current["tip"]["path"]
        or plan.get("tip") != current["tip"]
        or plan.get("original_fact_result") != digest(result)
        or plan.get("parent_inventory") != current
        or plan.get("previous_attempts") != previous
        or plan.get("input") != parent.base.manifest.checksum
        or plan.get("task") != parent.base.task.checksum
        or plan.get("provider") != digest(config)
        or plan.get("confirmed_observations_checksum") != current["confirmed_observations_checksum"]
        or plan.get("prior_deadline") != parent.protocol["deadline"]
        or not 1 <= len(targets) <= 20
        or len(set(targets)) != len(targets)
        or not set(targets) <= set(current["unattempted"])
        or set(targets) & (set(current["attempted"]) | set(current["quarantined"]))
        or plan.get("review_target_ids") != review_ids
        or set(review_ids) & set(current["quarantined"])
        or plan.get("reuse_assessment_ids") != []
        or plan.get("max_cumulative_requests") != 40
        or type(plan.get("max_new_requests")) is not int
        or not needed < plan["max_new_requests"] <= 32
        or plan.get("max_model_dispatches") != plan["max_new_requests"] - 1
        or sum(p["requests"] for p in previous) + plan["max_new_requests"] > 40
        or type(plan.get("seconds")) is not int
        or not 600 <= plan["seconds"] <= 2700
        or start.tzinfo is None
        or end != start + timedelta(seconds=plan["seconds"])
        or plan.get("execution_lease_profile") != PROFILE
        or plan.get("original_unknown_not_replayed") is not True
        or plan.get("new_public_search_allowed") is not False
        or plan.get("expansion_policy") is not None
        or plan.get("clinical_acceptance") is not False
        or not isinstance(plan.get("authorization"), str)
        or not plan["authorization"].strip()
        or len(plan["authorization"]) > 200
        or any(
            k in plan
            for k in (
                "pending_review",
                "unknown_acquisition",
                "review_clarification",
                "recorded_review_origin",
                "recovery_network_route",
                "recovery_http_client",
                "acquisition_increment_only",
                "continue_acquisition",
            )
        )
    ):
        raise ValueError("unscored_plan_binding")
    return config


def check(parent, plan):
    result = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    return validate_plan(
        plan, parent=parent, result=result, current=inventory(parent, result, plan["facts"])
    )


def prepare(
    *,
    previous,
    output,
    provider_config,
    authorization,
    request_limit=24,
    seconds=2700,
    acquisition_target=(),
    check_only=False,
):
    from .campaign_renewal import RenewedCampaign, completed_parent
    from .execution_lease import PROFILE
    from .implementation import freeze
    from .unknown_acquisition import check_network

    path, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (path, *path.parents, out, *out.parents)):
        raise ValueError("unscored_symlink")
    parent, original = completed_parent(path)
    current = inventory(parent, original, path)
    target = Path(provider_config).absolute()
    root = Path(__file__).resolve().parents[2]
    if not target.resolve().is_relative_to(root / "configs/agents"):
        raise ValueError("unscored_public_provider_required")
    selected = read_provider_config(json.loads(target.read_bytes()))
    binding = dict(
        schema_version="route-only-provider-transition/1.0.0",
        path=str(target.relative_to(root)),
        origin_checksum=digest(parent.config),
        provider_checksum=digest(selected),
    )
    config = configuration(parent.config, binding)
    check_network(config)  # Before an owner claim or credential load; no HTTP.
    selected_targets = list(acquisition_target) if acquisition_target else current["unattempted"]
    if out.exists():
        plan = json.loads((out / "protocol.json").read_bytes())
        if (
            plan.get("facts") != str(path)
            or plan.get("route_transition") != binding
            or plan.get("authorization") != authorization
            or plan.get("max_new_requests") != request_limit
            or plan.get("seconds") != seconds
            or plan.get("acquisition_targets") != selected_targets
            or plan.get("implementation") != freeze()
        ):
            raise ValueError("unscored_preparation_conflict")
    else:
        now = datetime.now(UTC)
        plan = dict(
            schema_version=VERSION,
            unscored_acquisition=SEMANTICS,
            mode="acquisition",
            facts=str(path),
            tip=current["tip"],
            original_fact_result=digest(original),
            previous_attempts=current["tip"]["previous_attempts"],
            parent_inventory=current,
            input=parent.base.manifest.checksum,
            task=parent.base.task.checksum,
            provider=digest(config),
            route_transition=binding,
            implementation=freeze(),
            authorization=authorization,
            max_new_requests=request_limit,
            max_model_dispatches=request_limit - 1,
            max_cumulative_requests=40,
            seconds=seconds,
            started_at=now.isoformat(),
            deadline=(now + timedelta(seconds=seconds)).isoformat(),
            prior_deadline=parent.protocol["deadline"],
            acquisition_targets=selected_targets,
            review_target_ids=sorted(
                set(selected_targets) | set(current["pending"]["selected_ids"])
            ),
            reuse_assessment_ids=[],
            confirmed_observations_checksum=current["confirmed_observations_checksum"],
            expansion_policy=None,
            new_public_search_allowed=False,
            original_unknown_not_replayed=True,
            clinical_acceptance=False,
            execution_lease_profile=PROFILE,
        )
    validate_plan(plan, parent=parent, result=original, current=current)
    # Exercise the actual runner assembly and freeze checks before claiming any
    # successor. This path never calls execute(), loads credentials or sends HTTP.
    RenewedCampaign(parent, out, plan).freeze()
    docs = parent.documents(plan["tip"]["run_id"])
    claim = dict(output=str(out), plan_checksum=digest(plan))
    if check_only:
        saved = docs.read("authorized-campaign-renewal")
        if saved is not None and saved != (claim,):
            raise ValueError("renewal_tip_already_claimed")
        return dict(
            status="ready_not_prepared",
            targets=selected_targets,
            review_target_ids=plan["review_target_ids"],
            quarantined=current["quarantined"],
            previous_requests=sum(p["requests"] for p in plan["previous_attempts"]),
            max_new_requests=request_limit,
            api_requests=0,
            database_writes=0,
            output_created=False,
            provider_checksum=digest(config),
        )
    with docs.session():
        saved = docs.read("authorized-campaign-renewal")
        if saved is not None and saved != (claim,):
            raise ValueError("renewal_tip_already_claimed")
        # Local intent is immutable before committing the owner claim. A lost
        # acknowledgement can use this same plan; another output cannot fork it.
        out.mkdir(mode=0o700, exist_ok=True)
        write_once(out / "protocol.json", plan)
        if saved is None:
            docs.write("authorized-campaign-renewal", (claim,))
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        targets=selected_targets,
        previous_requests=sum(p["requests"] for p in plan["previous_attempts"]),
        max_new_requests=request_limit,
        api_requests=0,
        status="already_recorded" if saved else "prepared",
    )


PROGRESS_REPAIR = "unscored-progress-repair.json"
PROGRESS_KEY = "multimodal:unscored-progress-repair"
REPAIR_FILES = {
    "packages/multimodal_research/report.py",
    "packages/multimodal_research/campaign.py",
    "packages/multimodal_research/campaign_renewal.py",
    "packages/multimodal_research/unscored_acquisition.py",
    "apps/multimodal_case.py",
}


def progress_repair_version(campaign):
    from .definition_alternative import VERSION as definition_version

    version = campaign.protocol.get("schema_version")
    if version == VERSION:
        return "unscored-progress-repair/1.0.0"
    if version == definition_version:
        return "unscored-progress-repair/1.1.0"
    raise ValueError("unscored_progress_failure_scope")


def progress_failure(campaign, *, unstarted_review=False):
    """Original owner proof of a confirmed first observation, not a new dispatch grant."""
    from .stage_failure import KEY

    version = progress_repair_version(campaign)
    if version == "unscored-progress-repair/1.1.0":
        from db.runtime.repository.case_pipeline import identity

        from .definition_alternative import SEMANTICS

        if (
            campaign.protocol.get("definition_alternative") != SEMANTICS
            or campaign.protocol.get("definition_increment_only") is not True
            or digest(campaign.base.definition_current())
            != campaign.protocol.get("source_revision_checksum")
        ):
            raise ValueError("unscored_progress_definition_changed")
        review_run = identity("RUN", "task-research:" + campaign.request_key(None))
        if (
            unstarted_review
            and campaign.contexts.read_research_run(review_run, missing_ok=True) is not None
        ):
            raise ValueError("unscored_progress_review_already_started")
    failure = json.loads((campaign.output / "stage-failure.json").read_bytes())
    phases = failure.get("confirmed_phases", [])
    if (
        failure.get("error_code") != "local_owner_error"
        or failure.get("failure_stage") != "acquisition"
        or failure.get("protocol_checksum") != digest(campaign.protocol)
        or failure.get("score_status") != "not_scored"
        or len(phases) != 1
        or not phases[0].get("accepted")
        or phases[0].get("error") is not None
        or phases[0].get("target") != campaign.protocol["acquisition_targets"][0]
        or phases[0].get("requests", {}).get("api_requests_unknown") != 0
        or phases[0].get("transport", {}).get("remote_response_confirmed") != 1
        or (campaign.output / "cancel").exists()
    ):
        raise ValueError("unscored_progress_failure_scope")
    phase = phases[0]
    docs = campaign.documents(phase["run_id"])
    control = campaign.sessions.read(phase["run_id"])
    if (
        docs.read(KEY) != (failure,)
        or docs.read("dynamic-phase-result") != (phase,)
        or control is None
        or control.phase != "stopped"
        or control.cancelled
        or control.pending
        or phase["target"] not in campaign.accepted_observations()
    ):
        raise ValueError("unscored_progress_original_owner_changed")
    return failure, phase, docs


def read_progress_repair(campaign, *, require_owner=True):
    from packages.object_store import checksum

    from .implementation import freeze

    saved = json.loads((campaign.output / PROGRESS_REPAIR).read_bytes())
    failure, phase, docs = progress_failure(campaign)
    proof_path = Path(saved["test_proof"])
    proof = json.loads(proof_path.read_bytes())
    current = freeze()
    changed = {
        p
        for p in current.keys() | campaign.protocol["implementation"].keys()
        if current.get(p) != campaign.protocol["implementation"].get(p)
    }
    if (
        saved.get("schema_version") != progress_repair_version(campaign)
        or saved.get("protocol_checksum") != digest(campaign.protocol)
        or saved.get("failure_checksum") != digest(failure)
        or saved.get("phase_checksum") != digest(phase)
        or saved.get("original_deadline") != campaign.protocol["deadline"]
        or saved.get("new_window") is not False
        or saved.get("additional_request_budget") != 0
        or saved.get("implementation_before") != campaign.protocol["implementation"]
        or saved.get("implementation_after") != current
        or not changed
        or not changed <= REPAIR_FILES
        or saved.get("test_checksum") != checksum(proof_path.read_bytes())
        or proof.get("exit_code") != 0
        or proof.get("implementation_before") != proof.get("implementation_after")
        or any(proof["implementation_after"].get(p) != current.get(p) for p in changed)
        or (proof_path.parent / "command.exit").read_text().strip() != "0"
        or checksum((proof_path.parent / "command.log").read_bytes()) != proof["log_checksum"]
        or (require_owner and docs.read(PROGRESS_KEY) != (saved,))
    ):
        raise ValueError("unscored_progress_repair_binding")
    return saved


def authorize_progress_repair(output, test_proof):
    from packages.object_store import checksum

    from .campaign_renewal import assemble
    from .implementation import freeze

    campaign = assemble(output)
    with campaign.execution_session():
        if (campaign.output / PROGRESS_REPAIR).exists():
            amendment = read_progress_repair(campaign, require_owner=False)
            campaign.guard()
            _, _, docs = progress_failure(campaign)
            with docs.session():
                saved = docs.read(PROGRESS_KEY)
                if saved is not None and saved != (amendment,):
                    raise ValueError("unscored_progress_repair_conflict")
                if saved is None:
                    docs.write(PROGRESS_KEY, (amendment,))
            read_progress_repair(campaign)
            return dict(status="already_recorded", api_requests=0)
        campaign.guard()
        failure, phase, docs = progress_failure(campaign, unstarted_review=True)
        if campaign.usage() != 1 or docs.read("multimodal-acquisition-progress") is not None:
            raise ValueError("unscored_progress_not_first_local_failure")
        before, after = campaign.protocol["implementation"], freeze()
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        proof_path = Path(test_proof).resolve(strict=True)
        proof = json.loads(proof_path.read_bytes())
        if (
            not changed
            or not changed <= REPAIR_FILES
            or "packages/multimodal_research/report.py" not in changed
            or proof.get("exit_code") != 0
            or "pytest" not in proof.get("command", [])
            or "tests/evaluation_preparation/test_unscored_acquisition.py" not in proof["command"]
            or proof.get("implementation_before") != proof.get("implementation_after")
            or any(proof["implementation_after"].get(p) != after.get(p) for p in changed)
            or (proof_path.parent / "command.exit").read_text().strip() != "0"
            or checksum((proof_path.parent / "command.log").read_bytes()) != proof["log_checksum"]
        ):
            raise ValueError("unscored_progress_repair_unverified_delta")
        amendment = dict(
            schema_version=progress_repair_version(campaign),
            protocol_checksum=digest(campaign.protocol),
            failure_checksum=digest(failure),
            phase_checksum=digest(phase),
            implementation_before=before,
            implementation_after=after,
            test_proof=str(proof_path),
            test_checksum=checksum(proof_path.read_bytes()),
            original_deadline=campaign.protocol["deadline"],
            original_failure_preserved=True,
            new_window=False,
            additional_request_budget=0,
            interpretation="resume_only_original_unattempted_targets_in_existing_window",
        )
        with docs.session():
            saved = docs.read(PROGRESS_KEY)
            if saved is not None and saved != (amendment,):
                raise ValueError("unscored_progress_repair_conflict")
            write_once(campaign.output / PROGRESS_REPAIR, amendment)
            if saved is None:
                docs.write(PROGRESS_KEY, (amendment,))
        read_progress_repair(campaign)
    return dict(
        status="repaired_original_window",
        api_requests=0,
        deadline=campaign.protocol["deadline"],
        prior_confirmed_request_replayed=False,
    )

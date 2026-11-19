"""Explicit provider change for an expired, never-executed acquired Review plan.

This delegates to the original campaign/Runtime/Review owners. It neither resumes
an old run nor changes its authorization, deadline, responses or scores.
"""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from db.runtime.repository.case_pipeline import identity
from packages.evidence_details.live_permission import validate_live_config
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config

VERSION = "multimodal-campaign-renewal/1.21.0"
PROFILE = "configs/agents/relayrouter_generic_evidence_v1.json"
ACQUISITION_VERSION = "multimodal-campaign-renewal/1.44.0"
ACQUISITION_PROFILE = "configs/agents/relayrouter_snapshot_evidence_v1.json"


def acquisition_configuration(origin, profile):
    """A different approved model on the same transport, retaining every role."""
    if profile != ACQUISITION_PROFILE:
        raise ValueError("acquisition_model_transition_profile")
    selected = validate_live_config(read_provider_config(json.loads(Path(profile).read_bytes())))
    before = origin.transport.model_dump(mode="json")
    after = selected.transport.model_dump(mode="json")
    before.pop("model")
    after.pop("model")
    if (
        before != after
        or origin.transport.model == selected.transport.model
        or origin.transport.max_retries != 0
    ):
        raise ValueError("acquisition_model_transition_contract")
    raw = origin.model_dump(mode="json")
    raw["transport"]["model"] = selected.transport.model
    return validate_live_config(read_provider_config(raw))


def acquisition_plan_for(old, *, previous, output, authorization, config, started, implementation):
    """One new bounded window over an expired, never-started selected increment."""
    targets = old.get("acquisition_targets", [])
    counts = [p.get("requests") for p in old.get("previous_attempts", [])]
    cap, seconds = old.get("max_new_requests"), old.get("seconds")
    if (
        old.get("schema_version") != "multimodal-campaign-renewal/1.26.0"
        or old.get("mode") != "acquisition"
        or old.get("acquisition_increment_only") is not True
        or not 1 <= len(targets) <= 3
        or len(set(targets)) != len(targets)
        or old.get("review_target_ids") != targets
        or old.get("acquisition_selection", {}).get("selected") != targets
        or not counts
        or any(type(c) is not int or c < 0 for c in counts)
        or type(cap) is not int
        or not 1 <= cap <= 8
        or sum(counts) + cap > 40
        or old.get("max_cumulative_requests") != 40
        or type(seconds) is not int
        or not 300 < seconds <= 5400
        or started.tzinfo is None
        or datetime.fromisoformat(old["deadline"]) > started
        or not authorization.strip()
        or len(authorization) > 200
        or old.get("original_unknown_not_replayed") is not True
        or digest(config) == old.get("provider")
    ):
        raise ValueError("acquisition_model_transition_scope_or_budget")
    plan = copy.deepcopy(old)
    plan.update(
        schema_version=ACQUISITION_VERSION,
        authorization=authorization,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=seconds)).isoformat(),
        prior_deadline=old["deadline"],
        provider=digest(config),
        implementation=implementation,
        continuation_kind="model_only_unstarted_acquisition_successor",
        provider_transition=dict(
            previous=str(Path(previous).absolute()),
            output=str(Path(output).absolute()),
            previous_protocol=digest(old),
            previous_provider=old["provider"],
            profile=ACQUISITION_PROFILE,
            profile_checksum=digest(json.loads(Path(ACQUISITION_PROFILE).read_bytes())),
            predecessor_requests=0,
            original_deadline_unchanged=True,
            old_judgments_not_rereviewed=True,
        ),
    )
    return plan


def configuration(origin, profile):
    if profile != PROFILE:
        raise ValueError("provider_transition_profile")
    selected = validate_live_config(read_provider_config(json.loads(Path(profile).read_bytes())))
    raw = origin.model_dump(mode="json")
    for field in ("base_url", "model", "api_key_env"):
        raw["transport"][field] = getattr(selected.transport, field)
    config = validate_live_config(read_provider_config(raw))
    if (
        config.transport.max_retries != 0
        or config.transport.base_url == origin.transport.base_url
        or next(r.prompt_version for r in config.roles if r.role == "review")
        != "review-direction-2.3.0"
    ):
        raise ValueError("provider_transition_contract")
    return config


def plan_for(old, *, previous, output, authorization, config, started, implementation):
    if (
        old.get("schema_version") != "multimodal-campaign-renewal/1.20.0"
        or old.get("mode") != "review"
        or not old.get("review_acquired_only")
        or not old.get("repair_acquired_contract")
        or old.get("max_model_dispatches") != 1
        or old.get("acquisition_targets") != []
        or len(old.get("review_target_ids", [])) != 1
        or datetime.fromisoformat(old["deadline"]) > started
        or started.tzinfo is None
        or not authorization.strip()
        or len(authorization) > 200
        or old.get("max_new_requests", 0) < 2
        or sum(p["requests"] for p in old["previous_attempts"]) + 2 > 40
    ):
        raise ValueError("provider_transition_scope_or_budget")
    plan = copy.deepcopy(old)
    plan.update(
        schema_version=VERSION,
        authorization=authorization,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=1800)).isoformat(),
        prior_deadline=old["deadline"],
        seconds=1800,
        max_new_requests=2,
        provider=digest(config),
        implementation=implementation,
        continuation_kind="authorized_new_provider_window_over_confirmed_facts",
        provider_transition=dict(
            previous=str(Path(previous).absolute()),
            output=str(Path(output).absolute()),
            previous_protocol=digest(old),
            previous_provider=old["provider"],
            profile=PROFILE,
            profile_checksum=digest(json.loads(Path(PROFILE).read_bytes())),
            predecessor_requests=0,
            original_deadline_unchanged=True,
        ),
    )
    return plan


def claim_key(old):
    return "authorized-provider-transition:" + digest(old)[7:39]


def verify_unexecuted(campaign, previous, old):
    from .campaign_renewal import stopped_tip

    path = Path(previous).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("provider_transition_symlink")
    if {p.name for p in path.iterdir()} - {"protocol.json", "lock"}:
        raise ValueError("provider_transition_predecessor_has_activity")
    if json.loads((path / "protocol.json").read_bytes()) != old:
        raise ValueError("provider_transition_predecessor_protocol_changed")
    run_id = identity("RUN", "task-research:" + campaign.request_key(None))
    if (
        campaign.contexts.read_research_run(run_id, missing_ok=True) is not None
        or campaign.sessions.read(run_id) is not None
    ):
        raise ValueError("provider_transition_predecessor_has_runtime")
    if stopped_tip(campaign.parent, old["tip"]["path"]) != old["tip"]:
        raise ValueError("provider_transition_owner_changed")
    if digest(campaign.parent.accepted_observations()) != old["confirmed_observations_checksum"]:
        raise ValueError("provider_transition_observations_changed")


def prepare(*, previous, output, authorization, model_only_acquisition=False):
    import fcntl

    from apps.vectorengine_single_case import write_once

    from .campaign_renewal import assemble
    from .implementation import freeze

    previous, out = Path(previous).absolute(), Path(output).absolute()
    old = json.loads((previous / "protocol.json").read_bytes())
    # Lock the predecessor without editing any of its frozen files. Its original
    # deadline remains expired; another process cannot dispatch it concurrently.
    lock_path = Path(".local") / ("provider-transition-" + digest(old)[7:39] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = assemble(previous)
        verify_unexecuted(campaign, previous, old)
        config = (
            acquisition_configuration(campaign.config, ACQUISITION_PROFILE)
            if model_only_acquisition
            else configuration(campaign.config, PROFILE)
        )
        builder = acquisition_plan_for if model_only_acquisition else plan_for
        plan = builder(
            old,
            previous=previous,
            output=out,
            authorization=authorization,
            config=config,
            started=datetime.now(UTC),
            implementation=freeze(),
        )
        if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
            raise ValueError("provider_transition_new_directory_required")
        docs = campaign.parent.documents(old["tip"]["run_id"])
        claim = dict(output=str(out), plan_checksum=digest(plan))
        with docs.session():
            if docs.read(claim_key(old)) is not None:
                raise ValueError("provider_transition_already_claimed")
            docs.write(claim_key(old), (claim,))
        out.mkdir(mode=0o700)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        api_requests=0,
        previous_requests=sum(p["requests"] for p in old["previous_attempts"]),
        max_model_dispatches=plan.get("max_model_dispatches"),
        max_new_requests=plan["max_new_requests"],
        acquisition_targets=plan["acquisition_targets"],
        new_vision_requests=0,
        provider_profile=ACQUISITION_PROFILE if model_only_acquisition else PROFILE,
    )


def read_ancestry(out, depth, ancestry):
    from .campaign_renewal import MAX_HISTORY_DEPTH

    # completed_parent already records its current node before dispatching here.
    # One terminal occurrence is normal; earlier repeats still denote a cycle.
    if depth > MAX_HISTORY_DEPTH or out in ancestry[:-1]:
        raise ValueError("provider_transition_version_or_cycle")
    if not ancestry or ancestry[-1] != out:
        ancestry = (*ancestry, out)
    return ancestry


def assemble(output, *, depth=0, ancestry=()):
    from .campaign_renewal import RenewedCampaign
    from .campaign_renewal import assemble as assemble_old

    out = Path(output).absolute()
    plan = json.loads((out / "protocol.json").read_bytes())
    ancestry = read_ancestry(out, depth, ancestry)
    if plan.get("schema_version") not in {VERSION, ACQUISITION_VERSION}:
        raise ValueError("provider_transition_version_or_cycle")
    acquisition = plan["schema_version"] == ACQUISITION_VERSION
    proof = plan["provider_transition"]
    previous = Path(proof["previous"])
    old = json.loads((previous / "protocol.json").read_bytes())
    if old.get("schema_version") != (
        "multimodal-campaign-renewal/1.26.0"
        if acquisition
        else "multimodal-campaign-renewal/1.20.0"
    ):
        raise ValueError("provider_transition_predecessor_version")
    origin = assemble_old(previous, depth=depth + 1, ancestry=ancestry)
    verify_unexecuted(origin, previous, old)
    configure = acquisition_configuration if acquisition else configuration
    config = configure(origin.config, proof["profile"])
    builder = acquisition_plan_for if acquisition else plan_for
    expected = builder(
        old,
        previous=previous,
        output=out,
        authorization=plan["authorization"],
        config=config,
        started=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
    )
    docs = origin.parent.documents(old["tip"]["run_id"])
    if plan != expected or docs.read(claim_key(old)) != (
        dict(output=str(out), plan_checksum=digest(plan)),
    ):
        raise ValueError("provider_transition_claim_binding")
    # The original constructor validates the complete acquired-observation proof,
    # previous authorized bounds and citation upgrade. Only the explicit new
    # protocol/config/window differ; execution and acceptance remain original.
    campaign = RenewedCampaign(origin.parent, out, old)
    campaign.protocol, campaign.config = plan, config
    campaign.provider_transition_origin = (origin, previous, old)
    return campaign


def recheck(campaign):
    origin, previous, old = campaign.provider_transition_origin
    verify_unexecuted(origin, previous, old)
    configure = (
        acquisition_configuration
        if campaign.protocol["schema_version"] == ACQUISITION_VERSION
        else configuration
    )
    expected = configure(origin.config, campaign.protocol["provider_transition"]["profile"])
    if digest(expected) != campaign.protocol["provider"]:
        raise ValueError("provider_transition_config_changed")

"""Explicit bounded new-material Review using original campaign and domain owners.

No replay, score copying, acquisition, search, new model or parallel persistence.
Legacy operations select at most two confirmed observations; the explicit
aggregate method profile selects at most six accepted null targets with new context.
"""

import json
from contextlib import contextmanager
from copy import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evidence_details.live_permission import validate_live_config
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config

from .campaign_renewal import RenewedCampaign, completed_parent, stopped_tip
from .implementation import freeze
from .provider_transition import PROFILE, read_ancestry
from .review_context import collect, supplementary

VERSION = "multimodal-campaign-renewal/1.22.0"
METHOD_VERSION = "multimodal-campaign-renewal/1.45.0"
HARVARD_METHOD_VERSION = "multimodal-campaign-renewal/1.46.0"
AGGREGATE_METHOD_VERSION = "multimodal-campaign-renewal/1.69.0"
RNFL_UNIT_PROFILE = "grape-rnfl-device-unit/1.0.0"
TONOPACHY_UNIT_PROFILE = "grape-tonopachy-device-unit/1.0.0"
TONOPACHY_METHOD_VERSION = "multimodal-campaign-renewal/1.72.0"


def aggregate_profile(parent, metadata_profile):
    # Preserve the RNFL aggregate-only contract. The new profile accepts either
    # an original single-Review owner or a restored reasoned aggregate owner.
    return metadata_profile == RNFL_UNIT_PROFILE or (
        metadata_profile == TONOPACHY_UNIT_PROFILE
        and getattr(parent, "aggregation_baseline", None) is not None
    )


def method_version(profile):
    from packages.dataset_adapters.multimodal.grape_semantics import METHOD_PROFILE as GRAPE
    from packages.dataset_adapters.multimodal.harvard_semantics import METHOD_PROFILE as HARVARD

    versions = {
        GRAPE: METHOD_VERSION,
        HARVARD: HARVARD_METHOD_VERSION,
        RNFL_UNIT_PROFILE: AGGREGATE_METHOD_VERSION,
        TONOPACHY_UNIT_PROFILE: TONOPACHY_METHOD_VERSION,
    }
    if profile not in versions:
        raise ValueError("context_review_metadata_profile_or_retry")
    return versions[profile]


def method_reuse_ids(parent, tip, selected):
    """Already owner-verified acceptance; new metadata must not replace an old score."""
    docs = parent.documents(tip["run_id"])
    result = docs.read("dynamic-phase-result")
    if not result or not result[0].get("accepted") or result[0].get("target") is not None:
        raise ValueError("context_review_reuse_origin")
    saved = docs.read("accepted:" + result[0]["acceptance_key"])
    if not saved or len(saved) != 4 or saved[-1].get("status") != "accepted":
        raise ValueError("context_review_reuse_origin")
    ids = sorted(
        i["assessment_id"] for i in saved[0]["candidate"]["items"] if i["status"] == "assessed"
    )
    if not ids or len(ids) != len(set(ids)) or set(ids) & set(selected):
        raise ValueError("context_review_new_material_not_existing_score")
    return ids


def selected_configuration(origin, metadata_profile):
    if metadata_profile is None:
        return configuration(origin)
    method_version(metadata_profile)
    if origin.transport.max_retries != 0:
        raise ValueError("context_review_metadata_profile_or_retry")
    # No route/model/Prompt change hidden inside a dataset metadata correction.
    return validate_live_config(read_provider_config(origin.model_dump(mode="json")))


def configuration(origin, *, profile=PROFILE):
    selected = validate_live_config(read_provider_config(json.loads(Path(profile).read_bytes())))
    if origin.schema_version not in {"agent-provider-config/8.0.0", "agent-provider-config/9.0.0"}:
        raise ValueError("context_review_provider_version")
    raw = origin.model_dump(mode="json")
    for field in ("base_url", "model", "api_key_env"):
        raw["transport"][field] = getattr(selected.transport, field)
    raw["transport"].update(timeout_seconds=300, max_retries=0)
    raw["review_batching"]["batch_size"] = 1
    review = next(r for r in raw["roles"] if r["role"] == "review")
    review.update(
        prompt_version="review-direction-2.4.0",
        prompt_checksum=digest(
            Path("configs/prompts/agents/review-direction-2.4.0.txt").read_text()
        ),
        max_output_tokens=4096,
    )
    return validate_live_config(read_provider_config(raw))


def aggregate_origin_proof(parent, result, tip):
    """Read the exact owner-restored aggregate, never scan for a favorable score."""
    from types import SimpleNamespace

    from agents.expert_agent.reporting import read_research_report
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.accepted_aggregation import AggregatedSubmission, _target_context

    baseline = getattr(parent, "aggregation_baseline", None)
    if baseline is None or type(baseline.submission) is not AggregatedSubmission:
        raise ValueError("context_review_requires_original_reasoned_aggregate")
    docs, state = baseline.documents, baseline.state
    control = parent.sessions.read(docs.run_id)
    if (
        result.get("schema_version") != "multimodal-accepted-aggregation-result/1.0.0"
        or result.get("complete") is not True
        or not result.get("phases")
        or result["phases"][-1].get("run_id") != docs.run_id
        or result["phases"][-1].get("acceptance_key") != baseline.acceptance_key
        or tip.get("run_id") != docs.run_id
        or tip.get("protocol_kind") != "accepted_aggregation_manifest"
        or tip.get("protocol") != digest(baseline.manifest)
        or tip.get("result") != digest(result)
        or control is None
        or control.cancelled
        or control.pending
        or control.phase != "stopped"
        or docs.read("accepted-aggregation-manifest") != (baseline.manifest,)
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),)
    ):
        raise ValueError("context_review_aggregate_current_binding")
    service = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
    )
    report = read_research_report(service, key=baseline.acceptance_key, parent=state)
    if (
        report["receipt"]["status"] != "accepted"
        or report["receipt"]["submission_checksum"] != digest(baseline.submission)
        or report["records"]["candidate"] != baseline.submission.candidate.model_dump(mode="json")
    ):
        raise ValueError("context_review_aggregate_acceptance")
    candidates = baseline.submission.candidate.items
    targets = sorted(
        i.assessment_id
        for i in candidates
        if i.judgment.value is None and i.status != "assessed" and i.observation_available
    )
    if not targets:
        raise ValueError("context_review_no_observed_null_targets")
    return dict(
        schema_version="aggregate-null-review-origin/1.0.0",
        run_id=docs.run_id,
        acceptance_key=baseline.acceptance_key,
        submission_checksum=digest(baseline.submission),
        acceptance_checksum=digest(docs.read("accepted:" + baseline.acceptance_key)),
        state_checksum=digest(state),
        manifest_checksum=digest(baseline.manifest),
        null_ids=targets,
        context_refs={
            t: [r.model_dump(mode="json") for r in _target_context(state, t)] for t in targets
        },
        current_original_owner_restored=True,
    )


def origin_proof(parent, result, tip, *, metadata_profile=None):
    from agents.expert_agent.committed_review import read_committed_review

    if aggregate_profile(parent, metadata_profile):
        return aggregate_origin_proof(parent, result, tip)

    if (
        parent.protocol.get("schema_version") in {VERSION, METHOD_VERSION, HARVARD_METHOD_VERSION}
        or result.get("complete") is not True
        or not result["phases"][-1].get("accepted")
        or result["phases"][-1].get("target") is not None
    ):
        raise ValueError("context_review_requires_completed_parent_not_recursive")
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(tip["run_id"])
    if control.cancelled or control.pending or control.phase != "stopped":
        raise ValueError("context_review_origin_control")
    state = docs.read("dynamic-phase-input")
    if not state:
        raise ValueError("context_review_origin_state")
    proof = read_committed_review(
        docs,
        parent=state[0],
        config=parent.config,
        confirmed_review_reader=getattr(parent.base, "confirmed_review_reader", None),
    )
    if proof is None or proof["acceptance_key"] != result["phases"][-1]["acceptance_key"]:
        raise ValueError("context_review_origin_acceptance")
    return proof


def plan_for(
    parent,
    result,
    tip,
    contexts,
    proof,
    config,
    *,
    output,
    authorization,
    started,
    metadata_profile=None,
):
    selected = sorted(contexts)
    aggregated = aggregate_profile(parent, metadata_profile)
    batches = (
        len(selected) + config.review_batching.batch_size - 1
    ) // config.review_batching.batch_size
    request_cap = batches + 1 if aggregated else len(selected) + 1
    if (
        not 1 <= len(selected) <= (6 if aggregated else 2)
        or not authorization.strip()
        or len(authorization) > 200
        or started.tzinfo is None
        or sum(p["requests"] for p in tip["previous_attempts"]) + request_cap > 40
    ):
        raise ValueError("context_review_scope_or_budget")
    # Check exact existing payload field bounds before committing a new window.
    supplementary(contexts)
    plan = dict(
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
        provider_profile=PROFILE,
        implementation=freeze(),
        max_new_requests=len(selected) + 1,
        max_model_dispatches=len(selected),
        max_cumulative_requests=40,
        seconds=1800,
        started_at=started.isoformat(),
        deadline=(started + timedelta(seconds=1800)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        expansion_policy=None,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
        new_vision_requests=0,
        new_public_search_allowed=False,
        acquisition_targets=[],
        review_target_ids=selected,
        output=str(Path(output).absolute()),
        material_context_review=dict(
            contexts=contexts,
            origin_acceptance=proof,
            purpose="new_material_direction_reassessment_not_old_score_replacement",
            cross_provider_aggregation="not_authorized_by_this_review",
        ),
    )
    if metadata_profile is not None:
        from .execution_lease import PROFILE as LEASE_PROFILE

        if config != selected_configuration(parent.config, metadata_profile):
            raise ValueError("context_review_metadata_provider_changed")
        if aggregated and (
            proof.get("schema_version") != "aggregate-null-review-origin/1.0.0"
            or not set(selected) <= set(proof["null_ids"])
        ):
            raise ValueError("context_review_only_accepted_null_targets")
        if aggregated:
            for target in selected:
                expected = {digest(o.reference) for o in supplementary({target: contexts[target]})}
                prior = {digest(r) for r in proof["context_refs"][target]}
                if expected <= prior:
                    raise ValueError("context_review_no_new_target_material")
        reused = [] if aggregated else method_reuse_ids(parent, tip, selected)
        plan.update(
            schema_version=method_version(metadata_profile),
            metadata_profile=metadata_profile,
            provider_profile="unchanged_parent_provider",
            execution_lease_profile=LEASE_PROFILE,
            seconds=2700,
            deadline=(started + timedelta(seconds=2700)).isoformat(),
            reassessment_ids=selected,
            reuse_assessment_ids=reused,
            review_target_ids=sorted([*selected, *reused]),
        )
        if aggregated:
            batches = (
                len(selected) + config.review_batching.batch_size - 1
            ) // config.review_batching.batch_size
            plan.update(
                seconds=5400,
                deadline=(started + timedelta(seconds=5400)).isoformat(),
                max_new_requests=batches + 1,
                max_model_dispatches=batches,
                aggregate_context_review=True,
                old_accepted_items_not_replayed=True,
            )
        if plan["schema_version"] == HARVARD_METHOD_VERSION:
            from .campaign_renewal import RETAINED_CONTEXT_SCOPE

            plan["review_context_scope"] = RETAINED_CONTEXT_SCOPE
    return plan


def claim_key(plan):
    return (
        "authorized-material-context-review:"
        + digest(
            dict(
                contexts=plan["material_context_review"]["contexts"],
                provider=plan["provider"],
            )
        )[7:39]
    )


def checked_path(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("context_review_symlink")
    return path


@contextmanager
def preparation_session(previous, parent, metadata_profile):
    import fcntl

    if metadata_profile is not None:
        # Existing original Runtime lease; no silent filesystem-lock fallback.
        from .execution_lease import session

        with session(parent):
            yield
        return
    lock_path = Path(".local") / ("context-review-" + digest(str(previous))[7:39] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(*, previous, output, assessment_id, authorization, metadata_profile=None):
    previous, out = checked_path(previous), checked_path(output)
    if out.exists() or len(set(assessment_id)) != len(assessment_id):
        raise ValueError("context_review_new_directory_and_unique_targets")
    parent, result = completed_parent(previous)
    with preparation_session(previous, parent, metadata_profile):
        parent.base.check()
        tip = stopped_tip(parent, previous)
        proof = origin_proof(parent, result, tip, metadata_profile=metadata_profile)
        contexts = collect(
            parent.base,
            parent.accepted_observations(),
            assessment_id,
            metadata_profile=metadata_profile,
        )
        config = selected_configuration(parent.config, metadata_profile)
        plan = plan_for(
            parent,
            result,
            tip,
            contexts,
            proof,
            config,
            output=out,
            authorization=authorization,
            started=datetime.now(UTC),
            metadata_profile=metadata_profile,
        )
        docs = parent.documents(tip["run_id"])
        claim = dict(output=str(out), plan_checksum=digest(plan))
        with docs.session():
            if docs.read(claim_key(plan)) is not None:
                raise ValueError("context_review_identical_material_already_claimed")
            docs.write(claim_key(plan), (claim,))
        out.mkdir(mode=0o700)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        review_ids=plan["review_target_ids"],
        api_requests=0,
        max_model_dispatches=plan["max_model_dispatches"],
        cumulative_before=sum(p["requests"] for p in plan["previous_attempts"]),
        context_checksum=digest(contexts),
        provider_profile=plan["provider_profile"],
    )


def assemble(output, *, depth=0, ancestry=()):
    out = checked_path(output)
    ancestry = read_ancestry(out, depth, ancestry)
    plan = json.loads((out / "protocol.json").read_bytes())
    if plan.get("schema_version") not in {
        VERSION,
        METHOD_VERSION,
        HARVARD_METHOD_VERSION,
        AGGREGATE_METHOD_VERSION,
        TONOPACHY_METHOD_VERSION,
    } or plan.get("output") != str(out):
        raise ValueError("context_review_plan_identity")
    metadata_profile = plan.get("metadata_profile")
    if plan["schema_version"] != (
        method_version(metadata_profile) if metadata_profile else VERSION
    ):
        raise ValueError("context_review_metadata_version")
    parent, result = completed_parent(plan["facts"], depth=depth + 1, ancestry=ancestry)
    tip = stopped_tip(parent, plan["facts"])
    proof = origin_proof(parent, result, tip, metadata_profile=metadata_profile)
    contexts = collect(
        parent.base,
        parent.accepted_observations(),
        plan["reassessment_ids"] if metadata_profile else plan["review_target_ids"],
        metadata_profile=metadata_profile,
    )
    config = selected_configuration(parent.config, metadata_profile)
    expected = plan_for(
        parent,
        result,
        tip,
        contexts,
        proof,
        config,
        output=out,
        authorization=plan["authorization"],
        started=datetime.fromisoformat(plan["started_at"]),
        metadata_profile=metadata_profile,
    )
    # Reading historical results must not retroactively replace their frozen implementation.
    expected["implementation"] = plan["implementation"]
    docs = parent.documents(tip["run_id"])
    if plan != expected or docs.read(claim_key(plan)) != (
        dict(output=str(out), plan_checksum=digest(plan)),
    ):
        raise ValueError("context_review_owner_binding")
    return from_verified_parent(parent, out, plan, contexts, config)


def from_verified_parent(parent, out, plan, contexts, config):
    """Build with original owners after exact context and claim verification."""
    metadata_profile = plan.get("metadata_profile")
    tip = plan["tip"]
    # Reuse original execution, budget, Review, acceptance, scoring and recovery ports.
    base_parent = copy(parent)
    base_parent.base = copy(parent.base)
    base_parent.base.confirmed_review_reader = None
    base_parent.base.review_clarification = None
    base_parent.base.review_target_ids = tuple(plan["review_target_ids"])
    base_parent.base.review_display_context = contexts
    base_parent.base.review_metadata_profile = metadata_profile
    base_parent.base.review_context_scope = None
    if metadata_profile is not None and not aggregate_profile(parent, metadata_profile):
        from .campaign_renewal import RETAINED_CONTEXT_SCOPE, accepted_reader

        # Reuse the parent's already assessed items through its original reader;
        # only newly contextualized, previously unscored items need a request.
        base_parent.base.confirmed_review_reader = accepted_reader(parent, tip["run_id"])
        base_parent.base.review_context_scope = RETAINED_CONTEXT_SCOPE
    campaign = RenewedCampaign(base_parent, out, plan)
    campaign.parent = parent
    campaign.config = config
    if metadata_profile is not None:
        campaign.reused_review_count = len(plan["reuse_assessment_ids"])
    return campaign

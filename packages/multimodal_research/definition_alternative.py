"""Bounded alternative observation for an admitted, not-yet-reviewed definition.

Original Source admission, image, Runtime, alternative-view and Review owners
remain authoritative. This is a successor plan, not a second case executor.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .execution_lease import PROFILE

VERSION = "multimodal-campaign-renewal/1.64.0"
SEMANTICS = "admitted-definition-alternative-observation/1.0.0"

PLAN_FIELDS = frozenset(
    "acquisition_selection acquisition_targets alternative_view authorization "
    "clinical_acceptance confirmed_observations_checksum deadline definition_alternative "
    "definition_increment_only exact_request_budget execution_lease_profile expansion_policy "
    "facts implementation input logical_campaign_id max_cumulative_requests "
    "max_model_dispatches max_new_requests mode new_models new_public_search_allowed "
    "original_fact_result original_unknown_not_replayed previous_attempts prior_deadline "
    "provider reuse_assessment_ids review_partition review_target_ids schema_version seconds "
    "source_provider_allowed source_revision_checksum started_at task tip ".split()
)


def selection(protocol, result, revision, events):
    """No completed/UNKNOWN Review may be erased by the alternative window."""
    phases = result.get("phases", [])
    replaced = [r["id"] for r in revision.get("replaced", [])]
    selected = sorted(set(revision.get("added", [])) | set(replaced))
    if (
        protocol.get("mode") != "definitions"
        or protocol.get("definition_increment_only") is not True
        or result.get("complete") is not True
        or len(selected) != 1
        or len(phases) != 2
        or phases[0].get("target") != selected[0]
        or phases[0].get("accepted") is not True
        or not phases[0].get("acceptance_key")
        or phases[1].get("target") is not None
        or phases[1].get("accepted") is not False
        or phases[1].get("acceptance_key") is not None
        or phases[1].get("error") is not None
        or phases[1].get("evidence_limited") is not True
        or any(e.get("phase") in {"reserved", "completed", "response_confirmation"} for e in events)
        or phases[1].get("requests", {}).get("api_requests_observed") != 0
        or phases[1].get("requests", {}).get("api_requests_unknown") != 0
    ):
        raise ValueError("definition_alternative_requires_confirmed_unreviewed_increment")
    rows = [r for r in revision["rows"] if r["id"] == selected[0]]
    if (
        len(rows) != 1
        or rows[0].get("acquisition_method") != "vision"
        or not rows[0].get("component")
        or rows[0].get("acquisition_allowed", True) is not True
        or rows[0].get("review_allowed", True) is not True
        or not revision.get("accepted")
    ):
        raise ValueError("definition_alternative_current_requirement")
    return selected


def confirmed_sources(parent, docs, events, protocol):
    """A confirmed schema correction is historical work, not an unknown request."""
    reservations = [e for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    keys = [e["request_id"] for e in reservations]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    if (
        not 1 <= len(keys) <= 2
        or len(set(keys)) != len(keys)
        or [r["request_id"] for r in records] != keys
        or confirmations
        != [dict(phase="response_confirmation", request_id=k, confirmed=True) for k in keys]
        or records[-1].get("error") is not None
    ):
        raise ValueError("definition_alternative_source_response_unconfirmed")
    operations = docs.read_many(
        tuple(
            f"definition:operation:{i}"
            for i in range(protocol["expansion_policy"]["max_operations"])
        )
    )
    llm = [o[0] for o in operations if o and o[0].get("kind") == "llm"]
    if len(llm) != 1 or not llm[0]["key"].endswith(":proposal"):
        raise ValueError("definition_alternative_source_operation_binding")
    proposal_key = llm[0]["key"].removesuffix(":proposal") + ":confirmed-proposal"
    proposal = docs.read(proposal_key)
    if not proposal or len(proposal) != 1 or digest(proposal[0]) != records[-1]["output_checksum"]:
        raise ValueError("definition_alternative_source_proposal_binding")
    for retry, (reserved, record) in enumerate(zip(reservations, records, strict=True)):
        raw = docs.read("definition:response:" + record["request_id"])
        if (
            record.get("role") != "source"
            or record.get("error") not in {None, "schema"}
            or record.get("simulated") is not False
            or record.get("api_requests") != 1
            or record.get("retry_index") != retry
            or record.get("invocation_id") != keys[0]
            or record.get("scope", {}).get("run_id") != docs.run_id
            or record["scope"].get("case_code") != parent.base.binding["case_code"]
            or record["scope"].get("input_checksum") != protocol["input"]
            or reserved.get("scope_checksum") != digest(record["scope"])
            or reserved.get("config_checksum") != record.get("config_checksum")
            or record.get("prompt_version") != protocol["expansion_policy"]["prompt_version"]
            or not raw
            or len(raw) != 1
            or raw[0].get("record") != record
            or not raw[0].get("raw_text")
            or (
                record.get("error") is not None
                and digest(raw[0]["raw_text"].encode("utf-8").hex())
                != record.get("output_checksum")
            )
        ):
            raise ValueError("definition_alternative_source_response_binding")
    return digest(records)


def unreviewed(parent, target):
    """Read current definition qualification, not a caller-supplied approval bit."""
    revision = parent.base.definition_current()
    result = json.loads((parent.output / "result.json").read_bytes())
    final = result["phases"][-1]
    docs = parent.documents(final["run_id"])
    if docs.read("dynamic-phase-result") != (final,):
        raise ValueError("definition_alternative_review_phase_binding")
    if selection(parent.protocol, result, revision, RuntimeAgentJournal(docs).read()) != [target]:
        raise ValueError("definition_alternative_target_binding")
    return dict(
        revision_checksum=digest(revision),
        review_run_id=final["run_id"],
        review_phase_checksum=digest(final),
        review_requests=0,
    )


def check(parent, plan):
    from .alternative_acquisition import TILED_METHOD, proof
    from .campaign_renewal import stopped_tip, validate_exact_acquisition_budget

    if set(plan) - {"alternative_view_metadata_checksum"} != PLAN_FIELDS:
        raise ValueError("definition_alternative_plan_fields")
    if (parent.output / "cancel").exists():
        raise ValueError("definition_alternative_parent_cancelled")
    tip = stopped_tip(parent, parent.output, alternative_definition=True)
    revision = parent.base.definition_current()
    if digest(revision) != tip["definition"]["revision"]:
        raise ValueError("definition_alternative_revision_changed")
    from .campaign_renewal import gate_selection

    parent.targets = parent.available_targets(parent.base)
    gate_selection(parent, revision)
    targets = plan.get("acquisition_targets", [])
    if len(targets) != 1:
        raise ValueError("definition_alternative_one_target")
    expected = proof(parent, targets[0], method=TILED_METHOD, definition_increment=True)
    start = datetime.fromisoformat(plan["started_at"])
    end = datetime.fromisoformat(plan["deadline"])
    if (
        plan.get("schema_version") != VERSION
        or plan.get("definition_alternative") != SEMANTICS
        or plan.get("execution_lease_profile") != PROFILE
        or plan.get("mode") != "acquisition"
        or plan.get("definition_increment_only") is not True
        or plan.get("facts") != str(parent.output)
        or plan.get("tip") != tip
        or plan.get("previous_attempts") != tip["previous_attempts"]
        or plan.get("original_fact_result") != tip["result"]
        or plan.get("input") != parent.base.manifest.checksum
        or plan.get("task") != parent.base.task.checksum
        or plan.get("provider") != digest(parent.config)
        or plan.get("confirmed_observations_checksum") != digest(parent.accepted_observations())
        or plan.get("alternative_view") != expected
        or plan.get("source_revision_checksum") != tip["definition"]["revision"]
        or plan.get("logical_campaign_id")
        != getattr(parent, "logical_campaign_id", parent.campaign_id)
        or plan.get("prior_deadline") != parent.protocol["deadline"]
        or plan.get("review_target_ids") != targets
        or plan.get("reuse_assessment_ids") != []
        or plan.get("seconds") != 5400
        or start.tzinfo is None
        or end != start + timedelta(seconds=5400)
        or not 2 <= plan.get("max_new_requests", 0) <= 3
        or parent.config.transport.max_retries != 0
        or plan.get("new_public_search_allowed") is not False
        or plan.get("source_provider_allowed") is not False
        or plan.get("expansion_policy") is not None
        or plan.get("original_unknown_not_replayed") is not True
        or plan.get("clinical_acceptance") is not False
        or plan.get("new_models") != 0
        or not isinstance(plan.get("authorization"), str)
        or not plan["authorization"].strip()
        or len(plan["authorization"]) > 200
        or any(
            plan.get(k)
            for k in (
                "pending_review",
                "unknown_acquisition",
                "review_clarification",
                "recorded_review_origin",
                "recovery_network_route",
                "recovery_http_client",
                "acquisition_increment_only",
                "continue_acquisition",
                "review_definition_increment",
            )
        )
    ):
        raise ValueError("definition_alternative_plan_binding")
    validate_exact_acquisition_budget(plan, parent.config)
    return targets


def prepare(*, previous, output, target, authorization, request_limit=3, check_only=False):
    from .active_continuation import select_acquisition_targets
    from .alternative_acquisition import TILED_METHOD, prepare_material, proof
    from .campaign_renewal import RenewedCampaign, assemble, stopped_tip
    from .implementation import freeze
    from .unknown_acquisition import check_network

    path, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (path, *path.parents, out, *out.parents)):
        raise ValueError("definition_alternative_symlink")
    parent = assemble(path)
    tip = stopped_tip(parent, path, alternative_definition=True)
    revision = parent.base.definition_current()
    if digest(revision) != tip["definition"]["revision"]:
        raise ValueError("definition_alternative_revision_changed")
    from .campaign_renewal import gate_selection

    parent.targets = parent.available_targets(parent.base)
    gate_selection(parent, revision)
    current = proof(parent, target, method=TILED_METHOD, definition_increment=True)
    check_network(parent.config)
    now = datetime.now(UTC)
    plan = dict(
        schema_version=VERSION,
        definition_alternative=SEMANTICS,
        execution_lease_profile=PROFILE,
        mode="acquisition",
        definition_increment_only=True,
        facts=str(path),
        tip=tip,
        original_fact_result=tip["result"],
        previous_attempts=tip["previous_attempts"],
        authorization=authorization,
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(parent.config),
        implementation=freeze(),
        max_new_requests=request_limit,
        max_model_dispatches=request_limit,
        max_cumulative_requests=40,
        seconds=5400,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=5400)).isoformat(),
        prior_deadline=parent.protocol["deadline"],
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        expansion_policy=None,
        source_provider_allowed=False,
        new_public_search_allowed=False,
        original_unknown_not_replayed=True,
        clinical_acceptance=False,
        new_models=0,
        logical_campaign_id=getattr(parent, "logical_campaign_id", parent.campaign_id),
        source_revision_checksum=tip["definition"]["revision"],
        acquisition_targets=[target],
        review_target_ids=[target],
        reuse_assessment_ids=[],
        acquisition_selection=select_acquisition_targets([target], [target], owner_methods=True),
        review_partition="owner_bound_reused_and_new/1.0.0",
        exact_request_budget="dispatch_only_local_completion_zero_request/1.0",
        alternative_view=current,
    )
    existing = out.exists()
    if existing:
        saved_plan = json.loads((out / "protocol.json").read_bytes())
        if (
            saved_plan.get("facts") != str(path)
            or saved_plan.get("acquisition_targets") != [target]
            or saved_plan.get("authorization") != authorization
            or saved_plan.get("max_new_requests") != request_limit
            or saved_plan.get("implementation") != freeze()
        ):
            raise ValueError("definition_alternative_preparation_conflict")
        plan = saved_plan
    check(parent, plan)
    docs = parent.documents(tip["run_id"])
    if check_only:
        claimed = docs.read("authorized-campaign-renewal")
        if claimed is not None and (
            not existing or claimed != (dict(output=str(out), plan_checksum=digest(plan)),)
        ):
            raise ValueError("renewal_tip_already_claimed")
        return dict(
            eligible=True,
            target=target,
            previous_requests=sum(p["requests"] for p in tip["previous_attempts"]),
            max_new_requests=request_limit,
            new_api_requests=0,
            database_writes=0,
            plan=plan,
        )
    work = None
    if not existing:
        if docs.read("authorized-campaign-renewal") is not None:
            raise ValueError("renewal_tip_already_claimed")
        work, metadata = prepare_material(parent, current, out)
        plan["alternative_view_metadata_checksum"] = digest(metadata)
    candidate = RenewedCampaign(parent, out, plan)
    candidate.freeze()
    if existing:
        candidate.base.alternative_view.read_material()
    claim = dict(output=str(out), plan_checksum=digest(plan))
    with docs.session():
        saved = docs.read("authorized-campaign-renewal")
        if saved is not None and saved != (claim,):
            raise ValueError("renewal_tip_already_claimed")
        if work is not None:
            out.mkdir(mode=0o700)
            (work / "result").rename(out / "alternative-view")
            work.rename(out / "alternative-preparation")
        write_once(out / "protocol.json", plan)
        if saved is None:
            docs.write("authorized-campaign-renewal", (claim,))
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        max_new_requests=request_limit,
        previous_requests=sum(p["requests"] for p in tip["previous_attempts"]),
        api_requests=0,
        status="already_recorded" if saved else "prepared",
    )

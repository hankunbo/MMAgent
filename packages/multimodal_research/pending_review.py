"""Compose the original partial acceptance reader with the existing campaign."""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.pending_review import read_pending_review
from packages.grounding.models import validate


def inventory(
    parent,
    tip,
    acknowledge_unknown_request=None,
    format_requests=(),
    *,
    initial_partition=False,
    historical_origin=None,
):
    if historical_origin is not None:
        if not acknowledge_unknown_request or format_requests or initial_partition:
            raise ValueError("historical_pending_explicit_unknown_only")
        from .historical_pending import inventory as historical_inventory

        return historical_inventory(parent, tip, acknowledge_unknown_request, historical_origin)
    if format_requests and acknowledge_unknown_request:
        raise ValueError("pending_review_separate_unknown_and_format")
    if len(set(format_requests)) != len(format_requests) or len(format_requests) > 4:
        raise ValueError("pending_review_format_ack_inventory")
    value = _inventory(
        parent,
        tip,
        acknowledge_unknown_request,
        inspect_format_failures=bool(format_requests),
        initial_partition=initial_partition,
    )
    if format_requests:
        value = select_known_formats(value, format_requests)
    if not value["selected_ids"]:
        raise ValueError("pending_review_no_safe_unstarted_work")
    return value


def select_known_formats(value, requested):
    """A precise new-run request; no UNKNOWN or already accepted item selection."""
    from copy import deepcopy

    value = deepcopy(value)
    # Composition may share row objects with nested historical proof records.
    # Authorization belongs only to this new selection, not those old proofs.
    value["quarantined_batches"] = [deepcopy(b) for b in value["quarantined_batches"]]
    found = set()
    selected = set(value["selected_ids"])
    for batch in value["quarantined_batches"]:
        proof = batch.get("known_format_failure")
        if not proof or proof["last_request_id"] not in requested:
            continue
        key = proof["last_request_id"]
        if key in found or set(batch["assessment_ids"]) & set(value["accepted_ids"]):
            raise ValueError("pending_review_format_duplicate_or_accepted")
        found.add(key)
        batch["new_reassessment_authorized"] = True
        batch["reason"] = "explicit_known_format_reassessment_not_original_replay"
        selected.update(batch["assessment_ids"])
    if found != set(requested):
        raise ValueError("pending_review_format_ack_mismatch")
    value.update(
        schema_version="pending-review-inventory/1.1.0",
        selected_ids=sorted(selected),
        acknowledged_format_requests=sorted(requested),
    )
    return value


def _inventory(
    parent,
    tip,
    acknowledge_unknown_request=None,
    *,
    inspect_format_failures=False,
    initial_partition=False,
):
    """Empty ancestor work is valid inventory, not an authorization to dispatch.

    Recheck every owner before composing the current attempt. Only the public
    boundary rejects a final empty selection; a failed child can still own work.
    """
    from .campaign import Phase

    parent.base.check()
    docs = parent.documents(tip["run_id"])
    parent.base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    phase = docs.read("dynamic-phase-result")
    if not phase or phase[0]["target"] is not None:
        raise ValueError("pending_review_accepted_phase_required")
    current, _ = Phase(parent.base, observation_reader=parent.accepted_observations).project(
        run_id=docs.run_id, config=parent.config
    )
    old = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if current != old:
        raise ValueError("pending_review_current_material_changed")
    if parent.protocol.get("pending_review"):
        # A completed increment may be followed by explicit treatment of an old
        # quarantined item. Never strand it by requiring a second claim on the
        # original run, and never rerun the completed increment's items.
        from agents.expert_agent.committed_review import read_committed_review
        from packages.evidence_details.acceptance_models import read_records
        from packages.grounding.models import digest

        if not phase[0]["accepted"]:
            from packages.evidence_details.pending_review import read_failed_review_attempt

            prior_inventory = parent.protocol["pending_review"]
            prior_ids = {
                r for b in prior_inventory["quarantined_batches"] for r in b["request_ids"]
            }
            earlier_ack = (
                acknowledge_unknown_request if acknowledge_unknown_request in prior_ids else None
            )
            prior = _inventory(
                parent.parent,
                parent.protocol["tip"],
                earlier_ack,
                inspect_format_failures=inspect_format_failures,
                initial_partition=parent.protocol.get("schema_version")
                == PARTITIONED_WINDOW_VERSION,
            )
            attempt = read_failed_review_attempt(
                documents=docs,
                state=current,
                config=parent.config,
                acknowledge_request=None if earlier_ack else acknowledge_unknown_request,
                **(dict(inspect_format_failures=True) if inspect_format_failures else {}),
            )
            selected = sorted(
                (set(prior["selected_ids"]) - set(parent.protocol["review_target_ids"]))
                | set(attempt["selected_ids"])
            )
            return dict(
                prior,
                selected_ids=selected,
                acknowledge_unknown_request=acknowledge_unknown_request,
                quarantined_batches=prior["quarantined_batches"] + attempt["quarantined_batches"],
                failed_increment=dict(
                    attempt=attempt, previous_inventory=prior, phase_checksum=digest(phase)
                ),
            )
        records = read_records(docs.read("accepted:" + phase[0]["acceptance_key"])[0])
        completed = {i.assessment_id for i in records.candidate.items}
        if completed != set(parent.protocol["review_target_ids"]):
            # A legitimately accepted partial child must not strand its untouched
            # batches. The same original owner quarantines every sent/rejected
            # batch; this is not permission to retry a malformed answer.
            attempt = read_pending_review(
                documents=docs,
                state=current,
                acceptance_key=phase[0]["acceptance_key"],
                acknowledge_unknown_request=acknowledge_unknown_request,
                allow_empty=True,
                **(dict(inspect_format_failures=True) if inspect_format_failures else {}),
            )
            # The partial-acceptance owner verifies the saved score/receipt,
            # exact parent and every batch/request before partitioning pending
            # work. A whole-stage transport proof would incorrectly reject its
            # quarantined UNKNOWN and strand unrelated, never-sent batches.
            # This inventory is not a whole-stage completion certificate.
            if set(attempt["accepted_ids"]) != completed or not completed.issubset(
                parent.protocol["review_target_ids"]
            ):
                raise ValueError("pending_review_partial_inventory_binding")
            try:
                # Preserve the exact proof shape of previously frozen partial
                # successors whose entire request journal was confirmed.
                proof = read_committed_review(docs, parent=current, config=parent.config)
            except ValueError as error:
                if str(error) != "committed_review_unknown_request":
                    raise
                proof = dict(
                    schema_version="pending-partial-owner-proof/1.0.0",
                    inventory_checksum=digest(attempt),
                    phase_checksum=digest(phase),
                    whole_stage_confirmed=False,
                )
            if proof is None:
                raise ValueError("pending_review_increment_not_confirmed")
            prior = _inventory(
                parent.parent,
                parent.protocol["tip"],
                inspect_format_failures=inspect_format_failures,
            )
            return dict(
                prior,
                selected_ids=sorted(
                    (set(prior["selected_ids"]) - set(parent.protocol["review_target_ids"]))
                    | set(attempt["selected_ids"])
                ),
                accepted_ids=sorted(set(prior["accepted_ids"]) | completed),
                quarantined_batches=prior["quarantined_batches"] + attempt["quarantined_batches"],
                acknowledge_unknown_request=acknowledge_unknown_request,
                partial_increment=dict(
                    attempt=attempt,
                    proof=proof,
                    records_checksum=digest(records),
                    previous_inventory=prior,
                ),
            )
        proof = read_committed_review(docs, parent=current, config=parent.config)
        if proof is None:
            raise ValueError("pending_review_increment_not_confirmed")
        prior = _inventory(
            parent.parent,
            parent.protocol["tip"],
            acknowledge_unknown_request,
            inspect_format_failures=inspect_format_failures,
            initial_partition=parent.protocol.get("schema_version") == PARTITIONED_WINDOW_VERSION,
        )
        remaining = sorted(set(prior["selected_ids"]) - completed)
        return dict(
            prior,
            selected_ids=remaining,
            accepted_ids=sorted(set(prior["accepted_ids"]) | completed),
            completed_increment=dict(
                proof=proof, records_checksum=digest(records), previous_inventory=prior
            ),
        )
    if not phase[0]["accepted"]:
        from packages.evidence_details.pending_review import read_failed_review_attempt
        from packages.grounding.models import digest

        if initial_partition:
            if not acknowledge_unknown_request or inspect_format_failures:
                raise ValueError("pending_initial_partition_explicit_unknown_required")
            attempt = read_failed_review_attempt(
                documents=docs,
                state=current,
                config=parent.config,
                acknowledge_request=acknowledge_unknown_request,
            )
            return retained_partition(parent, current, attempt, phase)
        if acknowledge_unknown_request or inspect_format_failures:
            raise ValueError("pending_initial_failure_unstarted_only")
        attempt = read_failed_review_attempt(
            documents=docs,
            state=current,
            config=parent.config,
        )
        return dict(
            schema_version="pending-review-inventory/1.2.0",
            accepted_ids=[],
            selected_ids=attempt["selected_ids"],
            quarantined_batches=attempt["quarantined_batches"],
            acknowledge_unknown_request=None,
            initial_failed_review=dict(attempt=attempt, phase_checksum=digest(phase)),
        )
    return read_pending_review(
        documents=docs,
        state=current,
        acceptance_key=phase[0]["acceptance_key"],
        acknowledge_unknown_request=acknowledge_unknown_request,
        allow_empty=True,
        **(dict(inspect_format_failures=True) if inspect_format_failures else {}),
    )


def retained_partition(parent, current, attempt, phase):
    """Exclude only currently compatible original accepted work, never caller flags."""
    from packages.grounding.models import digest

    reader = getattr(parent.base, "confirmed_review_reader", None)
    if (
        parent.protocol.get("mode") != "acquisition"
        or parent.protocol.get("pending_review")
        or not parent.protocol.get("reuse_assessment_ids")
        or reader is None
    ):
        raise ValueError("pending_initial_partition_original_owner_required")
    origin = reader.current(current)
    retained = set(origin.assessment_ids)
    batches = attempt["quarantined_batches"]
    authorized = [b for b in batches if b.get("new_reassessment_authorized")]
    raw = set(attempt["selected_ids"])
    selected = raw - retained
    if (
        not retained
        or retained != set(parent.protocol["reuse_assessment_ids"])
        or not retained <= raw
        or any(retained & set(b["assessment_ids"]) for b in batches)
        or len(authorized) != 1
        or selected != set(authorized[0]["assessment_ids"])
        or not selected
        or len(selected) > parent.config.review_batching.batch_size
        or phase[0].get("accepted") is not False
    ):
        raise ValueError("pending_initial_partition_binding")
    return dict(
        attempt,
        schema_version="pending-review-inventory/1.3.0",
        accepted_ids=sorted(retained),
        selected_ids=sorted(selected),
        acknowledge_unknown_request=attempt["acknowledge_request"],
        initial_failed_partition=dict(
            attempt=attempt,
            phase_checksum=digest(phase),
            origin_checksum=digest(origin),
            run_id=origin.run_id,
            acceptance_key=origin.acceptance_key,
            acceptance_checksum=origin.acceptance_checksum,
            retained_ids=sorted(retained),
        ),
    )


UNSTARTED_WINDOW_VERSION = "multimodal-campaign-renewal/1.39.0"
UNSTARTED_FOLLOWUP_VERSION = "multimodal-campaign-renewal/1.42.0"
RUNTIME_PENDING_VERSION = "multimodal-campaign-renewal/1.54.0"
REASSESSMENT_WINDOW_VERSION = "multimodal-campaign-renewal/1.55.0"
PARTITIONED_WINDOW_VERSION = "multimodal-campaign-renewal/1.57.0"


def projection_context(campaign, *, definition=False):
    """Bind a projection to its exact original context; no dispatch authority."""
    from db.runtime.repository.case_pipeline import identity

    target = "definition-expansion" if definition else None
    rid = identity("RUN", "task-research:" + campaign.request_key(target))
    context = campaign.contexts.read_research_run(rid)
    docs = campaign.documents(rid)
    if (
        context is None
        or context.run_id != rid
        or docs.run_id != rid
        or context.request.input_checksum != campaign.protocol["input"]
        or context.request.task.checksum != campaign.protocol["task"]
        or context.request.visible_checksum != campaign.base.visible_value.checksum
        or context.request.scope.case_code != campaign.base.binding["case_code"]
    ):
        raise ValueError("pending_partition_projection_started_or_changed")
    return docs, context


def partition_projection_unstarted(campaign, *, definition=False, budget_binding_only=False):
    """Original registered context, no Planner, phase, reservation or response."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    docs, context = projection_context(campaign, definition=definition)
    rid = docs.run_id
    journal = RuntimeAgentJournal(docs).read()
    if budget_binding_only:
        if not definition or tuple(journal) != (
            dict(
                schema_version="agent-budget/1.0.0",
                binding=dict(
                    run_id=rid,
                    deadline=campaign.protocol["deadline"],
                    max_requests=campaign.config.max_requests,
                ),
            ),
        ):
            raise ValueError("definition_projection_budget_binding")
        journal = []  # Exact budget initialization is not a request reservation.
    if (
        context is None
        or context.run_id != rid
        or docs.run_id != rid
        or context.request.input_checksum != campaign.protocol["input"]
        or context.request.task.checksum != campaign.protocol["task"]
        or context.request.visible_checksum != campaign.base.visible_value.checksum
        or context.request.scope.case_code != campaign.base.binding["case_code"]
        or campaign.sessions.read(rid) is not None
        or journal
        or any(
            docs.read(k)
            for k in (
                "dynamic-campaign",
                "dynamic-phase-input",
                "dynamic-phase-result",
                "live-stage-intent:review",
                "live-stage:review",
                "multimodal-case-result",
            )
        )
        or (campaign.phase_path(None) / "request.json").exists()
        or (
            definition
            and (
                any(docs.read(f"definition:operation:{n}") for n in range(100))
                or any(
                    docs.read(k)
                    for k in (
                        "definition:revision",
                        "definition:stage:sources",
                        "live-stage-intent:source",
                        "live-stage:source",
                    )
                )
            )
        )
    ):
        raise ValueError("pending_partition_projection_started_or_changed")
    return docs, context


def partition_projection_repair_current(campaign):
    """A tested same-window amendment permits the first dispatch, never a replay."""
    import json
    from pathlib import Path

    from packages.grounding.models import digest

    path = campaign.output / "prewire-repair.json"
    definition = campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.56.0"
    if (
        campaign.protocol.get("schema_version") != PARTITIONED_WINDOW_VERSION and not definition
    ) or not path.is_file():
        return False
    value = json.loads(path.read_bytes())
    if (
        value.get("schema_version")
        != ("renewal-prewire-repair/1.2.0" if definition else "renewal-prewire-repair/1.1.0")
        or value.get("reason")
        != (
            "definition_retained_context_projection"
            if definition
            else "partitioned_review_context_projection"
        )
        or value.get("protocol") != digest(campaign.protocol)
        or value.get("deadline") != campaign.protocol["deadline"]
        or value.get("original_failure_checksum")
        != digest(json.loads((campaign.output / "stage-closure-error.json").read_bytes()))
        or value.get("proof") != digest(json.loads(Path(value["proof_path"]).read_bytes()))
    ):
        raise ValueError("pending_partition_projection_amendment_binding")
    if definition:
        from .implementation import freeze

        implementation = value.get("implementation_after")
        if (campaign.output / "archive-repair.json").exists():
            from .archive_recovery import DEFINITION_PROJECTION_VERSION, read

            amendment = read(campaign)
            if amendment["schema_version"] != DEFINITION_PROJECTION_VERSION or amendment[
                "prior_projection_grant"
            ] != digest(value):
                raise ValueError("definition_projection_guard_chain")
            implementation = amendment["implementation_after"]
        # The persisted zero-wire grant is immutable. Later confirmed work is
        # restored by the original operation journal, not required to be empty.
        docs, context = projection_context(campaign, definition=True)
        if (
            campaign.protocol.get("mode") != "definitions"
            or not campaign.protocol.get("definition_increment_only")
            or value.get("context_checksum") != digest(context)
            or value.get("implementation_before") != campaign.protocol["implementation"]
            or implementation != freeze()
            or value.get("max_new_requests") != campaign.protocol["max_new_requests"]
            or value.get("original_journal") != digest([])
            or value.get("original_operations") != digest([])
        ):
            raise ValueError("definition_projection_amendment_binding")
    else:
        docs, _ = partition_projection_unstarted(campaign)
    if docs.run_id != value["run_id"] or docs.read("renewal:prewire-repair") != (value,):
        raise ValueError("pending_partition_projection_amendment_owner")
    return True


def authorize_partition_projection_repair(campaign, proof, authorization):
    """Append original-owner proof; retain failed CLI, key, budget and deadline."""
    import json
    from pathlib import Path

    from apps.vectorengine_single_case import write_once
    from packages.grounding.models import digest
    from packages.object_store import checksum

    from .campaign import Phase
    from .implementation import freeze

    with campaign.execution_session():
        definition = campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.56.0"
        if (
            not authorization.strip()
            or (
                campaign.protocol.get("schema_version") != PARTITIONED_WINDOW_VERSION
                and not definition
            )
            or (
                definition
                and (
                    campaign.protocol.get("mode") != "definitions"
                    or not campaign.protocol.get("definition_increment_only")
                    or campaign.parent.base.review_context_scope
                    != "current-owner-prior-visible/1.0.0"
                    or campaign.parent.base.confirmed_review_reader is None
                )
            )
            or (campaign.output / "cancel").exists()
        ):
            raise ValueError("pending_partition_projection_authorization")
        campaign.guard()
        failure = json.loads((campaign.output / "stage-closure-error.json").read_bytes())
        if failure != dict(
            schema_version="multimodal-local-closure-error/1.0.0",
            primary="local_owner_error",
            secondary="local_owner_error",
            stage="definitions" if definition else "review",
            durable_commit="unconfirmed",
            automatic_replay_authorized=False,
            complete=False,
        ):
            raise ValueError("pending_partition_projection_failure_scope")
        docs, context = partition_projection_unstarted(campaign, definition=definition)
        campaign.base.contexts[docs.run_id] = context
        state, _ = Phase(campaign.base, observation_reader=campaign.accepted_observations).project(
            run_id=docs.run_id, config=campaign.config
        )
        if sorted(i.id for i in state.policy.items) != (
            [] if definition else campaign.protocol["review_target_ids"]
        ):
            raise ValueError("pending_partition_projection_selection")
        before, after = campaign.protocol["implementation"], freeze()
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        allowed = {
            "packages/multimodal_research/campaign_renewal.py",
            "packages/multimodal_research/pending_review.py",
            "apps/multimodal_case.py" if definition else "packages/multimodal_research/campaign.py",
        }
        if before.keys() != after.keys() or changed != allowed:
            raise ValueError("pending_partition_projection_change_scope")
        proof_path = Path(proof).absolute()
        evidence = json.loads(proof_path.read_bytes())
        if (
            evidence.get("direct_exit") != 0
            or evidence.get("implementation_before") != before
            or evidence.get("implementation_after") != after
            or not evidence.get("commands")
            or any(
                row.get("direct_exit") != 0
                or row.get("log_checksum") != checksum(Path(row["log_path"]).read_bytes())
                for row in evidence["commands"]
            )
        ):
            raise ValueError("pending_partition_projection_tests_failed")
        value = dict(
            schema_version="renewal-prewire-repair/1.2.0"
            if definition
            else "renewal-prewire-repair/1.1.0",
            protocol=digest(campaign.protocol),
            run_id=docs.run_id,
            implementation_before=before,
            implementation_after=after,
            proof_path=str(proof_path),
            proof=digest(evidence),
            authorization=authorization,
            original_journal=digest([]),
            deadline=campaign.protocol["deadline"],
            requests=0,
            reason="definition_retained_context_projection"
            if definition
            else "partitioned_review_context_projection",
            original_failure_checksum=digest(failure),
            corrected_projection_checksum=digest(state),
        )
        if definition:
            value.update(
                context_checksum=digest(context),
                max_new_requests=campaign.protocol["max_new_requests"],
                original_operations=digest([]),
            )
        with docs.session():
            docs.write("renewal:prewire-repair", (value,))
        write_once(campaign.output / "prewire-repair.json", value)
        return dict(status="authorized_zero_wire_repair", requests=0, deadline=value["deadline"])


def current_unknown_proof(parent, pending):
    from packages.evidence_details.pending_review import unknown_response_proof

    return unknown_response_proof(
        parent.documents(pending["run_id"]),
        pending["acknowledge_unknown_request"],
        pending["journal_checksum"],
    )


def validate_reassessment_window(plan, parent):
    """One explicit new attempt of one batch; original UNKNOWN stays charged.

    The caller also compares inventory reconstructed by the original owner.
    This port checks expiry, exact selection, immutable request proof and bounds.
    """
    from datetime import datetime

    from packages.grounding.models import digest

    from .execution_lease import PROFILE

    pending = plan.get("pending_review") or {}
    ack = pending.get("acknowledge_unknown_request")
    selected = pending.get("selected_ids", [])
    authorized = [
        b for b in pending.get("quarantined_batches", []) if b.get("new_reassessment_authorized")
    ]
    start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
    from .historical_pending import VERSION as HISTORICAL_VERSION

    historical = plan.get("schema_version") == HISTORICAL_VERSION
    partitioned = plan.get("schema_version") == PARTITIONED_WINDOW_VERSION
    previous_end = datetime.fromisoformat(parent.protocol["deadline"])
    if partitioned:
        proof = pending.get("initial_failed_partition") or {}
        if (
            parent.protocol.get("mode") != "acquisition"
            or parent.protocol.get("pending_review")
            or not proof.get("origin_checksum")
            or proof.get("retained_ids") != pending.get("accepted_ids")
            or not proof.get("retained_ids")
            or (previous_end > start and end > previous_end)
        ):
            raise ValueError("pending_partitioned_window_binding")
    if historical:
        proof = pending.get("historical_unknown_origin") or {}
        if (
            not proof.get("current_acceptance")
            or not proof.get("material", {}).get("exact_item_observation_sources")
            or proof.get("current_config_checksum") != digest(parent.config)
            or not pending.get("accepted_ids")
        ):
            raise ValueError("historical_pending_window_binding")
    if (
        plan.get("schema_version")
        not in {REASSESSMENT_WINDOW_VERSION, PARTITIONED_WINDOW_VERSION, HISTORICAL_VERSION}
        or plan.get("pending_reassessment_window")
        != (
            "original_owner_ancestor_unknown_new_attempt/1.0.0"
            if historical
            else "original_owner_partitioned_unknown_new_attempt/1.0.0"
            if partitioned
            else "original_owner_unknown_batch_new_window/1.0.0"
        )
        or bool(pending.get("historical_unknown_origin")) != historical
        or bool(pending.get("initial_failed_partition")) != partitioned
        or plan.get("execution_lease_profile") != PROFILE
        or plan.get("mode") != "review"
        or plan.get("review_increment_only") is not True
        or not ack
        or len(authorized) != 1
        or authorized[0].get("request_ids") != [ack]
        or authorized[0].get("old_response_state_unchanged") is not True
        or selected != sorted(set(authorized[0]["assessment_ids"]))
        or not 1 <= len(selected) <= parent.config.review_batching.batch_size
        or plan.get("review_target_ids") != selected
        or set(selected) & set(pending.get("accepted_ids", []))
        or pending.get("acknowledged_format_requests")
        or plan.get("provider") != digest(parent.config)
        or plan.get("previous_window_deadline") != parent.protocol["deadline"]
        or (not partitioned and previous_end > start)
        or plan.get("whole_case_execution_profile")
        != getattr(parent.base, "execution_profile", None)
        or plan.get("whole_case_execution_profile")
        not in {"relay-context-v1", "relay-snapshot-context-v1", "relay-root-question-v1"}
        or not 300 < (end - start).total_seconds() <= 2700
        or plan.get("seconds") != (end - start).total_seconds()
        or parent.config.transport.max_retries != 0
        or plan.get("max_model_dispatches") != 1
        or plan.get("max_new_requests") != 2
        or plan.get("max_cumulative_requests") != 40
        or not plan.get("previous_attempts")
        or any(
            type(p["requests"]) is not int or p["requests"] < 0
            for p in plan.get("previous_attempts", [])
        )
        or sum(p["requests"] for p in plan["previous_attempts"]) + 2 > 40
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan.get("acquisition_targets", [])
        or plan.get("expansion_policy") is not None
        or plan.get("pending_new_window")
        or plan.get("runtime_pending_review")
        or plan.get("recovery_network_route")
        or plan.get("recovery_http_client")
        or plan.get("unknown_response_proof") != current_unknown_proof(parent, pending)
    ):
        raise ValueError("pending_reassessment_window_binding")
    ancestor = parent
    for _ in range(33):
        prior = ancestor.protocol.get("pending_review") or {}
        if prior.get("acknowledge_unknown_request") == ack:
            raise ValueError("pending_reassessment_window_already_used")
        ancestor = getattr(ancestor, "parent", None)
        if ancestor is None:
            return
    raise ValueError("pending_reassessment_window_history_depth")


def declare_reassessment_window(plan, parent):
    from datetime import datetime

    from .execution_lease import PROFILE
    from .historical_pending import VERSION as HISTORICAL_VERSION

    historical = bool(plan["pending_review"].get("historical_unknown_origin"))
    partitioned = bool(plan["pending_review"].get("initial_failed_partition"))
    if partitioned:
        start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
        previous_end = datetime.fromisoformat(parent.protocol["deadline"])
        if previous_end > start:
            end = min(end, previous_end)
            plan.update(deadline=end.isoformat(), seconds=(end - start).total_seconds())
    plan.update(
        schema_version=HISTORICAL_VERSION
        if historical
        else PARTITIONED_WINDOW_VERSION
        if partitioned
        else REASSESSMENT_WINDOW_VERSION,
        pending_reassessment_window=(
            "original_owner_ancestor_unknown_new_attempt/1.0.0"
            if historical
            else "original_owner_partitioned_unknown_new_attempt/1.0.0"
            if partitioned
            else "original_owner_unknown_batch_new_window/1.0.0"
        ),
        execution_lease_profile=PROFILE,
        previous_window_deadline=parent.protocol["deadline"],
        whole_case_execution_profile=getattr(parent.base, "execution_profile", None),
        unknown_response_proof=current_unknown_proof(parent, plan["pending_review"]),
    )
    validate_reassessment_window(plan, parent)


def validate_runtime_pending_window(plan, parent):
    """Explicit lease selection, never a new time window or UNKNOWN replay grant.

    The caller must additionally rebuild inventory through the original owner;
    these structural checks cannot make caller-supplied IDs authoritative.
    """
    from datetime import datetime

    from .campaign_renewal import validate_context_window
    from .execution_lease import PROFILE

    validate_context_window(plan, parent)
    pending = plan.get("pending_review") or {}
    selected = pending.get("selected_ids", [])
    quarantined = {
        item for batch in pending.get("quarantined_batches", []) for item in batch["assessment_ids"]
    }
    required = (len(selected) + parent.config.review_batching.batch_size - 1) // (
        parent.config.review_batching.batch_size
    )
    start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
    if (
        plan.get("schema_version") != RUNTIME_PENDING_VERSION
        or plan.get("runtime_pending_review") != "owner_verified_unstarted_same_window/1.0.0"
        or plan.get("execution_lease_profile") != PROFILE
        or plan.get("mode") != "review"
        or plan.get("review_increment_only") is not True
        or not selected
        or selected != sorted(set(selected))
        or plan.get("review_target_ids") != selected
        or set(selected) & (set(pending.get("accepted_ids", [])) | quarantined)
        or pending.get("acknowledge_unknown_request")
        or pending.get("acknowledged_format_requests")
        or any(b.get("new_reassessment_authorized") for b in pending.get("quarantined_batches", []))
        or plan.get("pending_new_window")
        or plan.get("initial_failed_review_window")
        or not 300 < (end - start).total_seconds() <= 2700
        or plan.get("seconds") != int((end - start).total_seconds())
        or parent.config.transport.max_retries != 0
        or plan.get("max_model_dispatches") != required
        or not required < plan.get("max_new_requests", 0) <= 8
        or plan.get("max_cumulative_requests") != 40
        or not plan.get("previous_attempts")
        or any(
            type(p["requests"]) is not int or p["requests"] < 0
            for p in plan.get("previous_attempts", [])
        )
        or sum(p["requests"] for p in plan.get("previous_attempts", [])) + plan["max_new_requests"]
        > 40
        or plan.get("acquisition_targets", [])
        or plan.get("expansion_policy") is not None
    ):
        raise ValueError("pending_runtime_window_binding")


def declare_runtime_pending_window(plan, parent):
    """Called only after ordinary pending inventory/current-window preparation."""
    from .execution_lease import PROFILE

    if plan.get("schema_version") != "multimodal-campaign-renewal/1.23.0":
        raise ValueError("pending_runtime_requires_current_context_window")
    plan.update(
        schema_version=RUNTIME_PENDING_VERSION,
        runtime_pending_review="owner_verified_unstarted_same_window/1.0.0",
        execution_lease_profile=PROFILE,
    )
    validate_runtime_pending_window(plan, parent)


def validate_unstarted_window(plan, parent):
    """One initial window, and one strict-progress unsent-only followup."""
    from datetime import datetime

    from packages.grounding.models import digest

    from .campaign_renewal import recovery_configuration

    start = datetime.fromisoformat(plan["started_at"])
    end = datetime.fromisoformat(plan["deadline"])
    pending = plan.get("pending_review") or {}
    selected = pending.get("selected_ids", [])
    quarantined = {
        item for batch in pending.get("quarantined_batches", []) for item in batch["assessment_ids"]
    }
    if (
        plan.get("schema_version") not in {UNSTARTED_WINDOW_VERSION, UNSTARTED_FOLLOWUP_VERSION}
        or plan.get("mode") != "review"
        or plan.get("pending_new_window") != "owner_verified_unstarted_only/1.0.0"
        or plan.get("review_increment_only") is not True
        or plan.get("review_target_ids") != selected
        or not 1 <= len(selected) <= 6
        or selected != sorted(set(selected))
        or set(selected) & (set(pending.get("accepted_ids", [])) | quarantined)
        or pending.get("acknowledge_unknown_request")
        or pending.get("acknowledged_format_requests")
        or any(b.get("new_reassessment_authorized") for b in pending.get("quarantined_batches", []))
        or plan.get("previous_window_deadline") != parent.protocol["deadline"]
        or datetime.fromisoformat(parent.protocol["deadline"]) > start
        or plan.get("whole_case_execution_profile")
        != getattr(parent.base, "execution_profile", None)
        or plan.get("whole_case_execution_profile")
        not in {"relay-context-v1", "relay-snapshot-context-v1", "relay-root-question-v1"}
        or plan["provider"] != digest(recovery_configuration(parent.config, plan))
        or not 300 < (end - start).total_seconds() <= 2700
        or plan["seconds"] != (end - start).total_seconds()
        or not 1 <= plan["max_model_dispatches"] < plan["max_new_requests"] <= 6
        or plan["max_cumulative_requests"] != 40
        or sum(p["requests"] for p in plan["previous_attempts"]) + plan["max_new_requests"] > 40
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan.get("acquisition_targets", [])
        or plan.get("expansion_policy") is not None
    ):
        raise ValueError("pending_unstarted_window_binding")
    followup = plan["schema_version"] == UNSTARTED_FOLLOWUP_VERSION
    if followup:
        previous = set((parent.protocol.get("pending_review") or {}).get("selected_ids", []))
        if (
            parent.protocol.get("schema_version") != UNSTARTED_WINDOW_VERSION
            or plan.get("pending_window_parent_checksum") != digest(parent.protocol)
            or not set(selected) < previous
            or not previous.intersection(pending.get("accepted_ids", []))
        ):
            raise ValueError("pending_unstarted_window_followup_scope")
    elif plan.get("pending_window_parent_checksum") is not None:
        raise ValueError("pending_unstarted_window_followup_version")
    ancestor = parent
    for index in range(33):
        version = ancestor.protocol.get("schema_version")
        if version == UNSTARTED_FOLLOWUP_VERSION or (
            version == UNSTARTED_WINDOW_VERSION and not (followup and index == 0)
        ):
            raise ValueError("pending_unstarted_window_already_used")
        ancestor = getattr(ancestor, "parent", None)
        if ancestor is None:
            return
    raise ValueError("pending_unstarted_window_history_depth")


def validate_initial_failed_window(plan, parent):
    """Recheck original unsent proof for one bounded initial-failure successor."""
    from datetime import datetime

    from packages.grounding.models import digest

    pending = inventory(parent, plan["tip"])
    proof = pending.get("initial_failed_review") or {}
    selected = pending["selected_ids"]
    batches = (len(selected) + parent.config.review_batching.batch_size - 1) // (
        parent.config.review_batching.batch_size
    )
    required = batches * (parent.config.transport.max_retries + 1)
    start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
    if (
        parent.protocol.get("pending_review")
        or plan.get("schema_version") != "multimodal-campaign-renewal/1.41.0"
        or plan.get("initial_failed_review_window") != "unstarted_only/1.0.0"
        or plan.get("mode") != "review"
        or plan.get("review_increment_only") is not True
        or plan.get("pending_review") != pending
        or pending["accepted_ids"]
        or not proof
        or proof["attempt"]["run_id"] != plan["tip"]["run_id"]
        or plan.get("review_target_ids") != selected
        or plan.get("provider") != digest(parent.config)
        or plan.get("previous_window_deadline") != parent.protocol["deadline"]
        or datetime.fromisoformat(parent.protocol["deadline"]) > start
        or not 300 < (end - start).total_seconds() <= 2700
        or plan.get("seconds") != (end - start).total_seconds()
        or not required <= plan["max_model_dispatches"] < plan["max_new_requests"] <= 10
        or plan.get("max_cumulative_requests") != 40
        or any(
            type(p["requests"]) is not int or p["requests"] < 0 for p in plan["previous_attempts"]
        )
        or sum(p["requests"] for p in plan["previous_attempts"]) + plan["max_new_requests"] > 40
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan.get("acquisition_targets", [])
        or plan.get("expansion_policy") is not None
    ):
        raise ValueError("initial_failed_window_binding")

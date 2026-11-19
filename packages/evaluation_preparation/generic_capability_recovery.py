"""Original-owner inventory of an interrupted capability increment.

No window is renewed here. Registered-but-unsent work, confirmed response without
acceptance, UNKNOWN and accepted facts are distinct; a response is never a score.
"""

from datetime import UTC, datetime
from pathlib import Path

from packages.grounding.models import digest, validate
from packages.llm_provider.models import read_request_record
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .generic_definition_successor import read
from .generic_input import directory


def software_chain(reference):
    """Read an explicit bounded history transition; never replace saved references."""
    path = Path(reference["path"]).resolve()
    if (
        not path.is_relative_to(Path("docs/acceptance").resolve())
        or checksum(path.read_bytes()) != reference["checksum"]
    ):
        raise ValueError("generic_capability_history_proof_reference")
    value = read(path)
    if value.get("schema_version") != "generic-capability-history-chain/1.0.0":
        return None
    windows = value.get("windows")
    if (
        value.get("scope") != "expired_history_read_only_not_dispatch"
        or not isinstance(windows, dict)
        or not 1 <= len(windows) <= 4
    ):
        raise ValueError("generic_capability_history_chain_scope")
    for key, entry in windows.items():
        directory(key)
        if set(entry) != {"selection_checksum", "software_reference"}:
            raise ValueError("generic_capability_history_chain_entry")
    return windows


def historical_protocol(campaign, proof_reference, *, now=None):
    """Explicit verified software transition for READ-ONLY expired history.

    All medical/input/selection/budget fields remain exact. This cannot renew
    dispatch permission; the original frozen protocol is kept byte for byte.
    """
    reference = proof_reference
    path = Path(reference["path"]).resolve()
    if (
        not path.is_relative_to(Path("docs/acceptance").resolve())
        or checksum(path.read_bytes()) != reference["checksum"]
    ):
        raise ValueError("generic_capability_history_proof_reference")
    proof = read(path)
    if (
        proof.get("schema_version")
        not in {
            "generic-capability-history-software/1.0.0",
            "generic-capability-history-software/1.1.0",
        }
        or proof.get("scope") != "expired_history_read_only_not_dispatch"
        or proof.get("passed") is not True
        or not proof.get("commands")
    ):
        raise ValueError("generic_capability_history_proof_scope")
    for command in proof["commands"]:
        folder = Path(command).resolve()
        if not folder.is_relative_to(Path("docs/acceptance").resolve()):
            raise ValueError("generic_capability_history_command_scope")
        result = read(folder / "command.json")
        if (
            result["exit_code"] != 0
            or (folder / "command.exit").read_text().strip() != "0"
            or checksum((folder / "command.log").read_bytes()) != result["log_checksum"]
        ):
            raise ValueError("generic_capability_history_unverified_change")
    frozen = read(campaign.output / "protocol.json")
    current = campaign.protocol_fields()
    if (now or datetime.now(UTC)) < datetime.fromisoformat(frozen["deadline"]):
        raise ValueError("generic_capability_history_requires_expired_window")
    historical_fields = {
        k: v for k, v in frozen.items() if k not in {"started_at", "deadline", "implementation"}
    }
    current_fields = {k: v for k, v in current.items() if k != "implementation"}
    capability_files = {}
    if proof["schema_version"] == "generic-capability-history-software/1.1.0":
        transition = proof.get("capability_implementation", {})
        old_observations, new_observations = (
            historical_fields.get("observations"),
            current_fields.get("observations"),
        )
        if (
            not isinstance(old_observations, dict)
            or not isinstance(new_observations, dict)
            or not isinstance(transition.get("before"), dict)
            or not isinstance(transition.get("after"), dict)
            or not transition["before"]
            or not transition["after"]
            or old_observations.get("capability_implementation") != transition["before"]
            or new_observations.get("capability_implementation") != transition["after"]
        ):
            raise ValueError("generic_capability_history_nested_implementation")
        capability_files = transition["after"]
        historical_fields["observations"] = {
            k: v for k, v in old_observations.items() if k != "capability_implementation"
        }
        current_fields["observations"] = {
            k: v for k, v in new_observations.items() if k != "capability_implementation"
        }
    if (
        historical_fields != current_fields
        or frozen["implementation"] != proof["before"]
        or current["implementation"] != proof["after"]
    ):
        raise ValueError("generic_capability_history_semantic_or_implementation_changed")
    covered = proof.get("files", {})
    source_readers = {"apps/vectorengine_round_materials.py", "apps/active_evidence_materials.py"}
    if not set(proof["after"]) | set(capability_files) | source_readers <= set(covered):
        raise ValueError("generic_capability_history_uncovered_dependency")
    for filename, expected in covered.items():
        source = Path(filename)
        # These are read dependencies in the original capability freeze, not
        # permission to execute a migration or to load arbitrary local files.
        capability_resource = filename in capability_files and any(
            filename.startswith(prefix) and source.suffix in suffixes
            for prefix, suffixes in (
                ("docs/sources/", {".xml", ".pdf", ".md", ".json", ".txt"}),
                ("apps/templates/", {".js", ".css", ".html"}),
                ("db/", {".py", ".sql"}),
            )
        )
        if (
            source.is_absolute()
            or not source.resolve().is_relative_to(Path.cwd())
            or any(part.startswith(".") for part in source.parts)
            or (source.suffix not in {".py", ".json", ".txt"} and not capability_resource)
            or (
                not filename.startswith(("apps/", "packages/", "configs/", "agents/"))
                and filename not in frozen["implementation"]
                and not capability_resource
            )
            or checksum(source.read_bytes()) != expected
        ):
            raise ValueError("generic_capability_history_dependency_changed")
    if any(covered[p] != h for p, h in (*proof["after"].items(), *capability_files.items())):
        raise ValueError("generic_capability_history_inconsistent_freeze")
    campaign.protocol = frozen
    campaign._capability_history_only = True
    return frozen


def response_inventory(documents, events):
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    records = [read_request_record(e["record"]) for e in events if e.get("phase") == "completed"]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    if (
        len(reserved) != len(set(reserved))
        or len({r.request_id for r in records}) != len(records)
        or any(r.request_id not in reserved or r.simulated for r in records)
        or len({e["request_id"] for e in confirmations}) != len(confirmations)
        or any(e["request_id"] not in reserved for e in confirmations)
    ):
        raise ValueError("generic_capability_recovery_request_accounting")
    completed = {r.request_id: r for r in records}
    confirmed = {e["request_id"]: e["confirmed"] for e in confirmations}
    rows = []
    for key in reserved:
        record = completed.get(key)
        raw = documents.read("live-response:" + key)
        status = "unknown"
        raw_checksum = None
        if record is not None and confirmed.get(key) is True:
            if (
                not raw
                or len(raw) != 1
                or raw[0].get("request_id") != key
                or raw[0].get("schema_version") != "bounded-live-response/1.0.0"
                or raw[0].get("untrusted_model_output") is not True
                or raw[0].get("accepted_medical_evidence") is not False
            ):
                raise ValueError("generic_capability_recovery_confirmed_response_missing")
            body = raw[0]["raw_model_text"].encode()
            if (
                checksum(body) != raw[0]["raw_checksum"]
                or confirmed_output_checksum(record, body, events) != record.output_checksum
                or raw[0].get("validation_error") != record.error
            ):
                raise ValueError("generic_capability_recovery_response_binding")
            status, raw_checksum = "confirmed_response", raw[0]["raw_checksum"]
        elif record is not None and confirmed.get(key) is False and record.api_requests == 0:
            if raw or record.output_checksum is not None:
                raise ValueError("generic_capability_recovery_unsent_response_conflict")
            status = "confirmed_not_sent"
        rows.append(
            dict(
                request_id=key,
                status=status,
                record_checksum=digest(record) if record else None,
                response_checksum=raw_checksum,
                error=record.error if record else None,
                replay_authorized=False,
            )
        )
    return rows


def confirmed_output_checksum(record, body, events):
    """Mirror the original Provider's raw-error versus parsed-success digest."""
    if record.error is not None:
        return digest(body.hex())
    from packages.llm_provider.models import PartialVisionProposal, VisionProposal
    from packages.llm_provider.provider import strict_json
    from packages.llm_provider.vision_content import decode

    if record.role != "vision":
        raise ValueError("generic_capability_recovery_nonvision_response")
    proofs = [
        e
        for e in events
        if e.get("schema_version") == "vision-content-binding/1.0.0"
        and e.get("request_id") == record.request_id
    ]
    if proofs:
        if len(proofs) != 1 or proofs[0]["raw_checksum"] != checksum(body):
            raise ValueError("generic_capability_recovery_vision_binding")
        candidate, proof = decode(body, proofs[0]["binding"], proofs[0]["binding_code"])
        if any(proofs[0].get(k) != v for k, v in proof.items()):
            raise ValueError("generic_capability_recovery_vision_binding")
    else:
        raw = strict_json(body)
        candidate = validate(
            PartialVisionProposal
            if record.prompt_version.startswith("vision-partial-")
            else VisionProposal,
            raw,
        )
    if candidate.scope_checksum != digest(record.scope):
        raise ValueError("generic_capability_recovery_response_scope")
    return digest(candidate)


def phase_inventory(campaign, target):
    path = campaign.phase_path(target)
    try:
        request = campaign.requests.for_key(campaign.request_key(target))
    except ValueError as exc:
        if str(exc) != "research_registration_pending":
            raise
        if (path / "request.json").exists() or (path / "result.json").exists():
            raise ValueError("generic_capability_recovery_unregistered_local_phase") from exc
        return dict(target=target, status="not_registered", reservations=0, requests=[])
    run = request.lock.run_id
    context = campaign.contexts.read_research_run(run, missing_ok=True)
    control = campaign.sessions.read(run)
    if context is None:
        if (
            control is not None
            or (path / "request.json").exists()
            or (path / "result.json").exists()
        ):
            raise ValueError("generic_capability_recovery_missing_context")
        return dict(
            target=target,
            run_id=run,
            request_checksum=digest(request),
            status="registered_without_context",
            reservations=0,
            requests=[],
        )
    if context.request != request or request.scope.case_code != campaign.base.binding["case_code"]:
        raise ValueError("generic_capability_recovery_context_binding")
    if (path / "request.json").is_file() and read(path / "request.json") != request.model_dump(
        mode="json"
    ):
        raise ValueError("generic_capability_recovery_local_request_binding")
    docs = campaign.documents(run)
    events = RuntimeAgentJournal(docs).read()
    phase = docs.read("dynamic-phase-result")
    if phase is not None and len(phase) != 1:
        raise ValueError("generic_capability_recovery_phase_shape")
    result = phase[0] if phase else None
    if result and (result["run_id"] != run or result.get("target") != target):
        raise ValueError("generic_capability_recovery_phase_binding")
    if (path / "result.json").is_file() and read(path / "result.json") != result:
        raise ValueError("generic_capability_recovery_local_result_binding")
    requests = response_inventory(docs, events)
    if control is None:
        if events or result:
            raise ValueError("generic_capability_recovery_control_missing")
        status = "registered_not_dispatched"
    else:
        if control.phase != "stopped" or control.pending is not None or control.cancelled:
            raise ValueError("generic_capability_recovery_not_terminal")
        if result and result.get("accepted"):
            status = "accepted"
        elif any(r["status"] == "unknown" for r in requests):
            status = "unknown_requires_explicit_resolution"
        elif any(r["status"] == "confirmed_response" for r in requests):
            status = "confirmed_response_requires_local_acceptance_check"
        else:
            status = "stopped_without_confirmed_response"
    return dict(
        target=target,
        run_id=run,
        request_checksum=digest(request),
        context_checksum=digest(context),
        control_checksum=digest(control) if control else None,
        journal_checksum=digest(events),
        phase_checksum=digest(result) if result else None,
        accepted=bool(result and result.get("accepted")),
        acceptance_key=result.get("acceptance_key") if result else None,
        error=result.get("error") if result else None,
        status=status,
        reservations=len(requests),
        requests=requests,
    )


def snapshot(campaign, *, now=None):
    """Current source selection plus exact registration, Control and request journals."""
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .generic_capability_projection import _current_selection_reads

    # Only the read-only inventory is scoped: no dispatch/persistence crosses it.
    with historical_fact_read_scope(), _current_selection_reads():
        return _snapshot(campaign, now=now)


def _snapshot(campaign, *, now=None):
    from .generic_capability_successor import VERSIONS

    key = campaign.review_key
    root = directory(key)
    if (root / "cancel-request.json").exists():
        raise ValueError("generic_capability_recovery_cancelled")
    if (now or datetime.now(UTC)) < datetime.fromisoformat(campaign.protocol["deadline"]):
        raise ValueError("generic_capability_recovery_window_not_expired")
    current, _ = campaign.selection_owner.current()
    history = current["original_history"]
    if (
        campaign.selection["schema_version"] not in VERSIONS
        or history != campaign.selection["original_history"]
        or history != campaign.checked.history
    ):
        raise ValueError("generic_capability_recovery_original_history_binding")
    rows = [phase_inventory(campaign, target) for target in (*campaign.visual_targets, None)]
    total = sum(row["reservations"] for row in rows)
    if campaign.usage() != total:
        raise ValueError("generic_capability_recovery_local_accounting_incomplete")
    observations = (
        campaign.accepted_observations()
    )  # Original acceptance and current source checks.
    for row in rows:
        if row.get("accepted") and row["target"] is not None and row["target"] not in observations:
            raise ValueError("generic_capability_recovery_accepted_observation_missing")
    inherited = campaign.selection.get("continuation")
    acquired = [row["target"] for row in rows if row.get("accepted") and row["target"] is not None]
    prior = history["reservations"]
    if inherited is not None:
        prior = campaign.prior()["reservations"]  # Fresh original-owner ancestor accounting.
        earlier = inherited["inventory"]["acquired_targets"]
        if set(earlier) & set(acquired) or not set(earlier) <= set(observations):
            raise ValueError("generic_capability_recovery_inherited_observation_binding")
        acquired = list(earlier) + acquired
    return dict(
        schema_version=(
            "generic-capability-recovery-inventory/1.1.0"
            if inherited is not None
            else "generic-capability-recovery-inventory/1.0.0"
        ),
        key=key,
        parent_key=campaign.spec["key"],
        selection_checksum=digest(campaign.selection),
        protocol_checksum=digest(campaign.protocol),
        observations_checksum=digest(observations),
        acquired_targets=acquired,
        phases=rows,
        new_requests=total,
        cumulative_requests=prior + total,
        dispatch_authorized=False,
        old_window_renewed=False,
    )


def rejected_deadline(campaign, row):
    """A confirmed answer is a new-attempt candidate only after exact domain rejection.

    No previous accepted fact or ambiguous delivery is permitted on this path.
    The returned evidence binds the refusal, not a fabricated medical acceptance.
    """
    docs = campaign.documents(row["run_id"])
    intent = docs.read("live-stage-intent:vision")
    if not intent or len(intent) != 1 or intent[0].get("role") != "vision":
        raise ValueError("generic_capability_recovery_missing_vision_intent")
    event = intent[0]["event"]["code"]
    key = event + ":domain"
    stage = docs.read("live-stage:vision")
    result = docs.read("review:" + event + ":result")
    submitted = docs.read("submission:" + key)
    accepted = docs.read("accepted:" + key)
    if (
        accepted is not None
        or not stage
        or len(stage) != 1
        or result != stage
        or stage[0].get("event") != intent[0]["event"]
        or stage[0].get("run_id") != row["run_id"]
        or stage[0].get("reason") != "deadline"
        or stage[0].get("domain_status") != "not_accepted"
        or stage[0].get("acceptance_key") is not None
        or not submitted
        or len(submitted) != 1
        or submitted[0].get("status") != "rejected"
        or submitted[0].get("receipt", {}).get("status") != "rejected"
        or row.get("accepted")
        or row.get("error") != "deadline"
        or len(row["requests"]) != 1
        or row["requests"][0]["status"] != "confirmed_response"
    ):
        raise ValueError("generic_capability_recovery_explicit_rejection_required")
    receipt = submitted[0]["receipt"]
    if receipt.get("submission_checksum") != submitted[0]["submission_checksum"]:
        raise ValueError("generic_capability_recovery_submission_receipt_binding")
    return dict(
        target=row["target"],
        run_id=row["run_id"],
        request_id=row["requests"][0]["request_id"],
        intent_checksum=digest(intent),
        stage_checksum=digest(stage),
        submission_checksum=digest(submitted),
        previous_acceptance_absent=True,
        reason="confirmed_response_domain_deadline_rejected_new_attempt",
        previous_answer_relabelled=False,
    )


def continuation_origin(
    key, software_reference, acknowledged_requests=(), *, history_chain=None, _seen=()
):
    """Build a typed successor from original owners; a saved inventory is never authority."""
    from types import SimpleNamespace

    from .generic_capability_execution import load_selection
    from .generic_capability_successor import CONTINUATION_VERSIONS

    options = dict(history_software=history_chain or software_reference)
    if _seen:
        options["_seen"] = _seen
    previous = load_selection(key, **options)
    inventory = snapshot(previous)
    continued = previous.selection["schema_version"] in CONTINUATION_VERSIONS
    if continued:
        return review_only_origin(previous, inventory, software_reference, acknowledged_requests)
    # This port handles the observed pre-Review cutoff. Partial Review has its
    # own receipt owner and cannot silently be treated as a new full batch.
    if inventory["phases"][-1]["status"] != "not_registered":
        raise ValueError("generic_capability_continuation_review_recovery_required")
    pending, rejected = [], []
    for row in inventory["phases"][:-1]:
        if row.get("accepted"):
            continue
        if row["status"] == "confirmed_response_requires_local_acceptance_check":
            rejected.append(rejected_deadline(previous, row))
        elif row["status"] not in {
            "not_registered",
            "registered_without_context",
            "registered_not_dispatched",
        }:
            raise ValueError("generic_capability_continuation_unresolved_request")
        pending.append(row["target"])
    acknowledgments = tuple(acknowledged_requests)
    required = {r["request_id"] for r in rejected}
    if len(acknowledgments) != len(set(acknowledgments)) or set(acknowledgments) != required:
        raise ValueError("generic_capability_rejected_request_acknowledgment")
    prior = inventory["cumulative_requests"]
    total_items = len(pending) + len(inventory["acquired_targets"])
    batch = previous.config.review_batching.batch_size
    review = (total_items + batch - 1) // batch
    cap = min(12, 40 - prior)
    if not total_items or cap <= 0 or len(pending) + review > cap:
        raise ValueError("generic_capability_continuation_budget")
    checked = SimpleNamespace(**vars(previous.checked))
    checked.continuation = dict(
        schema_version="generic-capability-continuation-binding/1.0.0",
        predecessor_key=key,
        root_claim=previous.selection,
        software_reference=software_reference,
        inventory=inventory,
        inventory_checksum=digest(inventory),
        reuse_targets=inventory["acquired_targets"],
        dispatch_targets=pending,
        rejected_attempts=rejected,
        acknowledged_requests=sorted(acknowledgments),
        old_window_renewed=False,
    )
    checked.predecessor = previous
    checked.limits = dict(
        max_new_requests=cap,
        max_cumulative_requests=40,
        expected_vision_requests=len(pending),
        minimum_review_requests=review,
        seconds=5400,
        reserve_review_seconds=max(720, review * 240),
        reserve_close_seconds=300,
    )
    return checked


def review_only_origin(previous, inventory, software_reference, acknowledged_requests):
    """A terminal, proven zero-request Review may receive a NEW bounded window.

    No claimed/UNKNOWN/confirmed/accepted Review is treated as unsent. Acquisition
    must already be complete; accepted observations retain their original owners.
    """
    from types import SimpleNamespace

    review = inventory["phases"][-1]
    acquisitions = inventory["phases"][:-1]
    if (
        acknowledged_requests
        or review.get("target") is not None
        or review.get("status") != "stopped_without_confirmed_response"
        or review.get("accepted")
        or review.get("acceptance_key") is not None
        or review.get("reservations") != 0
        or review.get("requests") != []
        or any(not row.get("accepted") for row in acquisitions)
    ):
        raise ValueError("generic_capability_review_only_requires_terminal_unsent")
    targets = inventory["acquired_targets"]
    expected = previous.checked.projection["dispatch_targets"]
    if not targets or len(targets) != len(set(targets)) or set(targets) != set(expected):
        raise ValueError("generic_capability_review_only_observation_coverage")
    prior = inventory["cumulative_requests"]
    batch = previous.config.review_batching.batch_size
    minimum = (len(targets) + batch - 1) // batch
    cap = min(10, 40 - prior)
    if type(prior) is not int or prior < 0 or cap < minimum:
        raise ValueError("generic_capability_continuation_budget")
    checked = SimpleNamespace(**vars(previous.checked))
    checked.continuation = dict(
        schema_version="generic-capability-continuation-binding/1.1.0",
        purpose="review_only_after_terminal_zero_request_review",
        predecessor_key=previous.review_key,
        root_claim=previous.root_claim(),
        software_reference=software_reference,
        inventory=inventory,
        inventory_checksum=digest(inventory),
        reuse_targets=targets,
        dispatch_targets=[],
        rejected_attempts=[],
        acknowledged_requests=[],
        old_window_renewed=False,
    )
    checked.predecessor = previous
    checked.limits = dict(
        max_new_requests=cap,
        max_cumulative_requests=40,
        expected_vision_requests=0,
        minimum_review_requests=minimum,
        seconds=5400,
        reserve_review_seconds=max(720, minimum * 240),
        reserve_close_seconds=300,
    )
    return checked

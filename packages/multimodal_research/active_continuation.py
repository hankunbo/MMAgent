"""one bounded continuation from closed active facts.

No old run or deadline is resumed. Attempted (including unknown) targets are
excluded from new acquisitions; only unattempted targets may be dispatched.
"""

import fcntl
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest, validate
from packages.multimodal_research.confirmed_observations import ConfirmedObservations
from packages.research_runtime.agent_records import RuntimeAgentJournal


def review_reserve_policy(value):
    from typing import Literal

    from pydantic import Field

    from packages.grounding.models import Value

    class Policy(Value):
        schema_version: Literal["review-work-reserve/1.0.0"]
        local_seconds_per_batch: int = Field(ge=60, le=180)
        close_seconds: int = Field(ge=300, le=600)

    return validate(Policy, value).model_dump(mode="json")


def reviewable_ids(observations, *, excluded=()):
    """Match the obtained-only Review projection, preserving legal zero values."""
    return sorted(
        k for k, v in observations.items() if v.status == "obtained" and k not in excluded
    )


def reserve_current_review(
    *,
    seconds_left,
    assessment_ids,
    batch_size,
    timeout_seconds,
    max_retries,
    requests_used,
    request_limit,
    policy,
    local_request_headroom=1,
):
    """Explicit opt-in estimate using actual transport bounds and pending work.

    Include one potential new obtained observation and the next acquisition.
    Local allowance is an engineering estimate, not a guaranteed latency bound.
    The absolute deadline/cancel/current-source checks remain authoritative.
    """
    policy = review_reserve_policy(policy)
    if (
        len(set(assessment_ids)) != len(assessment_ids)
        or batch_size < 1
        or timeout_seconds <= 0
        or max_retries < 0
        or type(local_request_headroom) is not int
        or local_request_headroom not in (0, 1)
    ):
        raise ValueError("review_reserve_invalid_inventory")
    batches = math.ceil((len(assessment_ids) + 1) / batch_size)
    attempts = 1 + max_retries
    per_batch = timeout_seconds * attempts + policy["local_seconds_per_batch"]
    reserve_seconds = policy["close_seconds"] + batches * per_batch
    reserve_requests = batches * attempts + local_request_headroom
    result = dict(
        schema_version=policy["schema_version"],
        stop_acquisition=(
            seconds_left < reserve_seconds + per_batch
            or requests_used + attempts + reserve_requests > request_limit
        ),
        reason="budget_reserved_for_pending_review",
        pending_assessment_ids=sorted(assessment_ids),
        projected_review_batches=batches,
        review_reserve_seconds=reserve_seconds,
        review_reserve_requests=reserve_requests,
        next_acquisition_seconds=per_batch,
        seconds_left=seconds_left,
        acquired_at_decision=len(assessment_ids),
        transport_timeout_seconds=timeout_seconds,
        local_seconds_per_batch=policy["local_seconds_per_batch"],
    )
    if not local_request_headroom:
        result["request_budget_semantics"] = "dispatch_only_local_completion_zero_request/1.0"
    return result


def reuse_dependencies(base, path, profile):
    """Reuse only owner-checked native artifacts and exact public search receipts."""
    from apps.vectorengine_single_case import write_once
    from packages.multimodal_research.native_fundus import recheck

    path = Path(path).absolute()
    load_parent(
        path,
        checksum=base.manifest.checksum,
        intent=base.task.model_dump(mode="json"),
        profile_checksum=digest(profile),
    )
    discovery = json.loads((path / "active-discovery.json").read_bytes())
    native = json.loads((path / "native-fundus/accepted.json").read_bytes())
    recheck(base, native)

    def discoverer(directory, selected):
        if selected != profile:
            raise ValueError("active_continuation_discovery_profile")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if [q["topic"] for q in discovery["queries"]] != profile["search_topics"]:
            raise ValueError("active_continuation_discovery_inventory")
        for query in discovery["queries"]:
            original = json.loads(
                (path / "knowledge-discovery" / (query["topic"] + ".json")).read_bytes()
            )
            if digest(original) != query["checksum"] or original != query["result"]:
                raise ValueError("active_continuation_discovery_receipt")
            write_once(directory / (query["topic"] + ".json"), original)
        return discovery

    def native_runner(current, requirement, *, before_dispatch):
        if (
            requirement["id"] != native["plan"]["target"]
            or requirement["component"] != native["plan"]["component"]
        ):
            raise ValueError("active_continuation_native_target")
        recheck(current, native)
        write_once(
            base.output / "native-reuse.json",
            dict(
                source=str(path),
                native_checksum=digest(native),
                new_model_runs=0,
            ),
        )
        return native

    return discoverer, native_runner


def ordered_targets(ledger, targets, observations):
    """Avoid starving an unobserved modality behind a long single-modality list."""
    modality = {row["id"]: row["input_modalities"][0] for row in ledger}
    counts = {}
    for target in observations:
        if target in modality:
            key = modality[target]
            counts[key] = counts.get(key, 0) + 1
    remaining, result = list(targets), []
    while remaining:
        chosen = min(remaining, key=lambda target: counts.get(modality[target], 0))
        remaining.remove(chosen)
        result.append(chosen)
        key = modality[chosen]
        counts[key] = counts.get(key, 0) + 1
    return tuple(result)


def select_acquisition_targets(eligible, selected, *, owner_methods=False):
    """Explicit finite work slice; the caller's original owner supplies eligibility."""
    if (
        not 1 <= len(selected) <= 3
        or len(set(selected)) != len(selected)
        or len(set(eligible)) != len(eligible)
        or not set(selected) <= set(eligible)
    ):
        raise ValueError("acquisition_selection_scope")
    value = dict(
        schema_version="bounded-acquisition-selection/1.1.0"
        if owner_methods
        else "bounded-acquisition-selection/1.0.0",
        eligible=list(eligible),
        selected=[target for target in eligible if target in selected],
        deferred=[target for target in eligible if target not in selected],
    )
    if owner_methods:
        value["eligibility_owner"] = "MultimodalCampaign.available_targets"
    return value


def reserve_review(
    *,
    seconds_left,
    acquired,
    batch_size,
    requests_used,
    request_limit=40,
    minimum_requests=0,
    local_request_headroom=1,
):
    """Conservative finite allowance; no medical effect or enlarged API quota."""
    batches = math.ceil((acquired + 1) / batch_size)
    reserve_seconds = 300 + batches * (240 + 180)
    if type(local_request_headroom) is not int or local_request_headroom not in (0, 1):
        raise ValueError("review_reserve_local_request_headroom")
    reserve_requests = max(minimum_requests, batches + local_request_headroom)
    result = dict(
        stop_acquisition=seconds_left < reserve_seconds + 420
        or requests_used + 1 + reserve_requests > request_limit,
        reason="budget_reserved_for_review",
        seconds_left=seconds_left,
        review_reserve_seconds=reserve_seconds,
        review_reserve_requests=reserve_requests,
        acquired_at_decision=acquired,
    )
    if not local_request_headroom:
        result["request_budget_semantics"] = "dispatch_only_local_completion_zero_request/1.0"
    return result


def load_parent(path, *, checksum, intent, profile_checksum):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("active_continuation_symlink")
    spec = json.loads((path / "entry.json").read_bytes())
    protocol = json.loads((path / "protocol.json").read_bytes())
    if (
        spec.get("schema_version")
        not in {"multimodal-live-entry/1.8.0", "multimodal-live-entry/1.9.0"}
        or spec.get("active_continuation_of")
        or any(spec.get(k) for k in ("previous", "continuation_of", "review_from", "known_only"))
        or spec.get("input_checksum") != checksum
        or spec.get("intent") != intent
        or spec.get("active_acquisition", {}).get("checksum") != profile_checksum
        or protocol.get("input") != checksum
        or protocol.get("task") != intent["checksum"]
        or protocol.get("implementation") != spec.get("implementation")
        or protocol.get("provider") != digest(spec["provider"])
        or (path / "cancel").exists()
    ):
        raise ValueError("active_continuation_parent_binding")
    if datetime.now(UTC) < datetime.fromisoformat(protocol["deadline"]):
        raise ValueError("active_continuation_parent_not_expired")
    with (path / "lock").open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("active_continuation_parent_running") from error
    if (path / "result.json").exists():
        result = json.loads((path / "result.json").read_bytes())
        if result.get("complete"):
            raise ValueError("active_continuation_parent_already_complete")
    return spec, protocol


class ActiveContinuation(ConfirmedObservations):
    def __init__(self, campaign, path):
        self.campaign, self.path = campaign, Path(path).absolute()
        base = campaign.base
        self.spec, self.protocol = load_parent(
            self.path,
            checksum=base.manifest.checksum,
            intent=base.task.model_dump(mode="json"),
            profile_checksum=base.active_owner.plan["profile_checksum"],
        )
        if self.spec["provider"] != campaign.config.model_dump(mode="json"):
            raise ValueError("active_continuation_provider_changed")
        self.config = validate(type(campaign.config), self.spec["provider"])
        self.snapshot = self.current_snapshot()
        self.result = dict(phases=self.snapshot["phases"])
        self.values, self.origins = self.read()  # Original current projection/receipt owner.
        self.binding = digest((self.values, self.origins))
        self.attempted = tuple(self.snapshot["attempted"])
        self.initial_attempts = self.attempts()

    def current_snapshot(self):
        c, b = self.campaign, self.campaign.base
        spec, protocol = load_parent(
            self.path,
            checksum=b.manifest.checksum,
            intent=b.task.model_dump(mode="json"),
            profile_checksum=b.active_owner.plan["profile_checksum"],
        )
        if spec != self.spec or protocol != self.protocol:
            raise ValueError("active_continuation_parent_changed")
        rows, attempted, phases, count = [], [], [], 0
        for target in (*protocol["targets"], None):
            key = "mm-live:" + digest((self.path.name, target or "review"))[7:55]
            run_id = identity("RUN", "task-research:" + key)
            try:
                context = c.contexts.read_research_run(run_id)
            except ValueError as error:
                if str(error) != "research_context_required" or c.sessions.read(run_id) is not None:
                    raise
                rows.append(dict(target=target, state="unregistered"))
                continue
            if (
                context.request.scope.case_code != b.binding["case_code"]
                or context.request.input_checksum != b.manifest.checksum
                or context.request.task != b.task
            ):
                raise ValueError("active_continuation_context_scope")
            docs = c.documents(run_id)
            control = c.sessions.read(run_id)
            events = RuntimeAgentJournal(docs).read()
            saved = docs.read("dynamic-phase-result")
            inputs = docs.read("dynamic-phase-input")
            reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
            records = [e["record"] for e in events if e.get("phase") == "completed"]
            if control is None:
                if events or saved:
                    raise ValueError("active_continuation_control_missing_with_facts")
                rows.append(
                    dict(target=target, state="registered_not_started", context=digest(context))
                )
                continue
            if (
                control.phase != "stopped"
                or control.pending is not None
                or len(reserved) != len(set(reserved))
                or len(records) != len(reserved)
                or set(reserved) != {r["request_id"] for r in records}
                or not saved
                or len(saved) != 1
                or not inputs
                or saved[0]["run_id"] != run_id
                or saved[0]["target"] != target
            ):
                raise ValueError("active_continuation_unconfirmed_phase")
            if docs.read("dynamic-campaign") != (protocol,):
                raise ValueError("active_continuation_original_campaign_binding")
            if target is None:
                raise ValueError("active_continuation_review_already_started")
            if any(r["role"] != "vision" for r in records):
                raise ValueError("active_continuation_not_read_only_vision")
            phase = saved[0]
            local = self.path / ("vision-" + digest(target)[7:19]) / "result.json"
            if local.exists() and json.loads(local.read_bytes()) != phase:
                raise ValueError("active_continuation_local_result_conflict")
            accepted = docs.read("accepted:" + phase["acceptance_key"]) if phase["accepted"] else ()
            if phase["accepted"] and not accepted:
                raise ValueError("active_continuation_accepted_document_missing")
            phases.append(phase)
            attempted.append(target)
            count += len(reserved)
            rows.append(
                dict(
                    target=target,
                    run_id=run_id,
                    context=digest(context),
                    control=digest(control),
                    events=digest(events),
                    phase=digest(phase),
                    inputs=digest(inputs),
                    accepted=digest(accepted),
                )
            )
        if count >= 40 or not phases:
            raise ValueError("active_continuation_budget_or_empty")
        return dict(rows=rows, phases=phases, attempted=attempted, requests=count)

    def attempts(self):
        return [
            dict(
                directory=str(self.path),
                result_checksum=digest(self.snapshot),
                requests=self.snapshot["requests"],
            )
        ]

    def check_attempts(self):
        if self.current_snapshot() != self.snapshot:
            raise ValueError("active_continuation_history_changed")
        return self.attempts()

    def check(self):
        # Input, source, policy and native owner checks are current; immutable
        # accepted documents and their inputs must remain byte-equivalent.
        # Rebuilding the same Phase N times per provider callback is unnecessary.
        self.campaign.base.check()
        self.check_attempts()
        return self.values

    def report(self):
        return dict(
            schema_version="confirmed-active-acquisition-continuation/1.0.0",
            previous=self.initial_attempts,
            origins=self.origins,
            observations_checksum=digest(self.values),
            attempted_targets=list(self.attempted),
            original_runs_resumed=False,
            old_native_model_reexecuted=False,
            unknown_vision_replayed=False,
            max_cumulative_requests=40,
            old_deadline_preserved=True,
        )

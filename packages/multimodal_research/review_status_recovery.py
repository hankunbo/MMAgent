"""One explicit same-deadline Review successor after a confirmed status conflict.

No failed text is corrected into evidence. Only a newly returned and accepted
Review can produce scores. Original observations, grants and failure stay intact.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .campaign import MultimodalCampaign, configuration
from .implementation import freeze
from .report_recovery import saved_campaign


def required_window(observations, batch_size):
    """Observed local setup was ~240s, in addition to each bounded remote request.

    Include two current-owner passes, one possible format correction, and the
    original close reserve. This is an estimate, never an enlarged deadline.
    """
    batches = math.ceil(observations / batch_size)
    return 300 + 2 * 240 + (batches + 1) * 240


def check_window(deadline, observations, batch_size):
    needed = required_window(observations, batch_size)
    if (datetime.fromisoformat(deadline) - datetime.now(UTC)).total_seconds() < needed:
        raise ValueError("review_status_recovery_insufficient_original_time")
    return needed


def check_failure(campaign, result):
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_batches import ReviewBatchPlan, batch_payload, check_batch
    from packages.llm_provider.review_response import decode
    from packages.object_store import checksum

    phase = result["phases"][-1]
    docs = campaign.documents(phase["run_id"])
    control = campaign.sessions.read(phase["run_id"])
    journal = RuntimeAgentJournal(docs).read()
    records = [e["record"] for e in journal if e.get("phase") == "completed"]
    reserved = [e["request_id"] for e in journal if e.get("phase") == "reserved"]
    confirmed = [
        e["request_id"]
        for e in journal
        if e.get("phase") == "response_confirmation" and e.get("confirmed") is True
    ]
    if (
        result["complete"]
        or phase["accepted"]
        or phase["target"] is not None
        or len(records) != 1
        or reserved != [records[0]["request_id"]]
        or records[0]["request_id"] not in confirmed
        or records[0]["error"] != "scope"
        or records[0]["role"] != "review"
        or records[0]["api_requests"] != 1
        or records[0]["simulated"]
        or control.phase != "stopped"
        or control.pending is not None
        or (campaign.output / "cancel").exists()
    ):
        raise ValueError("review_status_recovery_confirmed_failure_required")
    key = docs.read("live-stage-intent:review")[0]["event"]["code"] + ":A"
    plan = validate(ReviewBatchPlan, docs.read("review-batches:" + key + ":plan")[0])
    batch = plan.batches[0]
    response = docs.read(
        "review-batches:" + key + ":" + batch.id + ":attempt:" + records[0]["request_id"]
    )[0]
    if response["record_checksum"] != digest(records[0]) or response["finish_reason"] != "stop":
        raise ValueError("review_status_recovery_response_binding")
    state = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if (
        checksum(response["raw_model_text"].encode()) != response["raw_checksum"]
        or records[0]["input_checksum"] != digest(batch_payload(state, plan, batch))
        or records[0]["config_checksum"] != state.config_checksum
    ):
        raise ValueError("review_status_recovery_original_binding")
    candidate, _ = decode(response["raw_model_text"], response["provider_config"])
    try:
        check_batch(state, plan, batch, candidate)
    except ValueError as exc:
        if str(exc) != "review_ungradable_status":
            raise
    else:
        raise ValueError("review_status_recovery_no_status_failure")
    if docs.read("multimodal-case-result") != (result,):
        raise ValueError("review_status_recovery_original_result_changed")
    return dict(
        result=digest(result),
        control=digest(control),
        events=digest(journal),
        response=digest(response),
    )


class StatusReviewCampaign(MultimodalCampaign):
    def __init__(self, parent, result, output, plan):
        super().__init__(parent.base, parent.target, campaign_id=Path(output).name)
        self.parent, self.original_result = parent, result
        self.output = Path(output)
        self.config = configuration(
            resilient_vision=True, directional=True, exact_primary_status=True
        )
        self.targets = self.initial_targets = ()
        self.protocol = plan

    def previous_attempts(self):
        if check_failure(self.parent, self.original_result) != self.protocol["original_failure"]:
            raise ValueError("review_status_recovery_history_changed")
        return self.protocol["previous_attempts"]

    def accepted_observations(self):
        values = self.parent.accepted_observations()
        if digest(values) != self.protocol["confirmed_observations_checksum"]:
            raise ValueError("review_status_recovery_observations_changed")
        return values

    def expand_definitions(self, **kwargs):
        # Installed, current revision of the parent only. No new search or Source calls.
        self.base.check()
        return self.base.definition_revision

    def freeze(self):
        if self.protocol["implementation"] != freeze() or self.protocol["provider"] != digest(
            self.config
        ):
            raise ValueError("review_status_recovery_implementation_changed")
        self.previous_attempts()
        return self.protocol

    def guard(self, *, dispatch=False):
        from packages.llm_provider.provider import ProviderError

        used = self.usage()
        if used > 3 or (not dispatch and used >= 3):
            raise ProviderError("budget")
        super().guard(dispatch=dispatch)


def prepare(previous, output):
    parent, result = saved_campaign(previous)
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("review_status_recovery_new_directory_required")
    failure = check_failure(parent, result)
    config = configuration(resilient_vision=True, directional=True, exact_primary_status=True)
    facts = {**parent.base.observations, **parent.accepted_observations()}
    needed = check_window(
        parent.protocol["deadline"], len(facts), config.review_batching.batch_size
    )
    previous_attempts = parent.protocol["previous_attempts"] + [
        dict(directory=str(parent.output), requests=result["new_requests"], checksum=digest(result))
    ]
    if sum(p["requests"] for p in previous_attempts) + 3 > 40:
        raise ValueError("review_status_recovery_budget")
    plan = dict(
        schema_version="review-status-successor/1.1.0",
        parent=str(parent.output),
        original_failure=failure,
        previous_attempts=previous_attempts,
        deadline=parent.protocol["deadline"],
        started_at=datetime.now(UTC).isoformat(),
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(
            configuration(resilient_vision=True, directional=True, exact_primary_status=True)
        ),
        implementation=freeze(),
        max_new_requests=3,
        confirmed_observations_checksum=digest(parent.accepted_observations()),
        original_protocol_checksum=digest(parent.protocol),
        new_searches=0,
        new_images=0,
        new_models=0,
        required_window_seconds=needed,
    )
    docs = parent.documents(result["phases"][-1]["run_id"])
    claim = dict(output=str(out), plan_checksum=digest(plan))
    with docs.session():
        if docs.read("review-status-successor") is not None:
            raise ValueError("review_status_recovery_already_claimed")
        docs.write("review-status-successor", (claim,))
    out.mkdir(mode=0o700)
    write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        requests_used=sum(p["requests"] for p in previous_attempts),
        api_requests=0,
    )


def run(output):
    out = Path(output).absolute()
    plan = json.loads((out / "protocol.json").read_bytes())
    parent, result = saved_campaign(plan["parent"])
    docs = parent.documents(result["phases"][-1]["run_id"])
    if docs.read("review-status-successor") != (dict(output=str(out), plan_checksum=digest(plan)),):
        raise ValueError("review_status_recovery_claim")
    campaign = StatusReviewCampaign(parent, result, out, plan)
    # Preparation does not reserve time: recheck after current owner reads,
    # before registering a new phase or consuming a model reservation.
    facts = {**parent.base.observations, **parent.accepted_observations()}
    check_window(plan["deadline"], len(facts), campaign.config.review_batching.batch_size)
    return campaign.execute()

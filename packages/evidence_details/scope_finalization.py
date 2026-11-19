"""Zero-wire storage recovery of a confirmed survey rejected only by capacity.

The original response, failed request, Control and phase remain immutable. A new
local grant permits original ResearchAcceptance to save the unchanged findings
with the versioned candidate-reference observation representation. No provider,
reserve, dispatch or score method is exposed.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from packages.grounding.models import Checksum, Code, Value, digest, validate
from packages.llm_provider.input_scope import (
    PROMPT,
    STORAGE_PROMPT,
    ScopeVisionProposal,
    ScopeVisionProposalV2,
    ScopeVisionProposalV3,
)
from packages.llm_provider.models import read_request_record
from packages.object_store import checksum

from .acceptance_models import DomainState
from .input_scope import LiveScopeRecordsV2, ScopeSubmissionV2, payload
from .live_permission import ResearchLiveGrant, ResearchLivePermission, validate_live_config

GRANT_KEY = "recorded-input-scope:grant:1"
SUBMISSION_KEY = "recorded-input-scope:submission:1"
ACCEPTANCE_KEY = "recorded-input-scope:acceptance:1"


class ScopeStorageInterpretation(Value):
    schema_version: Literal["input-scope-storage-interpretation/1.0.0"] = (
        "input-scope-storage-interpretation/1.0.0"
    )
    raw_checksum: Checksum
    original_candidate_checksum: Checksum
    candidate_checksum: Checksum
    original_error: Literal["input_scope_original_observation_bound"] = (
        "input_scope_original_observation_bound"
    )
    changed_fields: tuple[Literal["schema_version"]] = ("schema_version",)
    findings_regions_bindings_unchanged: Literal[True] = True


class RecordedScopeSubmission(ScopeSubmissionV2):
    schema_version: Literal["research-input-scope-submission/1.2.0"] = (
        "research-input-scope-submission/1.2.0"
    )
    storage_interpretation: ScopeStorageInterpretation
    original_request_checksum: Checksum


class RecordedScopeRecords(LiveScopeRecordsV2):
    schema_version: Literal["case-input-scope-observation/2.2.0"] = (
        "case-input-scope-observation/2.2.0"
    )
    storage_interpretation: ScopeStorageInterpretation
    original_request_checksum: Checksum


class ScopeTextInterpretation(ScopeStorageInterpretation):
    schema_version: Literal["input-scope-storage-interpretation/1.1.0"] = (
        "input-scope-storage-interpretation/1.1.0"
    )
    original_error: Literal["input_scope_region_text_bound"] = "input_scope_region_text_bound"
    original_prompt: Literal["vision-input-scope-1.1.0"] = STORAGE_PROMPT
    original_text_limit: Literal[100] = 100
    current_text_limit: Literal[512] = 512
    # Exact paths, not model-content heuristics; original raw response remains unchanged.
    affected_fields: tuple[str, ...] = Field(min_length=1, max_length=8)


class RecordedScopeTextSubmission(RecordedScopeSubmission):
    schema_version: Literal["research-input-scope-submission/1.3.0"] = (
        "research-input-scope-submission/1.3.0"
    )
    candidate: ScopeVisionProposalV3
    storage_interpretation: ScopeTextInterpretation


class RecordedScopeTextRecords(RecordedScopeRecords):
    schema_version: Literal["case-input-scope-observation/2.3.0"] = (
        "case-input-scope-observation/2.3.0"
    )
    candidate: ScopeVisionProposalV3
    storage_interpretation: ScopeTextInterpretation


def recorded_records_type(submission):
    return (
        RecordedScopeTextRecords
        if isinstance(submission, RecordedScopeTextSubmission)
        else RecordedScopeRecords
    )


class RecordedScopeGrant(Value):
    schema_version: Literal["recorded-input-scope-grant/1.0.0"] = "recorded-input-scope-grant/1.0.0"
    purpose: Literal["save_confirmed_scope_storage_only"] = "save_confirmed_scope_storage_only"
    run_id: Code
    case_code: Code
    original_grant_checksum: Checksum
    failed_phase_checksum: Checksum
    control_checksum: Checksum
    state_checksum: Checksum
    submission_checksum: Checksum
    original_request_checksum: Checksum
    authorization: str = Field(min_length=1, max_length=200)
    issued_at: datetime
    deadline: datetime
    new_api_requests: Literal[0] = 0
    new_disease_scores: Literal[0] = 0

    @model_validator(mode="after")
    def bound(self):
        if (
            not self.authorization.strip()
            or self.issued_at.tzinfo is None
            or self.deadline.tzinfo is None
            or not timedelta(0) < self.deadline - self.issued_at <= timedelta(minutes=15)
        ):
            raise ValueError("recorded_scope_local_bound")
        return self


def interpret(raw_text, *, prompt=PROMPT):
    """Prove exactly one old storage-capacity error, not a medical correction."""
    if not isinstance(raw_text, str) or len(raw_text.encode()) > 32768:
        raise ValueError("recorded_scope_response_bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("recorded_scope_duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("recorded_scope_nonfinite_json")

    try:
        raw = json.loads(raw_text, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError as error:
        raise ValueError("recorded_scope_response_depth") from error
    if prompt == STORAGE_PROMPT:
        return interpret_text_capacity(raw_text, raw)
    if prompt != PROMPT:
        raise ValueError("recorded_scope_prompt_version")
    try:
        validate(ScopeVisionProposal, raw)
    except ValidationError as error:
        errors = error.errors()
        if not (
            len(errors) == 1
            and errors[0]["loc"] == ()
            and errors[0]["type"] == "value_error"
            and str(errors[0].get("ctx", {}).get("error"))
            == "input_scope_original_observation_bound"
        ):
            raise ValueError("recorded_scope_not_storage_only") from error
    else:
        raise ValueError("recorded_scope_no_capacity_failure")
    candidate = validate(
        ScopeVisionProposalV2,
        dict(raw, schema_version="visual-input-scope-candidate/1.1.0"),
    )
    return candidate, ScopeStorageInterpretation(
        raw_checksum=checksum(raw_text.encode()),
        original_candidate_checksum=digest(raw),
        candidate_checksum=digest(candidate),
    )


def interpret_text_capacity(raw_text, raw):
    """Only region text overflow may change representation; no content correction."""
    try:
        validate(ScopeVisionProposalV2, raw)
    except ValidationError as error:
        failures = error.errors()
        fields = []
        for failure in failures:
            loc = failure["loc"]
            if (
                failure["type"] == "string_too_long"
                and len(loc) == 3
                and loc[0] == "regions"
                and type(loc[1]) is int
                and loc[2] in {"finding", "limitation"}
                and failure.get("ctx", {}).get("max_length") == 100
            ):
                fields.append("/" + "/".join(map(str, loc)))
            elif loc == ("regions",) and failure["type"] == "too_short":
                # Pydantic removes invalid elements before checking tuple length.
                # Full V3 validation below still requires the original region count.
                continue
            else:
                raise ValueError("recorded_scope_not_text_capacity_only") from error
        if not fields:
            raise ValueError("recorded_scope_no_text_capacity_failure") from error
    else:
        raise ValueError("recorded_scope_no_text_capacity_failure")
    candidate = validate(
        ScopeVisionProposalV3,
        dict(raw, schema_version="visual-input-scope-candidate/1.2.0"),
    )
    unchanged = candidate.model_dump(mode="json")
    unchanged["schema_version"] = raw["schema_version"]
    if unchanged != raw:
        raise ValueError("recorded_scope_text_content_changed")
    return candidate, ScopeTextInterpretation(
        raw_checksum=checksum(raw_text.encode()),
        original_candidate_checksum=digest(raw),
        candidate_checksum=digest(candidate),
        affected_fields=tuple(fields),
    )


class RecordedScopeRequests:
    """Original immutable journal, no synthetic successful request projection."""

    def __init__(self, journal):
        if not journal.durable:
            raise ValueError("recorded_scope_durable_journal_required")
        self.journal = journal

    def records(self):
        events = self.journal.read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        records = [
            read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
        ]
        confirmed = [
            e["request_id"]
            for e in events
            if e.get("phase") == "response_confirmation" and e.get("confirmed") is True
        ]
        if len(records) != 1:
            raise ValueError("recorded_scope_single_confirmed_request_required")
        record = records[0]
        if (
            reserved != [record.request_id]
            or confirmed != [record.request_id]
            or record.simulated
            or record.api_requests != 1
            or record.role != "vision"
            or record.error != "schema"
            or record.retry_index != 0
            or record.prompt_version not in {PROMPT, STORAGE_PROMPT}
        ):
            raise ValueError("recorded_scope_unknown_or_wrong_request")
        return tuple(records)


def original_values(documents):
    phase = documents.read("dynamic-phase-result")
    intent = documents.read("live-stage-intent:vision")
    if (
        not phase
        or len(phase) != 1
        or phase[0].get("run_id") != documents.run_id
        or not str(phase[0].get("target", "")).startswith("input-scope:")
        or phase[0].get("accepted") is not False
        or phase[0].get("error") != "candidate_failed_or_context_changed"
        or not intent
        or len(intent) != 1
        or intent[0].get("role") != "vision"
    ):
        raise ValueError("recorded_scope_original_failed_phase_required")
    key = "review:" + intent[0]["event"]["code"]
    failure = documents.read(key + ":candidate-failure")
    frame = documents.read(key + ":intent")
    if (
        not failure
        or len(failure) != 1
        or failure[0].get("category") != "provider"
        or failure[0].get("code") != "schema"
        or not frame
        or len(frame) != 1
        or frame[0]["frame"]["event"] != intent[0]["event"]
        or not any(
            a["id"] == intent[0]["proposed_action_id"] and a["branch"] == "research_vision"
            for a in frame[0]["frame"]["actions"]["actions"]
        )
    ):
        raise ValueError("recorded_scope_original_review_binding")
    return phase[0], intent[0], frame[0]


def reconstruct(owner, documents, requests):
    state = validate(DomainState, owner.current())
    documents.validate_scope(state.material.scope)
    _, intent, frame = original_values(documents)
    record = requests.records()[0]
    expected = payload(owner, state)
    raw = documents.read("live-response:" + record.request_id)
    if (
        documents.read("dynamic-phase-input") != (state.model_dump(mode="json"),)
        or frame["state"] != state.model_dump(mode="json")
        or frame["frame"]["scope"] != state.material.scope.model_dump(mode="json")
        or intent["config_checksum"] != state.config_checksum
        or not raw
        or len(raw) != 1
        or raw[0].get("schema_version") != "bounded-live-response/1.0.0"
        or raw[0].get("request_id") != record.request_id
        or raw[0].get("validation_error") != "schema"
        or raw[0].get("untrusted_model_output") is not True
        or raw[0].get("accepted_medical_evidence") is not False
        or record.scope != state.material.scope
        or record.input_checksum != digest(expected)
        or record.prompt_checksum != state.prompt_checksum
        or record.config_checksum != state.config_checksum
    ):
        raise ValueError("recorded_scope_original_material_binding")
    text = raw[0]["raw_model_text"]
    if raw[0]["raw_checksum"] != checksum(text.encode()) or record.output_checksum != digest(
        text.encode().hex()
    ):
        raise ValueError("recorded_scope_response_checksum")
    candidate, proof = interpret(text, prompt=record.prompt_version)
    from packages.llm_provider.input_scope import check_regions

    check_regions(candidate, expected.scope_task)
    cls = (
        RecordedScopeTextSubmission
        if isinstance(proof, ScopeTextInterpretation)
        else RecordedScopeSubmission
    )
    sub = cls(
        scope=state.material.scope,
        actor="PLANNER",
        correlation="SOURCE:AGENT_RESEARCH",
        request_id=record.request_id,
        prompt_checksum=record.prompt_checksum,
        config_checksum=record.config_checksum,
        observation_checksums=tuple(o.reference for o in state.material.observations),
        source_checksums=tuple(s.reference for s in state.material.sources),
        candidate=candidate,
        storage_interpretation=proof,
        original_request_checksum=digest(record),
    )
    return state, sub, record


def check_request(submission, state, owner, documents, requests):
    current, expected, record = reconstruct(owner, documents, requests)
    if current != state or submission != expected:
        raise ValueError("recorded_scope_submission_changed")
    return record


class RecordedScopePermission(ResearchLivePermission):
    def __init__(self, *, grant, config, documents, requests, owner, clock=None):
        self.grant = validate(RecordedScopeGrant, grant)
        self.config = validate_live_config(config)
        self.documents, self.budget, self.owner = documents, requests, owner
        self.clock = clock or (lambda: datetime.now(UTC))
        if requests.journal.documents is not documents:
            raise ValueError("recorded_scope_owner_binding")

    @classmethod
    def prepare(
        cls, *, owner, documents, requests, config, authorization, clock=None, persist=True
    ):
        state, sub, record = reconstruct(owner, documents, requests)
        phase, _, _ = original_values(documents)
        control = documents.sessions.read(documents.run_id)
        old = validate(ResearchLiveGrant, documents.read("research-live-grant")[0])
        expected = dict(
            run_id=documents.run_id,
            case_code=state.material.scope.case_code,
            original_grant_checksum=digest(old),
            failed_phase_checksum=digest(phase),
            control_checksum=digest(control),
            state_checksum=digest(state),
            submission_checksum=digest(sub),
            original_request_checksum=digest(record),
            authorization=authorization,
        )
        now = (clock or (lambda: datetime.now(UTC)))()
        with documents.session():
            if documents.consumed(state.material.scope.graph):
                raise ValueError("parent_graph_already_consumed")
            saved = documents.read(GRANT_KEY)
            grant = (
                validate(RecordedScopeGrant, saved[0])
                if saved
                else RecordedScopeGrant(
                    **expected, issued_at=now, deadline=now + timedelta(minutes=15)
                )
            )
            if any(getattr(grant, k) != v for k, v in expected.items()):
                raise ValueError("recorded_scope_grant_conflict")
            permission = cls(
                grant=grant,
                config=config,
                documents=documents,
                requests=requests,
                owner=owner,
                clock=clock,
            )
            permission.check_bindings(state, role="vision", record=record, persisted=False)
            permission.check_deadline()
            if persist:
                documents.write(SUBMISSION_KEY, (sub,))
                documents.write(GRANT_KEY, (grant,))
        return permission, sub

    def check_deadline(self):
        if self.clock() >= self.grant.deadline:
            raise ValueError("recorded_scope_local_deadline")

    def check_bindings(self, state, *, role=None, record=None, persisted=True):
        g, docs, scope = self.grant, self.documents, state.material.scope
        current, sub, original = reconstruct(self.owner, docs, self.budget)
        old = validate(ResearchLiveGrant, docs.read("research-live-grant")[0])
        phase, _, _ = original_values(docs)
        control = docs.sessions.read(docs.run_id)
        if (
            role not in (None, "vision")
            or current != state
            or digest(state) != g.state_checksum
            or digest(sub) != g.submission_checksum
            or digest(original) != g.original_request_checksum
            or (record is not None and record != original)
            or digest(phase) != g.failed_phase_checksum
            or digest(old) != g.original_grant_checksum
            or control is None
            or control.phase != "stopped"
            or control.pending is not None
            or digest(control) != g.control_checksum
            or (scope.run_id, scope.case_code) != (g.run_id, g.case_code)
            or scope.graph.kind != "research_artifact"
            or (
                scope.run_id,
                scope.case_code,
                scope.input_checksum,
                scope.visible_checksum,
                scope.policy_checksum,
            )
            != (
                old.run_id,
                old.case_code,
                old.input_checksum,
                old.visible_checksum,
                old.policy_checksum,
            )
            or digest(self.config) != old.config_checksum
            or state.config_checksum != old.config_checksum
            or old.root_checksum != docs.root.checksum
            or "vision" not in old.roles
            or original.deadline > old.deadline
            or (original.provider, original.model, original.endpoint)
            != (
                self.config.transport.provider,
                self.config.transport.model,
                self.config.transport.endpoint,
            )
            or any(s.purpose != "research_material" for s in state.material.sources)
        ):
            raise ValueError("recorded_scope_current_binding")
        if persisted and (
            docs.read(GRANT_KEY) != (g.model_dump(mode="json"),)
            or docs.read(SUBMISSION_KEY) != (sub.model_dump(mode="json"),)
        ):
            raise ValueError("recorded_scope_saved_grant_changed")

    def check(self, state, *, role=None, record=None):
        self.check_deadline()
        self.check_bindings(state, role=role, record=record)
        self.check_deadline()


class SavedScopePermission(RecordedScopePermission):
    """Historical read only: never a replacement deadline or write permission."""

    def check(self, state, *, role=None, record=None):
        if (
            role is not None
            or record is not None
            or not self.documents.read("accepted:" + ACCEPTANCE_KEY)
        ):
            raise ValueError("recorded_scope_read_only")
        self.check_bindings(state)

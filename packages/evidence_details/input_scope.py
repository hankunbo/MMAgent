"""Input-survey admission through the existing research acceptance owner.

No new database or medical permission. The task reader must reconstruct the
current input binding; a saved survey is a research observation, not diagnosis.
"""

from pathlib import Path
from typing import Literal

from pydantic import model_validator

from packages.grounding.models import digest, validate
from packages.llm_provider.input_scope import (
    STORAGE_PROMPT,
    TEXT_PROMPT,
    ScopeTask,
    ScopeVisionPayload,
    ScopeVisionProposal,
    ScopeVisionProposalV2,
    ScopeVisionProposalV3,
)

from .acceptance_models import AcceptedRecordsV21, AcceptedRecordsV22, Submission, TaskPolicyV2
from .review_tasks import project_review_tasks


class ScopeSubmission(Submission):
    schema_version: Literal["research-input-scope-submission/1.0.0"] = (
        "research-input-scope-submission/1.0.0"
    )
    candidate: ScopeVisionProposal


class ScopeRecords(AcceptedRecordsV21):
    schema_version: Literal["case-input-scope-observation/1.0.0"] = (
        "case-input-scope-observation/1.0.0"
    )
    candidate: ScopeVisionProposal
    scope_task: ScopeTask
    interpretations: tuple[()] = ()
    invalidated_assessment_ids: tuple[()] = ()


class LiveScopeRecords(AcceptedRecordsV22):
    schema_version: Literal["case-input-scope-observation/1.1.0"] = (
        "case-input-scope-observation/1.1.0"
    )
    candidate: ScopeVisionProposal
    scope_task: ScopeTask
    interpretations: tuple[()] = ()
    invalidated_assessment_ids: tuple[()] = ()


class ScopeSubmissionV2(ScopeSubmission):
    schema_version: Literal["research-input-scope-submission/1.1.0"] = (
        "research-input-scope-submission/1.1.0"
    )
    candidate: ScopeVisionProposalV2


class ScopeRecordsV2(ScopeRecords):
    schema_version: Literal["case-input-scope-observation/2.0.0"] = (
        "case-input-scope-observation/2.0.0"
    )
    candidate: ScopeVisionProposalV2

    @model_validator(mode="after")
    def observation_binding(self):
        return check_storage(self)


class LiveScopeRecordsV2(LiveScopeRecords):
    schema_version: Literal["case-input-scope-observation/2.1.0"] = (
        "case-input-scope-observation/2.1.0"
    )
    candidate: ScopeVisionProposalV2

    @model_validator(mode="after")
    def observation_binding(self):
        return check_storage(self)


class ScopeSubmissionV3(ScopeSubmissionV2):
    schema_version: Literal["research-input-scope-submission/1.4.0"] = (
        "research-input-scope-submission/1.4.0"
    )
    candidate: ScopeVisionProposalV3


class ScopeRecordsV3(ScopeRecordsV2):
    schema_version: Literal["case-input-scope-observation/3.0.0"] = (
        "case-input-scope-observation/3.0.0"
    )
    candidate: ScopeVisionProposalV3


class LiveScopeRecordsV3(LiveScopeRecordsV2):
    schema_version: Literal["case-input-scope-observation/3.1.0"] = (
        "case-input-scope-observation/3.1.0"
    )
    candidate: ScopeVisionProposalV3


def check_storage(records):
    if (
        len(records.observations) != 1
        or records.observations[0].value != records.candidate.observation_value()
    ):
        raise ValueError("input_scope_saved_observation_binding")
    return records


def submission_type(candidate):
    if isinstance(candidate, ScopeVisionProposalV3):
        return ScopeSubmissionV3
    return ScopeSubmissionV2 if isinstance(candidate, ScopeVisionProposalV2) else ScopeSubmission


def records_type(candidate, *, live):
    if isinstance(candidate, ScopeVisionProposalV3):
        return LiveScopeRecordsV3 if live else ScopeRecordsV3
    if isinstance(candidate, ScopeVisionProposalV2):
        return LiveScopeRecordsV2 if live else ScopeRecordsV2
    return LiveScopeRecords if live else ScopeRecords


def payload(owner, state):
    reader = getattr(owner, "input_scope_task", None)
    if not callable(reader):
        raise ValueError("input_scope_current_owner_required")
    if type(state.policy) is not TaskPolicyV2:
        raise ValueError("input_scope_acquisition_policy_required")
    task = validate(ScopeTask, reader(state.material.scope))
    templates = project_review_tasks(state)
    if len(templates.tasks) != 1 or templates.tasks[0].method != "input_scope_survey":
        raise ValueError("input_scope_only_survey_task")
    return validate(
        ScopeVisionPayload,
        dict(
            schema_version="input-scope-vision-input/1.0.0",
            role="vision",
            material=state.material,
            vision_task=templates.tasks[0],
            scope_task=task,
        ),
    )


def check_submission(submission, state, owner, record):
    from packages.llm_provider.input_scope import check_regions

    from .scope_finalization import RecordedScopeSubmission

    if not isinstance(submission, ScopeSubmission):
        raise ValueError("input_scope_submission_version")
    expected = payload(owner, state)
    if record.input_checksum != digest(expected):
        raise ValueError("input_scope_request_task_binding")
    # A new output contract cannot silently consume an old prompt/request.
    if (
        isinstance(submission, ScopeSubmissionV2)
        and not isinstance(submission, RecordedScopeSubmission)
        and record.prompt_checksum
        != digest(
            Path(
                "configs/prompts/agents",
                (TEXT_PROMPT if isinstance(submission, ScopeSubmissionV3) else STORAGE_PROMPT)
                + ".txt",
            ).read_text()
        )
    ):
        raise ValueError("input_scope_request_contract_binding")
    check_regions(submission.candidate, expected.scope_task)
    return expected.scope_task

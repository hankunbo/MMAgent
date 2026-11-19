"""Operator-authorized new bounded window over original confirmed case facts.

Not resume/retry: old stops, unknown requests and deadlines remain immutable.
The same Runtime/Planner/Provider/definition/report owners execute each new phase.
"""

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.dataset_adapters.current_read_scope import _projection_visible_reads
from packages.evidence_details.accepted_read_scope import historical_fact_read_scope
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .alternative_acquisition import AGGREGATE_VERSION as AGGREGATE_ALTERNATIVE_VERSION
from .alternative_acquisition import AGGREGATE_VERSIONS as AGGREGATE_ALTERNATIVE_VERSIONS
from .alternative_acquisition import VERSIONS as ALTERNATIVE_ACQUISITION_VERSIONS
from .campaign import MultimodalCampaign, configuration
from .capability_window import (
    CURRENT_SCOPE_RECOVERY_VERSION as CAPABILITY_CURRENT_SCOPE_RECOVERY_VERSION,
)
from .capability_window import CURRENT_VERSIONS as CAPABILITY_CURRENT_VERSIONS
from .capability_window import RECOVERY_VERSION as CAPABILITY_RECOVERY_VERSION
from .capability_window import ROUTED_VERSION as CAPABILITY_ROUTED_VERSION
from .capability_window import SCOPE_CONFIRMED_VERSION as CAPABILITY_SCOPE_CONFIRMED_VERSION
from .capability_window import SCOPE_UNKNOWN_VERSION as CAPABILITY_SCOPE_UNKNOWN_VERSION
from .capability_window import VERSION as CAPABILITY_VERSION
from .capability_window import VERSIONS as CAPABILITY_VERSIONS
from .confirmed_definition_failure import VERSIONS as CONFIRMED_DEFINITION_FAILURE_VERSIONS
from .definition_alternative import VERSION as DEFINITION_ALTERNATIVE_VERSION
from .definition_expansion import read_expansion_policy
from .implementation import freeze
from .interrupted_definition import VERSIONS as INTERRUPTED_DEFINITION_VERSIONS
from .report_recovery import saved_campaign
from .source_checkpoint import KIND as SOURCE_CHECKPOINT_KIND
from .source_checkpoint import VERSION as SOURCE_CHECKPOINT_VERSION
from .unknown_acquisition import CONFIGURATION_VERSION as CONFIGURATION_ACQUISITION_VERSION
from .unknown_acquisition import CONFIRMED_INVALID_VERSION as CONFIRMED_INVALID_ACQUISITION_VERSION
from .unknown_acquisition import HISTORICAL_CLIENT_VERSION as HISTORICAL_ACQUISITION_VERSION
from .unknown_acquisition import LOCAL_VERSION as UNKNOWN_LOCAL_VERSION
from .unknown_acquisition import POST_DEFINITION_VERSION as POST_DEFINITION_ACQUISITION_VERSION
from .unknown_acquisition import VERSION as UNKNOWN_ACQUISITION_VERSION
from .unknown_acquisition import VERSIONS as UNKNOWN_ACQUISITION_VERSIONS

LIMITS = {"review": (1800, 3), "definitions": (3600, 12), "acquisition": (5400, 12)}
MAX_HISTORY_DEPTH = 16
RUNTIME_DEFINITION_VERSION = "multimodal-campaign-renewal/1.56.0"
RECORDED_DEFINITION_VERSION = "multimodal-campaign-renewal/1.78.0"
CASE_DEFINITION_VERSION = "multimodal-campaign-renewal/1.79.0"
CASE_DEFINITION_VERSIONS = {CASE_DEFINITION_VERSION, SOURCE_CHECKPOINT_VERSION}
DISCOVERED_CANDIDATE_POLICY = "campaign-definition-expansion/1.30.0"
CONFIRMED_INDEX_POLICY = "campaign-definition-expansion/1.31.0"
REVISED_INDEX_POLICY = "campaign-definition-expansion/1.32.0"
REVISED_PUBLIC_POLICY = "campaign-definition-expansion/1.33.0"
IDENTITY_INDEX_POLICY = "campaign-definition-expansion/1.34.0"
TYPED_INDEX_POLICY = "campaign-definition-expansion/1.35.0"
NUMERIC_PUBLIC_POLICY = "campaign-definition-expansion/1.36.0"
REVISED_ACQUISITION_POLICIES = {
    REVISED_INDEX_POLICY,
    REVISED_PUBLIC_POLICY,
    NUMERIC_PUBLIC_POLICY,
    "campaign-definition-expansion/1.37.0",
    "campaign-definition-expansion/1.38.0",
    "campaign-definition-expansion/1.39.0",
    "campaign-definition-expansion/1.40.0",
    IDENTITY_INDEX_POLICY,
    TYPED_INDEX_POLICY,
}
INDEX_ONLY_POLICIES = {
    "campaign-definition-expansion/1.38.0",
    "campaign-definition-expansion/1.39.0",
    "campaign-definition-expansion/1.40.0",
    CONFIRMED_INDEX_POLICY,
    REVISED_INDEX_POLICY,
    IDENTITY_INDEX_POLICY,
    TYPED_INDEX_POLICY,
}
RUNTIME_DEFINITION_POLICIES = {
    DISCOVERED_CANDIDATE_POLICY,
    CONFIRMED_INDEX_POLICY,
    *REVISED_ACQUISITION_POLICIES,
}
EXACT_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.32.0"
ACQUISITION_REVIEW_VERSION = "multimodal-campaign-renewal/1.33.0"
CLARIFIED_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.34.0"
RETAINED_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.37.0"
UNSTARTED_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.49.0"
ACTIVATED_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.50.0"
UNSTARTED_ACQUISITION_VERSIONS = {UNSTARTED_ACQUISITION_VERSION, ACTIVATED_ACQUISITION_VERSION}
UNSTARTED_REVIEW_VERSION = "multimodal-campaign-renewal/1.71.0"
UNSTARTED_AGGREGATE_ALTERNATIVE_VERSION = "multimodal-campaign-renewal/1.73.0"
UNSTARTED_WINDOW_VERSIONS = UNSTARTED_ACQUISITION_VERSIONS | {
    UNSTARTED_REVIEW_VERSION,
    UNSTARTED_AGGREGATE_ALTERNATIVE_VERSION,
}
RETAINED_ACQUISITION_VERSIONS = {
    *(ALTERNATIVE_ACQUISITION_VERSIONS - {DEFINITION_ALTERNATIVE_VERSION}),
    RETAINED_ACQUISITION_VERSION,
    *UNSTARTED_ACQUISITION_VERSIONS,
    *UNKNOWN_ACQUISITION_VERSIONS,
}
RETAINED_CONTEXT_SCOPE = "current-owner-prior-visible/1.0.0"
RETAINED_REVIEW_VERSION = "multimodal-campaign-renewal/1.38.0"
CAPACITY_REVIEW_VERSION = "multimodal-campaign-renewal/1.43.0"
UNSCORED_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.52.0"
TRUNCATED_REVIEW_VERSION = "multimodal-campaign-renewal/1.53.0"
ACQUISITION_REVIEW_VERSIONS = {ACQUISITION_REVIEW_VERSION, RETAINED_REVIEW_VERSION}
EXACT_ACQUISITION_VERSIONS = {
    *ALTERNATIVE_ACQUISITION_VERSIONS,
    EXACT_ACQUISITION_VERSION,
    CLARIFIED_ACQUISITION_VERSION,
    RETAINED_ACQUISITION_VERSION,
    *UNSTARTED_ACQUISITION_VERSIONS,
    UNKNOWN_ACQUISITION_VERSION,
    UNKNOWN_LOCAL_VERSION,
    CONFIGURATION_ACQUISITION_VERSION,
    CONFIRMED_INVALID_ACQUISITION_VERSION,
    HISTORICAL_ACQUISITION_VERSION,
    POST_DEFINITION_ACQUISITION_VERSION,
}
CURRENT_INCREMENT_POLICIES = {
    "campaign-definition-expansion/1.24.0",
    "campaign-definition-expansion/1.25.0",
    "campaign-definition-expansion/1.26.0",
    "campaign-definition-expansion/1.27.0",
    "campaign-definition-expansion/1.28.0",
    DISCOVERED_CANDIDATE_POLICY,
    CONFIRMED_INDEX_POLICY,
    *REVISED_ACQUISITION_POLICIES,
}


def validate_context_window(plan, parent):
    """Whole-case pending Review inherits the original deadline, never a reset."""
    profile = getattr(parent.base, "execution_profile", None)
    if (
        profile not in {"relay-context-v1", "relay-snapshot-context-v1", "relay-root-question-v1"}
        or plan.get("whole_case_execution_profile") != profile
        or plan.get("original_window_deadline") != parent.protocol["deadline"]
        or datetime.fromisoformat(plan["deadline"])
        > datetime.fromisoformat(parent.protocol["deadline"])
        or datetime.fromisoformat(plan["deadline"]) <= datetime.fromisoformat(plan["started_at"])
        or not plan.get("pending_review")
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan["provider"] != digest(parent.config)
    ):
        raise ValueError("renewal_context_original_window_binding")


def current_increment(plan):
    """Explicit successor semantics; never reinterpret a historical protocol."""
    return plan.get("schema_version") in {
        "multimodal-campaign-renewal/1.11.0",
        RUNTIME_DEFINITION_VERSION,
        RECORDED_DEFINITION_VERSION,
        *CASE_DEFINITION_VERSIONS,
        *INTERRUPTED_DEFINITION_VERSIONS,
        *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
    }


def definition_phase_base(base, *, original_requirements=False):
    """New definition scope does not inherit an accepted Review continuation.

    Keep facts and their readers; the coupled prior-review context flag belongs
    to the old Review reader, which cannot authorize a revised definition scope.
    """
    from copy import copy

    current = copy(base)
    if original_requirements:
        from .definition_expansion import DefinitionExpansion

        origin = getattr(getattr(base, "definition_current", None), "__self__", None)
        if not isinstance(origin, DefinitionExpansion) or origin.campaign.base is not base:
            raise ValueError("renewal_definition_original_owner_missing")
        # The successor installs the same admitted revision on its own base.
        # Supply the owner's original baseline, never today's revised ledger.
        current.ledger = list(origin.original_rows)
    current.confirmed_review_reader = None
    current.review_clarification = None
    current.review_context_scope = None
    return current


def validate_definition_runtime(plan):
    """New preparation only; never replace the lock on an old frozen plan."""
    from .execution_lease import PROFILE

    if (
        plan.get("schema_version")
        not in {
            RUNTIME_DEFINITION_VERSION,
            RECORDED_DEFINITION_VERSION,
            *CASE_DEFINITION_VERSIONS,
            *INTERRUPTED_DEFINITION_VERSIONS,
            *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
        }
        or plan.get("mode") != "definitions"
        or plan.get("definition_increment_only") is not True
        or plan.get("expansion_policy", {}).get("schema_version") not in RUNTIME_DEFINITION_POLICIES
        or plan.get("execution_lease_profile") != PROFILE
    ):
        raise ValueError("definition_runtime_execution_lease_profile_binding")
    if (
        plan["expansion_policy"]["schema_version"] in INDEX_ONLY_POLICIES
        and plan.get("new_public_search_allowed") is not False
    ):
        raise ValueError("definition_index_only_no_public_dispatch")
    if bool(plan.get("interrupted_definitions")) != (
        plan.get("schema_version") in INTERRUPTED_DEFINITION_VERSIONS
    ):
        raise ValueError("interrupted_definition_version")
    if bool(plan.get("confirmed_definition_failure")) != (
        plan.get("schema_version") in CONFIRMED_DEFINITION_FAILURE_VERSIONS
    ):
        raise ValueError("confirmed_definition_failure_version")
    if bool(plan.get("recorded_definition_origin")) != (
        plan.get("schema_version") == RECORDED_DEFINITION_VERSION
    ):
        raise ValueError("recorded_definition_protocol_version")
    if (plan.get("pending_definition_acquisition") is not None) != (
        plan.get("schema_version") in CASE_DEFINITION_VERSIONS
    ):
        raise ValueError("pending_definition_protocol_version")
    if bool(plan.get("source_checkpoint")) != (
        plan.get("schema_version") == SOURCE_CHECKPOINT_VERSION
    ):
        raise ValueError("source_checkpoint_protocol_version")
    if plan.get("confirmed_definition_failure"):
        from .confirmed_definition_failure import check_plan as check_confirmed_failure

        check_confirmed_failure(plan)
    if plan.get("interrupted_definitions"):
        from .interrupted_definition import check_plan

        check_plan(plan)


def validate_format_window(plan, parent):
    """Explicit separate closure window; never reinterpret or revive the old run."""
    start = datetime.fromisoformat(plan["started_at"])
    end = datetime.fromisoformat(plan["deadline"])
    if (
        plan.get("schema_version") != "multimodal-campaign-renewal/1.24.0"
        or plan.get("mode") != "review"
        or not plan.get("pending_review", {}).get("acknowledged_format_requests")
        or plan.get("previous_window_deadline") != parent.protocol["deadline"]
        or plan.get("whole_case_execution_profile")
        != getattr(parent.base, "execution_profile", None)
        or plan.get("whole_case_execution_profile")
        not in {"relay-context-v1", "relay-snapshot-context-v1", "relay-root-question-v1"}
        or plan["provider"] != digest(parent.config)
        or not 0 < (end - start).total_seconds() <= 2700
        or plan["seconds"] != (end - start).total_seconds()
        or not 1 <= plan["max_new_requests"] <= 10
        or sum(p["requests"] for p in plan["previous_attempts"]) + plan["max_new_requests"] > 40
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
    ):
        raise ValueError("renewal_known_format_window_binding")
    ancestor = parent
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if ancestor.protocol.get("schema_version") == "multimodal-campaign-renewal/1.24.0":
            raise ValueError("renewal_known_format_window_already_used")
        ancestor = getattr(ancestor, "parent", None)
        if ancestor is None:
            return
    raise ValueError("renewal_history_depth")


def validate_increment_window(policy, *, seconds, cap):
    """A changed-definition scope needs proposal capacity as well as Review reserve."""
    if (
        policy["reserve_review_requests"] >= cap
        or policy["reserve_review_seconds"] + policy["reserve_close_seconds"] >= seconds
    ):
        raise ValueError("renewal_increment_no_proposal_capacity")


@historical_fact_read_scope()
@_projection_visible_reads()
def completed_parent(path, *, depth=0, ancestry=()):
    """Read completed history; recheck input before outer fact/Runtime checks.

    Input proof reuse ends before historical_fact_read_scope performs its fresh
    return checks. No cached input survives into actual dispatch or acceptance.
    """
    if depth > MAX_HISTORY_DEPTH:
        raise ValueError("renewal_history_depth")
    path = Path(path).absolute()
    if path in ancestry:
        raise ValueError("renewal_history_cycle")
    ancestry = (*ancestry, path)
    if (path / "manifest.json").is_file() and not (path / "protocol.json").exists():
        from .accepted_aggregation import restore_aggregation

        return restore_aggregation(path, depth=depth, ancestry=ancestry)
    plan = json.loads((path / "protocol.json").read_bytes())
    if plan.get("schema_version", "").startswith("multimodal-campaign-renewal/"):
        campaign = assemble(path, depth=depth + 1, ancestry=ancestry)
        result = json.loads((path / "result.json").read_bytes())
        if campaign.documents(result["phases"][-1]["run_id"]).read("multimodal-case-result") != (
            result,
        ):
            raise ValueError("renewal_parent_result_binding")
        # Reading a completed zero-increment Source successor is not replaying
        # its request. The original terminal gate still checks every journal,
        # wire outcome, revision, Control and result binding below.
        completed_source = bool(
            current_increment(plan)
            and plan.get("mode") == "definitions"
            and plan.get("definition_increment_only")
            and result.get("complete") is True
        )
        stopped_tip(campaign, path, completed_source=completed_source)
        return campaign, result
    return saved_campaign(path)


def gate_selection(campaign, revision):
    """Only changed definitions are reviewed; never relabel a full-case score."""
    changed = set(revision["added"]) | {r["id"] for r in revision["replaced"]}
    selected = tuple(sorted(changed))
    separate = campaign.protocol.get("schema_version") in CASE_DEFINITION_VERSIONS
    pending = campaign.protocol.get("pending_definition_acquisition", {}).get("targets", [])
    acquisition_ids = selected
    if separate:
        allowed_acquisition = {
            r["id"] for r in revision["rows"] if r.get("acquisition_allowed") is True
        }
        acquisition_ids = tuple(
            t for t in [*pending, *sorted(changed - set(pending))] if t in allowed_acquisition
        )
    if any("dependency_assessment" in r for r in revision["rows"]):
        allowed = {r["id"] for r in revision["rows"] if r.get("review_allowed", True)}
        selected = tuple(r for r in selected if r in allowed)
    campaign.base.review_target_ids = selected
    if revised_acquisition(campaign):
        # Only incoming targets selected by the original input/method owner are
        # candidates. A definition never creates a patient observation or grants
        # permission to repeat an earlier UNKNOWN/failed acquisition.
        attempts = campaign.previous_attempts()
        history_owner = (
            campaign
            if campaign.protocol.get("schema_version") in CONFIRMED_DEFINITION_FAILURE_VERSIONS
            else campaign.parent
        )
        attempted = attempted_acquisition_targets(history_owner, attempts)
        confirmed = campaign.parent.accepted_observations()
        if digest(confirmed) != campaign.protocol["confirmed_observations_checksum"]:
            raise ValueError("renewal_observations_changed")
        prior = set(confirmed) | set(campaign.base.observations)
        candidates = acquisition_ids if separate else selected
        eligible = set(campaign.targets) - attempted - prior
        campaign.targets = tuple(t for t in candidates if t in eligible)
        if separate:
            # One new observation per bounded increment. Remaining admitted
            # checks stay visible in the ledger and may enter a successor.
            campaign.targets = campaign.targets[:1]
        # Keep this window's already-confirmed phase in the stable schedule on
        # restoration. execute_phase reads its exact receipt without dispatch;
        # accepted_observations also needs that target to read the saved fact.
    else:
        campaign.targets = tuple(t for t in campaign.targets if t in revision["added"])
    if len(campaign.targets) > 1 or len(selected) > 2:
        raise ValueError("definition_gate_increment_bound")
    return selected


def revised_acquisition(campaign):
    return (
        campaign.protocol.get("expansion_policy", {}).get("schema_version")
        in REVISED_ACQUISITION_POLICIES
    )


def definition_review_selection(protocol, result, revision, events):
    """A saved interpretation may outlive a zero-wire Review preparation timeout.

    This checks only eligibility shape; stopped_tip and restore_origin_materials
    separately prove original Control/journals and rebuild current definition.
    """
    phases = result.get("phases", [])
    selected = [r["id"] for r in revision.get("replaced", [])]
    if (
        not current_increment(protocol)
        or protocol.get("mode") != "definitions"
        or protocol.get("definition_increment_only") is not True
        or result.get("complete") is not False
        or len(phases) != 1
        or phases[0].get("target") is not None
        or phases[0].get("accepted") is not False
        or phases[0].get("error") != "deadline"
        or any(e.get("phase") in {"reserved", "completed"} for e in events)
        or revision.get("added")
        or not revision.get("accepted")
        or not 1 <= len(selected) <= 2
        or len(set(selected)) != len(selected)
        or any(r["definition"] not in revision["accepted"] for r in revision["replaced"])
    ):
        raise ValueError("renewal_definition_review_requires_zero_wire_timeout")
    return tuple(sorted(selected))


def declined_acquisition_tip(parent, path, protocol, result):
    """One confirmed 429 at the first new observation, never UNKNOWN or a retry loop.

    This is evidence for an explicitly requested successor, not permission to
    revive the stopped run. The original window and remaining budget are kept.
    """
    baseline = getattr(parent, "aggregation_baseline", None) or getattr(
        getattr(parent, "parent", None), "aggregation_baseline", None
    )
    phases = result.get("phases", [])
    if (
        baseline is None
        or protocol.get("schema_version") != "multimodal-campaign-renewal/1.17.0"
        or protocol.get("mode") != "acquisition"
        or protocol.get("acquisition_increment_only") is not True
        or Path(protocol["facts"]).absolute() != baseline.path
        or protocol["accepted_origin_checksum"] != digest(baseline.manifest)
        or protocol["accepted_state_checksum"] != digest(baseline.state)
        or protocol["input"] != parent.base.manifest.checksum
        or protocol["task"] != parent.base.task.checksum
        or result.get("case_code") != parent.base.binding["case_code"]
        or result.get("task") != parent.base.task.model_dump(mode="json")
        or result.get("complete") is not False
        or result.get("previous_attempts") != protocol["previous_attempts"]
        or result.get("new_requests") != 1
        or len(phases) != 1
        or phases[0].get("target") != protocol["acquisition_targets"][0]
        or phases[0].get("accepted")
        or phases[0].get("acceptance_key")
        or (path / "cancel").exists()
    ):
        raise ValueError("renewal_declined_acquisition_identity")
    phase = phases[0]
    docs = parent.documents(phase["run_id"])
    control = parent.sessions.read(docs.run_id)
    if (
        control is None
        or control.phase != "stopped"
        or control.cancelled
        or control.pending
        or docs.read("dynamic-campaign") != (protocol,)
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("multimodal-case-result") != (result,)
    ):
        raise ValueError("renewal_declined_acquisition_not_terminal")
    events = RuntimeAgentJournal(docs).read()
    reservations = [e for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    if len(reservations) != 1 or len(records) != 1 or len(confirmations) != 1:
        raise ValueError("renewal_declined_acquisition_unknown")
    record = records[0]
    diagnostics = [
        e["value"]
        for e in events
        if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
        and e.get("request_id") == record["request_id"]
    ]
    if (
        record.get("error") != "rate_limit"
        or record.get("role") != "vision"
        or record.get("simulated") is not False
        or record.get("api_requests") != 1
        or record.get("retry_index") != 0
        or record.get("output_checksum") is not None
        or record["scope"]["run_id"] != docs.run_id
        or record["scope"]["case_code"] != result["case_code"]
        or record["scope"]["input_checksum"] != protocol["input"]
        or reservations[0]["request_id"] != record["request_id"]
        or confirmations[0]
        != dict(phase="response_confirmation", request_id=record["request_id"], confirmed=True)
        or len(diagnostics) != 1
        or diagnostics[0].get("http_status") != 429
        or diagnostics[0].get("category") not in {"rate_limit", "unclassified_429"}
    ):
        raise ValueError("renewal_declined_acquisition_unknown")
    previous = protocol["previous_attempts"] + [
        dict(directory=str(path), requests=1, checksum=digest(result))
    ]
    return dict(
        path=str(path),
        result=digest(result),
        protocol=digest(protocol),
        journals=[digest(events)],
        previous_attempts=previous,
        run_id=docs.run_id,
        deadline=protocol["deadline"],
        declined_acquisition=dict(
            request_id=record["request_id"],
            journal_checksum=digest(events),
            deadline=protocol["deadline"],
            remaining_requests=protocol["max_new_requests"] - 1,
            targets=protocol["acquisition_targets"],
            old_run_remains_stopped=True,
            further_decline_successors_allowed=False,
        ),
    )


def confirmed_search_failure(revision, operation, outcome):
    """A terminal public-search failure is not a confirmed LLM response.

    Match the original connector receipt in the persisted search backlog. This
    only permits reading a stopped history; it never replays its wire request.
    """
    from packages.grounding.models import validate
    from packages.knowledge_connectors.ports import WireReceipt

    if not outcome or len(outcome) != 1:
        return False
    receipt = validate(WireReceipt, outcome[0]).model_dump(mode="json")
    if receipt["operation"] != "search" or receipt["http_status"] == 200:
        return False
    matches = []
    for rejected in revision.get("rejected", []):
        if rejected.get("reason") != "public_search_incomplete":
            continue
        result = rejected.get("result", {})
        if result.get("status") not in {"failed", "timeout", "partial"}:
            continue
        for run in result.get("runs", []):
            saved = run.get("outcome", {})
            if (
                saved.get("stop_reason") in {"provider_error", "timeout"}
                and saved.get("error_category")
                in {"network", "timeout", "retry_exhausted", "rate_limit", "http"}
                and receipt in saved.get("wire_receipts", [])
            ):
                matches.append(receipt)
    return len(matches) == 1 and operation.get("kind") == "wire"


def stopped_tip(
    parent,
    path,
    *,
    completed_source=False,
    accepted_definition=False,
    alternative_definition=False,
    interrupted=False,
    confirmed_failure=False,
    zero_increment_acquisition=False,
    source_checkpoint=False,
):
    """Original owner proves terminal status; unknown responses stay in history."""
    path = Path(path).absolute()
    if source_checkpoint:
        from .source_checkpoint import inspect_checkpoint

        old = json.loads((path / "protocol.json").read_bytes())
        return inspect_checkpoint(assemble_bound_parent(parent, path, old))[0]
    if confirmed_failure:
        from .confirmed_definition_failure import tip

        return tip(parent, path)
    if interrupted:
        from .interrupted_definition import tip

        return tip(parent, path, local_failure=interrupted == "local_index_failure")
    if (path / "manifest.json").is_file() and not (path / "protocol.json").exists():
        from .accepted_aggregation import aggregation_tip

        return aggregation_tip(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("renewal_tip_symlink")
    result = json.loads((path / "result.json").read_bytes())
    protocol = json.loads((path / "protocol.json").read_bytes())
    if (
        protocol.get("schema_version") == RETAINED_ACQUISITION_VERSION
        and result.get("complete") is False
        and len(result.get("phases", [])) == 1
        and result["phases"][0].get("target") is not None
        and result["phases"][0].get("requests", {}).get("api_requests_observed") == 0
        and result["phases"][0].get("requests", {}).get("api_requests_unknown") == 0
    ):
        from .unknown_acquisition import local_configuration_tip

        return local_configuration_tip(parent, path, protocol, result, ordinary=True)
    if (
        protocol.get("schema_version") == UNKNOWN_ACQUISITION_VERSION
        and result.get("complete") is False
        and result.get("phases")
        and result["phases"][-1].get("target") is not None
    ):
        from .unknown_acquisition import local_configuration_tip

        return local_configuration_tip(parent, path, protocol, result)
    if (
        protocol.get("schema_version") == "multimodal-campaign-renewal/1.19.0"
        and protocol.get("review_acquired_only")
        and result.get("complete") is False
    ):
        from .acquired_review import contract_repair_tip

        return contract_repair_tip(parent, path, protocol, result)
    if (
        protocol.get("schema_version")
        in {"multimodal-campaign-renewal/1.17.0", "multimodal-campaign-renewal/1.18.0"}
        and result.get("complete") is False
        and result.get("phases")
        and result["phases"][-1].get("target") is not None
        and any(p.get("accepted") for p in result["phases"])
    ):
        from .acquired_review import terminal_inventory

        return terminal_inventory(parent, path, protocol, result)
    if (
        protocol.get("schema_version") == "multimodal-campaign-renewal/1.17.0"
        and result.get("complete") is False
        and result.get("phases")
        and result["phases"][-1].get("target") is not None
    ):
        return declined_acquisition_tip(parent, path, protocol, result)
    if (
        result["case_code"] != parent.base.binding["case_code"]
        or result["task"] != parent.base.task.model_dump(mode="json")
        or protocol["input"] != parent.base.manifest.checksum
        or protocol["task"] != parent.base.task.checksum
        or not result["phases"]
        or result["phases"][-1]["target"] is not None
    ):
        raise ValueError("renewal_tip_identity")
    from .input_scope import terminal_dispatch_phases

    accounting_phases = terminal_dispatch_phases(parent, path, protocol, result) + result["phases"]
    if len({p["run_id"] for p in accounting_phases}) != len(accounting_phases):
        raise ValueError("renewal_tip_duplicate_phase")
    count, journals = 0, []
    for phase in accounting_phases:
        docs = parent.documents(phase["run_id"])
        control = parent.sessions.read(phase["run_id"])
        if (
            control is None
            or control.phase != "stopped"
            or control.pending is not None
            or docs.read("dynamic-phase-result") != (phase,)
            or docs.read("dynamic-campaign") != (protocol,)
        ):
            raise ValueError("renewal_tip_not_terminal")
        events = RuntimeAgentJournal(docs).read()
        count += sum(e.get("phase") == "reserved" for e in events)
        journals.append(digest(events))
    definition = None
    terminal_definition_backlog = False
    completed_adopted_increment = False
    if protocol.get("definition_expansion") and protocol.get("mode") != "definitions":
        rid = identity(
            "RUN", "task-research:mm-live:" + digest((path.name, "definition-expansion"))[7:55]
        )
        dd = parent.documents(rid)
        source_events = RuntimeAgentJournal(dd).read()
        reserved = [e["request_id"] for e in source_events if e.get("phase") == "reserved"]
        if len(reserved) != len(set(reserved)):
            raise ValueError("renewal_definition_duplicate_reservation")
        operations = dd.read_many(
            tuple(
                f"definition:operation:{n}"
                for n in range(protocol["definition_expansion"]["max_operations"])
            )
        )
        count += len(reserved) + sum(
            bool(op) and op[0]["kind"] in {"wire", "search", "fetch"} for op in operations
        )
        definition = dict(run_id=rid, journal=digest(source_events), operations=digest(operations))
    if protocol.get("mode") == "definitions":
        rid = identity(
            "RUN", "task-research:mm-live:" + digest((path.name, "definition-expansion"))[7:55]
        )
        dd = parent.documents(rid)
        revision = dd.read("definition:revision")
        completed_zero_increment = bool(
            protocol.get("schema_version")
            in {
                RUNTIME_DEFINITION_VERSION,
                RECORDED_DEFINITION_VERSION,
                *CASE_DEFINITION_VERSIONS,
                *INTERRUPTED_DEFINITION_VERSIONS,
                *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
            }
            and completed_source
            and not accepted_definition
            and result.get("complete") is True
            and protocol.get("definition_increment_only") is True
            and revision
            and revision[0].get("added") == []
            and revision[0].get("replaced") == []
        )
        completed_adopted_increment = bool(
            completed_source
            and not accepted_definition
            and current_increment(protocol)
            and protocol.get("schema_version") != "multimodal-campaign-renewal/1.11.0"
            and result.get("complete") is True
            and protocol.get("definition_increment_only") is True
            and revision
            and valid_completed_increment(revision[0])
        )
        if alternative_definition:
            from .definition_alternative import selection

            if revision is None or (path / "cancel").exists():
                raise ValueError("definition_alternative_current_revision_required")
            selection(protocol, result, revision[0], RuntimeAgentJournal(docs).read())
            for phase in result["phases"]:
                if parent.sessions.read(phase["run_id"]).cancelled:
                    raise ValueError("definition_alternative_parent_cancelled")
        elif accepted_definition:
            if revision is None:
                raise ValueError("renewal_definition_revision_required")
            definition_review_selection(
                protocol, result, revision[0], RuntimeAgentJournal(docs).read()
            )
        elif not (completed_zero_increment or completed_adopted_increment) and (
            revision is None or revision[0]["added"] or revision[0]["accepted"]
        ):
            raise ValueError("renewal_definition_continuation_requires_unstarted_proposal")
        source_events = RuntimeAgentJournal(dd).read()
        source_reserved = [e["request_id"] for e in source_events if e.get("phase") == "reserved"]
        source_records = [e["record"] for e in source_events if e.get("phase") == "completed"]
        if source_reserved and not (
            completed_source or accepted_definition or alternative_definition
        ):
            raise ValueError("renewal_definition_source_already_started")
        corrected_source_proof = None
        if zero_increment_acquisition:
            if not completed_zero_increment:
                raise ValueError("post_definition_acquisition_requires_zero_effective_increment")
            from .definition_alternative import confirmed_sources

            corrected_source_proof = confirmed_sources(parent, dd, source_events, protocol)
        if (completed_source or accepted_definition or alternative_definition) and (
            len(source_reserved) != len(set(source_reserved))
            or len(source_records) != len(source_reserved)
            or set(source_reserved) != {r["request_id"] for r in source_records}
            or any(
                r["role"] != "source"
                or (
                    r["error"] is not None
                    and not (alternative_definition and r["error"] == "schema")
                    and corrected_source_proof is None
                )
                for r in source_records
            )
        ):
            raise ValueError("renewal_definition_source_unconfirmed")
        wires = []
        failed_wires = []
        for n in range(24):
            op = dd.read(f"definition:operation:{n}")
            if op and op[0]["kind"] == "wire":
                outcome = dd.read(op[0]["key"] + ":outcome")
                if not outcome or outcome[0].get("http_status") != 200:
                    if not (
                        completed_zero_increment
                        and confirmed_search_failure(revision[0], op[0], outcome)
                    ):
                        raise ValueError("renewal_definition_wire_unknown")
                    failed_wires.append(op[0]["key"])
                if op[0]["key"] in {w["operation"]["key"] for w in wires}:
                    raise ValueError("renewal_definition_duplicate_wire")
                wires.append(dict(operation=op[0], outcome=outcome))
        count += len(wires) + len(source_reserved)
        definition = dict(run_id=rid, revision=digest(revision[0]), wires=digest(wires))
        if completed_source or accepted_definition or alternative_definition:
            definition["source_journal"] = digest(source_events)
        if corrected_source_proof is not None:
            definition["zero_increment_acquisition_source_records"] = corrected_source_proof
        if accepted_definition:
            if not source_records:
                raise ValueError("renewal_definition_source_origin_missing")
            definition["accepted_interpretation"] = True
        if alternative_definition:
            from .definition_alternative import confirmed_sources

            definition["confirmed_source_records"] = confirmed_sources(
                parent, dd, source_events, protocol
            )
            definition["accepted_alternative"] = True
        if completed_adopted_increment:
            if (
                not source_records
                or failed_wires
                or (path / "cancel").exists()
                or any(
                    getattr(parent.sessions.read(p["run_id"]), "cancelled", False)
                    for p in result["phases"]
                )
            ):
                raise ValueError("renewal_completed_increment_source_missing")
            definition["completed_increment"] = dict(
                schema_version="definition-completed-increment-proof/1.0.0",
                changed_ids=sorted(
                    set(revision[0]["added"]) | {r["id"] for r in revision[0]["replaced"]}
                ),
                replay_authorized=False,
            )
        terminal_definition_backlog = completed_zero_increment and bool(
            revision[0]["accepted"]
            or failed_wires
            or protocol.get("schema_version") in CASE_DEFINITION_VERSIONS
        )
        if terminal_definition_backlog:
            if (
                not source_records
                and protocol.get("schema_version") not in CASE_DEFINITION_VERSIONS
            ):
                raise ValueError("renewal_definition_source_origin_missing")
            definition["terminal_backlog"] = dict(
                schema_version="definition-terminal-backlog-proof/1.0.0",
                failed_public_search_wires=failed_wires,
                no_effective_increment=True,
                replay_authorized=False,
            )
    if (
        count != result["new_requests"]
        or result["previous_attempts"] != protocol["previous_attempts"]
    ):
        raise ValueError("renewal_tip_accounting")
    if docs.read("multimodal-case-result") != (result,):
        raise ValueError("renewal_tip_result")
    previous = result["previous_attempts"] + [
        dict(directory=str(path), requests=count, checksum=digest(result))
    ]
    value = dict(
        path=str(path),
        result=digest(result),
        protocol=digest(protocol),
        journals=journals,
        previous_attempts=previous,
        run_id=docs.run_id,
    )
    if definition:
        value["definition"] = definition
    if terminal_definition_backlog or alternative_definition or completed_adopted_increment:
        # Rebuild through the original definition/source owners. A stored
        # acceptance receipt alone does not authorize current use.
        restore_origin_materials(parent, value)
    return value


def valid_completed_increment(revision):
    """Shape of a committed increment; current medical/source owner checks follow."""
    added = revision.get("added", [])
    replaced = revision.get("replaced", [])
    if not all(isinstance(r, dict) and isinstance(r.get("id"), str) for r in replaced):
        return False
    changed = set(added) | {r["id"] for r in replaced}
    rows = {r["id"]: r for r in revision.get("rows", [])}
    accepted = revision.get("accepted", {})
    return bool(changed) and all(
        t in rows
        and rows[t].get("definition_receipt_key") in accepted
        and rows[t].get("definition_receipt_checksum")
        == digest(accepted[rows[t]["definition_receipt_key"]])
        for t in changed
    )


def pending_definition_acquisition(parent, tip):
    """Current admitted, input-compatible, never-attempted checks from this origin."""
    if tip.get("result_kind") == SOURCE_CHECKPOINT_KIND:
        old = json.loads((Path(tip["path"]) / "protocol.json").read_bytes())
        return pending_definition_acquisition(parent, old["tip"])
    definition = tip.get("definition", {})
    inherited = (
        parent.protocol.get("schema_version") in CASE_DEFINITION_VERSIONS
        and definition.get("terminal_backlog")
        and parent.protocol.get("pending_definition_acquisition")
    )
    if not definition.get("completed_increment") and not inherited:
        return None
    revision = parent.base.definition_current()
    if digest(revision) != tip["definition"]["revision"]:
        raise ValueError("renewal_pending_definition_revision_changed")
    confirmed = parent.accepted_observations()
    attempted = attempted_acquisition_targets(parent, tip["previous_attempts"])
    available = set(MultimodalCampaign.available_targets(parent.base))
    # Include prior admitted checks still in the ledger, not just this round's
    # added list. Existing fixed checks never acquire a definition receipt here.
    targets = sorted(
        r["id"]
        for r in parent.base.ledger
        if r.get("definition_receipt_key")
        and r.get("acquisition_allowed") is True
        and r["id"] in available
        and r["id"] not in attempted
        and r["id"] not in confirmed
        and r["id"] not in parent.base.observations
    )
    return dict(
        schema_version="pending-definition-acquisition/1.0.0",
        source_revision=tip["definition"]["revision"],
        targets=targets,
    )


def check_local_history(plan):
    if plan.get("schema_version") in {
        CAPABILITY_SCOPE_UNKNOWN_VERSION,
        CAPABILITY_SCOPE_CONFIRMED_VERSION,
    }:
        from .scope_unknown_successor import check_history as check_unknown_history

        check_unknown_history(plan)
    from .capability_window import SCOPE_RECOVERY_VERSIONS

    if plan.get("schema_version") in SCOPE_RECOVERY_VERSIONS:
        from .scope_successor import check_history

        check_history(plan)
    tip = Path(plan["tip"]["path"])
    filename = (
        "stages/sources.json"
        if plan.get("schema_version") == SOURCE_CHECKPOINT_VERSION
        else "result.json"
    )
    if plan.get("schema_version") in {
        *INTERRUPTED_DEFINITION_VERSIONS,
        *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
    }:
        validate_definition_runtime(plan)
        filename = "stage-failure.json"
    if (
        digest(json.loads((tip / filename).read_bytes())) != plan["tip"]["result"]
        or digest(
            json.loads(
                (
                    tip
                    / (
                        "manifest.json"
                        if plan["tip"].get("protocol_kind") == "accepted_aggregation_manifest"
                        else "protocol.json"
                    )
                ).read_bytes()
            )
        )
        != plan["tip"]["protocol"]
    ):
        raise ValueError("renewal_history_changed")


def definition_reassessment(parent, tip, request_id):
    """One explicitly acknowledged text reassessment; never recover an UNKNOWN response."""
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_batches import batch_payload, read_plan
    from packages.grounding.models import validate

    old = json.loads((Path(tip["path"]) / "protocol.json").read_bytes())
    if (
        old.get("schema_version") != "multimodal-campaign-renewal/1.13.0"
        or old.get("review_definition_increment") is not True
        or old.get("mode") != "review"
        or len(old.get("review_target_ids", [])) != 1
    ):
        raise ValueError("renewal_definition_reassessment_one_successor_only")
    origin = old["tip"]
    if stopped_tip(parent, origin["path"], accepted_definition=True) != origin:
        raise ValueError("renewal_definition_reassessment_origin_changed")
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(tip["run_id"])
    if control.phase != "stopped" or control.pending is not None or control.cancelled:
        raise ValueError("renewal_definition_reassessment_not_stopped")
    result = docs.read("multimodal-case-result")[0]
    events = RuntimeAgentJournal(docs).read()
    reserved = [e for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    if (
        result.get("complete") is not False
        or result.get("new_requests") != 1
        or len(result["phases"]) != 1
        or result["phases"][0].get("accepted") is not False
        or len(reserved) != 1
        or len(records) != 1
        or reserved[0]["request_id"] != request_id
        or records[0].get("request_id") != request_id
        or records[0].get("role") != "review"
        or records[0].get("error") != "transport"
        or records[0].get("retry_index") != 0
        or records[0].get("api_requests") != 1
        or records[0].get("simulated") is not False
    ):
        raise ValueError("renewal_definition_reassessment_request_binding")
    failures = [
        e.get("value", {}).get("transport_failure", {})
        for e in events
        if e.get("schema_version") == "agent-transport-diagnostic/1.0.0"
    ]
    if len(failures) != 1 or failures[0].get("remote_execution") != "unknown":
        raise ValueError("renewal_definition_reassessment_unknown_proof")
    intent = docs.read("live-stage-intent:review")[0]
    prefix = "review-batches:" + intent["event"]["code"] + ":A"
    plan = read_plan(docs.read(prefix + ":plan")[0])
    state = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if (
        list(plan.assessment_ids) != old["review_target_ids"]
        or len(plan.batches) != 1
        or plan.scope != state.material.scope
        or plan.config_checksum != old["provider"]
        or records[0]["scope"] != state.material.scope.model_dump(mode="json")
        or records[0]["config_checksum"] != plan.config_checksum
        or records[0]["input_checksum"] != digest(batch_payload(state, plan, plan.batches[0]))
        or docs.read("accepted:" + intent["event"]["code"] + ":domain") is not None
    ):
        raise ValueError("renewal_definition_reassessment_parent_binding")
    slot = prefix + ":" + plan.batches[0].id
    attempt = docs.read(slot + ":attempt:" + request_id)
    if (
        not attempt
        or attempt[0].get("error") != "transport"
        or attempt[0].get("raw_model_text")
        or attempt[0].get("record_checksum") != digest(records[0])
        or docs.read(slot + ":result") is not None
    ):
        raise ValueError("renewal_definition_reassessment_response_must_be_recovered")
    return dict(
        source_origin=origin,
        source_revision_checksum=old["source_revision_checksum"],
        logical_campaign_id=old["logical_campaign_id"],
        review_target_ids=old["review_target_ids"],
        request_id=request_id,
        request_checksum=digest(records[0]),
        journal_checksum=digest(events),
        parent_checksum=digest(state),
        remote_execution="unknown",
        duplicate_cost_possible=True,
        original_request_replayed=False,
    )


def unstarted_context_proof(parent, result, tip):
    """Original stopped Review with no request or accepted judgment, not UNKNOWN replay."""
    phase = result["phases"][-1]
    docs = parent.documents(tip["run_id"])
    control = parent.sessions.read(tip["run_id"])
    events = RuntimeAgentJournal(docs).read()
    stage = docs.read("live-stage:review")
    safe_stop = stage is None or (
        len(stage) == 1
        and stage[0].get("schema_version") == "research-review-result/3.0.0"
        and stage[0].get("run_id") == tip["run_id"]
        and stage[0].get("status") == "safe_stop"
        and stage[0].get("reason") == "deadline"
        and stage[0].get("domain_status") == "not_accepted"
        and stage[0].get("acceptance_key") is None
        and stage[0].get("new_medical_review") is False
        and stage[0].get("cancelled") is False
        and stage[0].get("recorded_agent_invocations") == 0
        and stage[0].get("requests") == phase.get("requests")
        and stage[0].get("event", {}).get("role") == "review_control"
        and any(
            ref.get("role") == "research_review"
            and ref.get("code") == stage[0]["event"].get("code")
            and ref.get("version") == stage[0]["schema_version"]
            and ref.get("checksum") == digest(stage[0])
            for ref in phase.get("planner", {}).get("result_refs", [])
        )
    )
    if (
        parent.protocol.get("schema_version") == "multimodal-campaign-renewal/1.25.0"
        or getattr(parent.base, "execution_profile", None)
        not in {"relay-snapshot-context-v1", "relay-root-question-v1"}
        or result["complete"]
        or phase["target"] is not None
        or phase["accepted"]
        or phase["acceptance_key"] is not None
        or phase["error"] != "deadline"
        or phase["run_id"] != tip["run_id"]
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or datetime.now(UTC) < control.binding.deadline
        or any(e.get("phase") in {"reserved", "completed"} for e in events)
        or docs.read("dynamic-phase-result") != (phase,)
        or not safe_stop
        or (Path(tip["path"]) / "cancel").exists()
    ):
        raise ValueError("renewal_unstarted_context_not_zero_wire_deadline")
    return dict(
        run_id=tip["run_id"],
        phase_checksum=digest(phase),
        journal_checksum=digest(events),
        stopped_review_checksum=digest(stage),
        control_checksum=digest(control),
        phase_deadline=control.binding.deadline.isoformat(),
        original_window_deadline=parent.protocol["deadline"],
        provider_checksum=digest(parent.config),
        new_vision_requests=0,
        new_public_search_allowed=False,
    )


def validate_unstarted_context(plan, parent, result):
    if (
        plan["schema_version"] != "multimodal-campaign-renewal/1.25.0"
        or plan["mode"] != "review"
        or plan["facts"] != plan["tip"]["path"]
        or plan["provider"] != digest(parent.config)
        or plan["input"] != parent.base.manifest.checksum
        or plan["task"] != parent.base.task.checksum
        or plan["previous_attempts"] != plan["tip"]["previous_attempts"]
        or plan["prior_deadline"] != parent.protocol["deadline"]
        or datetime.fromisoformat(plan["started_at"])
        < datetime.fromisoformat(plan["unstarted_context_review"]["phase_deadline"])
        or plan["seconds"] != 2700
        or datetime.fromisoformat(plan["deadline"])
        != datetime.fromisoformat(plan["started_at"]) + timedelta(seconds=2700)
        or not 2 <= plan["max_new_requests"] <= 8
        or plan["max_model_dispatches"] != plan["max_new_requests"] - 1
        or sum(r["requests"] for r in plan["previous_attempts"]) + plan["max_new_requests"] > 40
        or plan.get("acquisition_targets") != []
        or plan.get("new_vision_requests") != 0
        or plan.get("new_public_search_allowed") is not False
        or plan.get("expansion_policy") is not None
        or plan["unstarted_context_review"] != unstarted_context_proof(parent, result, plan["tip"])
    ):
        raise ValueError("renewal_unstarted_context_binding")


class RenewedCampaign(MultimodalCampaign):
    def execution_session(self):
        from .execution_lease import (
            AGGREGATE_VERSION,
            PATH,
            PROFILE,
            read,
            session,
            validate_aggregate,
        )

        if self.protocol.get("schema_version") in {
            DEFINITION_ALTERNATIVE_VERSION,
            *CAPABILITY_VERSIONS,
        }:
            if self.protocol.get("execution_lease_profile") != PROFILE:
                raise ValueError("definition_alternative_execution_lease_profile")
            return session(self)
        if (
            self.protocol.get("schema_version")
            in {AGGREGATE_VERSION, *AGGREGATE_ALTERNATIVE_VERSIONS}
            and "execution_lease_profile" in self.protocol
        ):
            validate_aggregate(self)
            return session(self)

        if self.protocol.get("schema_version") in {
            RUNTIME_DEFINITION_VERSION,
            RECORDED_DEFINITION_VERSION,
            *CASE_DEFINITION_VERSIONS,
            *INTERRUPTED_DEFINITION_VERSIONS,
            *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
        }:
            validate_definition_runtime(self.protocol)
            return session(self)
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.45.0",
            "multimodal-campaign-renewal/1.46.0",
            "multimodal-campaign-renewal/1.69.0",
            "multimodal-campaign-renewal/1.72.0",
            UNSTARTED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.74.0",
            UNSCORED_ACQUISITION_VERSION,
            TRUNCATED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.54.0",
            "multimodal-campaign-renewal/1.55.0",
            "multimodal-campaign-renewal/1.57.0",
            "multimodal-campaign-renewal/1.65.0",
        } or (
            self.protocol.get("schema_version") in RETAINED_ACQUISITION_VERSIONS
            and "execution_lease_profile" in self.protocol
        ):
            if self.protocol.get("execution_lease_profile") != PROFILE:
                raise ValueError("context_review_execution_lease_profile")
            return session(self)
        if not (self.output / PATH).exists():
            return super().execution_session()
        read(self)
        return session(self)

    def cumulative_request_limit(self):
        if self.protocol.get("schema_version") == "multimodal-campaign-renewal/1.36.0":
            from .confirmed_review_window import method_cumulative_limit

            return method_cumulative_limit(self.protocol)
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.35.0",
            "multimodal-campaign-renewal/1.40.0",
        }:
            from .review_method_window import cumulative_limit

            return cumulative_limit(self.protocol)
        return super().cumulative_request_limit()

    def __init__(self, parent, output, plan):
        if plan.get("schema_version") in {
            RUNTIME_DEFINITION_VERSION,
            RECORDED_DEFINITION_VERSION,
            *CASE_DEFINITION_VERSIONS,
            *INTERRUPTED_DEFINITION_VERSIONS,
            *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
        }:
            validate_definition_runtime(plan)
            if plan.get("schema_version") == RECORDED_DEFINITION_VERSION:
                recorded_definition_check(parent, plan)
        elif (plan.get("expansion_policy") or {}).get(
            "schema_version"
        ) in RUNTIME_DEFINITION_POLICIES:
            raise ValueError("definition_runtime_policy_requires_new_window")
        truncated = plan.get("schema_version") == TRUNCATED_REVIEW_VERSION
        if bool(plan.get("truncated_review")) != truncated:
            raise ValueError("truncated_review_protocol_version")
        if truncated:
            from .truncated_review import check

            truncated_config = check(parent, plan)
        unscored = plan.get("schema_version") == UNSCORED_ACQUISITION_VERSION
        if bool(plan.get("unscored_acquisition")) != unscored:
            raise ValueError("unscored_protocol_version")
        if unscored:
            from .unscored_acquisition import check

            unscored_config = check(parent, plan)
        if bool(plan.get("runtime_pending_review")) != (
            plan.get("schema_version") == "multimodal-campaign-renewal/1.54.0"
        ):
            raise ValueError("pending_runtime_window_version")
        if bool(plan.get("pending_reassessment_window")) != (
            plan.get("schema_version")
            in {
                "multimodal-campaign-renewal/1.55.0",
                "multimodal-campaign-renewal/1.57.0",
                "multimodal-campaign-renewal/1.65.0",
            }
        ):
            raise ValueError("pending_reassessment_window_version")
        if bool(plan.get("pending_new_window")) != (
            plan.get("schema_version")
            in {
                "multimodal-campaign-renewal/1.39.0",
                "multimodal-campaign-renewal/1.41.0",
                "multimodal-campaign-renewal/1.42.0",
            }
        ):
            raise ValueError("pending_unstarted_window_version")
        if bool(plan.get("initial_failed_review_window")) != (
            plan.get("schema_version") == "multimodal-campaign-renewal/1.41.0"
        ):
            raise ValueError("initial_failed_window_version")
        recovery_config = (
            recovery_configuration(parent.config, plan)
            if "recovery_network_route" in plan or "recovery_http_client" in plan
            else None
        )
        if bool(plan.get("alternative_view")) != (
            plan["schema_version"]
            in (ALTERNATIVE_ACQUISITION_VERSIONS | AGGREGATE_ALTERNATIVE_VERSIONS)
        ):
            raise ValueError("alternative_view_explicit_version_required")
        definition_alternative = plan.get("schema_version") == DEFINITION_ALTERNATIVE_VERSION
        if bool(plan.get("definition_alternative")) != definition_alternative:
            raise ValueError("definition_alternative_explicit_version")
        if definition_alternative:
            from .definition_alternative import check

            check(parent, plan)
        base = (
            definition_phase_base(parent.base, original_requirements=True)
            if definition_alternative
            else parent.base
        )
        if plan.get("schema_version") in CAPABILITY_VERSIONS:
            from .capability_window import install_base

            base = install_base(parent, output, plan)
        if plan.get("schema_version") == CAPACITY_REVIEW_VERSION:
            from copy import copy

            from .review_capacity_recovery import check
            from .review_capacity_recovery import configuration as capacity_configuration

            check(parent, plan)
            base = copy(base)
            base.review_target_ids = tuple(plan["review_target_ids"])
            base.review_context_scope = RETAINED_CONTEXT_SCOPE
            base.confirmed_review_reader = accepted_reader(
                parent.parent,
                parent.protocol["tip"]["run_id"],
                recorded_origin=parent.protocol.get("recorded_review_origin"),
                capacity_config=capacity_configuration(
                    parent.config, plan["review_capacity"]["required_batches"]
                ),
            )
        if plan.get("root_direction_review") or plan.get("schema_version") in {
            "multimodal-campaign-renewal/1.29.0",
            "multimodal-campaign-renewal/1.30.0",
        }:
            from copy import copy

            from .root_review_window import check

            check(plan, parent, json.loads((Path(plan["facts"]) / "result.json").read_bytes()))
            base = copy(base)
            base.confirmed_review_reader = None
            if plan.get("schema_version") not in {
                "multimodal-campaign-renewal/1.35.0",
                "multimodal-campaign-renewal/1.40.0",
            }:
                base.review_clarification = None
            base.review_target_ids = tuple(plan["review_target_ids"])
            base.root_direction_review = plan["root_direction_review"]["question"]
        if (
            plan.get("shape_contract_review")
            or plan.get("schema_version") == "multimodal-campaign-renewal/1.28.0"
        ):
            from copy import copy

            from .review_shape_window import check

            check(plan, parent, json.loads((Path(plan["facts"]) / "result.json").read_bytes()))
            base = copy(base)
            base.confirmed_review_reader = None
            base.review_clarification = None
            base.review_target_ids = tuple(plan["review_target_ids"])
        if plan.get("same_profile_review") or plan.get("schema_version") in {
            "multimodal-campaign-renewal/1.27.0",
            "multimodal-campaign-renewal/1.31.0",
            "multimodal-campaign-renewal/1.36.0",
        }:
            from .confirmed_review_window import check

            check(plan, parent, json.loads((Path(plan["facts"]) / "result.json").read_bytes()))
            if plan.get("schema_version") == "multimodal-campaign-renewal/1.36.0":
                from copy import copy

                # Installing the historical reader must not mutate its own parent.
                # Preserve the parent's root question and per-item clarification.
                base = copy(base)
        if (
            plan.get("review_acquisition_gap")
            or plan.get("schema_version") in ACQUISITION_REVIEW_VERSIONS
        ):
            from copy import copy

            from .acquired_review import validate_projection_repair

            validate_projection_repair(parent, plan)
            base = copy(base)
            base.review_target_ids = tuple(plan["review_target_ids"])
            retained = plan["schema_version"] == RETAINED_REVIEW_VERSION
            base.review_clarification = (
                parent.protocol.get("review_clarification") if retained else None
            )
            if retained:
                base.review_context_scope = RETAINED_CONTEXT_SCOPE
            base.confirmed_review_reader = accepted_reader(
                parent.parent,
                parent.protocol["tip"]["run_id"],
                recorded_origin=parent.protocol.get("recorded_review_origin"),
                **({"network_config": recovery_config} if recovery_config is not None else {}),
            )
        if plan.get("acquisition_selection") or plan.get("schema_version") in {
            "multimodal-campaign-renewal/1.26.0",
            *AGGREGATE_ALTERNATIVE_VERSIONS,
            *EXACT_ACQUISITION_VERSIONS,
        }:
            validate_acquisition_selection(parent, plan)
        if (
            plan.get("unstarted_context_review")
            or plan.get("schema_version") == "multimodal-campaign-renewal/1.25.0"
        ):
            result = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
            validate_unstarted_context(plan, parent, result)
        if plan.get("review_acquired_only"):
            from copy import copy

            from .acquired_review import bounds

            proof = plan["tip"].get("acquired_review")
            if (
                plan.get("schema_version")
                not in {"multimodal-campaign-renewal/1.19.0", "multimodal-campaign-renewal/1.20.0"}
                or plan.get("mode") != "review"
                or not proof
                or plan["facts"] != plan["tip"]["path"]
                or plan["review_target_ids"] != proof["selected_ids"]
                or plan["deadline"] != proof["deadline"]
                or plan["max_new_requests"] > proof["remaining_requests"]
                or plan.get("acquisition_targets") != []
                or plan.get("expansion_policy") is not None
            ):
                raise ValueError("acquired_review_plan_binding")
            selected_config = parent.config
            if plan.get("schema_version") == "multimodal-campaign-renewal/1.20.0":
                from packages.llm_provider.config import qualified_citation_upgrade

                selected_config = qualified_citation_upgrade(parent.config)
                if (
                    not plan.get("repair_acquired_contract")
                    or not plan["tip"].get("contract_repair")
                    or digest(selected_config) != plan["tip"]["contract_repair"]["new_provider"]
                ):
                    raise ValueError("acquired_contract_repair_plan")
            elif plan.get("repair_acquired_contract") or plan["tip"].get("contract_repair"):
                raise ValueError("acquired_contract_repair_version")
            # Verify originally granted bounds even during historical reads after expiry.
            _, _, _, dispatches = bounds(
                plan["tip"],
                selected_config,
                plan["max_new_requests"],
                now=datetime.fromisoformat(plan["started_at"]),
            )
            if plan["max_model_dispatches"] != dispatches:
                raise ValueError("acquired_review_dispatch_binding")
            base = copy(base)
            base.confirmed_review_reader = None
            base.review_clarification = None
            base.review_target_ids = tuple(proof["selected_ids"])
        if plan.get("acquisition_increment_only"):
            from copy import copy

            baseline = getattr(parent, "aggregation_baseline", None)
            if (
                plan.get("schema_version")
                not in {
                    "multimodal-campaign-renewal/1.17.0",
                    "multimodal-campaign-renewal/1.18.0",
                    "multimodal-campaign-renewal/1.26.0",
                    *AGGREGATE_ALTERNATIVE_VERSIONS,
                }
                or plan.get("mode") != "acquisition"
                or baseline is None
                or digest(baseline.state) != plan["accepted_state_checksum"]
                or digest(baseline.manifest) != plan["accepted_origin_checksum"]
                or plan.get("review_target_ids") != plan.get("acquisition_targets")
            ):
                raise ValueError("renewal_aggregation_increment_binding")
            base = copy(base)
            base.confirmed_review_reader = None
            base.review_target_ids = tuple(plan["review_target_ids"])
            self.progress_accepted_reader = baseline
            if plan.get("schema_version") == "multimodal-campaign-renewal/1.18.0":
                origin = plan["tip"].get("declined_acquisition")
                if (
                    not origin
                    or plan["deadline"] != origin["deadline"]
                    or plan["max_new_requests"] > origin["remaining_requests"]
                    or plan["acquisition_targets"] != origin["targets"]
                ):
                    raise ValueError("renewal_declined_acquisition_bounds")
        if plan.get("schema_version") in {
            "multimodal-campaign-renewal/1.15.0",
            "multimodal-campaign-renewal/1.16.0",
            "multimodal-campaign-renewal/1.23.0",
            "multimodal-campaign-renewal/1.24.0",
            "multimodal-campaign-renewal/1.39.0",
            "multimodal-campaign-renewal/1.41.0",
            "multimodal-campaign-renewal/1.42.0",
            "multimodal-campaign-renewal/1.54.0",
            "multimodal-campaign-renewal/1.55.0",
            "multimodal-campaign-renewal/1.57.0",
            "multimodal-campaign-renewal/1.65.0",
        }:
            from copy import copy

            from .pending_review import inventory

            pending = plan.get("pending_review", {})
            if plan["schema_version"] == "multimodal-campaign-renewal/1.41.0":
                from .pending_review import validate_initial_failed_window

                validate_initial_failed_window(plan, parent)
            elif plan["schema_version"] in {
                "multimodal-campaign-renewal/1.39.0",
                "multimodal-campaign-renewal/1.42.0",
            }:
                from .pending_review import validate_unstarted_window

                validate_unstarted_window(plan, parent)
            elif plan.get("pending_new_window"):
                raise ValueError("pending_unstarted_window_version")
            if plan["schema_version"] == "multimodal-campaign-renewal/1.23.0":
                validate_context_window(plan, parent)
            if plan["schema_version"] == "multimodal-campaign-renewal/1.54.0":
                from .pending_review import validate_runtime_pending_window

                validate_runtime_pending_window(plan, parent)
            elif plan.get("runtime_pending_review"):
                raise ValueError("pending_runtime_window_version")
            if plan["schema_version"] in {
                "multimodal-campaign-renewal/1.55.0",
                "multimodal-campaign-renewal/1.57.0",
                "multimodal-campaign-renewal/1.65.0",
            }:
                from .pending_review import validate_reassessment_window

                validate_reassessment_window(plan, parent)
            if plan["schema_version"] == "multimodal-campaign-renewal/1.24.0":
                validate_format_window(plan, parent)
            elif pending.get("acknowledged_format_requests"):
                raise ValueError("renewal_known_format_requires_explicit_window")
            if (
                plan.get("mode") != "review"
                or not plan.get("review_increment_only")
                or plan.get("review_target_ids") != pending.get("selected_ids")
                or inventory(
                    parent,
                    plan["tip"],
                    pending.get("acknowledge_unknown_request"),
                    pending.get("acknowledged_format_requests", ()),
                    initial_partition=plan["schema_version"]
                    == "multimodal-campaign-renewal/1.57.0",
                    historical_origin=(pending.get("historical_unknown_origin") or {}).get("path"),
                )
                != pending
            ):
                raise ValueError("renewal_pending_review_binding")
            base = copy(base)
            base.confirmed_review_reader = None
            base.review_clarification = None
            base.review_target_ids = tuple(pending["selected_ids"])
            if plan["schema_version"] in {
                "multimodal-campaign-renewal/1.57.0",
                "multimodal-campaign-renewal/1.65.0",
            }:
                # The old owner was rechecked above for partition exclusion.
                # This new increment does not install its judgments/context.
                base.review_context_scope = None
        elif plan.get("pending_review") or plan.get("review_increment_only"):
            raise ValueError("renewal_pending_review_version")
        if plan.get("review_definition_increment"):
            from copy import copy

            if (
                plan.get("schema_version")
                not in {
                    "multimodal-campaign-renewal/1.13.0",
                    "multimodal-campaign-renewal/1.14.0",
                }
                or plan.get("mode") != "review"
                or not plan.get("definition_increment_only")
                or not plan.get("definition_origin", plan["tip"])
                .get("definition", {})
                .get("accepted_interpretation")
            ):
                raise ValueError("renewal_definition_review_scope")
            base = copy(base)
            base.confirmed_review_reader = None
            base.review_clarification = None
            if plan.get("schema_version") == "multimodal-campaign-renewal/1.14.0" and (
                plan.get("max_model_dispatches") != 1
                or plan.get("max_new_requests") != 2
                or plan.get("seconds") != 3600
                or definition_reassessment(parent, plan["tip"], plan["reassessment"]["request_id"])
                != plan["reassessment"]
                or plan["definition_origin"] != plan["reassessment"]["source_origin"]
            ):
                raise ValueError("renewal_definition_reassessment_plan")
        if plan.get("schema_version") == "multimodal-campaign-renewal/1.12.0" or (
            plan.get("schema_version")
            in {
                "multimodal-campaign-renewal/1.26.0",
                *AGGREGATE_ALTERNATIVE_VERSIONS,
                *EXACT_ACQUISITION_VERSIONS,
            }
            and not plan.get("acquisition_increment_only")
        ):
            from copy import copy

            if (
                plan.get("mode") != "acquisition"
                or plan.get("review_partition") != "owner_bound_reused_and_new/1.0.0"
            ):
                raise ValueError("renewal_partitioned_acquisition_binding")
            base = copy(base)
        elif plan.get("recorded_review_origin") is not None:
            raise ValueError("renewal_recorded_origin_requires_new_protocol")
        if current_increment(plan):
            from copy import copy

            if (
                plan.get("mode") != "definitions"
                or not plan.get("definition_increment_only")
                or plan.get("expansion_policy", {}).get("schema_version")
                not in CURRENT_INCREMENT_POLICIES
            ):
                raise ValueError("renewal_current_increment_binding")
            validate_increment_window(
                plan["expansion_policy"], seconds=plan["seconds"], cap=plan["max_new_requests"]
            )
            if plan.get("schema_version") in CASE_DEFINITION_VERSIONS and (
                pending_definition_acquisition(parent, plan["tip"])
                != plan["pending_definition_acquisition"]
            ):
                raise ValueError("renewal_pending_definition_binding")
            # Same original input/Knowledge/Runtime owners and immutable facts.
            # Only this window's phase selectors differ. Do not inherit an old
            # accepted-review reader into a newly revised definition scope.
            base = definition_phase_base(base)
        if unscored or truncated:
            from copy import copy

            base = copy(base)
            base.confirmed_review_reader = None
            base.review_clarification = None
            base.review_target_ids = tuple(plan["review_target_ids"])
        super().__init__(base, parent.target, campaign_id=Path(output).name)
        self.parent, self.output, self.protocol = parent, Path(output), plan
        if plan.get("schema_version") == SOURCE_CHECKPOINT_VERSION:
            self.logical_campaign_id = plan["logical_campaign_id"]
        if plan.get("schema_version") in {
            *INTERRUPTED_DEFINITION_VERSIONS,
            *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
        }:
            self.logical_campaign_id = plan["logical_campaign_id"]
        self.config = configuration(
            resilient_vision=True, directional=True, exact_primary_status=True
        )
        if plan.get("continuation"):
            self.logical_campaign_id = plan["continuation"]["campaign_root_id"]
            self.config = compact_configuration()
        if plan.get("reuse_confirmed_review"):
            self.config = compact_configuration()
            origin = plan.get("confirmed_review_origin", plan["tip"])
            if plan.get("repair_review_format"):
                from packages.llm_provider.config import review_format_upgrade

                self.config = review_format_upgrade(parent.config)
            else:
                restore_origin_materials(parent, origin)
            if plan.get("same_profile_review"):
                self.config = parent.config
            self.base.confirmed_review_reader = historical_reader(
                parent,
                origin["run_id"],
                format_config=self.config if plan.get("repair_review_format") else None,
                preserve_prompt=bool(plan.get("same_profile_review")),
            )
            if plan.get("repair_review_format"):
                self.base.confirmed_review_reader.expected_format_origin = plan[
                    "format_origin_checksum"
                ]
        self.targets = self.initial_targets = ()
        if plan.get("pending_review") or plan.get("review_acquired_only"):
            self.config = parent.config
            if plan.get("repair_acquired_contract"):
                from packages.llm_provider.config import qualified_citation_upgrade

                self.config = qualified_citation_upgrade(parent.config)
        if plan["mode"] == "definitions":
            self.base.definition_expansion_policy = plan["expansion_policy"]
        if plan.get("definition_increment_only"):
            self.config = parent.config if current_increment(plan) else compact_configuration()
            self.base.review_target_ids = ()
        if plan.get("confirmed_facts_review"):
            self.config = confirmed_facts_configuration()
        if plan.get("unstarted_context_review"):
            self.config = parent.config
        if plan.get("review_definition_increment"):
            self.config = parent.config
            definition_origin = plan.get("definition_origin", plan["tip"])
            self.logical_campaign_id = Path(definition_origin["path"]).name
            restore_origin_materials(self, definition_origin)
            selected = tuple(sorted(r["id"] for r in self.base.definition_revision["replaced"]))
            if (
                list(selected) != plan.get("review_target_ids")
                or self.base.definition_revision["added"]
                or digest(self.base.definition_revision) != plan.get("source_revision_checksum")
                or self.logical_campaign_id != plan.get("logical_campaign_id")
            ):
                raise ValueError("renewal_definition_review_projection_changed")
            self.base.review_target_ids = selected
            self.targets = self.initial_targets = ()
        if definition_alternative:
            self.logical_campaign_id = plan["logical_campaign_id"]
            restore_origin_materials(self, plan["tip"])
            self.base.review_target_ids = tuple(plan["review_target_ids"])
        if plan["mode"] == "acquisition":
            self.config = parent.config
            if plan["schema_version"] in EXACT_ACQUISITION_VERSIONS:
                # The parent's text-only selector is not this acquisition's
                # Review scope. Only accepted history plus this bounded slice.
                self.base.review_target_ids = acquisition_review_ids(plan)
            if (
                not plan.get("acquisition_increment_only")
                and not unscored
                and not definition_alternative
                and plan.get("schema_version") not in CAPABILITY_VERSIONS
            ):
                from .unknown_acquisition import fact_review_run

                self.base.confirmed_review_reader = accepted_reader(
                    parent,
                    fact_review_run(parent, plan),
                    recorded_origin=plan.get("recorded_review_origin"),
                    client_config=recovery_config if "recovery_http_client" in plan else None,
                )
                if (
                    digest(self.base.confirmed_review_reader.state_reader())
                    != plan["accepted_state_checksum"]
                ):
                    raise ValueError("renewal_accepted_state_changed")
            if plan["schema_version"] in RETAINED_ACQUISITION_VERSIONS:
                self.base.review_context_scope = RETAINED_CONTEXT_SCOPE
            self.targets = self.initial_targets = tuple(plan["acquisition_targets"])
            self.reused_review_count = len(plan["reuse_assessment_ids"])
        if plan.get("review_acquisition_gap"):
            self.config = parent.config
            self.reused_review_count = len(plan["reuse_assessment_ids"])
        if plan.get("review_clarification"):
            self.config = parent.config
            self.base.confirmed_review_reader = accepted_reader(parent, plan["tip"]["run_id"])
            self.base.review_clarification = plan["review_clarification"]
            self.reused_review_count = len(plan["reuse_assessment_ids"])
        if plan.get("shape_contract_review"):
            from .review_shape_window import configuration as shape_configuration

            self.config = shape_configuration(parent.config)
        if plan.get("root_direction_review"):
            from .root_review_window import configuration_for

            self.config = configuration_for(parent.config, plan)
        if "recovery_network_route" in plan or "recovery_http_client" in plan:
            self.config = recovery_config
        if unscored:
            self.config = unscored_config
        if truncated:
            self.config = truncated_config
        if plan.get("schema_version") == CAPACITY_REVIEW_VERSION:
            from .review_capacity_recovery import configuration as capacity_configuration

            self.config = capacity_configuration(
                parent.config, plan["review_capacity"]["required_batches"]
            )
            self.reused_review_count = len(plan["reuse_assessment_ids"])
        if plan.get("alternative_view"):
            from .alternative_acquisition import AlternativeView

            self.base.alternative_view = AlternativeView(parent, plan, self.output)
        if plan.get("schema_version") in CAPABILITY_VERSIONS:
            from .capability_window import install_methods

            if plan["schema_version"] in {
                CAPABILITY_ROUTED_VERSION,
                CAPABILITY_SCOPE_UNKNOWN_VERSION,
                CAPABILITY_SCOPE_CONFIRMED_VERSION,
                CAPABILITY_CURRENT_SCOPE_RECOVERY_VERSION,
            }:
                from .capability_window import route_configuration

                self.config = route_configuration(parent, plan.get("route_transition"))
            self.logical_campaign_id = plan["logical_campaign_id"]
            install_methods(self)

    def previous_attempts(self):
        check_local_history(self.protocol)
        return self.protocol["previous_attempts"]

    def scoped_acquisition_inventory(self):
        """A scope result cannot broaden the original bounded renewal plan."""
        if self.protocol["mode"] != "acquisition":
            return ()
        allowed = frozenset(self.protocol["acquisition_targets"])
        return tuple(t for t in super().scoped_acquisition_inventory() if t in allowed)

    def work_deadline(self):
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.54.0",
            "multimodal-campaign-renewal/1.55.0",
            "multimodal-campaign-renewal/1.57.0",
            "multimodal-campaign-renewal/1.65.0",
        } and getattr(self, "_final_review_active", False):
            return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(seconds=60)
        if self.protocol.get("schema_version") in {
            UNSCORED_ACQUISITION_VERSION,
            TRUNCATED_REVIEW_VERSION,
            *CAPABILITY_VERSIONS,
        }:
            reserve = 60 if getattr(self, "_final_review_active", False) else 600
            return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(seconds=reserve)
        if self.protocol.get("schema_version") in {
            *ACQUISITION_REVIEW_VERSIONS,
            CAPACITY_REVIEW_VERSION,
        }:
            return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(seconds=300)
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.1.0",
            "multimodal-campaign-renewal/1.2.0",
            "multimodal-campaign-renewal/1.3.0",
            "multimodal-campaign-renewal/1.4.0",
            "multimodal-campaign-renewal/1.5.0",
            "multimodal-campaign-renewal/1.6.0",
            "multimodal-campaign-renewal/1.7.0",
            "multimodal-campaign-renewal/1.8.0",
            "multimodal-campaign-renewal/1.9.0",
            "multimodal-campaign-renewal/1.10.0",
            "multimodal-campaign-renewal/1.11.0",
            RUNTIME_DEFINITION_VERSION,
            RECORDED_DEFINITION_VERSION,
            *CASE_DEFINITION_VERSIONS,
            *INTERRUPTED_DEFINITION_VERSIONS,
            *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
            "multimodal-campaign-renewal/1.12.0",
            "multimodal-campaign-renewal/1.13.0",
            "multimodal-campaign-renewal/1.14.0",
            "multimodal-campaign-renewal/1.15.0",
            "multimodal-campaign-renewal/1.16.0",
            "multimodal-campaign-renewal/1.17.0",
            "multimodal-campaign-renewal/1.18.0",
            "multimodal-campaign-renewal/1.19.0",
            "multimodal-campaign-renewal/1.20.0",
            "multimodal-campaign-renewal/1.21.0",
            "multimodal-campaign-renewal/1.44.0",
            "multimodal-campaign-renewal/1.22.0",
            "multimodal-campaign-renewal/1.45.0",
            "multimodal-campaign-renewal/1.46.0",
            "multimodal-campaign-renewal/1.69.0",
            "multimodal-campaign-renewal/1.72.0",
            UNSTARTED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.74.0",
            "multimodal-campaign-renewal/1.23.0",
            "multimodal-campaign-renewal/1.24.0",
            "multimodal-campaign-renewal/1.39.0",
            "multimodal-campaign-renewal/1.41.0",
            "multimodal-campaign-renewal/1.42.0",
            "multimodal-campaign-renewal/1.25.0",
            "multimodal-campaign-renewal/1.26.0",
            *AGGREGATE_ALTERNATIVE_VERSIONS,
            "multimodal-campaign-renewal/1.27.0",
            "multimodal-campaign-renewal/1.28.0",
            "multimodal-campaign-renewal/1.29.0",
            "multimodal-campaign-renewal/1.30.0",
            "multimodal-campaign-renewal/1.31.0",
            *EXACT_ACQUISITION_VERSIONS,
            "multimodal-campaign-renewal/1.35.0",
            "multimodal-campaign-renewal/1.40.0",
            "multimodal-campaign-renewal/1.36.0",
        } and getattr(self, "_final_review_active", False):
            # Review consumes the acquisition reserve; leave one minute for saving.
            # Never extend the absolute campaign deadline or change old protocols.
            return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(seconds=60)
        return super().work_deadline()

    @contextmanager
    def phase_context(self, target):
        """Share the existing deadline with outer guards; nested calls restore it."""
        previous = getattr(self, "_final_review_active", False)
        self._final_review_active = target is None
        try:
            yield
        finally:
            self._final_review_active = previous

    def execute_phase(self, target=None):
        if (
            self.protocol.get("schema_version") in CAPABILITY_CURRENT_VERSIONS
            and target in self.protocol["input_scope"]["targets"]
            and target not in self.protocol["input_scope"]["dispatch_targets"]
        ):
            raise ValueError("input_scope_reused_target_dispatched")
        from .capability_window import SCOPE_RECOVERY_VERSIONS

        if self.protocol.get("schema_version") in SCOPE_RECOVERY_VERSIONS and target in {
            r["target"] for r in self.protocol["scope_reuse"]["reused"]
        }:
            raise ValueError("scope_successor_confirmed_target_no_dispatch")
        with self.phase_context(target):
            alternative = getattr(self.base, "alternative_view", None)
            if alternative is not None and target == alternative.target:
                self.guard()
                alternative.ensure()
                self.guard()
            result = super().execute_phase(target)
            if getattr(self, "protocol", {}).get("mode") == "acquisition" and target is not None:
                from .report import export_acquisition_progress

                print(
                    json.dumps(
                        dict(progress=export_acquisition_progress(self, result)), ensure_ascii=False
                    ),
                    flush=True,
                )
            return result

    def accepted_observations(self):
        values = self.parent.accepted_observations()
        if digest(values) != self.protocol["confirmed_observations_checksum"]:
            raise ValueError("renewal_observations_changed")
        if self.protocol.get("alternative_view"):
            # Retain the prior limitation in the immutable method proof; do not
            # present its old image as a result of the new view.
            values = {
                k: v for k, v in values.items() if k != self.protocol["alternative_view"]["target"]
            }
        return {**values, **super().accepted_observations()}

    @historical_fact_read_scope()
    @_projection_visible_reads()
    def freeze(self):
        if self.protocol.get("schema_version") in CAPABILITY_VERSIONS:
            from .capability_review_origin import resolve
            from .capability_window import check

            original = json.loads((self.output / "protocol.json").read_bytes())
            expected = (
                resolve(self.parent, original)
                if self.protocol["schema_version"] == CAPABILITY_VERSION
                else original
            )
            if expected != self.protocol:
                raise ValueError("capability_review_effective_plan_changed")
            check(self.parent, self.protocol)
        if self.protocol.get("schema_version") == SOURCE_CHECKPOINT_VERSION:
            from .source_checkpoint import recheck

            recheck(self)
        if self.protocol.get("schema_version") == "multimodal-campaign-renewal/1.74.0":
            from .local_review_recovery import recheck

            recheck(self)
        if self.protocol.get("schema_version") == UNSCORED_ACQUISITION_VERSION:
            from .unscored_acquisition import check

            if check(self.parent, self.protocol) != self.config:
                raise ValueError("unscored_provider_changed")
        if self.protocol.get("schema_version") in UNSTARTED_WINDOW_VERSIONS:
            from .unstarted_window import recheck

            recheck(self)
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.21.0",
            "multimodal-campaign-renewal/1.44.0",
        }:
            from .provider_transition import recheck

            recheck(self)
        expected = self.protocol["implementation"]
        if (self.output / "unscored-progress-repair.json").exists():
            from .unscored_acquisition import read_progress_repair

            expected = read_progress_repair(self)["implementation_after"]
        if (self.output / "execution-lease.json").exists():
            from .execution_lease import read as read_execution_lease

            expected = read_execution_lease(self)["implementation_after"]
        amendment_path = self.output / "prewire-repair.json"
        if amendment_path.exists():
            amendment = json.loads(amendment_path.read_bytes())
            docs = self.documents(amendment["run_id"])
            if (
                amendment["protocol"] != digest(self.protocol)
                or amendment["implementation_before"] != expected
                or docs.read("renewal:prewire-repair") != (amendment,)
                or digest(json.loads(Path(amendment["proof_path"]).read_bytes()))
                != amendment["proof"]
            ):
                raise ValueError("renewal_prewire_amendment_binding")
            expected = amendment["implementation_after"]
        if (self.output / "archive-repair.json").exists():
            from .archive_recovery import read

            amendment = read(self)
            expected = amendment["implementation_after"]
            if amendment["schema_version"] in {
                "confirmed-material-local-repair/1.5.0",
                "confirmed-material-local-repair/1.6.0",
            }:
                self.local_material_repair = amendment
            if amendment.get("local_operation_limit"):
                self.definition_operation_limit = amendment["local_operation_limit"]
                self.local_control_amendment = digest(amendment)
        if (self.output / "definition-review-repair.json").exists():
            from .stages import read_definition_review_repair

            expected = read_definition_review_repair(self)["implementation_after"]
        if expected != freeze() or self.protocol["provider"] != digest(self.config):
            raise ValueError("renewal_implementation_changed")
        if (
            stopped_tip(
                self.parent,
                self.protocol["tip"]["path"],
                completed_source=bool(
                    self.protocol.get("reuse_confirmed_review")
                    or self.protocol.get("retrieval_revision")
                    or self.protocol.get("root_direction_review")
                    or self.protocol.get("schema_version") == POST_DEFINITION_ACQUISITION_VERSION
                    or self.protocol.get("schema_version") in CAPABILITY_VERSIONS
                ),
                accepted_definition=bool(self.protocol.get("review_definition_increment")),
                zero_increment_acquisition=self.protocol.get("schema_version")
                == POST_DEFINITION_ACQUISITION_VERSION,
                alternative_definition=self.protocol.get("schema_version")
                == DEFINITION_ALTERNATIVE_VERSION,
                interrupted=(
                    "local_index_failure"
                    if self.protocol.get("confirmed_local_index_failure")
                    else self.protocol.get("schema_version") in INTERRUPTED_DEFINITION_VERSIONS
                ),
                source_checkpoint=self.protocol.get("schema_version") == SOURCE_CHECKPOINT_VERSION,
                confirmed_failure=self.protocol.get("schema_version")
                in CONFIRMED_DEFINITION_FAILURE_VERSIONS,
            )
            != self.protocol["tip"]
        ):
            raise ValueError("renewal_owner_history_changed")
        if self.protocol.get("schema_version") in CASE_DEFINITION_VERSIONS and (
            pending_definition_acquisition(self.parent, self.protocol["tip"])
            != self.protocol["pending_definition_acquisition"]
        ):
            raise ValueError("renewal_pending_definition_binding")
        origin = self.protocol.get("confirmed_review_origin")
        if origin and stopped_tip(self.parent, origin["path"], completed_source=True) != origin:
            raise ValueError("renewal_confirmed_origin_changed")
        proof = self.protocol.get("reassessment")
        if (
            proof
            and definition_reassessment(self.parent, self.protocol["tip"], proof["request_id"])
            != proof
        ):
            raise ValueError("renewal_definition_reassessment_changed")
        if self.protocol.get("pending_review"):
            from .pending_review import inventory

            pending = self.protocol["pending_review"]
            if (
                inventory(
                    self.parent,
                    self.protocol["tip"],
                    pending.get("acknowledge_unknown_request"),
                    pending.get("acknowledged_format_requests", ()),
                    initial_partition=self.protocol["schema_version"]
                    == "multimodal-campaign-renewal/1.57.0",
                    historical_origin=(pending.get("historical_unknown_origin") or {}).get("path"),
                )
                != pending
            ):
                raise ValueError("renewal_pending_review_origin_changed")
            if self.protocol.get("schema_version") in {
                "multimodal-campaign-renewal/1.55.0",
                "multimodal-campaign-renewal/1.57.0",
                "multimodal-campaign-renewal/1.65.0",
            }:
                from .pending_review import validate_reassessment_window

                validate_reassessment_window(self.protocol, self.parent)
        if self.protocol.get("schema_version") == TRUNCATED_REVIEW_VERSION:
            from .truncated_review import check

            check(self.parent, self.protocol)
        if self.protocol.get("unstarted_context_review"):
            result = json.loads((Path(self.protocol["facts"]) / "result.json").read_bytes())
            validate_unstarted_context(self.protocol, self.parent, result)
        if self.protocol.get("same_profile_review"):
            from .confirmed_review_window import check

            check(
                self.protocol,
                self.parent,
                json.loads((Path(self.protocol["facts"]) / "result.json").read_bytes()),
            )
        if self.protocol.get("shape_contract_review"):
            from .review_shape_window import check

            check(
                self.protocol,
                self.parent,
                json.loads((Path(self.protocol["facts"]) / "result.json").read_bytes()),
            )
        if self.protocol.get("root_direction_review"):
            from .root_review_window import check

            check(
                self.protocol,
                self.parent,
                json.loads((Path(self.protocol["facts"]) / "result.json").read_bytes()),
            )
        return self.protocol

    def budget_factory(self, **kwargs):
        if self.protocol.get("schema_version") in CONFIRMED_DEFINITION_FAILURE_VERSIONS:
            kwargs["bounded_phase"] = True
        if self.protocol.get("schema_version") in {
            *ACQUISITION_REVIEW_VERSIONS,
            CAPACITY_REVIEW_VERSION,
            *CAPABILITY_VERSIONS,
            UNSCORED_ACQUISITION_VERSION,
            TRUNCATED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.54.0",
            "multimodal-campaign-renewal/1.55.0",
            "multimodal-campaign-renewal/1.57.0",
            "multimodal-campaign-renewal/1.65.0",
        }:
            kwargs["bounded_phase"] = True
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.15.0",
            "multimodal-campaign-renewal/1.16.0",
            "multimodal-campaign-renewal/1.18.0",
            "multimodal-campaign-renewal/1.19.0",
            "multimodal-campaign-renewal/1.20.0",
            "multimodal-campaign-renewal/1.21.0",
            "multimodal-campaign-renewal/1.22.0",
            "multimodal-campaign-renewal/1.45.0",
            "multimodal-campaign-renewal/1.46.0",
            "multimodal-campaign-renewal/1.69.0",
            "multimodal-campaign-renewal/1.72.0",
            UNSTARTED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.74.0",
            "multimodal-campaign-renewal/1.23.0",
            "multimodal-campaign-renewal/1.24.0",
            "multimodal-campaign-renewal/1.39.0",
            "multimodal-campaign-renewal/1.41.0",
            "multimodal-campaign-renewal/1.42.0",
            "multimodal-campaign-renewal/1.25.0",
            "multimodal-campaign-renewal/1.27.0",
            "multimodal-campaign-renewal/1.28.0",
            "multimodal-campaign-renewal/1.29.0",
            "multimodal-campaign-renewal/1.30.0",
            "multimodal-campaign-renewal/1.31.0",
            *EXACT_ACQUISITION_VERSIONS,
            "multimodal-campaign-renewal/1.35.0",
            "multimodal-campaign-renewal/1.40.0",
            "multimodal-campaign-renewal/1.36.0",
        }:
            # Original Planner narrows this phase to the campaign deadline minus
            # closure reserve. Keep that shorter window, never invent 2700 seconds.
            kwargs["bounded_phase"] = True
        budget = super().budget_factory(**kwargs)
        if self.protocol.get("schema_version") in CONFIRMED_DEFINITION_FAILURE_VERSIONS:
            previous_reserve = budget.before_reserve

            def bounded_reserve(checksum):
                if self.usage() >= self.protocol["max_new_requests"]:
                    raise ProviderError("budget")
                return previous_reserve(checksum)

            budget.before_reserve = bounded_reserve
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.14.0",
            "multimodal-campaign-renewal/1.15.0",
            "multimodal-campaign-renewal/1.16.0",
            "multimodal-campaign-renewal/1.18.0",
            "multimodal-campaign-renewal/1.19.0",
            "multimodal-campaign-renewal/1.20.0",
            "multimodal-campaign-renewal/1.21.0",
            "multimodal-campaign-renewal/1.22.0",
            "multimodal-campaign-renewal/1.45.0",
            "multimodal-campaign-renewal/1.46.0",
            "multimodal-campaign-renewal/1.69.0",
            "multimodal-campaign-renewal/1.72.0",
            UNSTARTED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.74.0",
            "multimodal-campaign-renewal/1.23.0",
            "multimodal-campaign-renewal/1.24.0",
            "multimodal-campaign-renewal/1.39.0",
            "multimodal-campaign-renewal/1.41.0",
            "multimodal-campaign-renewal/1.42.0",
            "multimodal-campaign-renewal/1.25.0",
            "multimodal-campaign-renewal/1.27.0",
            "multimodal-campaign-renewal/1.28.0",
            "multimodal-campaign-renewal/1.29.0",
            "multimodal-campaign-renewal/1.30.0",
            "multimodal-campaign-renewal/1.31.0",
            *EXACT_ACQUISITION_VERSIONS,
            *ACQUISITION_REVIEW_VERSIONS,
            CAPACITY_REVIEW_VERSION,
            UNSCORED_ACQUISITION_VERSION,
            *CAPABILITY_VERSIONS,
            TRUNCATED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.35.0",
            "multimodal-campaign-renewal/1.54.0",
            "multimodal-campaign-renewal/1.55.0",
            "multimodal-campaign-renewal/1.57.0",
            "multimodal-campaign-renewal/1.65.0",
            "multimodal-campaign-renewal/1.40.0",
            "multimodal-campaign-renewal/1.36.0",
        }:
            original = budget.before_reserve

            def before_reserve(checksum):
                # Existing orchestration headroom allows post-response checks;
                # Dispatch cap excludes headroom reserved solely for local closure.
                if self.usage() >= self.protocol["max_model_dispatches"]:
                    raise ProviderError("budget")
                return original(checksum)

            budget.before_reserve = before_reserve
        return budget

    @property
    def request_finalization_headroom(self):
        return (
            0
            if self.protocol.get("schema_version")
            in {
                *EXACT_ACQUISITION_VERSIONS,
                RETAINED_REVIEW_VERSION,
                CAPACITY_REVIEW_VERSION,
                *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
            }
            else 1
        )

    def guard_control(self, *, dispatch=False):
        # Exact-cap local checks retain current input/source/cancel/deadline and
        # cumulative accounting. Only before_reserve authorizes another request.
        super().guard_control(dispatch=dispatch or not self.request_finalization_headroom)

    def guard(self, *, dispatch=False):
        if self.protocol.get("schema_version") == "multimodal-campaign-renewal/1.74.0":
            from .local_review_recovery import recheck

            recheck(self)
        if (self.output / "cancel").exists():
            raise ProviderError("cancelled")
        if self.protocol.get("schema_version") in {
            UNSCORED_ACQUISITION_VERSION,
            TRUNCATED_REVIEW_VERSION,
        }:
            if (Path(self.protocol["facts"]) / "cancel").exists():
                raise ProviderError("cancelled")
        if self.protocol.get("schema_version") in UNSTARTED_WINDOW_VERSIONS:
            from .unstarted_window import recheck

            recheck(self)
        used, cap = self.usage(), self.protocol["max_new_requests"]
        if used > cap or (self.request_finalization_headroom and not dispatch and used >= cap):
            raise ProviderError("budget")
        super().guard(dispatch=dispatch)
        if self.protocol.get("schema_version") in UNKNOWN_ACQUISITION_VERSIONS:
            from .unknown_acquisition import check, check_network

            check(self.parent, self.protocol)
            check_network(self.parent.config)

    def expand_definitions(self, **kwargs):
        if self.protocol["mode"] in {"review", "acquisition"}:
            self.base.check()
            return self.base.definition_revision
        revision = super().expand_definitions(**kwargs)
        if revision.get("schema_version") == "definition-stage-checkpoint/1.0.0":
            return revision
        # Historical policies schedule only additions. Explicit new policies
        # also consume an accepted revision of a still-unattempted check; all
        # other unmet checks remain outside this bounded increment.
        added = set(revision["added"])
        if not revised_acquisition(self):
            self.targets = tuple(t for t in self.targets if t in added)
        if self.protocol.get("definition_increment_only"):
            selected = gate_selection(self, revision)
            if not selected and not self.targets:
                write_once(
                    self.output / "increment-gate-stop.json",
                    dict(
                        reason="no_effective_increment",
                        revision_checksum=digest(revision),
                        requests=self.usage(),
                        whole_case_rescored=False,
                    ),
                )
        write_once(
            self.output / "increment-schedule.json",
            dict(
                revision_checksum=digest(revision),
                added=revision["added"],
                scheduled=list(self.targets),
                missing_input=[
                    r["id"] for r in revision["rows"] if r["id"] in added and not r["component"]
                ],
                no_old_observation_reexecution=True,
                **(
                    dict(review_ids=list(selected), whole_case_rescored=False)
                    if self.protocol.get("definition_increment_only")
                    else {}
                ),
                **(
                    dict(
                        inherited_pending=self.protocol["pending_definition_acquisition"],
                        acquisition_and_review_qualification_separate=True,
                    )
                    if self.protocol.get("schema_version") in CASE_DEFINITION_VERSIONS
                    else {}
                ),
            ),
        )
        return revision


def compact_configuration():
    """Explicit smaller text batch; never changes old requests or medical semantics."""
    from packages.llm_provider.config import read_provider_config

    raw = configuration(
        resilient_vision=True, directional=True, exact_primary_status=True
    ).model_dump(mode="json")
    raw["review_batching"]["batch_size"] = 2
    for role in raw["roles"]:
        if role["role"] in {"source", "review"}:
            role["max_output_tokens"] = 2048
    return read_provider_config(raw)


def confirmed_facts_configuration():
    """One item per real text request, explicit bounded longer worker window.

    Does not change old requests, prompts, model or acceptance/scoring semantics.
    All unmet requirements still get their original local missing evaluation.
    """
    from packages.llm_provider.config import read_provider_config

    raw = compact_configuration().model_dump(mode="json")
    raw["review_batching"]["batch_size"] = 1
    raw["transport"]["timeout_seconds"] = 300
    raw["transport"]["max_retries"] = 0
    for role in raw["roles"]:
        if role["role"] == "review":
            role["max_output_tokens"] = 4096
    return read_provider_config(raw)


def historical_reader(parent, run_id, *, format_config=None, preserve_prompt=False):
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_continuation import ConfirmedReviewReader
    from packages.grounding.models import validate
    from packages.llm_provider.models import read_request_record

    docs = parent.documents(run_id)

    class OriginalRequests:
        journal = RuntimeAgentJournal(docs)

        @property
        def used(self):
            return sum(e.get("phase") == "reserved" for e in self.journal.read())

        def records(self):
            return tuple(
                read_request_record(e["record"])
                for e in self.journal.read()
                if e.get("phase") == "completed"
            )

    def state():
        value = docs.read("dynamic-phase-input")
        if not value:
            raise ValueError("renewal_origin_state_missing")
        current = validate(DomainState, value[0])
        if format_config is not None or preserve_prompt:
            return current
        prompt = next(
            r.prompt_checksum for r in compact_configuration().roles if r.role == "review"
        )
        return validate(DomainState, current.model_copy(update={"prompt_checksum": prompt}))

    reader_type = ConfirmedReviewReader
    if (
        preserve_prompt
        and format_config is None
        and getattr(parent, "protocol", {}).get("schema_version")
        in {"multimodal-campaign-renewal/1.35.0", "multimodal-campaign-renewal/1.40.0"}
    ):
        from packages.evidence_details.review_continuation import AcceptedPartialConfirmedReader

        reader_type = AcceptedPartialConfirmedReader
    return reader_type(
        documents=docs, requests=OriginalRequests(), state_reader=state, format_config=format_config
    )


def recorded_acquisition_origin(parent, run_id):
    """Exact original owner confirmation, not the failed outer phase flag."""
    from agents.expert_agent.committed_review import read_committed_review
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.recorded_finalization import RESULT_KEY, original_failure
    from packages.grounding.models import validate

    docs = parent.documents(run_id)
    old = original_failure(docs)
    closed = docs.read(RESULT_KEY)
    if not closed or len(closed) != 1:
        raise ValueError("renewal_recorded_closure_required")
    closed = closed[0]
    state = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if closed.get("schema_version") == "multimodal-recorded-finalization-result/1.1.0":
        from types import SimpleNamespace

        from packages.evidence_details.recorded_finalization import (
            RecordedBatchRequests,
            read_saved_acceptance,
        )

        def current():
            parent.base.check()
            value = validate(DomainState, docs.read("dynamic-phase-input")[0])
            if value.config_checksum != digest(parent.config):
                raise ValueError("renewal_accepted_provider_changed")
            return value

        proof = read_saved_acceptance(
            owner=SimpleNamespace(
                current=current,
                confirmed_review_reader=getattr(parent.base, "confirmed_review_reader", None),
            ),
            documents=docs,
            requests=RecordedBatchRequests(RuntimeAgentJournal(docs)),
            config=parent.config,
        )
        if (
            closed.get("complete") is not True
            or closed.get("new_requests") != 0
            or closed.get("original_requests") != old["new_requests"]
            or closed.get("original_result_checksum") != digest(old)
            or closed.get("finalization_grant_checksum") != proof["grant_checksum"]
            or closed.get("phases")
            != [
                *old["phases"][:-1],
                dict(
                    old["phases"][-1],
                    accepted=True,
                    acceptance_key=proof["acceptance_key"],
                    error=None,
                    evidence_limited=True,
                ),
            ]
        ):
            raise ValueError("renewal_recorded_local_closure_binding")
        return dict(
            schema_version="renewal-recorded-origin/1.1.0",
            closure_checksum=digest(closed),
            original_result_checksum=digest(old),
            confirmation=proof,
        )
    reader = getattr(getattr(parent, "base", None), "confirmed_review_reader", None)
    proof = read_committed_review(
        docs,
        parent=state,
        config=parent.config,
        **({"confirmed_review_reader": reader} if reader is not None else {}),
    )
    if (
        proof is None
        or closed.get("schema_version") != "multimodal-recorded-finalization-result/1.2.0"
        or closed.get("complete") is not True
        or closed.get("new_requests") != 0
        or closed.get("original_requests") != old["new_requests"]
        or closed.get("original_result_checksum") != digest(old)
        or closed.get("local_review_closure", {}).get("confirmation") != proof
        or closed.get("phases")
        != [
            *old["phases"][:-1],
            dict(
                old["phases"][-1],
                accepted=True,
                acceptance_key=proof["acceptance_key"],
                error=None,
                evidence_limited=True,
            ),
        ]
    ):
        raise ValueError("renewal_recorded_closure_binding")
    return dict(
        schema_version="renewal-recorded-origin/1.0.0",
        closure_checksum=digest(closed),
        original_result_checksum=digest(old),
        confirmation=proof,
    )


def recorded_definition_origin(parent, original, local_finalization):
    """A verified local closure enables a NEW definition window, never old replay."""
    if original.get("complete") is not False or not original.get("phases"):
        raise ValueError("recorded_definition_failed_original_required")
    path = Path(local_finalization).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("recorded_definition_closure_symlink")
    proof = recorded_acquisition_origin(parent, original["phases"][-1]["run_id"])
    closed = json.loads((path / "result.json").read_bytes())
    if (
        digest(closed) != proof["closure_checksum"]
        or digest(original) != proof["original_result_checksum"]
    ):
        raise ValueError("recorded_definition_closure_binding")
    return dict(
        schema_version="recorded-definition-origin/1.0.0",
        local_finalization=str(path),
        proof=proof,
    )


def recorded_definition_check(parent, plan):
    original = json.loads((Path(plan["facts"]) / "result.json").read_bytes())
    expected = plan["recorded_definition_origin"]
    if (
        plan["facts"] != plan["tip"]["path"]
        or digest(original) != plan["original_fact_result"]
        or recorded_definition_origin(parent, original, expected["local_finalization"]) != expected
    ):
        raise ValueError("recorded_definition_origin_changed")


def accepted_reader(
    parent,
    run_id,
    lineage=(),
    *,
    recorded_origin=None,
    network_config=None,
    capacity_config=None,
    client_config=None,
):
    """Read accepted phase identity through Runtime, not exported model answers."""
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.accepted_review_reuse import (
        AcceptedReviewReader,
        RecordedAcceptedReviewReader,
    )
    from packages.grounding.models import validate
    from packages.llm_provider.models import read_request_record

    docs = parent.documents(run_id)
    result = docs.read("dynamic-phase-result")
    local = recorded_origin is not None and recorded_origin.get("schema_version") == (
        "renewal-recorded-origin/1.1.0"
    )

    def check_origin():
        if not local:
            if recorded_acquisition_origin(parent, run_id) != recorded_origin:
                raise ValueError("renewal_recorded_origin_changed")
            return
        from packages.evidence_details.recorded_finalization import RESULT_KEY, original_failure

        closed = docs.read(RESULT_KEY)
        proof = recorded_origin["confirmation"]
        if (
            not closed
            or len(closed) != 1
            or digest(closed[0]) != recorded_origin["closure_checksum"]
            or digest(original_failure(docs)) != recorded_origin["original_result_checksum"]
            or proof.get("schema_version") != "recorded-review-acceptance-proof/1.0.0"
            or proof.get("run_id") != run_id
        ):
            raise ValueError("renewal_recorded_origin_changed")
        # Identity only, not authorization. The reader's public current() below
        # verifies the entire grant/requests/acceptance proof exactly once; its
        # state_reader must not recursively invoke that full proof again.

    if recorded_origin is not None:
        check_origin()
        acceptance_key = recorded_origin["confirmation"]["acceptance_key"]
    elif not result or not result[0]["accepted"] or result[0]["target"] is not None:
        raise ValueError("renewal_accepted_review_required")
    else:
        acceptance_key = result[0]["acceptance_key"]

    class OriginalRequests:
        journal = RuntimeAgentJournal(docs)

        def records(self):
            return tuple(
                read_request_record(e["record"])
                for e in self.journal.read()
                if e.get("phase") == "completed"
            )

    def state():
        parent.base.check()
        if recorded_origin is not None:
            check_origin()
        current = validate(DomainState, docs.read("dynamic-phase-input")[0])
        if current.config_checksum != digest(parent.config):
            raise ValueError("renewal_accepted_provider_changed")
        prompt = next(r.prompt_checksum for r in parent.config.roles if r.role == "review")
        return validate(DomainState, current.model_copy(update={"prompt_checksum": prompt}))

    def ancestor(rid, chain):
        # Select only an already owner-verified predecessor, never a directory
        # nominated by the caller or an arbitrary historical accepted object.
        current = parent
        route_config = (
            recovery_configuration(parent.parent.config, parent.protocol)
            if parent.protocol.get("schema_version") == RETAINED_REVIEW_VERSION
            and "recovery_network_route" in parent.protocol
            else None
        )
        if parent.protocol.get("schema_version") in {
            CAPABILITY_ROUTED_VERSION,
            CAPABILITY_SCOPE_UNKNOWN_VERSION,
            CAPABILITY_SCOPE_CONFIRMED_VERSION,
            CAPABILITY_CURRENT_SCOPE_RECOVERY_VERSION,
        }:
            from .capability_window import route_configuration

            route_config = route_configuration(
                parent.parent, parent.protocol.get("route_transition")
            )
        route_options = {"network_config": route_config} if route_config is not None else {}
        if parent.protocol.get("schema_version") == CAPACITY_REVIEW_VERSION:
            route_options["capacity_config"] = parent.config
        if "recovery_http_client" in parent.protocol:
            route_options["client_config"] = recovery_configuration(
                parent.parent.config, parent.protocol
            )
        for _ in range(5):
            protocol = getattr(current, "protocol", {})
            # A zero-dispatch recovery may stop on a later Vision run while
            # retaining a separately committed Review. Resolve its frozen fact
            # identity; accepted_reader still verifies the original closure.
            fact_run = protocol.get("accepted_fact_run_id", protocol.get("tip", {}).get("run_id"))
            if protocol.get("recorded_review_origin") and fact_run == rid:
                return accepted_reader(
                    current.parent,
                    rid,
                    chain,
                    recorded_origin=protocol["recorded_review_origin"],
                    **route_options,
                )
            if protocol.get("repair_review_format") and protocol["tip"]["run_id"] == rid:
                reader = historical_reader(current.parent, rid, format_config=current.config)
                reader.expected_format_origin = protocol["format_origin_checksum"]
                return reader
            if protocol.get("schema_version") in {
                "multimodal-campaign-renewal/1.31.0",
                "multimodal-campaign-renewal/1.36.0",
            } and (protocol["tip"]["run_id"] == rid):
                # The accepted successor reused confirmed (not yet accepted)
                # batches. Resolve its bound original owner, not another accepted
                # result or an arbitrary response in the database.
                return historical_reader(current.parent, rid, preserve_prompt=True)
            # Runtime is shared: a later campaign can read an ancestor's run
            # and can have the same provider checksum. That does not make it
            # the owner of that run. Bind the original review identity first,
            # so nested reads retain its explicit route/continuation contract.
            output = getattr(current, "output", None)
            expected_run = (
                identity("RUN", "task-research:mm-live:" + digest((output.name, "review"))[7:55])
                if output is not None
                else None
            )
            origin = (
                current.documents(rid).read("dynamic-phase-input") if rid == expected_run else None
            )
            if origin and origin[0]["config_checksum"] == digest(current.config):
                result = current.documents(rid).read("dynamic-phase-result")
                if result and result[0]["accepted"]:
                    return accepted_reader(current, rid, chain, **route_options)
            current = getattr(current, "parent", None)
            if current is None:
                break
        raise ValueError("renewal_accepted_ancestor_unbound")

    reader_type = RecordedAcceptedReviewReader if local else AcceptedReviewReader
    return reader_type(
        documents=docs,
        requests=OriginalRequests(),
        state_reader=state,
        acceptance_key=acceptance_key,
        ancestor_reader=ancestor,
        lineage=lineage,
        network_configs=(parent.config, network_config) if network_config is not None else None,
        capacity_configs=(parent.config, capacity_config) if capacity_config is not None else None,
        client_configs=(parent.config, client_config) if client_config is not None else None,
        **(
            {"config": parent.config, "saved_proof": recorded_origin["confirmation"]}
            if local
            else {}
        ),
    )


def restore_origin_materials(parent, tip):
    """Read the exact prior source revision, never requery or rename its aliases."""
    from types import SimpleNamespace

    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
    from packages.grounding.models import validate
    from packages.llm_provider.models import Scope

    from .definition_expansion import DefinitionExpansion

    if "definition" not in tip:
        return
    prior = json.loads((Path(tip["path"]) / "protocol.json").read_bytes())
    installed = getattr(getattr(parent.base, "definition_current", None), "__self__", None)
    if (
        isinstance(installed, DefinitionExpansion)
        and installed.campaign.base is parent.base
        and installed.documents.run_id == tip["definition"]["run_id"]
        and installed.campaign.protocol == prior
    ):
        # Revalidate through the original owner. Its original_rows remain the
        # pre-revision ledger; rebuilding from the installed rows changes the
        # parent binding on the second check. This never caches authorization.
        revision = installed.read_revision()
        if digest(revision) != tip["definition"]["revision"]:
            raise ValueError("renewal_source_revision_changed")
        installed.install(revision)
        parent.base.definition_current = installed.read_revision
        restore_definition_schedule(parent, tip, installed.campaign.targets, revision)
        return
    docs = parent.documents(tip["definition"]["run_id"])
    records = [
        e["record"] for e in RuntimeAgentJournal(docs).read() if e.get("phase") == "completed"
    ]
    if not records and prior.get("schema_version") not in CASE_DEFINITION_VERSIONS:
        raise ValueError("renewal_source_origin_missing")
    if records:
        scope = validate(Scope, records[0]["scope"])
    else:
        from packages.evidence_details.acceptance_models import DomainState

        state = docs.read("dynamic-phase-input")
        if not state or len(state) != 1:
            raise ValueError("renewal_source_origin_missing")
        scope = validate(DomainState, state[0]).material.scope
        if scope.run_id != docs.run_id:
            raise ValueError("renewal_source_scope_binding")
    original = parent.base.sources.reader.original
    sources = KnowledgeDefinitionSources(
        repository=original.knowledge,
        store=original.knowledge_objects,
        index_keys=prior["expansion_policy"]["index_keys"],
        failure_policy=prior["expansion_policy"].get("search_failure_policy"),
    )
    historical = SimpleNamespace(
        base=parent.base,
        protocol=prior,
        campaign_id=Path(tip["path"]).name,
        logical_campaign_id=prior.get("logical_campaign_id")
        or prior.get("continuation", {}).get("campaign_root_id", Path(tip["path"]).name),
        select_targets=parent.select_targets,
    )
    if prior.get("schema_version") in CASE_DEFINITION_VERSIONS:
        origin = parent if str(parent.output) == prior["facts"] else parent.parent
        if str(origin.output) != prior["facts"]:
            raise ValueError("definition_inherited_owner_binding")
        historical.parent = origin
    expansion = DefinitionExpansion(
        historical,
        documents=docs,
        sources=sources,
        provider=None,
        budget=None,
        scope=scope,
        policy=prior["expansion_policy"],
    )
    revision = expansion.read_revision()
    if digest(revision) != tip["definition"]["revision"]:
        raise ValueError("renewal_source_revision_changed")
    expansion.install(revision)
    parent.base.definition_current = expansion.read_revision
    restore_definition_schedule(parent, tip, historical.targets, revision)


def restore_definition_schedule(parent, tip, targets, revision):
    """Cold readback retains this window's confirmed observation schedule."""
    if (
        parent.protocol.get("schema_version") in CASE_DEFINITION_VERSIONS
        and str(parent.output) == tip["path"]
    ):
        parent.targets = tuple(targets)
        gate_selection(parent, revision)


def continuation_bounds(old, used, now):
    if old.get("mode") != "definitions" or old.get("continuation"):
        raise ValueError("renewal_definition_one_continuation_only")
    until = datetime.fromisoformat(old["deadline"])
    cap = old["max_new_requests"] - used
    if now >= until - timedelta(seconds=300) or not 1 <= cap < old["max_new_requests"]:
        raise ValueError("renewal_definition_remaining_budget")
    return until, cap


def review_repair_bounds(old, result, now):
    if (
        old.get("schema_version") != "multimodal-campaign-renewal/1.2.0"
        or old.get("reuse_confirmed_review") is not True
        or result.get("new_requests") != 0
        or result.get("complete") is not False
        or len(result.get("phases", [])) != 1
        or result["phases"][0].get("error") != "review_continuation_current_binding"
        or result["phases"][0].get("target") is not None
    ):
        raise ValueError("renewal_review_repair_requires_zero_request_binding_failure")
    until = datetime.fromisoformat(old["deadline"])
    if now >= until - timedelta(seconds=300):
        raise ValueError("renewal_review_repair_remaining_time")
    return until, min(2, old["max_new_requests"])


def unattempted_targets(parent, attempted, confirmed, *, full_inventory=False, owner_methods=False):
    """A text-only successor does not erase its verified predecessor's plan."""
    if full_inventory and owner_methods:
        owner = getattr(parent.base, "active_owner", None)
        if owner is not None:
            owner.check()
        return [
            target
            for target in MultimodalCampaign.available_targets(parent.base)
            if target not in attempted and target not in confirmed
        ]
    if full_inventory:
        # Exact old selection/1.0 meaning for existing frozen plans.
        return [
            row["id"]
            for row in parent.base.ledger
            if row["id"] not in attempted
            and row["id"] not in confirmed
            and row.get("component") is not None
            and row.get("acquisition_method") == "vision"
            and row.get("acquisition_allowed", True)
        ]
    declared, ancestor = [], parent
    seen = set()
    for _ in range(5):
        if id(ancestor) in seen:
            raise ValueError("renewal_acquisition_history_cycle")
        seen.add(id(ancestor))
        declared.extend(t for t in ancestor.initial_targets if t not in declared)
        ancestor = getattr(ancestor, "parent", None)
        if ancestor is None:
            break
    if ancestor is not None:
        raise ValueError("renewal_acquisition_history_depth")
    current_rows = {r["id"]: r for r in parent.base.ledger}
    return [
        t
        for t in declared
        if t not in attempted
        and t not in confirmed
        and t in current_rows
        and current_rows[t]["component"] is not None
        and current_rows[t]["acquisition_method"] == "vision"
    ]


def verify_routed_source_history(parent, attempt):
    """A history label is not authority: recheck its original Source owner."""
    from .discovery_continuation import ConfirmedDefinitionContinuation

    ancestor, seen = parent, set()
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if ancestor is None or id(ancestor) in seen:
            break
        seen.add(id(ancestor))
        owner = getattr(ancestor, "definition_continuation", None)
        if isinstance(owner, ConfirmedDefinitionContinuation) and owner.routed_backlog:
            if attempt in owner.check_attempts():
                return
        ancestor = getattr(ancestor, "parent", None)
    raise ValueError("renewal_routed_source_history_owner")


def attempted_acquisition_targets(parent, attempts):
    """Read typed immutable history; Source failures are not acquisition results.

    Their original Runtime failure and absence of any acquisition/Review context
    must still agree. An UNKNOWN Source is retained, never replayed or counted as
    an attempted visual examination. Unrecognized history fails closed.
    """
    attempted = set()
    source_kinds = {
        "confirmed_source_schema_failure": "confirmed_response_or_domain_refusal",
        # Emitted by the original confirmed-backlog continuation after it
        # verifies a closed Source history. It is not a visual attempt and
        # grants no permission to replay that Source request.
        "confirmed_source_schema_deferred": "confirmed_response_or_domain_refusal",
        "terminal_source_unknown_deferred": "dispatched_unknown",
    }
    for attempt in attempts:
        path = Path(attempt["directory"])
        kind = attempt.get("kind")
        if kind is not None:
            from .accepted_aggregation import budget_source_owner

            branch = budget_source_owner(parent, attempt)
            if branch is not None:
                # The extra branch is budget ancestry only. Its original owner
                # must verify typed failure/UNKNOWN records; it supplies no
                # patient observation and grants no replay authorization.
                attempted_acquisition_targets(branch, [attempt])
                continue
        if kind == "confirmed_definition_format_failure":
            from .confirmed_definition_failure import verify_history as verify_confirmed_failure

            verify_confirmed_failure(parent, attempt)
            continue
        if kind in {
            "interrupted_public_material_deferred",
            "confirmed_public_local_failure_deferred",
        }:
            from .interrupted_definition import verify_history

            verify_history(parent, attempt)
            continue
        if kind == SOURCE_CHECKPOINT_KIND:
            from .source_checkpoint import verify_history

            verify_history(parent, attempt)
            continue
        if kind in {"unknown-scope-new-attempt/1.0.0", "confirmed-invalid-scope-new-attempt/1.0.0"}:
            from .scope_unknown_successor import verify_budget_history as verify_unknown_scope

            verify_unknown_scope(parent, attempt)
            continue
        if kind == "accepted-scope-local-recovery/1.0.0":
            from .scope_successor import verify_budget_history

            verify_budget_history(parent, attempt)
            # A qualified input survey is budget ancestry, not another medical
            # acquisition target and never permission to resend its request.
            continue
        routed = kind == "routed_source_failure_deferred"
        if routed:
            verify_routed_source_history(parent, attempt)
        if kind not in source_kinds and not routed:
            if kind is not None:
                raise ValueError("renewal_acquisition_history_kind")
            historical = json.loads((path / "result.json").read_bytes())
            if digest(historical) != attempt["checksum"]:
                raise ValueError("renewal_acquisition_history_changed")
            attempted.update(p["target"] for p in historical["phases"] if p["target"])
            continue
        failure = json.loads((path / "stage-failure.json").read_bytes())
        if digest(failure) != attempt["checksum"]:
            raise ValueError("renewal_acquisition_history_changed")
        protocol = json.loads((path / "protocol.json").read_bytes())
        run_id = identity(
            "RUN", "task-research:mm-live:" + digest((path.name, "definition-expansion"))[7:55]
        )
        if (
            (path / "result.json").exists()
            or failure["stage"] != "definitions"
            or failure["campaign_id"] != path.name
            or failure["failure_kind"]
            not in (
                {"confirmed_response_or_domain_refusal", "dispatched_unknown"}
                if routed
                else {source_kinds[kind]}
            )
            or failure["run_id"] != run_id
            or failure["protocol_checksum"] != digest(protocol)
            or failure["new_requests"] != attempt["requests"]
            or failure["input_checksum"] != parent.base.manifest.checksum
            or protocol["input"] != parent.base.manifest.checksum
            or protocol["task"] != parent.base.task.checksum
            or parent.documents(run_id).read("multimodal-stage-failure") != (failure,)
        ):
            raise ValueError("renewal_source_history_binding")
        for target in (*protocol["targets"], None):
            phase_run = identity(
                "RUN", "task-research:mm-live:" + digest((path.name, target or "review"))[7:55]
            )
            try:
                parent.contexts.read_research_run(phase_run)
            except ValueError as error:
                if str(error) != "research_context_required":
                    raise
            else:
                raise ValueError("renewal_source_history_acquisition_started")
    return attempted


def validate_review_capacity(policy, reused_ids, new_ids):
    """Check the complete owner plan, not just new request reservations.

    Historical items still occupy slots in the versioned batch-plan inventory.
    Reject impossible new preparation before acquisition; never rewrite an old
    plan, silently resize its batches or authorize historical requests again.
    """
    from packages.grounding.models import validate
    from packages.llm_provider.config import ReviewBatchPolicy

    policy = validate(ReviewBatchPolicy, policy)
    new = set(new_ids)
    reused = set(reused_ids) - new
    size = policy.batch_size

    def count(n):
        return (n + size - 1) // size

    ordinary = count(len(reused | new))
    partitioned = count(len(reused)) + count(len(new))
    if max(ordinary, partitioned) > policy.max_batches:
        raise ValueError("renewal_review_plan_capacity")


def validate_acquisition_selection(parent, plan):
    from .active_continuation import select_acquisition_targets

    selection = plan.get("acquisition_selection") or {}
    owner_methods = selection.get("schema_version") == "bounded-acquisition-selection/1.1.0"
    if (
        plan.get("schema_version")
        not in {
            "multimodal-campaign-renewal/1.26.0",
            *AGGREGATE_ALTERNATIVE_VERSIONS,
            *EXACT_ACQUISITION_VERSIONS,
        }
        or plan.get("mode") != "acquisition"
        or selection.get("selected") != plan.get("acquisition_targets")
        or selection.get("schema_version")
        not in {"bounded-acquisition-selection/1.0.0", "bounded-acquisition-selection/1.1.0"}
    ):
        raise ValueError("renewal_acquisition_selection_binding")
    exact = plan.get("schema_version") in EXACT_ACQUISITION_VERSIONS
    if exact:
        validate_exact_acquisition_budget(plan, parent.config)
    elif plan.get("exact_request_budget"):
        raise ValueError("renewal_exact_request_budget_version")
    if clarified_acquisition(plan):
        validate_acquisition_clarification(parent, plan)
    check_local_history(plan)
    attempted = attempted_acquisition_targets(parent, plan["previous_attempts"])
    if plan.get("schema_version") in (
        ALTERNATIVE_ACQUISITION_VERSIONS | AGGREGATE_ALTERNATIVE_VERSIONS
    ):
        from .alternative_acquisition import check

        eligible = check(parent, plan)
    elif plan.get("schema_version") in UNKNOWN_ACQUISITION_VERSIONS:
        from .unknown_acquisition import check

        eligible = check(parent, plan)
    else:
        if plan.get("unknown_acquisition") or plan.get("confirmed_invalid_acquisition"):
            raise ValueError("unknown_acquisition_requires_explicit_version")
        eligible = unattempted_targets(
            parent,
            attempted,
            parent.accepted_observations(),
            full_inventory=True,
            owner_methods=owner_methods,
        )
    if (
        select_acquisition_targets(eligible, selection["selected"], owner_methods=owner_methods)
        != selection
    ):
        raise ValueError("renewal_acquisition_selection_changed")


def clarified_acquisition(plan):
    return plan.get("schema_version") == CLARIFIED_ACQUISITION_VERSION or (
        plan.get("schema_version") == RETAINED_ACQUISITION_VERSION
        and bool(plan.get("review_clarification"))
    )


def validate_exact_acquisition_budget(plan, config):
    targets = plan.get("acquisition_targets", [])
    cap = plan.get("max_new_requests")
    clarified = plan.get("review_clarification", {}).get("assessment_ids", [])
    combined = clarified_acquisition(plan)
    batches = (
        len(targets) + len(clarified) + config.review_batching.batch_size - 1
    ) // config.review_batching.batch_size
    required = (len(targets) + batches) * (1 + config.transport.max_retries)
    if (
        plan.get("schema_version") not in EXACT_ACQUISITION_VERSIONS
        or plan.get("mode") != "acquisition"
        or plan.get("exact_request_budget") != "dispatch_only_local_completion_zero_request/1.0"
        or plan.get("acquisition_increment_only")
        or plan.get("continue_acquisition")
        or not 1 <= len(targets) <= 3
        or (combined and not 1 <= len(clarified) <= 2)
        or (combined and plan.get("new_observation_execution_allowed") is not True)
        or (not combined and bool(plan.get("review_clarification")))
        or len(clarified) != len(set(clarified))
        or bool(set(clarified) & set(targets))
        or type(cap) is not int
        or not required <= cap <= 12
        or plan.get("max_model_dispatches") != cap
        or plan.get("max_cumulative_requests") != 40
        or sum(p["requests"] for p in plan["previous_attempts"]) + cap > 40
        or plan.get("review_partition") != "owner_bound_reused_and_new/1.0.0"
        or plan.get("acquisition_selection", {}).get("schema_version")
        != "bounded-acquisition-selection/1.1.0"
    ):
        raise ValueError("renewal_exact_request_budget_binding")
    retained = plan.get("schema_version") in RETAINED_ACQUISITION_VERSIONS
    if (retained and plan.get("review_context_scope") != RETAINED_CONTEXT_SCOPE) or (
        not retained and plan.get("review_context_scope") is not None
    ):
        raise ValueError("renewal_retained_context_scope")


def validate_acquisition_clarification(parent, plan):
    """Existing owner-bound question correction, not an edited old answer."""
    from packages.evidence_details.review_tasks import project_review_tasks

    from .single_proposition import clarification_spec

    reader = accepted_reader(parent, plan["tip"]["run_id"])
    state = reader.state_reader()
    spec = plan["review_clarification"]
    ids = set(project_review_tasks(state).assessment_ids)
    clarified = set(spec["assessment_ids"])
    reused = set(plan["reuse_assessment_ids"])
    rows = [dict(id=r.id, definition=r.definition) for r in state.policy.requirements]
    if (
        spec != clarification_spec(rows, parent.base.task, spec["assessment_ids"], version="1.1.0")
        or state.material.goal_code != parent.base.task.target_code
        or digest(state) != plan["accepted_state_checksum"]
        or not clarified < ids
        or reused != ids - clarified
        or set(plan["acquisition_targets"]) & ids
        or plan.get("recorded_review_origin") is not None
    ):
        raise ValueError("renewal_acquisition_clarification_binding")


def acquisition_review_ids(plan):
    """Explicit new-version scope; never inherit a parent's subset silently."""
    new = plan["acquisition_targets"]
    old = plan["reuse_assessment_ids"]
    clarified = (
        plan["review_clarification"]["assessment_ids"] if clarified_acquisition(plan) else []
    )
    if not new or set(new) & set(old) or len(new) != len(set(new)) or len(old) != len(set(old)):
        raise ValueError("renewal_acquisition_review_scope")
    if set(clarified) & (set(new) | set(old)) or len(clarified) != len(set(clarified)):
        raise ValueError("renewal_acquisition_review_scope")
    return tuple(sorted(set(new) | set(old) | set(clarified)))


def recovery_configuration(origin, plan):
    """Explicit transport-only successors; historical receipts retain their provider."""
    if "recovery_http_client" in plan:
        from packages.llm_provider.client_selection import apply_selection

        if (
            plan.get("schema_version") != RETAINED_ACQUISITION_VERSION
            or plan.get("mode") != "acquisition"
            or plan["recovery_http_client"] is None
            or any(
                k in plan
                for k in ("recovery_network_route", "unknown_acquisition", "review_clarification")
            )
        ):
            raise ValueError("recovery_http_client_scope")
        config = apply_selection(origin, plan["recovery_http_client"])
        if digest(config) != plan.get("provider"):
            raise ValueError("recovery_http_client_provider_binding")
        return config
    if "recovery_network_route" not in plan:
        return origin
    from packages.llm_provider.config import with_network_route
    from packages.llm_provider.network_route import read_route, route_opener

    if (
        plan.get("schema_version")
        not in {
            RETAINED_REVIEW_VERSION,
            "multimodal-campaign-renewal/1.39.0",
            "multimodal-campaign-renewal/1.42.0",
        }
        or plan.get("mode") != "review"
        or plan.get("recovery_network_route") is None
    ):
        raise ValueError("recovery_network_route_scope")
    route = read_route(plan["recovery_network_route"])
    existing = getattr(origin, "network_route", None)
    if existing is not None:
        if existing != route:
            raise ValueError("recovery_network_route_changed")
        config = origin
    else:
        config = with_network_route(origin, route)
    # Check current environment before claims or credentials; no HTTP request.
    route_opener(route, config.transport.base_url)
    return config


def prepare(
    *,
    facts,
    previous,
    output,
    mode,
    authorization,
    expansion_policy=None,
    continuation=False,
    reuse_confirmed_review=False,
    request_limit=None,
    definition_increment_only=False,
    confirmed_facts_review=False,
    clarify_review=(),
    repair_review_format=False,
    review_definition_increment=False,
    acknowledge_unknown_request=None,
    pending_review=False,
    pending_new_window=False,
    pending_runtime_lease=False,
    pending_reassessment_window=False,
    historical_review_origin=None,
    recovery_network_route=None,
    acquisition_increment_only=False,
    continue_acquisition=False,
    review_acquired_only=False,
    repair_acquired_contract=False,
    acknowledge_format_request=(),
    format_closure_seconds=None,
    unstarted_context_review=False,
    acquisition_target=(),
    same_profile_review=False,
    budget_stop_review=False,
    shape_contract_review=False,
    root_direction_review=False,
    local_finalization=None,
    review_batch_size=None,
    exact_request_budget=False,
    review_acquisition_gap=False,
    review_method_profile=None,
    method_diagnostic=(),
    review_capacity_recovery=False,
    acknowledge_unknown_acquisition_request=(),
    acknowledge_confirmed_invalid_acquisition_request=(),
    historical_acquisition_origin=None,
    recovery_http_client=None,
    acknowledge_truncated_request=None,
    alternative_view_method=None,
):
    if historical_acquisition_origin is not None:
        options = locals().copy()
        allowed = {
            "facts",
            "previous",
            "output",
            "mode",
            "authorization",
            "request_limit",
            "acquisition_target",
            "exact_request_budget",
            "historical_acquisition_origin",
            "acknowledge_unknown_acquisition_request",
        }
        if (
            mode != "acquisition"
            or not exact_request_budget
            or request_limit != 2
            or len(acquisition_target) != 1
            or len(acknowledge_unknown_acquisition_request) != 1
            or Path(facts).absolute() != Path(previous).absolute()
            or any(value for key, value in options.items() if key not in allowed)
        ):
            raise ValueError("historical_acquisition_explicit_scope")
    confirmed_invalid_acquisition = bool(acknowledge_confirmed_invalid_acquisition_request)
    if confirmed_invalid_acquisition:
        options = locals().copy()
        allowed = {
            "facts",
            "previous",
            "output",
            "mode",
            "authorization",
            "request_limit",
            "acquisition_target",
            "exact_request_budget",
            "acknowledge_confirmed_invalid_acquisition_request",
            "confirmed_invalid_acquisition",
        }
        if (
            mode != "acquisition"
            or not exact_request_budget
            or len(acquisition_target) != 1
            or len(acknowledge_confirmed_invalid_acquisition_request) != 1
            or Path(facts).absolute() != Path(previous).absolute()
            or request_limit != 2
            or any(value for key, value in options.items() if key not in allowed)
        ):
            raise ValueError("confirmed_invalid_acquisition_explicit_scope")
        # Share the bounded owner reconstruction, not UNKNOWN semantics. The
        # persisted proof and protocol version below distinguish the states.
        acknowledge_unknown_acquisition_request = acknowledge_confirmed_invalid_acquisition_request
    if alternative_view_method is not None:
        options = locals().copy()
        from .alternative_acquisition import METHOD_VERSIONS

        allowed = {
            "facts",
            "previous",
            "output",
            "mode",
            "authorization",
            "request_limit",
            "acquisition_target",
            "exact_request_budget",
            "alternative_view_method",
            "acquisition_increment_only",
        }
        if (
            alternative_view_method not in METHOD_VERSIONS
            or mode != "acquisition"
            or exact_request_budget == acquisition_increment_only
            or (acquisition_increment_only and request_limit != 3)
            or len(acquisition_target) != 1
            or request_limit not in (2, 3)
            or Path(facts).absolute() != Path(previous).absolute()
            or any(value for key, value in options.items() if key not in allowed)
        ):
            raise ValueError("alternative_view_explicit_bounded_scope")
    if historical_review_origin is not None and not pending_reassessment_window:
        raise ValueError("historical_pending_explicit_window_required")
    if pending_reassessment_window and (
        not pending_review
        or mode != "review"
        or not acknowledge_unknown_request
        or request_limit != 2
        or any(
            (
                pending_new_window,
                pending_runtime_lease,
                expansion_policy,
                continuation,
                reuse_confirmed_review,
                definition_increment_only,
                confirmed_facts_review,
                clarify_review,
                repair_review_format,
                review_definition_increment,
                recovery_network_route,
                recovery_http_client,
                acquisition_increment_only,
                continue_acquisition,
                review_acquired_only,
                repair_acquired_contract,
                acknowledge_format_request,
                format_closure_seconds,
                unstarted_context_review,
                acquisition_target,
                same_profile_review,
                budget_stop_review,
                shape_contract_review,
                root_direction_review,
                local_finalization,
                review_batch_size,
                exact_request_budget,
                review_acquisition_gap,
                review_method_profile,
                method_diagnostic,
                review_capacity_recovery,
                acknowledge_unknown_acquisition_request,
                acknowledge_truncated_request,
            )
        )
    ):
        raise ValueError("pending_reassessment_explicit_scope")
    if pending_runtime_lease and (
        not pending_review
        or mode != "review"
        or pending_new_window
        or acknowledge_unknown_request
        or acknowledge_format_request
        or acknowledge_truncated_request
        or review_capacity_recovery
        or review_method_profile
    ):
        raise ValueError("pending_runtime_explicit_unstarted_scope")
    if acknowledge_truncated_request:
        options = locals().copy()
        from .truncated_review import prepare as prepare_truncated

        return prepare_truncated(**options)
    if recovery_http_client is not None and (
        mode != "acquisition"
        or not exact_request_budget
        or not 1 <= len(acquisition_target) <= 3
        or any(
            (
                expansion_policy,
                continuation,
                reuse_confirmed_review,
                definition_increment_only,
                confirmed_facts_review,
                clarify_review,
                repair_review_format,
                review_definition_increment,
                acknowledge_unknown_request,
                pending_review,
                pending_new_window,
                recovery_network_route,
                acquisition_increment_only,
                continue_acquisition,
                review_acquired_only,
                repair_acquired_contract,
                acknowledge_format_request,
                format_closure_seconds,
                unstarted_context_review,
                same_profile_review,
                budget_stop_review,
                shape_contract_review,
                root_direction_review,
                local_finalization,
                review_batch_size,
                review_acquisition_gap,
                review_method_profile,
                method_diagnostic,
                review_capacity_recovery,
                acknowledge_unknown_acquisition_request,
            )
        )
    ):
        raise ValueError("recovery_http_client_scope")
    if acknowledge_unknown_acquisition_request and (
        mode != "acquisition"
        or not exact_request_budget
        or not acquisition_target
        or any(
            (
                expansion_policy,
                continuation,
                reuse_confirmed_review,
                definition_increment_only,
                confirmed_facts_review,
                clarify_review,
                repair_review_format,
                review_definition_increment,
                acknowledge_unknown_request,
                pending_review,
                pending_new_window,
                recovery_network_route,
                acquisition_increment_only,
                continue_acquisition,
                review_acquired_only,
                repair_acquired_contract,
                acknowledge_format_request,
                format_closure_seconds,
                unstarted_context_review,
                same_profile_review,
                budget_stop_review,
                shape_contract_review,
                root_direction_review,
                local_finalization,
                review_batch_size,
                review_acquisition_gap,
                review_method_profile,
                method_diagnostic,
                review_capacity_recovery,
            )
        )
    ):
        raise ValueError("unknown_acquisition_explicit_scope")
    if review_capacity_recovery:
        options = locals().copy()
        from .review_capacity_recovery import prepare as prepare_capacity

        return prepare_capacity(**options)
    if recovery_network_route is not None and not (
        mode == "review" and (pending_new_window or review_acquisition_gap)
    ):
        raise ValueError("recovery_network_route_scope")
    if pending_new_window and (
        not pending_review
        or mode != "review"
        or acknowledge_unknown_request
        or acknowledge_format_request
        or review_method_profile is not None
    ):
        raise ValueError("pending_unstarted_window_explicit_scope")
    if review_method_profile is not None:
        options = locals().copy()
        from .review_method_window import prepare as prepare_method

        return prepare_method(**options)
    if method_diagnostic:
        raise ValueError("review_method_diagnostic_requires_explicit_method")
    if review_acquisition_gap:
        options = locals().copy()
        from .acquired_review import prepare_projection_repair

        return prepare_projection_repair(**options)
    if exact_request_budget and (
        mode != "acquisition"
        or not acquisition_target
        or acquisition_increment_only
        or continue_acquisition
        or root_direction_review
        or shape_contract_review
        or same_profile_review
        or budget_stop_review
    ):
        raise ValueError("renewal_exact_request_budget_scope")
    if budget_stop_review and same_profile_review:
        raise ValueError("confirmed_review_window_explicit_text_only_scope")
    if root_direction_review:
        options = locals().copy()
        options.pop("root_direction_review")
        from .root_review_window import prepare as prepare_root_direction

        return prepare_root_direction(**options)
    recorded_definition = bool(
        local_finalization is not None
        and mode == "definitions"
        and definition_increment_only
        and not continuation
        and not reuse_confirmed_review
        and Path(facts).absolute() == Path(previous).absolute()
    )
    if (
        local_finalization is not None and not recorded_definition
    ) or review_batch_size is not None:
        raise ValueError("root_review_local_closure_requires_explicit_mode")
    if shape_contract_review:
        options = locals().copy()
        options.pop("shape_contract_review")
        from .review_shape_window import prepare as prepare_shape_contract

        return prepare_shape_contract(**options)
    if same_profile_review or budget_stop_review:
        options = locals().copy()
        options.pop("same_profile_review")
        from .confirmed_review_window import prepare as prepare_same_profile

        return prepare_same_profile(**options)
    if mode not in LIMITS or not authorization.strip() or len(authorization) > 200:
        raise ValueError("renewal_explicit_authorization_required")
    combined_clarification = bool(
        mode == "acquisition" and exact_request_budget and acquisition_target and clarify_review
    )
    if acquisition_target and (mode != "acquisition" or continue_acquisition):
        raise ValueError("renewal_acquisition_selection_mode")
    if repair_acquired_contract and not review_acquired_only:
        raise ValueError("acquired_contract_repair_text_only_required")
    if acquisition_target:
        # Validate only bounded caller selection shape before expensive history.
        # Eligibility still comes from the original owner below, never this list.
        from .active_continuation import select_acquisition_targets

        select_acquisition_targets(acquisition_target, acquisition_target, owner_methods=True)
    parent, original = completed_parent(facts)
    initial_failed_window = bool(
        pending_review
        and pending_new_window
        and not original["complete"]
        and not parent.protocol.get("pending_review")
    )
    partitioned_failed_window = bool(
        pending_reassessment_window
        and not original["complete"]
        and not parent.protocol.get("pending_review")
        and parent.protocol.get("mode") == "acquisition"
        and parent.protocol.get("reuse_assessment_ids")
    )
    if initial_failed_window and recovery_network_route is not None:
        raise ValueError("initial_failed_window_same_provider_required")
    if unstarted_context_review and (
        mode != "review"
        or Path(facts).absolute() != Path(previous).absolute()
        or any(
            (
                expansion_policy,
                continuation,
                reuse_confirmed_review,
                definition_increment_only,
                confirmed_facts_review,
                clarify_review,
                repair_review_format,
                review_definition_increment,
                acknowledge_unknown_request,
                pending_review,
                acquisition_increment_only,
                continue_acquisition,
                review_acquired_only,
                repair_acquired_contract,
                acknowledge_format_request,
                format_closure_seconds,
            )
        )
    ):
        raise ValueError("renewal_unstarted_context_explicit_scope")
    if bool(acknowledge_format_request) != (format_closure_seconds is not None) or (
        acknowledge_format_request
        and (not pending_review or acknowledge_unknown_request or mode != "review")
    ):
        raise ValueError("renewal_known_format_explicit_scope")
    if review_acquired_only and (
        mode != "review"
        or Path(facts).absolute() != Path(previous).absolute()
        or any(
            (
                continuation,
                reuse_confirmed_review,
                definition_increment_only,
                confirmed_facts_review,
                clarify_review,
                repair_review_format,
                review_definition_increment,
                pending_review,
                expansion_policy,
                acknowledge_unknown_request,
                acquisition_increment_only,
                continue_acquisition,
            )
        )
    ):
        raise ValueError("acquired_review_scope")
    if continue_acquisition and (
        not acquisition_increment_only
        or mode != "acquisition"
        or any(
            (
                continuation,
                reuse_confirmed_review,
                definition_increment_only,
                confirmed_facts_review,
                clarify_review,
                repair_review_format,
                review_definition_increment,
                pending_review,
                expansion_policy,
                acknowledge_unknown_request,
            )
        )
    ):
        raise ValueError("renewal_declined_acquisition_scope")
    if acquisition_increment_only and (
        mode != "acquisition" or not getattr(parent, "aggregation_baseline", None)
    ):
        raise ValueError("renewal_increment_requires_accepted_aggregation")
    if getattr(parent, "aggregation_baseline", None) and not acquisition_increment_only:
        raise ValueError("renewal_aggregation_requires_explicit_increment")
    if acknowledge_unknown_request and not (review_definition_increment or pending_review):
        raise ValueError("renewal_unknown_ack_requires_definition_review")
    if pending_review and (
        mode != "review"
        or Path(facts).absolute() != Path(previous).absolute()
        or (
            not original["complete"]
            and not parent.protocol.get("pending_review")
            and not initial_failed_window
            and not partitioned_failed_window
        )
        or expansion_policy is not None
        or continuation
        or reuse_confirmed_review
        or definition_increment_only
        or confirmed_facts_review
        or clarify_review
        or repair_review_format
        or review_definition_increment
    ):
        raise ValueError("renewal_pending_review_scope")
    if review_definition_increment and (
        mode != "review"
        or expansion_policy is not None
        or continuation
        or reuse_confirmed_review
        or definition_increment_only
        or confirmed_facts_review
        or clarify_review
        or repair_review_format
        or not original["complete"]
        or Path(facts).absolute() == Path(previous).absolute()
    ):
        raise ValueError("renewal_definition_review_scope")
    definition_origin = None
    if recorded_definition:
        definition_origin = recorded_definition_origin(parent, original, local_finalization)
    recorded_origin = None
    if mode == "acquisition" and not original["complete"]:
        recorded_origin = recorded_acquisition_origin(parent, original["phases"][-1]["run_id"])
    if repair_review_format and (
        mode != "review"
        or not reuse_confirmed_review
        or continuation
        or clarify_review
        or confirmed_facts_review
        or definition_increment_only
        or Path(facts).absolute() != Path(previous).absolute()
        or original["complete"]
    ):
        raise ValueError("renewal_format_repair_scope")
    if clarify_review and (
        (mode != "review" and not combined_clarification)
        or continuation
        or reuse_confirmed_review
        or definition_increment_only
        or confirmed_facts_review
        or Path(facts).absolute() != Path(previous).absolute()
        or not original["complete"]
    ):
        raise ValueError("renewal_clarification_requires_completed_origin")
    if mode == "acquisition" and (
        continuation
        or reuse_confirmed_review
        or definition_increment_only
        or confirmed_facts_review
        or (
            Path(facts).absolute() != Path(previous).absolute()
            and not continue_acquisition
            and not acknowledge_unknown_acquisition_request
        )
        or (not original["complete"] and recorded_origin is None)
    ):
        raise ValueError("renewal_acquisition_requires_completed_fact_origin")
    if reuse_confirmed_review and mode != "review":
        raise ValueError("renewal_reuse_review_only")
    if definition_increment_only and (
        mode != "definitions" or continuation or reuse_confirmed_review
    ):
        raise ValueError("renewal_definition_increment_scope")
    if confirmed_facts_review and (
        mode != "review" or continuation or reuse_confirmed_review or definition_increment_only
    ):
        raise ValueError("renewal_confirmed_facts_scope")
    policy = None
    if mode == "definitions":
        if expansion_policy is None:
            raise ValueError("renewal_expansion_policy_required")
        policy = read_expansion_policy(json.loads(Path(expansion_policy).read_bytes())).model_dump(
            mode="json"
        )
    contextual = bool(
        policy
        and policy["schema_version"]
        in {
            "campaign-definition-expansion/1.4.0",
            "campaign-definition-expansion/1.5.0",
            "campaign-definition-expansion/1.6.0",
            *CURRENT_INCREMENT_POLICIES,
        }
    )
    if contextual and (not definition_increment_only or continuation):
        raise ValueError("renewal_contextual_increment_required")
    new_increment = bool(policy and policy["schema_version"] in CURRENT_INCREMENT_POLICIES)
    if new_increment and (
        Path(facts).absolute() != Path(previous).absolute()
        or (not original["complete"] and definition_origin is None)
    ):
        raise ValueError("renewal_current_increment_requires_completed_origin")
    if recorded_definition and (
        not new_increment or policy["schema_version"] not in RUNTIME_DEFINITION_POLICIES
    ):
        raise ValueError("recorded_definition_runtime_policy_required")
    post_definition_proof = None
    if (
        acknowledge_unknown_acquisition_request
        and Path(facts).absolute() != Path(previous).absolute()
        and (Path(previous) / "protocol.json").is_file()
        and json.loads((Path(previous) / "protocol.json").read_bytes()).get("mode") == "definitions"
    ):
        from .unknown_acquisition import post_definition_tip

        tip, post_definition_proof = post_definition_tip(parent, previous, facts)
    else:
        tip = stopped_tip(
            parent,
            previous,
            completed_source=reuse_confirmed_review or contextual,
            accepted_definition=review_definition_increment,
        )
    if (
        acknowledge_unknown_acquisition_request
        and Path(facts).absolute() != Path(previous).absolute()
        and not tip.get("local_configuration_stop")
        and post_definition_proof is None
    ):
        raise ValueError("unknown_local_configuration_predecessor_required")
    if continue_acquisition and "declined_acquisition" not in tip:
        raise ValueError("renewal_confirmed_decline_required")
    if "acquired_review" in tip and not review_acquired_only:
        raise ValueError("acquired_review_explicit_text_only_required")
    if review_acquired_only and "acquired_review" not in tip:
        raise ValueError("acquired_review_terminal_proof_required")
    if bool(tip.get("contract_repair")) != repair_acquired_contract:
        raise ValueError("acquired_contract_repair_explicit_required")
    reassessment = (
        definition_reassessment(parent, tip, acknowledge_unknown_request)
        if acknowledge_unknown_request and not pending_review
        else None
    )
    if str(Path(facts).absolute()) not in {r["directory"] for r in tip["previous_attempts"]}:
        raise ValueError("renewal_fact_parent_not_in_history")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("renewal_new_directory_required")
    seconds, cap = LIMITS[mode]
    if acknowledge_unknown_acquisition_request:
        seconds = 2700
    if review_definition_increment:
        seconds, cap = 3600, 2
    if repair_review_format:
        seconds, cap = 3600, 8
    if confirmed_facts_review:
        seconds, cap = 5400, 12
    if pending_review:
        seconds, cap = 2700, 10 if initial_failed_window else (6 if pending_new_window else 8)
    if unstarted_context_review:
        seconds, cap = 2700, 8
    if acknowledge_format_request:
        if not 1 <= format_closure_seconds <= 2700:
            raise ValueError("renewal_known_format_window_bound")
        seconds, cap = format_closure_seconds, 10
    if request_limit is not None:
        if not 1 <= request_limit <= cap:
            raise ValueError("renewal_request_limit")
        cap = request_limit
    if continue_acquisition:
        cap = min(cap, tip["declined_acquisition"]["remaining_requests"])
    if reassessment and cap != 2:
        raise ValueError("renewal_reassessment_requires_local_close_headroom")
    if reuse_confirmed_review and not repair_review_format:
        cap = min(cap, 2)
    if definition_increment_only:
        cap = min(cap, 8)
    if sum(p["requests"] for p in tip["previous_attempts"]) + cap > 40:
        raise ValueError("renewal_cumulative_budget")
    if mode == "definitions":
        if definition_increment_only and policy["schema_version"] not in {
            "campaign-definition-expansion/1.3.0",
            "campaign-definition-expansion/1.4.0",
            "campaign-definition-expansion/1.5.0",
            "campaign-definition-expansion/1.6.0",
            *CURRENT_INCREMENT_POLICIES,
        }:
            raise ValueError("renewal_definition_increment_policy")
    elif expansion_policy is not None:
        raise ValueError("renewal_review_cannot_search")
    config = configuration(resilient_vision=True, directional=True, exact_primary_status=True)
    if reuse_confirmed_review or definition_increment_only:
        config = compact_configuration()
    if confirmed_facts_review:
        config = confirmed_facts_configuration()
    if repair_review_format:
        from packages.llm_provider.config import review_format_upgrade

        config = review_format_upgrade(parent.config)
    if (
        mode == "acquisition"
        or clarify_review
        or review_definition_increment
        or pending_review
        or review_acquired_only
        or unstarted_context_review
    ):
        config = parent.config
    if mode == "acquisition":
        from .unknown_acquisition import check_network

        check_network(config)  # No credentials, reservation or HTTP before route validation.
    if new_increment:
        validate_increment_window(policy, seconds=seconds, cap=cap)
        config = parent.config
    inherited_definition_acquisition = (
        pending_definition_acquisition(parent, tip) if new_increment else None
    )
    if repair_acquired_contract:
        from packages.llm_provider.config import qualified_citation_upgrade

        config = qualified_citation_upgrade(parent.config)
    client_binding = None
    if recovery_http_client is not None:
        from packages.llm_provider.client_selection import apply_selection, load_selection

        client_binding = load_selection(recovery_http_client)
        config = apply_selection(config, client_binding)
    route = None
    if recovery_network_route is not None:
        from packages.llm_provider.network_route import load_route

        route = load_route(recovery_network_route).model_dump(mode="json")
        config = recovery_configuration(
            config,
            dict(
                schema_version="multimodal-campaign-renewal/1.39.0",
                mode="review",
                recovery_network_route=route,
            ),
        )
    if mode == "definitions":
        from .unknown_acquisition import check_network

        # The historical provider freezes its proxy/client. Fail before creating
        # a successor claim if the launching terminal does not supply that route.
        check_network(config)
    confirmed = parent.accepted_observations()
    now = datetime.now(UTC)
    continued = None
    review_origin = None
    deadline = now + timedelta(seconds=seconds)
    if review_acquired_only:
        from .acquired_review import bounds

        deadline, seconds, cap, dispatches = bounds(tip, config, cap, now)
    if continue_acquisition:
        deadline = datetime.fromisoformat(tip["declined_acquisition"]["deadline"])
        seconds = int((deadline - now).total_seconds())
        if seconds <= 300 or cap < 2:
            raise ValueError("renewal_declined_acquisition_window_exhausted")
    if continuation and reuse_confirmed_review:
        old = json.loads((Path(previous) / "protocol.json").read_bytes())
        result = json.loads((Path(previous) / "result.json").read_bytes())
        deadline, cap = review_repair_bounds(old, result, now)
        seconds = int((deadline - now).total_seconds())
        review_origin = old["tip"]
        if stopped_tip(parent, review_origin["path"], completed_source=True) != review_origin:
            raise ValueError("renewal_review_origin_changed")
    elif continuation:
        if mode != "definitions" or "definition" not in tip or policy["public_topics"]:
            raise ValueError("renewal_definition_continuation_scope")
        if not policy["index_keys"]:
            raise ValueError("renewal_definition_index_required")
        old = json.loads((Path(previous) / "protocol.json").read_bytes())
        deadline, cap = continuation_bounds(old, tip["previous_attempts"][-1]["requests"], now)
        seconds = int((deadline - now).total_seconds())
        config = compact_configuration()
        continued = dict(
            campaign_root_id=Path(previous).name,
            previous_revision=tip["definition"]["revision"],
            previous_protocol=digest(old),
            external_limit=old["max_new_requests"],
            previous_external_requests=tip["previous_attempts"][-1]["requests"],
            deadline_unchanged=True,
            new_public_http_allowed=False,
        )
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.1.0",
        mode=mode,
        authorization=authorization,
        facts=str(Path(facts).absolute()),
        tip=tip,
        original_fact_result=digest(original),
        previous_attempts=tip["previous_attempts"],
        input=parent.base.manifest.checksum,
        task=parent.base.task.checksum,
        provider=digest(config),
        implementation=freeze(),
        max_new_requests=cap,
        max_cumulative_requests=40,
        seconds=seconds,
        started_at=now.isoformat(),
        deadline=deadline.isoformat(),
        prior_deadline=tip["deadline"]
        if acquisition_increment_only
        else json.loads((Path(previous) / "protocol.json").read_bytes())["deadline"],
        confirmed_observations_checksum=digest(confirmed),
        expansion_policy=policy,
        original_unknown_not_replayed=True,
        new_models=0,
        clinical_acceptance=False,
    )
    if route is not None:
        plan["recovery_network_route"] = route
    if client_binding is not None:
        plan["recovery_http_client"] = client_binding
    if unstarted_context_review:
        plan.update(
            schema_version="multimodal-campaign-renewal/1.25.0",
            unstarted_context_review=unstarted_context_proof(parent, original, tip),
            max_model_dispatches=cap - 1,
            acquisition_targets=[],
            new_vision_requests=0,
            new_public_search_allowed=False,
        )
        validate_unstarted_context(plan, parent, original)
    if review_acquired_only:
        plan.update(
            schema_version="multimodal-campaign-renewal/1.20.0"
            if repair_acquired_contract
            else "multimodal-campaign-renewal/1.19.0",
            review_acquired_only=True,
            review_target_ids=tip["acquired_review"]["selected_ids"],
            reuse_assessment_ids=tip["acquired_review"]["historical_review_ids"],
            acquisition_targets=[],
            max_model_dispatches=dispatches,
            new_vision_requests=0,
            new_public_search_allowed=False,
            continuation_kind="confirmed_acquired_text_closure_same_window",
        )
        if repair_acquired_contract:
            plan.update(repair_acquired_contract=True)
    if pending_review:
        from .pending_review import inventory

        pending = inventory(
            parent,
            tip,
            acknowledge_unknown_request,
            acknowledge_format_request,
            initial_partition=partitioned_failed_window,
            historical_origin=historical_review_origin,
        )
        required = (
            len(pending["selected_ids"]) + config.review_batching.batch_size - 1
        ) // config.review_batching.batch_size
        if required * (config.transport.max_retries + 1) + 1 > cap:
            raise ValueError("renewal_pending_review_budget")
        plan.update(
            schema_version="multimodal-campaign-renewal/1.16.0"
            if parent.protocol.get("pending_review") and not original["complete"]
            else "multimodal-campaign-renewal/1.15.0",
            pending_review=pending,
            review_target_ids=pending["selected_ids"],
            new_public_search_allowed=False,
            new_vision_requests=0,
            review_increment_only=True,
            max_model_dispatches=required * (config.transport.max_retries + 1),
        )
        if initial_failed_window:
            from .pending_review import validate_initial_failed_window

            plan.update(
                schema_version="multimodal-campaign-renewal/1.41.0",
                initial_failed_review_window="unstarted_only/1.0.0",
                pending_new_window="owner_verified_unstarted_only/1.0.0",
                previous_window_deadline=parent.protocol["deadline"],
            )
            validate_initial_failed_window(plan, parent)
        elif pending_new_window:
            from .pending_review import (
                UNSTARTED_FOLLOWUP_VERSION,
                UNSTARTED_WINDOW_VERSION,
                validate_unstarted_window,
            )

            followup = parent.protocol.get("schema_version") == UNSTARTED_WINDOW_VERSION
            plan.update(
                schema_version=UNSTARTED_FOLLOWUP_VERSION if followup else UNSTARTED_WINDOW_VERSION,
                pending_new_window="owner_verified_unstarted_only/1.0.0",
                whole_case_execution_profile=getattr(parent.base, "execution_profile", None),
                previous_window_deadline=parent.protocol["deadline"],
                max_model_dispatches=min(
                    cap - 1, required * (1 + getattr(config, "format_correction_budget", 0))
                ),
            )
            if followup:
                plan["pending_window_parent_checksum"] = digest(parent.protocol)
            validate_unstarted_window(plan, parent)
        elif acknowledge_format_request:
            plan.update(
                schema_version="multimodal-campaign-renewal/1.24.0",
                whole_case_execution_profile=getattr(parent.base, "execution_profile", None),
                previous_window_deadline=parent.protocol["deadline"],
                max_model_dispatches=min(
                    cap - 1, required * (1 + getattr(config, "format_correction_budget", 0))
                ),
            )
            validate_format_window(plan, parent)
        elif pending_reassessment_window:
            from .pending_review import declare_reassessment_window

            declare_reassessment_window(plan, parent)
        elif getattr(parent.base, "execution_profile", None):
            bounded = min(deadline, datetime.fromisoformat(parent.protocol["deadline"]))
            if (bounded - now).total_seconds() <= 300:
                raise ValueError("renewal_context_original_window_exhausted")
            plan.update(
                schema_version="multimodal-campaign-renewal/1.23.0",
                whole_case_execution_profile=parent.base.execution_profile,
                original_window_deadline=parent.protocol["deadline"],
                deadline=bounded.isoformat(),
                seconds=int((bounded - now).total_seconds()),
            )
            validate_context_window(plan, parent)
        if pending_runtime_lease:
            from .pending_review import declare_runtime_pending_window

            declare_runtime_pending_window(plan, parent)
    if continued:
        plan["continuation"] = continued
    if clarify_review:
        from packages.grounding.models import validate

        from .single_proposition import clarification_spec, clarify_review_state

        reader = accepted_reader(parent, tip["run_id"])
        old = reader.state_reader()
        spec = clarification_spec(
            [dict(id=r.id, definition=r.definition) for r in old.policy.requirements],
            parent.base.task,
            clarify_review,
            version="1.1.0" if combined_clarification else "1.0.0",
        )
        raw = old.model_dump(mode="json")
        raw["material"]["scope"]["run_id"] = identity("RUN", "clarify-preflight:" + out.name)
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        all_ids = set(reader.current(validate(type(old), raw)).assessment_ids)
        from packages.evidence_details.review_tasks import project_review_tasks

        if (
            not set(clarify_review) < all_ids
            or (config.review_batching.batch_size != 1 and not combined_clarification)
            or set(project_review_tasks(old).assessment_ids) != all_ids
        ):
            raise ValueError("renewal_clarification_requires_accepted_subset")
        origin = reader.current(
            clarify_review_state(validate(type(old), raw), parent.base.task, spec)
        )
        if set(origin.assessment_ids) != all_ids - set(clarify_review) or (
            not combined_clarification and cap <= len(clarify_review)
        ):
            raise ValueError("renewal_clarification_reuse_or_closure_budget")
        plan.update(
            schema_version="multimodal-campaign-renewal/1.9.0",
            review_clarification=spec,
            reuse_assessment_ids=list(origin.assessment_ids),
            accepted_origin_checksum=digest(origin),
            new_public_search_allowed=False,
            new_observation_execution_allowed=combined_clarification,
        )
    if mode == "acquisition":
        fact_run = tip["run_id"]
        if tip.get("local_configuration_stop") or post_definition_proof is not None:
            fact_run = original["phases"][-1]["run_id"]
            plan["accepted_fact_run_id"] = fact_run
        reader = (
            None
            if acquisition_increment_only
            else accepted_reader(
                parent,
                fact_run,
                recorded_origin=recorded_origin,
                client_config=config if client_binding is not None else None,
            )
        )
        old = (
            parent.aggregation_baseline.state
            if acquisition_increment_only
            else reader.state_reader()
        )
        # Only execution identity changes for this initial owner preflight. At
        # actual dispatch/acceptance the reader rechecks the new complete state.
        raw = old.model_dump(mode="json")
        if client_binding is not None:
            raw["config_checksum"] = digest(config)
        raw["material"]["scope"]["run_id"] = identity("RUN", "reuse-preflight:" + out.name)
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        from packages.grounding.models import validate

        current_state = validate(type(old), raw)
        if combined_clarification:
            from .single_proposition import clarify_review_state

            current_state = clarify_review_state(
                current_state, parent.base.task, plan["review_clarification"]
            )
        origin = None if acquisition_increment_only else reader.current(current_state)
        attempted = attempted_acquisition_targets(parent, tip["previous_attempts"])
        unknown_proof = None
        alternative_proof = None
        if alternative_view_method:
            from .alternative_acquisition import proof

            alternative_proof = proof(parent, acquisition_target[0], method=alternative_view_method)
            targets = [acquisition_target[0]]
        elif acknowledge_unknown_acquisition_request:
            from .unknown_acquisition import selection as unknown_selection

            if config.transport.max_retries != 0:
                raise ValueError("unknown_acquisition_no_automatic_retry")
            unknown_proof = unknown_selection(
                parent,
                tip["previous_attempts"],
                acknowledge_unknown_acquisition_request,
                confirmed,
                **(
                    {"confirmed_invalid": True}
                    if confirmed_invalid_acquisition
                    else {"historical_origin": historical_acquisition_origin}
                    if historical_acquisition_origin
                    else {"local_request_ids": acknowledge_unknown_acquisition_request}
                    if tip.get("configuration_attempt_stop")
                    else {}
                ),
            )
            targets = unknown_proof["selected"]
            if set(targets) != set(acquisition_target):
                raise ValueError("unknown_acquisition_explicit_targets")
        else:
            targets = unattempted_targets(
                parent,
                attempted,
                confirmed,
                full_inventory=bool(acquisition_target),
                owner_methods=bool(acquisition_target),
            )
        selection = None
        if acquisition_target:
            from .active_continuation import select_acquisition_targets

            selection = select_acquisition_targets(targets, acquisition_target, owner_methods=True)
            targets = selection["selected"]
        if continue_acquisition:
            origin = tip["declined_acquisition"]
            targets = list(origin["targets"])
            if any(t not in set(unattempted_targets(parent, set(), confirmed)) for t in targets):
                raise ValueError("renewal_declined_acquisition_inputs_changed")
        if not targets or not 1 <= config.review_batching.batch_size <= 3:
            raise ValueError("renewal_acquisition_requires_unattempted_targets_and_bounded_batches")
        plan.update(
            schema_version="multimodal-campaign-renewal/1.8.0",
            acquisition_targets=targets,
            accepted_state_checksum=digest(old),
            accepted_origin_checksum=digest(parent.aggregation_baseline.manifest)
            if acquisition_increment_only
            else digest(origin),
            reuse_assessment_ids=[
                i.assessment_id for i in parent.aggregation_baseline.submission.candidate.items
            ]
            if acquisition_increment_only
            else list(origin.assessment_ids),
            reuse_semantics="historical_progress_only_new_observations_reviewed_separately"
            if acquisition_increment_only
            else origin.semantics,
            new_public_search_allowed=False,
            independent_observation_failure_limit=2,
            independent_observation_policy="bounded-independent-observations/1.0.0",
        )
        if recorded_origin is not None or config.review_batching.batch_size != 1:
            plan.update(
                schema_version="multimodal-campaign-renewal/1.12.0",
                recorded_review_origin=recorded_origin,
                review_partition="owner_bound_reused_and_new/1.0.0",
            )
        if acquisition_increment_only:
            plan.pop("recorded_review_origin", None)
            plan.pop("review_partition", None)
            plan.update(
                schema_version="multimodal-campaign-renewal/1.17.0",
                acquisition_increment_only=True,
                review_target_ids=targets,
            )
            required = (
                len(targets)
                + (len(targets) + config.review_batching.batch_size - 1)
                // config.review_batching.batch_size
            )
            if cap < required * (config.transport.max_retries + 1) + 1:
                raise ValueError("renewal_increment_acquisition_review_budget")
            if continue_acquisition:
                plan.update(
                    schema_version="multimodal-campaign-renewal/1.18.0",
                    continue_acquisition=True,
                    continuation_kind="one_confirmed_decline_successor_same_window",
                    max_model_dispatches=required * (config.transport.max_retries + 1),
                )
        if selection is not None:
            # Both original plan construction and reuse-separated partitions
            # must fit before any newly selected observation is dispatched.
            validate_review_capacity(
                config.review_batching,
                () if acquisition_increment_only else plan["reuse_assessment_ids"],
                (*targets, *(clarify_review if combined_clarification else ())),
            )
            required = (
                len(targets)
                + (
                    len(targets)
                    + (len(clarify_review) if combined_clarification else 0)
                    + config.review_batching.batch_size
                    - 1
                )
                // config.review_batching.batch_size
            )
            if cap < required * (config.transport.max_retries + 1) + (
                0 if exact_request_budget else 1
            ):
                raise ValueError("renewal_selection_acquisition_review_budget")
            plan.update(
                schema_version=RETAINED_ACQUISITION_VERSION
                if exact_request_budget
                else "multimodal-campaign-renewal/1.26.0",
                acquisition_selection=selection,
                review_partition="owner_bound_reused_and_new/1.0.0",
            )
            if unknown_proof is not None:
                plan.update(
                    schema_version=POST_DEFINITION_ACQUISITION_VERSION
                    if post_definition_proof is not None
                    else CONFIRMED_INVALID_ACQUISITION_VERSION
                    if confirmed_invalid_acquisition
                    else HISTORICAL_ACQUISITION_VERSION
                    if historical_acquisition_origin
                    else CONFIGURATION_ACQUISITION_VERSION
                    if tip.get("configuration_attempt_stop")
                    else UNKNOWN_LOCAL_VERSION
                    if tip.get("local_configuration_stop")
                    else UNKNOWN_ACQUISITION_VERSION,
                    **{
                        "confirmed_invalid_acquisition"
                        if confirmed_invalid_acquisition
                        else "unknown_acquisition": unknown_proof
                    },
                )
                if post_definition_proof is not None:
                    plan["post_definition_acquisition"] = post_definition_proof
                if historical_acquisition_origin:
                    plan["historical_acquisition_origin"] = str(
                        Path(historical_acquisition_origin).absolute()
                    )
            if exact_request_budget:
                from .execution_lease import declare_selected

                plan.update(
                    review_context_scope=RETAINED_CONTEXT_SCOPE,
                    exact_request_budget="dispatch_only_local_completion_zero_request/1.0",
                    max_model_dispatches=cap,
                )
                declare_selected(plan)
            if alternative_proof is not None:
                plan.update(
                    schema_version=AGGREGATE_ALTERNATIVE_VERSION
                    if acquisition_increment_only
                    else METHOD_VERSIONS[alternative_view_method],
                    alternative_view=alternative_proof,
                )
            validate_acquisition_selection(parent, plan)
            if acquisition_increment_only:
                from .execution_lease import declare_aggregate

                declare_aggregate(plan, parent)
    if confirmed_facts_review:
        plan.update(
            schema_version="multimodal-campaign-renewal/1.6.0",
            confirmed_facts_review=True,
            review_execution_policy="single-item-bounded-text/1.0.0",
        )
    if definition_increment_only:
        plan.update(
            schema_version="multimodal-campaign-renewal/1.4.0", definition_increment_only=True
        )
    if contextual:
        prior_policy = json.loads((Path(previous) / "protocol.json").read_bytes()).get(
            "expansion_policy"
        )
        if prior_policy == policy:
            raise ValueError("renewal_unchanged_retrieval_not_a_revision")
        plan.update(
            schema_version="multimodal-campaign-renewal/1.5.0",
            retrieval_revision=dict(
                previous_policy_checksum=digest(prior_policy),
                current_policy_checksum=digest(policy),
                reason="explicit_operator_authorized_contextual_retrieval_revision",
            ),
        )
    if new_increment:
        plan.update(schema_version="multimodal-campaign-renewal/1.11.0")
        if policy["schema_version"] in RUNTIME_DEFINITION_POLICIES:
            from .execution_lease import PROFILE

            plan.update(
                schema_version=RUNTIME_DEFINITION_VERSION,
                execution_lease_profile=PROFILE,
            )
            if policy["schema_version"] in INDEX_ONLY_POLICIES:
                plan["new_public_search_allowed"] = False
            if definition_origin is not None:
                plan.update(
                    schema_version=RECORDED_DEFINITION_VERSION,
                    recorded_definition_origin=definition_origin,
                )
            elif inherited_definition_acquisition is not None:
                plan.update(
                    schema_version=CASE_DEFINITION_VERSION,
                    pending_definition_acquisition=inherited_definition_acquisition,
                )
            validate_definition_runtime(plan)
    if review_definition_increment:
        # Rebuild through the same original definition owner. This reads the
        # original Source output, not a new proposal or invented authorization.
        from copy import copy

        staging = copy(parent)
        staging.base = copy(parent.base)
        definition_origin = reassessment["source_origin"] if reassessment else tip
        restore_origin_materials(staging, definition_origin)
        revision = staging.base.definition_revision
        selected = tuple(sorted(r["id"] for r in revision["replaced"]))
        if (
            not 1 <= len(selected) <= 2
            or revision["added"]
            or any(
                not r.get("review_allowed", True) for r in revision["rows"] if r["id"] in selected
            )
            or any(t not in {**staging.base.observations, **confirmed} for t in selected)
        ):
            raise ValueError("renewal_definition_review_observation_missing")
        plan.update(
            schema_version="multimodal-campaign-renewal/1.13.0",
            review_definition_increment=True,
            definition_increment_only=True,
            review_target_ids=list(selected),
            logical_campaign_id=Path(definition_origin["path"]).name,
            source_revision_checksum=digest(revision),
            source_provider_allowed=False,
            new_public_search_allowed=False,
            new_vision_requests=0,
            old_deadline_unchanged=True,
            continuation_kind="authorized_new_review_window_not_old_run_resume",
        )
        if reassessment:
            if reassessment["review_target_ids"] != list(selected) or reassessment[
                "source_revision_checksum"
            ] != digest(revision):
                raise ValueError("renewal_reassessment_definition_changed")
            plan.update(
                schema_version="multimodal-campaign-renewal/1.14.0",
                definition_origin=definition_origin,
                reassessment=reassessment,
                max_model_dispatches=1,
            )
    if reuse_confirmed_review:
        plan.update(schema_version="multimodal-campaign-renewal/1.2.0", reuse_confirmed_review=True)
    if repair_review_format:
        from packages.evidence_details.acceptance_models import DomainState
        from packages.evidence_details.review_tasks import project_review_tasks
        from packages.grounding.models import validate

        from .campaign import Phase

        reader = historical_reader(parent, tip["run_id"], format_config=config)
        old = reader.state_reader()
        current, _ = Phase(parent.base, observation_reader=parent.accepted_observations).project(
            run_id=tip["run_id"], config=parent.config
        )
        if current != old:
            raise ValueError("renewal_format_current_parent_changed")
        raw = old.model_dump(mode="json")
        raw["config_checksum"] = digest(config)
        raw["material"]["scope"]["run_id"] = identity("RUN", "format-preflight:" + out.name)
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        origin = reader.current(validate(DomainState, raw))
        reused = {i.assessment_id for i in origin.submission.candidate.items}
        remaining = set(project_review_tasks(old).assessment_ids) - reused
        if not remaining or len(remaining) + 1 > cap or config.review_batching.batch_size != 1:
            raise ValueError("renewal_format_remaining_budget")
        plan.update(
            schema_version="multimodal-campaign-renewal/1.10.0",
            repair_review_format=True,
            format_origin_checksum=digest(origin),
            reuse_assessment_ids=sorted(reused),
            remaining_assessment_ids=sorted(remaining),
            original_failed_requests=[r.request_id for r in origin.confirmed_format_failures],
            format_transition=origin.format_transition,
            new_vision_requests=0,
        )
    if review_origin:
        plan.update(
            schema_version="multimodal-campaign-renewal/1.3.0",
            confirmed_review_origin=review_origin,
            local_prewire_continuation=dict(
                parent=str(Path(previous).absolute()),
                deadline_unchanged=True,
                remaining_requests=cap,
                campaign_root_id=Path(previous).name,
            ),
        )
    if client_binding is not None:
        recovery_configuration(parent.config, plan)
    alternative_work = None
    if alternative_view_method:
        from .alternative_acquisition import prepare_material

        alternative_work, alternative_metadata = prepare_material(
            parent, plan["alternative_view"], out
        )
        plan["alternative_view_metadata_checksum"] = digest(alternative_metadata)
        validate_acquisition_selection(parent, plan)
    docs = parent.documents(tip["run_id"])
    claim = dict(output=str(out), plan_checksum=digest(plan))
    with docs.session():
        if docs.read("authorized-campaign-renewal") is not None:
            raise ValueError("renewal_tip_already_claimed")
        docs.write("authorized-campaign-renewal", (claim,))
    out.mkdir(mode=0o700)
    if alternative_work is not None:
        (alternative_work / "result").rename(out / "alternative-view")
        alternative_work.rename(out / "alternative-preparation")
    write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        mode=mode,
        previous_requests=sum(p["requests"] for p in tip["previous_attempts"]),
        max_new_requests=cap,
        api_requests=0,
    )


def assemble(output, *, depth=0, ancestry=()):
    out = Path(output).absolute()
    plan = json.loads((out / "protocol.json").read_bytes())
    if plan.get("schema_version") == "multimodal-campaign-renewal/1.74.0":
        from .local_review_recovery import assemble as assemble_local_review

        return assemble_local_review(out, depth=depth, ancestry=ancestry)
    if plan.get("schema_version") in UNSTARTED_WINDOW_VERSIONS:
        from .unstarted_window import assemble as assemble_window

        return assemble_window(out, depth=depth, ancestry=ancestry)
    if plan.get("schema_version") in {
        "multimodal-campaign-renewal/1.22.0",
        "multimodal-campaign-renewal/1.45.0",
        "multimodal-campaign-renewal/1.46.0",
        "multimodal-campaign-renewal/1.69.0",
        "multimodal-campaign-renewal/1.72.0",
    }:
        from .context_review import assemble as assemble_context

        return assemble_context(out, depth=depth, ancestry=ancestry)
    if plan.get("schema_version") in {
        "multimodal-campaign-renewal/1.21.0",
        "multimodal-campaign-renewal/1.44.0",
    }:
        from .provider_transition import assemble as assemble_transition

        return assemble_transition(out, depth=depth, ancestry=ancestry)
    if plan["schema_version"] not in {
        *CAPABILITY_VERSIONS,
        "multimodal-campaign-renewal/1.0.0",
        "multimodal-campaign-renewal/1.1.0",
        "multimodal-campaign-renewal/1.2.0",
        "multimodal-campaign-renewal/1.3.0",
        "multimodal-campaign-renewal/1.4.0",
        "multimodal-campaign-renewal/1.5.0",
        "multimodal-campaign-renewal/1.6.0",
        "multimodal-campaign-renewal/1.7.0",
        "multimodal-campaign-renewal/1.8.0",
        "multimodal-campaign-renewal/1.9.0",
        "multimodal-campaign-renewal/1.10.0",
        "multimodal-campaign-renewal/1.11.0",
        RUNTIME_DEFINITION_VERSION,
        RECORDED_DEFINITION_VERSION,
        *CASE_DEFINITION_VERSIONS,
        *INTERRUPTED_DEFINITION_VERSIONS,
        *CONFIRMED_DEFINITION_FAILURE_VERSIONS,
        "multimodal-campaign-renewal/1.12.0",
        "multimodal-campaign-renewal/1.13.0",
        "multimodal-campaign-renewal/1.14.0",
        "multimodal-campaign-renewal/1.15.0",
        "multimodal-campaign-renewal/1.16.0",
        "multimodal-campaign-renewal/1.17.0",
        "multimodal-campaign-renewal/1.18.0",
        "multimodal-campaign-renewal/1.19.0",
        "multimodal-campaign-renewal/1.20.0",
        "multimodal-campaign-renewal/1.23.0",
        "multimodal-campaign-renewal/1.54.0",
        "multimodal-campaign-renewal/1.55.0",
        "multimodal-campaign-renewal/1.57.0",
        "multimodal-campaign-renewal/1.65.0",
        "multimodal-campaign-renewal/1.24.0",
        "multimodal-campaign-renewal/1.39.0",
        "multimodal-campaign-renewal/1.41.0",
        "multimodal-campaign-renewal/1.42.0",
        "multimodal-campaign-renewal/1.25.0",
        "multimodal-campaign-renewal/1.26.0",
        *AGGREGATE_ALTERNATIVE_VERSIONS,
        "multimodal-campaign-renewal/1.27.0",
        "multimodal-campaign-renewal/1.28.0",
        "multimodal-campaign-renewal/1.29.0",
        "multimodal-campaign-renewal/1.30.0",
        "multimodal-campaign-renewal/1.31.0",
        *EXACT_ACQUISITION_VERSIONS,
        *ACQUISITION_REVIEW_VERSIONS,
        CAPACITY_REVIEW_VERSION,
        UNSCORED_ACQUISITION_VERSION,
        TRUNCATED_REVIEW_VERSION,
        "multimodal-campaign-renewal/1.35.0",
        "multimodal-campaign-renewal/1.40.0",
        "multimodal-campaign-renewal/1.36.0",
    }:
        raise ValueError("renewal_protocol_version")
    if plan["schema_version"] == DEFINITION_ALTERNATIVE_VERSION:
        from .provider_transition import read_ancestry

        # completed_parent has already appended this current node. Reuse the
        # original reader's convention; only an earlier occurrence is a cycle.
        ancestry = read_ancestry(out, depth, ancestry)
        if Path(plan["facts"]).absolute() in ancestry or depth >= MAX_HISTORY_DEPTH:
            raise ValueError("renewal_history_cycle_or_depth")
        parent = assemble(plan["facts"], depth=depth + 1, ancestry=ancestry)
        original = json.loads((parent.output / "result.json").read_bytes())
    else:
        parent, original = completed_parent(plan["facts"], depth=depth, ancestry=ancestry)
    return assemble_bound_parent(parent, out, plan, original=original)


def assemble_bound_parent(parent, out, plan, *, original=None):
    """Same original claim gate for an already reconstructed fact owner."""
    out = Path(out).absolute()
    if str(parent.output.absolute()) != plan["facts"]:
        raise ValueError("renewal_fact_parent_changed")
    if original is None:
        original = json.loads((parent.output / "result.json").read_bytes())
    if digest(original) != plan["original_fact_result"]:
        raise ValueError("renewal_fact_parent_changed")
    from .capability_window import SCOPE_RECOVERY_VERSIONS

    if plan.get("schema_version") in {
        *SCOPE_RECOVERY_VERSIONS,
        CAPABILITY_SCOPE_UNKNOWN_VERSION,
        CAPABILITY_SCOPE_CONFIRMED_VERSION,
    }:
        # install_base performs the exact new claim/current-owner gate.
        return RenewedCampaign(parent, out, plan)
    if plan.get("schema_version") == CAPABILITY_RECOVERY_VERSION:
        from .capability_recovery import check

        check(parent, plan)
        return RenewedCampaign(parent, out, plan)
    docs = parent.documents(plan["tip"]["run_id"])
    from .source_checkpoint import claim as checkpoint_claim

    expected_claim = (
        checkpoint_claim(plan, out)
        if plan.get("schema_version")
        in {SOURCE_CHECKPOINT_VERSION, CAPABILITY_VERSION, *CAPABILITY_CURRENT_VERSIONS}
        else dict(output=str(out), plan_checksum=digest(plan))
    )
    if docs.read("authorized-campaign-renewal") != (expected_claim,):
        raise ValueError("renewal_claim_binding")
    if plan.get("schema_version") == CAPABILITY_VERSION:
        from .capability_review_origin import resolve

        plan = resolve(parent, plan)
    return RenewedCampaign(parent, out, plan)


def run(output):
    """Dispatch a prepared successor only after its original route is current."""
    from .unknown_acquisition import check_network

    campaign = assemble(output)
    check_network(campaign.config)
    return campaign.execute()


def authorize_prewire_repair(output, proof, authorization):
    """Repair the zero-wire retry-ceiling gate, not an unknown response or new window."""
    import fcntl

    c = assemble(output)
    if c.protocol.get("schema_version") == "multimodal-campaign-renewal/1.57.0":
        from .pending_review import authorize_partition_projection_repair

        return authorize_partition_projection_repair(c, proof, authorization)
    with (c.output / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = c.protocol.get("schema_version") == "multimodal-campaign-renewal/1.15.0"
        if not authorization.strip() or not (c.protocol.get("confirmed_facts_review") or pending):
            raise ValueError("renewal_prewire_authorization")
        c.guard()
        path = c.phase_path(None)
        request = json.loads((path / "request.json").read_bytes())
        run_id = identity("RUN", "task-research:" + c.request_key(None))
        context = c.contexts.read_research_run(run_id)
        if context.request.model_dump(mode="json") != request:
            raise ValueError("renewal_prewire_request_binding")
        c.base.contexts[run_id] = context
        docs = c.documents(run_id)
        journal = RuntimeAgentJournal(docs).read()
        if (
            c.sessions.read(docs.run_id) is not None
            or any(e.get("phase") in {"reserved", "completed"} for e in journal)
            or docs.read("dynamic-phase-result") is not None
            or docs.read("dynamic-campaign") != (c.protocol,)
        ):
            raise ValueError("renewal_prewire_not_unstarted")
        before, after = c.protocol["implementation"], freeze()
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        allowed = {
            "packages/evidence_details/live_permission.py",
            "packages/multimodal_research/campaign_renewal.py",
            "apps/multimodal_case.py",
        }
        if pending:
            allowed |= {
                "packages/llm_provider/bounded_live.py",
                "packages/multimodal_research/campaign.py",
            }
        if not changed or not changed <= allowed:
            raise ValueError("renewal_prewire_change_scope")
        proof_path = Path(proof).absolute()
        evidence = json.loads(proof_path.read_bytes())
        if evidence.get("direct_exit") != 0:
            raise ValueError("renewal_prewire_tests_failed")
        value = dict(
            schema_version="renewal-prewire-repair/1.0.0",
            protocol=digest(c.protocol),
            run_id=docs.run_id,
            implementation_before=before,
            implementation_after=after,
            proof_path=str(proof_path),
            proof=digest(evidence),
            authorization=authorization,
            original_journal=digest(journal),
            deadline=c.protocol["deadline"],
            requests=0,
            reason="honor_narrowed_pending_review_window"
            if pending
            else "allow_stricter_zero_retry_configuration",
        )
        if pending:
            failure_path = c.output / "stage-failure.json"
            failure = json.loads(failure_path.read_bytes())
            if failure["error_code"] != "configuration" or failure["new_requests"] != 0:
                raise ValueError("renewal_prewire_failure_scope")
            value["original_failure_checksum"] = digest(failure)
        with docs.session():
            docs.write("renewal:prewire-repair", (value,))
        write_once(c.output / "prewire-repair.json", value)
        return dict(status="authorized_zero_wire_repair", requests=0, deadline=value["deadline"])

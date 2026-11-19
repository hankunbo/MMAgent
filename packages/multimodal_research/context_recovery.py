"""Explicit tested-code amendment for a registered, never-started context Review.

Retains the original campaign/window/budget and every acquisition outcome. This
is not transport recovery and cannot dispatch another image or search request.
"""

import fcntl
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.object_store import checksum

NAME = "review-context-repair.json"
REVISION = "review-context-repair-r02.json"
VERSION = "pre-wire-review-context-repair/1.0.0"
ALLOWED = frozenset(
    {
        "packages/multimodal_research/review_context.py",
        "packages/multimodal_research/stage_failure.py",
        "packages/multimodal_research/context_recovery.py",
        "packages/multimodal_research/implementation.py",
        "packages/multimodal_research/campaign.py",
        "apps/multimodal_case.py",
    }
)

BUDGET_NAME = "phase-budget-repair.json"
BUDGET_REVISION = "phase-budget-repair-r02.json"
BUDGET_ALLOWED = frozenset(
    {
        "packages/evaluation_preparation/dynamic_campaign.py",
        "packages/multimodal_research/context_recovery.py",
        "packages/multimodal_research/implementation.py",
        "packages/multimodal_research/campaign.py",
        "apps/multimodal_case.py",
    }
)


def read_budget_amendment(output):
    from .implementation import freeze

    p = Path(output)
    revised = (p / BUDGET_REVISION).exists()
    value = json.loads((p / (BUDGET_REVISION if revised else BUDGET_NAME)).read_bytes())
    before, after = value["implementation_before"], freeze()
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    if (
        value["schema_version"]
        != (
            "pre-wire-phase-budget-repair/1.1.0"
            if revised
            else "pre-wire-phase-budget-repair/1.0.0"
        )
        or value["campaign_id"] != p.name
        or not changed
        or not changed <= BUDGET_ALLOWED
        or before.keys() != after.keys()
        or value["implementation_after"] != after
        or value["delta"] != sorted(changed)
        or value["entry_checksum"] != checksum((p / "entry.json").read_bytes())
        or value["protocol_checksum"] != checksum((p / "protocol.json").read_bytes())
        or value["failure_checksum"] != checksum((p / "stage-failure.json").read_bytes())
        or value["request_checksum"] != checksum(Path(value["request_path"]).read_bytes())
    ):
        raise ValueError("phase_budget_repair_binding")
    if revised:
        previous = json.loads((p / BUDGET_NAME).read_bytes())
        if (
            value["previous_amendment_checksum"] != checksum((p / BUDGET_NAME).read_bytes())
            or previous["implementation_before"] != before
            or any(
                previous[k] != value[k]
                for k in (
                    "campaign_id",
                    "run_id",
                    "entry_checksum",
                    "protocol_checksum",
                    "failure_checksum",
                    "request_path",
                    "request_checksum",
                    "deadline",
                )
            )
        ):
            raise ValueError("phase_budget_repair_history_changed")
    for proof in value["tests"]:
        if checksum(Path(proof["path"]).read_bytes()) != proof["checksum"]:
            raise ValueError("phase_budget_repair_proof_changed")
    return value


def authorize_budget_repair(output, tests, *, supersede=False):
    """Original Runtime proof: configuration failure before any Planner/request start."""
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    from .entry import owners
    from .implementation import freeze

    p = Path(output).absolute()
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("phase_budget_repair_symlink")
    with (p / "lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads((p / "entry.json").read_bytes())
        protocol = json.loads((p / "protocol.json").read_bytes())
        failure = json.loads((p / "stage-failure.json").read_bytes())
        requests = list(p.glob("vision-*/request.json"))
        previous = json.loads((p / BUDGET_NAME).read_bytes()) if supersede else None
        if (
            spec["schema_version"] != "multimodal-live-entry/1.22.0"
            or protocol["schema_version"] != "multimodal-live-campaign/1.14.0"
            or spec["implementation"] != protocol["implementation"]
            or protocol["campaign_id"] != p.name
            or len(requests) != 1
            or failure["stage"] != "acquisition"
            or failure["error_code"] != "configuration"
            or failure["failure_kind"] != "local_stage_failure"
            or failure["new_requests"] != 0
            or not failure["definition_revision_saved"]
            or failure["protocol_checksum"] != digest(protocol)
            or datetime.now(UTC) >= datetime.fromisoformat(protocol["deadline"])
            or any(
                (p / n).exists()
                for n in (
                    BUDGET_REVISION if supersede else BUDGET_NAME,
                    "cancel",
                    "result.json",
                    "stage-closure-error.json",
                )
            )
        ):
            raise ValueError("phase_budget_repair_failure_scope")
        runtime, store, _, target = owners()
        contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
        rid = failure["run_id"]
        context = contexts.read_research_run(rid)
        request = context.request
        root = runtime.read_case_artifact(request.scope.case_code, request.artifacts[0].asset_code)
        docs = RuntimeDocuments(
            repo=runtime, contexts=contexts, sessions=sessions, store=store, run_id=rid, root=root
        )
        if (
            request.model_dump(mode="json") != json.loads(requests[0].read_bytes())
            or request.input_checksum != protocol["input"]
            or request.task.checksum != protocol["task"]
            or request.visible_checksum != protocol["visible"]
            or rid
            != identity(
                "RUN",
                "task-research:mm-live:" + digest((p.name, failure["control"]["target"]))[7:55],
            )
            or sessions.read(rid) is not None
            or RuntimeAgentJournal(docs).read()
            or docs.read("dynamic-phase-result") is not None
            or docs.read("dynamic-campaign") != (protocol,)
            or docs.read("multimodal-stage-failure") != (failure,)
            or not docs.read("dynamic-phase-input")
        ):
            raise ValueError("phase_budget_repair_started_or_unknown")
        if supersede and (
            previous["schema_version"] != "pre-wire-phase-budget-repair/1.0.0"
            or previous["implementation_before"] != spec["implementation"]
            or previous["run_id"] != rid
            or docs.read("multimodal:phase-budget-repair") != (previous,)
        ):
            raise ValueError("phase_budget_repair_history_changed")
        after = freeze()
        before = spec["implementation"]
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        if not changed or not changed <= BUDGET_ALLOWED or before.keys() != after.keys():
            raise ValueError("phase_budget_repair_uncovered_delta")
        proofs = []
        for filename in tests:
            path = Path(filename).absolute()
            proof = json.loads(path.read_bytes())
            if (
                proof["exit_code"] != 0
                or proof["implementation_before"] != after
                or proof["implementation_after"] != after
                or proof["log_checksum"] != checksum(path.with_suffix(".log").read_bytes())
            ):
                raise ValueError("phase_budget_repair_tests_failed")
            proofs.append(dict(path=str(path), checksum=checksum(path.read_bytes())))
        if len(proofs) < 2:
            raise ValueError("phase_budget_repair_tests_required")
        value = dict(
            schema_version="pre-wire-phase-budget-repair/1.0.0",
            campaign_id=p.name,
            run_id=rid,
            entry_checksum=checksum((p / "entry.json").read_bytes()),
            protocol_checksum=checksum((p / "protocol.json").read_bytes()),
            failure_checksum=checksum((p / "stage-failure.json").read_bytes()),
            request_path=str(requests[0]),
            request_checksum=checksum(requests[0].read_bytes()),
            implementation_before=before,
            implementation_after=after,
            delta=sorted(changed),
            tests=proofs,
            deadline=protocol["deadline"],
            requests=0,
            unknown_replay_allowed=False,
        )
        name, key = BUDGET_NAME, "multimodal:phase-budget-repair"
        if supersede:
            value.update(
                schema_version="pre-wire-phase-budget-repair/1.1.0",
                previous_amendment_checksum=checksum((p / BUDGET_NAME).read_bytes()),
                definition_restore_only=True,
            )
            name, key = BUDGET_REVISION, key + ":r02"
        with docs.session():
            if sessions.read(rid) is not None or RuntimeAgentJournal(docs).read():
                raise ValueError("phase_budget_repair_started_or_unknown")
            docs.write(key, (value,))
        write_once(p / name, value)
        read_budget_amendment(p)
        return dict(amendment=str(p / name), api_requests=0, deadline=value["deadline"])


def check_delta(before, after):
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if not changed or not changed <= ALLOWED or any(key not in after for key in before):
        raise ValueError("review_context_repair_uncovered_delta")
    return sorted(changed)


def read_amendment(output):
    from .implementation import freeze

    p = Path(output)
    path = p / (REVISION if (p / REVISION).exists() else NAME)
    value = json.loads(path.read_bytes())
    if path.name == REVISION and value.get("supersedes") != checksum((p / NAME).read_bytes()):
        raise ValueError("review_context_repair_previous_binding")
    if (
        value["schema_version"] != VERSION
        or value["campaign_id"] != p.name
        or value["entry_checksum"] != checksum((p / "entry.json").read_bytes())
        or value["protocol_checksum"] != checksum((p / "protocol.json").read_bytes())
        or value["failure_checksum"] != checksum((p / "stage-closure-error.json").read_bytes())
        or value["implementation_after"] != freeze()
        or check_delta(value["implementation_before"], value["implementation_after"])
        != value["delta"]
    ):
        raise ValueError("review_context_repair_binding")
    for proof in value["tests"]:
        if checksum(Path(proof["path"]).read_bytes()) != proof["checksum"]:
            raise ValueError("review_context_repair_proof_changed")
    for phase in value["phases"]:
        if checksum(Path(phase["path"]).read_bytes()) != phase["checksum"]:
            raise ValueError("review_context_repair_phase_changed")
    return value


def authorize(output, tests):
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    from .entry import owners
    from .implementation import freeze

    p = Path(output).absolute()
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("review_context_repair_symlink")
    with (p / "lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads((p / "entry.json").read_bytes())
        protocol = json.loads((p / "protocol.json").read_bytes())
        failure = json.loads((p / "stage-closure-error.json").read_bytes())
        if (
            spec["schema_version"]
            not in {"multimodal-live-entry/1.18.0", "multimodal-live-entry/1.19.0"}
            or spec["implementation"] != protocol["implementation"]
            or protocol["campaign_id"] != p.name
            or datetime.now(UTC) >= datetime.fromisoformat(protocol["deadline"])
            or any(
                (p / n).exists() for n in (REVISION, "cancel", "result.json", "stage-failure.json")
            )
            or failure
            != dict(
                schema_version="multimodal-local-closure-error/1.0.0",
                primary="local_owner_error",
                secondary="local_owner_error",
                stage="review",
                durable_commit="unconfirmed",
                automatic_replay_authorized=False,
                complete=False,
            )
        ):
            raise ValueError("review_context_repair_not_unstarted_local_failure")
        after = freeze()
        delta = check_delta(spec["implementation"], after)
        runtime, store, _, target = owners()
        contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)

        def documents(rid):
            context = contexts.read_research_run(rid)
            root = runtime.read_case_artifact(
                context.request.scope.case_code, context.request.artifacts[0].asset_code
            )
            return RuntimeDocuments(
                repo=runtime,
                contexts=contexts,
                sessions=sessions,
                store=store,
                run_id=rid,
                root=root,
            )

        rid = identity("RUN", "task-research:mm-live:" + digest((p.name, "review"))[7:55])
        docs = documents(rid)
        req = contexts.read_research_run(rid).request
        if (
            req.input_checksum != protocol["input"]
            or req.task.checksum != protocol["task"]
            or req.visible_checksum != protocol["visible"]
            or sessions.read(rid) is not None
            or RuntimeAgentJournal(docs).read()
            or any(
                docs.read(k)
                for k in (
                    "dynamic-campaign",
                    "dynamic-phase-input",
                    "dynamic-phase-result",
                    "live-stage-intent:review",
                    "live-stage:review",
                )
            )
            or (p / "review" / "request.json").exists()
        ):
            raise ValueError("review_context_repair_started_or_unknown")
        phases = []
        for path in sorted(p.glob("vision-*/result.json")):
            phase = json.loads(path.read_bytes())
            expected = identity(
                "RUN", "task-research:mm-live:" + digest((p.name, phase["target"]))[7:55]
            )
            old = documents(expected)
            control = sessions.read(expected)
            if (
                phase["run_id"] != expected
                or old.read("dynamic-phase-result") != (phase,)
                or old.read("dynamic-campaign") != (protocol,)
                or control is None
                or control.phase != "stopped"
                or control.pending is not None
            ):
                raise ValueError("review_context_repair_acquisition_binding")
            phases.append(
                dict(path=str(path), checksum=checksum(path.read_bytes()), target=phase["target"])
            )
        if not phases:
            raise ValueError("review_context_repair_no_confirmed_phases")
        proofs = []
        for filename in tests:
            path = Path(filename).absolute()
            proof = json.loads(path.read_bytes())
            if (
                proof["exit_code"] != 0
                or proof.get("implementation_before") != after
                or proof.get("implementation_after") != after
                or proof["log_checksum"] != checksum(path.with_suffix(".log").read_bytes())
            ):
                raise ValueError("review_context_repair_tests_failed")
            proofs.append(dict(path=str(path), checksum=checksum(path.read_bytes())))
        if len(proofs) < 2:
            raise ValueError("review_context_repair_tests_required")
        value = dict(
            schema_version=VERSION,
            campaign_id=p.name,
            entry_checksum=checksum((p / "entry.json").read_bytes()),
            protocol_checksum=checksum((p / "protocol.json").read_bytes()),
            failure_checksum=checksum((p / "stage-closure-error.json").read_bytes()),
            implementation_before=spec["implementation"],
            implementation_after=after,
            tests=proofs,
            delta=delta,
            phases=phases,
            review_run_id=rid,
            deadline=protocol["deadline"],
            max_requests=protocol["max_new_requests"],
            new_acquisition_allowed=False,
            retry_unknown_allowed=False,
        )
        name = REVISION if (p / NAME).exists() else NAME
        if name == REVISION:
            value["supersedes"] = checksum((p / NAME).read_bytes())
        write_once(p / name, value)
        return dict(amendment=str(p / name), deadline=value["deadline"], api_requests=0)


def execute(campaign):
    """Called under original run lock; never resume a started/unknown Review."""
    from .stage_failure import unstarted_phase
    from .stages import confirmed_phase

    value = read_amendment(campaign.output)
    # Saved revision's current source/definition checks remain active, but a
    # closed definition-stage dispatch guard must not authorize this Review.
    # Review uses its own original campaign/current-input/cancel/budget guards.
    campaign.expand_definitions(restore_only=True, report_only=True)
    phases = [confirmed_phase(campaign, p["target"]) for p in value["phases"]]
    if any(p is None for p in phases):
        raise ValueError("review_context_repair_phase_missing")
    review = confirmed_phase(campaign, None)
    if review is not None:
        return campaign.finish(phases + [review])  # Zero-request confirmed local closure.
    rid = value["review_run_id"]
    if not unstarted_phase(campaign, None, rid, campaign.documents(rid)):
        raise ValueError("review_context_repair_started_or_unknown")
    campaign.guard()
    # Validate the entire actual request state before reading a key or dispatching.
    phase = campaign.phase_class(
        campaign.base,
        target=None,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    campaign.register_phase(phase, campaign.request_key(None))
    campaign.guard()
    spec = importlib.util.spec_from_file_location("private_launcher", ".local/vectorengine/run.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    name = campaign.config.transport.api_key_env
    previous = os.environ.get(name)
    try:
        os.environ[name] = launcher.load_secret()
        review = campaign.execute_phase(None)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    return campaign.finish(phases + [review])

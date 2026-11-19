"""Original terminal supervision for a locked, stopped Source backlog successor.

Keeps the original cohort/command immutable after a narrow implementation repair.
The original definition continuation owns history, budget and the unique claim.
"""

import sys
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

from . import terminal_cohort as terminal

REPAIR_FILES = {
    "apps/case_cohort.py",
    "apps/multimodal_case.py",
    "packages/evaluation_preparation/terminal_cohort.py",
    "packages/evaluation_preparation/terminal_source_recovery.py",
    "packages/multimodal_research/execution_lease.py",
    "packages/multimodal_research/entry.py",
    "packages/multimodal_research/discovery_continuation.py",
}


def prepare_command(manifest, case, cohort, output):
    from packages.multimodal_research.execution_lease import SOURCE_PROFILE

    argv = terminal.commands(manifest, case, cohort, "prepare")[0]
    index = argv.index("--output") + 1
    parent = Path(argv[index]).resolve()
    child = output / ("source-backlog-" + digest((str(cohort), case.id, str(output)))[7:39])
    argv[index] = str(child)
    argv[argv.index("--entry-execution-lock") + 1] = SOURCE_PROFILE
    argv += [
        "--definition-failure-of",
        str(parent),
        "--defer-failed-definition-expansion",
        "--retain-closed-public-wire-failures",
    ]
    return argv, parent, child


def run(cohort, case_id, output, phase):
    from packages.multimodal_research.accepted_aggregation import checked_path

    if phase not in {"prepare", "preflight", "run", "export"}:
        raise ValueError("source_runtime_unknown_phase")
    cohort, manifest, old_freeze = terminal.load(cohort, check_current=False)
    selected = [case for case in manifest.cases if case.id == case_id]
    if not isinstance(manifest, terminal.LocalProviderManifest) or len(selected) != 1:
        raise ValueError("source_runtime_original_cohort_required")
    case = selected[0]
    if not terminal.source_recovery_ready(cohort, manifest, case, runtime_backlog=True):
        raise ValueError("source_runtime_original_terminal_failure_required")
    current = terminal.implementation()
    if old_freeze.get("runtime_owner") != terminal.runtime_owner_implementation():
        raise ValueError("source_runtime_original_runtime_owner_changed")
    previous = old_freeze["implementation"]
    changed = {p for p in set(previous) | set(current) if previous.get(p) != current.get(p)}
    if not changed <= REPAIR_FILES:
        raise ValueError("source_runtime_unrelated_implementation_delta")
    if any(
        checksum(terminal.public_config(p).read_bytes()) != expected
        for p, expected in old_freeze["configurations"].items()
    ):
        raise ValueError("source_runtime_configuration_changed")
    out = checked_path(output)
    argv, parent, child = prepare_command(manifest, case, cohort, out)
    for action in ("prepare", "preflight"):
        folder = cohort / "commands" / case.id / action
        receipt = terminal.read(folder / "result.json")
        if (
            receipt.get("exit_code") != 0
            or receipt.get("status") != "command_exited"
            or receipt.get("command") != terminal.commands(manifest, case, cohort, action)[0]
            or receipt.get("log_checksum") != checksum((folder / "stdout.log").read_bytes())
        ):
            raise ValueError("source_runtime_original_preparation_unconfirmed")
    if any(out == p or out in p.parents or p in out.parents for p in (cohort, parent)):
        raise ValueError("source_runtime_output_overlaps_history")
    names = ("entry.json", "protocol.json", "stage-failure.json")
    record = dict(
        schema_version="terminal-source-runtime-recovery/1.1.0",
        original_cohort=str(cohort),
        original_freeze_checksum=digest(old_freeze),
        original_case=case.id,
        parent=str(parent),
        child=str(child),
        parent_files={n: checksum((parent / n).read_bytes()) for n in names},
        implementation=current,
        changed_files=sorted(changed),
        mode="original_deadline_backlog_no_source_replay",
        automatic_unknown_replay=False,
        prepare_command=argv,
    )
    metadata = out / "source-recovery.json"
    if out.exists() and (not metadata.is_file() or terminal.read(metadata) != record):
        raise ValueError("source_runtime_recovery_freeze_conflict")
    if phase != "prepare" and not metadata.is_file():
        raise ValueError("source_runtime_prepare_required")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Shared original supervisor excludes all other original-cohort operations;
    # the child retains its own Runtime lease and unique domain successor claim.
    with terminal.runtime_supervision_session(cohort) as parent_guard:
        with terminal.runtime_supervision_session(out) as child_guard:

            def guard():
                parent_guard()
                child_guard()

            guard()
            terminal.check_pending_commands(cohort)
            terminal.check_pending_commands(out)
            if terminal.implementation() != current:
                raise ValueError("source_runtime_implementation_changed")
            prerequisites = {
                "prepare": (),
                "preflight": ("prepare",),
                "run": ("prepare", "preflight"),
                "export": ("prepare",),
            }[phase]
            for action in prerequisites:
                path = out / "commands" / case.id / action / "result.json"
                if not path.exists() or terminal.read(path).get("exit_code") != 0:
                    raise ValueError("source_runtime_prerequisite_not_completed")
            terminal.publish_json(metadata, record)
            if phase != "prepare":
                argv = [
                    sys.executable,
                    "-u",
                    "-m",
                    "apps.multimodal_case",
                    "restore-report" if phase == "export" else phase,
                    "--input",
                    str(child),
                ]
            receipt = terminal.operation(
                out, case, phase, argv, manifest.command_timeout_seconds, lease_guard=guard
            )
    return dict(
        case=case.id,
        action=phase,
        **receipt,
        child=str(child),
        original_unknown_preserved=True,
        whole_case_gate_passed=False,
    )

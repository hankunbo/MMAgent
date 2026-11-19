"""Explicit tested-code repair before a campaign has registered any phase.

Keeps the original entry, native results, clock and failure immutable. This is
not permission to replay an Agent request or amend an already started protocol.
"""

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.object_store import checksum

NAME = "assembly-repair.json"
ALLOWED = {
    "packages/multimodal_research/campaign.py",
    "packages/multimodal_research/entry.py",
    "packages/multimodal_research/report_recovery.py",
    "packages/multimodal_research/assembly_recovery.py",
    "packages/multimodal_research/implementation.py",
    "packages/multimodal_research/entry_preflight.py",
    "apps/multimodal_case.py",
}


def read(path):
    return json.loads(Path(path).read_bytes())


def unchanged_proofs(paths):
    proofs = []
    for filename in paths:
        path = Path(filename).absolute()
        value = read(path)
        if value.get("exit_code") != 0:
            raise ValueError("assembly_repair_test_failed")
        from .implementation import freeze

        if value.get("implementation_after") != freeze():
            raise ValueError("assembly_repair_test_implementation")
        proofs.append(dict(path=str(path), checksum=checksum(path.read_bytes())))
    if len({p["path"] for p in proofs}) < 2:
        raise ValueError("assembly_repair_targeted_and_static_required")
    return proofs


def validate_candidate(path, current):
    spec = read(path / "entry.json")
    original = spec["implementation"]
    delta = sorted(k for k in original.keys() | current.keys() if original.get(k) != current.get(k))
    start = read(path / "active-start.json")
    if (
        spec["schema_version"] != "multimodal-live-entry/1.29.0"
        or spec["provider"]["network_route"] != spec["network_route"]
        or not delta
        or not set(delta) <= ALLOWED
        or start["input_checksum"] != spec["input_checksum"]
        or datetime.now(UTC) >= datetime.fromisoformat(start["deadline"])
        or any(
            (path / name).exists()
            for name in ("protocol.json", "stage-failure.json", "result.json", "cancel", NAME)
        )
        or list(path.glob("**/request.json"))
    ):
        raise ValueError("assembly_repair_not_unstarted")
    failures = list(path.glob("entry-failure-*.json"))
    eligible = [
        p
        for p in failures
        if read(p).get("phase") == "owner_assembly"
        and read(p).get("error_code") == "qualified_provider_not_installed"
    ]
    if len(eligible) != 1:
        raise ValueError("assembly_repair_failure_required")
    discovery = read(path / "active-discovery.json")
    if discovery["queries"] or discovery["status"] != "deferred_to_campaign_revision":
        raise ValueError("assembly_repair_discovery_started")
    return spec, start, delta, eligible[0]


def confirm_no_phases(path, *, contexts, sessions):
    # Before definition expansion, only frozen plan targets and these two owners
    # can register a phase. No generic scan for an arbitrary "accepted" object.
    plan = read(path / "active-plan.json")
    targets = {row["id"] for row in plan["rows"]} | {"review", "definition-expansion"}
    checked = []
    for target in sorted(targets):
        key = "mm-live:" + digest((path.name, target))[7:55]
        run_id = identity("RUN", "task-research:" + key)
        if contexts.read_research_run(run_id, missing_ok=True) is not None:
            raise ValueError("assembly_repair_phase_registered")
        if sessions.read(run_id) is not None:
            raise ValueError("assembly_repair_control_present")
        checked.append(run_id)
    return checked


def authorize(output, tests):
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository

    from .entry import owners
    from .implementation import freeze

    path = Path(output).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("assembly_repair_symlink")
    with (path / "assembly.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = freeze()
        spec, start, delta, failure = validate_candidate(path, current)
        proofs = unchanged_proofs(tests)
        _, _, _, target = owners()
        run_ids = confirm_no_phases(
            path, contexts=ResearchRunRepository(target), sessions=PlannerRepository(target)
        )
        protected = [
            path / name
            for name in (
                "entry.json",
                "active-start.json",
                "active-plan.json",
                "active-discovery.json",
            )
        ] + [failure]
        protected += [p for p in (path / "native-fundus").rglob("*") if p.is_file()]
        if any(p.is_symlink() for p in protected):
            raise ValueError("assembly_repair_symlink")
        value = dict(
            schema_version="pre-campaign-assembly-repair/1.0.0",
            campaign_id=path.name,
            implementation_before=spec["implementation"],
            implementation_after=current,
            delta=delta,
            tests=proofs,
            protected={str(p.relative_to(path)): checksum(p.read_bytes()) for p in protected},
            deadline=start["deadline"],
            max_requests=spec["max_requests"],
            unregistered_phase_ids=run_ids,
            confirmed_prior_model_requests=0,
            native_reuse="original_owner_recheck_required",
        )
        # A second full identity check before publishing a recovery permission.
        validate_candidate(path, current)
        write_once(path / NAME, value)
        return dict(
            authorized=True, api_requests=0, deadline=start["deadline"], checksum=digest(value)
        )


def read_amendment(output):
    from .implementation import freeze

    path = Path(output)
    value = read(path / NAME)
    before, after = value["implementation_before"], freeze()
    delta = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if (
        value["schema_version"] != "pre-campaign-assembly-repair/1.0.0"
        or value["campaign_id"] != path.name
        or value["implementation_after"] != after
        or value["delta"] != delta
        or not delta
        or not set(delta) <= ALLOWED
        or read(path / "entry.json")["implementation"] != before
        or any(checksum((path / p).read_bytes()) != h for p, h in value["protected"].items())
        or any(checksum(Path(p["path"]).read_bytes()) != p["checksum"] for p in value["tests"])
    ):
        raise ValueError("assembly_repair_binding_changed")
    return value

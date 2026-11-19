"""Explicit one-time repair of a locally failed, never-dispatched search assembly.

Not a transport retry. A reserved wire, any Agent request, a current Control, a
result or cancellation refuses admission. Original deadline and campaign remain.
"""

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.object_store import checksum


def read_amendment(output):
    p = Path(output)
    v = json.loads((p / "local-search-repair.json").read_bytes())
    if (
        v["schema_version"] != "pre-wire-search-repair/1.0.0"
        or v["campaign_id"] != p.name
        or v["entry_checksum"] != checksum((p / "entry.json").read_bytes())
        or v["protocol_checksum"] != checksum((p / "protocol.json").read_bytes())
        or v["failure_checksum"] != checksum(Path(v["failure_log"]).read_bytes())
    ):
        raise ValueError("local_search_repair_binding")
    for proof in v["tests"]:
        if checksum(Path(proof["path"]).read_bytes()) != proof["checksum"]:
            raise ValueError("local_search_repair_test_changed")
    return v


def pre_wire_journal(events, operations, *, run_id, deadline):
    expected = dict(
        schema_version="agent-budget/1.0.0",
        binding=dict(run_id=run_id, deadline=deadline, max_requests=40),
    )
    return list(events) == [expected] and [o["kind"] for o in operations] == [
        "local_search",
        "search_group",
    ]


def authorize(output, *, failure_log, tests):
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    from .entry import owners
    from .implementation import freeze

    p = Path(output).absolute()
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("local_search_repair_symlink")
    with (p / "lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads((p / "entry.json").read_bytes())
        protocol = json.loads((p / "protocol.json").read_bytes())
        failure = Path(failure_log).absolute()
        if (
            spec["schema_version"] != "multimodal-live-entry/1.11.0"
            or protocol["implementation"] != spec["implementation"]
            or protocol["campaign_id"] != p.name
            or datetime.now(UTC) >= datetime.fromisoformat(protocol["deadline"])
            or any((p / n).exists() for n in ("cancel", "result.json", "local-search-repair.json"))
            or not failure.read_text()
            .rstrip()
            .endswith("ValueError: all clients must use the executor governor")
        ):
            raise ValueError("local_search_repair_not_pre_wire_failure")
        runtime, store, _, target = owners()
        contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
        run_id = identity(
            "RUN", "task-research:mm-live:" + digest((p.name, "definition-expansion"))[7:55]
        )
        context = contexts.read_research_run(run_id)
        root = runtime.read_case_artifact(
            context.request.scope.case_code, context.request.artifacts[0].asset_code
        )
        docs = RuntimeDocuments(
            repo=runtime,
            contexts=contexts,
            sessions=sessions,
            store=store,
            run_id=run_id,
            root=root,
        )
        ops = [docs.read(f"definition:operation:{n}") for n in range(12)]
        operations = [o[0] for o in ops if o]
        events = RuntimeAgentJournal(docs).read()
        if (
            sessions.read(run_id) is not None
            or not pre_wire_journal(
                events, operations, run_id=run_id, deadline=protocol["deadline"]
            )
            or any(docs.read(o["key"] + ":result") for o in operations)
            or docs.read("definition:revision")
            or context.request.input_checksum != spec["input_checksum"]
        ):
            raise ValueError("local_search_repair_dispatch_or_unknown")
        for t in (*protocol["targets"], None):
            rid = identity("RUN", "task-research:mm-live:" + digest((p.name, t or "review"))[7:55])
            try:
                contexts.read_research_run(rid)
            except ValueError as exc:
                if str(exc) != "research_context_required":
                    raise
            else:
                raise ValueError("local_search_repair_phase_already_registered")
            if sessions.read(rid) is not None:
                raise ValueError("local_search_repair_control_present")
        proofs = []
        for filename in tests:
            file = Path(filename).absolute()
            command = json.loads(file.read_bytes())
            if command["direct_exit"] != 0:
                raise ValueError("local_search_repair_tests_failed")
            proofs.append(dict(path=str(file), checksum=checksum(file.read_bytes())))
        if len(proofs) < 2:
            raise ValueError("local_search_repair_targeted_and_static_required")
        value = dict(
            schema_version="pre-wire-search-repair/1.0.0",
            campaign_id=p.name,
            entry_checksum=checksum((p / "entry.json").read_bytes()),
            protocol_checksum=checksum((p / "protocol.json").read_bytes()),
            failure_log=str(failure),
            failure_checksum=checksum(failure.read_bytes()),
            implementation_before=spec["implementation"],
            implementation_after=freeze(),
            tests=proofs,
            operations=operations,
            run_id=run_id,
            deadline=protocol["deadline"],
            max_requests=protocol["max_new_requests"],
            confirmed_pre_repair_wire_requests=0,
            model_outputs_reused=True,
        )
        write_once(p / "local-search-repair.json", value)
        return dict(
            authorized=True,
            campaign_id=p.name,
            deadline=protocol["deadline"],
            api_requests=0,
            amendment_checksum=digest(value),
        )


def admit_operation(expansion, key, ops):
    """Retain the old reservation and add one immutable repair reservation."""
    v = read_amendment(expansion.campaign.output)
    from .implementation import freeze

    old = v["operations"]
    eligible = [o for o in old if o["key"] == key and o["binding"] == expansion.binding]
    repairs = [o for o in ops[len(old) :] if o["kind"] == "local_repair"]
    if (
        v["implementation_after"] != freeze()
        or v["run_id"] != expansion.scope.run_id
        or len(eligible) != 1
        or ops[: len(old)] != old
        or len(repairs) != len(ops) - len(old)
        or any(o["key"] == key + ":local-repair" for o in repairs)
        or len(ops) >= expansion.policy.max_operations
        or expansion.budget.used
    ):
        raise ValueError("local_search_repair_current_binding")
    expansion.documents.write("definition:local-search-repair", (v,))
    expansion.documents.write(
        f"definition:operation:{len(ops)}",
        (
            dict(
                key=key + ":local-repair",
                kind="local_repair",
                binding=expansion.binding,
                amendment_checksum=digest(v),
            ),
        ),
    )

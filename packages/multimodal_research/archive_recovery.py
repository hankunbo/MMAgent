"""One local archive repair over confirmed search/fetch, no transport replay.

The original campaign, deadline, operation claims and cumulative budget remain.
Only its incomplete local orchestration slot may be completed after verification.
"""

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .implementation import freeze

MATERIAL_FILTER_VERSION = "confirmed-material-local-repair/1.5.0"
DEFINITION_PROJECTION_VERSION = "confirmed-material-local-repair/1.6.0"
MATERIAL_FILTER_FILES = {
    "packages/evidence_reuse/definition_sources.py",
    "packages/multimodal_research/archive_recovery.py",
    "packages/multimodal_research/campaign.py",
    "packages/multimodal_research/campaign_renewal.py",
    "packages/multimodal_research/definition_expansion.py",
    "packages/multimodal_research/stage_failure.py",
    "apps/multimodal_case.py",
}


def authorize_definition_projection_guard(campaign, *, proof, authorization):
    """Bridge the exact zero-wire secondary stop; preserve both prior failures."""
    from .pending_review import partition_projection_unstarted
    from .stage_failure import read_saved_failure

    p = campaign.output
    with campaign.execution_session():
        campaign.guard()
        if (
            not authorization.strip()
            or campaign.protocol.get("schema_version") != "multimodal-campaign-renewal/1.56.0"
            or campaign.protocol.get("mode") != "definitions"
            or not campaign.protocol.get("definition_increment_only")
            or any((p / n).exists() for n in ("cancel", "archive-repair.json", "result.json"))
        ):
            raise ValueError("definition_projection_guard_scope")
        docs, context = partition_projection_unstarted(
            campaign, definition=True, budget_binding_only=True
        )
        prior = json.loads((p / "prewire-repair.json").read_bytes())
        closure = json.loads((p / "stage-closure-error.json").read_bytes())
        failure = read_saved_failure(campaign)
        if (
            prior.get("schema_version") != "renewal-prewire-repair/1.2.0"
            or prior.get("reason") != "definition_retained_context_projection"
            or prior.get("protocol") != digest(campaign.protocol)
            or prior.get("context_checksum") != digest(context)
            or prior.get("original_failure_checksum") != digest(closure)
            or prior.get("run_id") != docs.run_id
            or docs.read("renewal:prewire-repair") != (prior,)
            or not failure
            or failure.get("error_code") != "stage_previous_failure_read_only"
            or failure.get("stage") != "definitions"
            or failure.get("failure_kind") != "no_provider_reservation"
            or failure.get("new_requests") != 0
            or failure.get("requests")
            or failure.get("definition_revision_saved")
            or failure.get("scores") is not None
        ):
            raise ValueError("definition_projection_guard_history")
        proof_path = Path(proof).absolute()
        evidence = json.loads(proof_path.read_bytes())
        before, after = campaign.protocol["implementation"], freeze()
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        if before.keys() != after.keys() or changed != {
            "apps/multimodal_case.py",
            "packages/multimodal_research/campaign_renewal.py",
            "packages/multimodal_research/pending_review.py",
            "packages/multimodal_research/archive_recovery.py",
            "packages/multimodal_research/definition_expansion.py",
        }:
            raise ValueError("definition_projection_guard_delta")
        if (
            evidence.get("direct_exit") != 0
            or evidence.get("implementation_before") != before
            or evidence.get("implementation_after") != after
            or not evidence.get("commands")
            or any(
                c.get("direct_exit") != 0
                or c.get("log_checksum") != checksum(Path(c["log_path"]).read_bytes())
                for c in evidence["commands"]
            )
        ):
            raise ValueError("definition_projection_guard_proof")
        value = dict(
            schema_version=DEFINITION_PROJECTION_VERSION,
            protocol_checksum=checksum((p / "protocol.json").read_bytes()),
            implementation_before=before,
            implementation_after=after,
            failure_log=str(p / "stage-failure.json"),
            failure_checksum=checksum((p / "stage-failure.json").read_bytes()),
            original_failure_checksum=digest(failure),
            original_closure_checksum=digest(closure),
            prior_projection_grant=digest(prior),
            context_checksum=digest(context),
            tests=[dict(path=str(proof_path), checksum=checksum(proof_path.read_bytes()))],
            run_id=docs.run_id,
            deadline=campaign.protocol["deadline"],
            max_new_requests=campaign.protocol["max_new_requests"],
            new_wire_requests=0,
            confirmed_wire_requests=0,
            original_operations=digest([]),
            original_agent_journal=digest(RuntimeAgentJournal(docs).read()),
            authorization=authorization,
            local_failure="definition_projection_secondary_local_guard",
        )
        with docs.session():
            docs.write("definition:local-archive-repair", (value,))
        write_once(p / "archive-repair.json", value)
        return dict(
            status="authorized_definition_guard_repair", requests=0, deadline=value["deadline"]
        )


def authorize_material_filter(campaign, *, diagnostic, tests):
    """One current-window local retry; no new campaign, deadline or remote retry."""
    from .stage_failure import check_definition_binding, read_saved_failure

    p = campaign.output
    with campaign.execution_session():
        if (
            campaign.protocol.get("schema_version") != "multimodal-campaign-renewal/1.58.0"
            or campaign.protocol.get("new_public_search_allowed") is not False
            or datetime.now(UTC) >= datetime.fromisoformat(campaign.protocol["deadline"])
            or any((p / n).exists() for n in ("cancel", "result.json", "archive-repair.json"))
        ):
            raise ValueError("material_filter_repair_not_active")
        failure = read_saved_failure(campaign)
        if (
            not failure
            or failure.get("stage") != "definitions"
            or failure.get("error_code") != "local_owner_error"
            or failure.get("failure_kind") != "no_provider_reservation"
            or failure.get("requests")
            or failure.get("new_requests") != 0
            or failure.get("scores") is not None
            or failure.get("definition_revision_saved")
        ):
            raise ValueError("material_filter_repair_failure")
        d = documents(campaign)
        check_definition_binding(campaign, d)
        operations = [v[0] for n in range(100) if (v := d.read(f"definition:operation:{n}"))]
        results = {o["key"]: d.read(o["key"] + ":result") for o in operations}
        outcomes = {o["key"]: d.read(o["key"] + ":outcome") for o in operations}
        events = RuntimeAgentJournal(d).read()
        if (
            not confirmed_index_failure(operations, results, events)
            or campaign.sessions.read(d.run_id) is not None
            or any(
                d.read(k)
                for k in ("definition:revision", "definition:stage:sources", "dynamic-campaign")
            )
            or any(outcomes.values())
        ):
            raise ValueError("material_filter_repair_not_local")
        diagnostic = Path(diagnostic).absolute()
        proof = json.loads(diagnostic.read_bytes())
        if (
            proof.get("error", {}).get("type") != "Rejected"
            or proof["error"].get("safe_code") != "table_unavailable"
            or proof.get("new_requests") != 0
            or proof.get("database_writes") != 0
        ):
            raise ValueError("material_filter_repair_diagnostic")
        before, after = campaign.protocol["implementation"], freeze()
        changed = {k for k in before.keys() & after.keys() if before[k] != after[k]}
        if (
            before.keys() != after.keys()
            or not changed <= MATERIAL_FILTER_FILES
            or "packages/evidence_reuse/definition_sources.py" not in changed
        ):
            raise ValueError("material_filter_repair_implementation_scope")
        proofs = []
        for filename in tests:
            file = Path(filename).absolute()
            record = json.loads(file.read_bytes())
            if record.get("direct_exit", record.get("exit_code")) != 0:
                raise ValueError("material_filter_repair_tests_failed")
            proofs.append(dict(path=str(file), checksum=checksum(file.read_bytes())))
        if len(proofs) < 2:
            raise ValueError("material_filter_repair_tests_required")
        value = dict(
            schema_version=MATERIAL_FILTER_VERSION,
            protocol_checksum=checksum((p / "protocol.json").read_bytes()),
            implementation_before=before,
            implementation_after=after,
            failure_log=str(diagnostic),
            failure_checksum=checksum(diagnostic.read_bytes()),
            original_failure_checksum=digest(failure),
            tests=proofs,
            operations=operations,
            pending_key=operations[0]["key"],
            results_checksum=digest(results),
            outcomes_checksum=digest(outcomes),
            run_id=d.run_id,
            deadline=campaign.protocol["deadline"],
            new_wire_requests=0,
            confirmed_wire_requests=0,
            local_failure="unusable_table_is_not_whole_corpus_failure",
        )
        with d.session():
            d.write("definition:local-archive-repair", (value,))
        write_once(p / "archive-repair.json", value)
        return dict(
            schema_version=MATERIAL_FILTER_VERSION,
            deadline=value["deadline"],
            amendment_checksum=digest(value),
            new_requests=0,
        )


def documents(campaign):
    rid = identity(
        "RUN",
        "task-research:mm-live:" + digest((campaign.campaign_id, "definition-expansion"))[7:55],
    )
    return campaign.documents(rid)


def confirmed_local_failure(operations, results, outcomes, events):
    pending = [o for o in operations if results.get(o["key"]) is None]
    return (
        len(pending) == 1
        and pending[0]["kind"] == "local_search"
        and all(
            o["kind"] in {"local_search", "search_group", "fetch_group", "wire"} for o in operations
        )
        and any(o["kind"] == "fetch_group" for o in operations)
        and any(o["kind"] == "wire" for o in operations)
        and all(
            outcomes.get(o["key"])
            and len(outcomes[o["key"]]) == 1
            and outcomes[o["key"]][0]["http_status"] == 200
            and outcomes[o["key"]][0]["response_checksum"]
            for o in operations
            if o["kind"] == "wire"
        )
        and len(events) == 1
        and events[0].get("schema_version") == "agent-budget/1.0.0"
    )


def confirmed_index_failure(operations, results, events):
    return (
        len(operations) == 1
        and operations[0]["kind"] == "local_search"
        and results.get(operations[0]["key"]) is None
        and len(events) == 1
        and events[0].get("schema_version") == "agent-budget/1.0.0"
    )


def confirmed_proposal_failure(operations, results, events):
    """Only the first, not-yet-reserved model dispatch after confirmed local retrieval."""
    return (
        2 <= len(operations) <= 3
        and operations[-1]["kind"] == "llm"
        and operations[-1]["key"] == "definition:proposal"
        and results.get("definition:proposal") is None
        and all(o["kind"] == "local_search" and results.get(o["key"]) for o in operations[:-1])
        and len(events) == 1
        and events[0].get("schema_version") == "agent-budget/1.0.0"
    )


def authorize(campaign, *, failure_log, tests):
    p = campaign.output
    followup = (p / "archive-repair.json").exists()
    prior = read(campaign, current=False) if followup else None
    with (p / "lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            campaign.protocol["mode"] != "definitions"
            or datetime.now(UTC) >= datetime.fromisoformat(campaign.protocol["deadline"])
            or any(
                (p / n).exists() for n in ("cancel", "result.json", "index-contract-repair.json")
            )
        ):
            raise ValueError("archive_repair_not_active_local_failure")
        failure = Path(failure_log).absolute()
        text = failure.read_text().rstrip()
        table_repair = (
            not followup
            and campaign.protocol.get("definition_increment_only") is True
            and text.endswith("ValueError: table_chunk_limit")
        )
        index_repair = text.endswith("packages.knowledge_claims.evidence.Rejected: legacy_fidelity")
        proposal_repair = (
            not followup
            and campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.5.0"
            and campaign.protocol.get("definition_increment_only") is True
            and campaign.protocol.get("expansion_policy", {}).get("prompt_version")
            == "definition-draft-2.1.0"
            and text.endswith("packages.llm_provider.provider.ProviderError: configuration")
        )
        if proposal_repair:
            previous, current = campaign.protocol["implementation"], freeze()
            changed = {k for k in previous.keys() & current.keys() if previous[k] != current[k]}
            if (
                previous.keys() != current.keys()
                or not changed
                <= {
                    "packages/llm_provider/provider.py",
                    "packages/llm_provider/source_snippets.py",
                    "packages/multimodal_research/campaign.py",
                    "packages/multimodal_research/archive_recovery.py",
                }
                or "packages/llm_provider/provider.py" not in changed
            ):
                raise ValueError("proposal_repair_implementation_scope")
        if followup:
            if (
                prior.get("local_failure") != "index_fidelity"
                or prior["schema_version"] != "confirmed-material-local-repair/1.0.0"
                or not text.endswith(
                    "packages.knowledge_claims.evidence.Rejected: derivation_config"
                )
            ):
                raise ValueError("index_contract_repair_scope")
            index_repair = True
        if (
            not index_repair
            and not table_repair
            and not proposal_repair
            and not text.endswith(
                "KnowledgeConflictError: rejected or incomplete provider run "
                "cannot produce a source"
            )
        ):
            raise ValueError("archive_repair_wrong_failure")
        d = documents(campaign)
        operations = [v[0] for n in range(24) if (v := d.read(f"definition:operation:{n}"))]
        results = {o["key"]: d.read(o["key"] + ":result") for o in operations}
        outcomes = {o["key"]: d.read(o["key"] + ":outcome") for o in operations}
        events = RuntimeAgentJournal(d).read()
        if (
            campaign.sessions.read(d.run_id) is not None
            or d.read("definition:revision") is not None
            or not (
                confirmed_proposal_failure(operations, results, events)
                if proposal_repair
                else confirmed_index_failure(operations, results, events)
                if index_repair and campaign.protocol.get("continuation")
                else confirmed_local_failure(operations, results, outcomes, events)
            )
        ):
            raise ValueError("archive_repair_unknown_or_agent_already_started")
        proofs = []
        for filename in tests:
            path = Path(filename).absolute()
            proof = json.loads(path.read_bytes())
            if proof.get("direct_exit", proof.get("exit_code")) != 0:
                raise ValueError("archive_repair_tests_failed")
            proofs.append(dict(path=str(path), checksum=checksum(path.read_bytes())))
        if len(proofs) < 2:
            raise ValueError("archive_repair_tests_and_static_required")
        value = dict(
            schema_version="confirmed-material-local-repair/1.0.0",
            protocol_checksum=checksum((p / "protocol.json").read_bytes()),
            implementation_before=campaign.protocol["implementation"],
            implementation_after=freeze(),
            failure_log=str(failure),
            failure_checksum=checksum(failure.read_bytes()),
            tests=proofs,
            operations=operations,
            pending_key=next(k for k, v in results.items() if v is None),
            results_checksum=digest(results),
            outcomes_checksum=digest(outcomes),
            run_id=d.run_id,
            deadline=campaign.protocol["deadline"],
            new_wire_requests=0,
            confirmed_wire_requests=sum(o["kind"] == "wire" for o in operations),
            local_failure="index_fidelity" if index_repair else "bounded_archive",
        )
        if followup:
            value.update(
                schema_version="confirmed-material-local-repair/1.2.0",
                previous_amendment_checksum=digest(prior),
                implementation_before=prior["implementation_after"],
                local_failure="index_json_contract",
            )
        if table_repair:
            value.update(
                schema_version="confirmed-material-local-repair/1.3.0",
                local_failure="table_chunk_limit",
                selection="non-table-blocks/1.0.0",
            )
        if proposal_repair:
            value.update(
                schema_version="confirmed-material-local-repair/1.4.0",
                local_failure="definition_prompt_dispatch_before_first_reservation",
                prompt_version="definition-draft-2.1.0",
                original_model_requests=0,
            )
        with d.session():
            d.write(
                "definition:index-contract-repair"
                if followup
                else "definition:local-archive-repair",
                (value,),
            )
        write_once(p / ("index-contract-repair.json" if followup else "archive-repair.json"), value)
        return value


def read(campaign, *, current=True):
    p = campaign.output
    value = json.loads((p / "archive-repair.json").read_bytes())
    d = documents(campaign)
    if (
        value["schema_version"]
        not in {
            "confirmed-material-local-repair/1.0.0",
            "confirmed-material-local-repair/1.3.0",
            "confirmed-material-local-repair/1.4.0",
            MATERIAL_FILTER_VERSION,
            DEFINITION_PROJECTION_VERSION,
        }
        or value["protocol_checksum"] != checksum((p / "protocol.json").read_bytes())
        or value["implementation_before"] != campaign.protocol["implementation"]
        or value["failure_checksum"] != checksum(Path(value["failure_log"]).read_bytes())
        or value["deadline"] != campaign.protocol["deadline"]
        or value["run_id"] != d.run_id
        or d.read("definition:local-archive-repair") != (value,)
        or any(checksum(Path(t["path"]).read_bytes()) != t["checksum"] for t in value["tests"])
    ):
        raise ValueError("archive_repair_binding")
    second = p / "local-operation-repair.json"
    if second.exists():
        next_value = json.loads(second.read_bytes())
        if (
            next_value["previous_amendment_checksum"] != digest(value)
            or next_value["implementation_before"] != value["implementation_after"]
            or next_value["protocol_checksum"] != value["protocol_checksum"]
            or next_value["deadline"] != value["deadline"]
            or next_value["local_operation_limit"] != 24
            or next_value["max_external_requests"] != campaign.protocol["max_new_requests"]
            or d.read("definition:local-operation-repair") != (next_value,)
            or checksum(Path(next_value["failure_log"]).read_bytes())
            != next_value["failure_checksum"]
            or any(
                checksum(Path(t["path"]).read_bytes()) != t["checksum"] for t in next_value["tests"]
            )
        ):
            raise ValueError("local_operation_repair_binding")
        value = next_value
    contract = p / "index-contract-repair.json"
    if contract.exists():
        next_value = json.loads(contract.read_bytes())
        if (
            next_value["schema_version"] != "confirmed-material-local-repair/1.2.0"
            or next_value["previous_amendment_checksum"] != digest(value)
            or next_value["implementation_before"] != value["implementation_after"]
            or next_value["protocol_checksum"] != value["protocol_checksum"]
            or next_value["deadline"] != value["deadline"]
            or d.read("definition:index-contract-repair") != (next_value,)
            or checksum(Path(next_value["failure_log"]).read_bytes())
            != next_value["failure_checksum"]
            or any(
                checksum(Path(t["path"]).read_bytes()) != t["checksum"] for t in next_value["tests"]
            )
        ):
            raise ValueError("index_contract_repair_binding")
        value = next_value
    if current and value["implementation_after"] != freeze():
        raise ValueError("archive_repair_implementation_changed")
    if value["schema_version"] in {MATERIAL_FILTER_VERSION, DEFINITION_PROJECTION_VERSION}:
        from .stage_failure import read_saved_failure

        if digest(read_saved_failure(campaign)) != value["original_failure_checksum"]:
            raise ValueError("material_filter_repair_failure_changed")
    if value["schema_version"] == DEFINITION_PROJECTION_VERSION:
        prior = json.loads((p / "prewire-repair.json").read_bytes())
        if (
            digest(prior) != value["prior_projection_grant"]
            or d.read("renewal:prewire-repair") != (prior,)
            or digest(json.loads((p / "stage-closure-error.json").read_bytes()))
            != value["original_closure_checksum"]
            or value["max_new_requests"] != campaign.protocol["max_new_requests"]
            or value["original_operations"] != digest([])
        ):
            raise ValueError("definition_projection_guard_amendment")
    return value


def admit(expansion, key, operations):
    value = read(expansion.campaign)
    d = expansion.documents
    if value["schema_version"] == "confirmed-material-local-repair/1.1.0":
        marker = "definition:local-operation-repair-started:" + digest(key)[7:31]
        if (
            key not in value["pending_keys"]
            or operations[: len(value["operations"])] != value["operations"]
            or any(d.read(k + ":result") != tuple(v) for k, v in value["confirmed_results"].items())
            or any(
                d.read(k + ":outcome") != tuple(v) for k, v in value["confirmed_outcomes"].items()
            )
            or expansion.budget.used
            or d.read(marker) is not None
        ):
            raise ValueError("local_operation_repair_not_local")
        expansion.control()
        d.write(marker, (dict(amendment=digest(value), key=key),))
        return
    results = {o["key"]: d.read(o["key"] + ":result") for o in operations}
    outcomes = {o["key"]: d.read(o["key"] + ":outcome") for o in operations}
    marker = (
        "definition:index-contract-repair-started"
        if value["schema_version"] == "confirmed-material-local-repair/1.2.0"
        else "definition:local-archive-repair-started"
    )
    if (
        key != value["pending_key"]
        or operations != value["operations"]
        or digest(results) != value["results_checksum"]
        or digest(outcomes) != value["outcomes_checksum"]
        or expansion.budget.used
        or d.read(marker) is not None
    ):
        raise ValueError("archive_repair_not_confirmed_local_slot")
    expansion.control()
    d.write(marker, (dict(amendment=digest(value), key=key),))


def exhausted_before_wire(operations, results, outcomes, events):
    """Exact bounded gate shape: both pending slots are local; no unconfirmed wire."""
    pending = [o for o in operations if results.get(o["key"]) is None]
    return (
        len(operations) == 12
        and [o["kind"] for o in pending] == ["local_search", "fetch_group"]
        and operations[-1] == pending[-1]
        and all(
            o["kind"] in {"local_search", "search_group", "fetch_group", "wire"} for o in operations
        )
        and all(
            outcomes.get(o["key"])
            and outcomes[o["key"]][0]["http_status"] == 200
            and outcomes[o["key"]][0]["response_checksum"]
            for o in operations
            if o["kind"] == "wire"
        )
        and len(events) == 1
        and events[0].get("schema_version") == "agent-budget/1.0.0"
    )


def authorize_operation_repair(campaign, *, failure_log, tests):
    p = campaign.output
    with (p / "lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read(campaign, current=False)
        if (
            prior["schema_version"] != "confirmed-material-local-repair/1.0.0"
            or datetime.now(UTC) >= datetime.fromisoformat(campaign.protocol["deadline"])
            or any(
                (p / n).exists() for n in ("cancel", "result.json", "local-operation-repair.json")
            )
        ):
            raise ValueError("local_operation_repair_not_active")
        d = documents(campaign)
        ops = [v[0] for n in range(12) if (v := d.read(f"definition:operation:{n}"))]
        results = {o["key"]: d.read(o["key"] + ":result") for o in ops}
        outcomes = {o["key"]: d.read(o["key"] + ":outcome") for o in ops}
        if (
            campaign.sessions.read(d.run_id) is not None
            or d.read("definition:revision")
            or not exhausted_before_wire(ops, results, outcomes, RuntimeAgentJournal(d).read())
        ):
            raise ValueError("local_operation_repair_unconfirmed")
        failure = Path(failure_log).absolute()
        if not failure.read_text().rstrip().endswith("ConnectorError: security"):
            raise ValueError("local_operation_repair_wrong_failure")
        proofs = []
        for filename in tests:
            f = Path(filename).absolute()
            if json.loads(f.read_bytes())["direct_exit"] != 0:
                raise ValueError("local_operation_repair_test_failed")
            proofs.append(dict(path=str(f), checksum=checksum(f.read_bytes())))
        if len(proofs) < 2:
            raise ValueError("local_operation_repair_tests_required")
        value = dict(
            schema_version="confirmed-material-local-repair/1.1.0",
            previous_amendment_checksum=digest(prior),
            protocol_checksum=prior["protocol_checksum"],
            implementation_before=prior["implementation_after"],
            implementation_after=freeze(),
            deadline=prior["deadline"],
            max_external_requests=campaign.protocol["max_new_requests"],
            local_operation_limit=24,
            prior_local_operation_limit=12,
            operations=ops,
            pending_keys=[k for k, v in results.items() if v is None],
            confirmed_results={k: v for k, v in results.items() if v is not None},
            confirmed_outcomes={k: v for k, v in outcomes.items() if v is not None},
            tests=proofs,
            failure_log=str(failure),
            failure_checksum=checksum(failure.read_bytes()),
            no_prior_wire_replay=True,
        )
        with d.session():
            d.write("definition:local-operation-repair", (value,))
        write_once(p / "local-operation-repair.json", value)
        return value

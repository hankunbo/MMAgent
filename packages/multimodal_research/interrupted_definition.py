"""Bounded successor over confirmed pre-checkpoint public work, never replay.

Runtime owns history and the single successor claim; Knowledge owns every index
and current material read. This module supplies no medical or scoring decisions.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.knowledge_connectors.ports import WireReceipt
from packages.research_runtime.agent_records import RuntimeAgentJournal

VERSION = "multimodal-campaign-renewal/1.58.0"
LOCAL_VERSION = "multimodal-campaign-renewal/1.62.0"
VERSIONS = {VERSION, LOCAL_VERSION}
KIND = "interrupted-definition-history/1.0.0"
HISTORY_KIND = "interrupted_public_material_deferred"
LOCAL_KIND = "confirmed-public-local-failure/1.0.0"
LOCAL_HISTORY_KIND = "confirmed_public_local_failure_deferred"
REPAIR_KEY = "definition:confirmed-local-index-repair"


def tip(parent, path, *, local_failure=False, allow_pending=False):
    from .campaign_renewal import RenewedCampaign
    from .stage_failure import check_definition_binding, read_saved_failure

    path = Path(path).absolute()
    plan = json.loads((path / "protocol.json").read_bytes())
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or (path / "result.json").exists()
        or (path / "cancel").exists()
        or plan.get("schema_version") != "multimodal-campaign-renewal/1.56.0"
        or plan.get("mode") != "definitions"
        or not plan.get("definition_increment_only")
        or plan.get("facts") != str(parent.output.absolute())
        or plan.get("input") != parent.base.manifest.checksum
        or plan.get("task") != parent.base.task.checksum
        or plan.get("provider") != digest(parent.config)
    ):
        raise ValueError("interrupted_definition_origin")
    campaign = RenewedCampaign(parent, path, plan)
    failure = read_saved_failure(campaign)
    if (
        not failure
        or failure.get("error_code")
        != ("local_owner_error" if local_failure else "execution_process_interrupted")
        or failure.get("stage") != "definitions"
        or failure.get("failure_kind") != "no_provider_reservation"
        or failure.get("requests")
        or failure.get("scores") is not None
        or failure.get("definition_revision_saved")
    ):
        raise ValueError("interrupted_definition_failure_binding")
    docs = campaign.documents(failure["run_id"])
    check_definition_binding(campaign, docs)
    for key in (
        "dynamic-campaign",
        "definition:stage:sources",
        "definition:stage:definitions",
        "definition:revision",
    ):
        if docs.read(key) is not None:
            raise ValueError("interrupted_definition_stage_already_confirmed")
    events = RuntimeAgentJournal(docs).read()
    if (
        list(events)
        != [
            dict(
                schema_version="agent-budget/1.0.0",
                binding=dict(
                    run_id=docs.run_id,
                    deadline=plan["deadline"],
                    max_requests=campaign.config.max_requests,
                ),
            )
        ]
        or campaign.sessions.read(docs.run_id) is not None
    ):
        raise ValueError("interrupted_definition_provider_or_control_started")
    for target in (*campaign.targets, None):
        rid = identity("RUN", "task-research:" + campaign.request_key(target))
        if campaign.contexts.read_research_run(rid, missing_ok=True) is not None:
            raise ValueError("interrupted_definition_patient_phase_started")
    records = docs.read_many(
        tuple(
            f"definition:operation:{n}" for n in range(plan["expansion_policy"]["max_operations"])
        )
    )
    operations = [r[0] for r in records if r]
    keys = tuple(k for op in operations for k in (op["key"] + ":result", op["key"] + ":outcome"))
    saved = dict(zip(keys, docs.read_many(keys), strict=True))
    wires, indexes, history, pending = [], [], [], []
    for op in operations:
        result, outcome = (saved[op["key"] + suffix] for suffix in (":result", ":outcome"))
        permitted_pending = {"local_search"}
        if local_failure:
            permitted_pending.add("local_index_window")
        if op["kind"] == "llm" or (result is None and op["kind"] not in permitted_pending):
            raise ValueError("interrupted_definition_unconfirmed_operation")
        history.append(dict(operation=op, result=digest(result), outcome=digest(outcome)))
        if op["kind"] == "wire":
            if not outcome or len(outcome) != 1:
                raise ValueError("interrupted_definition_wire_unknown")
            receipt = WireReceipt.model_validate_json(json.dumps(outcome[0]))
            if (
                receipt.http_status != 200
                or not receipt.response_checksum
                or receipt.operation != op.get("wire_kind")
                or result != (dict(provider=receipt.provider_code, reservation_only=True),)
                or not datetime.fromisoformat(plan["started_at"])
                <= receipt.requested_at
                <= datetime.fromisoformat(plan["deadline"])
            ):
                raise ValueError("interrupted_definition_wire_binding")
            wires.append(op["key"])
        if op["kind"] == "local_index_window":
            if local_failure and result is None:
                pending.append(pending_material(docs, op["key"]))
                continue
            if len(result) != 1 or result[0].get("status") != "indexed":
                raise ValueError("interrupted_definition_index_unconfirmed")
            # Original Runtime owns these immutable addresses. Current source
            # qualification is separately checked by Knowledge at consumption.
            actual = result[0]
            indexes.append(dict(key=actual["key"], checksum=digest(actual)))
    if (
        not 1 <= len(indexes) <= 8
        or len({v["key"] for v in indexes}) != len(indexes)
        or not wires
        or len(wires) != failure.get("new_requests")
        or len(wires) >= plan["max_new_requests"]
    ):
        raise ValueError("interrupted_definition_history_accounting")
    repair = docs.read(REPAIR_KEY) if local_failure else None
    if local_failure:
        if not 1 <= len(pending) <= 4:
            raise ValueError("confirmed_local_index_pending_bound")
        if repair is not None:
            if (
                len(repair) != 1
                or repair[0].get("schema_version") != "confirmed-local-index-repair/1.0.0"
                or repair[0].get("history") != digest(history)
                or repair[0].get("failure") != digest(failure)
                or repair[0].get("pending") != pending
                or repair[0].get("new_requests") != 0
                or len(repair[0].get("indexes", [])) != len(pending)
            ):
                raise ValueError("confirmed_local_index_repair_binding")
            indexes.extend(repair[0]["indexes"])
        elif not allow_pending:
            raise ValueError("confirmed_local_index_repair_required")
        if len(indexes) > 8 or len({r["key"] for r in indexes}) != len(indexes):
            raise ValueError("confirmed_local_index_pool_bound")
    proof = dict(
        path=str(path),
        result=digest(failure),
        result_kind=LOCAL_KIND if local_failure else KIND,
        protocol=digest(plan),
        journals=[digest(events)],
        run_id=docs.run_id,
        previous_attempts=plan["previous_attempts"]
        + [
            dict(
                directory=str(path),
                requests=len(wires),
                checksum=digest(failure),
                kind=LOCAL_HISTORY_KIND if local_failure else HISTORY_KIND,
            )
        ],
        interruption=dict(
            indexes=indexes,
            history=digest(history),
            wire_keys=wires,
            old_request_limit=plan["max_new_requests"],
            old_deadline=plan["deadline"],
            facts=plan["facts"],
        ),
    )
    if local_failure:
        proof["interruption"].update(pending=pending, repair=digest(repair))
    return proof


def pending_material(docs, key):
    """Bind a local window to its original fetched bytes, hit and frozen policy."""
    import re

    match = re.fullmatch(r"(.+):fetch:([0-9]+):index:window:([0-9]+)", key)
    if match is None:
        raise ValueError("confirmed_local_index_key")
    prefix, number, window = match.groups()
    fetch_key = prefix + ":fetch:" + number
    search = docs.read(prefix + ":search:result")
    fetched = docs.read(fetch_key + ":result")
    planned = docs.read(fetch_key + ":index:plan:result")
    if (
        not search
        or len(search) != 1
        or not isinstance(search[0], list)
        or not fetched
        or len(fetched) != 1
        or not planned
        or len(planned) != 1
        or not 0 <= int(number) < len(search[0])
        or not 0 <= int(window) < len(planned[0].get("policies", [])) <= 4
    ):
        raise ValueError("confirmed_local_index_material_missing")
    return dict(
        key=key,
        search_key=prefix + ":search",
        fetch_key=fetch_key,
        hit=search[0][int(number)],
        fetch_checksum=digest(fetched[0]),
        plan_checksum=digest(planned[0]),
        window=int(window),
        source_checksum=planned[0]["source_checksum"],
        policy=planned[0]["policies"][int(window)],
    )


def build_pending_index(original, docs, descriptor):
    """Original archive/index owners; no connector or model is installed here."""
    from db.knowledge.models import KnowledgeReference
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources, restore_fetch
    from packages.grounding.models import validate
    from packages.knowledge_index.models import IndexConfig, parse_chunk_policy

    current = pending_material(docs, descriptor["key"])
    if current != descriptor:
        raise ValueError("confirmed_local_index_material_changed")
    fetched = docs.read(descriptor["fetch_key"] + ":result")[0]
    request = restore_fetch(fetched, original.knowledge_objects)
    if request.document.response.response_checksum != descriptor["source_checksum"]:
        raise ValueError("confirmed_local_index_source_changed")
    request = request.model_copy(update={"pipeline_version": "1.2.0"})
    archived = original.knowledge.archive_hit(
        validate(KnowledgeReference, descriptor["hit"]["reference"]),
        request,
        original.knowledge_objects,
    )
    sources = KnowledgeDefinitionSources(
        repository=original.knowledge,
        store=original.knowledge_objects,
        material_policy="xml-jats-citations-windowed/1.0.0",
    )
    options = sources.index_options()
    options["chunk_policy"] = parse_chunk_policy(descriptor["policy"])
    actual = original.knowledge.build_document_index(
        archived,
        original.knowledge_objects,
        IndexConfig(
            mode="fts_only",
            vector_unavailable_reason="bounded_definition_text_retrieval",
            vector_weight=0.0,
        ),
        applicability_refs=sources.index_scope(archived),
        **options,
    )
    if actual.get("status") != "indexed":
        raise ValueError("confirmed_local_index_not_indexed")
    return dict(key=actual["key"], checksum=digest(actual))


def policy_for(old, proof):
    from .definition_expansion import read_expansion_policy

    policy = dict(old["expansion_policy"])
    policy["index_keys"] = list(
        dict.fromkeys(
            (*policy["index_keys"], *(i["key"] for i in proof["interruption"]["indexes"]))
        )
    )
    return read_expansion_policy(policy).model_dump(mode="json")


def verify_history(parent, attempt):
    """Original-owner proof for downstream accounting, never an acquisition."""
    from .campaign_renewal import MAX_HISTORY_DEPTH

    ancestor, seen = parent, set()
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if ancestor is None or id(ancestor) in seen:
            break
        seen.add(id(ancestor))
        saved = ancestor.protocol.get("tip", {})
        if ancestor.protocol.get("schema_version") == "multimodal-campaign-renewal/1.63.0":
            # This format successor retains the intervening failed 1.62 window
            # as history, not as its completed-facts parent. Follow that exact
            # original-owner-verified link once; never scan unrelated histories.
            from .confirmed_definition_failure import tip as confirmed_tip

            if confirmed_tip(ancestor.parent, saved["path"]) != saved:
                raise ValueError("interrupted_definition_history_owner")
            previous = json.loads((Path(saved["path"]) / "protocol.json").read_bytes())
            nested = previous.get("tip", {})
            if (
                previous.get("schema_version") == LOCAL_VERSION
                and nested.get("result_kind") == LOCAL_KIND
                and nested.get("path") == attempt["directory"]
            ):
                current = tip(ancestor.parent, nested["path"], local_failure=True)
                if current == nested and current["previous_attempts"][-1] == attempt:
                    return
                break
        if (
            saved.get("result_kind") in {KIND, LOCAL_KIND}
            and saved.get("path") == attempt["directory"]
        ):
            current = tip(
                ancestor.parent, saved["path"], local_failure=saved["result_kind"] == LOCAL_KIND
            )
            if current == saved and current["previous_attempts"][-1] == attempt:
                return
            break
        ancestor = getattr(ancestor, "parent", None)
    raise ValueError("interrupted_definition_history_owner")


def check_plan(plan):
    old = json.loads((Path(plan["tip"]["path"]) / "protocol.json").read_bytes())
    proof = plan["tip"]
    start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
    local = plan.get("schema_version") == LOCAL_VERSION
    if (
        plan.get("schema_version") not in VERSIONS
        or plan.get("interrupted_definitions") is not True
        or bool(plan.get("confirmed_local_index_failure")) != local
        or proof.get("result_kind") != (LOCAL_KIND if local else KIND)
        or (local and proof["interruption"].get("repair") in {None, digest(None)})
        or plan.get("new_public_search_allowed") is not False
        or plan.get("logical_campaign_id") != Path(proof["path"]).name
        or plan.get("expansion_policy") != policy_for(old, proof)
        or plan["facts"] != old["facts"]
        or plan["prior_deadline"] != old["deadline"]
        or plan["previous_attempts"] != proof["previous_attempts"]
        or plan["provider"] != old["provider"]
        or plan["input"] != old["input"]
        or plan["task"] != old["task"]
        or plan["original_fact_result"] != old["original_fact_result"]
        or not 0 < (end - start).total_seconds() == plan["seconds"] <= 3600
        or (not local and start <= datetime.fromisoformat(old["deadline"]))
        or (local and start <= datetime.fromisoformat(old["started_at"]))
        or not 1
        <= plan["max_new_requests"]
        <= old["max_new_requests"] - proof["previous_attempts"][-1]["requests"]
        or plan.get("max_cumulative_requests") != 40
        or sum(p["requests"] for p in plan["previous_attempts"]) + plan["max_new_requests"] > 40
    ):
        raise ValueError("interrupted_definition_successor_binding")


def prepare(
    *,
    previous,
    output,
    authorization,
    request_limit=5,
    seconds=3600,
    check_only=False,
    local_index_failure=False,
):
    from .campaign_renewal import RenewedCampaign, completed_parent, validate_increment_window
    from .execution_lease import PROFILE
    from .implementation import freeze

    path, out = Path(previous).absolute(), Path(output).absolute()
    if (
        not authorization.strip()
        or len(authorization) > 200
        or out.exists()
        or any(p.is_symlink() for p in (out, *out.parents))
    ):
        raise ValueError("interrupted_definition_explicit_new_window")
    old = json.loads((path / "protocol.json").read_bytes())
    parent, result = completed_parent(old["facts"])
    # Acquire the original execution lease, so terminal proof and the unique
    # successor claim cannot race an old process returning after disconnection.
    interrupted = RenewedCampaign(parent, path, old)
    with interrupted.execution_session():
        proof = tip(
            parent, path, local_failure=local_index_failure, allow_pending=local_index_failure
        )
        original = parent.base.sources.reader.original
        if local_index_failure:
            if (
                not 1
                <= request_limit
                <= old["max_new_requests"] - proof["previous_attempts"][-1]["requests"]
                or sum(p["requests"] for p in proof["previous_attempts"]) + request_limit > 40
            ):
                raise ValueError("confirmed_local_index_window_budget")
            validate_increment_window(policy_for(old, proof), seconds=seconds, cap=request_limit)
            docs = parent.documents(proof["run_id"])
            if docs.read("authorized-campaign-renewal") is not None:
                raise ValueError("renewal_tip_already_claimed")
            if check_only:
                return dict(
                    ready_for_local_repair=True,
                    ready_to_dispatch=False,
                    previous_requests=sum(p["requests"] for p in proof["previous_attempts"]),
                    max_new_requests=request_limit,
                    pending_indexes=len(proof["interruption"]["pending"]),
                    api_requests=0,
                    database_writes=0,
                    proof=proof,
                )
            if docs.read(REPAIR_KEY) is None:
                repair_start = datetime.now(UTC)
                repair_deadline = repair_start + timedelta(seconds=900)
                indexed = []
                for descriptor in proof["interruption"]["pending"]:
                    if (path / "cancel").exists() or datetime.now(UTC) >= repair_deadline:
                        raise ValueError("confirmed_local_index_repair_stopped")
                    indexed.append(build_pending_index(original, docs, descriptor))
                if (path / "cancel").exists() or datetime.now(UTC) >= repair_deadline:
                    raise ValueError("confirmed_local_index_repair_stopped")
                repair = dict(
                    schema_version="confirmed-local-index-repair/1.0.0",
                    history=proof["interruption"]["history"],
                    failure=proof["result"],
                    pending=proof["interruption"]["pending"],
                    indexes=indexed,
                    new_requests=0,
                    started_at=repair_start.isoformat(),
                    deadline=repair_deadline.isoformat(),
                )
                with docs.session():
                    docs.write(REPAIR_KEY, (repair,))
            proof = tip(parent, path, local_failure=True)
        for receipt in proof["interruption"]["indexes"]:
            actual = original.knowledge.replay_document_index(
                receipt["key"], original.knowledge_objects
            )
            if digest(actual) != receipt["checksum"] or actual.get("status") != "indexed":
                raise ValueError("interrupted_definition_index_changed")
        policy = policy_for(old, proof)
        validate_increment_window(policy, seconds=seconds, cap=request_limit)
        now = datetime.now(UTC)
        plan = dict(
            schema_version=LOCAL_VERSION if local_index_failure else VERSION,
            mode="definitions",
            authorization=authorization,
            facts=old["facts"],
            tip=proof,
            original_fact_result=digest(result),
            previous_attempts=proof["previous_attempts"],
            input=old["input"],
            task=old["task"],
            provider=digest(parent.config),
            implementation=freeze(),
            max_new_requests=request_limit,
            max_cumulative_requests=40,
            seconds=seconds,
            started_at=now.isoformat(),
            deadline=(now + timedelta(seconds=seconds)).isoformat(),
            prior_deadline=old["deadline"],
            confirmed_observations_checksum=digest(parent.accepted_observations()),
            expansion_policy=policy,
            original_unknown_not_replayed=True,
            new_models=0,
            clinical_acceptance=False,
            definition_increment_only=True,
            interrupted_definitions=True,
            new_public_search_allowed=False,
            logical_campaign_id=path.name,
            execution_lease_profile=PROFILE,
        )
        if local_index_failure:
            plan["confirmed_local_index_failure"] = True
        check_plan(plan)
        docs = parent.documents(proof["run_id"])
        if check_only:
            if docs.read("authorized-campaign-renewal") is not None:
                raise ValueError("renewal_tip_already_claimed")
            return dict(ready=True, plan=plan, api_requests=0, database_writes=0)
        with docs.session():
            if docs.read("authorized-campaign-renewal") is not None:
                raise ValueError("renewal_tip_already_claimed")
            docs.write(
                "authorized-campaign-renewal", (dict(output=str(out), plan_checksum=digest(plan)),)
            )
        out.mkdir(mode=0o700)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        previous_requests=sum(p["requests"] for p in plan["previous_attempts"]),
        max_new_requests=request_limit,
        reused_indexes=len(proof["interruption"]["indexes"]),
        new_public_requests_allowed=False,
        api_requests=0,
    )

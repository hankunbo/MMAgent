"""One format-corrected successor of fully confirmed Source failures.

Original Runtime owns every response, terminal failure and unique claim. This
port cannot resume UNKNOWN, change medical content or reset cumulative budget.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

VERSION = "multimodal-campaign-renewal/1.61.0"
PASSAGE_VERSION = "multimodal-campaign-renewal/1.63.0"
VERSIONS = {VERSION, PASSAGE_VERSION}
KIND = "confirmed-definition-format-failure/1.0.0"
HISTORY_KIND = "confirmed_definition_format_failure"


def logical_campaign(old, path):
    if old.get("schema_version") == "multimodal-campaign-renewal/1.62.0":
        value = old.get("logical_campaign_id")
        if not isinstance(value, str) or not value:
            raise ValueError("confirmed_definition_failure_campaign_identity")
        return value
    return Path(path).name


def tip(parent, path):
    from .campaign_renewal import RUNTIME_DEFINITION_VERSION, RenewedCampaign
    from .stage_failure import check_definition_binding, read_saved_failure

    path = Path(path).absolute()
    old = json.loads((path / "protocol.json").read_bytes())
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or (path / "result.json").exists()
        or (path / "cancel").exists()
        or old.get("schema_version")
        not in {RUNTIME_DEFINITION_VERSION, "multimodal-campaign-renewal/1.62.0"}
        or old.get("mode") != "definitions"
        or old.get("definition_increment_only") is not True
        or old.get("facts") != str(parent.output.absolute())
        or old.get("input") != parent.base.manifest.checksum
        or old.get("task") != parent.base.task.checksum
        or old.get("provider") != digest(parent.config)
        or old.get("new_public_search_allowed") is not False
        or old.get("expansion_policy", {}).get("schema_version")
        not in {"campaign-definition-expansion/1.34.0", "campaign-definition-expansion/1.36.0"}
    ):
        raise ValueError("confirmed_definition_failure_origin")
    policy_for(old)  # Only an explicit compatible transition is authorized.
    campaign = RenewedCampaign(parent, path, old)
    failure = read_saved_failure(campaign)
    rid = identity("RUN", "task-research:" + campaign.request_key("definition-expansion"))
    if (
        not failure
        or failure.get("run_id") != rid
        or failure.get("error_code") != "schema"
        or failure.get("stage") != "definitions"
        or failure.get("failure_kind") != "confirmed_response_or_domain_refusal"
        or failure.get("scores") is not None
        or failure.get("definition_revision_saved")
        or failure.get("confirmed_proposal_saved")
        or failure.get("cancel_marker")
        or failure.get("previous_attempts") != old["previous_attempts"]
    ):
        raise ValueError("confirmed_definition_failure_terminal")
    docs = campaign.documents(rid)
    check_definition_binding(campaign, docs)
    if campaign.sessions.read(rid) is not None or any(
        docs.read(k) is not None
        for k in ("dynamic-campaign", "definition:revision", "definition:stage:definitions")
    ):
        raise ValueError("confirmed_definition_failure_phase_started")
    for target in (*campaign.targets, None):
        target_run = identity("RUN", "task-research:" + campaign.request_key(target))
        if campaign.contexts.read_research_run(target_run, missing_ok=True) is not None:
            raise ValueError("confirmed_definition_failure_phase_started")
    events = RuntimeAgentJournal(docs).read()
    if old["schema_version"] == "multimodal-campaign-renewal/1.62.0" and not any(
        e.get("schema_version") == "source-definition-contract-diagnostic/1.0.0"
        and any(f.get("type") == "verbatim_request_quote_required" for f in e.get("failures", ()))
        for e in events
    ):
        raise ValueError("confirmed_definition_failure_passage_diagnostic")
    reservations = [e for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    keys = [r["request_id"] for r in reservations]
    if (
        not 1 <= len(keys) <= 2
        or len(set(keys)) != len(keys)
        or [r["request_id"] for r in records] != keys
        or confirmations
        != [dict(phase="response_confirmation", request_id=k, confirmed=True) for k in keys]
        or len(keys) != failure["new_requests"]
        or events[0]
        != dict(
            schema_version="agent-budget/1.0.0",
            binding=dict(
                run_id=rid, deadline=old["deadline"], max_requests=campaign.config.max_requests
            ),
        )
    ):
        raise ValueError("confirmed_definition_failure_unknown_or_accounting")
    raw_proofs = []
    for retry, (reserved, record) in enumerate(zip(reservations, records, strict=True)):
        raw = docs.read("definition:response:" + record["request_id"])
        if (
            record.get("error") != "schema"
            or record.get("role") != "source"
            or record.get("simulated") is not False
            or record.get("api_requests") != 1
            or record.get("retry_index") != retry
            or record.get("invocation_id") != keys[0]
            or record.get("scope", {}).get("run_id") != rid
            or record["scope"].get("case_code") != parent.base.binding["case_code"]
            or record["scope"].get("input_checksum") != old["input"]
            or reserved.get("scope_checksum") != digest(record["scope"])
            or reserved.get("config_checksum") != record.get("config_checksum")
            or record.get("prompt_version") != old["expansion_policy"]["prompt_version"]
            or not raw
            or len(raw) != 1
            or raw[0].get("record") != record
            or not raw[0].get("raw_text")
            or digest(raw[0]["raw_text"].encode("utf-8").hex()) != record.get("output_checksum")
        ):
            raise ValueError("confirmed_definition_failure_response_binding")
        raw_proofs.append(digest(raw[0]))
    operations = docs.read_many(
        tuple(
            f"definition:operation:{i}"
            for i in range(old["expansion_policy"]["max_operations"] + 1)
        )
    )
    present = [v[0] for v in operations if v]
    if (
        not present
        or any(v is None for v in operations[: len(present)])
        or len(present) > old["expansion_policy"]["max_operations"]
        or sum(op["kind"] == "llm" for op in present) != 1
    ):
        raise ValueError("confirmed_definition_failure_operations")
    history = []
    for op in present:
        result = docs.read(op["key"] + ":result")
        if (
            op["kind"] not in {"local_search", "local_pool_reconciliation", "llm"}
            or (op["kind"] == "llm" and result is not None)
            or (op["kind"] != "llm" and result is None)
        ):
            raise ValueError("confirmed_definition_failure_operations")
        history.append(dict(operation=op, result=digest(result)))
    return dict(
        path=str(path),
        result=digest(failure),
        result_kind=KIND,
        protocol=digest(old),
        journals=[digest(events)],
        run_id=rid,
        previous_attempts=old["previous_attempts"]
        + [
            dict(
                directory=str(path), requests=len(keys), checksum=digest(failure), kind=HISTORY_KIND
            )
        ],
        confirmed_failure=dict(
            responses=raw_proofs,
            history=digest(history),
            request_ids=keys,
            old_request_limit=old["max_new_requests"],
            deadline=old["deadline"],
        ),
    )


def policy_for(old):
    from .definition_expansion import read_expansion_policy

    policy = dict(old["expansion_policy"])
    if (
        old.get("schema_version") == "multimodal-campaign-renewal/1.62.0"
        and policy["schema_version"] == "campaign-definition-expansion/1.36.0"
        and policy["prompt_version"] == "definition-draft-6.7.0"
    ):
        policy.update(
            schema_version="campaign-definition-expansion/1.37.0",
            prompt_version="definition-draft-6.8.0",
            reserve_review_requests=2,
        )
        return read_expansion_policy(policy).model_dump(mode="json")
    if (
        old.get("schema_version") != "multimodal-campaign-renewal/1.56.0"
        or policy["schema_version"] != "campaign-definition-expansion/1.34.0"
        or policy["prompt_version"] != "definition-draft-6.6.0"
    ):
        raise ValueError("confirmed_definition_failure_format_transition")
    policy.update(
        schema_version="campaign-definition-expansion/1.35.0",
        prompt_version="definition-draft-6.7.0",
    )
    return read_expansion_policy(policy).model_dump(mode="json")


def check_plan(plan):
    old = json.loads((Path(plan["tip"]["path"]) / "protocol.json").read_bytes())
    start, end = (datetime.fromisoformat(plan[k]) for k in ("started_at", "deadline"))
    proof = plan["tip"]
    if (
        plan.get("schema_version")
        != (
            PASSAGE_VERSION
            if old["schema_version"] == "multimodal-campaign-renewal/1.62.0"
            else VERSION
        )
        or plan.get("confirmed_definition_failure") is not True
        or proof.get("result_kind") != KIND
        or plan.get("new_public_search_allowed") is not False
        or plan.get("logical_campaign_id") != logical_campaign(old, proof["path"])
        or plan.get("expansion_policy") != policy_for(old)
        or any(
            plan.get(k) != old[k]
            for k in ("facts", "provider", "input", "task", "original_fact_result")
        )
        or plan.get("prior_deadline") != old["deadline"]
        or plan.get("previous_attempts") != proof["previous_attempts"]
        or not 0 < (end - start).total_seconds() == plan["seconds"] <= 3600
        or not 1
        <= plan["max_new_requests"]
        <= old["max_new_requests"] - proof["previous_attempts"][-1]["requests"]
        or plan.get("max_cumulative_requests") != 40
        or sum(p["requests"] for p in plan["previous_attempts"]) + plan["max_new_requests"] > 40
    ):
        raise ValueError("confirmed_definition_failure_successor_binding")


def verify_history(parent, attempt):
    from .campaign_renewal import MAX_HISTORY_DEPTH

    ancestor, seen = parent, set()
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if ancestor is None or id(ancestor) in seen:
            break
        seen.add(id(ancestor))
        saved = ancestor.protocol.get("tip", {})
        if saved.get("result_kind") == KIND and saved.get("path") == attempt["directory"]:
            current = tip(ancestor.parent, saved["path"])
            if current == saved and current["previous_attempts"][-1] == attempt:
                return
            break
        ancestor = getattr(ancestor, "parent", None)
    raise ValueError("confirmed_definition_failure_history_owner")


def prepare(*, previous, output, authorization, request_limit=4, seconds=3600, check_only=False):
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
        raise ValueError("confirmed_definition_failure_explicit_window")
    old = json.loads((path / "protocol.json").read_bytes())
    parent, result = completed_parent(old["facts"])
    failed = RenewedCampaign(parent, path, old)
    with failed.execution_session():
        proof = tip(parent, path)
        policy = policy_for(old)
        # Knowledge current qualification is rechecked at material consumption.
        original = parent.base.sources.reader.original
        for key in policy["index_keys"]:
            current = original.knowledge.replay_document_index(key, original.knowledge_objects)
            if current.get("status") != "indexed":
                raise ValueError("confirmed_definition_failure_index_changed")
        validate_increment_window(policy, seconds=seconds, cap=request_limit)
        now = datetime.now(UTC)
        plan = dict(
            schema_version=PASSAGE_VERSION
            if old["schema_version"] == "multimodal-campaign-renewal/1.62.0"
            else VERSION,
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
            confirmed_definition_failure=True,
            new_public_search_allowed=False,
            logical_campaign_id=logical_campaign(old, path),
            execution_lease_profile=PROFILE,
        )
        check_plan(plan)
        docs = parent.documents(proof["run_id"])
        claim = dict(output=str(out), plan_checksum=digest(plan))
        with docs.session():
            if docs.read("authorized-campaign-renewal") is not None:
                raise ValueError("renewal_tip_already_claimed")
            if check_only:
                return dict(ready=True, plan=plan, api_requests=0, database_writes=0)
            docs.write("authorized-campaign-renewal", (claim,))
        out.mkdir(mode=0o700)
        write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        deadline=plan["deadline"],
        previous_requests=sum(p["requests"] for p in plan["previous_attempts"]),
        max_new_requests=request_limit,
        api_requests=0,
        new_public_requests_allowed=False,
    )

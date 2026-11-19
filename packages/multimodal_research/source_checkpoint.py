"""A bounded successor of expired sources-only work, using original owners.

No failed/UNKNOWN model call is replayed. The old Runtime claim, checkpoint and
deadline remain immutable; only a separately claimed successor may dispatch.
"""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

VERSION = "multimodal-campaign-renewal/1.80.0"
ORIGIN = "multimodal-campaign-renewal/1.79.0"
KIND = "expired-definition-source-checkpoint/1.0.0"


def inspect_checkpoint(campaign):
    """Prove a confirmed sources checkpoint with no model/patient work started."""
    from .stage_failure import check_definition_binding

    path, old = campaign.output.absolute(), campaign.protocol
    if (
        old.get("schema_version") != ORIGIN
        or old.get("mode") != "definitions"
        or old.get("definition_increment_only") is not True
        or datetime.fromisoformat(old["deadline"]) >= datetime.now(UTC)
        or any(p.is_symlink() for p in (path, *path.parents))
        or any(
            (path / p).exists()
            for p in ("cancel", "result.json", "stage-failure.json", "stage-closure-error.json")
        )
    ):
        raise ValueError("source_checkpoint_origin_not_expired_sources_only")
    rid = identity("RUN", "task-research:" + campaign.request_key("definition-expansion"))
    docs = campaign.documents(rid)
    check_definition_binding(campaign, docs)
    context = campaign.contexts.read_research_run(rid, missing_ok=True)
    if (
        context is None
        or context.run_id != rid
        or context.request.input_checksum != campaign.base.manifest.checksum
        or context.request.task != campaign.base.task
        or context.request.scope.case_code != campaign.base.binding["case_code"]
        or context.request.policy_ref.checksum
        != digest((campaign.base.ledger, campaign.config, None))
    ):
        raise ValueError("source_checkpoint_original_context_binding")
    receipt = json.loads((path / "stages/sources.json").read_bytes())
    checkpoint = docs.read("definition:stage:sources")
    if (
        not checkpoint
        or len(checkpoint) != 1
        or receipt.get("run_id") != rid
        or receipt.get("stage") != "sources"
        or receipt.get("binding") != digest(old)
        or receipt.get("payload") != checkpoint[0]
        or docs.read("multimodal-stage-operation:sources") != (receipt,)
        or checkpoint[0].get("binding", {}).get("protocol_checksum") != digest(old)
        or checkpoint[0]["binding"].get("input_checksum") != campaign.base.manifest.checksum
        or checkpoint[0]["binding"].get("target") != campaign.base.task.target_code
        or checkpoint[0]["binding"].get("parent_requirements_checksum")
        != digest(campaign.base.ledger)
        or checkpoint[0]["binding"].get("policy_checksum") != digest(old["expansion_policy"])
        or campaign.sessions.read(rid) is not None
    ):
        raise ValueError("source_checkpoint_owner_binding")
    forbidden = (
        "definition:revision",
        "definition:confirmed-proposal",
        "definition:stage:definitions",
        "definition:gap:final-candidate",
        "multimodal-stage-failure",
        "dynamic-phase-result",
        "multimodal-case-result",
    )
    if any(v is not None for v in docs.read_many(forbidden)):
        raise ValueError("source_checkpoint_later_work_started")
    for target in (*campaign.targets, None):
        phase_run = identity("RUN", "task-research:" + campaign.request_key(target))
        if campaign.contexts.read_research_run(phase_run, missing_ok=True) is not None:
            raise ValueError("source_checkpoint_patient_phase_started")
    events = RuntimeAgentJournal(docs).read()
    expected = dict(
        schema_version="agent-budget/1.0.0",
        binding=dict(
            run_id=rid, deadline=old["deadline"], max_requests=campaign.config.max_requests
        ),
    )
    if events != (expected,):
        raise ValueError("source_checkpoint_request_started_or_unknown")
    bound = old["expansion_policy"]["max_operations"]
    slots = docs.read_many(tuple(f"definition:operation:{n}" for n in range(bound + 1)))
    operations = [v[0] for v in slots if v]
    if (
        not operations
        or len(operations) > bound
        or any(v is None for v in slots[: len(operations)])
    ):
        raise ValueError("source_checkpoint_operation_inventory")
    history = []
    for op in operations:
        result = docs.read(op["key"] + ":result")
        if (
            op["kind"] not in {"local_search", "local_pool_reconciliation"}
            or op.get("binding") != checkpoint[0]["binding"]
            or result is None
        ):
            raise ValueError("source_checkpoint_external_or_unconfirmed_operation")
        history.append(dict(operation=op, result_checksum=digest(result)))
    if not checkpoint[0].get("materials"):
        raise ValueError("source_checkpoint_no_materials")
    proof = dict(
        path=str(path),
        run_id=rid,
        protocol=digest(old),
        result=digest(receipt),
        result_kind=KIND,
        journals=[digest(events)],
        checkpoint=digest(checkpoint[0]),
        history=digest(history),
        previous_attempts=old["previous_attempts"]
        + [dict(directory=str(path), requests=0, checksum=digest(receipt), kind=KIND)],
    )
    return proof, checkpoint[0]


def plan_for(old, proof, *, previous, output, authorization, now, implementation):
    """Keep original facts, policy, model and cumulative cap; refresh one window."""
    if (
        old.get("schema_version") != ORIGIN
        or proof.get("result_kind") != KIND
        or proof["protocol"] != digest(old)
        or proof["path"] != str(Path(previous).absolute())
        or not authorization.strip()
        or len(authorization) > 200
        or now.tzinfo is None
        or datetime.fromisoformat(old["deadline"]) >= now
        or old.get("max_cumulative_requests") != 40
        or not 300 < old["seconds"] <= 3600
        or not 1 <= old["max_new_requests"] <= 12
        or any(
            type(r.get("requests")) is not int or r["requests"] < 0
            for r in proof["previous_attempts"]
        )
        or sum(r["requests"] for r in proof["previous_attempts"]) + old["max_new_requests"] > 40
    ):
        raise ValueError("source_checkpoint_successor_bounds")
    plan = copy.deepcopy(old)
    plan.update(
        schema_version=VERSION,
        authorization=authorization,
        implementation=implementation,
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=old["seconds"])).isoformat(),
        prior_deadline=old["deadline"],
        tip=proof,
        previous_attempts=proof["previous_attempts"],
        source_checkpoint=dict(
            previous=str(Path(previous).absolute()),
            output=str(Path(output).absolute()),
            origin_protocol=digest(old),
            checkpoint=proof["checkpoint"],
        ),
        logical_campaign_id=Path(previous).name,
    )
    return plan


def recheck(campaign):
    from .campaign_renewal import assemble_bound_parent

    plan = campaign.protocol
    previous = plan.get("source_checkpoint", {}).get("previous")
    if (
        plan.get("schema_version") != VERSION
        or not previous
        or previous == str(campaign.output.absolute())
    ):
        raise ValueError("source_checkpoint_successor_version")
    old = json.loads((Path(previous) / "protocol.json").read_bytes())
    original = assemble_bound_parent(campaign.parent, previous, old)
    proof, material = inspect_checkpoint(original)
    expected = plan_for(
        original.protocol,
        proof,
        previous=previous,
        output=campaign.output,
        authorization=plan["authorization"],
        now=datetime.fromisoformat(plan["started_at"]),
        implementation=plan["implementation"],
    )
    if expected != plan:
        raise ValueError("source_checkpoint_successor_changed")
    return original, material


def prepare(*, previous, output, authorization, check_only=False):
    from .campaign_renewal import assemble
    from .implementation import freeze
    from .unknown_acquisition import check_network

    old = assemble(previous)
    # Validate the original public route before claiming a new bounded window.
    # run() still rechecks it immediately before any credential or dispatch.
    check_network(old.config)
    proof, _ = inspect_checkpoint(old)
    out = Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("source_checkpoint_output_symlink")
    plan = plan_for(
        old.protocol,
        proof,
        previous=previous,
        output=out,
        authorization=authorization,
        now=datetime.now(UTC),
        implementation=freeze(),
    )
    docs = old.documents(proof["run_id"])
    if check_only:
        if docs.read("authorized-campaign-renewal") is not None:
            raise ValueError("renewal_tip_already_claimed")
        return dict(
            check_only=True,
            protocol_checksum=digest(plan),
            previous=proof,
            max_new_requests=plan["max_new_requests"],
            cumulative_limit=plan["max_cumulative_requests"],
            previous_requests=sum(r["requests"] for r in plan["previous_attempts"]),
            source_checkpoint=plan["source_checkpoint"],
            actual_output_created=False,
            api_requests=0,
            database_writes=0,
        )
    with docs.session():
        if inspect_checkpoint(old)[0] != proof:
            raise ValueError("source_checkpoint_changed_before_claim")
        saved = docs.read("authorized-campaign-renewal")
        if saved is not None:
            if len(saved) != 1 or saved[0].get("output") != str(out) or "plan" not in saved[0]:
                raise ValueError("renewal_tip_already_claimed")
            plan = saved[0]["plan"]
            expected = plan_for(
                old.protocol,
                proof,
                previous=previous,
                output=out,
                authorization=authorization,
                now=datetime.fromisoformat(plan["started_at"]),
                implementation=freeze(),
            )
            if expected != plan or saved != (claim(plan, out),):
                raise ValueError("source_checkpoint_saved_claim_changed")
        else:
            if out.exists():
                raise ValueError("source_checkpoint_new_directory_required")
            docs.write("authorized-campaign-renewal", (claim(plan, out),))
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "protocol.json", plan)
    return dict(
        output=str(out),
        protocol_checksum=digest(plan),
        max_new_requests=plan["max_new_requests"],
        previous_requests=sum(r["requests"] for r in plan["previous_attempts"]),
        api_requests=0,
    )


def claim(plan, output):
    return dict(output=str(Path(output).absolute()), plan_checksum=digest(plan), plan=plan)


def retained_tip(plan):
    if plan.get("schema_version") != VERSION:
        return plan["tip"]
    reference = plan["source_checkpoint"]
    old = json.loads((Path(reference["previous"]) / "protocol.json").read_bytes())
    if digest(old) != reference["origin_protocol"]:
        raise ValueError("source_checkpoint_retained_origin_changed")
    return old["tip"]


def verify_history(parent, attempt):
    from .campaign_renewal import MAX_HISTORY_DEPTH, assemble, assemble_bound_parent

    # The fact owner is often already reconstructed by the caller. Reuse that
    # owner, not a stored qualification: the original claim constructor and the
    # checkpoint/Runtime readers below still run on each call.
    path = Path(attempt["directory"]).absolute()
    old = json.loads((path / "protocol.json").read_bytes())
    ancestor, seen, original = parent, set(), None
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if ancestor is None:
            break
        if id(ancestor) in seen:
            raise ValueError("source_checkpoint_history_ancestry_cycle")
        seen.add(id(ancestor))
        output = getattr(ancestor, "output", None)
        if output is not None and str(Path(output).absolute()) == old.get("facts"):
            original = assemble_bound_parent(ancestor, path, old)
            break
        ancestor = getattr(ancestor, "parent", None)
    else:
        if ancestor is not None:
            raise ValueError("source_checkpoint_history_ancestry_depth")
    if original is None:
        original = assemble(path)
    proof, _ = inspect_checkpoint(original)
    if (
        attempt != proof["previous_attempts"][-1]
        or original.base.manifest.checksum != parent.base.manifest.checksum
        or original.base.task != parent.base.task
    ):
        raise ValueError("source_checkpoint_history_changed")


def materials(expansion, planned_queries):
    """Reuse first direction's exact material through current Knowledge checks."""
    if (
        expansion.campaign.protocol.get("schema_version") != VERSION
        or getattr(expansion, "gap_attempt", None) != 0
    ):
        return None
    original, checkpoint = recheck(expansion.campaign)
    expected = dict(checkpoint["binding"], protocol_checksum=digest(expansion.campaign.protocol))
    if expected != expansion.binding:
        raise ValueError("source_checkpoint_definition_binding")
    planned = [
        r["plan"] for r in checkpoint["backlog"] if r.get("reason") == "owner_gap_retrieval_plan"
    ]
    if planned != planned_queries:
        raise ValueError("source_checkpoint_direction_changed")
    expansion.sources.restore(checkpoint["materials"])
    return (
        {m["code"]: m for m in checkpoint["materials"]},
        checkpoint["backlog"]
        + [
            dict(
                reason="confirmed_source_checkpoint_reused",
                checkpoint=expansion.campaign.protocol["source_checkpoint"],
            )
        ],
    )

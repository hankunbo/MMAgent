"""Explicit ancestor UNKNOWN batch on current facts, through existing owners.

This proves one new text attempt. It does not replay, dispatch, accept, or score.
"""

from copy import copy
from pathlib import Path

from packages.evidence_details.acceptance_models import DomainState, read_records
from packages.evidence_details.pending_review import (
    read_failed_review_attempt,
    unknown_response_proof,
)
from packages.grounding.models import digest, validate

VERSION = "multimodal-campaign-renewal/1.65.0"


def selected_material(old, current, selected):
    """Exact clinical item/observation/source equality; unrelated additions allowed."""
    a, b = old.material, current.material
    if any(
        getattr(a.scope, k) != getattr(b.scope, k)
        for k in ("case_code", "input_checksum", "visible_checksum", "graph")
    ) or (a.goal_code, a.goal_summary, a.visibility) != (b.goal_code, b.goal_summary, b.visibility):
        raise ValueError("historical_pending_case_or_goal_changed")
    before = {i.id: i for i in old.policy.items}
    after = {i.id: i for i in current.policy.items}
    observations = [{o.reference.code: o for o in m.observations} for m in (a, b)]
    sources = [{s.reference.code: s for s in m.sources} for m in (a, b)]
    for key in selected:
        item = before.get(key)
        if item is None or item != after.get(key):
            raise ValueError("historical_pending_definition_changed")
        code = item.observation_code
        observation = observations[0].get(code)
        if (
            observation is None
            or observation.status != "obtained"
            or observation != observations[1].get(code)
            or any(
                sources[0].get(s) is None or sources[0][s] != sources[1].get(s)
                for s in item.source_codes
            )
        ):
            raise ValueError("historical_pending_observation_or_source_changed")
    return dict(
        old_state_checksum=digest(old),
        current_state_checksum=digest(current),
        selected_ids=sorted(selected),
        exact_item_observation_sources=True,
    )


def inventory(parent, tip, request, origin_path):
    from agents.expert_agent.committed_review import read_committed_review

    from .campaign import Phase
    from .campaign_renewal import completed_parent, stopped_tip

    path = Path(origin_path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("historical_pending_origin_symlink")
    matches = [p for p in tip["previous_attempts"] if Path(p["directory"]).absolute() == path]
    if len(matches) != 1 or matches[0].get("kind") or not request:
        raise ValueError("historical_pending_origin_not_in_current_lineage")
    parent.base.check()
    docs = parent.documents(tip["run_id"])
    phase = docs.read("dynamic-phase-result")
    if not phase or not phase[0]["accepted"] or phase[0]["target"] is not None:
        raise ValueError("historical_pending_current_acceptance_required")
    parent.base.contexts[docs.run_id] = parent.contexts.read_research_run(docs.run_id)
    projector = Phase(parent.base, observation_reader=parent.accepted_observations)
    current_selected, _ = projector.project(run_id=docs.run_id, config=parent.config)
    saved = validate(DomainState, docs.read("dynamic-phase-input")[0])
    if current_selected != saved:
        raise ValueError("historical_pending_current_context_changed")
    projector.verify(saved)
    committed = read_committed_review(
        docs,
        parent=saved,
        config=parent.config,
        confirmed_review_reader=getattr(parent.base, "confirmed_review_reader", None),
    )
    if not committed or committed["acceptance_key"] != phase[0]["acceptance_key"]:
        raise ValueError("historical_pending_current_commit_binding")
    records = read_records(docs.read("accepted:" + phase[0]["acceptance_key"])[0])
    retained = sorted(i.assessment_id for i in records.candidate.items)
    base = copy(parent.base)
    base.review_target_ids = base.confirmed_review_reader = base.review_context_scope = None
    full = Phase(base, observation_reader=parent.accepted_observations)
    current, _ = full.project(run_id=docs.run_id, config=parent.config)
    full.verify(current)

    ancestor, result = completed_parent(path)
    old_tip = stopped_tip(ancestor, path)
    if matches[0]["checksum"] != digest(result) or matches[0]["requests"] != result["new_requests"]:
        raise ValueError("historical_pending_origin_history_changed")
    last = result["phases"][-1]
    if last["target"] is not None or last["accepted"] or last["run_id"] != old_tip["run_id"]:
        raise ValueError("historical_pending_origin_not_failed_review")
    old_docs = ancestor.documents(last["run_id"])
    old = validate(DomainState, old_docs.read("dynamic-phase-input")[0])
    attempt = read_failed_review_attempt(
        documents=old_docs, state=old, config=ancestor.config, acknowledge_request=request
    )
    authorized = [b for b in attempt["quarantined_batches"] if b.get("new_reassessment_authorized")]
    if len(authorized) != 1 or authorized[0]["request_ids"] != [request]:
        raise ValueError("historical_pending_exact_unknown_batch_required")
    selected = sorted(authorized[0]["assessment_ids"])
    if not selected or set(selected) & set(retained):
        raise ValueError("historical_pending_already_accepted")
    material = selected_material(old, current, selected)
    unknown = unknown_response_proof(old_docs, request, attempt["journal_checksum"])
    # Prior successor proofs remain discoverable through the actual parent chain.
    ancestor = parent
    seen = set()
    for _ in range(33):
        if ancestor is None:
            break
        if id(ancestor) in seen:
            raise ValueError("historical_pending_parent_cycle")
        seen.add(id(ancestor))
        pending = ancestor.protocol.get("pending_review") or {}
        if pending.get("acknowledge_unknown_request") == request:
            raise ValueError("historical_pending_request_already_reassessed")
        ancestor = getattr(ancestor, "parent", None)
    else:
        raise ValueError("historical_pending_history_depth")
    return dict(
        schema_version="pending-review-inventory/1.4.0",
        run_id=old_docs.run_id,
        journal_checksum=attempt["journal_checksum"],
        accepted_ids=retained,
        selected_ids=selected,
        quarantined_batches=attempt["quarantined_batches"],
        acknowledge_unknown_request=request,
        historical_unknown_origin=dict(
            path=str(path),
            result_checksum=digest(result),
            original_attempt=attempt,
            original_unknown=unknown,
            material=material,
            current_acceptance=committed,
            old_config_checksum=old.config_checksum,
            current_config_checksum=digest(parent.config),
        ),
    )

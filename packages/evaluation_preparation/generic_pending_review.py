"""Current-owner selection for a bounded legacy Review-only successor.

Selection is not acceptance: old judgments stay in their original receipts;
only confirmed truncated or never-sent tasks enter the new request plan.
"""

import json
from pathlib import Path

from packages.evidence_details.pending_review import read_pending_review
from packages.grounding.models import digest, validate
from packages.llm_provider.config import read_provider_config

from .dynamic_stages import rebuild
from .generic_aliases import GenericPhase

PROFILE = "relay-pending-single-v1"
CURRENT_PROFILE = "bound-pending-single-v1"
UNSTARTED_PROFILE = "bound-unstarted-batches-v1"
PROFILES = (PROFILE, CURRENT_PROFILE, UNSTARTED_PROFILE)
PROVIDER = "configs/agents/fairvlmed_relay_review_single_v1.json"


def configuration(origin, profile=PROFILE):
    from packages.evidence_details.live_permission import validate_live_config

    expected = origin.model_dump(mode="json")
    role = next(r for r in expected["roles"] if r["role"] == "review")
    if (
        origin.schema_version
        not in (
            {"agent-provider-config/7.0.0", "agent-provider-config/7.1.0"}
            if profile in {CURRENT_PROFILE, UNSTARTED_PROFILE}
            else {"agent-provider-config/7.0.0"}
        )
        or role["prompt_version"] != "review-batch-1.9.0"
        or expected["review_batching"]["batch_size"] != 3
    ):
        raise ValueError("generic_pending_profile_origin")
    if profile == UNSTARTED_PROFILE:
        # A new unstarted-only selection is not a method/prompt/batch-size change.
        return validate_live_config(origin)
    expected["review_batching"]["batch_size"] = 1
    if profile == CURRENT_PROFILE:
        # Derive only batching from the already-bound provider. Never substitute
        # the legacy file's model, credentials environment, prompt or route.
        return validate_live_config(read_provider_config(expected))
    if profile != PROFILE:
        raise ValueError("generic_pending_profile_unknown")
    selected = validate_live_config(read_provider_config(json.loads(Path(PROVIDER).read_bytes())))
    if selected.model_dump(mode="json") != expected:
        raise ValueError("generic_pending_profile_delta")
    return selected


def select(inventory, acknowledge_truncated_request):
    """No UNKNOWN acknowledgment route; an exact confirmed failure is required."""
    selected = set(inventory["selected_ids"])
    found = False
    for batch in inventory["quarantined_batches"]:
        proof = batch.get("known_truncated_failure")
        if proof and proof["last_request_id"] == acknowledge_truncated_request:
            if found or proof["original_answer_not_accepted"] is not True:
                raise ValueError("generic_pending_truncation_ambiguous")
            selected.update(batch["assessment_ids"])
            found = True
    if acknowledge_truncated_request is not None and not found:
        raise ValueError("generic_pending_truncation_ack_mismatch")
    if not selected or selected.intersection(inventory["accepted_ids"]):
        raise ValueError("generic_pending_selection_empty_or_accepted")
    return dict(
        schema_version="generic-pending-review-proof/1.0.0",
        inventory=inventory,
        selected_ids=sorted(selected),
        acknowledge_truncated_request=acknowledge_truncated_request,
        selection_kind="new_review_not_old_request_replay",
    )


def original_phase(campaign, *, require_accepted=True):
    """Reproject through original inputs/sources, not merely a saved-state lambda."""
    old = json.loads((campaign.output / "result.json").read_bytes())
    final = old["phases"][-1]
    if final.get("target") is not None or (
        not final.get("accepted")
        if require_accepted
        else final.get("accepted") is not False or final.get("acceptance_key") is not None
    ):
        raise ValueError("generic_pending_original_not_accepted_review")
    docs = campaign.documents(final["run_id"])
    if docs.read("dynamic-case-result") != (old,):
        raise ValueError("generic_pending_original_result_binding")
    phase = GenericPhase(
        campaign.base,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    state, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    phase.verify(state)
    if docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),):
        raise ValueError("generic_pending_original_material_changed")
    return docs, state, phase, final


def current_selection(
    campaign, acknowledge_truncated_request, *, profile=PROFILE, acknowledge_unknown_request=None
):
    if profile not in PROFILES:
        raise ValueError("generic_pending_profile_unknown")
    if profile == UNSTARTED_PROFILE:
        if acknowledge_truncated_request is not None or acknowledge_unknown_request is not None:
            raise ValueError("generic_unstarted_no_reassessment_authorization")
        return unstarted_selection(campaign)
    if profile == CURRENT_PROFILE:
        if acknowledge_truncated_request is not None or not acknowledge_unknown_request:
            raise ValueError("generic_pending_exact_unknown_ack_required")
    elif acknowledge_unknown_request is not None:
        raise ValueError("generic_pending_unknown_requires_current_profile")
    docs, state, _, final = original_phase(campaign)
    inventory = read_pending_review(
        documents=docs,
        state=state,
        acceptance_key=final["acceptance_key"],
        allow_empty=True,
        allow_legacy_partial=True,
        inspect_truncated_failures=True,
        acknowledge_unknown_request=acknowledge_unknown_request,
    )
    if profile == CURRENT_PROFILE:
        return select_unknown(inventory, acknowledge_unknown_request)
    return select(inventory, acknowledge_truncated_request)


def unstarted_selection(campaign):
    """Original failed-attempt owner selects unsent work, never claimed/UNKNOWN tasks."""
    from packages.evidence_details.pending_review import read_failed_review_attempt

    docs, state, _, _ = original_phase(campaign, require_accepted=False)
    inventory = read_failed_review_attempt(documents=docs, state=state, config=campaign.config)
    selected = inventory["selected_ids"]
    quarantined = {
        item for batch in inventory["quarantined_batches"] for item in batch["assessment_ids"]
    }
    if (
        not selected
        or inventory["acknowledge_request"] is not None
        or set(selected) & quarantined
        or any(b["new_reassessment_authorized"] for b in inventory["quarantined_batches"])
    ):
        raise ValueError("generic_unstarted_selection_empty_or_claimed")
    return dict(
        schema_version="generic-unstarted-review-proof/1.0.0",
        inventory=dict(inventory, accepted_ids=[]),
        selected_ids=selected,
        selection_kind="unstarted_only_original_failed_review_unchanged",
    )


def minimum_requests(proof, config, profile):
    size = config.review_batching.batch_size if profile == UNSTARTED_PROFILE else 1
    return (len(proof["selected_ids"]) + size - 1) // size


def select_unknown(inventory, request_id):
    """Consume original-owner authorization; do not infer eligibility from errors."""
    batches = [
        b for b in inventory["quarantined_batches"] if b.get("new_reassessment_authorized") is True
    ]
    if (
        not request_id
        or inventory.get("acknowledge_unknown_request") != request_id
        or len(batches) != 1
        or batches[0]["request_ids"] != [request_id]
        or batches[0].get("old_response_state_unchanged") is not True
    ):
        raise ValueError("generic_pending_unknown_ack_mismatch")
    selected = set(batches[0]["assessment_ids"])
    if (
        not selected
        or not selected <= set(inventory["selected_ids"])
        or selected.intersection(inventory["accepted_ids"])
    ):
        raise ValueError("generic_pending_selection_empty_or_accepted")
    return dict(
        schema_version="generic-pending-review-proof/1.1.0",
        inventory=inventory,
        selected_ids=sorted(selected),
        acknowledge_unknown_request=request_id,
        selection_kind="explicit_new_reassessment_old_unknown_unchanged",
    )


def selected_state(state, selected_ids):
    selected = set(selected_ids)
    if not selected or not selected <= {i.id for i in state.policy.items}:
        raise ValueError("generic_pending_current_items_changed")
    raw = state.policy.model_dump(mode="json")
    raw["items"] = [i for i in raw["items"] if i["id"] in selected]
    policy = validate(type(state.policy), raw)
    # Preserve all observations, sources, requirements and dependency groups.
    # Only the dispatchable Review items are narrowed; this is not a full score.
    return rebuild(state, policy, state.material.observations)


def phase_type(reader):
    class PendingPhase(GenericPhase):
        def project_current(self):
            state = super().project_current()
            if self.target is not None:
                raise ValueError("generic_pending_no_acquisition")
            return selected_state(state, reader()["selected_ids"])

    return PendingPhase


def implementation():
    from packages.object_store import checksum

    files = [
        "packages/evidence_details/pending_review.py",
        "packages/evidence_details/accepted_aggregation.py",
        "packages/evidence_details/accepted_review_reuse.py",
        "packages/evidence_details/review_batches.py",
        "packages/evidence_details/acceptance.py",
        "packages/evidence_details/acceptance_models.py",
    ]
    return {p: checksum(Path(p).read_bytes()) for p in files}


def completion(proof, scores):
    reviewed = {r["id"] for r in (scores or {}).get("reviews", [])}
    selected = set(proof["selected_ids"])
    return dict(
        schema_version="generic-pending-review-completion/1.0.0",
        selected_ids=sorted(selected),
        previous_accepted_ids=proof["inventory"]["accepted_ids"],
        pending_ids=sorted(selected - reviewed),
        historical_judgments_included_in_this_score=False,
        whole_case_aggregation_required=True,
        selection_checksum=digest(proof),
    )


def export_increment(campaign, result):
    """Save an explicitly partial display using original accepted bytes, no scoring here."""
    from types import SimpleNamespace

    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.acceptance_models import DomainState
    from packages.grounding.models import canonical

    docs = campaign.documents(result["phases"][-1]["run_id"])
    saved = docs.read("dynamic-case-report")
    if saved:
        if saved[0]["result"] != result or saved[1]["report_checksum"] != digest(saved[0]):
            raise ValueError("generic_pending_saved_report_binding")
        return saved[0]
    state = validate(DomainState, docs.read("dynamic-phase-input")[0])
    final = result["phases"][-1]
    detail = None
    if final["accepted"]:
        service = ResearchAcceptance(
            owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
        )
        detail = read_research_report(service, key=final["acceptance_key"], parent=state)
    scores = detail["scores"] if detail else None
    proof = campaign.pending_review_proof
    scope = completion(proof, scores)
    old = set(scope["previous_accepted_ids"])
    rows = [
        dict(
            id=r.id,
            examination=r.definition.examination,
            modality=list(r.input_modalities),
            status=(
                "原已评阅，本增量不重复计分"
                if r.id in old
                else "本轮补评"
                if r.id in proof["selected_ids"]
                else "保留原需求及缺项"
            ),
            observation=None,
            sources=[s.model_dump(mode="json") for s in r.definition.source_refs],
        )
        for r in state.policy.requirements
    ]
    report = json.loads(
        canonical(
            dict(
                schema_version="generic-pending-review-report/1.0.0",
                result=result,
                requirements=rows,
                required_count=len(rows),
                acquired_count=len(set(proof["selected_ids"]) | old),
                scores=scores,
                original_review=detail,
                pending_review_increment=scope,
                completion=dict(
                    report_saved=True,
                    domain_accepted=final["accepted"],
                    all_obtained_reviewed=False,
                    pending_review_ids=scope["pending_ids"],
                    whole_case_aggregation_required=True,
                ),
                reasoning_scope="selected_review_increment_not_whole_case",
                calibrated_probability=False,
                clinical_diagnosis=False,
            )
        )
    )
    rendered = (
        render_research_html(detail) if detail else "<!doctype html><html><body></body></html>"
    )
    body = rendered.find(">", rendered.find("<body")) + 1
    notice = (
        "<section><h1>待评阅项目补评增量</h1><p>本页不是完整病例评分。"
        "旧判读未重复请求，未计入本页分数；须经原接纳入口组合后另存完整结果。</p></section>"
    )
    rendered = rendered[:body] + notice + rendered[body:]
    with docs.session():
        docs.write(
            "dynamic-case-report", (report, dict(html=rendered, report_checksum=digest(report)))
        )
    return report


def restore_campaign(path):
    """Read original generic owners for local composition, never install a dispatcher."""
    from .case_execution import owners
    from .generic_campaign import GenericCampaign
    from .generic_input import directory, load, provider_config
    from .generic_review import GenericReview

    path = Path(path).absolute()
    if path != directory(path.name):
        raise ValueError("generic_aggregation_path_scope")
    selection = json.loads((path / "input.json").read_bytes())
    if selection.get("schema_version") in {
        "generic-capability-successor/1.0.0",
        "generic-capability-successor/1.1.0",
        "generic-capability-successor/1.2.0",
        "generic-capability-successor/1.3.0",
    }:
        from .generic_definition_execution import restore

        return restore(path)
    if selection.get("schema_version") in {
        "generic-acquisition-successor/1.0.0",
        "generic-acquisition-successor/2.0.0",
        "generic-acquisition-successor/3.0.0",
    }:
        from .generic_acquisition_successor import restore

        return restore(path)
    if selection.get("schema_version") in {
        "generic-review-selection/1.7.0",
        "generic-review-selection/1.8.0",
    }:
        from .generic_review_continuation import restore

        return restore(path)
    if selection.get("schema_version") == "generic-review-selection/1.6.0":
        from .generic_reassessment import restore

        return restore(path)
    if selection.get("schema_version") == "generic-review-selection/1.5.0":
        from .generic_review_followup import restore

        return restore(path)
    if selection.get("schema_version") == "generic-definition-review-successor/1.0.0":
        from .generic_definition_execution import restore

        return restore(path)
    pending = selection.get("schema_version") in {
        "generic-review-selection/1.2.0",
        "generic-review-selection/1.3.0",
        "generic-review-selection/1.4.0",
    }
    parent_key = selection["parent_key"] if pending else path.name
    spec = load(parent_key)
    parent = directory(parent_key)
    binding = json.loads((parent / "binding.json").read_bytes())
    runtime, store, _ = owners()
    registered = (parent, binding, runtime, store)
    if pending:
        current = selection["schema_version"] == "generic-review-selection/1.3.0"
        unstarted = selection["schema_version"] == "generic-review-selection/1.4.0"
        profile = UNSTARTED_PROFILE if unstarted else CURRENT_PROFILE if current else PROFILE
        if (
            selection.get("key") != path.name
            or selection.get("review_key") != path.name
            or selection.get("review_profile") != profile
            or (current and "acknowledge_truncated_request" in selection)
            or (not current and "acknowledge_unknown_request" in selection)
            or (unstarted and "acknowledge_truncated_request" in selection)
            or (
                unstarted
                and selection.get("provider_derivation") != "bound-provider-unchanged/1.0.0"
            )
            or (
                current
                and selection.get("provider_derivation") != "bound-provider-batch-size-only/1.0.0"
            )
            or (
                (current or unstarted)
                and selection.get("origin_provider_checksum") != digest(provider_config(spec))
            )
            or selection.get("input_checksum") != digest(spec)
            or json.loads((parent / "review-successor.json").read_bytes())
            != {k: v for k, v in selection.items() if k != "key"}
            or json.loads((path / "binding.json").read_bytes()) != binding
        ):
            raise ValueError("generic_aggregation_successor_binding")
        campaign = GenericReview(
            spec,
            registered,
            path.name,
            review_profile=profile,
            **(
                {}
                if unstarted
                else dict(acknowledge_unknown_request=selection["acknowledge_unknown_request"])
                if current
                else dict(acknowledge_truncated_request=selection["acknowledge_truncated_request"])
            ),
        )
        if (
            selection["eligibility_checksum"] != digest(campaign.pending_review_proof)
            or selection["selected_ids"] != campaign.pending_review_proof["selected_ids"]
            or selection["provider_checksum"] != digest(campaign.config)
        ):
            raise ValueError("generic_aggregation_eligibility_changed")
    else:
        campaign = GenericCampaign(spec, registered)
    return confirmed_campaign(campaign)


def confirmed_campaign(campaign):
    """Common exact saved acceptance gate for legacy and explicit followup callers."""
    old = json.loads((campaign.output / "result.json").read_bytes())
    final = old["phases"][-1]
    docs = campaign.documents(final["run_id"])
    if docs.read("dynamic-case-result") != (old,) or not final.get("accepted"):
        raise ValueError("generic_aggregation_original_not_accepted")
    saved = docs.read("dynamic-case-report")
    if not saved or saved[0]["result"] != old:
        raise ValueError("generic_aggregation_original_report_binding")
    for request_path in campaign.paths():
        request_file = request_path / "request.json"
        if request_file.exists():
            run_id = json.loads(request_file.read_bytes())["lock"]["run_id"]
            control = campaign.sessions.read(run_id)
            if (
                control is None
                or control.phase != "stopped"
                or control.pending
                or control.cancelled
            ):
                raise ValueError("generic_aggregation_original_not_closed")
    protocol = json.loads((campaign.output / "protocol.json").read_bytes())
    if docs.read("dynamic-campaign") != (protocol,) or protocol["provider_checksum"] != digest(
        campaign.config
    ):
        raise ValueError("generic_aggregation_protocol_binding")
    campaign.protocol = protocol
    return campaign, old

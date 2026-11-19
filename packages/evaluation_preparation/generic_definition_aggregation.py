"""Original-owner whole-case fold after one explicitly adopted definition revision.

No new scores, facts, requests, or arbitrary item selection are authored here.
The changed item's new accepted judgment replaces its old interpretation; all
other original judgments must be compatible and retained by the existing owner.
"""

from pathlib import Path

from packages.evidence_details.accepted_aggregation import AcceptedOriginReader
from packages.grounding.models import digest

from .generic_definition_execution import committed_state, restore
from .generic_definition_projection import phase_type
from .generic_definition_selection import (
    DefinitionRevisionSelection,
    RetainedDefinitionOriginReader,
)
from .generic_definition_successor import DefinitionReview
from .generic_input import directory


def restore_plan(plan):
    campaign, result = restore(Path(plan.facts).absolute())
    if type(campaign) is not DefinitionReview or not campaign.definition_keys:
        raise ValueError("generic_definition_aggregate_owner_required")
    origins = [Path(item.campaign).absolute() for item in plan.origins]
    expected = [directory(campaign.spec["key"]), directory(campaign.review_key)]
    if origins != expected or any(item.local_finalization for item in plan.origins):
        raise ValueError("generic_definition_aggregate_exact_origins")
    phase = phase_type(
        campaign.checked.facts,
        campaign.definition_stage.expansion.owner,
        campaign.definition_keys,
        selected_only=False,
    )
    return campaign, result, phase


def select_readers(plan, readers, campaign):
    from .generic_aliases import GenericPhase

    if type(campaign) is not DefinitionReview or len(readers) != 2:
        raise ValueError("generic_definition_aggregate_owner_required")
    origins = [Path(item.campaign).absolute() for item in plan.origins]
    if origins != [directory(campaign.spec["key"]), directory(campaign.review_key)]:
        raise ValueError("generic_definition_aggregate_exact_origins")
    if (
        type(readers[0]) is not AcceptedOriginReader
        or type(readers[1]) is not DefinitionCommitOriginReader
        or readers[1].campaign.review_key != campaign.review_key
    ):
        raise ValueError("generic_definition_aggregate_origin_readers")

    def seed_reader(run_id):
        phase = GenericPhase(
            campaign.base,
            observation_reader=campaign.accepted_observations,
            model_reader=campaign.read_model,
        )
        state, _ = phase.project(run_id=run_id, config=campaign.config)
        phase.verify(state)
        return state

    selection = DefinitionRevisionSelection(
        owner=campaign.definition_stage.expansion.owner,
        keys=campaign.definition_keys,
        seed_reader=seed_reader,
    )
    original = readers[0]
    if (
        original.client_transition
        or original.reasoned_batch_resize
        or original.confirmed_review_reader is not None
        or original.reused_origin_new_items_only
        or original.alias_manifests is not None
        or original.retained_identity_aliases
    ):
        raise ValueError("generic_definition_aggregate_no_combined_transition")
    retained = RetainedDefinitionOriginReader(
        definition_selection=selection,
        documents=original.documents,
        requests=original.requests,
        state_reader=original.state_reader,
        acceptance_key=original.acceptance_key,
        current_config=original.current_config,
    )
    retained.accounting = original.accounting
    return (retained, readers[1])


class DefinitionCommitOriginReader(AcceptedOriginReader):
    """Exact committed Review is still readable after technical cancellation."""

    def __init__(self, *, campaign, **kwargs):
        if type(campaign) is not DefinitionReview:
            raise ValueError("generic_definition_committed_owner_required")
        super().__init__(**kwargs)
        self.campaign = campaign

    def _committed_cancellation(self, old, control):
        docs, state, _, committed = committed_state(self.campaign)
        proof = committed[0] if committed else {}
        if (
            docs.run_id != self.documents.run_id
            or state != old
            or proof.get("acceptance_key") != self.acceptance_key
            or proof.get("parent_checksum") != digest(old)
            or proof.get("accepted_checksum")
            != digest(docs.read("accepted:" + self.acceptance_key))
            or proof.get("journal_checksum") != digest(self.requests.journal.read())
            or docs.sessions.read(docs.run_id) != control
        ):
            raise ValueError("generic_definition_committed_cancel_binding")
        return True


def prepare_plan(key, output):
    """Freeze the exact two origins for the existing aggregate-accepted command."""
    from apps.vectorengine_single_case import write_once
    from packages.multimodal_research.accepted_aggregation import (
        GenericDefinitionAggregationPlan,
        OriginPath,
        checked_path,
    )

    campaign, _ = restore(directory(key))
    original = directory(campaign.spec["key"])
    successor = directory(campaign.review_key)
    destination = checked_path(output)
    if any(destination == p or p in destination.parents for p in (original, successor)):
        raise ValueError("generic_definition_plan_overlaps_origin")
    plan = GenericDefinitionAggregationPlan(
        facts=str(successor),
        origins=(OriginPath(campaign=str(original)), OriginPath(campaign=str(successor))),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_once(destination, plan)
    return dict(plan=str(destination), plan_checksum=digest(plan), new_api_requests=0)


def audit_wholecase(path, report_path, tree_path):
    """Existing current-aggregation owner and saved report/tree must agree exactly."""
    import json
    from types import SimpleNamespace

    from agents.expert_agent.reporting import read_research_report
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.accepted_aggregation import ACCEPTANCE_KEY
    from packages.multimodal_research.accepted_aggregation import restore_aggregation
    from packages.multimodal_research.report_storage import unpack
    from packages.object_store import checksum

    from .cohort_recovery import check_tree
    from .generic_definition_successor import read

    campaign, result = restore_aggregation(Path(path).absolute())
    if type(campaign) is not DefinitionReview or not result.get("definition_revision_policy"):
        raise ValueError("generic_definition_audit_wholecase_scope")
    baseline = campaign.aggregation_baseline
    docs, state = baseline.documents, baseline.state
    stored, display = unpack(
        docs.read("accepted-aggregation-report"), key="accepted-aggregation-report"
    )
    report_path, tree_path = Path(report_path), Path(tree_path)
    report = json.loads(report_path.read_bytes())
    service = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
    )
    detail = read_research_report(service, key=ACCEPTANCE_KEY, parent=state)
    if (
        report != stored
        or report["result"] != result
        or report["scores"] != detail["scores"]
        or report["original_review"] != detail
        or report.get("review_scope", {}).get("kind") != "whole_case_definition_reaggregation"
        or report["definition_revision"]["definition_projection"]
        != docs.read("generic-definition-wholecase-projection")[0]
        or report_path.with_suffix(".html").read_text() != display["html"]
        or not check_tree(report, tree_path.read_text())
    ):
        raise ValueError("generic_definition_audit_saved_bundle_mismatch")
    old = read(campaign.acquisition.output / "review/review-report.json")
    new = read(campaign.output / "review/review-report.json")
    before = {r["id"]: r for r in old["scores"]["reviews"]}
    changed = {r["id"]: r for r in new["scores"]["reviews"]}
    after = {r["id"]: r for r in report["scores"]["reviews"]}
    expected = set(
        r["assessment_id"]
        for r in report["definition_revision"]["definition_projection"]["interpretation_revisions"]
    )
    if (
        set(changed) != expected
        or set(after) != set(before)
        or any(after[k] != v for k, v in before.items() if k not in changed)
        or any(after[k] != v for k, v in changed.items())
        or result["new_requests"] != 0
    ):
        raise ValueError("generic_definition_audit_exact_judgment_selection")
    values = [r["judgment"]["value"] for r in after.values()]
    return dict(
        schema_version="generic-definition-wholecase-audit/1.0.0",
        original_owner_verified=True,
        original_score_json_html_equal=True,
        tree_packet_verified=True,
        report=str(report_path),
        report_checksum=checksum(report_path.read_bytes()),
        tree=str(tree_path),
        tree_checksum=checksum(tree_path.read_bytes()),
        requirements=len(report["requirements"]),
        reviews=len(after),
        revised_ids=sorted(changed),
        retained_reviews=len(before) - len(changed),
        nonzero=sum(v is not None and v != 0 for v in values),
        neutral=sum(v == 0 for v in values),
        null=sum(v is None for v in values),
        cumulative_requests=result["cumulative_requests"],
        new_requests=0,
        new_joint_medical_review=False,
        clinical_acceptance=False,
    )

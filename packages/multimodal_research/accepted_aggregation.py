"""CLI composition of original accepted Review owners; no model/Tool dispatcher."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from agents.expert_agent.reporting import read_research_report
from agents.planner.service import Planner
from apps.vectorengine_live_assembly import make_binding
from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    GRANT_KEY,
    AcceptedOriginReader,
    AcceptedReviewSet,
    AggregationPermission,
    AggregationRequests,
    HistoricalRequests,
    NullReviewRevision,
)
from packages.grounding.models import Value, digest, validate
from packages.planner_control.models import Command
from packages.research_runtime.agent_records import RuntimeAgentJournal
from packages.research_runtime.live_snapshots import LiveSnapshots
from packages.research_runtime.planner_steps import ResearchPlannerSteps

from .campaign import Phase
from .campaign_renewal import MAX_HISTORY_DEPTH, completed_parent


class OriginPath(Value):
    campaign: str = Field(min_length=1)
    local_finalization: str | None = None


class AggregationPlan(Value):
    schema_version: Literal["accepted-review-aggregation-plan/1.0.0"] = (
        "accepted-review-aggregation-plan/1.0.0"
    )
    facts: str = Field(min_length=1)
    origins: tuple[OriginPath, ...] = Field(min_length=1, max_length=8)


class FullAggregationPlan(AggregationPlan):
    schema_version: Literal["accepted-review-aggregation-plan/1.1.0"] = (
        "accepted-review-aggregation-plan/1.1.0"
    )
    scope: Literal["all_current_obtained"] = "all_current_obtained"


class GenericAggregationPlan(AggregationPlan):
    schema_version: Literal["accepted-review-aggregation-plan/1.2.0"] = (
        "accepted-review-aggregation-plan/1.2.0"
    )
    scope: Literal["all_current_obtained"] = "all_current_obtained"
    owner_kind: Literal["fairvlmed_generic_pending_review"] = "fairvlmed_generic_pending_review"


class LegacyCapabilityAggregationPlan(GenericAggregationPlan):
    """Original unchanged judgments plus new capability-selected observations."""

    schema_version: Literal["accepted-review-aggregation-plan/1.14.0"] = (
        "accepted-review-aggregation-plan/1.14.0"
    )
    owner_kind: Literal["fairvlmed_generic_capability_increment"] = (
        "fairvlmed_generic_capability_increment"
    )
    selection_policy: Literal["current_capability_increment/1.0.0"] = (
        "current_capability_increment/1.0.0"
    )


class GenericCapabilityAggregationPlan(LegacyCapabilityAggregationPlan):
    """Include original acquisition/failure/unsent windows, not only Review origins."""

    schema_version: Literal["accepted-review-aggregation-plan/1.15.0"] = (
        "accepted-review-aggregation-plan/1.15.0"
    )
    accounting: Literal["capability_owner_ancestry/1.0.0"] = "capability_owner_ancestry/1.0.0"
    history_software: dict[str, str] | None = None


class GenericDefinitionAggregationPlan(GenericAggregationPlan):
    """Replace exactly owner-revised definitions; preserve every other judgment."""

    schema_version: Literal["accepted-review-aggregation-plan/1.12.0"] = (
        "accepted-review-aggregation-plan/1.12.0"
    )
    owner_kind: Literal["fairvlmed_generic_definition_revision"] = (
        "fairvlmed_generic_definition_revision"
    )
    selection_policy: Literal["owner_adopted_definition_revision/1.0.0"] = (
        "owner_adopted_definition_revision/1.0.0"
    )


class GenericClientAggregationPlan(GenericAggregationPlan):
    """Exact same-contract generic judgments across one HTTP-only client upgrade."""

    schema_version: Literal["accepted-review-aggregation-plan/1.10.0"] = (
        "accepted-review-aggregation-plan/1.10.0"
    )
    reuse_kind: Literal["same_medical_contract_http_client_only"] = (
        "same_medical_contract_http_client_only"
    )


class AccountedAggregationPlan(FullAggregationPlan):
    """Full current facts, including failed/UNKNOWN windows in the original ledger."""

    schema_version: Literal["accepted-review-aggregation-plan/1.3.0"] = (
        "accepted-review-aggregation-plan/1.3.0"
    )
    accounting: Literal["fact_lineage_and_review_origins"] = "fact_lineage_and_review_origins"


class ReasonedBatchAggregationPlan(AccountedAggregationPlan):
    """Same model and medical contract, explicit 3-to-1 task partition change."""

    schema_version: Literal["accepted-review-aggregation-plan/1.4.0"] = (
        "accepted-review-aggregation-plan/1.4.0"
    )
    reuse_kind: Literal["same_method_partition_only"] = "same_method_partition_only"


class RetainedOriginAggregationPlan(AccountedAggregationPlan):
    """Current ancestor recheck, then select only the successor's new batches."""

    schema_version: Literal["accepted-review-aggregation-plan/1.5.0"] = (
        "accepted-review-aggregation-plan/1.5.0"
    )
    reuse_kind: Literal["owner_verified_new_batches_only"] = "owner_verified_new_batches_only"


class AliasAggregationPlan(AccountedAggregationPlan):
    """Exact original source identity across newly numbered source aliases."""

    schema_version: Literal["accepted-review-aggregation-plan/1.6.0"] = (
        "accepted-review-aggregation-plan/1.6.0"
    )
    reuse_kind: Literal["original_owner_exact_source_aliases"] = (
        "original_owner_exact_source_aliases"
    )


class SourceBudgetOrigin(Value):
    campaign: str = Field(min_length=1)
    result_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class SourceBudgetAggregationPlan(AccountedAggregationPlan):
    """Completed Source branches contribute costs, never judgments or new facts."""

    schema_version: Literal["accepted-review-aggregation-plan/1.7.0"] = (
        "accepted-review-aggregation-plan/1.7.0"
    )
    source_budget_origins: tuple[SourceBudgetOrigin, ...] = Field(min_length=1, max_length=8)


class RetainedSourceBudgetAggregationPlan(
    SourceBudgetAggregationPlan, RetainedOriginAggregationPlan
):
    """Preserve selected Review ancestry while counting terminal Source branches."""

    schema_version: Literal["accepted-review-aggregation-plan/1.8.0"] = (
        "accepted-review-aggregation-plan/1.8.0"
    )


class RetainedAliasAggregationPlan(AliasAggregationPlan, RetainedOriginAggregationPlan):
    """Exact historical aliases plus identity-only retained successor contexts."""

    schema_version: Literal["accepted-review-aggregation-plan/1.9.0"] = (
        "accepted-review-aggregation-plan/1.9.0"
    )
    reuse_kind: Literal["owner_verified_aliases_and_new_batches"] = (
        "owner_verified_aliases_and_new_batches"
    )


class NullRevisionAggregationPlan(RetainedOriginAggregationPlan):
    """Original accepted nulls superseded only by exact new-context Reviews."""

    schema_version: Literal["accepted-review-aggregation-plan/1.11.0"] = (
        "accepted-review-aggregation-plan/1.11.0"
    )
    null_revisions: tuple[NullReviewRevision, ...] = Field(min_length=1, max_length=6)
    selection_policy: Literal["new_context_accepted_null_revision/1.0.0"] = (
        "new_context_accepted_null_revision/1.0.0"
    )


class MethodContextAggregationPlan(NullRevisionAggregationPlan):
    """Full current observation context plus exact newly accepted method metadata."""

    schema_version: Literal["accepted-review-aggregation-plan/1.13.0"] = (
        "accepted-review-aggregation-plan/1.13.0"
    )
    material_context_policy: Literal["full-owner-and-selected-method-context/1.0.0"] = (
        "full-owner-and-selected-method-context/1.0.0"
    )


def review_set(plan, readers, config, *, campaign=None):
    """The same explicit selection on initial composition and cold restoration."""
    if isinstance(plan, LegacyCapabilityAggregationPlan):
        from packages.evaluation_preparation.generic_capability_aggregation import select_readers

        readers = select_readers(plan, readers, campaign)
    if isinstance(plan, GenericDefinitionAggregationPlan):
        from packages.evaluation_preparation.generic_definition_aggregation import select_readers

        readers = select_readers(plan, readers, campaign)
    revisions = plan.null_revisions if isinstance(plan, NullRevisionAggregationPlan) else ()
    return AcceptedReviewSet(readers, config, null_revisions=revisions)


def read_plan(raw):
    cls = {
        "accepted-review-aggregation-plan/1.0.0": AggregationPlan,
        "accepted-review-aggregation-plan/1.1.0": FullAggregationPlan,
        "accepted-review-aggregation-plan/1.2.0": GenericAggregationPlan,
        "accepted-review-aggregation-plan/1.10.0": GenericClientAggregationPlan,
        "accepted-review-aggregation-plan/1.11.0": NullRevisionAggregationPlan,
        "accepted-review-aggregation-plan/1.12.0": GenericDefinitionAggregationPlan,
        "accepted-review-aggregation-plan/1.13.0": MethodContextAggregationPlan,
        "accepted-review-aggregation-plan/1.14.0": LegacyCapabilityAggregationPlan,
        "accepted-review-aggregation-plan/1.15.0": GenericCapabilityAggregationPlan,
        "accepted-review-aggregation-plan/1.3.0": AccountedAggregationPlan,
        "accepted-review-aggregation-plan/1.4.0": ReasonedBatchAggregationPlan,
        "accepted-review-aggregation-plan/1.5.0": RetainedOriginAggregationPlan,
        "accepted-review-aggregation-plan/1.6.0": AliasAggregationPlan,
        "accepted-review-aggregation-plan/1.7.0": SourceBudgetAggregationPlan,
        "accepted-review-aggregation-plan/1.8.0": RetainedSourceBudgetAggregationPlan,
        "accepted-review-aggregation-plan/1.9.0": RetainedAliasAggregationPlan,
    }.get(raw.get("schema_version"))
    if cls is None:
        raise ValueError("accepted_aggregation_plan_version")
    return validate(cls, raw)


def plan_campaign(plan):
    if isinstance(plan, LegacyCapabilityAggregationPlan):
        from packages.evaluation_preparation.generic_capability_aggregation import restore_plan

        return restore_plan(plan)
    if isinstance(plan, GenericDefinitionAggregationPlan):
        from packages.evaluation_preparation.generic_definition_aggregation import restore_plan

        return restore_plan(plan)
    if isinstance(plan, GenericAggregationPlan):
        from packages.evaluation_preparation.generic_aliases import GenericPhase
        from packages.evaluation_preparation.generic_pending_review import restore_campaign
        from packages.evaluation_preparation.generic_review import GenericReview

        campaign, old = restore_campaign(checked_path(plan.facts))
        if not isinstance(campaign, GenericReview) or campaign.pending_review_proof is None:
            raise ValueError("generic_aggregation_requires_pending_successor")
        if isinstance(plan, GenericClientAggregationPlan) and (
            campaign.config.schema_version
            not in {"agent-provider-config/7.2.0", "agent-provider-config/9.2.0"}
            or getattr(campaign.config, "http_client", None) is None
        ):
            raise ValueError("generic_aggregation_client_version_required")
        return campaign, old, GenericPhase
    campaign, old = completed_parent(checked_path(plan.facts))
    full_scope(campaign, plan)
    return campaign, old, Phase


def full_scope(campaign, plan):
    if isinstance(plan, ReasonedBatchAggregationPlan):
        from .truncated_review import VERSION, configuration

        if campaign.protocol.get("schema_version") != VERSION or campaign.config != configuration(
            campaign.parent.config
        ):
            raise ValueError("accepted_aggregation_reasoned_partition_origin")
    if isinstance(plan, FullAggregationPlan):
        from copy import copy

        campaign.base = copy(campaign.base)
        campaign.base.review_target_ids = None
        campaign.base.confirmed_review_reader = None
        if isinstance(plan, MethodContextAggregationPlan):
            if not getattr(campaign.base, "review_metadata_profile", None):
                raise ValueError("accepted_aggregation_method_profile_required")
            campaign.base.aggregation_method_context_policy = plan.material_context_policy
        if isinstance(plan, AccountedAggregationPlan):
            # Full projection rebuilds current context for every obtained item.
            # The incremental retained-context mode requires the reader above;
            # origins remain independently checked by AcceptedReviewSet.
            campaign.base.review_context_scope = None


def restore_aggregation(path, *, depth=0, ancestry=()):
    """Read original owners behind a sealed local result; never dispatch/write."""
    from types import SimpleNamespace

    path = checked_path(path)
    plan = read_plan(json.loads((path / "plan.json").read_bytes()))
    if depth > MAX_HISTORY_DEPTH or checked_path(plan.facts) == path:
        raise ValueError("accepted_aggregation_history_depth")
    if not ancestry or ancestry[-1] != path:
        if path in ancestry:
            raise ValueError("renewal_history_cycle")
        ancestry = (*ancestry, path)
    generic = isinstance(plan, GenericAggregationPlan)
    if generic:
        campaign, old, phase_class = plan_campaign(plan)
    else:
        campaign, old = completed_parent(plan.facts, depth=depth + 1, ancestry=ancestry)
        full_scope(campaign, plan)
        phase_class = Phase
    result = json.loads((path / "result.json").read_bytes())
    manifest = json.loads((path / "manifest.json").read_bytes())
    if (
        result.get("schema_version")
        != (
            "generic-accepted-aggregation-result/1.0.0"
            if generic
            else "multimodal-accepted-aggregation-result/1.0.0"
        )
        or result.get("original_result_checksum") != digest(old)
        or manifest.get("plan") != plan.model_dump(mode="json")
        or manifest.get("original_result_checksum") != digest(old)
    ):
        raise ValueError("accepted_aggregation_restore_identity")
    if (
        isinstance(plan, GenericDefinitionAggregationPlan)
        and result.get("definition_revision_policy") != plan.selection_policy
    ):
        raise ValueError("generic_definition_aggregate_policy_changed")
    if isinstance(plan, LegacyCapabilityAggregationPlan) and (
        result.get("capability_revision_policy") != plan.selection_policy
    ):
        raise ValueError("generic_capability_aggregate_policy_changed")
    if isinstance(plan, NullRevisionAggregationPlan) and (
        result.get("review_revision_policy") != plan.selection_policy
        or result.get("review_revisions")
        != [r.model_dump(mode="json") for r in plan.null_revisions]
    ):
        raise ValueError("accepted_aggregation_revision_restore_binding")
    docs = campaign.documents(manifest["run_id"])
    if (
        docs.read("accepted-aggregation-manifest") != (manifest,)
        or docs.read("multimodal-case-result") != (result,)
        or (generic and docs.read("dynamic-case-result") != (result,))
        or result["phases"][-1]["run_id"] != docs.run_id
        or docs.read("dynamic-phase-result") != (result["phases"][-1],)
    ):
        raise ValueError("accepted_aggregation_restore_persistence")
    if not generic:
        campaign.base.contexts[docs.run_id] = campaign.contexts.read_research_run(docs.run_id)
    phase = (
        phase_class(
            campaign.base,
            observation_reader=campaign.accepted_observations,
            model_reader=campaign.read_model,
        )
        if generic
        else phase_class(campaign.base, observation_reader=campaign.accepted_observations)
    )
    state, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    phase.verify(state)
    if docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),):
        raise ValueError("accepted_aggregation_restore_current_material")
    if manifest.get("recovery") is not None:
        from packages.evidence_details.accepted_aggregation import check_recovery_content
        from packages.evidence_details.review_batches import read_submission

        previous_docs, proof, previous_state, previous_sub = recovery_origin(
            campaign, manifest["recovery"]["previous_output"], plan, old
        )
        if (
            manifest.get("schema_version") != "accepted-review-aggregation-manifest/1.1.0"
            or manifest["recovery"] != proof
            or result.get("local_aggregation_recovery") != proof
        ):
            raise ValueError("accepted_aggregation_restore_recovery_binding")
        claim = previous_docs.read("accepted-aggregation-recovery-successor")
        expected = recovery_claim(path, plan, proof)
        if claim != (expected,):
            raise ValueError("accepted_aggregation_restore_recovery_claim")
        current_sub = read_submission(docs.read("submission:" + ACCEPTANCE_KEY)[0]["submission"])
        check_recovery_content(previous_state, previous_sub, state, current_sub)
    if isinstance(plan, AliasAggregationPlan) and docs.read("generic-review-source-aliases") != (
        phase.source_alias_manifest,
    ):
        raise ValueError("accepted_aggregation_current_alias_owner_binding")
    if isinstance(plan, GenericDefinitionAggregationPlan) and docs.read(
        "generic-definition-wholecase-projection"
    ) != (phase.definition_revision_manifest,):
        raise ValueError("generic_definition_aggregate_projection_changed")
    if isinstance(plan, GenericDefinitionAggregationPlan) and result.get(
        "definition_revision_projection_checksum"
    ) != digest(phase.definition_revision_manifest):
        raise ValueError("generic_definition_aggregate_projection_changed")
    if isinstance(plan, LegacyCapabilityAggregationPlan) and (
        docs.read("generic-capability-wholecase-projection") != (phase.capability_projection,)
        or result.get("capability_revision_projection_checksum")
        != digest(phase.capability_projection)
    ):
        raise ValueError("generic_capability_aggregate_projection_changed")
    source = review_set(
        plan,
        tuple(
            origin_reader(
                p,
                generic=generic,
                current_config=campaign.config
                if generic or isinstance(plan, ReasonedBatchAggregationPlan)
                else None,
                reasoned_batch_resize=isinstance(plan, ReasonedBatchAggregationPlan),
                client_transition=isinstance(plan, GenericClientAggregationPlan),
                network_transition=isinstance(plan, LegacyCapabilityAggregationPlan)
                and bool(campaign.selection.get("route_transition")),
                reused_origin_new_items_only=isinstance(plan, RetainedOriginAggregationPlan),
                retained_identity_aliases=isinstance(plan, RetainedAliasAggregationPlan),
                current_alias_reader=(lambda: phase.source_alias_manifest)
                if isinstance(plan, AliasAggregationPlan)
                else None,
                history_software=plan.history_software
                if isinstance(plan, GenericCapabilityAggregationPlan) and p.campaign == plan.facts
                else None,
            )
            for p in plan.origins
        ),
        campaign.config,
        campaign=campaign,
    )
    verify_request_history(plan, campaign, old, source.readers, manifest, result)
    sub, _ = source.current(state)
    grant = docs.read(GRANT_KEY)
    control = campaign.sessions.read(docs.run_id)
    if (
        digest(state) != manifest["state_checksum"]
        or digest(sub) != manifest["submission_checksum"]
        or not grant
        or digest(control) != grant[0]["control_checksum"]
        or control.phase != "stopped"
        or control.cancelled
        or control.pending
        or RuntimeAgentJournal(docs).read()
    ):
        raise ValueError("accepted_aggregation_restore_authority")
    service = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
    )
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=state)
    if report["receipt"]["submission_checksum"] != digest(sub) or report["receipt"][
        "permission_checksum"
    ] != digest(grant[0]):
        raise ValueError("accepted_aggregation_restore_acceptance")
    campaign.aggregation_baseline = SimpleNamespace(
        documents=docs,
        acceptance_key=ACCEPTANCE_KEY,
        state=state,
        manifest=manifest,
        path=path,
        submission=sub,
        grant=grant[0],
    )
    return campaign, result


def aggregation_tip(path):
    campaign, result = restore_aggregation(path)
    baseline = campaign.aggregation_baseline
    previous = result["previous_attempts"] + [
        dict(directory=str(checked_path(path)), requests=0, checksum=digest(result))
    ]
    if sum(a["requests"] for a in previous) != result["cumulative_requests"]:
        raise ValueError("accepted_aggregation_history_accounting")
    return dict(
        path=str(checked_path(path)),
        result=digest(result),
        protocol=digest(baseline.manifest),
        protocol_kind="accepted_aggregation_manifest",
        journals=[digest(())],
        previous_attempts=previous,
        run_id=baseline.documents.run_id,
        deadline=baseline.grant["deadline"],
    )


def checked_path(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("accepted_aggregation_symlink")
    return path


def source_budget_history(plan, campaign, fact_history):
    """Recheck the original Source/Runtime owners; no caller-supplied cost totals.

    A terminal Source branch must share exact original ancestry and current
    input/proposition with the fact branch. Its definitions are not installed in
    the scoring campaign. UNKNOWN ancestry remains typed and is checked again by
    request_history; this proof never authorizes a new dispatch.
    """
    from .campaign_renewal import stopped_tip

    known = {str(checked_path(a["directory"])): a for a in fact_history}
    seen, groups = set(), []
    for origin in plan.source_budget_origins:
        path = checked_path(origin.campaign)
        if path in seen or path == checked_path(plan.facts):
            raise ValueError("accepted_aggregation_source_budget_duplicate")
        seen.add(path)
        branch, result = completed_parent(path)
        protocol = branch.protocol
        if (
            digest(result) != origin.result_checksum
            or protocol.get("mode") != "definitions"
            or protocol.get("definition_increment_only") is not True
            or result.get("complete") is not True
            or branch.base.manifest.checksum != campaign.base.manifest.checksum
            or branch.base.task != campaign.base.task
            or branch.base.binding["case_code"] != campaign.base.binding["case_code"]
        ):
            raise ValueError("accepted_aggregation_source_budget_scope")
        proof = stopped_tip(branch, path, completed_source=True)
        attempts = proof["previous_attempts"]
        # Historical terminal Source results have no cumulative_requests field.
        # stopped_tip binds each original reservation and previous_attempts to
        # the saved protocol; derive the total from those original fields.
        total = sum(a["requests"] for a in result["previous_attempts"]) + result["new_requests"]
        if (
            proof["result"] != origin.result_checksum
            or not proof.get("definition")
            or not any(known.get(str(checked_path(a["directory"]))) == a for a in attempts)
            or sum(a["requests"] for a in attempts) != total
            or result.get("cumulative_requests", total) != total
        ):
            raise ValueError("accepted_aggregation_source_budget_lineage")
        groups.append((branch, attempts))
    return groups


def budget_source_owner(parent, attempt):
    """Resolve typed costs through a persisted aggregation's exact branch plan.

    A later acquisition walks its ordinary parent chain. Only a manifest read
    back from the original Runtime can supply extra accounting ancestry. The
    branch's original terminal proof is rebuilt before returning its owner.
    """
    seen = set()
    for _ in range(MAX_HISTORY_DEPTH + 1):
        if parent is None or id(parent) in seen:
            break
        seen.add(id(parent))
        baseline = getattr(parent, "aggregation_baseline", None)
        if baseline is not None:
            manifest = baseline.manifest
            plan = read_plan(manifest["plan"])
            if isinstance(plan, SourceBudgetAggregationPlan):
                docs = baseline.documents
                result = docs.read("multimodal-case-result")
                if (
                    docs.read("accepted-aggregation-manifest") != (manifest,)
                    or not result
                    or digest(result[0]["previous_attempts"])
                    != manifest.get("request_history_checksum")
                ):
                    raise ValueError("accepted_aggregation_source_budget_persistence")
                for owner, entries in source_budget_history(
                    plan, parent, result[0]["previous_attempts"]
                ):
                    if attempt in entries:
                        return owner
        parent = getattr(parent, "parent", None)
    return None


def capability_request_history(campaign, result):
    """Original recovery owner proves ALL windows, including failed/unsent ones.

    The current prior() rechecks the bound predecessor inventories recursively;
    their exact hashes retain request states and failure records. They contribute
    costs only, never judgments or permission to replay UNKNOWN requests.
    """
    from packages.evaluation_preparation.generic_capability_successor import CapabilitySuccessor
    from packages.evaluation_preparation.generic_input import directory

    if type(campaign) is not CapabilitySuccessor:
        raise ValueError("capability_accounting_owner_required")
    prior, used = campaign.prior(), campaign.usage()
    root = prior["completed_origin"]
    if (
        prior != campaign.protocol["previous"]
        or root != campaign.checked.history
        or root["root_key"] != campaign.spec["key"]
        or root["input_checksum"] != digest(campaign.spec)
        or result["prior_graph_requests"] != prior["reservations"]
        or result["new_requests"] != used
        or type(used) is not int
        or used < 0
    ):
        raise ValueError("capability_accounting_current_binding")
    windows, seen = [], {campaign.review_key}
    child = campaign
    while child.selection.get("continuation") is not None:
        binding = child.selection["continuation"]
        previous = getattr(child.checked, "predecessor", None)
        inventory = binding["inventory"]
        if (
            type(previous) is not CapabilitySuccessor
            or previous.review_key in seen
            or len(seen) >= 4
            or binding["predecessor_key"] != previous.review_key
            or binding["inventory_checksum"] != digest(inventory)
            or inventory["key"] != previous.review_key
            or inventory["parent_key"] != campaign.spec["key"]
            or inventory["selection_checksum"] != digest(previous.selection)
            or inventory["protocol_checksum"] != digest(previous.protocol)
            or previous.checked.history != root
            or inventory["cumulative_requests"] != child.protocol["previous"]["reservations"]
            or type(inventory["new_requests"]) is not int
            or inventory["new_requests"] < 0
            or inventory["new_requests"] + previous.protocol["previous"]["reservations"]
            != inventory["cumulative_requests"]
            or sum(row["reservations"] for row in inventory["phases"]) != inventory["new_requests"]
        ):
            raise ValueError("capability_accounting_ancestry_binding")
        seen.add(previous.review_key)
        windows.append(
            dict(
                directory=str(directory(previous.review_key)),
                requests=inventory["new_requests"],
                checksum=digest(inventory),
            )
        )
        child = previous
    if child.protocol["previous"]["reservations"] != root["reservations"]:
        raise ValueError("capability_accounting_missing_root")
    history = [
        dict(
            directory=str(directory(root["root_key"])),
            requests=root["reservations"],
            checksum=root["result_checksum"],
        ),
        *reversed(windows),
        dict(directory=str(directory(campaign.review_key)), requests=used, checksum=digest(result)),
    ]
    if (
        len({row["directory"] for row in history}) != len(history)
        or sum(row["requests"] for row in history) != prior["reservations"] + used
        or prior["reservations"] + used > 40
    ):
        raise ValueError("capability_accounting_budget_or_duplicate")
    return history


def request_history(plan, campaign, old, readers):
    """Merge only owner-restored records; never infer confirmation from request counts.

    plan_campaign/completed_parent verify the fact result, journals and lineage;
    origin_reader independently verifies accepted Review origins. Failed facts
    can contribute budget history, but can never supply an accepted judgment.
    Versions 1.0-1.2 retain their historical origin-only accounting interpretation.
    """
    from .campaign_renewal import attempted_acquisition_targets
    from .confirmed_definition_failure import HISTORY_KIND
    from .demand_history import SOURCE_KINDS

    accounted = isinstance(plan, AccountedAggregationPlan)
    typed_kinds = SOURCE_KINDS | (
        {HISTORY_KIND} if isinstance(plan, AliasAggregationPlan) else set()
    )
    groups = [reader.accounting for reader in readers]
    if isinstance(plan, GenericCapabilityAggregationPlan):
        groups.insert(0, capability_request_history(campaign, old))
    if accounted:
        groups.insert(
            0,
            [
                *old["previous_attempts"],
                dict(
                    directory=str(checked_path(plan.facts)),
                    requests=old["new_requests"],
                    checksum=digest(old),
                ),
            ],
        )
    source_branches = []
    if isinstance(plan, SourceBudgetAggregationPlan):
        source_branches = source_budget_history(plan, campaign, groups[0])
        groups.extend(entries for _, entries in source_branches)
    attempts = {}
    for group in groups:
        for attempt in group:
            typed = (
                set(attempt) == {"directory", "requests", "checksum", "kind"}
                and attempt.get("kind") in typed_kinds
            )
            if accounted and (
                (set(attempt) != {"directory", "requests", "checksum"} and not typed)
                or type(attempt["requests"]) is not int
                or attempt["requests"] < 0
            ):
                raise ValueError("accepted_aggregation_request_history_invalid")
            directory = str(checked_path(attempt["directory"]))
            if directory in attempts and attempts[directory] != attempt:
                raise ValueError("accepted_aggregation_request_history_conflict")
            if accounted and typed and directory not in attempts:
                # Preserve the typed failure, its UNKNOWN status and cost. The
                # original Source/Runtime owner verifies the binding; the label
                # is never permission to consume a judgment or replay a request.
                owner = next(
                    (branch for branch, entries in source_branches if attempt in entries),
                    campaign,
                )
                attempted_acquisition_targets(owner, [attempt])
            attempts[directory] = attempt
    if (
        accounted
        or isinstance(plan, (GenericDefinitionAggregationPlan, LegacyCapabilityAggregationPlan))
    ) and sum(a["requests"] for a in attempts.values()) > 40:
        raise ValueError("accepted_aggregation_request_history_budget")
    return list(attempts.values())


def verify_request_history(plan, campaign, old, readers, manifest, result):
    if not isinstance(
        plan,
        (
            AccountedAggregationPlan,
            GenericDefinitionAggregationPlan,
            LegacyCapabilityAggregationPlan,
        ),
    ):
        return
    attempts = request_history(plan, campaign, old, readers)
    if (
        manifest.get("request_history_checksum") != digest(attempts)
        or result.get("previous_attempts") != attempts
        or result.get("cumulative_requests") != sum(a["requests"] for a in attempts)
        or result.get("new_requests") != 0
    ):
        raise ValueError("accepted_aggregation_request_history_binding")


def origin_reader(
    entry,
    *,
    generic=False,
    current_config=None,
    reasoned_batch_resize=False,
    client_transition=False,
    network_transition=False,
    reused_origin_new_items_only=False,
    current_alias_reader=None,
    retained_identity_aliases=False,
    history_software=None,
):
    if generic:
        from packages.evaluation_preparation.generic_pending_review import restore_campaign

        if entry.local_finalization:
            raise ValueError("generic_aggregation_no_foreign_local_finalization")
        if history_software is not None:
            from packages.evaluation_preparation.generic_definition_execution import restore

            campaign, old = restore(checked_path(entry.campaign), history_software=history_software)
        else:
            campaign, old = restore_campaign(checked_path(entry.campaign))
    else:
        campaign, old = completed_parent(checked_path(entry.campaign))
    tip = old["phases"][-1]
    docs = campaign.documents(tip["run_id"])
    if entry.local_finalization is not None:
        local = json.loads((checked_path(entry.local_finalization) / "result.json").read_bytes())
        tip = local_origin_phase(docs, old, local, campaign=campaign)
    if not tip.get("accepted") or tip.get("target") is not None:
        raise ValueError("accepted_aggregation_review_not_accepted")
    if not generic:
        campaign.base.contexts[docs.run_id] = campaign.contexts.read_research_run(docs.run_id)
    parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
    phase = (
        campaign.phase_class(
            campaign.base,
            observation_reader=campaign.accepted_observations,
            model_reader=campaign.read_model,
        )
        if generic
        else Phase(campaign.base, observation_reader=campaign.accepted_observations)
    )

    def current():
        now, _ = phase.project(run_id=docs.run_id, config=campaign.config)
        if now != parent:
            raise ValueError("accepted_aggregation_original_material_changed")
        phase.verify(parent)
        return parent

    def alias_manifests():
        original = docs.read("generic-review-source-aliases")
        if original is None or original != (getattr(phase, "source_alias_manifest", None),):
            raise ValueError("accepted_aggregation_original_alias_owner_binding")
        return original[0], current_alias_reader()

    reader_type, closure_options = AcceptedOriginReader, {}
    if entry.local_finalization is not None and local.get("schema_version") == (
        "multimodal-recorded-finalization-result/1.2.0"
    ):
        from packages.evidence_details.accepted_aggregation import CommittedClosureOriginReader

        reader_type = CommittedClosureOriginReader
        closure_options = dict(campaign=campaign, closure_checksum=digest(local))
    if generic:
        from packages.evaluation_preparation.generic_capability_successor import CapabilitySuccessor
        from packages.evaluation_preparation.generic_definition_successor import DefinitionReview

        if type(campaign) is CapabilitySuccessor:
            from packages.evaluation_preparation.generic_capability_aggregation import (
                CapabilityCommitOriginReader,
            )

            reader_type = CapabilityCommitOriginReader
            closure_options = dict(campaign=campaign)
        if type(campaign) is DefinitionReview:
            from packages.evaluation_preparation.generic_definition_aggregation import (
                DefinitionCommitOriginReader,
            )

            reader_type = DefinitionCommitOriginReader
            closure_options = dict(campaign=campaign)
    reader = reader_type(
        **closure_options,
        documents=docs,
        requests=HistoricalRequests(RuntimeAgentJournal(docs)),
        state_reader=current,
        acceptance_key=tip["acceptance_key"],
        current_config=current_config,
        reasoned_batch_resize=reasoned_batch_resize,
        client_transition=client_transition,
        network_transition=network_transition,
        confirmed_review_reader=getattr(campaign.base, "confirmed_review_reader", None)
        if reused_origin_new_items_only
        else None,
        reused_origin_new_items_only=reused_origin_new_items_only,
        alias_manifests=alias_manifests if current_alias_reader is not None else None,
        retained_identity_aliases=retained_identity_aliases,
    )
    if generic:
        original = getattr(campaign, "acquisition", None)
        previous = []
        if original is not None:
            prior = json.loads((original.output / "result.json").read_bytes())
            if original.documents(prior["phases"][-1]["run_id"]).read("dynamic-case-result") != (
                prior,
            ):
                raise ValueError("generic_aggregation_accounting_binding")
            previous = [
                dict(
                    directory=str(original.output.parent),
                    checksum=digest(prior),
                    requests=prior["new_requests"],
                )
            ]
        from packages.evaluation_preparation.generic_acquisition_successor import (
            FailedAcquisitionSuccessor,
        )
        from packages.evaluation_preparation.generic_review_followup import FollowupReview

        if isinstance(campaign, FailedAcquisitionSuccessor):
            previous = campaign.original_accounting()
        if isinstance(campaign, FollowupReview):
            from packages.evaluation_preparation.generic_reassessment import aggregation_history

            previous = aggregation_history(campaign)
        reader.accounting = [
            *previous,
            dict(
                directory=str(checked_path(entry.campaign)),
                checksum=digest(old),
                requests=old["new_requests"],
            ),
        ]
        return reader
    reader.accounting = [
        *campaign.protocol.get("previous_attempts", []),
        dict(
            directory=str(checked_path(entry.campaign)),
            checksum=digest(old),
            requests=old["new_requests"],
        ),
    ]
    return reader


def local_origin_phase(documents, original, local, *, campaign=None):
    """Explicit local-result versions; eligibility remains with the Review owner."""
    from packages.evidence_details.partial_finalization import (
        CITATION_ACCEPTANCE_KEY,
        DEADLINE_ACCEPTANCE_KEY,
    )
    from packages.evidence_details.partial_finalization import RESULT_KEY as PARTIAL_RESULT_KEY
    from packages.evidence_details.recorded_explanation import POLICY, RESULT_KEY

    version = local.get("schema_version")
    if version == "multimodal-recorded-finalization-result/1.2.0":
        from packages.evidence_details.recorded_finalization import RESULT_KEY as RECORDED_KEY

        from .campaign_renewal import recorded_acquisition_origin

        if campaign is None:
            raise ValueError("accepted_aggregation_committed_origin_owner_required")
        proof = recorded_acquisition_origin(campaign, documents.run_id)
        if proof["closure_checksum"] != digest(local) or proof[
            "original_result_checksum"
        ] != digest(original):
            raise ValueError("accepted_aggregation_committed_origin_binding")
        key = RECORDED_KEY
    elif version == "multimodal-recorded-finalization-result/1.3.0":
        if local.get("finalization_policy") != POLICY:
            raise ValueError("accepted_aggregation_local_finalization_binding")
        key = RESULT_KEY
    elif version == "multimodal-partial-finalization-result/1.1.0":
        scope = local.get("review_closure_scope", {})
        if (
            scope.get("local_confirmed_only") is not True
            or scope.get("historical_answers_reaccepted") is not False
            or local.get("phases", [{}])[-1].get("acceptance_key") != DEADLINE_ACCEPTANCE_KEY
        ):
            raise ValueError("accepted_aggregation_local_finalization_binding")
        key = PARTIAL_RESULT_KEY
    elif version == "multimodal-partial-finalization-result/1.2.0":
        scope = local.get("review_closure_scope", {})
        if (
            scope.get("local_confirmed_only") is not True
            or scope.get("historical_answers_reaccepted") is not False
            or scope.get("rejected_answers_repaired") is not False
            or scope.get("campaign_execution_complete") is not False
            or local.get("phases", [{}])[-1].get("acceptance_key") != CITATION_ACCEPTANCE_KEY
        ):
            raise ValueError("accepted_aggregation_local_finalization_binding")
        key = "citation-partial-finalization-result"
    else:
        raise ValueError("accepted_aggregation_local_finalization_version")
    if (
        local.get("original_result_checksum") != digest(original)
        or documents.read(key) != (local,)
        or local["phases"][-1]["run_id"] != documents.run_id
    ):
        raise ValueError("accepted_aggregation_local_finalization_binding")
    return local["phases"][-1]


def recovery_origin(campaign, previous, plan, original):
    """Locate the immutable rejection through its original Runtime owner."""
    from packages.evidence_details.accepted_aggregation import rejected_aggregation

    path = checked_path(previous)
    manifest = json.loads((path / "manifest.json").read_bytes())
    if json.loads((path / "plan.json").read_bytes()) != plan.model_dump(mode="json"):
        raise ValueError("accepted_aggregation_recovery_plan_changed")
    if (path / "result.json").exists():
        raise ValueError("accepted_aggregation_recovery_already_completed")
    docs = campaign.documents(manifest["run_id"])
    proof, state, sub = rejected_aggregation(
        docs,
        manifest=manifest,
        plan=plan.model_dump(mode="json"),
        original_result_checksum=digest(original),
    )
    attempt = manifest.get("recovery", {}).get("attempt", 0) + 1
    if attempt > 2:
        raise ValueError("accepted_aggregation_recovery_attempt_limit")
    return docs, dict(proof, previous_output=str(path), attempt=attempt), state, sub


def check_saved_aggregation(prior, submission, documents):
    """A rejected receipt is a terminal fact, not a four-object score bundle."""
    if len(prior) == 1 and prior[0].get("status") == "rejected":
        raise ValueError("accepted_aggregation_previous_rejection_requires_explicit_successor")
    grant = documents.read(GRANT_KEY)
    if (
        len(prior) != 4
        or prior[-1].get("status") != "accepted"
        or not grant
        or len(grant) != 1
        or prior[-1]["submission_checksum"] != digest(submission)
        or prior[-1]["permission_checksum"] != digest(grant[0])
    ):
        raise ValueError("accepted_aggregation_saved_binding")


def recovery_claim(output, plan, proof):
    return dict(
        schema_version="rejected-aggregation-successor/1.0.0",
        previous=proof,
        output=str(output),
        registration_key="local-accepted-fold:" + digest((str(output), plan))[7:51],
        plan_checksum=digest(plan),
        new_model_requests=0,
    )


def claim_recovery(documents, output, plan, proof):
    """Append one explicit link; never replace the old rejection/grant or journal."""
    claim = recovery_claim(output, plan, proof)
    with documents.session():
        if (
            digest(documents.sessions.read(documents.run_id)) != proof["control_checksum"]
            or RuntimeAgentJournal(documents).read()
        ):
            raise ValueError("accepted_aggregation_recovery_control_changed")
        previous = documents.read("accepted-aggregation-recovery-successor")
        if previous is not None and previous != (claim,):
            raise ValueError("accepted_aggregation_recovery_successor_conflict")
        documents.write("accepted-aggregation-recovery-successor", (claim,))


def aggregate_entry(plan_path, output, *, progress=None, retry_from=None):
    """A new local result, not a restart of any stopped real experiment.

    Repeating the same output validates current origin owners and returns exactly
    the saved acceptance/report. No API permission, credentials or tool port is
    installed. The local grant authorizes composition only, for fifteen minutes.
    """
    emit = progress or (lambda _: None)
    plan = read_plan(json.loads(checked_path(plan_path).read_bytes()))
    out = checked_path(output)
    source_paths = [checked_path(plan.facts)] + [checked_path(p.campaign) for p in plan.origins]
    source_paths += [
        checked_path(p.local_finalization) for p in plan.origins if p.local_finalization
    ]
    if isinstance(plan, SourceBudgetAggregationPlan):
        source_paths += [checked_path(p.campaign) for p in plan.source_budget_origins]
    if retry_from is not None:
        source_paths.append(checked_path(retry_from))
    if any(out == p or out in p.parents or p in out.parents for p in source_paths):
        raise ValueError("accepted_aggregation_output_overlaps_history")
    if out.exists() and (
        not (out / "plan.json").is_file()
        or json.loads((out / "plan.json").read_bytes()) != plan.model_dump(mode="json")
    ):
        raise ValueError("accepted_aggregation_output_conflict")
    campaign, old, phase_class = plan_campaign(plan)
    recovery = recovery_origin(campaign, retry_from, plan, old) if retry_from is not None else None
    if out.exists() and (out / "manifest.json").is_file():
        saved_manifest = json.loads((out / "manifest.json").read_bytes())
        if saved_manifest.get("recovery") != (recovery[1] if recovery else None):
            raise ValueError("accepted_aggregation_recovery_reference_required")
    generic = isinstance(plan, GenericAggregationPlan)
    source = review_set(
        plan,
        tuple(
            origin_reader(
                p,
                generic=generic,
                current_config=campaign.config
                if generic or isinstance(plan, ReasonedBatchAggregationPlan)
                else None,
                reasoned_batch_resize=isinstance(plan, ReasonedBatchAggregationPlan),
                client_transition=isinstance(plan, GenericClientAggregationPlan),
                network_transition=isinstance(plan, LegacyCapabilityAggregationPlan)
                and bool(campaign.selection.get("route_transition")),
                reused_origin_new_items_only=isinstance(plan, RetainedOriginAggregationPlan),
                retained_identity_aliases=isinstance(plan, RetainedAliasAggregationPlan),
                current_alias_reader=(lambda: phase.source_alias_manifest)
                if isinstance(plan, AliasAggregationPlan)
                else None,
                history_software=plan.history_software
                if isinstance(plan, GenericCapabilityAggregationPlan) and p.campaign == plan.facts
                else None,
            )
            for p in plan.origins
        ),
        campaign.config,
        campaign=campaign,
    )
    attempts = request_history(plan, campaign, old, source.readers)
    emit(
        dict(phase="aggregation_origins_restored", origins=len(source.readers), new_api_requests=0)
    )
    phase = (
        phase_class(
            campaign.base,
            observation_reader=campaign.accepted_observations,
            model_reader=campaign.read_model,
        )
        if generic
        else phase_class(campaign.base, observation_reader=campaign.accepted_observations)
    )
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(out / "plan.json", plan)
    key = "local-accepted-fold:" + digest((str(out), plan))[7:51]
    context, parent = campaign.register_phase(phase, key)
    docs = campaign.documents(context.run_id)

    class Owner:
        accepted_review_set = source

        def current(self):
            phase.verify(parent)
            return parent

    owner = Owner()
    started, sha = campaign.sessions.origin(docs.run_id)
    binding, _ = make_binding(
        initial=parent,
        config=campaign.config,
        started=started,
        context_checksum=sha,
        catalog=context.request.lock.catalog_ref,
    )
    planner = Planner(
        runtime=campaign.sessions, owner=ResearchPlannerSteps(snapshots=LiveSnapshots(owner))
    )
    control = campaign.sessions.read(docs.run_id)
    if control is None:
        planner.begin(binding)
        control = planner.read(docs.run_id)
    if control.phase != "stopped":
        if control.pending or control.cancelled or control.binding != binding:
            raise ValueError("accepted_aggregation_unexpected_control")
        planner.step(
            docs.run_id,
            Command(
                key="local-aggregation-stop", action="stop", expected_revision=control.revision
            ),
        )
    journal = RuntimeAgentJournal(docs)
    requests = AggregationRequests(source, owner, journal)
    before = campaign.sessions.read(docs.run_id)
    sub, _ = source.current(owner.current())
    if recovery is not None:
        from packages.evidence_details.accepted_aggregation import check_recovery_content

        check_recovery_content(recovery[2], recovery[3], parent, sub)
        history_sha = recovery[1]["request_history_checksum"]
        if history_sha is not None and history_sha != digest(attempts):
            raise ValueError("accepted_aggregation_recovery_accounting_changed")
        claim_recovery(recovery[0], out, plan, recovery[1])
    emit(
        dict(
            phase="aggregation_current_bindings_verified",
            items=len(sub.candidate.items),
            new_api_requests=0,
        )
    )
    if journal.read():
        raise ValueError("accepted_aggregation_unexpected_dispatch")
    manifest = dict(
        schema_version="accepted-review-aggregation-manifest/1.0.0",
        plan=plan.model_dump(mode="json"),
        state_checksum=digest(parent),
        origins_checksum=digest(sub.origins),
        submission_checksum=digest(sub),
        original_result_checksum=digest(old),
        run_id=docs.run_id,
    )
    if isinstance(
        plan,
        (
            AccountedAggregationPlan,
            GenericDefinitionAggregationPlan,
            LegacyCapabilityAggregationPlan,
        ),
    ):
        manifest["request_history_checksum"] = digest(attempts)
    if recovery is not None:
        manifest.update(
            schema_version="accepted-review-aggregation-manifest/1.1.0", recovery=recovery[1]
        )
    with docs.session():
        docs.write("accepted-aggregation-manifest", (manifest,))
        docs.write("dynamic-phase-input", (parent,))
        if isinstance(plan, GenericDefinitionAggregationPlan):
            docs.write(
                "generic-definition-wholecase-projection", (phase.definition_revision_manifest,)
            )
        if isinstance(plan, LegacyCapabilityAggregationPlan):
            docs.write("generic-capability-wholecase-projection", (phase.capability_projection,))
        if isinstance(plan, AliasAggregationPlan):
            docs.write("generic-review-source-aliases", (phase.source_alias_manifest,))
    write_once(out / "manifest.json", manifest)
    prior = docs.read("accepted:" + ACCEPTANCE_KEY)
    if prior is None:
        permission, sub = AggregationPermission.prepare(
            owner=owner, documents=docs, requests=requests, config=campaign.config
        )
        service = ResearchAcceptance(
            owner=owner, documents=docs, requests=requests, live_permission=permission
        )
        submitted = service.submit(sub, key=ACCEPTANCE_KEY)
        if not isinstance(submitted, dict) or submitted.get("status") != "submitted_not_accepted":
            raise ValueError("accepted_aggregation_submission_rejected")
        if service.accept(key=ACCEPTANCE_KEY).status != "accepted":
            raise ValueError("accepted_aggregation_acceptance_rejected")
    else:
        check_saved_aggregation(prior, sub, docs)
        service = ResearchAcceptance(owner=owner, documents=docs, requests=requests)
    emit(dict(phase="aggregation_score_saved", run_id=docs.run_id, new_api_requests=0))
    review = read_research_report(service, key=ACCEPTANCE_KEY, parent=parent)
    final = dict(
        run_id=docs.run_id,
        target=None,
        accepted=True,
        acceptance_key=ACCEPTANCE_KEY,
        error=None,
        execution_kind="local_accepted_review_aggregation",
    )
    result = dict(
        old,
        schema_version="multimodal-accepted-aggregation-result/1.0.0",
        complete=True,
        new_requests=0,
        original_requests=old["new_requests"],
        cumulative_requests=sum(a["requests"] for a in attempts),
        previous_attempts=attempts,
        origins=[o.model_dump(mode="json") for o in sub.origins],
        original_result_checksum=digest(old),
        execution_kind="local_accepted_review_aggregation_not_new_model_review",
        phases=[*old["phases"][:-1], final],
    )
    if recovery is not None:
        result["local_aggregation_recovery"] = recovery[1]
    if isinstance(plan, NullRevisionAggregationPlan):
        result.update(
            review_revision_policy=plan.selection_policy,
            review_revisions=[r.model_dump(mode="json") for r in plan.null_revisions],
        )
    if isinstance(plan, GenericDefinitionAggregationPlan):
        result["definition_revision_policy"] = plan.selection_policy
        result["definition_revision_projection_checksum"] = digest(
            phase.definition_revision_manifest
        )
    if isinstance(plan, LegacyCapabilityAggregationPlan):
        result["capability_revision_policy"] = plan.selection_policy
        result["capability_revision_projection_checksum"] = digest(phase.capability_projection)
    if generic:
        result.update(
            schema_version="generic-accepted-aggregation-result/1.0.0",
            prior_graph_requests=result["cumulative_requests"],
        )
    with docs.session():
        docs.write("dynamic-phase-result", (final,))
        docs.write("multimodal-case-result", (result,))
        if generic:
            docs.write("dynamic-case-result", (result,))
    (out / "review").mkdir(exist_ok=True)
    write_once(out / "result.json", result)
    write_once(out / "review/review-report.json", review)
    campaign.report_name = out.name
    campaign.report_key = "accepted-aggregation-report"
    campaign.review_report_path = out / "review/review-report.json"
    from .finalization import _saved_export
    from .report import export

    saved = docs.read(campaign.report_key)
    if generic:
        from packages.evaluation_preparation.dynamic_report import (
            GenericAggregateReportView,
        )
        from packages.evaluation_preparation.dynamic_report import (
            export as generic_export,
        )

        report_view = GenericAggregateReportView(campaign, out)
        # Full current policy with all original observations; never render the
        # selected-increment report as a whole-case score.
        if not saved:
            generic_export(report_view, result)
            saved = docs.read(campaign.report_key)
        exported = _saved_export(
            saved, key=campaign.report_key, result=result, review=review, name=out.name
        )
        from packages.evaluation_preparation.evidence_media import build
        from packages.evaluation_preparation.evidence_view import convert
        from packages.evaluation_preparation.report_production import media_links
        from packages.object_store import checksum

        source_path = Path(exported["json"])
        report = json.loads(source_path.read_bytes())
        media = build(
            report,
            source_checksum=checksum(source_path.read_bytes()),
            root=campaign.base.directory,
            binding=campaign.base.binding,
            runtime=campaign.base.runtime,
            objects=campaign.base.objects,
        )
        if report.get("report_production"):
            media = media_links(report, media)
        media_path = out / "media.json"
        write_once(media_path, media)
        exported["tree"] = convert(
            source_path,
            source_path.with_name(source_path.stem + "_证据树.html"),
            layout="tree-v3",
            media_path=media_path,
        )
    else:
        exported = (
            _saved_export(
                saved, key=campaign.report_key, result=result, review=review, name=out.name
            )
            if saved
            else export(campaign, result)
        )
    if journal.read() or campaign.sessions.read(docs.run_id) != before:
        raise ValueError("accepted_aggregation_changed_control_or_requests")
    return dict(
        exported,
        execution_kind=result["execution_kind"],
        accepted_items=len(sub.candidate.items),
        new_model_requests=0,
        memory=dict(
            status="not_archived_aggregation_reuse_not_installed", eligible_for_planning=False
        ),
    )

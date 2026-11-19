"""Project accepted 07 research facts; scores and coverage stay owned by 07."""

from agents.expert_agent.reporting import read_research_report
from packages.evidence_details.acceptance_models import DomainState, Submission
from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.evidence_details.review_tasks import project_review_tasks, review_payload
from packages.evidence_scoring.engine import path_available
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material, ReviewProposal
from packages.planner_control.models import Graph

from .models import PathGraphSummary


def seal_summary(**value):
    raw = dict(schema_version="path-graph-validation/1.0.0", **value)
    # Include defaults in the digest, so JSON and HTML have one exact fact set.
    model = PathGraphSummary(**raw, checksum=digest(None))
    raw = model.model_dump(mode="json", exclude={"checksum"})
    return validate(PathGraphSummary, {**raw, "checksum": digest(raw)})


class ResearchPathReader:
    """Installed acceptance + parent + key, not Agent-supplied rows or a score blob.

    owner.current must revalidate source/observation/reset/qualification dependencies.
    RuntimeDocuments additionally verifies actual objects and case/run identity.
    """

    def __init__(self, *, acceptance, parent, key):
        self.acceptance, self.parent, self.key = acceptance, validate(DomainState, parent), key

    def __call__(self):
        a, parent = self.acceptance, self.parent
        current = validate(DomainState, a.owner.current())
        a.documents.validate_scope(current.material.scope)
        report = read_research_report(a, key=self.key, parent=parent)
        if report.get("schema_version") != "research-successor-report/1.0.0":
            raise ValueError("accepted_research_required")
        records, receipt = report["records"], report["receipt"]
        if (
            records["schema_version"] != "case-observation-interpretation/2.1.0"
            or parent.policy.schema_version != "research-task-policy/2.0.0"
            or report["scores"] is None
            or report["scores"]["schema_version"] != "case-research-scoring/6.1.0"
        ):
            raise ValueError("path_graph_exact_versions_required")
        saved = a.documents.read("submission:" + self.key)
        if not saved or saved[0].get("status") != "submitted_not_accepted":
            raise ValueError("path_graph_submission_missing")
        sub = validate(Submission, saved[0]["submission"])
        if (
            not isinstance(sub.candidate, ReviewProposal)
            or digest(sub) != receipt["submission_checksum"]
            or sub.candidate.model_dump(mode="json") != records["candidate"]
            or sub.scope != parent.material.scope
            or sub.request_id != records["request_id"]
            or sub.source_checksums != tuple(s.reference for s in parent.material.sources)
            or sub.observation_checksums != tuple(o.reference for o in parent.material.observations)
            or sub.prompt_checksum != parent.prompt_checksum
            or sub.config_checksum != parent.config_checksum
        ):
            raise ValueError("path_graph_acceptance_binding")
        requests = [r for r in a.requests.records() if r.request_id == sub.request_id]
        if len(requests) != 1 or (
            requests[0].scope != sub.scope
            or requests[0].role != "review"
            or requests[0].prompt_checksum != sub.prompt_checksum
            or requests[0].config_checksum != sub.config_checksum
            or requests[0].input_checksum
            != digest(
                review_payload(parent.material, project_review_tasks(parent), sub.candidate.round)
            )
            or requests[0].output_checksum != digest(sub.candidate)
            or requests[0].error is not None
            or not requests[0].simulated
        ):
            raise ValueError("path_graph_request_binding")
        # Derive the exact successor Material using the original 08 envelope rules.
        raw = parent.material.model_dump(mode="json")
        raw["scope"]["graph"] = Graph(
            **receipt["graph"], kind="research_artifact", parent=receipt["parent_graph"]
        ).model_dump(mode="json")
        raw["scope"].pop("material_checksum")
        raw["scope"]["material_checksum"] = digest(raw)
        successor = parent.model_copy(update={"material": validate(Material, raw)})
        if current not in (parent, successor):
            raise CurrentResearchRejected("path_graph_source_or_version_changed")
        if digest(validate(DomainState, a.owner.current())) != digest(current):
            raise CurrentResearchRejected("path_graph_read_changed")
        return project_research(parent, report, current.material.scope.model_dump(mode="json"))


def project_research(parent, report, current_scope):
    """Internal projection: caller already joined immutable receipt and current owner."""
    scores = report["scores"]
    rows = {r["id"]: r for r in scores["reviews"]}
    items = {i["assessment_id"]: i for i in report["records"]["candidate"]["items"]}
    observations = {o.reference.code: o for o in parent.material.observations}
    resources = {r.code: r for r in parent.resources}
    paths, evidence, unresolved, groups = [], [], [], {}
    interpreted = set()
    for policy in parent.policy.items:
        row = rows[policy.id]
        obs = observations[policy.observation_code]
        status = (
            items[policy.id]["status"]
            if policy.id in items
            else ("assessed" if row["judgment"]["value"] is not None else "reference_missing")
        )
        usable = status == "assessed" and row["judgment"]["value"] is not None
        if usable:
            interpreted.add(policy.id)
        else:
            unresolved.append(
                dict(
                    target=policy.node,
                    item=policy.id,
                    reason=status,
                    method=policy.method,
                    observation_status=obs.status,
                )
            )
        paths.append(
            dict(
                target=policy.node,
                item=policy.id,
                operator="OR_OF_AND_DEPENDENCIES",
                branches=[
                    [resources[k].model_dump(mode="json") for k in p]
                    for p in policy.dependency_paths
                ],
                available=path_available(policy.dependency_paths, set()),
                interpreted=usable,
                definition=policy.definition.model_dump(mode="json"),
            )
        )
        evidence.append(
            dict(
                target=policy.node,
                item=policy.id,
                observation=obs.model_dump(mode="json"),
                source_refs=[
                    s.reference.model_dump(mode="json")
                    for s in parent.material.sources
                    if s.reference.code in policy.source_codes
                ],
                status=status,
                direction=row["direction"],
                judgment=row["judgment"],
                clinical_effect="NONE",
                correlation_group=policy.correlation_group,
                dependence_block=policy.dependence_block,
            )
        )
        groups.setdefault((policy.correlation_group, policy.dependence_block), []).append(policy.id)
    goals = []
    unclosed_items = set()
    for node in parent.nodes:
        ids = {p.id for p in parent.policy.items if p.node == node}
        slots = [
            dict(
                group=g.id,
                alternatives=sorted(ids & set(slot)),
                interpreted=bool(ids & set(slot) & interpreted),
            )
            for g in parent.policy.groups
            for slot in g.slots
            if ids & set(slot)
        ]
        unclosed_items.update(
            i for slot in slots if not slot["interpreted"] for i in slot["alternatives"]
        )
        goals.append(
            dict(
                target=node,
                declared_items=sorted(ids),
                interpreted_items=sorted(ids & interpreted),
                required_slots=slots,
                requirements_closed=bool(slots) and all(s["interpreted"] for s in slots),
                conclusion="research_requirements_only",
            )
        )
        if not slots:
            unresolved.append(
                dict(target=node, reason="no_declared_interpretation", required_for_coverage=True)
            )
    for gap in unresolved:
        if "item" in gap:
            gap["required_for_coverage"] = gap["item"] in unclosed_items
    summary = scores["root"]["summary"]
    risk = dict(
        original_channels={k: summary[k] for k in ("risk", "diagnostic", "model")},
        background_dominant=summary["risk"]["support"]
        > max(summary["diagnostic"]["support"], summary["model"]["support"]),
        score_is_not_diagnosis=True,
        automatic_removal=False,
    )
    # Same-target opposed directions are a review signal, never a new disease formula.
    conflicting = [
        g["target"]
        for g in goals
        if {e["direction"] for e in evidence if e["target"] == g["target"]} >= {"support", "refute"}
    ]
    return seal_summary(
        domain="research",
        purpose="research_offline",
        case_code=parent.material.scope.case_code,
        run_id=parent.material.scope.run_id,
        graph=report["receipt"]["graph"],
        binding=dict(
            parent_state=parent.model_dump(mode="json"),
            source_admission="original_owner.current_revalidated",
            parent_graph=report["receipt"]["parent_graph"],
            current_scope=current_scope,
            policy_checksum=digest(parent.policy),
            receipt_checksum=digest(report["receipt"]),
            records_checksum=digest(report["records"]),
            score_checksum=digest(scores),
            source_uses=report["source_uses"],
        ),
        paths=tuple(paths),
        goals=tuple(goals),
        evidence=tuple(evidence),
        unresolved=tuple(unresolved),
        duplicate_groups=tuple(
            dict(
                correlation_group=g,
                dependence_block=b,
                items=ids,
                handling="original_6.1_family_max_shared_input_discount",
            )
            for (g, b), ids in sorted(groups.items())
            if len(ids) > 1
        ),
        coverage=scores["root"]["coverage"],
        risk=risk,
        review_facts=dict(
            conflict=bool(conflicting),
            uncertain=any(g["required_for_coverage"] for g in unresolved),
            high_risk=False,
            required=bool(conflicting),
            no_progress=0,
        ),
        relations=tuple(
            dict(target=t, relation="opposed_research_directions") for t in conflicting
        ),
        original_report=report,
        original_report_checksum=digest(report),
    )

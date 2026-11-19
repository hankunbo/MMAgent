"""Produce source-bound display semantics before saving the authoritative report.

No new interpretation, medical threshold, score or clinical graph is created.
Organization uses original requirement modalities, not disease-name templates.
"""

import copy
import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

from .evidence_assets import asset_identity, reference_values
from .evidence_presentation import pointer_part, resolve

PROFILE = "case-report-production/1.0.0"


def implementation():
    files = [
        "report_production.py",
        "dynamic_report.py",
        "comprehensive_materials.py",
        "generic_input.py",
        "generic_repair.py",
        "generic_campaign.py",
        "quality_materials.py",
        "generic_run.py",
        "evidence_media.py",
        "evidence_assets.py",
        "evidence_presentation.py",
        "presentation_contracts.py",
        "evidence_tree_v3.py",
        "evidence_view.py",
        "evidence_tree.py",
    ]
    paths = [Path(__file__).parent / name for name in files]
    paths += [Path("apps/fairvlmed_case.py")]
    paths += [
        Path("apps/templates") / name
        for name in (
            "evidence_tree_reader_v3.js",
            "evidence_tree_reader_v3.css",
            "evidence_tree_reader_v2.css",
            "readable_evidence_graph.html",
        )
    ]
    return {str(p): checksum(p.read_bytes()) for p in paths}


def _body(observation):
    try:
        value = json.loads((observation or {}).get("value") or "null")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def review_revisions(report):
    """Project the accepted owner's explicit selection; never infer a new judgment."""
    result = report.get("result", {})
    raw = result.get("review_revisions")
    if raw is None and "review_revision_policy" not in result:
        return {}
    from packages.evidence_details.accepted_aggregation import NullReviewRevision

    if (
        result.get("execution_kind") != "local_accepted_review_aggregation_not_new_model_review"
        or result.get("review_revision_policy") != "new_context_accepted_null_revision/1.0.0"
        or not isinstance(raw, list)
        or not 1 <= len(raw) <= 6
    ):
        raise ValueError("report_review_revision_policy")
    revisions = [NullReviewRevision.model_validate(r) for r in raw]
    detail = report.get("original_review") or {}
    candidates = {
        i["assessment_id"]: i
        for i in detail.get("records", {}).get("candidate", {}).get("items", [])
    }
    if detail.get("receipt", {}).get("status") != "accepted" or len(
        {r.assessment_id for r in revisions}
    ) != len(revisions):
        raise ValueError("report_review_revision_acceptance")
    result = {}
    for revision in revisions:
        current = candidates.get(revision.assessment_id)
        if current is None or digest(current) != revision.replacement_item_checksum:
            raise ValueError("report_review_revision_selected_item")
        result[revision.assessment_id] = dict(
            schema_version="accepted-review-revision-presentation/1.0.0",
            **revision.model_dump(mode="json"),
            current_judgment=current["judgment"],
            previous_records_preserved=True,
            added_evidence_votes=0,
            explanation="补充方法上下文后取得新的已接纳评阅；本次组合只选择新版本，旧评阅保留，不新增独立证据票。",
        )
    return result


def definition_rationales(report, state):
    """Dispatch saved owner formats; a display binding never grants qualification."""
    revision = report.get("definition_revision")
    if revision is None:
        return {}
    version = revision.get("schema_version")
    if version in (
        None,
        "campaign-requirement-revision/1.0.0",
        "campaign-requirement-revision/1.1.0",
    ):
        result = {}
        for key in set(revision["added"]) | {r["id"] for r in revision["replaced"]}:
            origin = next(r for r in revision["rows"] if r["id"] == key)
            definitions = [
                (k, v)
                for k, v in revision["accepted"].items()
                if digest(v) == origin["definition_receipt_checksum"]
            ]
            if len(definitions) != 1:
                raise ValueError("report_definition_receipt_binding")
            definition_key, accepted = definitions[0]
            result[key] = dict(
                purpose=accepted["draft"]["extraction_reason"],
                research_definition=accepted,
                revision=revision["revision"],
                denominator_before=revision["denominator_before"],
                denominator_after=revision["denominator_after"],
                definition_pointer="/definition_revision/accepted/" + pointer_part(definition_key),
            )
        return result
    increment = version == "generic-definition-review-report-scope/1.0.0"
    if not increment and version != "generic-definition-wholecase-report-scope/1.0.0":
        raise ValueError("report_definition_unsupported_version")
    projection = revision["definition_projection"]
    items = {i["id"]: i for i in state["policy"]["items"]}
    if projection is None:
        if (
            not increment
            or items
            or revision["interpretation_ids"]
            or (
                revision["whole_case_aggregation_required"]
                or revision["status"] != "no_adopted_definition"
                or revision["new_requirements"]
            )
        ):
            raise ValueError("report_definition_no_adoption_binding")
        return {}
    requirements = {r["id"]: r for r in state["policy"]["requirements"]}
    observations = {o["reference"]["code"]: o for o in state["material"]["observations"]}
    selected = projection["selected_ids"]
    rows = projection["interpretation_revisions"]
    if (
        projection["schema_version"]
        not in (
            "generic-confirmed-interpretation-projection/1.0.0",
            "generic-registered-interpretation-projection/1.0.0",
        )
        or projection["projected_state_checksum"] != digest(state)
        or projection["phase_run_id"] != state["material"]["scope"]["run_id"]
        or projection["selected_only"] != increment
        or projection["requirements_after"] != len(requirements)
        or len(selected) != len(set(selected))
        or sorted(r["assessment_id"] for r in rows) != sorted(selected)
        or not selected
    ):
        raise ValueError("report_definition_projection_binding")
    if increment and (
        revision["interpretation_ids"] != sorted(items)
        or sorted(selected) != sorted(items)
        or revision["status"] != "new_interpretation_increment"
        or not revision["whole_case_aggregation_required"]
    ):
        raise ValueError("report_definition_selection_binding")
    result = {}
    for index, row in enumerate(rows):
        key = row["assessment_id"]
        item, requirement = items.get(key), requirements.get(key)
        observed = observations.get((item or {}).get("observation_code"))
        if (
            item is None
            or requirement is None
            or observed is None
            or row["current_definition"] != requirement["definition"]
            or item["definition"] != requirement["definition"]
            or row["observation"] != observed["reference"]
            or (row["method"], row["unit"]) != (item["method"], item["unit"])
            or row["clinical_acceptance"] is not False
            or row["qualification"] != "campaign_research_only"
        ):
            raise ValueError("report_definition_item_binding")
        pointer = "/definition_revision/definition_projection/interpretation_revisions/" + str(
            index
        )
        result[key] = dict(
            purpose="使用原研究入口已接纳的解释修订；观察、方法、来源及适用限制由保存记录精确绑定。",
            definition_pointer=pointer,
            denominator_before=projection["requirements_before"],
            denominator_after=projection["requirements_after"],
            accepted_definition_revision=dict(
                schema_version="accepted-definition-revision-presentation/1.0.0",
                **copy.deepcopy(row),
                denominator_before=projection["requirements_before"],
                denominator_after=projection["requirements_after"],
                source_pointer=pointer,
                added_evidence_votes=0,
                historical_records_preserved=True,
            ),
        )
    return result


def capability_rationales(report, state):
    """Saved selection/denominator provenance; no new medical interpretation."""
    revision = report.get("capability_revision")
    if revision is None:
        return {}
    version = revision.get("schema_version")
    increment = version == "generic-capability-increment-report-scope/1.0.0"
    if not increment and version != "generic-capability-wholecase-report-scope/1.0.0":
        raise ValueError("report_capability_unsupported_version")
    projection = revision["projection"]
    scope = state["material"]["scope"]
    required = {r["id"] for r in state["policy"]["requirements"]}
    retained, added = projection["retained_requirement_ids"], projection["added_requirement_ids"]
    dispatched = projection["dispatch_targets"]
    if (
        projection["schema_version"] != "generic-capability-demand-projection/1.0.0"
        or projection["phase_input_checksum"] != digest(state)
        or projection["run_id"] != scope["run_id"]
        or projection["case_code"] != scope["case_code"]
        or projection["target"] != state["material"]["goal_code"]
        or len(retained) != len(set(retained))
        or len(added) != len(set(added))
        or set(retained) & set(added)
        or set(retained) | set(added) != required
        or len(dispatched) != len(set(dispatched))
        or not set(dispatched) <= set(added)
        or projection["requirements_before"] != len(retained)
        or projection["requirements_after"] != len(required)
        or projection["independent_evidence_count_claimed"] is not False
        or revision["whole_case_aggregation_required"] != increment
        or (increment and not {i["id"] for i in state["policy"]["items"]} <= set(dispatched))
    ):
        raise ValueError("report_capability_projection_binding")
    result = {}
    for index, key in enumerate(added):
        result[key] = dict(
            kind="current_capability_selected_requirement",
            selection_checksum=projection["selection_checksum"],
            source_pointer="/capability_revision/projection/added_requirement_ids/" + str(index),
            denominator_before=len(retained),
            denominator_after=len(required),
            selected_for_acquisition=key in dispatched,
            observation_completion_not_inferred=True,
            independent_evidence_count_claimed=False,
        )
    for row in projection["covered_by_existing_scope"]:
        if row["existing"] not in retained:
            raise ValueError("report_capability_existing_scope_binding")
        result.setdefault(
            row["existing"], dict(kind="existing_question_scope_reused", scope_links=[])
        )["scope_links"].append(copy.deepcopy(row))
    return result


def produce(report, state, *, artifacts=(), measured=None, origins=None):
    """Called with the original Runtime's frozen phase and verified Artifact records.

    Missing/unscored requirements are retained; original accepted bytes are untouched.
    Native task origins come from the SAME measurement projection that made the values.
    """
    result = copy.deepcopy(report)
    if "report_production" in result:
        raise ValueError("report_production_already_present")
    raw = state.model_dump(mode="json") if hasattr(state, "model_dump") else state
    scope, policy = raw["material"]["scope"], raw["policy"]
    if result["result"]["case_code"] != scope["case_code"]:
        raise ValueError("report_production_case_binding")
    required = {r["id"]: r for r in policy["requirements"]}
    if len(required) != len(policy["requirements"]) or [
        r["id"] for r in result["requirements"]
    ] != list(required):
        raise ValueError("report_production_requirement_binding")
    items = {i["id"]: i for i in policy["items"]}
    observations = {o["reference"]["code"]: o for o in raw["material"]["observations"]}
    sources = {s["reference"]["code"]: s for s in raw["material"]["sources"]}
    assets = {}
    for artifact in artifacts:
        a = artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else artifact
        if a["scope"]["case_code"] != scope["case_code"]:
            raise ValueError("report_production_artifact_scope")
        if a["asset_code"] in assets and assets[a["asset_code"]] != a:
            raise ValueError("report_production_artifact_identity")
        assets[a["asset_code"]] = a
    result["production_inputs"] = dict(
        schema_version="report-production-inputs/1.0.0",
        scope=scope,
        policy=policy,
        sources=list(sources.values()),
        artifacts=list(assets.values()),
    )
    revisions = review_revisions(report)
    definitions = definition_rationales(report, raw)
    capabilities = capability_rationales(report, raw)
    if revisions:
        result["review_revision_provenance"] = revisions
    nodes, edges, groups = [], [], set()
    scored = {r["id"]: r for r in (report.get("scores") or {}).get("reviews", [])}
    for index, row in enumerate(result["requirements"]):
        key, requirement = row["id"], required[row["id"]]
        definition = requirement["definition"]
        if row["sources"] != definition["source_refs"]:
            raise ValueError("report_production_source_binding")
        for ref in definition["source_refs"]:
            if ref["code"] not in sources or sources[ref["code"]]["reference"] != ref:
                raise ValueError("report_production_source_missing")
        item = items.get(key, {})
        observation = observations.get(item.get("observation_code"))
        if item and observation is None:
            raise ValueError("report_production_observation_missing")
        dependencies = item.get("dependency_paths", [])
        codes = {code for path in dependencies for code in path}
        bound_assets = [a for code, a in assets.items() if code in codes]
        method = item.get("method")
        binding = dict(
            schema_version="check-execution-binding/1.0.0",
            method=method,
            observation_reference=(observation or {}).get("reference"),
            dependency_paths=dependencies,
            artifacts=bound_assets,
            unresolved_resource_codes=sorted(codes - set(assets)),
            native_task=None,
            native_result_pointer=None,
            semantics="saved_execution_provenance_not_independent_disease_evidence",
        )
        measured_row = (measured or {}).get(key)
        body = _body(observation)
        if measured_row and all(body.get(k) == v for k, v in measured_row.items()):
            task = (origins or {}).get(key)
            full = body.get("full_record")
            if task and full in codes and full in assets:
                ptr = "/native_tools/" + (
                    "original_measurements"
                    if task == "original_measurements"
                    else "results/" + pointer_part(task)
                )
                saved = resolve(report, ptr)
                binding.update(
                    native_task=task,
                    native_result_pointer=ptr,
                    native_result_checksum=digest(saved),
                    native_record=saved,
                )
        if observation and observation.get("image"):
            binding["view"] = observation["image"]
        row["execution_binding"] = binding
        if scores := result.get("scores"):
            if scores.get("schema_version") == "case-research-scoring/11.0.0":
                row["direction_assessment"] = scores["direction_assessments"].get(key)
        ptr = "/production_inputs/policy/requirements/" + str(index)
        row["presentation_sections"] = dict(
            schema_version="evidence-sections/1.0.0",
            rationale=dict(
                purpose="根据当前来源中的目标关联保留此检查；不是新增诊断结论。",
                target=raw["material"]["goal_summary"],
                relationship=definition["interpretation"],
                source_refs=definition["source_refs"],
                basis_pointer=ptr + "/definition/interpretation",
            ),
            acquisition=dict(
                examination=definition["examination"],
                input_modalities=requirement["input_modalities"],
                method=method,
                execution_binding_pointer="/requirements/" + str(index) + "/execution_binding",
                missing_reason=requirement.get("missing_reason"),
                method_status="executed_observation_method"
                if method
                else "not_obtained_no_model_claim",
            ),
            reference=dict(
                interpretation=definition["interpretation"],
                applicability_conditions=definition["applicability_conditions"],
                interpretation_status=definition["interpretation_status"],
                definition_reference=definition["reference"],
                source_refs=definition["source_refs"],
                basis_pointer=ptr + "/definition",
                threshold_invented=False,
            ),
            sources=[sources[r["code"]] for r in definition["source_refs"]],
            observation=observation,
            score_pointer="/scores/reviews/" + str(list(scored).index(key))
            if key in scored
            else None,
            dependencies=binding,
        )
        if key in revisions:
            row["presentation_sections"]["rationale"]["accepted_review_revision"] = revisions[key]
        if key in definitions:
            row["presentation_sections"]["rationale"].update(definitions[key])
        if key in capabilities:
            row["presentation_sections"]["rationale"]["capability_selection"] = capabilities[key]
        # Upstream-declared navigation, not score grouping or clinical causality.
        modalities = tuple(sorted(requirement["input_modalities"]))
        hypothesis = policy.get("hypothesis_bindings", {}).get(key)
        parent_node = "root"
        if hypothesis:
            hid = "hypothesis:" + hypothesis
            if hid not in groups:
                groups.add(hid)
                h = next(h for h in policy["hypotheses"] if h["code"] == hypothesis)
                nodes.append(
                    dict(
                        id=hid,
                        type="goal",
                        title=h["title"],
                        semantics="research_hypothesis_not_independent_observation",
                    )
                )
                edges.append(dict(source="root", target=hid, relation="contains"))
            parent_node = hid
        gid = "organization:" + digest((hypothesis, modalities) if hypothesis else modalities)[7:23]
        if gid not in groups:
            groups.add(gid)
            nodes.append(
                dict(
                    id=gid,
                    type="goal",
                    title="检查输入 · " + " / ".join(modalities),
                    semantics="input_modality_organization_not_causal_path",
                )
            )
            edges.append(dict(source=parent_node, target=gid, relation="contains"))
        edges.append(dict(source=gid, target="check:" + key, relation="contains"))
        task = binding["native_task"]
        if task and task != "original_measurements":
            edges.append(dict(source="tool:" + task, target="check:" + key, relation="produces"))
    result["presentation_structure"] = dict(
        schema_version="case-presentation-structure/1.0.0", nodes=nodes, edges=edges
    )
    if result.get("upstream_reuse"):
        outcomes = []
        for plan in result["upstream_reuse"]["plan"]["skills"]["plans"]:
            identifier = plan["requirement_id"]
            outcomes.append(
                dict(
                    requirement_id=identifier,
                    memory_key=plan.get("memory_key"),
                    memory_checksum=plan.get("memory_checksum"),
                    selected_at_planning=True,
                    observation_obtained=identifier in items,
                    review_saved=identifier in scored,
                    observation_reference=observations[items[identifier]["observation_code"]][
                        "reference"
                    ]
                    if identifier in items
                    else None,
                    semantics="selected_revalidated_hint_and_observed_outcome_not_proof_of_benefit",
                    clinical_publication=False,
                )
            )
        result["upstream_reuse"]["outcomes"] = outcomes
    result["report_production"] = dict(
        schema_version=PROFILE,
        input_checksum=scope["input_checksum"],
        case_code=scope["case_code"],
        policy_checksum=digest(policy),
        original_scores_checksum=digest(report.get("scores")),
        original_review_checksum=digest(report.get("original_review")),
        organization="declared_input_modality_groups_not_clinical_causal_edges",
        clinical_acceptance=False,
    )
    return result


def campaign_report(report, campaign, state):
    from .comprehensive_materials import measured_projection

    assets = []
    for resource in state.resources:
        a = campaign.base.runtime.read_case_artifact(state.material.scope.case_code, resource.code)
        if a is not None:
            if a.checksum != resource.checksum:
                raise ValueError("report_production_current_artifact")
            model = report.get("model_output")
            if model and a.asset_code == model["source"]["artifact_code"]:
                # Model output uses its original deployment-owned output store,
                # not the visible-input store. That owner verifies bytes/scope.
                if (
                    a.checksum != model["source"]["artifact_checksum"]
                    or campaign.read_model() != model
                ):
                    raise ValueError("report_production_model_owner_binding")
                assets.append(a)
                continue
            raw = campaign.base.objects.read_bounded(a.uri, 4194304)
            if checksum(raw) != a.checksum:
                raise ValueError("report_production_current_content")
            extra = getattr(campaign.base, "extra_artifact", None)
            if (
                extra
                and a.asset_code == extra.asset_code
                and json.loads(raw) != report.get("native_tools")
            ):
                raise ValueError("report_production_native_artifact_binding")
            assets.append(a)
    tools = report.get("native_tools")
    measured, origins = measured_projection(tools, include_origins=True) if tools else ({}, {})
    return produce(report, state, artifacts=assets, measured=measured, origins=origins)


def media_links(report, legacy):
    """Precise per-check image binding derived from saved execution records only."""
    items = [dict(v, id="legacy:" + k, kind="image", role=k) for k, v in legacy["images"].items()]
    links = []
    for index, row in enumerate(report["requirements"]):
        pointer = "/requirements/" + str(index) + "/execution_binding"
        value = resolve(report, pointer)
        refs = reference_values(value)
        for asset in items:
            if refs & asset_identity(asset):
                links.append(
                    dict(
                        node_id="check:" + row["id"],
                        asset_id=asset["id"],
                        source_pointer=pointer,
                        source_value_checksum=digest(value),
                    )
                )
    return dict(
        schema_version="evidence-assets/1.0.0",
        source_checksum=legacy["source_checksum"],
        case_code=legacy["case_code"],
        input_checksum=legacy["input_checksum"],
        items=items,
        links=links,
    )

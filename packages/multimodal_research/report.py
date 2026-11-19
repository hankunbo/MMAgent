"""Original accepted records -> six-section tree -> immutable Runtime readback."""

import base64
import html
import json
from pathlib import Path

from packages.evaluation_preparation.evidence_tree_v3 import render
from packages.evaluation_preparation.report_production import produce
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import canonical, digest, validate

from .report_io import authorize_recovery, publish_html, publish_json, review_mirror


def whole_case_rescored(final, scores):
    """Partial domain acceptance is not full Review completion."""
    return bool(
        scores
        and final["accepted"]
        and final["target"] is None
        and not scores.get("review_completion", {}).get("pending_ids", [])
    )


def pending_review_scope(protocol, review_ids):
    """Describe saved recovery provenance; never invent an earlier acceptance."""
    if protocol.get("truncated_review"):
        proof = protocol["truncated_review"]
        if (
            protocol.get("schema_version") != "multimodal-campaign-renewal/1.53.0"
            or list(review_ids) != proof["review_target_ids"]
        ):
            raise ValueError("truncated_review_report_scope")
        return dict(
            kind="confirmed_truncation_review_increment",
            baseline=protocol["tip"],
            original_acceptance=proof["inventory"],
            review_ids=list(review_ids),
            original_failed_request=proof["request_id"],
            whole_case_rescored=False,
            explanation="仅补评已确认截断批次；保留原已接纳评分与失败，增量分须经原owner汇总后才是整例结果。",
        )
    pending = protocol["pending_review"]
    if protocol.get("schema_version") == "multimodal-campaign-renewal/1.41.0":
        proof = pending.get("initial_failed_review")
        if (
            not proof
            or pending["accepted_ids"]
            or list(review_ids) != pending["selected_ids"]
            or proof["attempt"]["run_id"] != protocol["tip"]["run_id"]
        ):
            raise ValueError("initial_failed_report_origin")
        return dict(
            kind="initial_failed_review_unstarted_increment",
            baseline=protocol["tip"],
            original_acceptance=None,
            initial_failed_review=proof,
            review_ids=list(review_ids),
            quarantined_batches=pending["quarantined_batches"],
            whole_case_rescored=False,
            explanation="原首批评阅失败、没有旧接纳评分。本次仅评阅原未发送项目；"
            "旧未知批次保持隔离，增量评分不表示整例评阅完成。",
        )
    return dict(
        kind="pending_review_increment",
        baseline=protocol["tip"],
        original_acceptance=pending,
        review_ids=list(review_ids),
        whole_case_rescored=False,
        explanation="仅补评原未完成项目；原已接纳评分不重评、不改写。未知请求默认隔离，增量分非整例重算。",
    )


def acquired_observations(campaign, *, images_only=False):
    """Confirmed installed and phase facts; scope filtering is a separate step."""
    result = {
        key: value
        for key, value in campaign.base.observations.items()
        if not images_only or value.image is not None
    }
    for key, value in campaign.accepted_observations().items():
        if images_only and value.image is None:
            continue
        if key in result and canonical(result[key]) != canonical(value):
            raise ValueError("multimodal_acquired_observation_conflict")
        result[key] = value
    return result


def export_acquisition_progress(campaign, phase):
    """Immutable original-owner checkpoint, never a premature accepted root score.

    Historical accepted judgments are displayed separately from new observations
    awaiting Review. The final original acceptance transaction is not split.
    """
    from datetime import UTC, datetime

    from apps.vectorengine_single_case import write_once
    from packages.evidence_details.acceptance_models import read_records

    docs = campaign.documents(phase["run_id"])
    if docs.read("dynamic-phase-result") != (phase,):
        raise ValueError("progress_phase_not_saved")
    key = "multimodal-acquisition-progress"
    saved = docs.read(key)
    if saved:
        envelope = saved[0]
        if envelope["snapshot"]["phase"] != phase:
            raise ValueError("progress_phase_changed")
    else:
        reader = (
            getattr(campaign, "progress_accepted_reader", None)
            or campaign.base.confirmed_review_reader
        )
        from .definition_alternative import SEMANTICS
        from .definition_alternative import VERSION as definition_version

        definition_increment = campaign.protocol.get("schema_version") == definition_version
        unscored = (
            campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.52.0"
            or definition_increment
        )
        if definition_increment and (
            campaign.protocol.get("definition_alternative") != SEMANTICS
            or campaign.protocol.get("definition_increment_only") is not True
            or digest(campaign.base.definition_current())
            != campaign.protocol.get("source_revision_checksum")
        ):
            raise ValueError("progress_current_definition_changed")
        if reader is None:
            if not unscored or campaign.protocol.get("reuse_assessment_ids") != []:
                raise ValueError("progress_historical_acceptance_missing")
            bundle, interpretations = None, ()
        else:
            bundle = reader.documents.read("accepted:" + reader.acceptance_key)
            interpretations = read_records(bundle[0]).interpretations
        observations = acquired_observations(campaign)
        old_ids = set(campaign.protocol["reuse_assessment_ids"])
        rows = []
        for requirement in campaign.base.ledger:
            item = next(
                (i for i in interpretations if i.review.assessment_id == requirement["id"]),
                None,
            )
            observation = observations.get(requirement["id"])
            rows.append(
                dict(
                    id=requirement["id"],
                    examination=requirement["definition"].examination,
                    source_refs=[
                        r.model_dump(mode="json") for r in requirement["definition"].source_refs
                    ],
                    observation=observation.model_dump(mode="json") if observation else None,
                    historical_review=item.model_dump(mode="json")
                    if item and item.review.assessment_id in old_ids
                    else None,
                    status="historical_accepted_local_judgment"
                    if requirement["id"] in old_ids
                    else "observation_obtained_pending_review"
                    if observation is not None and observation.status == "obtained"
                    else "observation_" + observation.status
                    if observation is not None
                    else "not_obtained",
                )
            )
        snapshot = dict(
            schema_version="multimodal-acquisition-progress/1.3.0"
            if definition_increment
            else "multimodal-acquisition-progress/1.2.0"
            if unscored
            else "multimodal-acquisition-progress/1.1.0",
            final=False,
            created_at=datetime.now(UTC).isoformat(),
            campaign=campaign.campaign_id,
            protocol_checksum=digest(campaign.protocol),
            phase=phase,
            historical_acceptance_checksum=digest(bundle) if bundle is not None else None,
            rows=rows,
            requirements=len(rows),
            observation_records=sum(r["observation"] is not None for r in rows),
            observations=sum(
                r["observation"] is not None and r["observation"]["status"] == "obtained"
                for r in rows
            ),
            observation_status_counts={
                status: sum(
                    r["observation"] is not None and r["observation"]["status"] == status
                    for r in rows
                )
                for status in ("obtained", "unknown", "missing", "ungradable")
            },
            historical_assessed=len(old_ids),
            new_accepted_score=None,
            requests_this_window=campaign.usage(),
            limitation="新增观察尚未完成本轮方向接纳；历史评分不是本轮新增评分。",
        )
        if definition_increment:
            snapshot["definition_revision_checksum"] = campaign.protocol["source_revision_checksum"]
        data = html.escape(json.dumps(snapshot, ensure_ascii=False, indent=2))
        table = "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(row[k])) + "</td>" for k in ("id", "examination", "status")
            )
            + "</tr>"
            for row in rows
        )
        page = (
            '<!doctype html><html lang="zh"><meta charset="utf-8"><title>取证中期结果</title>'
            "<h1>取证中期结果（非最终评分）</h1><p>"
            + html.escape(snapshot["limitation"])
            + '</p><table border="1"><tr><th>检查</th><th>需求</th><th>状态</th></tr>'
            + table
            + "</table><details><summary>原观察、历史评阅与精确引用</summary><pre>"
            + data
            + "</pre></details></html>"
        )
        envelope = dict(
            snapshot=snapshot,
            snapshot_checksum=digest(snapshot),
            html=page,
            html_checksum=digest(page),
        )
        docs.write(key, (envelope,))
    if (
        digest(envelope["snapshot"]) != envelope["snapshot_checksum"]
        or digest(envelope["html"]) != envelope["html_checksum"]
    ):
        raise ValueError("progress_envelope_binding")
    root = campaign.output / "progress"
    root.mkdir(mode=0o700, exist_ok=True)
    stem = "phase-" + digest(phase)[7:23]
    write_once(root / (stem + ".json"), envelope["snapshot"])
    publish_html(root / (stem + ".html"), envelope["html"])
    return dict(
        json=str(root / (stem + ".json")),
        html=str(root / (stem + ".html")),
        observations=envelope["snapshot"]["observations"],
        historical_assessed=envelope["snapshot"]["historical_assessed"],
        new_accepted_score=None,
    )


def restore_export(campaign, result, directory=Path("docs/exports")):
    """Exact saved bytes are the recovery authority, not the current renderer."""
    from .report_storage import unpack

    final = result["phases"][-1]
    docs = campaign.documents(final["run_id"])
    key = getattr(campaign, "report_key", "dynamic-case-report")
    report, content = unpack(docs.read(key), key=key)
    detail = report["original_review"]
    if report["result"] != result or report["task"] != result["task"]:
        raise ValueError("multimodal_saved_export_binding")
    if final["target"] is None and final["accepted"]:
        accepted = docs.read("accepted:" + final["acceptance_key"])
        if (
            not accepted
            or detail is None
            or (detail["records"], detail["graph"], report["scores"], detail["receipt"])
            != (accepted[0], accepted[1], accepted[2]["scores"], accepted[3])
        ):
            raise ValueError("multimodal_saved_export_binding")
    elif detail is not None or report["scores"] is not None:
        raise ValueError("multimodal_saved_export_binding")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("Multimodal_" + getattr(campaign, "report_name", campaign.output.name))

    def recover():
        authorize_recovery(campaign, final)

    json_path = publish_json(path.with_suffix(".json"), report, recover=recover)
    page = publish_html(path.with_suffix(".html"), content["html"], recover=recover)
    return dict(
        complete=result["complete"],
        requests=result["new_requests"],
        json=str(json_path),
        html=str(page),
        report_checksum=digest(report),
        task_clinically_validated=False,
    )


def native_artifacts(base):
    """Project original single/collection records without changing their identity."""
    from .native_fundus import records

    native = getattr(getattr(base, "active_owner", None), "native", None)
    return tuple(artifact for record in records(native) for artifact in record["artifacts"])


def media(campaign, report):
    images = {}
    # Package views actually referenced by saved observations, not every planned
    # check. A plan can contain hundreds of absent inputs without creating images.
    # Native outputs below retain their own identities, independently of Review.
    targets = []
    acquired = report.get("acquisition_observations", {})
    # Media binds only image-bearing facts. A legacy map's numeric projection
    # is not a competing image, and must not hide its accepted visual result.
    # Two different image-bearing facts under one ID still conflict.
    originals = acquired_observations(campaign, images_only=True)
    for row in report["requirements"]:
        observation = (
            row.get("observation") or row.get("unscored_observation") or acquired.get(row["id"])
        )
        if observation and observation.get("image"):
            matches = [
                key
                for key, value in originals.items()
                if value.model_dump(mode="json") == observation
            ]
            if len(matches) != 1:
                raise ValueError("multimodal_report_image_observation_binding")
            targets.append((row["id"], matches[0]))
    for target, source_target in targets:
        requirement = next(r for r in campaign.base.ledger if r["id"] == source_target)
        if requirement.get("acquisition_method") == "source_measurement":
            continue  # Numeric evidence belongs in saved facts, not a fabricated image.
        image, raw, view = campaign.base.view(source_target)
        if image.model_dump(mode="json") != originals[source_target].image.model_dump(mode="json"):
            raise ValueError("multimodal_report_image_observation_binding")
        component = next(
            c for c in campaign.base.manifest.components if c.id == requirement["component"]
        )
        images[target] = dict(
            reference=image.reference.model_dump(mode="json"),
            checksum=image.reference.checksum,
            shape=[image.height, image.width],
            data_uri="data:image/png;base64," + base64.b64encode(raw).decode(),
            label=target,
            limits=view["limits"],
            transform=view["transform"],
            sampled_fraction=view["inspected_fraction"],
            title=requirement["definition"].examination,
            modality=view["modality"],
            laterality=component.laterality,
            timepoint=component.visit,
        )
    artifacts = native_artifacts(campaign.base)
    if artifacts:
        for artifact in artifacts:
            if artifact["media_type"] != "image/png":
                continue
            raw = campaign.base.objects.read_bounded(artifact["uri"], 4194304)
            images[artifact["asset_code"]] = dict(
                reference=artifact,
                checksum=artifact["checksum"],
                shape=artifact["dimensions"],
                data_uri="data:image/png;base64," + base64.b64encode(raw).decode(),
                title="AutoMorph · " + artifact["extensions"]["output_port"],
                modality="color_fundus",
                label=artifact["extensions"]["output_port"],
                transform="RGB_bilinear_512_original_prediction_grid",
                limits=["研究分割；不是人工标注；未验证分期"],
            )
    return dict(
        schema_version="evidence-media/1.0.0",
        source_checksum=digest(report),
        case_code=report["result"]["case_code"],
        input_checksum=campaign.base.manifest.checksum,
        images=images,
    )


def export(campaign, result, directory=Path("docs/exports")):
    if "scope_method_plan" in result:
        from .scope_selection import ensure

        # Install only owner-confirmed saved native facts BEFORE constructing
        # observations, quality, dependencies and media. Export cannot launch.
        if ensure(campaign, allow_native=False) != result["scope_method_plan"]:
            raise ValueError("scope_method_report_result_binding")
    final = result["phases"][-1]
    docs = campaign.documents(final["run_id"])
    parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
    detail, scores, mirror_diagnostic = None, None, None
    if final["target"] is None and final["accepted"]:
        # A missing local Review report is not a missing score. Reconstruct only
        # through the original read-only report owner; never accept/score again.
        from types import SimpleNamespace

        from agents.expert_agent.reporting import read_research_report
        from packages.evidence_details.acceptance import ResearchAcceptance

        acceptance = ResearchAcceptance(
            owner=SimpleNamespace(current=lambda: parent), documents=docs, requests=None
        )
        detail = read_research_report(acceptance, key=final["acceptance_key"], parent=parent)
        saved = docs.read("accepted:" + final["acceptance_key"])
        if (detail["records"], detail["graph"], detail["receipt"], detail["scores"]) != (
            saved[0],
            saved[1],
            saved[3],
            saved[2]["scores"],
        ) or detail["material"] != parent.material.model_dump(mode="json"):
            raise ValueError("multimodal_report_owner_binding")
        scores = detail["scores"]
        mirror_diagnostic = review_mirror(campaign, final, detail)
    ledger = campaign.base.ledger
    if final["target"] is None:
        # Review's original owner has already frozen the exact observations used
        # for this score. Do not first merge unrelated current projections: an
        # older map check may legitimately have both numeric and visual forms.
        observations = {
            p.id: next(
                o for o in parent.material.observations if o.reference.code == p.observation_code
            )
            for p in parent.policy.items
        }
        ledger = [
            dict(id=r.id, definition=r.definition, input_modalities=r.input_modalities)
            for r in parent.policy.requirements
        ]
    else:
        observations = acquired_observations(campaign)
    rows = [
        dict(
            id=r["id"],
            examination=r["definition"].examination,
            modality=list(r["input_modalities"]),
            status="已有结果"
            if r["id"] in observations and observations[r["id"]].status == "obtained"
            else "尚未取得可判读观察",
            observation=observations[r["id"]].model_dump(mode="json")
            if r["id"] in observations
            else None,
            sources=[s.model_dump(mode="json") for s in r["definition"].source_refs],
        )
        for r in ledger
    ]
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        multimodal_profile="multimodal-task-report/1.0.0",
        result=result,
        requirements=rows,
        acquired_count=sum(o.status == "obtained" for o in observations.values()),
        required_count=len(rows),
        scores=scores,
        original_review=detail,
        model_output=None,
        segmentation_qc=None,
        reasoning_scope="task_relative_source_bound_multimodal_research",
        missing_is_negative=False,
        calibrated_probability=False,
        clinical_diagnosis=False,
        views=campaign.base.views,
        task=result["task"],
        task_conclusion=dict(
            status="evidence_reviewed" if scores else "incomplete_technical_execution",
            target=result["task"]["target_code"],
            class_label=None,
            ordinal_grade=None,
            limitation="方向评分并非分类概率；未建立经验证的诊断/分级阈值。",
        ),
        completion=dict(
            report_saved=True,
            domain_accepted=final["accepted"] and final["target"] is None,
            all_obtained_reviewed=bool(
                scores and not scores.get("review_completion", {}).get("pending_ids", [])
            ),
            pending_review_ids=scores.get("review_completion", {}).get("pending_ids", [])
            if scores
            else list(observations),
            evidence_limited=True,
        ),
        counts=dict(
            requirements=len(rows),
            acquired=sum(o.status == "obtained" for o in observations.values()),
            reviewed=len(scores.get("reviews", [])) if scores else 0,
            directed=sum(r["judgment"]["value"] is not None for r in scores.get("reviews", []))
            if scores
            else 0,
        ),
    )
    if mirror_diagnostic:
        report["local_report_recovery"] = mirror_diagnostic
    if campaign.protocol.get("execution_profile") in {
        "relay-context-v1",
        "relay-snapshot-context-v1",
        "relay-root-question-v1",
    }:
        from .review_context import collect_campaign

        report["review_display_context"] = dict(
            contexts=collect_campaign(campaign.base, observations),
            execution_profile=campaign.protocol["execution_profile"],
            purpose="whole_case_current_owner_context_not_extra_votes",
        )
        report["review_scope"] = dict(
            kind="whole_case_uniform_configuration",
            whole_case_rescored=whole_case_rescored(final, scores),
            cross_provider_historical_scores_merged=False,
        )
    if campaign.protocol.get("material_context_review"):
        report["review_display_context"] = campaign.protocol["material_context_review"]
        report["review_scope"] = dict(
            kind="material_context_reassessment",
            review_ids=campaign.protocol["review_target_ids"],
            whole_case_rescored=False,
            baseline=campaign.protocol["tip"],
            explanation="补充输入/显示上下文后的新评阅；旧评分保留，非整例重评或跨配置合并。",
        )
    if campaign.protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.47.0",
        "multimodal-campaign-renewal/1.48.0",
        "multimodal-campaign-renewal/1.51.0",
        "multimodal-campaign-renewal/1.66.0",
        "multimodal-campaign-renewal/1.67.0",
        "multimodal-campaign-renewal/1.70.0",
    }:
        invalid_response = (
            campaign.protocol["schema_version"] == "multimodal-campaign-renewal/1.66.0"
        )
        proof = campaign.protocol[
            "confirmed_invalid_acquisition" if invalid_response else "unknown_acquisition"
        ]
        report["analytical_recovery"] = dict(
            schema_version=proof["schema_version"],
            original_attempts=proof["attempts"],
            new_phase_bindings=[
                dict(target=p["target"], run_id=p["run_id"], accepted=p["accepted"])
                for p in result["phases"]
                if p.get("target") in proof["selected"]
            ],
            old_failures_preserved=True,
            duplicate_billing_possible=True,
            historical_unknown_is_not_new_evidence=True,
        )
        if invalid_response:
            report["analytical_recovery"].pop("historical_unknown_is_not_new_evidence")
            report["analytical_recovery"]["confirmed_invalid_response_is_not_observation"] = True
    if campaign.protocol.get("review_clarification"):
        report["review_clarification"] = campaign.protocol["review_clarification"]
    if campaign.protocol.get("root_direction_review"):
        report["root_direction_review"] = campaign.protocol["root_direction_review"]
    if campaign.protocol.get("shape_contract_review"):
        report["review_scope"] = dict(
            kind="uniform_format_contract_reassessment",
            baseline=campaign.protocol["tip"],
            transition=campaign.protocol["shape_contract_review"],
            whole_case_rescored=whole_case_rescored(final, scores),
            old_scores_reused=False,
            observations_reused=True,
            explanation="新格式提示词下的统一文字评阅；保留旧评分，不跨配置拼接，未重跑取证。",
        )
    if (
        campaign.protocol.get("review_increment_only") or campaign.protocol.get("truncated_review")
    ) and result.get("execution_kind") != "local_accepted_review_aggregation_not_new_model_review":
        report["review_scope"] = pending_review_scope(
            campaign.protocol, campaign.base.review_target_ids
        )
        report["completion"]["all_obtained_reviewed"] = False
        report["completion"]["selected_increment_reviewed"] = bool(
            scores
            and set(campaign.base.review_target_ids) == {r["id"] for r in scores.get("reviews", [])}
        )
        report["task_conclusion"]["limitation"] = report["review_scope"]["explanation"]
    if (
        campaign.protocol.get("acquisition_increment_only")
        or campaign.protocol.get("review_acquired_only")
    ) and result.get("execution_kind") != "local_accepted_review_aggregation_not_new_model_review":
        report["review_scope"] = dict(
            kind="new_observation_review_increment",
            whole_case_rescored=False,
            baseline=campaign.protocol["tip"],
            review_ids=campaign.protocol["review_target_ids"],
            explanation="仅新取得观察进入本轮评阅；历史已接纳判读不重发，整例结果须经原owner显式汇总。",
        )
        report["completion"]["all_obtained_reviewed"] = False
        report["completion"]["selected_increment_reviewed"] = bool(
            scores and not scores.get("review_completion", {}).get("pending_ids", [])
        )
        report["task_conclusion"]["limitation"] += report["review_scope"]["explanation"]
        if campaign.protocol.get("review_acquired_only"):
            reviewed = {r["id"] for r in scores.get("reviews", [])} if scores else set()
            report["completion"]["pending_review_ids"] = [
                t for t in campaign.protocol["review_target_ids"] if t not in reviewed
            ]
            report["review_scope"]["historical_review_ids"] = campaign.protocol[
                "reuse_assessment_ids"
            ]
            report["review_scope"]["acquisition_stop"] = campaign.protocol["tip"]
    if result.get("execution_kind") == "local_accepted_review_aggregation_not_new_model_review":
        report["review_scope"] = dict(
            kind="accepted_local_judgments_reaggregated",
            whole_case_rescored=True,
            new_medical_review=False,
            new_model_requests=0,
            origins=result["origins"],
            explanation="复用兼容的已接纳逐项判读，由原评分公式重新汇总；不是新的模型综合评阅，缺项仍保留。",
        )
        report["task_conclusion"]["limitation"] += report["review_scope"]["explanation"]
        report["completion"]["all_obtained_reviewed"] = bool(
            scores and not scores.get("review_completion", {}).get("pending_ids", [])
        )
    if scores and "hypothesis_comparison" in scores:
        report["task_conclusion"].update(
            status="research_hypotheses_reviewed",
            hypothesis_comparison=scores["hypothesis_comparison"],
            evaluation_mapping=scores["evaluation_mapping"],
            limitation="四个命题分数不是概率；GAMMA仅提供无、早、中晚合并标签，不能验证四分类。",
        )
        report["counts"]["distinct_evidence_units"] = len(
            scores["root"]["summary"]["distinct_evidence_units"]
        )
    if (
        campaign.protocol.get("definition_increment_only")
        and result.get("execution_kind") != "local_accepted_review_aggregation_not_new_model_review"
    ):
        component_review = (
            campaign.protocol.get("definition_trial", {}).get("review_scope")
            == "definition_increment_and_confirmed_component"
        )
        report["review_scope"] = dict(
            kind=(
                "definition_increment_and_confirmed_component"
                if component_review
                else "definition_increment_only"
            ),
            review_ids=list(campaign.base.review_target_ids),
            baseline=campaign.protocol.get("tip")
            or dict(
                kind="fresh_independent_experiment",
                entry_checksum=campaign.protocol["definition_trial"]["entry_checksum"],
                original_case_rescored=False,
            ),
            whole_case_rescored=False,
            explanation=(
                "仅新定义增量与原owner确认组件的本轮评阅；非整例评分，不改原结果。"
                if component_review
                else "仅新增或改变定义的项目；增量研究分不是整例重新汇总。原病例结果保留。"
            ),
        )
        report["completion"]["all_obtained_reviewed"] = False
        report["task_conclusion"]["limitation"] = report["review_scope"]["explanation"]
        report["definition_gate_positive"] = bool(
            campaign.base.definition_revision["added"]
            and final["accepted"]
            and any(
                p["target"] in campaign.base.definition_revision["added"] and p["accepted"]
                for p in result["phases"]
            )
        )
        if not campaign.base.review_target_ids:
            report["task_conclusion"]["status"] = "no_effective_increment"
        scoped_rows = [r for r in campaign.base.ledger if "dependency_assessment" in r]
        if scoped_rows:
            # An acquired-only increment is useful but is not the B directional gate.
            reviewed = {r["id"] for r in scores.get("reviews", [])} if scores else set()
            permitted = {r["id"] for r in scoped_rows if r["review_allowed"]}
            report["definition_gate_positive"] = bool(
                report["definition_gate_positive"]
                and set(campaign.base.definition_revision["added"]) & reviewed & permitted
            )
            report["definition_dependency_scope"] = {
                r["id"]: dict(
                    assessment=r["dependency_assessment"],
                    acquisition_allowed=r["acquisition_allowed"],
                    review_allowed=r["review_allowed"],
                )
                for r in scoped_rows
            }
            if not campaign.base.review_target_ids and any(
                p["target"] in {r["id"] for r in scoped_rows} and p["accepted"]
                for p in result["phases"]
            ):
                report["task_conclusion"]["status"] = "observed_review_dependencies_unresolved"
    if getattr(campaign.base, "definition_revision", None):
        report["definition_revision"] = campaign.base.definition_revision
        if "review_source_scope" in campaign.base.definition_revision:
            report["review_source_scope"] = campaign.base.definition_revision["review_source_scope"]
        dependency_rows = {r["id"]: r for r in campaign.base.ledger if "dependency_assessment" in r}
        if dependency_rows:
            # Use already accepted original observations, never fabricate or re-score.
            actual = acquired_observations(campaign)
            report["acquisition_observations"] = {
                k: v.model_dump(mode="json") for k, v in actual.items()
            }
            for row in report["requirements"]:
                bound = dependency_rows.get(row["id"])
                if bound:
                    row["dependency_assessment"] = bound["dependency_assessment"]
                    if row["id"] in actual and row["observation"] is None:
                        row["unscored_observation"] = actual[row["id"]].model_dump(mode="json")
                        row["status"] = "已取得观察，判读依赖未满足"
            report["counts"]["acquired"] = sum(v.status == "obtained" for v in actual.values())
            report["acquired_count"] = report["counts"]["acquired"]
    from .observation_closure import saved_closure, saved_independent_failures

    closure = saved_closure(campaign, result["phases"])
    if closure:
        report["observation_closure"] = closure
    independent = saved_independent_failures(campaign, result["phases"])
    if independent:
        report["independent_observation_failures"] = independent
    if scores and scores.get("schema_version") == "case-research-scoring/11.0.0":
        report["counts"].update(scores["evidence_counts"])
    attach_retained_context(campaign, report, parent)
    if getattr(campaign, "report_projection", None):
        report["report_projection"] = campaign.report_projection
    if hasattr(campaign, "procedure_reuse"):
        report["procedure_reuse"] = campaign.procedure_reuse.report()
    if hasattr(campaign, "confirmed_observations"):
        report["confirmed_observations"] = campaign.confirmed_observations.report()
    if hasattr(campaign, "continuation"):
        report["acquisition_continuation"] = campaign.continuation.report()
    if hasattr(campaign, "remaining_review"):
        report["remaining_review"] = campaign.remaining_review.report()
    if getattr(campaign.base, "detailed", False):
        from .correspondence import describe

        report["cross_modal_correspondence"] = describe(campaign.base.manifest)
        report["inspection_profile"] = "detail-v1"
    if (campaign.output / "local-search-repair.json").exists():
        from .search_recovery import read_amendment

        report["implementation_amendment"] = read_amendment(campaign.output)
    if final["target"] is None:
        report = produce(
            report, parent, artifacts=(campaign.base.root, *native_artifacts(campaign.base))
        )
        present_retained_context(report)
        scoped = {r["id"]: r for r in campaign.base.ledger if "dependency_assessment" in r}
        for row in report["requirements"]:
            if row["id"] not in scoped:
                continue
            row["dependency_presentation_version"] = "1.0.0"
            row["presentation_sections"]["reference"]["dependency_assessment"] = scoped[row["id"]][
                "dependency_assessment"
            ]
            actual = report.get("acquisition_observations", {}).get(row["id"])
            if actual and row.get("unscored_observation") == actual:
                # These bytes were read above through accepted_observations, not
                # inferred by the renderer or copied from an unconfirmed response.
                binding = row["execution_binding"]
                binding.update(
                    method=actual["method"],
                    observation_reference=actual["reference"],
                    unscored=True,
                )
                if actual.get("image"):
                    binding["view"] = actual["image"]
                row["presentation_sections"]["observation"] = actual
                row["presentation_sections"]["acquisition"].update(
                    method=actual["method"],
                    method_status="observed_interpretation_pending",
                    missing_reason="definition_review_dependency_unresolved",
                )
                row["presentation_sections"]["dependencies"] = binding
        if getattr(campaign.base, "component_observations", None):
            installed = campaign.base.component_observations
            installed.check()
            report["component_observations"] = installed.report()
            for row in report["requirements"]:
                if row["id"] == installed.target:
                    row["execution_binding"]["confirmed_component"] = report[
                        "component_observations"
                    ]["origin"]
                    row["presentation_sections"]["dependencies"] = row["execution_binding"]
        if getattr(campaign.base, "active_owner", None):
            from .active_acquisition import structure

            report = structure(report, campaign.base)
            budget_stop = campaign.output / "acquisition-budget-stop.json"
            if budget_stop.exists():
                report["acquisition_budget_stop"] = json.loads(budget_stop.read_bytes())
            report["acquisition_observations"] = {
                k: v.model_dump(mode="json") for k, v in acquired_observations(campaign).items()
            }
            for row in report["requirements"]:
                # Missing observations remain visible, but are not silently given e=0.
                if row["observation"] is None and row["id"] in report["acquisition_observations"]:
                    row["unscored_observation"] = report["acquisition_observations"][row["id"]]
        from .alternative_acquisition import report_binding
        from .demand_coverage import project, render_summary
        from .input_scope import attach_report

        attach_report(campaign, report, result)
        report_binding(campaign, report)
        report["demand_coverage"] = project(report)
        rendered = render(report, input_checksum=digest(report), media=media(campaign, report))
        rendered = rendered.replace(
            "</body>", render_summary(report["demand_coverage"]) + "</body>"
        )
        if report.get("review_display_context"):
            context_html = (
                '<details id="review-display-context"><summary>'
                "本轮实际评阅的数值与显示上下文（不额外计票）</summary><pre>"
                + html.escape(
                    json.dumps(report["review_display_context"], ensure_ascii=False, indent=2)
                )
                + "</pre></details>"
            )
            rendered = rendered.replace("</body>", context_html + "</body>")
    else:
        rendered = (
            "<!doctype html><meta charset='utf-8'><h1>多模态运行：技术停止，非医学阴性</h1><pre>"
            + html.escape(json.dumps(report, ensure_ascii=False, indent=2))
            + "</pre>"
        )
    if report.get("analytical_recovery"):
        recovery_html = (
            '<details id="analytical-recovery"><summary>'
            "历史UNKNOWN与本轮新分析尝试（保留原记录，可能重复计费）</summary><pre>"
            + html.escape(json.dumps(report["analytical_recovery"], ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
        rendered = (
            rendered.replace("</body>", recovery_html + "</body>")
            if "</body>" in rendered
            else rendered + recovery_html
        )
    report = json.loads(canonical(report))
    report_key = getattr(campaign, "report_key", "dynamic-case-report")
    from .report_storage import save

    save(docs, report, rendered, key=report_key)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("Multimodal_" + getattr(campaign, "report_name", campaign.output.name))

    def recover():
        authorize_recovery(campaign, final)

    json_path = publish_json(path.with_suffix(".json"), report, recover=recover)
    page = publish_html(path.with_suffix(".html"), rendered, recover=recover)
    return dict(
        complete=result["complete"],
        requests=result["new_requests"],
        json=str(json_path),
        html=str(page),
        report_checksum=digest(report),
        task_clinically_validated=False,
    )


def attach_retained_context(campaign, report, parent):
    projected = getattr(campaign, "report_projection", None)
    from .alternative_acquisition import VERSIONS as alternative_versions

    if campaign.protocol.get("schema_version") in alternative_versions:
        if (
            projected
            and projected.get("schema_version") != "retained-context-report-projection/1.2.0"
        ):
            raise ValueError("report_retained_context_projection_version")
        attach_unscored_facts(campaign, report, parent, alternative=True)
        return
    if campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.52.0":
        if (
            projected
            and projected.get("schema_version") != "retained-context-report-projection/1.1.0"
        ):
            raise ValueError("report_retained_context_projection_version")
        attach_unscored_facts(campaign, report, parent)
        return
    method_projection = bool(
        projected
        and projected.get("schema_version") == "retained-context-report-projection/1.0.0"
        and campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.45.0"
    )
    if projected and not method_projection:
        raise ValueError("report_retained_context_projection_version")
    if not method_projection and campaign.protocol.get("schema_version") not in {
        "multimodal-campaign-renewal/1.37.0",
        "multimodal-campaign-renewal/1.38.0",
        "multimodal-campaign-renewal/1.43.0",
        "multimodal-campaign-renewal/1.46.0",
        "multimodal-campaign-renewal/1.47.0",
        "multimodal-campaign-renewal/1.48.0",
        "multimodal-campaign-renewal/1.51.0",
        "multimodal-campaign-renewal/1.66.0",
        "multimodal-campaign-renewal/1.67.0",
        "multimodal-campaign-renewal/1.70.0",
    }:
        return
    scope = (
        getattr(campaign.base, "review_context_scope", None)
        if method_projection
        else campaign.protocol.get("review_context_scope")
    )
    if scope != "current-owner-prior-visible/1.0.0":
        raise ValueError("report_retained_context_scope")
    scored_codes = {p.observation_code for p in parent.policy.items}
    saved_context = {
        o.reference.code: o
        for o in parent.material.observations
        if o.reference.code not in scored_codes
    }
    rows = {r["id"]: r for r in report["requirements"]}
    retained = {}
    for target, observation in acquired_observations(campaign).items():
        code = observation.reference.code
        if code not in saved_context:
            continue
        if observation != saved_context[code] or target not in rows:
            raise ValueError("report_retained_context_current_fact_changed")
        if observation.status != "obtained":
            continue
        if rows[target].get("observation") is not None:
            raise ValueError("report_retained_context_already_scored")
        retained[target] = observation.reference.model_dump(mode="json")
        rows[target]["unscored_observation"] = observation.model_dump(mode="json")
        rows[target]["status"] = "已取得背景观察，本轮不新增判读"
        rows[target]["unscored_reason"] = "retained_prior_visible_context_outside_selected_review"
    reviewed = {r["id"] for r in (report.get("scores") or {}).get("reviews", [])}
    selected = {p.id for p in parent.policy.items}
    reused = set(campaign.protocol.get("reuse_assessment_ids", []))
    report["retained_review_context"] = dict(
        schema_version="retained-review-context-report/1.0.0",
        observations=retained,
        review_ids=sorted(selected),
        reused_review_ids=sorted(reused),
        new_review_ids=sorted(selected - reused),
        extra_scoring_votes=0,
        semantics="current_owner_facts_retained_without_new_judgment",
    )
    observed_rows = {
        r["id"]
        for r in report["requirements"]
        if (r.get("observation") or r.get("unscored_observation") or {}).get("status") == "obtained"
    }
    report["counts"]["acquired"] = report["acquired_count"] = len(observed_rows)
    if (
        method_projection
        or campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.46.0"
    ):
        report["counts"]["obtained_observations"] = len(observed_rows)
    report["counts"]["retained_unscored_observations"] = len(retained)
    report["counts"]["review_scope_observations"] = len(parent.policy.items)
    report["completion"]["selected_increment_reviewed"] = bool(
        report.get("scores") and reviewed == selected
    )
    if retained:
        report["completion"]["all_obtained_reviewed"] = False
    report["completion"]["retained_context_not_newly_reviewed_ids"] = sorted(retained)
    report["task_conclusion"]["limitation"] += (
        " 本轮仅对选定项目新增或复用判读；已取得的保留背景不新增投票，"
        "既有局部判读不冒充新的综合评阅。"
    )


def attach_unscored_facts(campaign, report, parent, *, alternative=False):
    """Count confirmed facts outside this Review; never imply the model saw them."""
    actual = acquired_observations(campaign)
    projection = getattr(campaign, "report_projection", None)
    if projection and projection.get("acquisition_observations_checksum") != digest(
        {k: v.model_dump(mode="json") for k, v in actual.items()}
    ):
        raise ValueError("report_retained_context_current_fact_changed")
    selected = {p.id for p in parent.policy.items}
    reviewed = {r["id"] for r in (report.get("scores") or {}).get("reviews", [])}
    rows = {r["id"]: r for r in report["requirements"]}
    visible = {o.reference.code: o for o in parent.material.observations}
    quarantined = (
        set() if alternative else set(campaign.protocol["parent_inventory"]["quarantined"])
    )
    if alternative:
        # The composed score was rebuilt by the original owner above; preserve its
        # exact rows rather than calling a fresh Review or counting unscored facts.
        selected = reviewed
        if not selected <= actual.keys():
            raise ValueError("report_retained_context_selected_fact_missing")
        if projection and projection.get("scores_checksum") != digest(report.get("scores")):
            raise ValueError("report_retained_context_score_changed")
    retained, reasons, seen = {}, {}, {}
    for target, observation in actual.items():
        if observation.status != "obtained":
            continue
        if target not in rows:
            raise ValueError("report_retained_context_missing_target")
        if target in selected:
            if rows[target].get("observation") != observation.model_dump(mode="json"):
                raise ValueError("report_retained_context_selected_fact_changed")
            continue
        if rows[target].get("observation") is not None or target in reviewed:
            raise ValueError("report_retained_context_already_scored")
        old = visible.get(observation.reference.code)
        if old is not None and old != observation:
            raise ValueError("report_retained_context_current_fact_changed")
        retained[target] = observation.reference.model_dump(mode="json")
        seen[target] = old is not None
        reasons[target] = (
            "historical_unknown_review_quarantined"
            if target in quarantined
            else "confirmed_observation_outside_selected_review"
        )
        rows[target].update(
            unscored_observation=observation.model_dump(mode="json"),
            status="已取得观察，历史评阅UNKNOWN隔离"
            if target in quarantined
            else "已取得观察，本轮未评阅",
            unscored_reason=reasons[target],
        )
    report["retained_review_context"] = dict(
        schema_version="retained-review-context-report/1.2.0"
        if alternative
        else "retained-review-context-report/1.1.0",
        observations=retained,
        reasons=reasons,
        visible_in_current_review=seen,
        review_ids=sorted(selected),
        reused_review_ids=sorted(selected & set(campaign.protocol.get("reuse_assessment_ids", [])))
        if alternative
        else [],
        new_review_ids=sorted(selected - set(campaign.protocol.get("reuse_assessment_ids", [])))
        if alternative
        else sorted(selected),
        extra_scoring_votes=0,
        semantics="confirmed_facts_not_additional_judgments",
    )
    obtained = sum(o.status == "obtained" for o in actual.values())
    report["counts"].update(
        acquired=obtained,
        obtained_observations=obtained,
        retained_unscored_observations=len(retained),
        review_scope_observations=len(selected),
    )
    report["acquired_count"] = obtained
    report["completion"].update(
        all_obtained_reviewed=bool(report.get("scores") and reviewed == selected and not retained),
        selected_increment_reviewed=bool(report.get("scores") and reviewed == selected),
        retained_context_not_newly_reviewed_ids=sorted(retained),
    )
    report["review_scope"] = dict(
        kind="alternative_method_retained_facts"
        if alternative
        else "unscored_parent_unattempted_acquisition_review",
        whole_case_rescored=False,
        review_ids=sorted(selected),
        historical_unknown_review_ids=sorted(quarantined),
        unreviewed_observation_ids=sorted(retained),
        explanation="替代取证的已保存观察与原接纳判读分别保留；未评阅事实不新增投票，不把局部完成称为全部已取得观察均已评阅。"
        if alternative
        else "本轮仅评阅已确认的新观察；原UNKNOWN评阅保持隔离，已有事实不丢失、不额外计票。",
    )
    report["task_conclusion"]["limitation"] += report["review_scope"]["explanation"]


def present_retained_context(report):
    """Bind saved background facts to the same six sections, never score them."""
    retained = report.get("retained_review_context", {}).get("observations", {})
    for row in report["requirements"]:
        if row["id"] not in retained:
            continue
        actual = row["unscored_observation"]
        if actual["reference"] != retained[row["id"]]:
            raise ValueError("report_retained_context_presentation_binding")
        binding = row["execution_binding"]
        binding.update(
            method=actual["method"],
            observation_reference=actual["reference"],
            unscored=True,
        )
        if actual.get("image"):
            binding["view"] = actual["image"]
        sections = row["presentation_sections"]
        sections["observation"] = actual
        sections["acquisition"].update(
            method=actual["method"],
            method_status=report["retained_review_context"]
            .get("reasons", {})
            .get(row["id"], "retained_observation_outside_selected_review"),
            missing_reason=None,
        )
        sections["dependencies"] = binding

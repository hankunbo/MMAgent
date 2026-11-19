"""Technical closure on original Runtime/report owners, including zero-phase runs.

No fake Control, plan revision, observation or medical score is created here.
"""

import html
import json
from pathlib import Path

from packages.grounding.models import canonical, digest
from packages.llm_provider.provider import ProviderError

from . import report_storage
from .report_io import mirror_state, publish_html, publish_json

KEY = "multimodal-stage-failure"
REPORT_KEY = "multimodal-stage-failure-report"


def check_definition_binding(campaign, docs):
    """Check actual original definition records; never invent a phase document."""
    from .definition_expansion import read_expansion_policy

    phase = docs.read("dynamic-campaign")
    if phase is not None:
        if phase != (campaign.protocol,):
            raise ValueError("stage_phase_binding")
        return
    initialization = docs.read("definition-trial-initialization")
    if initialization is not None:
        if initialization != (campaign.protocol,):
            raise ValueError("stage_phase_binding")
        return
    expected = dict(
        campaign_id=getattr(campaign, "logical_campaign_id", campaign.campaign_id),
        input_checksum=campaign.base.manifest.checksum,
        target=campaign.base.task.target_code,
        protocol_checksum=digest(campaign.protocol),
        parent_requirements_checksum=digest(
            list(
                getattr(
                    getattr(campaign, "definition_expansion", None),
                    "original_rows",
                    campaign.base.ledger,
                )
            )
        ),
        policy_checksum=digest(read_expansion_policy(campaign.base.definition_expansion_policy)),
    )
    records = [
        docs.read(key)
        for key in (
            "definition:stage:sources",
            "definition:stage:definitions",
            "definition:revision",
        )
    ]
    bindings = [record[0].get("binding", {}) for record in records if len(record or ()) == 1]
    if not any(record is not None for record in records):
        # A process can disappear before the first stage checkpoint. Immutable
        # per-operation claims still prove historical work, not current-use or
        # replay permission. Keep this recovery confined to the current window.
        check_interrupted_definition_binding(campaign, docs, expected)
        return
    if (
        not bindings
        or any(len(record) != 1 for record in records if record)
        or any(
            binding != bindings[0]
            or any(binding.get(key) != value for key, value in expected.items())
            for binding in bindings
        )
    ):
        raise ValueError("stage_phase_binding")


def check_interrupted_definition_binding(campaign, docs, expected):
    """Exact original run/plan and contiguous operation history, for closure only."""
    from db.runtime.repository.case_pipeline import identity

    from .definition_expansion import read_expansion_policy

    if campaign.protocol.get("schema_version") not in {
        "multimodal-campaign-renewal/1.56.0",
        "multimodal-campaign-renewal/1.78.0",
        "multimodal-campaign-renewal/1.79.0",
        "multimodal-campaign-renewal/1.80.0",
        "multimodal-campaign-renewal/1.58.0",
        "multimodal-campaign-renewal/1.61.0",
        "multimodal-campaign-renewal/1.62.0",
        "multimodal-campaign-renewal/1.63.0",
    }:
        raise ValueError("stage_phase_binding")
    rid = identity("RUN", "task-research:" + campaign.request_key("definition-expansion"))
    if docs.run_id != rid:
        raise ValueError("stage_phase_binding")
    context = campaign.contexts.read_research_run(rid, missing_ok=True)
    if context is None:
        raise ValueError("stage_phase_binding")
    request = context.request
    if (
        context.run_id != rid
        or request.input_checksum != campaign.base.manifest.checksum
        or request.visible_checksum != campaign.base.visible_value.checksum
        or request.task != campaign.base.task
        or request.scope.case_code != campaign.base.binding["case_code"]
        or request.policy_ref.checksum != digest((campaign.base.ledger, campaign.config, None))
    ):
        raise ValueError("stage_phase_binding")
    policy = read_expansion_policy(campaign.base.definition_expansion_policy)
    bound = getattr(campaign, "definition_operation_limit", policy.max_operations)
    if type(bound) is not int or not 1 <= bound <= 100:
        raise ValueError("stage_phase_binding")
    # Read one beyond the original bound to reject overflow, holes and duplicates.
    keys = tuple(f"definition:operation:{i}" for i in range(bound + 1))
    values = docs.read_many(keys)
    present = [v for v in values if v is not None]
    if (
        not present
        or len(present) > bound
        or any(v is None for v in values[: len(present)])
        or any(len(v) != 1 or not isinstance(v[0], dict) for v in present)
    ):
        raise ValueError("stage_phase_binding")
    operations = [v[0] for v in present]
    operation_keys = [v.get("key") for v in operations]
    kinds = {
        "local_search",
        "search_group",
        "wire",
        "fetch_order",
        "fetch_group",
        "local_index_plan",
        "local_index_window",
        "local_pool_reconciliation",
        "llm",
    }
    if (
        any(v.get("binding") != expected for v in operations)
        or any(not isinstance(k, str) or not k.startswith("definition:") for k in operation_keys)
        or len(set(operation_keys)) != len(operation_keys)
        or any(v.get("kind") not in kinds for v in operations)
    ):
        raise ValueError("stage_phase_binding")


def historical_targets(campaign):
    """Finite identities from this campaign's original saved plan, not a reschedule."""
    from .stages import durable_receipt

    # Scope runs before medical acquisition and is deliberately absent from
    # campaign.targets. Its confirmed dispatch still belongs in failure history.
    targets = [*getattr(campaign, "scope_targets", ()), *campaign.targets]
    if not hasattr(campaign, "contexts") or not hasattr(campaign, "request_key"):
        return tuple(targets)
    requirements = durable_receipt(campaign, "requirements")
    acquisition = durable_receipt(campaign, "acquisition")
    if requirements:
        docs = campaign.documents(requirements["run_id"])
        check_definition_binding(campaign, docs)
        if docs.read("definition:revision") != (requirements["payload"],):
            raise ValueError("stage_phase_binding")
        targets.extend(r["id"] for r in requirements["payload"]["rows"])
    if acquisition:
        docs = campaign.documents(acquisition["run_id"])
        check_definition_binding(campaign, docs)
        targets.extend(p["target"] for p in acquisition["payload"]["phases"])
    if any(type(t) is not str or not t for t in targets):
        raise ValueError("stage_phase_binding")
    return tuple(dict.fromkeys(targets))


def read_saved_failure(campaign):
    """Only exact campaign-owned failure records can explain a truncated mirror."""
    from db.runtime.repository.case_pipeline import identity

    path = campaign.output / "stage-failure.json"
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        candidates = []
        # Finite original run identities, never a database scan of accepted objects.
        for target in (None, "definition-expansion", *historical_targets(campaign)):
            rid = identity("RUN", "task-research:" + campaign.request_key(target))
            if campaign.contexts.read_research_run(rid, missing_ok=True) is None:
                continue
            values = campaign.documents(rid).read(KEY)
            if not values or len(values) != 1:
                continue
            saved = values[0]
            if (
                saved.get("run_id") != rid
                or saved.get("campaign_id") != campaign.campaign_id
                or saved.get("protocol_checksum") != digest(campaign.protocol)
            ):
                continue
            payload = canonical(saved) + b"\n"
            if len(raw) < len(payload) and payload.startswith(raw):
                candidates.append(saved)
        if len(candidates) != 1:
            raise ValueError("stage_failure_mirror_unconfirmed") from None
        value = candidates[0]
    if not isinstance(value, dict) or value.get("protocol_checksum") != digest(campaign.protocol):
        raise ValueError("stage_failure_protocol_binding")
    if value.get("campaign_id") != campaign.campaign_id:
        raise ValueError("stage_failure_owner_binding")
    if value.get("run_id") and value["run_id"] not in {
        identity("RUN", "task-research:" + campaign.request_key(target))
        for target in (None, "definition-expansion", *historical_targets(campaign))
    }:
        raise ValueError("stage_failure_owner_binding")
    if value.get("run_id") and campaign.documents(value["run_id"]).read(KEY) != (value,):
        raise ValueError("stage_failure_owner_binding")
    mirror_state(path, canonical(value) + b"\n", kind="json")
    return value


def safe_error(error):
    from packages.knowledge_connectors.ports import ConnectorError

    if isinstance(error, ConnectorError):
        return "knowledge_" + error.category.value
    code = error.code if isinstance(error, ProviderError) else str(error)
    allowed = {
        "configuration",
        "scope",
        "transport",
        "timeout",
        "schema",
        "cancelled",
        "deadline",
        "budget",
        "definition_draft_exact_excerpt",
        "definition_quote_ambiguous",
        "definition_quote_not_found",
        "definition_revision_source_changed",
        "definition_pending_unknown_no_replay",
        "definition_public_request_budget",
        "definition_operation_budget",
        "definition_review_reserve",
        "definition_close_reserve",
        "stage_previous_failure_read_only",
        "stage_precondition",
        "agent_idempotency_conflict",
        "execution_process_interrupted",
    }
    return code if code in allowed else "local_owner_error"


def unstarted_phase(campaign, target, rid, docs):
    """Registered task with no phase initialization is not a failed domain judgment."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    ctx = campaign.contexts.read_research_run(rid)
    req = ctx.request
    if (
        req.input_checksum != campaign.protocol["input"]
        or req.visible_checksum != campaign.protocol["visible"]
        or req.task.checksum != campaign.protocol["task"]
        or req.scope.case_code != campaign.base.binding["case_code"]
        or req.policy_ref.checksum != digest((campaign.base.ledger, campaign.config, target))
    ):
        raise ValueError("stage_phase_binding")
    keys = (
        "dynamic-campaign",
        "dynamic-phase-input",
        "dynamic-phase-result",
        "live-stage-intent:review",
        "live-stage:review",
        "live-stage-intent:vision",
        "live-stage:vision",
        "multimodal-case-result",
    )
    return (
        campaign.sessions.read(rid) is None
        and not RuntimeAgentJournal(docs).read()
        and not any(docs.read(key) for key in keys)
        and not (campaign.phase_path(target) / "request.json").exists()
    )


def saved_progress(campaign, stage):
    """Read original phase/acceptance owners, not an inferred domain outcome."""
    from db.runtime.repository.case_pipeline import identity
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .stages import confirmed_phase

    phases, controls = [], []
    phase_reservations = 0
    active_run = None
    for target in (*historical_targets(campaign), None):
        rid = identity("RUN", "task-research:" + campaign.request_key(target))
        if campaign.contexts.read_research_run(rid, missing_ok=True) is None:
            continue
        phase_docs = campaign.documents(rid)
        if phase_docs.read("dynamic-campaign") != (campaign.protocol,):
            if unstarted_phase(campaign, target, rid, phase_docs):
                controls.append(
                    dict(
                        run_id=rid,
                        target=target,
                        exists=False,
                        phase="registered_unstarted",
                        pending=False,
                    )
                )
                continue
            raise ValueError("stage_phase_binding")
        phase_reservations += sum(
            event.get("phase") == "reserved" for event in RuntimeAgentJournal(phase_docs).read()
        )
        control = campaign.sessions.read(rid)
        controls.append(
            dict(
                run_id=rid,
                target=target,
                exists=control is not None,
                phase=control.phase if control else None,
                pending=control.pending is not None if control else None,
            )
        )
        active_run = rid
        phase = confirmed_phase(campaign, target)
        if phase:
            phases.append(phase)
    review = next((p for p in phases if p["target"] is None), None)
    original_review = review
    from .stages import review_for_export

    review = review_for_export(campaign)
    accepted = None
    if review and review["accepted"]:
        from .stages import acceptance_payload

        acceptance_payload(campaign, review)
        accepted = campaign.documents(review["run_id"]).read("accepted:" + review["acceptance_key"])
    docs = campaign.documents(active_run) if active_run else None
    case_result = docs.read("multimodal-case-result") if docs else None
    if case_result and (not original_review or case_result[0]["phases"][-1] != original_review):
        raise ValueError("stage_phase_binding")
    return dict(
        schema_version="multimodal-stage-failure/1.2.0"
        if review and review.get("closure")
        else "multimodal-stage-failure/1.1.0",
        run_id=active_run,
        confirmed_phases=phases,
        phase_controls=controls,
        phase_reservations=phase_reservations,
        control=controls[-1] if controls else dict(exists=False, phase=None, pending=None),
        original_review=original_review,
        committed_review=review.get("closure") if review else None,
        accepted_reference=dict(
            run_id=review["run_id"],
            key="accepted:" + review["acceptance_key"],
            checksum=digest(accepted),
        )
        if accepted
        else None,
        scores=accepted[2]["scores"] if accepted else None,
        score_status="saved" if accepted else "not_scored",
        case_result=case_result[0] if case_result else None,
        report_saved=bool(
            docs and docs.read(getattr(campaign, "report_key", "dynamic-case-report"))
        ),
        display_completed=False,
        failure_kind="local_post_score_failure" if accepted else "local_stage_failure",
        next_action="explicit_zero_request_stage_export"
        if review and (review["accepted"] or review["evidence_limited"])
        else "read_only_owner_confirmation_not_replay",
        not_executed=([] if review else ["review"]) + ([] if accepted else ["acceptance_scoring"]),
        failure_stage=stage,
    )


def frozen_component_summary(campaign):
    """Historical confirmation, never current consumption permission or a score.

    Failure closure must work after cancellation. Keep the exact campaign-frozen
    origin and measurements rather than running the tool/current permission
    checks again. Subsequent consumption must still recheck the original owner.
    """
    component = campaign.protocol.get("component_observations")
    if component is None:
        return None
    return dict(
        protocol_checksum=digest(campaign.protocol),
        historical_confirmation=component,
        current_qualification="not_rechecked_technical_closure_only",
        medical_direction=None,
        new_score=False,
    )


def saved_definition_state(campaign):
    """Read the exact definition run without installing providers or new budget.

    A cold campaign has no in-memory expansion. Its absence cannot erase durable
    requests. Reuse original RequestBudget readers over an immutable journal
    snapshot; construction is forbidden from creating a missing journal.
    """
    from contextlib import nullcontext
    from datetime import datetime

    from db.runtime.repository.case_pipeline import identity
    from packages.llm_provider.provider import RequestBudget
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    if not hasattr(campaign, "contexts") or not hasattr(campaign, "request_key"):
        return None, None, ()
    rid = identity("RUN", "task-research:" + campaign.request_key("definition-expansion"))
    if campaign.contexts.read_research_run(rid, missing_ok=True) is None:
        return None, None, ()
    docs = campaign.documents(rid)
    check_definition_binding(campaign, docs)
    events = RuntimeAgentJournal(docs).read()

    class SavedJournal:
        def read(self):
            return events

        def session(self):
            return nullcontext()

        def validate_binding(self, run_id, deadline):
            if run_id != rid or deadline.isoformat() != campaign.protocol["deadline"]:
                raise ValueError("stage_phase_binding")

        def append(self, event):
            raise ValueError("stage_missing_budget_read_only")

    budget = (
        RequestBudget(
            run_id=rid,
            deadline=datetime.fromisoformat(campaign.protocol["deadline"]),
            max_requests=campaign.config.max_requests,
            journal=SavedJournal(),
        )
        if events
        else None
    )
    from .definition_expansion import read_expansion_policy

    policy = read_expansion_policy(campaign.base.definition_expansion_policy)
    bound = getattr(campaign, "definition_operation_limit", policy.max_operations)
    if type(bound) is not int or not 1 <= bound <= 100:
        raise ValueError("stage_definition_operation_bound")
    keys = tuple(f"definition:operation:{n}" for n in range(bound))
    values = (
        docs.read_many(keys)
        if type(docs) is RuntimeDocuments
        else tuple(docs.read(k) for k in keys)
    )
    return docs, budget, tuple(v[0] for v in values if v)


def snapshot(campaign, error, stage):
    expansion = getattr(campaign, "definition_expansion", None)
    docs = (
        expansion.documents
        if expansion
        else campaign.documents(campaign.definition_run_id)
        if getattr(campaign, "definition_run_id", None)
        else None
    )
    recovered = False
    if expansion:
        budget, operations = expansion.budget, expansion.operations()
    else:
        docs, budget, operations = saved_definition_state(campaign)
        recovered = docs is not None
    records = budget.records() if budget else ()
    transport = budget.transport_counts() if budget else None
    diagnostics = [
        v[0] for r in records if (v := docs.read("definition:diagnostic:" + r.request_id))
    ]
    from packages.knowledge_connectors.ports import ConnectorError

    if isinstance(error, ConnectorError):
        diagnostics.append(
            dict(
                owner="knowledge_connector",
                category=error.category.value,
                last_error=error.last_error.value if error.last_error else None,
                metadata=error.metadata.model_dump(mode="json") if error.metadata else None,
                http_status=error.metadata.attempts[-1].http_status
                if error.metadata and error.metadata.attempts
                else None,
            )
        )
    operation_refs = [
        dict(
            key=o["key"],
            kind=o["kind"],
            confirmed=bool(
                docs.read(o["key"] + ":outcome")
                if o["kind"] == "wire"
                else docs.read(o["key"] + ":result")
            ),
        )
        for o in operations
    ]
    unknown = bool(
        transport and (transport["dispatch_unknown"] or transport["remote_response_unknown"])
    )
    # LLM operation without a confirmed proposal is not independently replayable.
    unknown = unknown or any(
        not o["confirmed"] for o in operation_refs if o["kind"] in {"wire", "search", "fetch"}
    )
    control = campaign.sessions.read(docs.run_id) if docs else None
    value = dict(
        schema_version="multimodal-stage-failure/1.0.0",
        campaign_id=campaign.campaign_id,
        input_checksum=campaign.base.manifest.checksum,
        target=campaign.base.task.target_code,
        protocol_checksum=digest(campaign.protocol),
        run_id=docs.run_id if docs else None,
        stage=stage,
        error_code=safe_error(error),
        failure_kind="dispatched_unknown"
        if unknown
        else "known_http_failure"
        if any((d.get("http_status") or 0) >= 400 for d in diagnostics)
        else "knowledge_connector_failure"
        if isinstance(error, ConnectorError)
        else "proven_not_dispatched"
        if records and all(r.api_requests == 0 and not r.simulated for r in records)
        else "confirmed_response_or_domain_refusal"
        if records
        else "no_provider_reservation"
        if not transport or not transport["reserved_attempts"]
        else "commit_or_dispatch_confirmation_unknown",
        control=dict(
            exists=control is not None,
            phase=control.phase if control else None,
            pending=control.pending is not None if control else None,
        ),
        cancel_marker=(campaign.output / "cancel").exists(),
        transport=transport,
        requests=[
            dict(
                request_id=r.request_id,
                error=r.error,
                api_requests=r.api_requests,
                simulated=r.simulated,
            )
            for r in records
        ],
        diagnostics=diagnostics,
        operations=operation_refs,
        definition_revision_saved=bool(docs and docs.read("definition:revision")),
        confirmed_proposal_saved=bool(docs and docs.read("definition:confirmed-proposal")),
        previous_attempts=campaign.protocol.get("previous_attempts", []),
        deadline=campaign.protocol["deadline"],
        max_new_requests=campaign.protocol["max_new_requests"],
        new_requests=campaign.usage()
        + (
            (budget.used if budget else 0)
            + sum(o["kind"] in {"wire", "search", "fetch"} for o in operations)
            if recovered
            else 0
        ),
        complete=False,
        scores=None,
        original_review=None,
        clinical_acceptance=False,
        live_domain_acceptance_not_enabled=True,
        automatic_replay_authorized=False,
        next_action="read_only_owner_confirmation_or_explicit_new_plan_not_replay",
        not_executed=["acquisition", "review", "acceptance_scoring"]
        if not any(
            p.exists()
            for p in (campaign.phase_path(t) / "result.json" for t in campaign.targets + (None,))
        )
        else [],
    )
    if stage in {"acquisition", "review", "acceptance", "export"}:
        # Source diagnostics retain their own owner; they are not the Review's
        # Control or an assertion that a later request was never dispatched.
        value["definition_progress"] = {
            k: value[k] for k in ("run_id", "control", "transport", "requests", "operations")
        }
        value.update(saved_progress(campaign, stage))
        if recovered:
            value["new_requests"] = (
                value["phase_reservations"]
                + (budget.used if budget else 0)
                + sum(o["kind"] in {"wire", "search", "fetch"} for o in operations)
            )
        value["requests"] = [
            dict(run_id=p["run_id"], requests=p["requests"], transport=p["transport"])
            for p in value["confirmed_phases"]
        ]
        value["transport"] = None  # Per-phase counts above, not Source counts.
    component = frozen_component_summary(campaign)
    if component is not None:
        value["schema_version"] = "multimodal-stage-failure/1.3.0"
        value["confirmed_component_snapshot"] = component
    if recovered:
        value["schema_version"] = "multimodal-stage-failure/1.4.0"
        value["definition_history_reconstructed"] = True
    return value


def render(value):
    title = "评分已保存、展示未完成" if value.get("scores") is not None else "本轮未评分"
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>阶段技术收尾</title>'
        f"<body><h1>阶段技术收尾：{title}</h1><p>此页只展示已保存事实。"
        "无观察阶段不创建医学分数；UNKNOWN 不得重发。"
        "Control 不存在不等于 Planner 已停止。</p><pre>"
        + html.escape(json.dumps(value, ensure_ascii=False, indent=2))
        + "</pre></body></html>"
    )


def export_saved(campaign, *, directory=None):
    value = read_saved_failure(campaign)
    if value["protocol_checksum"] != digest(campaign.protocol):
        raise ValueError("stage_failure_protocol_binding")
    if value["run_id"]:
        docs = campaign.documents(value["run_id"])
        if docs.read(KEY) != (value,):
            raise ValueError("stage_failure_owner_binding")
        saved = docs.read(REPORT_KEY)
        if saved:
            report, envelope = report_storage.unpack(saved, key=REPORT_KEY)
            if report != value:
                raise ValueError("stage_failure_report_binding")
            page = envelope["html"]
        else:
            page = render(value)
            report_storage.save(docs, value, page, key=REPORT_KEY)
    else:
        page = render(value)
    out = Path(directory) if directory else campaign.output / "technical-report"
    out.mkdir(parents=True, exist_ok=True)

    def current():
        if read_saved_failure(campaign) != value:
            raise ValueError("stage_failure_owner_binding")

    json_path = publish_json(out / "report.json", value, recover=current)
    path = publish_html(out / "report.html", page, recover=current)
    return dict(
        complete=False,
        json=str(json_path),
        html=str(path),
        report_checksum=digest(value),
        requests=value["new_requests"],
        score_computed=value["scores"] is not None,
    )


def export_local_repair_failure(campaign):
    path = campaign.output / "local-repair-failure.json"
    value = json.loads(path.read_bytes())
    docs = campaign.documents(value["run_id"])
    key = "multimodal-local-repair-failure"
    if value["repair"] != digest(campaign.local_material_repair) or docs.read(key) != (value,):
        raise ValueError("local_repair_failure_binding")
    saved = docs.read(key + ":report")
    if saved:
        report, envelope = report_storage.unpack(saved, key=key + ":report")
        if report != value:
            raise ValueError("local_repair_report_binding")
        page = envelope["html"]
    else:
        page = render(value)
        report_storage.save(docs, value, page, key=key + ":report")
    out = campaign.output / "local-repair-report"
    out.mkdir(exist_ok=True)
    publish_json(out / "report.json", value)
    publish_html(out / "report.html", page)
    return dict(
        complete=False,
        requests=value["new_requests"],
        json=str(out / "report.json"),
        html=str(out / "report.html"),
        report_checksum=digest(value),
        score_computed=value["scores"] is not None,
    )


def close(campaign, error, *, stage="definitions"):
    """Preserve primary and secondary failures independently; never dispatch here."""
    path = campaign.output / "stage-failure.json"
    if path.exists() and getattr(campaign, "local_material_repair", None):
        # A failed repair must not masquerade as the original zero-request failure.
        target = campaign.output / "local-repair-failure.json"
        if not target.exists():
            value = snapshot(campaign, error, stage)
            value["repair"] = digest(campaign.local_material_repair)
            value["original_failure"] = campaign.local_material_repair["original_failure_checksum"]
            docs = campaign.documents(value["run_id"])
            key = "multimodal-local-repair-failure"
            with docs.session():
                saved = docs.read(key)
                if saved:
                    value = saved[0]
                else:
                    docs.write(key, (value,))
            publish_json(target, value)
        return export_local_repair_failure(campaign)
    if path.exists():
        return export_saved(campaign)
    try:
        value = snapshot(campaign, error, stage)
        if value["run_id"]:
            docs = campaign.documents(value["run_id"])
            with docs.session():
                old = docs.read(KEY)
                if old:
                    value = old[0]  # Recover committed closure after acknowledgement loss.
                else:
                    docs.write(KEY, (value,))
            if docs.read(KEY) != (value,):
                raise ValueError("stage_failure_owner_binding")
        publish_json(path, value)
        return export_saved(campaign)
    except Exception as secondary:
        # No raw exception strings, and no claim that the database commit failed.
        fallback = dict(
            schema_version="multimodal-local-closure-error/1.0.0",
            primary=safe_error(error),
            secondary=safe_error(secondary),
            stage=stage,
            durable_commit="unconfirmed",
            automatic_replay_authorized=False,
            complete=False,
        )
        publish_json(campaign.output / "stage-closure-error.json", fallback)
        return fallback

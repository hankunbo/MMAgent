"""Read original case facts and reports only; never restart or install a candidate."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from agents.expert_agent.reporting import read_research_report, render_research_html
from agents.provider_agents.batched_review import read_progress
from apps.vectorengine_public_diagnostic import original_documents
from apps.vectorengine_round import (
    check_round_gate,
    evidence_for_round,
    load_round,
    read_deployment,
)
from apps.vectorengine_single_case import load_config, open_case
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_batches import (
    ReviewBatchPlan,
    ReviewBatchProposal,
    batch_payload,
    check_batch,
    read_submission,
    validate_batch_receipts,
)
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.config import prompt
from packages.llm_provider.http_recovery import observation_for
from packages.llm_provider.models import SourceProposal, VisionProposal
from packages.llm_provider.provider import RequestBudget, strict_json
from packages.llm_provider.transport import wire_request
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal

BASE = None


def save(name, value):
    with (BASE / name).open("xb") as f:
        f.write(canonical(value))


def main():
    import argparse

    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--round-config",
        default="configs/research/vectorengine_batch_integrated_case.json",
        choices=(
            "configs/research/vectorengine_batch_integrated_case.json",
            "configs/research/vectorengine_batch_integrated_r01_case.json",
            "configs/research/vectorengine_batch_integrated_r01_case_runtime_r01.json",
            "configs/research/vectorengine_active_acquisition_v1_case.json",
            "configs/research/vectorengine_active_acquisition_r01_case.json",
            "configs/research/vectorengine_active_acquisition_r02_case.json",
        ),
    )
    args = parser.parse_args()
    BASE = Path(args.output)
    BASE.mkdir(parents=True, exist_ok=False)
    r = load_round(args.round_config)
    c = load_config(r.provider_config)
    check_round_gate(r)
    original, target = open_case()
    evidence = evidence_for_round(original, r)
    if r.schema_version in {
        "bounded-live-round/9.0.0",
        "bounded-live-round/10.0.0",
        "bounded-live-round/11.0.0",
    }:
        prepared, _ = evidence.project(run_id=r.run_id, config=c)
        check_round_gate(r, prepared)
    deployment = read_deployment(evidence, target, r.run_id)
    if not deployment["target_run_exists"] or deployment["target_control_phase"] is None:
        save(
            "readback.json",
            dict(
                deployment=deployment,
                complete=False,
                reason="run_or_control_not_created",
                no_reexecution=True,
            ),
        )
        print(json.dumps(dict(complete=False, reason="run_or_control_not_created")))
        return
    docs = original_documents(evidence, target, r.run_id)
    context = docs.contexts.read_research_run(r.run_id)
    control = docs.sessions.read(r.run_id)
    assert control.phase == "stopped" and control.pending is None, "execution_not_terminal"
    journal = RuntimeAgentJournal(docs)
    events = journal.read()
    assert events and events[0]["schema_version"] == "agent-budget/1.0.0"
    budget = RequestBudget(
        run_id=r.run_id, deadline=control.binding.deadline, max_requests=40, journal=journal
    )
    records = budget.records()
    acceptance = ResearchAcceptance(owner=None, documents=docs, requests=budget)
    values = {}

    def read(key):
        value = docs.read(key)
        values[key] = value
        return value

    grant = read("research-live-grant")
    stages = {role: read("live-stage:" + role) for role in ("source", "vision", "review")}
    intents = {role: read("live-stage-intent:" + role) for role in stages}
    parents = {}
    progress = None
    reports = {}
    candidate_checks = {}
    for role, intent in intents.items():
        if not intent:
            continue
        event = intent[0]["event"]
        prefix = event["code"]
        body = read("review:" + prefix + ":intent")
        if body is None:
            continue
        parent = validate(DomainState, body[0]["state"])
        parents[role] = parent
        for suffix in (
            "selected",
            "decision",
            "result",
            "report",
            "html",
            "successor",
            "install-attempt",
            "installed",
        ):
            read("review:" + prefix + ":" + suffix)
        candidate = read(
            "review:" + prefix + ":" + ("A" if role == "review" else role) + ":candidate"
        )
        accepted_key = prefix + ":domain"
        read("submission:" + accepted_key)
        accepted = read("accepted:" + accepted_key)
        if role == "review":
            progress = read_progress(docs, prefix + ":A")
            if candidate:
                sub = read_submission(candidate[0])
                checked = validate_batch_receipts(parent, sub, docs, budget)
                candidate_checks[role] = dict(
                    validated_original_receipts=len(checked),
                    complete_ids=len(sub.candidate.items),
                    plan_checksum=digest(sub.plan),
                )
        if accepted and len(accepted) == 4:
            report = read_research_report(acceptance, key=accepted_key, parent=parent)
            html = render_research_html(report)
            exported = Path(r.output) / (role + "-report.json")
            exported_html = Path(r.output) / (role + "-report.html")
            assert exported.exists() and json.loads(exported.read_text()) == report, (
                "report_export_missing_or_mismatch"
            )
            assert exported_html.exists() and exported_html.read_text() == html, (
                "html_export_missing_or_mismatch"
            )
            reports[role] = dict(
                acceptance_key=accepted_key,
                report_checksum=digest(report),
                receipt=report["receipt"],
                report_path=str(exported),
                html_path=str(exported_html),
                html_checksum=checksum(exported_html.read_bytes()),
                scores_version=report["scores"]["schema_version"] if report["scores"] else None,
                scores_recomputed_and_equal=report["scores"] is not None,
                graph=report["receipt"]["graph"],
            )
    installs = []
    for n in range(4):
        value = read("research-owner-install:" + str(n))
        if value:
            installs.append(value[0])
    plan = validate(ReviewBatchPlan, progress["plan"]) if progress else None
    request_rows = []
    for record in records:
        raw_response = read("live-response:" + record.request_id)
        observation = observation_for(journal, record.request_id)
        role = record.role
        meta = dict(
            request_id=record.request_id,
            record=record.model_dump(mode="json"),
            http_observation=observation,
            raw_response=raw_response,
            finish_reason=None,
            finish_reason_status="not_retained_by_legacy_role_record",
            wire_reconstructed=False,
        )
        if role in parents:
            parent = parents[role]
            payload = None
            batch = None
            if role == "review" and plan:
                matches = [
                    b
                    for b in plan.batches
                    if digest(batch_payload(parent, plan, b)) == record.input_checksum
                ]
                assert len(matches) == 1
                batch = matches[0]
                payload = batch_payload(parent, plan, batch)
                response = read(
                    "review-batches:" + plan.key + ":" + batch.id + ":attempt:" + record.request_id
                )
                if response:
                    meta.update(
                        finish_reason=response[0]["finish_reason"],
                        finish_reason_status="original_batch_response",
                    )
            elif role in {"source", "vision"}:
                payload = dict(role=role, material=parent.material.model_dump(mode="json"))
                if role == "vision" and r.schema_version in {
                    "bounded-live-round/9.0.0",
                    "bounded-live-round/10.0.0",
                    "bounded-live-round/11.0.0",
                }:
                    from packages.evidence_details.review_tasks import project_review_tasks

                    image_observation = next(o for o in parent.material.observations if o.image)
                    task = next(
                        t
                        for t in project_review_tasks(parent).tasks
                        if t.observation == image_observation.reference
                    )
                    payload.update(
                        schema_version="targeted-vision-input/1.0.0",
                        vision_task=task.model_dump(mode="json"),
                    )
            if payload:
                assert digest(payload) == record.input_checksum
                setting, text = prompt(c, role)
                images = ()
                if role == "vision":
                    binding = next(o.image for o in parent.material.observations if o.image)
                    images = ((binding, evidence.image(binding.reference)),)
                cls = {
                    "source": SourceProposal,
                    "vision": VisionProposal,
                    "review": ReviewBatchProposal,
                }[role]
                wire = wire_request(
                    c.transport,
                    setting,
                    text,
                    payload,
                    cls.model_json_schema(),
                    images,
                    scope_binding=digest(parent.material.scope),
                    wire_policy=c.wire_policy,
                    wire_mode=c.wire_mode,
                )
                from packages.llm_provider.models import RolePayload
                from packages.llm_provider.source_snippets import project_wire

                if role == "source":
                    wire = project_wire(wire, c, role, validate(RolePayload, payload))
                corrections = [
                    e
                    for e in events
                    if e.get("schema_version")
                    in {"review-schema-correction/1.0.0", "source-quote-correction/1.0.0"}
                    and e.get("invocation_id") == record.invocation_id
                ]
                if corrections and record.retry_index == 1:
                    from packages.llm_provider.review_correction import (
                        corrected_wire,
                        verify_correction,
                    )

                    verify_correction(record, budget)
                    wire = corrected_wire(wire, corrections[0]["failures"], role=role)
                if observation:
                    assert digest(wire) == observation["wire_checksum"]
                    assert observation["config_checksum"] == record.config_checksum == digest(c)
                meta.update(
                    wire_reconstructed=True,
                    wire_checksum=digest(wire),
                    wire_bytes=len(canonical(wire)),
                    payload_bytes=len(canonical(payload)),
                    image_count=len(images),
                    image_bytes=sum(len(raw) for _, raw in images),
                    batch_id=batch.id if batch else None,
                    assessment_ids=batch.assessment_ids if batch else None,
                )
                if raw_response:
                    value = raw_response[0]
                    assert checksum(value["raw_model_text"].encode()) == value["raw_checksum"]
                    try:
                        if role == "review":
                            from packages.evidence_details.review_batches import (
                                read_response,
                                response_candidate,
                            )
                            from packages.llm_provider.review_response import decode

                            if not record.error:
                                candidate = response_candidate(
                                    read_response(response[0]),
                                    config_checksum=digest(c),
                                    prompt_checksum=record.prompt_checksum,
                                )
                            else:
                                candidate, _ = decode(value["raw_model_text"], c)
                        else:
                            candidate = validate(cls, strict_json(value["raw_model_text"]))
                        meta["schema_valid"] = True
                        if not record.error:
                            assert digest(candidate) == record.output_checksum
                        if role == "review":
                            check_batch(parent, plan, batch, candidate)
                            meta["original_batch_owner_passed"] = True
                    except ValueError as error:
                        meta["validation_error_type"] = type(error).__name__
                        meta["original_batch_owner_passed"] = False
        request_rows.append(meta)
    public = load_round(r.public_config)
    public_docs = original_documents(evidence, target, public.run_id)
    public_control = public_docs.sessions.read(public.run_id)
    public_events = RuntimeAgentJournal(public_docs).read()
    from apps.vectorengine_integrated_gate import historical_usage
    from packages.llm_provider.public_batches import check_public_batches

    public_budget = RequestBudget(
        run_id=public.run_id,
        deadline=public_control.binding.deadline,
        max_requests=40,
        journal=RuntimeAgentJournal(public_docs),
    )
    check_public_batches(
        c,
        public_budget,
        public_docs,
        public_docs.read("public-qualification")[0],
        digest(public),
        require_live=True,
    )
    history = (
        ()
        if r.schema_version == "bounded-live-round/9.0.0"
        else historical_usage(r, evidence, target)
    )
    if r.schema_version in {
        "bounded-live-round/9.0.0",
        "bounded-live-round/10.0.0",
        "bounded-live-round/11.0.0",
    }:
        source_binding = read("active-acquisition:source-binding")
        assert source_binding == [evidence.manifest] or source_binding == (evidence.manifest,)
        if r.schema_version == "bounded-live-round/11.0.0":
            aliases = read("active-acquisition:source-aliases")
            assert list(aliases) == [evidence.alias_manifest]
        for stage in (
            "before-source",
            "after-source",
            "before-vision",
            "after-vision",
            "before-review",
            "after-review",
        ):
            read("active-acquisition:" + stage)
        if "review" in reports:
            final_report = json.loads(Path(reports["review"]["report_path"]).read_text())
            evidence.verify(
                validate(
                    DomainState, prepared.model_copy(update={"material": final_report["material"]})
                )
            )
    save("historical-accounting.json", history)
    assert public_docs.sessions.read(public.run_id) == public_control
    assert RuntimeAgentJournal(public_docs).read() == public_events
    assert docs.sessions.read(r.run_id) == control and journal.read() == events, (
        "readback_mutated_runtime"
    )
    save("original-case-documents.json", values)
    save("original-case-journal.json", events)
    save("original-case-control.json", control)
    save("actual-requests.json", request_rows)
    save("batch-progress.json", progress)
    save("reports-readback.json", reports)
    complete = bool(
        stages["review"]
        and stages["review"][0].get("status") == "accepted"
        and "review" in reports
        and len(installs) == 3
    )
    result = dict(
        checked_at=datetime.now(UTC).isoformat(),
        run_id=r.run_id,
        synthetic=context.synthetic,
        context_checksum=digest(context),
        phase=control.phase,
        pending=control.pending,
        steps=control.steps,
        cancelled=control.cancelled,
        started_at=control.binding.started_at,
        deadline=control.binding.deadline,
        control_checksum=digest(control),
        journal_checksum=digest(events),
        stages={k: v[0] if v else None for k, v in stages.items()},
        requests=budget.summary(),
        transport_counts=budget.transport_counts(),
        public_requests=public_budget.used,
        historical_requests=sum(p.used for p in history),
        session_requests=public_budget.used + sum(p.used for p in history) + budget.used,
        retries=sum(x.retry_index > 0 for x in records),
        request_roles=dict(Counter(x.role for x in records)),
        case_image_requests=sum(x.get("image_count", 0) for x in request_rows),
        complete=complete,
        review_plan=plan.model_dump(mode="json") if plan else None,
        successful_review_responses=sum(x.role == "review" and x.error is None for x in records),
        saved_review_batch_results=sum(bool(b["result"]) for b in progress["batches"])
        if progress
        else 0,
        aggregate_saved=bool(progress and progress["aggregate"]),
        acceptance_roles=list(reports),
        owner_installations=len(installs),
        reports=reports,
        original_batch_receipts=candidate_checks,
        grant_checksum=digest(grant[0]) if grant else None,
        readback_journal_control_unchanged=True,
        public_journal_control_unchanged=True,
        new_tool_executions=0,
        formal_publication=False,
        tests_rerun=0,
    )
    save("readback.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "run_id",
                    "phase",
                    "steps",
                    "session_requests",
                    "request_roles",
                    "case_image_requests",
                    "complete",
                    "successful_review_responses",
                    "saved_review_batch_results",
                    "aggregate_saved",
                    "acceptance_roles",
                    "owner_installations",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

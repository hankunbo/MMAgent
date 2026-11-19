"""Explicit historical partial-fact closure, without importing an API launcher."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from agents.expert_agent.reporting import read_research_report
from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from packages.llm_provider.config import read_provider_config
from packages.research_runtime.agent_records import RuntimeAgentJournal


def _progress(phase):
    """Local diagnostic only; never an acceptance receipt or dispatch authority."""
    try:
        print(
            json.dumps(
                dict(
                    schema_version="local-finalization-progress/1.0.0",
                    phase=phase,
                    observed_at=datetime.now(UTC).isoformat(),
                    new_api_requests=0,
                    authoritative_result=False,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
    except OSError:
        # Losing a diagnostic sink must not invalidate a committed medical record.
        pass


def scope_recovery_owner(campaign, target):
    """Rebuild original input owner, without installing a dispatcher or live key."""
    from packages.llm_provider.images import validate_image

    from .campaign import Phase
    from .input_scope import configuration, current_entry

    if target not in campaign.scope_targets:
        raise ValueError("recorded_scope_target_not_in_original_plan")
    current_entry(campaign.base, target)
    original = json.loads((campaign.phase_path(target) / "result.json").read_bytes())
    docs = campaign.documents(original["run_id"])
    if docs.read("dynamic-phase-result") != (original,) or original["target"] != target:
        raise ValueError("recorded_scope_original_phase_binding")
    context = campaign.contexts.read_research_run(docs.run_id)
    if context.request.task != campaign.base.task:
        raise ValueError("recorded_scope_task_binding")
    campaign.base.contexts[docs.run_id] = context
    config = configuration(campaign.config, campaign.base.input_scope_plan)
    parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
    phase = Phase(campaign.base, target=target)
    expected, _ = phase.project(run_id=docs.run_id, config=config)
    if expected != parent:
        raise ValueError("recorded_scope_current_parent_changed")

    class Owner:
        def current(self):
            phase.verify(parent)
            return parent

        def verify_image(self, image):
            if image not in [o.image for o in self.current().material.observations]:
                raise ValueError("recorded_scope_image_binding")
            validate_image(image, campaign.base.image(image.reference))

    return phase.wrap_owner(Owner()), docs, parent, config


def finalize_scope(campaign, target, output, *, authorization, check_only=False):
    """Original acceptance of unchanged retained findings, never resume a stopped run."""
    from agents.expert_agent.reporting import render_research_html
    from packages.evidence_details.scope_finalization import (
        ACCEPTANCE_KEY,
        GRANT_KEY,
        RecordedScopePermission,
        RecordedScopeRequests,
        SavedScopePermission,
        reconstruct,
    )

    from . import report_storage
    from .input_scope import accepted_fact, current_entry

    out = Path(output).absolute()
    old_output = Path(campaign.output).absolute()
    if (
        out == old_output
        or out.is_relative_to(old_output)
        or old_output.is_relative_to(out)
        or any(p.is_symlink() for p in (out, *out.parents))
    ):
        raise ValueError("recorded_scope_output_path")
    if not authorization or len(authorization) > 200:
        raise ValueError("recorded_scope_authorization")
    _progress("restoring_scope_owner")
    owner, docs, parent, config = scope_recovery_owner(campaign, target)
    journal = RuntimeAgentJournal(docs)
    requests = RecordedScopeRequests(journal)
    before = dict(
        control=digest(campaign.sessions.read(docs.run_id)), journal=digest(journal.read())
    )
    original = docs.read("dynamic-phase-result")[0]
    _, sub, record = reconstruct(owner, docs, requests)
    prior = docs.read("accepted:" + ACCEPTANCE_KEY)
    if prior is not None and (len(prior) != 4 or prior[3].get("status") != "accepted"):
        raise ValueError("recorded_scope_previous_rejection")
    kwargs = dict(owner=owner, documents=docs, requests=requests, config=config)
    if check_only:
        if prior is not None:
            SavedScopePermission(grant=docs.read(GRANT_KEY)[0], **kwargs).check(parent)
        else:
            RecordedScopePermission.prepare(**kwargs, authorization=authorization, persist=False)
        return dict(
            eligible=True,
            target=target,
            run_id=docs.run_id,
            original_error=record.error,
            original_request_checksum=digest(record),
            interpretation=sub.storage_interpretation.model_dump(mode="json"),
            already_accepted=prior is not None,
            new_api_requests=0,
            database_writes=0,
        )
    report_key = "recorded-input-scope:report:1"
    result_key = "recorded-input-scope:result:1"
    saved_result = docs.read(result_key)
    if out.exists() and (
        not (out / "original-failure.json").exists()
        or json.loads((out / "original-failure.json").read_bytes()) != original
    ):
        raise ValueError("recorded_scope_output_conflict")
    if prior is None:
        permission, sub = RecordedScopePermission.prepare(**kwargs, authorization=authorization)
        acceptance = ResearchAcceptance(
            owner=owner, documents=docs, requests=requests, live_permission=permission
        )
        # Diagnose through the same original validator before reserving an immutable key.
        acceptance._check(sub, parent)
        submitted = acceptance.submit(sub, key=ACCEPTANCE_KEY)
        if not isinstance(submitted, dict) or submitted.get("status") != "submitted_not_accepted":
            raise ValueError("recorded_scope_submission_rejected")
        if acceptance.accept(key=ACCEPTANCE_KEY).status != "accepted":
            raise ValueError("recorded_scope_acceptance_rejected")
    else:
        permission = SavedScopePermission(grant=docs.read(GRANT_KEY)[0], **kwargs)
        permission.check(parent)
        acceptance = ResearchAcceptance(
            owner=owner, documents=docs, requests=requests, live_permission=permission
        )
    _progress("scope_acceptance_readback")
    report = read_research_report(acceptance, key=ACCEPTANCE_KEY, parent=parent)
    if report["receipt"]["permission_checksum"] != digest(permission.grant):
        raise ValueError("recorded_scope_receipt_grant_binding")
    result = dict(
        schema_version="recorded-input-scope-result/1.0.0",
        target=target,
        run_id=docs.run_id,
        accepted=True,
        acceptance_key=ACCEPTANCE_KEY,
        original_phase=original,
        original_phase_checksum=digest(original),
        original_request_checksum=digest(record),
        grant_checksum=digest(permission.grant),
        records_checksum=report["receipt"]["records_checksum"],
        new_api_requests=0,
        new_disease_scores=0,
        campaign_resumed=False,
    )
    fact = accepted_fact(
        acceptance.read(key=ACCEPTANCE_KEY),
        parent,
        result,
        current_entry(campaign.base, target)["task"],
        campaign.sessions.read(docs.run_id),
    )
    if saved_result is not None and saved_result != (result,):
        raise ValueError("recorded_scope_result_conflict")
    existing_report = docs.read(report_key)
    if existing_report:
        stored, rendered = report_storage.unpack(existing_report, key=report_key)
        if stored != report:
            raise ValueError("recorded_scope_report_changed")
        page = rendered["html"]
    else:
        page = render_research_html(report)
        report_storage.save(docs, report, page, key=report_key)
    with docs.session():
        docs.write(result_key, (result,))
    after = dict(
        control=digest(campaign.sessions.read(docs.run_id)), journal=digest(journal.read())
    )
    if before != after or docs.read("dynamic-phase-result") != (original,):
        raise ValueError("recorded_scope_history_changed")
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(out / "original-failure.json", original)
    write_once(out / "result.json", result)
    write_once(out / "scope-fact.json", fact)
    write_once(out / "report.json", report)
    html_path = out / "report.html"
    if html_path.exists():
        if html_path.read_text() != page:
            raise ValueError("recorded_scope_html_conflict")
    else:
        with html_path.open("x") as stream:
            stream.write(page)
    return dict(result, html=str(html_path), original_history_unchanged=True)


def read_finalized_scope(campaign, target):
    """Current owner-qualified use of a local recovery, never an execution receipt."""
    from packages.evidence_details.scope_finalization import (
        ACCEPTANCE_KEY,
        GRANT_KEY,
        RecordedScopeRequests,
        SavedScopePermission,
    )

    from .input_scope import accepted_fact, current_entry

    owner, docs, parent, config = scope_recovery_owner(campaign, target)
    result = docs.read("recorded-input-scope:result:1")
    if not result or len(result) != 1:
        raise ValueError("recorded_scope_result_missing")
    result = result[0]
    permission = SavedScopePermission(
        grant=docs.read(GRANT_KEY)[0],
        config=config,
        documents=docs,
        requests=RecordedScopeRequests(RuntimeAgentJournal(docs)),
        owner=owner,
    )
    permission.check(parent)
    original = docs.read("dynamic-phase-result")[0]
    values = docs.read("accepted:" + ACCEPTANCE_KEY)
    if (
        result.get("schema_version") != "recorded-input-scope-result/1.0.0"
        or result.get("target") != target
        or result.get("run_id") != docs.run_id
        or result.get("acceptance_key") != ACCEPTANCE_KEY
        or result.get("original_phase") != original
        or result.get("original_phase_checksum") != digest(original)
        or result.get("original_request_checksum") != permission.grant.original_request_checksum
        or result.get("grant_checksum") != digest(permission.grant)
        or result.get("records_checksum") != digest(values[0])
        or values[3].get("permission_checksum") != digest(permission.grant)
        or result.get("new_api_requests") != 0
        or result.get("new_disease_scores") != 0
        or result.get("campaign_resumed") is not False
    ):
        raise ValueError("recorded_scope_result_binding")
    # Original reporting validates the graph, receipt, records and null score together.
    acceptance = ResearchAcceptance(owner=owner, documents=docs, requests=permission.budget)
    read_research_report(acceptance, key=ACCEPTANCE_KEY, parent=parent)
    return accepted_fact(
        values,
        parent,
        result,
        current_entry(campaign.base, target)["task"],
        campaign.sessions.read(docs.run_id),
    )


def finalize_entry(
    previous,
    output,
    *,
    complete=False,
    lossless_explanations=False,
    citation_rejection=False,
    recover_confirmed_raw=False,
):
    """New output directory and authorization; never call old run/execute or rewrite it."""
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.dataset_adapters.multimodal.preparation import verify
    from packages.evidence_details import partial_finalization, recorded_finalization
    from packages.research_runtime.task_context import TaskIntent

    from .campaign import InputEvidence, MultimodalCampaign
    from .entry import owners
    from .material import MedicalSources

    module = recorded_finalization if complete else partial_finalization
    if recover_confirmed_raw and (complete or lossless_explanations or citation_rejection):
        raise ValueError("raw_batch_recovery_partial_deadline_only")
    if citation_rejection and (complete or lossless_explanations):
        raise ValueError("citation_partial_closure_only")
    if lossless_explanations:
        if not complete:
            raise ValueError("explanation_complete_closure_required")
        from packages.evidence_details import recorded_explanation

        module = recorded_explanation

    path, out = Path(previous).absolute(), Path(output).absolute()
    if any(p.is_symlink() for root in (path, out) for p in (root, *root.parents)) or path == out:
        raise ValueError("partial_finalization_path")
    old = json.loads((path / "result.json").read_bytes())
    if out.exists() and (
        not (out / "original-failure.json").is_file()
        or json.loads((out / "original-failure.json").read_bytes()) != old
    ):
        raise ValueError("partial_finalization_output_conflict")
    _progress("restoring_original_owner")
    if not (path / "entry.json").exists():
        # A completed execution window can still have a failed outer summary.
        # Restore only an existing owner-bound acquisition or definition Review;
        # never invoke execute/dispatch or overwrite its historical result.
        plan = json.loads((path / "protocol.json").read_bytes())
        deadline_partial = (
            not complete
            and plan.get("schema_version") == "multimodal-campaign-renewal/1.27.0"
            and plan.get("mode") == "review"
            and old.get("complete") is False
            and old.get("phases")
            and old["phases"][-1].get("error") == "deadline"
        )
        raw_deadline_partial = (
            recover_confirmed_raw
            and plan.get("schema_version")
            in {"multimodal-campaign-renewal/1.35.0", "multimodal-campaign-renewal/1.40.0"}
            and plan.get("mode") == "review"
            and isinstance(plan.get("root_direction_review"), dict)
            and isinstance(plan.get("text_review_method"), dict)
            and old.get("complete") is False
            and old.get("phases")
            and old["phases"][-1].get("error") == "deadline"
        )
        if recover_confirmed_raw and not raw_deadline_partial:
            raise ValueError("raw_batch_recovery_original_version")
        citation_partial = (
            citation_rejection
            and plan.get("schema_version") == "multimodal-campaign-renewal/1.29.0"
            and plan.get("mode") == "review"
            and isinstance(plan.get("root_direction_review"), dict)
            and old.get("complete") is False
            and old.get("phases")
            and old["phases"][-1].get("error") == "candidate_failed_or_context_changed"
        )
        if citation_rejection and not citation_partial:
            raise ValueError("citation_partial_original_version")
        partition = (
            plan.get("schema_version")
            in {
                "multimodal-campaign-renewal/1.12.0",
                "multimodal-campaign-renewal/1.26.0",
                "multimodal-campaign-renewal/1.47.0",
                "multimodal-campaign-renewal/1.48.0",
                "multimodal-campaign-renewal/1.49.0",
                "multimodal-campaign-renewal/1.50.0",
                "multimodal-campaign-renewal/1.51.0",
                "multimodal-campaign-renewal/1.60.0",
                "multimodal-campaign-renewal/1.66.0",
                "multimodal-campaign-renewal/1.67.0",
            }
            and plan.get("mode") == "acquisition"
            and plan.get("review_partition") == "owner_bound_reused_and_new/1.0.0"
            and not plan.get("acquisition_increment_only")
        )
        # A failed outer acquisition-increment summary can follow an already
        # committed Review. This entry only exports that exact commit: it may
        # never turn an unaccepted increment into a new acceptance.
        committed_increment = (
            complete
            and plan.get("schema_version") == "multimodal-campaign-renewal/1.26.0"
            and plan.get("mode") == "acquisition"
            and plan.get("review_partition") == "owner_bound_reused_and_new/1.0.0"
            and plan.get("acquisition_increment_only") is True
            and old.get("complete") is False
            and old.get("phases")
            and old["phases"][-1].get("target") is None
            and old["phases"][-1].get("accepted") is False
            and old["phases"][-1].get("error") == "deadline"
        )
        definition_review = (
            plan.get("schema_version")
            in {"multimodal-campaign-renewal/1.13.0", "multimodal-campaign-renewal/1.14.0"}
            and plan.get("mode") == "review"
            and plan.get("review_definition_increment") is True
            and plan.get("definition_increment_only") is True
        )
        pending_explanation = (
            lossless_explanations
            and plan.get("schema_version")
            in {"multimodal-campaign-renewal/1.15.0", "multimodal-campaign-renewal/1.16.0"}
            and plan.get("mode") == "review"
            and isinstance(plan.get("pending_review"), dict)
            and plan["pending_review"].get("schema_version") == "pending-review-inventory/1.0.0"
            and plan.get("review_increment_only") is True
        )
        if not (
            citation_partial
            or deadline_partial
            or raw_deadline_partial
            or committed_increment
            or complete
            and (partition or definition_review or pending_explanation)
        ):
            raise ValueError("recorded_finalization_renewal_version")
        from .campaign_renewal import completed_parent

        # This original owner verifies lineage, stopped Control, request journal,
        # accepted definition revision and exact selected IDs. Flags are not authority.
        campaign, restored = completed_parent(path)
        if restored != old:
            raise ValueError("recorded_finalization_original_result_changed")
        return _finish_entry(
            campaign,
            old,
            out,
            module,
            complete=complete,
            active=True,
            **({"citation_rejection": True} if citation_partial else {}),
            **({"recover_confirmed_raw": True} if raw_deadline_partial else {}),
            **({"require_committed": True} if committed_increment else {}),
        )
    if recover_confirmed_raw:
        raise ValueError("raw_batch_recovery_original_version")
    if citation_rejection:
        raise ValueError("citation_partial_original_version")
    spec = json.loads((path / "entry.json").read_bytes())
    if complete and spec.get("active_acquisition"):
        # Restore original facts/definitions through the existing read-only owner.
        # Never use assemble_entry: its normal installation may dispatch native work.
        from .report_recovery import saved_campaign

        campaign, restored = saved_campaign(path)
        if restored != old:
            raise ValueError("recorded_finalization_original_result_changed")
        return _finish_entry(campaign, old, out, module, complete=True, active=True)
    runtime, objects, inputs, target = owners()
    visible = inputs.read_visible(old["case_code"])
    prepared = prepare(
        spec["dataset"],
        spec["root"],
        spec["sample"],
        selection=spec["selection"],
        cutoff_visit=spec["cutoff_visit"],
    )
    if (
        prepared.manifest != verify(path / "input/preparation.json")
        or prepared.manifest.checksum != spec["input_checksum"]
    ):
        raise ValueError("partial_finalization_input_changed")
    from types import SimpleNamespace

    base = InputEvidence(
        registered=SimpleNamespace(visible=visible),
        inputs=inputs,
        runtime=runtime,
        objects=objects,
        prepared=prepared,
        task=validate(TaskIntent, spec["intent"]),
        output=path,
        sources=MedicalSources(),
    )
    base.known_only = spec["known_only"]
    base.previous = spec.get("previous")
    base.review_replacement = spec.get("review_replacement", False)
    if not base.known_only:
        raise ValueError("partial_finalization_known_measurements_only")
    if not spec.get("confirmed_observations_only"):
        for row in base.ledger:
            if row["id"] not in base.observations:
                row["missing_reason"] = "visual_attempt_not_replayed_review_known_measurements_only"
    campaign = MultimodalCampaign(base, target, campaign_id=path.name)
    campaign.config = read_provider_config(spec["provider"])
    if spec.get("confirmed_observations_only"):
        from .confirmed_observations import ConfirmedObservations

        campaign.confirmed_observations = ConfirmedObservations(campaign)
    if spec.get("reuse_procedures"):
        from .reuse import ProcedureReuse, repository

        campaign.procedure_reuse = ProcedureReuse(campaign, repository(campaign))
        campaign.procedure_reuse.restore(
            json.loads((path / "protocol.json").read_bytes())["procedure_reuse"]
        )
        campaign.procedure_reuse.check()
    return _finish_entry(campaign, old, out, module, complete=complete)


def _finish_entry(
    campaign,
    old,
    out,
    module,
    *,
    complete,
    active=False,
    citation_rejection=False,
    recover_confirmed_raw=False,
    require_committed=False,
):
    """Shared original acceptance, score and report path; no request dispatcher."""
    from .campaign import Phase
    from .report import export

    base = campaign.base
    ACCEPTANCE_KEY, GRANT_KEY, REPORT_KEY, RESULT_KEY = (
        module.ACCEPTANCE_KEY,
        module.GRANT_KEY,
        module.REPORT_KEY,
        module.RESULT_KEY,
    )
    original_failure = module.original_failure
    request_type = module.RecordedBatchRequests if complete else module.ClosedReviewRequests
    permission_type = (
        module.RecordedReviewPermission if complete else module.PartialReviewPermission
    )
    docs = campaign.documents(old["phases"][-1]["run_id"])
    if original_failure(docs) != old:
        raise ValueError("partial_finalization_original_result_mismatch")
    if not complete:
        GRANT_KEY, _, ACCEPTANCE_KEY = module.closure_keys(
            docs, **({"citation_rejection": True} if citation_rejection else {})
        )
    if citation_rejection:
        RESULT_KEY = "citation-partial-finalization-result"
        REPORT_KEY = "citation-partial-finalization-report"
    context = campaign.contexts.read_research_run(docs.run_id)
    if context.request.task != base.task:
        raise ValueError("partial_finalization_task_binding")
    base.contexts[docs.run_id] = context
    parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
    phase = Phase(base, observation_reader=campaign.accepted_observations)
    expected, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    if expected != parent:
        raise ValueError("partial_finalization_current_parent_changed")
    _progress("current_parent_verified")

    class Owner:
        confirmed_review_reader = getattr(base, "confirmed_review_reader", None)

        def current(self):
            phase.verify(parent)
            return parent

    journal = RuntimeAgentJournal(docs)
    before, events = campaign.sessions.read(docs.run_id), journal.read()
    requests = request_type(journal)
    raw_recovery = None
    if recover_confirmed_raw:
        from packages.evidence_details.batch_local_recovery import KEY, recover

        receipt = recover(owner=Owner(), documents=docs, requests=requests)
        raw_recovery = docs.reference(
            KEY, 0, receipt, role="recovery", version=receipt["schema_version"]
        ).model_dump(mode="json")
        _progress("confirmed_raw_batches_recovered_no_dispatch")
    committed = None
    acceptance_key = ACCEPTANCE_KEY
    if active:
        from agents.expert_agent.committed_review import read_committed_review

        # An outer deadline is not evidence that the original domain commit
        # failed. Resolve its exact intent before proposing a second acceptance.
        committed = read_committed_review(
            docs,
            parent=parent,
            config=campaign.config,
            confirmed_review_reader=getattr(base, "confirmed_review_reader", None),
        )
        if committed is not None:
            acceptance_key = committed["acceptance_key"]
    if require_committed and committed is None:
        raise ValueError("recorded_increment_requires_original_commit")
    if complete and committed is None:
        GRANT_KEY, _, acceptance_key = module.closure_keys(docs)
    prior = docs.read("accepted:" + acceptance_key)
    if committed is not None:
        Owner().current()
        acceptance = ResearchAcceptance(owner=Owner(), documents=docs, requests=None)
    elif prior is None:
        permission, submission = permission_type.prepare(
            owner=Owner(),
            documents=docs,
            requests=requests,
            config=campaign.config,
            **({"citation_rejection": True} if citation_rejection else {}),
        )
        _progress("local_grant_prepared")
        acceptance = ResearchAcceptance(
            owner=Owner(), documents=docs, requests=requests, live_permission=permission
        )
        submitted = acceptance.submit(submission, key=acceptance_key)
        if not isinstance(submitted, dict) or submitted.get("status") != "submitted_not_accepted":
            raise ValueError("partial_finalization_submission_rejected")
        _progress("submission_saved_not_yet_accepted")
        if acceptance.accept(key=acceptance_key).status != "accepted":
            raise ValueError("partial_finalization_acceptance_rejected")
        _progress("acceptance_saved")
    else:
        Owner().current()
        if citation_rejection:
            module.accepted_citation_permission(parent, docs, requests)
        acceptance = ResearchAcceptance(owner=Owner(), documents=docs, requests=requests)
    review = read_research_report(acceptance, key=acceptance_key, parent=parent)
    _progress("accepted_report_readback_verified")
    grant = None if committed else docs.read(GRANT_KEY)[0]
    result = dict(
        old,
        schema_version="multimodal-recorded-finalization-result/1.2.0"
        if committed
        else "multimodal-recorded-finalization-result/1.1.0"
        if active and complete
        else "multimodal-partial-finalization-result/1.1.0"
        if active
        else "multimodal-recorded-finalization-result/1.0.0"
        if complete
        else "multimodal-partial-finalization-result/1.0.0",
        complete=True,
        new_requests=0,
        original_result_checksum=digest(old),
        finalization_grant_checksum=digest(grant) if grant else None,
        phases=[
            *(old["phases"][:-1] if active else []),
            dict(
                old["phases"][-1],
                accepted=True,
                acceptance_key=acceptance_key,
                error=None,
                evidence_limited=True,
            ),
        ],
        execution_kind="committed_review_export_not_reacceptance"
        if committed
        else "complete_recorded_fact_finalization_not_run_resume"
        if complete
        else "partial_historical_fact_finalization_not_run_resume",
        original_requests=old["new_requests"],
    )
    if (
        not complete
        and grant
        and grant.get("schema_version") == "partial-review-finalization/1.1.0"
    ):
        result["review_closure_scope"] = dict(
            local_confirmed_only=True,
            historical_answers_reaccepted=False,
            excluded_historical_checksum=grant["excluded_historical_checksum"],
            pending_semantics="not_accepted_not_negative",
        )
    if citation_rejection:
        result.update(
            schema_version="multimodal-partial-finalization-result/1.2.0",
            review_closure_scope=dict(
                local_confirmed_only=True,
                rejected_answers_repaired=False,
                historical_answers_reaccepted=False,
                rejected_response_checksums=grant["rejected_response_checksums"],
                pending_semantics="not_accepted_not_negative",
                campaign_execution_complete=False,
            ),
        )
    if raw_recovery:
        # Same partial-score semantics; independently versioned recovery provenance.
        result["local_raw_batch_recovery"] = raw_recovery
        result["review_closure_scope"]["campaign_execution_complete"] = False
    if complete and not committed and GRANT_KEY == module.PARTITION_GRANT_KEY:
        result["finalization_policy"] = module.PARTITION_POLICY
    if getattr(module, "POLICY", None) == "identical-review-explanation-keys/1.0.0":
        result.update(
            schema_version="multimodal-recorded-finalization-result/1.3.0",
            finalization_policy=module.POLICY,
            local_response_interpretation=docs.read(module.SUBMISSION_KEY)[0][
                "local_interpretation"
            ],
        )
    if committed:
        from .stage_models import LocalReviewClosure

        result["local_review_closure"] = LocalReviewClosure(
            original_phase=old["phases"][-1],
            original_phase_checksum=digest(old["phases"][-1]),
            confirmation=committed,
        ).model_dump(mode="json")
    with docs.session():
        docs.write(RESULT_KEY, (result,))
    _progress("local_result_saved")
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    write_once(out / "original-failure.json", old)
    write_once(out / "review/review-report.json", review)
    write_once(out / "result.json", result)
    if active:
        # The original phase paths remain the source of confirmed visual facts,
        # native artifacts and the acquisition-stop receipt. Only export naming
        # changes; moving output would silently hide those original phase files.
        campaign.report_name = out.name
    else:
        campaign.output = out
    campaign.report_key = REPORT_KEY
    saved_report = docs.read(REPORT_KEY) if active else None
    exported = (
        _saved_export(saved_report, key=REPORT_KEY, result=result, review=review, name=out.name)
        if saved_report
        else export(campaign, result)
    )
    if (
        campaign.sessions.read(docs.run_id) != before
        or journal.read() != events
        or original_failure(docs) != old
    ):
        raise ValueError("partial_finalization_mutated_history")
    _progress("export_saved_history_unchanged")
    from .memory import archive

    return dict(
        exported,
        original_failure_preserved=True,
        control_unchanged=True,
        requests_unchanged=True,
        memory=archive(campaign, result),
    )


def _saved_export(values, *, key, result, review, name, directory=Path("docs/exports")):
    """Retry downstream archival without rendering or overwriting a saved report."""
    from .report_io import publish_html, publish_json
    from .report_storage import unpack

    report, page = unpack(values, key=key)
    if (
        report["result"] != result
        or report["original_review"] != review
        or report["scores"] != review["scores"]
    ):
        raise ValueError("recorded_finalization_saved_report_binding")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("Multimodal_" + name)
    return dict(
        complete=result["complete"],
        requests=result["new_requests"],
        json=str(publish_json(path.with_suffix(".json"), report)),
        html=str(publish_html(path.with_suffix(".html"), page["html"])),
        report_checksum=digest(report),
        task_clinically_validated=False,
    )

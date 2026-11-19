"""Bounded continuation of stopped reassessments through the original Review owner.

Never replay a claimed request. Explicit acknowledgments select NEW assessments;
unstarted batches are separately selected by the original failed-attempt reader.
"""

from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from packages.evidence_details.pending_review import read_failed_review_attempt
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

from . import generic_pending_review as pending
from . import generic_reassessment as reassessment
from .generic_input import directory

LEGACY_VERSION = "generic-review-selection/1.7.0"
VERSION = "generic-review-selection/1.8.0"
EXECUTION_LOCK = "original-runtime-case-pipeline-session/1.0.0"
PROFILE = "stopped-reassessment-continuation-v1"
MAX_HISTORY = 3
read = reassessment.read


def failed_snapshot(campaign, key, selection):
    """Exact local/DB history plus freshly projected source and original intent."""
    path = directory(key)
    if (path / "cancel-request.json").exists():
        raise ValueError("generic_continuation_history_cancelled")
    result = read(path / "evidence/result.json")
    if len(result["phases"]) != 1:
        raise ValueError("generic_continuation_review_only")
    final = result["phases"][0]
    if (
        final.get("target") is not None
        or final.get("accepted") is not False
        or final.get("acceptance_key") is not None
    ):
        raise ValueError("generic_continuation_requires_failed_review")
    docs = campaign.documents(final["run_id"])
    request = read(path / "evidence/review/request.json")
    context = campaign.contexts.read_research_run(docs.run_id)
    protocol = read(path / "evidence/protocol.json")
    report = docs.read("dynamic-case-report")
    phase = campaign.phase_class(
        campaign.base,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    state, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    phase.verify(state)
    journal = RuntimeAgentJournal(docs).read()
    reserved = sum(e.get("phase") == "reserved" for e in journal)
    if (
        context is None
        or context.request.model_dump(mode="json") != request
        or request["lock"]["run_id"] != docs.run_id
        or docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),)
        or docs.read("dynamic-phase-result") != (final,)
        or docs.read("dynamic-case-result") != (result,)
        or docs.read("dynamic-campaign") != (protocol,)
        or protocol["input"] != digest(campaign.spec)
        or protocol["provider_checksum"] != digest(campaign.config)
        or protocol["observations"]["pending_review"] != campaign.pending_review_proof
        or not report
        or report[0]["result"] != result
        or report[1]["report_checksum"] != digest(report[0])
        or report[0].get("scores") is not None
        or result["new_requests"] != reserved
        or result["prior_graph_requests"] != campaign.prior()["reservations"]
    ):
        raise ValueError("generic_continuation_failed_history_binding")
    inventory = read_failed_review_attempt(documents=docs, state=state, config=campaign.config)
    proof = dict(
        key=key,
        run_id=docs.run_id,
        selection=digest(selection),
        result=digest(result),
        report=digest(report),
        protocol=digest(protocol),
        request=digest(request),
        state=digest(state),
        journal=digest(journal),
        control=inventory["control_checksum"],
        reservations=reserved,
        inventory=digest(inventory),
    )
    return SimpleNamespace(
        key=key,
        docs=docs,
        state=state,
        result=result,
        report=report,
        protocol=protocol,
        request=request,
        selection=selection,
        inventory=inventory,
        proof=proof,
        reservations=reserved,
    )


def select_work(snapshot, acknowledgments, accepted_ids, config):
    selected = set(snapshot.inventory["selected_ids"])
    proofs = []
    if acknowledgments:
        acknowledged, proofs = reassessment.acknowledged_selection(
            [(snapshot.key, snapshot.docs, snapshot.state)],
            acknowledgments,
            accepted_ids=accepted_ids,
            all_ids=[v.id for v in snapshot.state.policy.items],
            config=config,
        )
        selected.update(acknowledged)
    all_ids = {v.id for v in snapshot.state.policy.items}
    if not selected or not selected <= all_ids or selected & set(accepted_ids):
        raise ValueError("generic_continuation_no_safe_selected_work")
    return sorted(selected), proofs


def origin(prior_key, acknowledgments, *, seen=()):
    if prior_key in seen or len(seen) >= MAX_HISTORY:
        raise ValueError("generic_continuation_history_limit_or_cycle")
    path = directory(prior_key)
    selection = read(path / "input.json")
    if selection.get("key") != prior_key or selection.get("review_key") != prior_key:
        raise ValueError("generic_continuation_prior_identity")
    version = selection.get("schema_version")
    if version == reassessment.VERSION:
        base = reassessment.origin(
            selection["accepted_review_key"], selection["proof"]["acknowledge_requests"]
        )
        expected = reassessment.selection_for(base, prior_key)
        issuer = selection["accepted_review_key"]
        previous_campaign = reassessment.ReassessmentReview(base, prior_key)
        history = []
    elif version in (VERSION, LEGACY_VERSION):
        previous = origin(
            selection["continuation_of"],
            selection["acknowledge_requests"],
            seen=(*seen, prior_key),
        )
        base = previous.base
        expected = selection_for(previous, prior_key, legacy=version == LEGACY_VERSION)
        issuer = selection["continuation_of"]
        previous_campaign = ContinuationReview(
            previous, prior_key, legacy_lock=version == LEGACY_VERSION
        )
        history = list(previous.history)
    else:
        raise ValueError("generic_continuation_unsupported_history")
    if (
        selection != dict(expected, key=prior_key)
        or read(directory(issuer) / "review-successor.json") != expected
        or read(path / "binding.json") != read(directory(base.proof["parent_key"]) / "binding.json")
    ):
        raise ValueError("generic_continuation_claim_binding")
    snapshot = failed_snapshot(previous_campaign, prior_key, selection)
    history.append(snapshot)
    selected, proofs = select_work(
        snapshot, acknowledgments, base.snapshot.ids, previous_campaign.config
    )
    previous_requests = base.proof["previous_requests"] + sum(s.reservations for s in history)
    maximum = min(10, 40 - previous_requests)
    size = previous_campaign.config.review_batching.batch_size
    if maximum < (len(selected) + size - 1) // size:
        raise ValueError("generic_continuation_cumulative_budget")
    proof = dict(
        schema_version="generic-review-continuation-proof/1.0.0",
        parent_key=base.proof["parent_key"],
        accepted_review_key=base.proof["accepted_review_key"],
        continuation_of=prior_key,
        original_reassessment=base.proof,
        failed_history=[s.proof for s in history],
        previous_requests=previous_requests,
        max_new_requests=maximum,
        selected_ids=selected,
        acknowledge_requests=sorted(acknowledgments),
        owner_proofs=proofs,
        inventory=dict(snapshot.inventory, accepted_ids=base.snapshot.ids),
        provider_checksum=digest(previous_campaign.config),
        max_cumulative_requests=40,
        new_vision_requests=0,
        selection_kind="unstarted_and_exact_acknowledged_new_review_old_unknown_unchanged",
    )
    return SimpleNamespace(base=base, history=history, proof=proof)


class ContinuationReview(reassessment.ReassessmentReview):
    def __init__(self, checked, key, *, legacy_lock=False):
        self.continuation = checked
        self.legacy_lock = legacy_lock
        super().__init__(checked.base, key)
        self.pending_review_proof = checked.proof
        self.phase_class = pending.phase_type(lambda: self.pending_review_proof)
        self.version = (
            "generic-review-continuation-v1:" if legacy_lock else "generic-review-continuation-v2:"
        ) + key
        self.max_new_requests = checked.proof["max_new_requests"]

    def execution_session(self):
        if self.legacy_lock:
            return super().execution_session()
        from db.runtime.repository.case_pipeline import CasePipelineRepository

        # Existing Runtime owner, existing role and session advisory lock. No
        # migration, alternate database, long transaction or file-lock fallback.
        key = "generic-review-execution:" + digest([self.spec["key"], self.review_key])[7:]
        return CasePipelineRepository(self.target).session(self.base.binding["case_code"], key)

    def prior(self):
        base = super().prior()
        history = self.continuation.history
        for s in history:
            p, docs = s.proof, s.docs
            path = directory(s.key)
            if (
                digest(RuntimeAgentJournal(docs).read()) != p["journal"]
                or digest(self.sessions.read(docs.run_id)) != p["control"]
                or docs.read("dynamic-case-result") != (s.result,)
                or docs.read("dynamic-case-report") != s.report
                or docs.read("dynamic-campaign") != (s.protocol,)
                or read(path / "input.json") != s.selection
                or read(path / "evidence/result.json") != s.result
                or read(path / "evidence/protocol.json") != s.protocol
                or read(path / "evidence/review/request.json") != s.request
                or read(
                    directory(
                        s.selection.get("continuation_of", s.selection["accepted_review_key"])
                    )
                    / "review-successor.json"
                )
                != {k: v for k, v in s.selection.items() if k != "key"}
                or read(path / "binding.json") != read(directory(self.spec["key"]) / "binding.json")
            ):
                raise ValueError("generic_continuation_history_changed")
        total = base["reservations"] + sum(s.reservations for s in history)
        if total != self.continuation.proof["previous_requests"]:
            raise ValueError("generic_continuation_accounting")
        return dict(reservations=total, original=base, continuation_proof=self.continuation.proof)

    def before_review(self):
        p = self.continuation.proof
        if origin(p["continuation_of"], p["acknowledge_requests"]).proof != p:
            raise ValueError("generic_continuation_current_history_changed")

    def guard(self, *, dispatch=False):
        from packages.llm_provider.provider import ProviderError

        if any(
            (directory(s.key) / "cancel-request.json").exists() for s in self.continuation.history
        ):
            raise ProviderError("cancelled")
        return super().guard(dispatch=dispatch)

    def cancellation_signal(self):
        signal = super().cancellation_signal()
        original = signal.is_set
        signal.is_set = lambda: (
            original()
            or any(
                (directory(s.key) / "cancel-request.json").exists()
                for s in self.continuation.history
            )
        )
        return signal


def selection_for(checked, key, *, legacy=False):
    p = checked.proof
    result = dict(
        schema_version=LEGACY_VERSION if legacy else VERSION,
        review_profile=PROFILE,
        review_key=key,
        parent_key=p["parent_key"],
        accepted_review_key=p["accepted_review_key"],
        continuation_of=p["continuation_of"],
        acknowledge_requests=p["acknowledge_requests"],
        proof=p,
        input_checksum=digest(checked.base.origin.campaign.spec),
        provider_checksum=p["provider_checksum"],
        max_new_requests=p["max_new_requests"],
        max_cumulative_requests=40,
        seconds=2700,
        new_vision_requests=0,
    )
    if not legacy:
        result["execution_lock"] = EXECUTION_LOCK
    return result


def run(key, review_key, prior_key, acknowledgments=(), *, preflight=False):
    import fcntl

    from packages.llm_provider.network_route import route_opener

    from .generic_control import start
    from .generic_run import finish_export, persist_export

    root, prior, out = directory(key), directory(prior_key), directory(review_key)
    with (root / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = origin(prior_key, acknowledgments)
        excluded = {
            key,
            checked.proof["accepted_review_key"],
            checked.base.proof["prior_review_key"],
        }
        excluded.update(s.key for s in checked.history)
        if key != checked.proof["parent_key"] or review_key in excluded:
            raise ValueError("generic_continuation_distinct_keys")
        config = checked.base.accepted.config
        route_opener(config.network_route, config.transport.base_url)
        selection = selection_for(checked, review_key)
        claim = prior / "review-successor.json"
        if claim.exists() and read(claim) != selection:
            raise ValueError("generic_continuation_successor_conflict")
        if (out / "input.json").exists() and read(out / "input.json") != dict(
            selection, key=review_key
        ):
            raise ValueError("generic_continuation_output_conflict")
        started = (out / "started.json").exists()
        if (
            started
            and not (out / "export-reference.json").exists()
            and not (out / "evidence/result.json").exists()
        ):
            raise ValueError("generic_continuation_claimed_use_status")
        if preflight:
            return dict(
                status="saved_export_only" if started else "ready",
                selection=selection,
                api_requests=0,
                database_writes=0,
            )
        campaign = ContinuationReview(checked, review_key)
        write_once(claim, selection)
        write_once(out / "input.json", dict(selection, key=review_key))
        write_once(out / "binding.json", read(root / "binding.json"))
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(review_key))
        if started:
            saved, result = restore(out)
            return persist_export(review_key, saved, result)
        start(dict(selection, key=review_key))
        return persist_export(review_key, campaign, campaign.execute())


def restore(path):
    selection = read(path / "input.json")
    checked = origin(selection["continuation_of"], selection["acknowledge_requests"])
    legacy = selection["schema_version"] == LEGACY_VERSION
    expected = selection_for(checked, path.name, legacy=legacy)
    if (
        selection != dict(expected, key=path.name)
        or read(directory(selection["continuation_of"]) / "review-successor.json") != expected
        or read(path / "binding.json") != read(directory(selection["parent_key"]) / "binding.json")
    ):
        raise ValueError("generic_continuation_saved_binding")
    return pending.confirmed_campaign(ContinuationReview(checked, path.name, legacy_lock=legacy))

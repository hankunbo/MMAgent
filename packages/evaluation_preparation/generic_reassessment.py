"""Explicit new Review after two quarantined histories and an accepted sibling.

Uses the existing GenericReview, original pending/acceptance owners and export.
Old UNKNOWN requests remain quarantined; only exact acknowledged work is selected.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from packages.evaluation_preparation import generic_pending_review as pending
from packages.evaluation_preparation import generic_review_followup as followup
from packages.evaluation_preparation.generic_input import directory
from packages.evidence_details.pending_review import read_failed_review_attempt
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

VERSION = "generic-review-selection/1.6.0"
PROFILE = "acknowledged-chain-reassessment-v1"


def read(path):
    return json.loads(Path(path).read_bytes())


def acknowledged_selection(histories, acknowledgments, *, accepted_ids, all_ids, config):
    """Original failed-attempt owner authorizes exact requests, never all UNKNOWNs."""
    acknowledgments = tuple(acknowledgments)
    if (
        not 1 <= len(acknowledgments) <= 4
        or len(set(acknowledgments)) != len(acknowledgments)
        or any(not isinstance(v, str) or not v.startswith("request:") for v in acknowledgments)
    ):
        raise ValueError("generic_reassessment_exact_requests_required")
    found, selected, proofs = set(), set(), []
    for history, documents, state in histories:
        inventory = read_failed_review_attempt(documents=documents, state=state, config=config)
        for batch in inventory["quarantined_batches"]:
            matched = set(batch["request_ids"]) & set(acknowledgments)
            if not matched:
                continue
            if len(matched) != 1 or len(batch["request_ids"]) != 1 or matched & found:
                raise ValueError("generic_reassessment_ambiguous_request")
            request_id = next(iter(matched))
            current = read_failed_review_attempt(
                documents=documents,
                state=state,
                config=config,
                acknowledge_request=request_id,
            )
            allowed = [
                b for b in current["quarantined_batches"] if b["new_reassessment_authorized"]
            ]
            if len(allowed) != 1 or allowed[0]["request_ids"] != [request_id]:
                raise ValueError("generic_reassessment_owner_acknowledgment")
            ids = set(allowed[0]["assessment_ids"])
            if (
                not ids
                or not ids <= set(current["selected_ids"])
                or not ids <= set(all_ids)
                or ids & set(accepted_ids)
                or ids & selected
                or allowed[0]["old_response_state_unchanged"] is not True
            ):
                raise ValueError("generic_reassessment_selected_scope")
            found.add(request_id)
            selected.update(ids)
            proofs.append(
                dict(
                    history=history,
                    run_id=documents.run_id,
                    request_id=request_id,
                    inventory_checksum=digest(current),
                    assessment_ids=sorted(ids),
                    journal_checksum=current["journal_checksum"],
                    control_checksum=current["control_checksum"],
                    old_response_state_unchanged=True,
                )
            )
    if found != set(acknowledgments) or not selected:
        raise ValueError("generic_reassessment_unmatched_request")
    return sorted(selected), proofs


def accepted_snapshot(campaign, result):
    """Read the original acceptance and current source projection, not an export label."""
    from agents.expert_agent.reporting import read_research_report
    from packages.evidence_details.acceptance import ResearchAcceptance

    final = result["phases"][-1]
    docs = campaign.documents(final["run_id"])
    phase = campaign.phase_class(
        campaign.base,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    state, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    phase.verify(state)
    saved = docs.read("dynamic-case-report")
    if (
        docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),)
        or docs.read("dynamic-phase-result") != (final,)
        or not saved
        or saved[0]["result"] != result
        or saved[1]["report_checksum"] != digest(saved[0])
    ):
        raise ValueError("generic_reassessment_accepted_binding")
    report = read_research_report(
        ResearchAcceptance(
            owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
        ),
        key=final["acceptance_key"],
        parent=state,
    )
    if report.get("scores") != saved[0].get("scores") or report["receipt"]["status"] != "accepted":
        raise ValueError("generic_reassessment_accepted_score_binding")
    events = RuntimeAgentJournal(docs).read()
    reservations = sum(e.get("phase") == "reserved" for e in events)
    if reservations != result["new_requests"]:
        raise ValueError("generic_reassessment_accepted_accounting")
    return SimpleNamespace(
        docs=docs,
        state=state,
        report=saved,
        result=result,
        ids=sorted(r["id"] for r in report["scores"]["reviews"]),
        reservations=reservations,
        proof=dict(
            run_id=docs.run_id,
            result=digest(result),
            report=digest(saved),
            state=digest(state),
            acceptance_key=final["acceptance_key"],
            acceptance=digest(docs.read("accepted:" + final["acceptance_key"])),
            journal=digest(events),
            control=digest(docs.sessions.read(docs.run_id)),
        ),
    )


def origin(accepted_key, acknowledgments):
    if (directory(accepted_key) / "cancel-request.json").exists():
        raise ValueError("generic_reassessment_history_cancelled")
    selection = read(directory(accepted_key) / "input.json")
    if selection.get("schema_version") != followup.VERSION or selection.get("key") != accepted_key:
        raise ValueError("generic_reassessment_requires_bound_accepted_followup")
    accepted, result = followup.restore(directory(accepted_key))
    checked = accepted.chain
    snapshot = accepted_snapshot(accepted, result)
    root_docs, root_state, _, _ = pending.original_phase(checked.campaign, require_accepted=False)
    histories = (
        ("root", root_docs, root_state),
        ("failed_successor", checked.child.documents, checked.child.state),
    )
    selected, proofs = acknowledged_selection(
        histories,
        acknowledgments,
        accepted_ids=snapshot.ids,
        all_ids=[i.id for i in root_state.policy.items],
        config=checked.campaign.config,
    )
    previous = checked.proof["previous_requests"] + snapshot.reservations
    if result["prior_graph_requests"] != checked.proof["previous_requests"]:
        raise ValueError("generic_reassessment_prior_accounting")
    maximum = min(10, 40 - previous)
    if (
        maximum
        < (len(selected) + accepted.config.review_batching.batch_size - 1)
        // accepted.config.review_batching.batch_size
    ):
        raise ValueError("generic_reassessment_cumulative_budget")
    if accepted.config.transport.max_retries != 0:
        raise ValueError("generic_reassessment_no_automatic_retry")
    proof = dict(
        schema_version="generic-chain-reassessment-proof/1.0.0",
        parent_key=selection["parent_key"],
        prior_review_key=selection["prior_review_key"],
        accepted_review_key=accepted_key,
        accepted_snapshot=snapshot.proof,
        previous_requests=previous,
        max_new_requests=maximum,
        original_followup=checked.proof,
        accepted_selection=digest(selection),
        acknowledge_requests=sorted(acknowledgments),
        owner_proofs=proofs,
        selected_ids=selected,
        inventory=dict(accepted_ids=snapshot.ids),
        provider_checksum=digest(accepted.config),
        selection_kind="explicit_new_reassessment_old_unknown_unchanged",
        new_vision_requests=0,
        max_cumulative_requests=40,
    )
    return SimpleNamespace(
        origin=checked, accepted=accepted, snapshot=snapshot, selection=selection, proof=proof
    )


class ReassessmentReview(followup.FollowupReview):
    def __init__(self, checked, key):
        self.reassessment = checked
        super().__init__(checked.origin, key)
        self.pending_review_proof = checked.proof
        self.phase_class = pending.phase_type(lambda: self.pending_review_proof)
        self.version = "generic-chain-reassessment-v1:" + key
        self.max_new_requests = checked.proof["max_new_requests"]

    def prior(self):
        prior = super().prior()
        checked, snapshot = self.reassessment, self.reassessment.snapshot
        docs = snapshot.docs
        if (
            prior["reservations"] + snapshot.reservations != checked.proof["previous_requests"]
            or docs.read("dynamic-case-result") != (snapshot.result,)
            or docs.read("dynamic-case-report") != snapshot.report
            or digest(RuntimeAgentJournal(docs).read()) != snapshot.proof["journal"]
            or digest(self.sessions.read(docs.run_id)) != snapshot.proof["control"]
            or digest(docs.read("accepted:" + snapshot.proof["acceptance_key"]))
            != snapshot.proof["acceptance"]
            or read(directory(checked.proof["accepted_review_key"]) / "input.json")
            != checked.selection
            or read(directory(checked.proof["prior_review_key"]) / "review-successor.json")
            != {k: v for k, v in checked.selection.items() if k != "key"}
            or read(directory(checked.proof["accepted_review_key"]) / "binding.json")
            != read(directory(checked.proof["parent_key"]) / "binding.json")
        ):
            raise ValueError("generic_reassessment_accepted_history_changed")
        return dict(
            reservations=checked.proof["previous_requests"],
            original=prior,
            reassessment_proof=checked.proof,
        )

    def before_review(self):
        proof = self.reassessment.proof
        if origin(proof["accepted_review_key"], proof["acknowledge_requests"]).proof != proof:
            raise ValueError("generic_reassessment_current_history_changed")

    def guard(self, *, dispatch=False):
        from packages.llm_provider.provider import ProviderError

        if (
            directory(self.reassessment.proof["accepted_review_key"]) / "cancel-request.json"
        ).exists():
            raise ProviderError("cancelled")
        return super().guard(dispatch=dispatch)

    def cancellation_signal(self):
        signal = super().cancellation_signal()
        original = signal.is_set
        previous = directory(self.reassessment.proof["accepted_review_key"]) / "cancel-request.json"
        signal.is_set = lambda: original() or previous.exists()
        return signal


def selection_for(checked, review_key):
    p = checked.proof
    return dict(
        schema_version=VERSION,
        review_profile=PROFILE,
        review_key=review_key,
        parent_key=p["parent_key"],
        prior_review_key=p["prior_review_key"],
        accepted_review_key=p["accepted_review_key"],
        proof=p,
        input_checksum=digest(checked.origin.campaign.spec),
        provider_checksum=digest(checked.accepted.config),
        max_new_requests=p["max_new_requests"],
        max_cumulative_requests=40,
        seconds=2700,
        new_vision_requests=0,
    )


def aggregation_history(campaign):
    """Exact ancestral journals/results, including the failed middle attempt.

    Used for both a bound 1.5 followup and a bound 1.6 reassessment. The current
    reader appends its own attempt; aggregation deduplicates identical ancestors.
    """
    if not isinstance(campaign, followup.FollowupReview):
        raise ValueError("generic_reassessment_accounting_campaign")
    prior = campaign.prior()
    original = campaign.acquisition
    root = read(original.output / "result.json")
    if original.documents(root["phases"][-1]["run_id"]).read("dynamic-case-result") != (root,):
        raise ValueError("generic_reassessment_accounting_root")
    chain = campaign.chain
    rows = [
        dict(
            directory=str(original.output.parent),
            checksum=digest(root),
            requests=root["new_requests"],
        ),
        dict(
            directory=str(directory(chain.proof["prior_review_key"])),
            checksum=digest(chain.child.result),
            requests=chain.child.reserved,
        ),
    ]
    if isinstance(campaign, ReassessmentReview):
        snapshot = campaign.reassessment.snapshot
        rows.append(
            dict(
                directory=str(directory(campaign.reassessment.proof["accepted_review_key"])),
                checksum=digest(snapshot.result),
                requests=snapshot.reservations,
            )
        )
    from .generic_review_continuation import ContinuationReview

    if isinstance(campaign, ContinuationReview):
        rows.extend(
            dict(
                directory=str(directory(s.key)), checksum=digest(s.result), requests=s.reservations
            )
            for s in campaign.continuation.history
        )
    if (
        len({v["directory"] for v in rows}) != len(rows)
        or sum(v["requests"] for v in rows) != prior["reservations"]
    ):
        raise ValueError("generic_reassessment_accounting_ancestry")
    return rows


def run(key, review_key, accepted_review_key, acknowledge_requests, *, preflight=False):
    import fcntl

    from packages.evaluation_preparation.generic_control import start
    from packages.evaluation_preparation.generic_run import finish_export, persist_export
    from packages.llm_provider.network_route import route_opener

    root, accepted_path, out = directory(key), directory(accepted_review_key), directory(review_key)
    with (root / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = origin(accepted_review_key, acknowledge_requests)
        if (
            key != checked.proof["parent_key"]
            or len({key, review_key, accepted_review_key, checked.proof["prior_review_key"]}) != 4
        ):
            raise ValueError("generic_reassessment_distinct_chain_keys")
        route_opener(
            checked.accepted.config.network_route, checked.accepted.config.transport.base_url
        )
        selection = selection_for(checked, review_key)
        claim = accepted_path / "review-successor.json"
        if claim.exists() and read(claim) != selection:
            raise ValueError("generic_reassessment_successor_conflict")
        if (out / "input.json").exists() and read(out / "input.json") != dict(
            selection, key=review_key
        ):
            raise ValueError("generic_reassessment_output_identity_changed")
        if (
            (out / "started.json").exists()
            and not (out / "export-reference.json").exists()
            and not (out / "evidence/result.json").exists()
        ):
            raise ValueError("generic_reassessment_claimed_use_status_or_saved_export")
        if preflight:
            return dict(
                status="saved_export_only" if (out / "started.json").exists() else "ready",
                selection=selection,
                api_requests=0,
                database_writes=0,
            )
        campaign = ReassessmentReview(checked, review_key)
        write_once(accepted_path / "review-successor.json", selection)
        write_once(out / "input.json", dict(selection, key=review_key))
        write_once(out / "binding.json", read(root / "binding.json"))
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(review_key))
        if (out / "started.json").exists():
            if (out / "evidence/result.json").exists():
                saved, result = restore(out)
                return persist_export(review_key, saved, result)
            raise ValueError("generic_reassessment_claimed_use_status_or_saved_export")
        start(dict(selection, key=review_key))
        return persist_export(review_key, campaign, campaign.execute())


def restore(path):
    selection = read(path / "input.json")
    checked = origin(selection["accepted_review_key"], selection["proof"]["acknowledge_requests"])
    if (
        selection != dict(selection_for(checked, path.name), key=path.name)
        or read(directory(selection["accepted_review_key"]) / "review-successor.json")
        != {k: v for k, v in selection.items() if k != "key"}
        or read(path / "binding.json") != read(directory(selection["parent_key"]) / "binding.json")
    ):
        raise ValueError("generic_reassessment_saved_selection_changed")
    return pending.confirmed_campaign(ReassessmentReview(checked, path.name))

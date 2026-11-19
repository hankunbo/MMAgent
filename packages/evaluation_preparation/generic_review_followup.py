"""One route-only, unstarted-work followup of a failed generic Review successor.

Original owners rebuild both histories. This is a bounded GenericReview variant,
not a replay permission for UNKNOWN batches or a new acquisition engine.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from packages.evidence_details.live_permission import validate_live_config
from packages.evidence_details.pending_review import read_failed_review_attempt
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config
from packages.llm_provider.network_route import read_route, route_opener
from packages.research_runtime.agent_records import RuntimeAgentJournal

from . import generic_pending_review as pending
from .generic_input import directory, load
from .generic_review import GenericReview

VERSION = "generic-review-selection/1.5.0"
PROFILE = "unstarted-route-followup-v1"
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_bytes())


def configuration(origin, route_path):
    path = (ROOT / route_path).resolve()
    if not path.is_relative_to(ROOT / "configs/agents") or path.suffix != ".json":
        raise ValueError("generic_followup_public_route_required")
    role = next(r for r in origin.roles if r.role == "review")
    if (
        origin.schema_version != "agent-provider-config/7.1.0"
        or role.prompt_version != "review-batch-1.9.0"
        or origin.transport.max_retries not in (0, 1)
        or origin.review_batching.batch_size != 3
    ):
        raise ValueError("generic_followup_original_provider")
    route = read_route(read(path))
    if route == origin.network_route:
        raise ValueError("generic_followup_requires_explicit_route_change")
    raw = origin.model_dump(mode="json")
    # Narrow the legacy one-retry slot; never replay UNKNOWN after a route change.
    raw["transport"]["max_retries"] = 0
    raw["network_route"] = route.model_dump(mode="json")
    return validate_live_config(read_provider_config(raw))


def stopped_successor(campaign, prior_key, original_proof):
    """Bind the exact original child, reproject it, and ask the same inventory owner."""
    root, child = directory(campaign.spec["key"]), directory(prior_key)
    if root == child:
        raise ValueError("generic_followup_distinct_history")
    claim, selection = read(root / "review-successor.json"), read(child / "input.json")
    if (
        claim.get("schema_version") != "generic-review-selection/1.4.0"
        or claim.get("parent_key") != campaign.spec["key"]
        or claim.get("review_key") != prior_key
        or claim.get("review_profile") != pending.UNSTARTED_PROFILE
        or claim.get("input_checksum") != digest(campaign.spec)
        or claim.get("provider_checksum") != digest(campaign.config)
        or claim.get("origin_provider_checksum") != digest(campaign.config)
        or claim.get("provider_derivation") != "bound-provider-unchanged/1.0.0"
        or claim.get("eligibility_checksum") != digest(original_proof)
        or claim.get("selected_ids") != original_proof["selected_ids"]
        or selection != dict(claim, key=prior_key)
        or read(child / "binding.json") != read(root / "binding.json")
    ):
        raise ValueError("generic_followup_original_child_binding")
    if (root / "cancel-request.json").exists() or (child / "cancel-request.json").exists():
        raise ValueError("generic_followup_history_cancelled")
    result = read(child / "evidence/result.json")
    if len(result["phases"]) != 1:
        raise ValueError("generic_followup_review_only_history")
    final = result["phases"][0]
    if (
        final.get("target") is not None
        or final.get("accepted") is not False
        or final.get("acceptance_key") is not None
    ):
        raise ValueError("generic_followup_requires_unaccepted_child")
    docs = campaign.documents(final["run_id"])
    request = read(child / "evidence/review/request.json")
    context = campaign.contexts.read_research_run(docs.run_id)
    protocol = read(child / "evidence/protocol.json")
    saved = docs.read("dynamic-case-report")
    if (
        context is None
        or context.request.model_dump(mode="json") != request
        or request["lock"]["run_id"] != docs.run_id
        or docs.read("dynamic-case-result") != (result,)
        or docs.read("dynamic-phase-result") != (final,)
        or docs.read("dynamic-campaign") != (protocol,)
        or protocol["input"] != digest(campaign.spec)
        or protocol["provider_checksum"] != digest(campaign.config)
        or protocol["observations"]["pending_review"] != original_proof
        or not saved
        or saved[0]["result"] != result
        or saved[1]["report_checksum"] != digest(saved[0])
    ):
        raise ValueError("generic_followup_child_result_binding")
    phase = pending.phase_type(lambda: original_proof)(
        campaign.base,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    state, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    phase.verify(state)
    if docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),):
        raise ValueError("generic_followup_child_current_material_changed")
    inventory = read_failed_review_attempt(documents=docs, state=state, config=campaign.config)
    events = RuntimeAgentJournal(docs).read()
    reserved = sum(e.get("phase") == "reserved" for e in events)
    if result["new_requests"] != reserved or result["prior_graph_requests"] != campaign.usage():
        raise ValueError("generic_followup_request_accounting")
    return SimpleNamespace(
        documents=docs,
        inventory=inventory,
        result=result,
        report=saved,
        protocol=protocol,
        claim=claim,
        selection=selection,
        reserved=reserved,
        journal_checksum=digest(events),
        state=state,
    )


def origin(key, prior_key, route_path):
    from .case_execution import owners
    from .generic_campaign import GenericCampaign

    spec = load(key)
    runtime, store, _ = owners()
    root = directory(key)
    registered = (root, read(root / "binding.json"), runtime, store)
    campaign = GenericCampaign(spec, registered)
    config = configuration(campaign.config, route_path)
    original = pending.unstarted_selection(campaign)
    child = stopped_successor(campaign, prior_key, original)
    inventory = child.inventory
    quarantine = original["inventory"]["quarantined_batches"] + inventory["quarantined_batches"]
    blocked = {i for b in quarantine for i in b["assessment_ids"]}
    ids = inventory["selected_ids"]
    if (
        not ids
        or set(ids) & blocked
        or not set(ids) < set(original["selected_ids"])
        or any(b["new_reassessment_authorized"] for b in quarantine)
    ):
        raise ValueError("generic_followup_no_unstarted_work")
    previous = campaign.usage() + child.reserved
    cap = min(10, 40 - previous)
    if cap < (len(ids) + 2) // 3 or cap <= 0:
        raise ValueError("generic_followup_cumulative_budget")
    proof = dict(
        schema_version="generic-unstarted-followup-proof/1.0.0",
        parent_key=key,
        prior_review_key=prior_key,
        original=original,
        inventory=dict(inventory, accepted_ids=[], quarantined_batches=quarantine),
        selected_ids=ids,
        prior_result=digest(child.result),
        prior_report=digest(child.report),
        prior_selection=digest(child.selection),
        prior_protocol=digest(child.protocol),
        previous_requests=previous,
        max_new_requests=cap,
        provider_checksum=digest(config),
        original_provider_checksum=digest(campaign.config),
        provider_derivation="route-change-zero-retry/1.0.0",
        route_path=str((ROOT / route_path).resolve().relative_to(ROOT)),
        route_checksum=digest(config.network_route),
        new_vision_requests=0,
        selection_kind="unstarted_only_after_bound_failed_successor",
    )
    return SimpleNamespace(
        campaign=campaign, registered=registered, child=child, config=config, proof=proof
    )


class FollowupReview(GenericReview):
    def __init__(self, checked, key):
        self.chain = checked
        super().__init__(checked.campaign.spec, checked.registered, key)
        self.config = checked.config
        self.pending_review_proof = checked.proof
        self.phase_class = pending.phase_type(lambda: self.pending_review_proof)
        self.version = "generic-review-followup-v1:" + key
        self.max_new_requests = checked.proof["max_new_requests"]
        self.duration_seconds = 2700

    def prior(self):
        original = GenericReview.prior(self)
        child = self.chain.child
        docs = child.documents
        ancestor = self.chain.proof["original"]["inventory"]
        root_docs = self.documents(ancestor["run_id"])
        if (
            original["reservations"] + child.reserved != self.chain.proof["previous_requests"]
            or digest(RuntimeAgentJournal(root_docs).read()) != ancestor["journal_checksum"]
            or digest(self.sessions.read(root_docs.run_id)) != ancestor["control_checksum"]
            or digest(RuntimeAgentJournal(docs).read()) != child.journal_checksum
            or digest(self.sessions.read(docs.run_id)) != child.inventory["control_checksum"]
            or docs.read("dynamic-case-result") != (child.result,)
            or docs.read("dynamic-case-report") != child.report
            or read(directory(self.spec["key"]) / "review-successor.json") != child.claim
            or read(directory(self.chain.proof["prior_review_key"]) / "input.json")
            != child.selection
        ):
            raise ValueError("generic_followup_ancestor_changed")
        return dict(
            reservations=self.chain.proof["previous_requests"],
            original=original,
            followup_proof=self.chain.proof,
        )

    def before_review(self):
        p = self.chain.proof
        if origin(p["parent_key"], p["prior_review_key"], p["route_path"]).proof != p:
            raise ValueError("generic_followup_current_origin_changed")

    def guard(self, *, dispatch=False):
        from packages.llm_provider.provider import ProviderError

        if (directory(self.chain.proof["prior_review_key"]) / "cancel-request.json").exists():
            raise ProviderError("cancelled")
        super().guard(dispatch=dispatch)

    def cancellation_signal(self):
        signal = super().cancellation_signal()
        original = signal.is_set
        prior = directory(self.chain.proof["prior_review_key"]) / "cancel-request.json"
        signal.is_set = lambda: original() or prior.exists()
        return signal


def selection_for(checked, review_key):
    proof = checked.proof
    return dict(
        schema_version=VERSION,
        parent_key=proof["parent_key"],
        prior_review_key=proof["prior_review_key"],
        review_key=review_key,
        review_profile=PROFILE,
        input_checksum=digest(checked.campaign.spec),
        provider_checksum=digest(checked.config),
        network_route=proof["route_path"],
        proof=proof,
        max_new_requests=proof["max_new_requests"],
        max_cumulative_requests=40,
        seconds=2700,
        new_vision_requests=0,
    )


def run(key, review_key, prior_review_key, network_route, *, preflight=False):
    import fcntl

    from .generic_control import start
    from .generic_run import finish_export, persist_export

    if len({key, review_key, prior_review_key}) != 3:
        raise ValueError("generic_followup_distinct_keys")
    root, prior, out = directory(key), directory(prior_review_key), directory(review_key)
    # Root lock serializes sibling branches without replacing the original one-child claim.
    with (root / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = origin(key, prior_review_key, network_route)
        route_opener(checked.config.network_route, checked.config.transport.base_url)
        selection = selection_for(checked, review_key)
        if preflight:
            return dict(status="ready", selection=selection, api_requests=0, database_writes=0)
        campaign = FollowupReview(checked, review_key)
        write_once(prior / "review-successor.json", selection)
        write_once(out / "input.json", dict(selection, key=review_key))
        write_once(out / "binding.json", read(root / "binding.json"))
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(review_key))
        # A claimed attempt is not permission to start another request after a crash.
        if (out / "started.json").exists():
            raise ValueError("generic_followup_claimed_use_status_or_saved_export")
        start(dict(selection, key=review_key))
        return persist_export(review_key, campaign, campaign.execute())


def restore(path):
    """Current-owner readback for the original aggregation/export callers; zero API."""
    selection = read(path / "input.json")
    checked = origin(
        selection["parent_key"], selection["prior_review_key"], selection["network_route"]
    )
    if (
        selection != dict(selection_for(checked, path.name), key=path.name)
        or read(directory(selection["prior_review_key"]) / "review-successor.json")
        != {k: v for k, v in selection.items() if k != "key"}
        or read(path / "binding.json") != read(directory(selection["parent_key"]) / "binding.json")
    ):
        raise ValueError("generic_followup_saved_selection_changed")
    return pending.confirmed_campaign(FollowupReview(checked, path.name))

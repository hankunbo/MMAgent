"""One exact failed Vision's new attempt, then selected Review through original owners.

Historical UNKNOWN stays unchanged. This adapter does not run models/tools,
replay old requests, renew old grants, or score historical judgments again.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

from . import generic_pending_review as pending
from .generic_aliases import GenericPhase
from .generic_campaign import GenericCampaign
from .generic_input import directory, load
from .generic_review import GenericReview

VERSION = "generic-acquisition-successor/1.0.0"
PROFILE = "exact-unknown-vision-and-selected-review/1.0.0"
CHAIN_VERSION = "generic-acquisition-successor/2.0.0"
CHAIN_PROFILE = "failed-vision-client-followup/1.0.0"
RECOVERY_VERSION = "generic-acquisition-successor/3.0.0"
RECOVERY_PROFILE = "verified-failed-client-new-attempt/1.0.0"


def implementation():
    from packages.object_store import checksum

    files = (
        "packages/evaluation_preparation/generic_acquisition_successor.py",
        "packages/evaluation_preparation/generic_pending_review.py",
        "packages/multimodal_research/unknown_acquisition.py",
        "packages/evidence_details/successor_owner.py",
    )
    return dict(pending.implementation(), **{p: checksum(Path(p).read_bytes()) for p in files})


def read(path):
    return json.loads(Path(path).read_bytes())


def origin(key, target, request_id):
    """Current projection plus exact old intent/grant/journal, never an error heuristic."""
    from agents.expert_agent.reporting import read_research_report
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.acceptance_models import DomainState
    from packages.grounding.models import validate
    from packages.multimodal_research.unknown_acquisition import original_attempt

    from .case_execution import owners

    spec = load(key)
    root = directory(key)
    if (root / "cancel-request.json").exists():
        raise ValueError("generic_acquisition_parent_cancelled")
    binding = read(root / "binding.json")
    runtime, objects, _ = owners()
    campaign = GenericCampaign(spec, (root, binding, runtime, objects))
    if target not in campaign.visual_targets or campaign.config.transport.max_retries != 0:
        raise ValueError("generic_acquisition_target_or_retry_scope")
    old = read(campaign.output / "result.json")
    docs, state, _, final = pending.original_phase(campaign)
    accepted = read_research_report(
        ResearchAcceptance(
            owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
        ),
        key=final["acceptance_key"],
        parent=state,
    )
    saved = docs.read("dynamic-case-report")
    accepted_ids = sorted(r["id"] for r in accepted["scores"]["reviews"])
    if saved[0].get("scores") != accepted["scores"] or target in accepted_ids:
        raise ValueError("generic_acquisition_original_score_or_already_reviewed")
    observations = campaign.accepted_observations()
    if target in observations:
        raise ValueError("generic_acquisition_already_observed")
    failed = read(campaign.phase_path(target) / "result.json")
    failed_docs = campaign.documents(failed["run_id"])
    projected = GenericPhase(campaign.base, target=target)
    current, _ = projected.project(run_id=failed["run_id"], config=campaign.config)
    projected.verify(current)
    if validate(DomainState, failed_docs.read("dynamic-phase-input")[0]) != current:
        raise ValueError("generic_acquisition_current_material_changed")
    adapter = SimpleNamespace(
        documents=campaign.documents,
        sessions=campaign.sessions,
        config=campaign.config,
        base=SimpleNamespace(
            binding=campaign.base.binding,
            manifest=SimpleNamespace(checksum=current.material.scope.input_checksum),
            task=SimpleNamespace(target_code=current.material.goal_code),
        ),
    )
    attempt = original_attempt(adapter, failed, request_id)
    journals = []
    for path in campaign.paths():
        if not (path / "request.json").exists():
            continue
        run_id = read(path / "request.json")["lock"]["run_id"]
        control = campaign.sessions.read(run_id)
        if control is None or control.phase != "stopped" or control.pending or control.cancelled:
            raise ValueError("generic_acquisition_parent_not_closed")
        events = RuntimeAgentJournal(campaign.documents(run_id)).read()
        journals.append(
            dict(
                run_id=run_id,
                checksum=digest(events),
                requests=sum(e.get("phase") == "reserved" for e in events),
            )
        )
    previous = sum(j["requests"] for j in journals)
    if previous != campaign.usage() or previous != old["new_requests"] or previous + 2 > 40:
        raise ValueError("generic_acquisition_cumulative_budget")
    proof = dict(
        schema_version=VERSION,
        parent_key=key,
        target=target,
        request_id=request_id,
        input_checksum=digest(spec),
        original_result=digest(old),
        original_report=digest(saved),
        original_acceptance=digest(accepted),
        original_observations=digest(observations),
        original_attempt=attempt,
        journals=journals,
        previous_requests=previous,
        provider_checksum=digest(campaign.config),
        selected_ids=[target],
        inventory=dict(accepted_ids=accepted_ids),
        max_new_requests=2,
        selection_kind="new_vision_and_review_old_unknown_unchanged",
    )
    return SimpleNamespace(campaign=campaign, proof=proof, binding=binding)


class AcquisitionSuccessor(GenericReview):
    def __init__(self, checked, key):
        # Reuse the assembled original owners without rerunning tool preparation.
        self.__dict__.update(vars(checked.campaign))
        self.checked = checked
        self.acquisition = checked.campaign
        self.review_key = key
        self.selected_target = checked.proof["target"]
        self.pending_review_proof = checked.proof
        self.output = directory(key) / "evidence"
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.version = VERSION + ":" + key
        self.visual_targets = (self.selected_target,)
        self.max_new_requests, self.max_vision_requests = 2, 1
        self.duration_seconds = 2700
        selected = self.selected_target

        class SelectedPhase(GenericPhase):
            def project_current(inner):
                state = super().project_current()
                return (
                    state if inner.target is not None else pending.selected_state(state, [selected])
                )

        self.phase_class = SelectedPhase

    def prior(self):
        c, p = self.acquisition, self.checked.proof
        old = read(c.output / "result.json")
        docs = c.documents(old["phases"][-1]["run_id"])
        if (
            digest(old) != p["original_result"]
            or docs.read("dynamic-case-result") != (old,)
            or digest(docs.read("dynamic-case-report")) != p["original_report"]
            or c.usage() != p["previous_requests"]
        ):
            raise ValueError("generic_acquisition_parent_history_changed")
        return dict(reservations=p["previous_requests"], acquisition_proof=p)

    def paths(self):
        return [self.phase_path(self.selected_target), self.phase_path(None)]

    def phase_path(self, target):
        if target not in (None, self.selected_target):
            raise ValueError("generic_acquisition_unselected_phase")
        return self.output / ("vision" if target is not None else "review")

    def request_key(self, target):
        self.phase_path(target)
        return (
            "fv:acquisition-successor:"
            + digest(dict(parent=self.spec["key"], successor=self.review_key, target=target))[7:]
        )

    def accepted_observations(self):
        from .comprehensive_campaign import ComprehensiveCampaign

        old = self.acquisition.accepted_observations()
        fresh = ComprehensiveCampaign.accepted_observations(self)
        if set(old) & set(fresh):
            raise ValueError("generic_acquisition_duplicate_observation")
        return dict(old, **fresh)

    def freeze_observations(self):
        return dict(
            original=self.acquisition.freeze_observations(),
            proof=self.checked.proof,
            new_target=self.selected_target,
            successor_implementation=implementation(),
        )

    def before_review(self):
        p = self.checked.proof
        if origin(p["parent_key"], p["target"], p["request_id"]).proof != p:
            raise ValueError("generic_acquisition_origin_changed_before_dispatch")
        self.execute_phase(self.selected_target)

    def execute_phase(self, target=None):
        if target is None:
            acquired = read(self.phase_path(self.selected_target) / "result.json")
            if not acquired["accepted"]:
                # Preserve the original failed phase and stop; no empty Review or retry.
                return acquired
            from .partial_observation import evaluate

            observation = self.accepted_observations().get(self.selected_target)
            if observation is None:
                raise ValueError("generic_acquisition_accepted_observation_missing")
            if not evaluate(observation)["availability_score"]:
                # An accepted Vision response is not necessarily an interpretable
                # finding. Preserve it without asking an empty Review or changing
                # the original response/acceptance to force a direction.
                return acquired
        return super().execute_phase(target)

    def phase_completed(self, phase):
        return phase.get("target") is None and super().phase_completed(phase)

    def export_result(self, result):
        phase = result["phases"][-1]
        if phase.get("target") is None:
            return super().export_result(result)
        from html import escape

        from packages.grounding.models import canonical

        docs = self.documents(phase["run_id"])
        saved = docs.read("dynamic-case-report")
        if saved:
            if saved[0]["result"] != result or saved[1]["report_checksum"] != digest(saved[0]):
                raise ValueError("generic_acquisition_saved_report_binding")
            return saved[0]
        observation = self.accepted_observations().get(self.selected_target)
        reason = (
            "accepted_observation_not_interpretable_for_selected_requirement"
            if phase["accepted"]
            else "acquisition_not_accepted"
        )
        report = json.loads(
            canonical(
                dict(
                    schema_version="generic-acquisition-incomplete-report/1.0.0",
                    result=result,
                    scores=None,
                    selected_target=self.selected_target,
                    acquisition_phase=phase,
                    acquisition_observation=observation,
                    previous_accepted_ids=self.checked.proof["inventory"]["accepted_ids"],
                    completion=dict(
                        report_saved=True,
                        domain_accepted=False,
                        review_dispatched=False,
                        observation_domain_accepted=phase["accepted"],
                        reason=reason,
                    ),
                    clinical_diagnosis=False,
                    calibrated_probability=False,
                )
            )
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'></head><body>"
            "<h1>新增取证记录：尚未形成方向评阅</h1>"
            "<p>原已接纳评阅保持；本页没有新增疾病评分。</p><pre>"
            + escape(json.dumps(report, ensure_ascii=False, indent=2))
            + "</pre></body></html>"
        )
        with docs.session():
            docs.write(
                "dynamic-case-report", (report, dict(html=html, report_checksum=digest(report)))
            )
        return report

    @contextmanager
    def execution_session(self):
        from db.runtime.repository.case_pipeline import CasePipelineRepository

        key = "generic-acquisition-execution:" + digest([self.spec["key"], self.review_key])[7:]
        with CasePipelineRepository(self.target).guarded_session(
            self.base.binding["case_code"], key
        ) as guard:
            self._execution_guard = guard
            try:
                guard()
                yield
            finally:
                self._execution_guard = None

    def guard(self, *, dispatch=False):
        guard = getattr(self, "_execution_guard", None)
        parent_guard = getattr(self, "_parent_execution_guard", None)
        if guard is None or parent_guard is None:
            raise ValueError("generic_acquisition_original_lease_required")
        parent_guard()
        guard()
        return super().guard(dispatch=dispatch)


def failed_lineage(proof):
    """At most two exact failed successors; no arbitrary retry chain."""
    rows, current = [], proof
    for _ in range(2):
        child = current.get("failed_successor")
        if child is None:
            break
        rows.append(child)
        current = current.get("original", {})
    if current.get("failed_successor") is not None:
        raise ValueError("generic_acquisition_failed_chain_bound")
    keys = [proof.get("parent_key"), *(r["key"] for r in rows)]
    if len(keys) != len(set(keys)):
        raise ValueError("generic_acquisition_failed_chain_cycle")
    return tuple(reversed(rows))


def client_implementation(proof):
    from packages.object_store import checksum

    paths = [
        proof["client_binding"]["path"],
        "packages/llm_provider/client_selection.py",
        "packages/llm_provider/curl_client.py",
    ]
    return {name: checksum(Path(name).read_bytes()) for name in paths}


def persisted_failed_protocol(protocol, proof, *, prior_chain):
    """v2 writer appended exactly three client hashes after its local write.

    Preserve both original records. Only that exact historical representation is
    recognized, with current byte verification; no other mismatch is normalized.
    v3 freezes these values inside observations before either owner writes.
    """
    if not prior_chain:
        return protocol
    extension = client_implementation(proof)
    if set(extension) & set(protocol):
        raise ValueError("generic_acquisition_legacy_protocol_collision")
    return dict(protocol, **extension)


def failed_successor(checked, previous, request_id):
    """Rebuild one exact failed child using the original Runtime and request owners."""
    from packages.evidence_details.acceptance_models import DomainState
    from packages.grounding.models import validate
    from packages.multimodal_research.unknown_acquisition import original_attempt

    c, original = checked.campaign, checked.proof
    root, child = directory(original["parent_key"]), directory(previous)
    child_config = getattr(checked, "config", c.config)
    prior_chain = original.get("schema_version") == CHAIN_VERSION
    claim_parent = directory(original["failed_successor"]["key"]) if prior_chain else root
    allowed_version = CHAIN_VERSION if prior_chain else VERSION
    selection = read(child / "input.json")
    expected = selection_for(checked, previous)
    expected["implementation"] = selection.get("implementation")
    if (
        selection != expected
        or not isinstance(selection.get("implementation"), dict)
        or not selection["implementation"]
        or child == root
        or selection != read(claim_parent / "acquisition-successor.json")
        or selection.get("schema_version") != allowed_version
        or selection.get("key") != previous
        or selection.get("parent_key") != original["parent_key"]
        or selection.get("proof") != original
        or read(child / "binding.json") != checked.binding
    ):
        raise ValueError("generic_acquisition_failed_child_binding")
    if (child / "cancel-request.json").exists():
        raise ValueError("generic_acquisition_failed_child_cancelled")
    result = read(child / "evidence/result.json")
    phases = result.get("phases", [])
    if result.get("complete") is not False or len(phases) != 1:
        raise ValueError("generic_acquisition_failed_child_not_single_failed_vision")
    phase = phases[0]
    if phase.get("target") != original["target"] or phase.get("accepted") is not False:
        raise ValueError("generic_acquisition_failed_child_not_single_failed_vision")
    docs = c.documents(phase["run_id"])
    request = read(child / "evidence/vision/request.json")
    context = c.contexts.read_research_run(docs.run_id)
    protocol = read(child / "evidence/protocol.json")
    saved = docs.read("dynamic-case-report")
    persisted_protocol = persisted_failed_protocol(protocol, original, prior_chain=prior_chain)
    if (
        context is None
        or context.request.model_dump(mode="json") != request
        or request["lock"]["run_id"] != docs.run_id
        or docs.read("dynamic-case-result") != (result,)
        or docs.read("dynamic-phase-result") != (phase,)
        or docs.read("dynamic-campaign") != (persisted_protocol,)
        or protocol["input"] != digest(c.spec)
        or protocol["provider_checksum"] != digest(child_config)
        or protocol["observations"].get("proof") != original
        or not saved
        or saved[0]["result"] != result
        or saved[1]["report_checksum"] != digest(saved[0])
    ):
        raise ValueError("generic_acquisition_failed_child_result_binding")
    projection = GenericPhase(c.base, target=original["target"])
    state, _ = projection.project(run_id=docs.run_id, config=child_config)
    projection.verify(state)
    if state != validate(DomainState, docs.read("dynamic-phase-input")[0]):
        raise ValueError("generic_acquisition_failed_child_current_material_changed")
    adapter = SimpleNamespace(
        documents=c.documents,
        sessions=c.sessions,
        config=child_config,
        base=SimpleNamespace(
            binding=c.base.binding,
            manifest=SimpleNamespace(checksum=state.material.scope.input_checksum),
            task=SimpleNamespace(target_code=state.material.goal_code),
        ),
    )
    attempt = original_attempt(adapter, phase, request_id)
    journal = RuntimeAgentJournal(docs).read()
    reservations = sum(e.get("phase") == "reserved" for e in journal)
    if (
        reservations != 1
        or result["new_requests"] != reservations
        or result["prior_graph_requests"] != original["previous_requests"]
    ):
        raise ValueError("generic_acquisition_failed_child_budget")
    return dict(
        key=previous,
        request_id=request_id,
        selection=digest(selection),
        result=digest(result),
        report=digest(saved),
        protocol=digest(protocol),
        **({"persisted_protocol": digest(persisted_protocol)} if prior_chain else {}),
        journal=digest(journal),
        requests=reservations,
        attempt=attempt,
    )


def chained_origin(key, target, request_id, previous, previous_request, client_path):
    from packages.evidence_details.live_permission import validate_live_config
    from packages.llm_provider.client_selection import load_selection
    from packages.llm_provider.config import with_http_client

    previous_value = read(directory(previous) / "input.json")
    prior_version = previous_value.get("schema_version")
    if prior_version not in {VERSION, CHAIN_VERSION}:
        raise ValueError("generic_acquisition_failed_chain_bound")
    recovered = prior_version == CHAIN_VERSION
    if recovered:
        old = previous_value["proof"]
        first = old["failed_successor"]
        if (
            old["parent_key"] != key
            or old["target"] != target
            or old["request_id"] != request_id
            or first["key"] in {key, previous}
            or first["request_id"] in {request_id, previous_request}
            or read(directory(first["key"]) / "input.json").get("schema_version") != VERSION
        ):
            raise ValueError("generic_acquisition_failed_chain_identity")
        checked = chained_origin(
            key,
            target,
            request_id,
            first["key"],
            first["request_id"],
            old["client_binding"]["path"],
        )
    else:
        checked = origin(key, target, request_id)
    if previous == key or previous_request == request_id:
        raise ValueError("generic_acquisition_failed_child_distinct_identity")
    child = failed_successor(checked, previous, previous_request)
    binding = load_selection(client_path)
    if recovered and binding != checked.proof["client_binding"]:
        raise ValueError("generic_acquisition_recovery_client_changed")
    if checked.campaign.config.schema_version != "agent-provider-config/7.1.0":
        raise ValueError("generic_acquisition_client_original_version")
    config = validate_live_config(with_http_client(checked.campaign.config, binding["client"]))
    prior = checked.proof["previous_requests"] + child["requests"]
    if prior + 2 > 40:
        raise ValueError("generic_acquisition_cumulative_budget")
    proof = dict(
        checked.proof,
        schema_version=RECOVERY_VERSION if recovered else CHAIN_VERSION,
        original=checked.proof,
        previous_requests=prior,
        failed_successor=child,
        client_binding=binding,
        provider_checksum=digest(config),
        selection_kind=(
            "explicit_new_attempt_after_verified_failed_client_successor"
            if recovered
            else "explicit_new_attempt_after_failed_successor_client_only"
        ),
    )
    failed_lineage(proof)
    return SimpleNamespace(
        campaign=checked.campaign,
        binding=checked.binding,
        proof=proof,
        config=config,
        original=checked,
    )


class FailedAcquisitionSuccessor(AcquisitionSuccessor):
    """One explicit new attempt; both preceding UNKNOWN requests remain accounted."""

    def __init__(self, checked, key):
        super().__init__(checked, key)
        self.config = checked.config
        self.version = checked.proof["schema_version"] + ":" + key

    def _current(self):
        p = self.checked.proof
        current = chained_origin(
            p["parent_key"],
            p["target"],
            p["request_id"],
            p["failed_successor"]["key"],
            p["failed_successor"]["request_id"],
            p["client_binding"]["path"],
        )
        if current.proof != p:
            raise ValueError("generic_acquisition_failed_chain_changed")
        return current

    def prior(self):
        self._current()
        return dict(
            reservations=self.checked.proof["previous_requests"],
            acquisition_proof=self.checked.proof,
        )

    def before_review(self):
        self._current()
        self.execute_phase(self.selected_target)

    def guard(self, *, dispatch=False):
        from packages.llm_provider.client_selection import load_selection

        p = self.checked.proof
        if any(
            (directory(row["key"]) / "cancel-request.json").exists() for row in failed_lineage(p)
        ):
            raise ValueError("generic_acquisition_failed_child_cancelled")
        if load_selection(p["client_binding"]["path"]) != p["client_binding"]:
            raise ValueError("generic_acquisition_client_binding_changed")
        return super().guard(dispatch=dispatch)

    def cancellation_signal(self):
        inherited = super().cancellation_signal()
        from threading import Event

        children = [directory(row["key"]) for row in failed_lineage(self.checked.proof)]

        class ChainCancellation(Event):
            def is_set(inner):
                return (
                    super().is_set()
                    or inherited.is_set()
                    or any((child / "cancel-request.json").exists() for child in children)
                )

        return ChainCancellation()

    def freeze_observations(self):
        value = super().freeze_observations()
        if self.checked.proof["schema_version"] == RECOVERY_VERSION:
            value["client_implementation"] = client_implementation(self.checked.proof)
        return value

    def freeze(self):
        value = super().freeze()
        if self.checked.proof["schema_version"] == CHAIN_VERSION:
            # Historical v2 writer semantics. New recovery uses explicit v3.
            value.update(client_implementation(self.checked.proof))
        return value

    def original_accounting(self):
        """Only exact owner-verified saved results; includes the failed middle child."""
        self._current()
        p = self.checked.proof
        values = []
        for key in (p["parent_key"], *(row["key"] for row in failed_lineage(p))):
            result = read(directory(key) / "evidence/result.json")
            docs = self.documents(result["phases"][-1]["run_id"])
            if docs.read("dynamic-case-result") != (result,):
                raise ValueError("generic_acquisition_chain_accounting_changed")
            values.append(
                dict(
                    directory=str(directory(key)),
                    checksum=digest(result),
                    requests=result["new_requests"],
                )
            )
        if sum(v["requests"] for v in values) != p["previous_requests"]:
            raise ValueError("generic_acquisition_chain_accounting_changed")
        return values


def selection_for(checked, key):
    if key in {checked.proof["parent_key"], *(r["key"] for r in failed_lineage(checked.proof))}:
        raise ValueError("generic_acquisition_distinct_successor")
    return dict(
        schema_version=checked.proof.get("schema_version", VERSION),
        profile={
            VERSION: PROFILE,
            CHAIN_VERSION: CHAIN_PROFILE,
            RECOVERY_VERSION: RECOVERY_PROFILE,
        }[checked.proof.get("schema_version", VERSION)],
        key=key,
        parent_key=checked.proof["parent_key"],
        proof=checked.proof,
        max_new_requests=2,
        max_cumulative_requests=40,
        new_vision_requests=1,
        seconds=2700,
        no_new_tools=True,
        implementation=implementation(),
    )


def restore(path):
    path = Path(path).absolute()
    value = read(path / "input.json")
    if path != directory(path.name) or value.get("schema_version") not in {
        VERSION,
        CHAIN_VERSION,
        RECOVERY_VERSION,
    }:
        raise ValueError("generic_acquisition_restore_scope")
    p = value["proof"]
    chained = value["schema_version"] in {CHAIN_VERSION, RECOVERY_VERSION}
    checked = (
        chained_origin(
            p["parent_key"],
            p["target"],
            p["request_id"],
            p["failed_successor"]["key"],
            p["failed_successor"]["request_id"],
            p["client_binding"]["path"],
        )
        if chained
        else origin(p["parent_key"], p["target"], p["request_id"])
    )
    expected = selection_for(checked, path.name)
    # Current eligibility is rebuilt above. Historical readback is not dispatch
    # authority; preserve the saved implementation snapshot instead of rewriting it.
    if chained:
        if not isinstance(value.get("implementation"), dict) or not value["implementation"]:
            raise ValueError("generic_acquisition_restore_implementation_missing")
        expected["implementation"] = value["implementation"]
    claim_parent = p["failed_successor"]["key"] if chained else p["parent_key"]
    if (
        value != expected
        or read(directory(claim_parent) / "acquisition-successor.json") != value
        or read(path / "binding.json") != checked.binding
    ):
        raise ValueError("generic_acquisition_restore_binding")
    result = read(path / "evidence/result.json")
    if not result["complete"] or result["phases"][-1].get("target") is not None:
        raise ValueError("generic_acquisition_no_accepted_review_to_combine")
    cls = FailedAcquisitionSuccessor if chained else AcquisitionSuccessor
    return pending.confirmed_campaign(cls(checked, path.name))


def verified_stage_recovery(campaign):
    """Allow only exact confirmed stages; an in-flight/UNKNOWN request is not replayed."""
    confirmed = []
    for path in campaign.paths():
        request_file = path / "request.json"
        result_file = path / "result.json"
        if not request_file.exists():
            if result_file.exists():
                raise ValueError("generic_acquisition_recovery_result_without_request")
            continue
        if not result_file.exists():
            raise ValueError("generic_acquisition_recovery_unconfirmed_request")
        request, phase = read(request_file), read(result_file)
        run_id = request["lock"]["run_id"]
        context = campaign.contexts.read_research_run(run_id)
        control = campaign.sessions.read(run_id)
        if (
            context is None
            or context.request.model_dump(mode="json") != request
            or phase["run_id"] != run_id
            or campaign.documents(run_id).read("dynamic-phase-result") != (phase,)
            or control is None
            or control.phase != "stopped"
            or control.pending
            or control.cancelled
        ):
            raise ValueError("generic_acquisition_recovery_original_stage_binding")
        confirmed.append(run_id)
    return confirmed


def run(
    key,
    successor,
    target,
    request_id,
    *,
    preflight=False,
    recover=False,
    previous_successor=None,
    previous_request=None,
    client_profile=None,
):
    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.multimodal_research.unknown_acquisition import check_network

    from .generic_control import start
    from .generic_execution_lock import target as runtime_target
    from .generic_run import finish_export, persist_export

    chained = any(v is not None for v in (previous_successor, previous_request, client_profile))
    if chained and not all((previous_successor, previous_request, client_profile)):
        raise ValueError("generic_acquisition_failed_chain_all_bindings_required")
    root, out = directory(key), directory(successor)
    # Same parent exclusion identity as the original Generic runner, without
    # activating its expired control or confusing its key with this successor.
    lease = "generic-case-execution:" + digest(str(root.resolve()))[7:]
    with CasePipelineRepository(runtime_target()).guarded_session(
        "CASE:GENERIC_EXECUTION", lease
    ) as guard:
        checked = (
            chained_origin(
                key, target, request_id, previous_successor, previous_request, client_profile
            )
            if chained
            else origin(key, target, request_id)
        )
        check_network(checked.campaign.config)
        value = selection_for(checked, successor)
        claim = (directory(previous_successor) if chained else root) / "acquisition-successor.json"
        if claim.exists() and read(claim) != value:
            raise ValueError("generic_acquisition_successor_conflict")
        if (out / "input.json").exists() and read(out / "input.json") != value:
            raise ValueError("generic_acquisition_output_conflict")
        if preflight:
            return dict(selection=value, api_requests=0, new_run_created=False)
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(successor))
        incomplete = (out / "started.json").exists() and not (out / "evidence/result.json").exists()
        if incomplete and not recover:
            raise ValueError("generic_acquisition_claimed_use_saved_stage_recovery")
        guard()
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_once(claim, value)
        write_once(out / "input.json", value)
        write_once(out / "binding.json", checked.binding)
        start(value)
        cls = FailedAcquisitionSuccessor if chained else AcquisitionSuccessor
        campaign = cls(checked, successor)
        campaign._parent_execution_guard = guard
        if incomplete:
            verified_stage_recovery(campaign)
        result = campaign.execute()
        guard()
        return persist_export(successor, campaign, result)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--successor", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--acknowledge-request", required=True)
    parser.add_argument("--previous-successor")
    parser.add_argument("--acknowledge-previous-request")
    parser.add_argument("--http-client-profile")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Reuse only exact confirmed stages")
    a = parser.parse_args()
    result = run(
        a.key,
        a.successor,
        a.target,
        a.acknowledge_request,
        preflight=a.preflight,
        recover=a.recover,
        previous_successor=a.previous_successor,
        previous_request=a.acknowledge_previous_request,
        client_profile=a.http_client_profile,
    )
    print(json.dumps(result, ensure_ascii=False))
    return (
        0
        if a.preflight or result.get("status") == "already_recorded" or result["result"]["complete"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

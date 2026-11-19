"""One explicit different-method reassessment through the original generic owner.

Completed historical judgments never become pending. Only retained observations
enter new requests; original aggregation, requests, scores and limits stay intact.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    GRANT_KEY,
    AcceptedReviewSet,
)
from packages.evidence_details.live_permission import validate_live_config
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest
from packages.llm_provider.config import prompt, read_provider_config
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .generic_input import directory, load
from .generic_review import GenericReview

PROFILE = "relay-retained-generic-v1"
VERSION = "generic-accepted-method-selection/1.0.0"
PROVIDER = "configs/agents/relayrouter_generic_evidence_v1.json"


def request_limit(count, previous):
    if not 1 <= count <= 15 or not 0 <= previous <= 40:
        raise ValueError("generic_method_request_history_or_scope")
    cap = (count + 2) // 3 + 2
    if previous + cap > 40:
        raise ValueError("generic_method_original_cumulative_cap")
    return cap


def configuration(origin, profile):
    if profile != PROFILE:
        raise ValueError("generic_method_profile")
    role, _ = prompt(origin, "review")
    selected = read_provider_config(json.loads(Path(PROVIDER).read_bytes()))
    if (
        origin.schema_version != "agent-provider-config/7.0.0"
        or origin.mode != "live"
        or role.prompt_version != "review-batch-1.9.0"
        or role.images
        or origin.transport.max_retries != 1
        or origin.recovery != "confirmed_http_once/1.0.0"
        or origin.transport.model == selected.transport.model
        or origin.transport.base_url != selected.transport.base_url
        or origin.transport.endpoint != selected.transport.endpoint
        or origin.transport.api_key_env != selected.transport.api_key_env
    ):
        raise ValueError("generic_method_origin_configuration")
    raw = origin.model_dump(mode="json")
    # This legacy config uses its one retry slot for format correction too.
    # Preserve that literal legacy contract; narrow total dispatches at the
    # original budget hook below. Confirmed-HTTP recovery rejects UNKNOWN replay.
    raw["transport"].update(model=selected.transport.model, timeout_seconds=180)
    raw["review_batching"]["batch_size"] = 3
    return validate_live_config(read_provider_config(raw))


def current_origin(path, profile):
    """Rebuild current original sources, observations and exact accepted origins."""
    from agents.expert_agent.reporting import read_research_report
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.multimodal_research.accepted_aggregation import (
        GenericAggregationPlan,
        checked_path,
        origin_reader,
        plan_campaign,
        read_plan,
    )
    from packages.multimodal_research.report_storage import unpack

    path = checked_path(path)
    plan = read_plan(json.loads((path / "plan.json").read_bytes()))
    if not isinstance(plan, GenericAggregationPlan):
        raise ValueError("generic_method_requires_generic_aggregation")
    manifest = json.loads((path / "manifest.json").read_bytes())
    result = json.loads((path / "result.json").read_bytes())
    campaign, old, phase_class = plan_campaign(plan)
    config = configuration(campaign.config, profile)
    docs = campaign.documents(manifest["run_id"])
    if (
        result.get("schema_version") != "generic-accepted-aggregation-result/1.0.0"
        or result.get("original_result_checksum") != digest(old)
        or manifest.get("plan") != plan.model_dump(mode="json")
        or manifest.get("original_result_checksum") != digest(old)
        or docs.read("accepted-aggregation-manifest") != (manifest,)
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-case-result") != (result,)
        or docs.read("dynamic-phase-result") != (result["phases"][-1],)
    ):
        raise ValueError("generic_method_original_result_binding")
    phase = phase_class(
        campaign.base,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    state, _ = phase.project(run_id=docs.run_id, config=campaign.config)
    phase.verify(state)
    if docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),):
        raise ValueError("generic_method_original_material_changed")
    source = AcceptedReviewSet(
        tuple(origin_reader(p, generic=True, current_config=campaign.config) for p in plan.origins),
        campaign.config,
    )
    submission, _ = source.current(state)
    control = campaign.sessions.read(docs.run_id)
    grant = docs.read(GRANT_KEY)
    if (
        digest(state) != manifest["state_checksum"]
        or digest(submission) != manifest["submission_checksum"]
        or not grant
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or control.cancelled
        or digest(control) != grant[0]["control_checksum"]
        or RuntimeAgentJournal(docs).read()
    ):
        raise ValueError("generic_method_original_authority")
    report, _ = unpack(docs.read("accepted-aggregation-report"), key="accepted-aggregation-report")
    service = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
    )
    accepted = read_research_report(service, key=ACCEPTANCE_KEY, parent=state)
    if (
        report["result"] != result
        or report["scores"] != accepted["scores"]
        or accepted["receipt"]["submission_checksum"] != digest(submission)
        or accepted["receipt"]["permission_checksum"] != digest(grant[0])
    ):
        raise ValueError("generic_method_original_score_binding")
    accounting = {}
    for reader in source.readers:
        for attempt in reader.accounting:
            name = str(checked_path(attempt["directory"]))
            if name in accounting and accounting[name] != attempt:
                raise ValueError("generic_method_history_conflict")
            accounting[name] = attempt
    previous = sum(row["requests"] for row in accounting.values())
    projection = project_review_tasks(state)
    count = len(projection.assessment_ids)
    if previous != result["cumulative_requests"]:
        raise ValueError("generic_method_request_history_or_scope")
    cap = request_limit(count, previous)
    proof = dict(
        schema_version="generic-accepted-method-proof/1.0.0",
        origin=str(path),
        root_key=campaign.spec["key"],
        result=digest(result),
        manifest=digest(manifest),
        state=digest(state),
        submission=digest(submission),
        report=digest(report),
        control=digest(control),
        grant=digest(grant[0]),
        previous_requests=previous,
        attempts=accounting,
        model_ids=sorted(projection.assessment_ids),
        fixed_ids=sorted(projection.fixed_assessment_ids),
        config=digest(config),
        max_new_requests=cap,
        old_answers_reused=False,
        new_vision_requests=0,
        profile=profile,
        origins=[o.model_dump(mode="json") for o in submission.origins],
    )
    return SimpleNamespace(
        campaign=campaign,
        documents=docs,
        state=state,
        source=source,
        proof=proof,
        config=config,
        result=result,
        manifest=manifest,
    )


class AcceptedMethodReview(GenericReview):
    """Original Review executor with explicit completed-origin eligibility."""

    def __init__(self, origin, key):
        from .case_execution import owners

        self.method_origin = origin
        root = directory(origin.proof["root_key"])
        spec = load(root.name)
        binding = json.loads((root / "binding.json").read_bytes())
        runtime, store, _ = owners()
        super().__init__(spec, (root, binding, runtime, store), key)
        self.config = origin.config
        self.provider_path = PROVIDER
        self.version = "generic-accepted-method-v1:" + key
        self.export_suffix = "_保留事实统一文字方法复评_" + key
        self.max_new_requests = origin.proof["max_new_requests"]
        self.duration_seconds = 2700

    def prior(self):
        origin = self.method_origin
        docs, proof = origin.documents, origin.proof
        if (
            docs.read("multimodal-case-result") != (origin.result,)
            or docs.read("accepted-aggregation-manifest") != (origin.manifest,)
            or digest(origin.campaign.sessions.read(docs.run_id)) != proof["control"]
            or RuntimeAgentJournal(docs).read()
        ):
            raise ValueError("generic_method_origin_changed")
        for prior in proof["origins"]:
            old = origin.campaign.documents(prior["run_id"])
            if (
                digest(origin.campaign.sessions.read(old.run_id)) != prior["control_checksum"]
                or digest(RuntimeAgentJournal(old).read()) != prior["journal_checksum"]
                or digest(old.read("accepted:" + prior["acceptance_key"]))
                != prior["acceptance_checksum"]
            ):
                raise ValueError("generic_method_ancestor_changed")
        return dict(reservations=proof["previous_requests"], accepted_method_origin=proof)

    def before_review(self):
        current = current_origin(self.method_origin.proof["origin"], PROFILE)
        if current.proof != self.method_origin.proof:
            raise ValueError("generic_method_current_origin_changed")

    def budget_factory(self, **kwargs):
        from packages.llm_provider.provider import ProviderError

        budget = super().budget_factory(**kwargs)
        original = budget.before_reserve

        def before_reserve(checksum):
            # Normal batches plus one total extra dispatch; retain one request
            # unit for zero-request finalization. No widening of legacy config.
            if self.usage() >= self.max_new_requests - 1:
                raise ProviderError("budget")
            original(checksum)

        budget.before_reserve = before_reserve
        return budget

    def freeze_observations(self):
        return dict(
            **super().freeze_observations(), accepted_method_origin=self.method_origin.proof
        )

    def archive_completed(self, result):
        # A method comparison alone does not grant a new Memory success/publication.
        return result


def run(key, review_key, *, method_origin, review_method_profile):
    """Reuse original locks, controls, executor, acceptance and exact export ports."""
    import fcntl

    from .generic_control import guard, start

    if key == review_key or review_method_profile != PROFILE:
        raise ValueError("generic_method_explicit_new_identity")
    parent, out = directory(key), directory(review_key)
    with (parent / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        origin = current_origin(method_origin, review_method_profile)
        if origin.proof["root_key"] != key or (parent / "cancel-request.json").exists():
            raise ValueError("generic_method_root_or_cancel")
        selection = dict(
            schema_version=VERSION,
            key=review_key,
            parent_key=key,
            origin=origin.proof,
            profile=PROFILE,
            old_scores_reused=False,
            new_vision_requests=0,
            new_tool_executions=0,
        )
        # Separate from the historical pending-only successor. One such new method
        # per original case; changing output keys cannot reset this finite grant.
        claim = parent / "accepted-method-successor.json"
        if claim.exists() and json.loads(claim.read_bytes()) != selection:
            raise ValueError("generic_method_successor_already_claimed")
        if out.exists() and (
            not (out / "input.json").exists()
            or json.loads((out / "input.json").read_bytes()) != selection
        ):
            raise ValueError("generic_method_output_already_exists")
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_once(claim, selection)
        write_once(out / "input.json", selection)
        write_once(out / "binding.json", json.loads((parent / "binding.json").read_bytes()))
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_method_export(review_key))
        campaign = AcceptedMethodReview(origin, review_key)
        start(selection)
        guard(review_key)
        result = campaign.execute()
        return persist_method_export(review_key, campaign, result)


def media_origin(report, key):
    """Resolve this version's exact saved acquisition; never grant new execution."""
    from .generic_run import export

    out = directory(key)
    selection = json.loads((out / "input.json").read_bytes())
    parent_key = selection.get("parent_key")
    if (
        selection.get("schema_version") != VERSION
        or selection.get("profile") != PROFILE
        or selection.get("key") != key
        or not isinstance(parent_key, str)
        or parent_key == key
        or selection.get("origin", {}).get("root_key") != parent_key
    ):
        raise ValueError("generic_method_media_identity")
    parent = directory(parent_key)
    binding = json.loads((parent / "binding.json").read_bytes())
    spec = json.loads((parent / "input.json").read_bytes())
    if (
        json.loads((parent / "accepted-method-successor.json").read_bytes()) != selection
        or json.loads((out / "binding.json").read_bytes()) != binding
        or spec.get("key") != parent_key
        or not spec.get("schema_version", "").startswith("fairvlmed-run-input/")
    ):
        raise ValueError("generic_method_media_acquisition_binding")
    original = json.loads(Path(export(parent_key)["json"]).read_bytes())
    if original["result"]["case_code"] != report["result"]["case_code"] or original.get(
        "native_tools"
    ) != report.get("native_tools"):
        raise ValueError("generic_method_media_original_report")
    return parent, binding


def finish_method_export(key):
    """Original saved-report export and converter with explicit retained media."""
    from packages.object_store import checksum

    from .case_execution import owners
    from .evidence_media import build
    from .generic_run import export, export_presentation
    from .report_production import PROFILE as REPORT_PROFILE
    from .report_production import media_links

    saved = export(key)  # Original Runtime envelopes, artifacts and exact HTML.
    report = json.loads(Path(saved["json"]).read_bytes())
    parent, binding = media_origin(report, key)
    if report.get("report_production"):
        if report["report_production"].get("schema_version") != REPORT_PROFILE:
            raise ValueError("generic_method_report_production_version")
        runtime, objects, _ = owners()
        media = build(
            report,
            source_checksum=checksum(Path(saved["json"]).read_bytes()),
            root=parent,
            binding=binding,
            runtime=runtime,
            objects=objects,
        )
        media_path = directory(key) / "retained-method-media.json"
        write_once(media_path, media_links(report, media))
        destination = Path(saved["html"]).with_name(Path(saved["html"]).stem + "_证据树.html")
        saved["tree"] = export_presentation(key, destination, media_path=media_path)
    return saved


def persist_method_export(key, campaign, result):
    """Bind the original report bundle; the new selection needs its own media adapter.

    Same generic-export-reference/1.0.0, read by the unchanged generic export
    owner. No report rendering, score writing or new database transaction here.
    """
    docs = campaign.documents(result["phases"][-1]["run_id"])
    report_key = getattr(campaign, "report_key", "dynamic-case-report")
    key_ref = docs.key(report_key)
    artifacts = campaign.base.runtime.read_observation_bundle(key_ref)
    if not artifacts or len(artifacts) != 2:
        raise ValueError("generic_saved_report_missing")
    reference = dict(
        schema_version="generic-export-reference/1.0.0",
        run_key=key,
        case_code=campaign.base.binding["case_code"],
        bundle_key=key_ref,
        run_id=docs.run_id,
        artifacts=[v.model_dump(mode="json") for v in artifacts],
        complete=result["complete"],
        clinical_acceptance=False,
        root_checksum=docs.root.checksum,
        report_key_checksum=digest(report_key),
    )
    write_once(directory(key) / "export-reference.json", reference)
    return dict(result=result, exports=finish_method_export(key))

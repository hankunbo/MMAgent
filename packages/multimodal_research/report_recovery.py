"""Zero-dispatch report recovery from original stopped phase results and owners."""

import json
from pathlib import Path
from types import SimpleNamespace

from packages.evidence_details.accepted_read_scope import _historical_medical_sources
from packages.grounding.models import digest, validate


def saved_scope_version(spec, protocol):
    """Recognize explicit survey versions without weakening legacy bindings."""
    version = protocol.get("schema_version")
    if version not in {"multimodal-live-campaign/1.20.0", "multimodal-live-campaign/1.21.0"}:
        if "input_scope" in protocol or "scope_selection_policy" in protocol:
            raise ValueError("saved_report_scope_version")
        return None
    from .active_requirements import load

    reference = spec.get("active_acquisition", {})
    if set(reference) != {"path", "checksum"}:
        raise ValueError("saved_report_scope_profile_binding")
    profile = load(reference["path"])
    selected = version == "multimodal-live-campaign/1.21.0"
    if (
        digest(profile) != reference["checksum"]
        or not profile.get("input_scope_policy")
        or not isinstance(protocol.get("input_scope"), dict)
        or selected != ("scope_selection_policy" in profile)
        or selected != ("scope_selection_policy" in protocol)
        or selected
        and protocol["scope_selection_policy"] != profile["scope_selection_policy"]
    ):
        raise ValueError("saved_report_scope_profile_binding")
    return version


def restore_scope_context(campaign, result):
    """Read original accepted survey and method records; never dispatch or save."""
    from .input_scope import configuration, read_results

    plan = getattr(campaign.base, "input_scope_plan", None)
    if plan is None:
        if "input_scope" in campaign.protocol or "input_scopes" in result:
            raise ValueError("saved_report_scope_plan_binding")
        return
    expected = dict(
        plan_checksum=digest(plan),
        provider_checksum=digest(configuration(campaign.config, plan)),
        targets=list(campaign.scope_targets),
        budget_owner="original_campaign_and_RuntimeAgentJournal",
        disease_votes=0,
    )
    if (
        campaign.protocol.get("input_scope") != expected
        or result.get("input_scope_plan_checksum") != digest(plan)
        or result.get("input_scopes") != read_results(campaign, require_all=True)
    ):
        raise ValueError("saved_report_scope_plan_binding")
    for scope in result["input_scopes"]:
        if campaign.documents(scope["phase"]["run_id"]).read("dynamic-campaign") != (
            campaign.protocol,
        ):
            raise ValueError("saved_report_scope_campaign_binding")
    selected = "scope_selection_policy" in campaign.protocol
    if selected != ("scope_method_plan" in result):
        raise ValueError("saved_report_scope_method_binding")
    if selected:
        from .scope_selection import ensure

        if ensure(campaign, allow_native=False) != result["scope_method_plan"]:
            raise ValueError("saved_report_scope_method_binding")


def restore_execution_context(base, spec, protocol):
    """Rebuild opt-in question from the original task, never from an HTML label."""
    from packages.llm_provider.client_selection import check_entry_binding as check_client

    scope_version = saved_scope_version(spec, protocol)
    client = check_client(spec, protocol)
    if client is not None:
        base.http_client_selection = client
    from packages.llm_provider.model_selection import check_entry_binding
    from packages.llm_provider.review_rating import check_entry_binding as check_review
    from packages.llm_provider.vision_correction import check_entry_binding as check_vision

    review_contract = check_review(spec, protocol)
    if review_contract is not None:
        base.review_contract = review_contract
    feedback = check_vision(spec, protocol)
    if feedback is not None:
        base.vision_feedback = feedback
    binding = check_entry_binding(spec, protocol)
    if binding is not None:
        base.model_selection = binding
    contextual = {
        "multimodal-live-entry/1.18.0": "relay-context-v1",
        **dict.fromkeys(
            (f"multimodal-live-entry/1.{v}.0" for v in range(19, 25)),
            "relay-snapshot-context-v1",
        ),
        "multimodal-live-entry/1.25.0": "relay-root-question-v1",
        "multimodal-live-entry/1.26.0": "relay-root-question-v1",
        "multimodal-live-entry/1.27.0": "relay-root-question-v1",
        "multimodal-live-entry/1.28.0": "relay-root-question-v1",
        "multimodal-live-entry/1.29.0": "relay-root-question-v1",
        "multimodal-live-entry/1.30.0": "relay-root-question-v1",
    }.get(spec["schema_version"])
    if contextual:
        if (
            spec.get("execution_profile") != contextual
            or protocol.get("execution_profile") != contextual
        ):
            raise ValueError("saved_report_execution_profile_binding")
        base.execution_profile = contextual
    elif spec.get("execution_profile") or protocol.get("execution_profile"):
        raise ValueError("saved_report_execution_profile_version")
    if contextual == "relay-root-question-v1":
        from .single_proposition import root_direction_spec

        question = root_direction_spec(base.task)
        batch_size = spec.get("review_batch_size", 1)
        routed_backlog = spec["schema_version"] == "multimodal-live-entry/1.30.0"
        root_confirmed = spec["schema_version"] == "multimodal-live-entry/1.28.0"
        root_backlog = (
            routed_backlog
            or root_confirmed
            or spec["schema_version"] == "multimodal-live-entry/1.27.0"
        )
        batched = root_backlog or spec["schema_version"] in {
            "multimodal-live-entry/1.26.0",
            "multimodal-live-entry/1.29.0",
        }
        networked = routed_backlog or spec["schema_version"] == "multimodal-live-entry/1.29.0"
        if networked != ("network_route" in spec) or (
            networked and spec["provider"].get("network_route") != spec["network_route"]
        ):
            raise ValueError("saved_report_network_route_binding")
        if networked:
            base.network_route = spec["network_route"]
        if root_backlog and (
            spec.get("definition_failure_of", {}).get("mode")
            != (
                "routed_root_source_backlog_no_source_replay"
                if routed_backlog
                else "root_confirmed_schema_backlog_no_source_replay"
                if root_confirmed
                else "root_technical_backlog_no_source_replay"
            )
            or protocol.get("definition_continuation") != spec["definition_failure_of"]
        ):
            raise ValueError("saved_report_root_backlog_binding")
        if (
            type(batch_size) is not int
            or batched
            and batch_size not in (2, 3)
            or not batched
            and ("review_batch_size" in spec or "review_batch_size" in protocol)
            or batched
            and protocol.get("review_batch_size") != batch_size
        ):
            raise ValueError("saved_report_review_batch_size")
        if (
            spec.get("root_direction_review") != question
            or protocol.get("root_direction_review") != question
            or protocol.get("schema_version")
            != (
                scope_version
                or (
                    "multimodal-live-campaign/1.19.0"
                    if routed_backlog
                    else "multimodal-live-campaign/1.18.0"
                    if root_confirmed
                    else "multimodal-live-campaign/1.17.0"
                    if root_backlog
                    else "multimodal-live-campaign/1.16.0"
                    if batched
                    else "multimodal-live-campaign/1.15.0"
                )
            )
        ):
            raise ValueError("saved_report_root_question_binding")
        base.root_direction_review = question
        base.review_batch_size = batch_size
        window = spec.get("source_fallback_window")
        if window != protocol.get("source_fallback_window") or (
            window is not None and not routed_backlog
        ):
            raise ValueError("saved_report_fallback_window_binding")
        if window is not None:
            from .discovery_continuation import check_fallback_window

            parent = json.loads(
                (Path(spec["definition_failure_of"]["parent"]) / "protocol.json").read_bytes()
            )
            clock = check_fallback_window(window, parent)
            if any(protocol[k] != v for k, v in clock.items()):
                raise ValueError("saved_report_fallback_window_clock")
            base.source_fallback_window = window
    elif spec.get("root_direction_review") or protocol.get("root_direction_review"):
        raise ValueError("saved_report_root_question_version")


def saved_campaign(path):
    """Private reconstruction for export/review recovery, not an execution entry."""
    from packages.dataset_adapters.current_read_scope import _historical_input_owners
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.dataset_adapters.multimodal.preparation import verify
    from packages.llm_provider.config import read_provider_config
    from packages.research_runtime.task_context import TaskIntent

    from .active_acquisition import install
    from .active_requirements import load
    from .campaign import InputEvidence, MultimodalCampaign
    from .entry import owners
    from .material import MedicalSources
    from .native_fundus import read_saved

    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("saved_report_symlink")
    spec = json.loads((path / "entry.json").read_bytes())
    protocol = json.loads((path / "protocol.json").read_bytes())
    result = json.loads((path / "result.json").read_bytes())
    if (
        spec["schema_version"]
        not in {
            "multimodal-live-entry/1.12.0",
            "multimodal-live-entry/1.15.0",
            "multimodal-live-entry/1.16.0",
            "multimodal-live-entry/1.17.0",
            "multimodal-live-entry/1.18.0",
            "multimodal-live-entry/1.19.0",
            "multimodal-live-entry/1.20.0",
            "multimodal-live-entry/1.21.0",
            "multimodal-live-entry/1.22.0",
            "multimodal-live-entry/1.23.0",
            "multimodal-live-entry/1.24.0",
            "multimodal-live-entry/1.25.0",
            "multimodal-live-entry/1.26.0",
            "multimodal-live-entry/1.27.0",
            "multimodal-live-entry/1.28.0",
            "multimodal-live-entry/1.29.0",
            "multimodal-live-entry/1.30.0",
        }
        or spec["implementation"] != protocol["implementation"]
        or spec["input_checksum"] != protocol["input"]
        or spec["intent"] != result["task"]
        or digest(spec["provider"]) != protocol["provider"]
        or not result["phases"]
        or result["phases"][-1]["target"] is not None
    ):
        raise ValueError("saved_report_parent_binding")
    if (
        spec["schema_version"] in {"multimodal-live-entry/1.23.0", "multimodal-live-entry/1.24.0"}
        or (
            spec["schema_version"]
            in {
                "multimodal-live-entry/1.25.0",
                "multimodal-live-entry/1.26.0",
                "multimodal-live-entry/1.27.0",
                "multimodal-live-entry/1.28.0",
                "multimodal-live-entry/1.29.0",
                "multimodal-live-entry/1.30.0",
            }
            and spec["dataset"] == "grape"
        )
    ) != (spec.get("input_semantics") == "grape-v1" and spec["dataset"] == "grape"):
        raise ValueError("saved_report_semantic_entry_binding")
    prepared = prepare(
        spec["dataset"],
        spec["root"],
        spec["sample"],
        selection=spec["selection"],
        cutoff_visit=spec["cutoff_visit"],
        semantic_profile=spec.get("input_semantics"),
    )
    if prepared.manifest != verify(path / "input/preparation.json"):
        raise ValueError("saved_report_input_changed")
    runtime, objects, inputs, target = _historical_input_owners(owners)
    visible = inputs.read_visible(result["case_code"])
    base = InputEvidence(
        registered=SimpleNamespace(visible=visible),
        inputs=inputs,
        runtime=runtime,
        objects=objects,
        prepared=prepared,
        task=validate(TaskIntent, spec["intent"]),
        output=path,
        sources=_historical_medical_sources(MedicalSources),
    )
    if base.manifest.checksum != spec["input_checksum"]:
        raise ValueError("saved_report_input_binding")
    profile = load(spec["active_acquisition"]["path"])
    if digest(profile) != spec["active_acquisition"]["checksum"]:
        raise ValueError("saved_report_profile_changed")
    native = None
    if spec.get("discovery_continuation_of"):
        native_path = Path(spec["discovery_continuation_of"]) / "native-fundus/accepted.json"
        if any(p.is_symlink() for p in (native_path, *native_path.parents)):
            raise ValueError("saved_report_native_symlink")
        native = json.loads(native_path.read_bytes())
    elif (path / "native-fundus/accepted.json").is_file():
        native_path = path / "native-fundus/accepted.json"
        if any(p.is_symlink() for p in (native_path, *native_path.parents)):
            raise ValueError("saved_report_native_symlink")
        native = json.loads(native_path.read_bytes())

    def native_reader(current, requirement, *, before_dispatch, component_scoped=False):
        if spec.get("definition_failure_of", {}).get("mode") in {
            "root_confirmed_schema_backlog_no_source_replay",
            "routed_root_source_backlog_no_source_replay",
        }:
            from .discovery_continuation import reuse_failed_native

            return reuse_failed_native(
                current,
                spec["definition_failure_of"],
                read_only=True,
                history_base=current,
                following_provider=spec["provider"],
                following_review_contract=spec.get("review_contract"),
            )(
                current,
                requirement,
                before_dispatch=before_dispatch,
                component_scoped=component_scoped,
            )
        return read_saved(
            current, requirement, component_scoped=component_scoped, legacy_record=native
        )

    base.active_seconds = spec["seconds"]
    base.active_schedule_v19 = True
    base.resilient_vision = spec["resilient_vision"]
    base.exact_primary_status = spec["schema_version"] != "multimodal-live-entry/1.12.0"
    base.known_only = False
    restore_execution_context(base, spec, protocol)
    component = spec.get("component_consumer")
    if bool(component) != (
        spec["schema_version"]
        in {
            "multimodal-live-entry/1.16.0",
            "multimodal-live-entry/1.20.0",
            "multimodal-live-entry/1.21.0",
            "multimodal-live-entry/1.22.0",
        }
    ):
        raise ValueError("saved_report_component_binding")
    install(
        base, profile, native_runner=native_reader, defer_native=bool(component), read_only=True
    )
    if base.active_owner.methods is not None and protocol.get("active_acquisition") != digest(
        base.active_owner.report()
    ):
        raise ValueError("saved_report_method_selection_changed")
    campaign = MultimodalCampaign(base, target, campaign_id=path.name)
    campaign.config = read_provider_config(spec["provider"])
    campaign.protocol = protocol
    if component:
        from .component_consumer import install_campaign, load, registration

        selection = load(component["path"], expected=component["checksum"])
        registered = registration(inputs, selection, base.manifest, base.task)
        if registered.visible != visible:
            raise ValueError("saved_report_component_case_binding")
        install_campaign(campaign, selection, component)
    if spec.get("definition_failure_of"):
        from .discovery_continuation import ConfirmedDefinitionContinuation

        campaign.definition_continuation = ConfirmedDefinitionContinuation(
            campaign, spec["definition_failure_of"], read_only=True, history_base=base
        )
    for phase in result["phases"]:
        context = campaign.contexts.read_research_run(phase["run_id"])
        base.contexts[phase["run_id"]] = context
        docs = campaign.documents(phase["run_id"])
        control = campaign.sessions.read(phase["run_id"])
        if (
            context.request.scope.case_code != result["case_code"]
            or context.request.input_checksum != spec["input_checksum"]
            or context.request.task != base.task
            or control is None
            or control.phase != "stopped"
            or control.pending is not None
            or docs.read("dynamic-phase-result") != (phase,)
            or docs.read("dynamic-campaign") != (protocol,)
        ):
            raise ValueError("saved_report_phase_binding")
    docs = campaign.documents(result["phases"][-1]["run_id"])
    if docs.read("multimodal-case-result") != (result,):
        raise ValueError("saved_report_result_binding")
    # This port reads an existing immutable revision; no Provider is constructed.
    campaign.expand_definitions(restore_only=True, report_only=True)
    restore_scope_context(campaign, result)
    campaign.base.check()
    return campaign, result


def saved_method_campaign(path):
    """Read a finished method window; never reassemble it for execution."""
    from .campaign_renewal import assemble
    from .memory import method_archive_status
    from .report_storage import unpack

    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("saved_report_symlink")
    campaign = assemble(path)  # Original claim, parent, input and source owners.
    result = json.loads((path / "result.json").read_bytes())
    if campaign.protocol.get("mode") != "review" or method_archive_status(campaign, result) is None:
        raise ValueError("saved_method_report_version")
    phases = result.get("phases", [])
    if (
        len(phases) != 1
        or not result.get("complete")
        or not phases[0].get("accepted")
        or phases[0].get("target") is not None
        or result["case_code"] != campaign.base.binding["case_code"]
        or result["task"] != campaign.base.task.model_dump(mode="json")
        or result["dataset"] != campaign.base.manifest.dataset
    ):
        raise ValueError("saved_method_report_scope")
    final = phases[0]
    docs = campaign.documents(final["run_id"])
    control = campaign.sessions.read(docs.run_id)
    if (
        control is None
        or control.phase != "stopped"
        or control.pending is not None
        or docs.read("dynamic-phase-result") != (final,)
        or docs.read("multimodal-case-result") != (result,)
    ):
        raise ValueError("saved_method_report_owner_binding")
    # This narrow recovery only publishes an existing exact report, never
    # silently renders a new interpretation under changed code or grants.
    report, _ = unpack(docs.read("dynamic-case-report"), key="dynamic-case-report")
    if report["result"] != result:
        raise ValueError("saved_method_report_result_binding")
    campaign.base.check()
    return campaign, result


def restore_method_report(path):
    """Zero API/score/Memory writes, including after a post-report archive error."""
    import fcntl

    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .memory import method_archive_status
    from .report import restore_export
    from .report_io import publish_json

    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("saved_report_symlink")
    with (path / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign, result = saved_method_campaign(path)
        docs = campaign.documents(result["phases"][-1]["run_id"])
        journal = RuntimeAgentJournal(docs)
        before = campaign.sessions.read(docs.run_id), journal.read()
        failure = docs.read("multimodal-stage-failure")
        local_failure = path / "stage-failure.json"
        if local_failure.exists() and failure != (json.loads(local_failure.read_bytes()),):
            raise ValueError("saved_method_failure_binding")
        exported = restore_export(campaign, result)
        if before != (campaign.sessions.read(docs.run_id), journal.read()):
            raise ValueError("saved_report_execution_changed")
        memory = method_archive_status(campaign, result)
        record = dict(
            schema_version="multimodal-saved-method-report-restoration/1.0.0",
            run_id=docs.run_id,
            result_checksum=digest(result),
            report_checksum=exported["report_checksum"],
            original_failure_checksum=digest(failure[0]) if failure else None,
            original_failure_preserved=True,
            report_saved=True,
            display_completed=True,
            memory=memory,
            new_api_requests=0,
            new_model_runs=0,
            database_writes=0,
        )
        record_path = publish_json(path / "report-restoration.json", record)
        return dict(
            exported,
            report_recovery=True,
            restoration_record=str(record_path),
            memory=memory,
            new_api_requests=0,
            new_model_runs=0,
            database_writes=0,
        )


def alternative_report_delta(before, after):
    """Only the diagnosed display projection and this zero-dispatch port may differ."""
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if (
        changed
        - {
            "packages/multimodal_research/alternative_acquisition.py",
            "packages/multimodal_research/report_recovery.py",
        }
        or before.keys() != after.keys()
    ):
        raise ValueError("alternative_report_recovery_implementation_scope")
    return sorted(changed)


def restore_alternative_report(path):
    """Finish a committed alternative result without dispatch or rescore."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .alternative_acquisition import VERSIONS
    from .campaign_renewal import assemble
    from .implementation import freeze
    from .report import export, restore_export
    from .report_io import publish_json
    from .stage_failure import read_saved_failure
    from .stages import local_current

    campaign = assemble(path)
    if campaign.protocol.get("schema_version") not in VERSIONS:
        raise ValueError("alternative_report_recovery_version")
    delta = alternative_report_delta(campaign.protocol["implementation"], freeze())
    with campaign.execution_session():
        result = json.loads((path / "result.json").read_bytes())
        final = result["phases"][-1]
        docs = campaign.documents(final["run_id"])
        failure = read_saved_failure(campaign)
        if (
            not result["complete"]
            or final["target"] is not None
            or not final["accepted"]
            or docs.read("multimodal-case-result") != (result,)
            or not failure
            or failure["failure_stage"] != "export"
            or failure["score_status"] != "saved"
            or failure["protocol_checksum"] != digest(campaign.protocol)
            or failure["case_result"] != result
            or docs.read("multimodal-stage-failure") != (failure,)
        ):
            raise ValueError("alternative_report_recovery_committed_result_required")
        local_current(campaign, None)
        for phase in result["phases"]:
            if campaign.documents(phase["run_id"]).read("dynamic-phase-result") != (phase,):
                raise ValueError("alternative_report_recovery_phase_binding")

        def execution_state():
            return (
                tuple(
                    (
                        campaign.sessions.read(p["run_id"]),
                        RuntimeAgentJournal(campaign.documents(p["run_id"])).read(),
                    )
                    for p in result["phases"]
                ),
                docs.read("accepted:" + final["acceptance_key"]),
                docs.read("multimodal-case-result"),
                docs.read("multimodal-stage-failure"),
            )

        before = execution_state()
        if not before[1] or digest(before[1]) != failure["accepted_reference"]["checksum"]:
            raise ValueError("alternative_report_recovery_score_binding")
        restored = (restore_export if docs.read("dynamic-case-report") else export)(
            campaign, result
        )
        if execution_state() != before:
            raise ValueError("alternative_report_recovery_execution_changed")
        receipt = dict(
            schema_version="alternative-report-recovery/1.0.0",
            protocol_checksum=digest(campaign.protocol),
            failure_checksum=digest(failure),
            accepted_checksum=digest(before[1]),
            implementation=freeze(),
            implementation_delta=delta,
            report_checksum=restored["report_checksum"],
            added_requests=0,
            added_scores=0,
            historical_failure_preserved=True,
        )
        docs.write("alternative-report-recovery", (receipt,))
        publish_json(path / "alternative-report-recovery.json", receipt)
        return dict(restored, report_recovery=receipt)


def restore_definition_report(path):
    """Recover a committed definition/observation successor without dispatch."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .campaign_renewal import CASE_DEFINITION_VERSIONS, completed_parent
    from .implementation import freeze
    from .report import export, restore_export
    from .report_io import publish_json
    from .stage_failure import read_saved_failure
    from .stages import local_current

    campaign, result = completed_parent(path)
    before_code, current_code = campaign.protocol["implementation"], freeze()
    changed = sorted(
        k
        for k in before_code.keys() | current_code.keys()
        if before_code.get(k) != current_code.get(k)
    )
    if (
        campaign.protocol["schema_version"] not in CASE_DEFINITION_VERSIONS
        or before_code.keys() != current_code.keys()
        or set(changed)
        - {
            "packages/multimodal_research/active_acquisition.py",
            "packages/multimodal_research/campaign_renewal.py",
            "packages/multimodal_research/report_recovery.py",
        }
    ):
        raise ValueError("definition_report_recovery_implementation_scope")
    with campaign.execution_session():
        final = result["phases"][-1]
        docs = campaign.documents(final["run_id"])
        failure = read_saved_failure(campaign)
        if (
            not result["complete"]
            or final["target"] is not None
            or not (final["accepted"] or final.get("evidence_limited"))
            or not failure
            or failure["failure_stage"] != "export"
            or failure["protocol_checksum"] != digest(campaign.protocol)
            or failure["case_result"] != result
            or docs.read("multimodal-stage-failure") != (failure,)
        ):
            raise ValueError("definition_report_recovery_committed_result_required")
        local_current(campaign, None)
        for phase in result["phases"]:
            if campaign.documents(phase["run_id"]).read("dynamic-phase-result") != (phase,):
                raise ValueError("definition_report_recovery_phase_binding")

        def execution_state():
            return (
                tuple(
                    (
                        campaign.sessions.read(p["run_id"]),
                        RuntimeAgentJournal(campaign.documents(p["run_id"])).read(),
                        campaign.documents(p["run_id"]).read("dynamic-phase-result"),
                        campaign.documents(p["run_id"]).read("accepted:" + p["acceptance_key"])
                        if p.get("acceptance_key")
                        else (),
                    )
                    for p in result["phases"]
                ),
                docs.read("multimodal-case-result"),
                docs.read("multimodal-stage-failure"),
            )

        before = execution_state()
        restored = (restore_export if docs.read("dynamic-case-report") else export)(
            campaign, result
        )
        if execution_state() != before:
            raise ValueError("definition_report_recovery_execution_changed")
        receipt = dict(
            schema_version="definition-report-recovery/1.0.0",
            protocol_checksum=digest(campaign.protocol),
            failure_checksum=digest(failure),
            implementation=current_code,
            implementation_delta=changed,
            report_checksum=restored["report_checksum"],
            added_requests=0,
            added_scores=0,
            historical_failure_preserved=True,
        )
        docs.write("definition-report-recovery", (receipt,))
        publish_json(path / "definition-report-recovery.json", receipt)
        return dict(restored, report_recovery=receipt)


def restore(path):
    """Append missing export only; no run, API, tool, score or Control transition."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .report import export, restore_export

    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("saved_report_symlink")
    protocol = json.loads((path / "protocol.json").read_bytes())
    if protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.79.0",
        "multimodal-campaign-renewal/1.80.0",
    }:
        return restore_definition_report(path)
    if protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.59.0",
        "multimodal-campaign-renewal/1.60.0",
    }:
        return restore_alternative_report(path)
    if protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.35.0",
        "multimodal-campaign-renewal/1.40.0",
        "multimodal-campaign-renewal/1.36.0",
    }:
        return restore_method_report(path)
    campaign, result = saved_campaign(path)
    docs = campaign.documents(result["phases"][-1]["run_id"])
    journal = RuntimeAgentJournal(docs)
    before = (campaign.sessions.read(docs.run_id), journal.read())
    restored = (restore_export if docs.read("dynamic-case-report") else export)(campaign, result)
    if before != (campaign.sessions.read(docs.run_id), journal.read()):
        raise ValueError("saved_report_execution_changed")
    return dict(restored, report_recovery=True, new_api_requests=0, new_model_runs=0)


def restore_context_report(path):
    """Explicit producer revision, not a new Review or replacement of saved bytes."""
    from copy import copy

    from agents.expert_agent.committed_review import read_committed_review
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .campaign_renewal import completed_parent
    from .context_review import METHOD_VERSION
    from .execution_lease import session
    from .report import export, restore_export
    from .report_storage import unpack
    from .stages import local_current

    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("saved_report_symlink")
    campaign, result = completed_parent(path)
    final = result["phases"][-1]
    unscored = campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.52.0"
    from .alternative_acquisition import VERSIONS as alternative_versions

    alternative = campaign.protocol.get("schema_version") in alternative_versions
    if (
        campaign.protocol.get("schema_version") != METHOD_VERSION
        and not unscored
        and not alternative
    ) or (not result.get("complete") or final["target"] is not None or not final["accepted"]):
        raise ValueError("retained_context_report_requires_completed_method_review")
    with session(campaign):
        # Fresh source/input, original DomainState, current Control and acceptance
        # checks; neither expiry nor a caller flag authorizes any new dispatch.
        local_current(campaign, None)
        docs = campaign.documents(final["run_id"])
        parent = docs.read("dynamic-phase-input")[0]
        proof = read_committed_review(
            docs,
            parent=parent,
            config=campaign.config,
            confirmed_review_reader=getattr(campaign.base, "confirmed_review_reader", None),
        )
        if proof is None or proof["acceptance_key"] != final["acceptance_key"]:
            raise ValueError("retained_context_report_acceptance_binding")
        original = docs.read("dynamic-case-report")
        report, _ = unpack(original, key="dynamic-case-report")
        if report["result"] != result:
            raise ValueError("retained_context_report_result_binding")
        accepted = docs.read("accepted:" + final["acceptance_key"])
        if report["scores"] != accepted[2]["scores"]:
            raise ValueError("retained_context_report_score_binding")
        projection = dict(
            schema_version="retained-context-report-projection/1.2.0"
            if alternative
            else "retained-context-report-projection/1.1.0"
            if unscored
            else "retained-context-report-projection/1.0.0",
            original_report_checksum=digest(report),
            original_report_key="dynamic-case-report",
            result_checksum=digest(result),
            accepted_checksum=digest(accepted),
            original_bytes_preserved=True,
            new_api_requests=0,
            new_scores=0,
        )
        if alternative:
            projection["scores_checksum"] = digest(report["scores"])
        if unscored or alternative:
            from .report import acquired_observations

            actual = {
                k: v.model_dump(mode="json") for k, v in acquired_observations(campaign).items()
            }
            if actual != report.get("acquisition_observations"):
                raise ValueError("retained_context_report_acquisition_changed")
            projection["acquisition_observations_checksum"] = digest(actual)
        current = copy(campaign)
        current.report_projection = projection
        revision = "v3" if alternative else "v2" if unscored else "v1"
        current.report_key = "retained-context-report-" + revision
        current.report_name = campaign.output.name + "-retained-context-" + revision
        current.local_export_recovery = True
        journal = RuntimeAgentJournal(docs)
        before = campaign.sessions.read(docs.run_id), journal.read()
        existing = docs.read(current.report_key)
        if existing:
            saved, _ = unpack(existing, key=current.report_key)
            if saved.get("report_projection") != projection:
                raise ValueError("retained_context_report_projection_binding")
        exported = (restore_export if existing else export)(current, result)
        if (
            before != (campaign.sessions.read(docs.run_id), journal.read())
            or docs.read("dynamic-case-report") != original
            or docs.read("accepted:" + final["acceptance_key"]) != accepted
            or docs.read("multimodal-case-result") != (result,)
        ):
            raise ValueError("retained_context_report_history_changed")
        return dict(
            exported,
            report_key=current.report_key,
            report_projection=projection,
            new_api_requests=0,
            new_scores=0,
            report_revision_created=not bool(existing),
            original_history_preserved=True,
        )

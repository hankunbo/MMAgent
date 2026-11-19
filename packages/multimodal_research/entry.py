"""Generic adapter entry; existing stores are not discovered through FairCLIP."""

import json
import subprocess
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.preparation import persist, verify
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest


def owners():
    from apps.patient_service_assembly import OBJECT_ROOTS
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from db.runtime.repository.dataset_input import DatasetInputRepository
    from packages.dataset_adapters.service import DatasetPreparationService
    from packages.research_runtime.deployment import existing_store

    target = credential_boundary().resolve_target("runtime")
    runtime = CaseEvidenceRepository(target)
    visible = existing_store(OBJECT_ROOTS["visible"], "visible")
    hidden = existing_store("/opt/data/test/MMAgent/.local/07-patient-evaluation-v1", "evaluation")
    return (
        runtime,
        visible,
        DatasetPreparationService(DatasetInputRepository(runtime), visible, hidden),
        target,
    )


def registration_key(out, spec):
    """A new bounded attempt keeps the original registered patient identity."""

    def parent(value):
        previous = (
            value.get("previous")
            or value.get("continuation_of")
            or value.get("review_from")
            or value.get("active_continuation_of")
            or value.get("discovery_continuation_of")
        )
        failure = value.get("definition_failure_of")
        if failure is not None:
            if previous is not None:
                raise ValueError("multimodal_previous_ambiguous")
            return failure["parent"], failure["entry"]
        return previous, None

    name, seen = out.name, set()
    previous, entry_checksum = parent(spec)
    while previous is not None:
        path = Path(previous).absolute()
        if path in seen or len(seen) >= 8 or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("multimodal_previous_chain")
        seen.add(path)
        old = json.loads((path / "entry.json").read_bytes())
        if entry_checksum is not None and digest(old) != entry_checksum:
            raise ValueError("multimodal_previous_entry_changed")
        if old["input_checksum"] != spec["input_checksum"] or old["intent"] != spec["intent"]:
            raise ValueError("multimodal_previous_input_identity")
        name = path.name
        previous, entry_checksum = parent(old)
    return "mm-input:" + digest((name, spec["input_checksum"]))[7:55]


def prepare_entry(
    *,
    dataset,
    root,
    sample,
    task,
    output,
    selection=None,
    cutoff_visit=None,
    previous=None,
    known_only=False,
    review_replacement=False,
    registry=None,
    reuse_procedures=False,
    confirmed_observations_only=False,
    view_profile="overview-v1",
    independent_propositions=False,
    resilient_vision=False,
    continuation_of=None,
    review_from=None,
    active_acquisition=None,
    automatic_requirements=False,
    active_continuation_of=None,
    discovery_continuation_of=None,
    component_consumer=None,
    input_semantics=None,
    execution_profile=None,
    network_route=None,
    http_client=None,
    model_selection=None,
    vision_feedback=None,
    review_contract=None,
    review_batch_size=1,
    definition_failure_of=None,
    defer_failed_definition_expansion=False,
    source_fallback_seconds=None,
    source_fallback_authorization=None,
    retain_resolved_definition_merges=False,
    retain_closed_public_wire_failures=False,
    entry_execution_lock=None,
):
    from .campaign import configuration
    from .implementation import freeze

    # Only the new automatic entry defaults to the SLO-capable task registry.
    # An explicitly named historical registry retains its original contract.
    if registry is None:
        registry = (
            "configs/research/multimodal_tasks_v2.json"
            if automatic_requirements
            else "configs/research/multimodal_tasks_v1.json"
        )

    if automatic_requirements:
        from .requirement_catalog import DEFAULT_PROFILE

        if active_acquisition is not None:
            raise ValueError("automatic_requirements_conflicts_with_manual_profile")
        active_acquisition = DEFAULT_PROFILE
        if execution_profile == "relay-root-question-v1":
            from .capability_window import CORRECTED_SCOPE_PROFILE

            # New automatic runs use the current input-survey contract even
            # when optional Memory hints are disabled. Historical entries
            # continue to bind and read their exact saved template.
            active_acquisition = CORRECTED_SCOPE_PROFILE
        # Dataset-specific decoding is separate from medical requirement selection.
        if input_semantics is None:
            input_semantics = {"harvard_gdp": "harvard-gdp-v1", "grape": "grape-v1"}.get(dataset)

    route = None
    selected_model = None
    selected_client = None
    if http_client is not None:
        from packages.llm_provider.client_selection import load_selection

        if network_route is None or execution_profile != "relay-root-question-v1":
            raise ValueError("client_selection_requires_routed_root_entry")
        selected_client = load_selection(http_client)
    if review_contract is not None:
        from packages.llm_provider.review_rating import POLICIES

        if (
            review_contract not in POLICIES
            or network_route is None
            or execution_profile != "relay-root-question-v1"
        ):
            raise ValueError("review_rating_requires_routed_root_entry")
    if vision_feedback is not None:
        from packages.llm_provider.vision_correction import POLICY

        if (
            vision_feedback != POLICY
            or network_route is None
            or execution_profile != "relay-root-question-v1"
            or not resilient_vision
        ):
            raise ValueError("vision_feedback_requires_routed_root_entry")
    if model_selection is not None:
        from packages.llm_provider.model_selection import load_selection

        if network_route is None or execution_profile != "relay-root-question-v1":
            raise ValueError("model_selection_requires_routed_root_entry")
        selected_model = load_selection(model_selection)
    routed_backlog = bool(
        network_route is not None and definition_failure_of and defer_failed_definition_expansion
    )
    if retain_resolved_definition_merges and not routed_backlog:
        raise ValueError("definition_resolved_merge_requires_routed_parent")
    if retain_closed_public_wire_failures and not routed_backlog:
        raise ValueError("definition_public_wire_policy_requires_routed_parent")
    if (source_fallback_seconds is None) != (source_fallback_authorization is None) or (
        source_fallback_seconds is not None and not routed_backlog
    ):
        raise ValueError("definition_fallback_window_requires_routed_parent")
    if network_route is not None:
        from packages.llm_provider.network_route import load_route, route_opener

        if execution_profile != "relay-root-question-v1" or review_batch_size not in (2, 3):
            raise ValueError("network_route_requires_batched_root_entry")
        if any(
            (
                previous,
                continuation_of,
                review_from,
                active_continuation_of,
                discovery_continuation_of,
                definition_failure_of and not routed_backlog,
                defer_failed_definition_expansion and not routed_backlog,
                independent_propositions,
                component_consumer,
            )
        ):
            raise ValueError("network_route_requires_fresh_entry_not_replay")
        route = load_route(network_route)
        route_opener(route, "https://api.relayrouter.ai/v1")
    if (
        type(review_batch_size) is not int
        or review_batch_size not in (1, 2, 3)
        or (review_batch_size != 1 and execution_profile != "relay-root-question-v1")
    ):
        raise ValueError("root_review_batch_size")
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("multimodal_new_output_required")
    if execution_profile is not None and (
        execution_profile
        not in {"relay-context-v1", "relay-snapshot-context-v1", "relay-root-question-v1"}
        or not active_acquisition
        or not resilient_vision
        or any(
            (
                previous,
                continuation_of,
                review_from,
                active_continuation_of,
                discovery_continuation_of,
                component_consumer and execution_profile != "relay-snapshot-context-v1",
                independent_propositions,
            )
        )
    ):
        raise ValueError("contextual_execution_requires_fresh_active_entry")
    root_backlog = bool(
        definition_failure_of
        and defer_failed_definition_expansion
        and execution_profile == "relay-root-question-v1"
        and review_batch_size in (2, 3)
    )
    if execution_profile == "relay-root-question-v1" and any(
        (
            component_consumer,
            definition_failure_of and not root_backlog,
            defer_failed_definition_expansion and not root_backlog,
            known_only,
            review_replacement,
            confirmed_observations_only,
        )
    ):
        raise ValueError("root_question_requires_fresh_observation_entry")
    if input_semantics and (not active_acquisition or component_consumer or previous):
        raise ValueError("qualified_input_requires_fresh_active_entry")
    if input_semantics == "grape-v1" and execution_profile not in {
        "relay-snapshot-context-v1",
        "relay-root-question-v1",
    }:
        raise ValueError("grape_semantics_requires_contextual_entry")
    confirmed_backlog = bool(
        definition_failure_of
        and defer_failed_definition_expansion
        and input_semantics == "grape-v1"
        and not component_consumer
        and not root_backlog
    )
    if definition_failure_of and (
        (not component_consumer and not confirmed_backlog and not root_backlog)
        or (execution_profile != "relay-snapshot-context-v1" and not root_backlog)
    ):
        raise ValueError("definition_successor_component_profile_required")
    if defer_failed_definition_expansion and not definition_failure_of:
        raise ValueError("definition_successor_parent_required")
    prepared = prepare(
        dataset,
        root,
        sample,
        selection=selection,
        cutoff_visit=cutoff_visit,
        semantic_profile=input_semantics,
    )
    registry_path = str(Path(registry).absolute())
    intent, registry = load_intent(task, registry_path)
    automatic_selection = None
    if automatic_requirements:
        from .automatic_profile import derive

        selected_profile, automatic_selection = derive(active_acquisition, intent)
        out.mkdir(mode=0o700, parents=True)
        active_acquisition = str(out / "automatic-requirements.json")
        write_once(Path(active_acquisition), selected_profile)
    active_profile = None
    consumer = None
    if component_consumer:
        from .component_consumer import load

        consumer = load(component_consumer)
        component_consumer = str(Path(component_consumer).absolute())
        if (
            consumer.input_checksum != prepared.manifest.checksum
            or consumer.task_checksum != intent.checksum
            or not active_acquisition
            or any(
                (
                    previous,
                    continuation_of,
                    review_from,
                    active_continuation_of,
                    discovery_continuation_of,
                    independent_propositions,
                    known_only,
                    confirmed_observations_only,
                    review_replacement,
                    reuse_procedures,
                )
            )
        ):
            raise ValueError("component_consumer_fresh_active_binding")
    if discovery_continuation_of:
        from .discovery_continuation import load_parent

        if not active_acquisition or active_continuation_of:
            raise ValueError("discovery_continuation_profile_required")
        discovery_continuation_of = str(Path(discovery_continuation_of).absolute())
        load_parent(
            discovery_continuation_of,
            checksum=prepared.manifest.checksum,
            intent=intent.model_dump(mode="json"),
        )
    if active_continuation_of and not active_acquisition:
        raise ValueError("active_continuation_requires_active_profile")
    if active_acquisition:
        from .active_requirements import load

        active_acquisition = str(Path(active_acquisition).absolute())
        active_profile = load(active_acquisition)
        if (
            "requirement_catalog" not in active_profile
            and active_profile.get("input_semantics") != input_semantics
        ):
            raise ValueError("active_input_semantics_binding")
        if (
            discovery_continuation_of
            and active_profile["schema_version"] != "source-bound-acquisition-profile/4.0.0"
        ):
            raise ValueError("discovery_continuation_profile_version")
        if active_continuation_of and "definition_expansion" in active_profile:
            raise ValueError("definition_expansion_requires_fresh_campaign")
        if (
            intent.target_code not in active_profile["goals"]
            or any((previous, continuation_of, review_from, independent_propositions, known_only))
            or view_profile != "detail-v1"
            or not resilient_vision
        ):
            raise ValueError("active_acquisition_requires_fresh_single_task_detail_resilient")
        if active_continuation_of:
            from .active_continuation import load_parent

            active_continuation_of = str(Path(active_continuation_of).absolute())
            load_parent(
                active_continuation_of,
                checksum=prepared.manifest.checksum,
                intent=intent.model_dump(mode="json"),
                profile_checksum=digest(active_profile),
            )
    procedure_contract = None
    if execution_profile == "relay-root-question-v1" and reuse_procedures:
        from .reuse import root_entry_contract

        if definition_failure_of:
            raise ValueError("root_method_reuse_requires_fresh_entry")
        procedure_contract = root_entry_contract(active_profile, intent, prepared.manifest.checksum)
    completed_propositions = 0
    if consumer and active_profile["schema_version"] not in {
        "source-bound-acquisition-profile/4.0.0",
        "source-bound-acquisition-profile/5.0.0",
    }:
        raise ValueError("component_consumer_profile_version")
    if review_from:
        from .remaining_review import load

        if previous or continuation_of or not independent_propositions or not resilient_vision:
            raise ValueError("remaining_review_requires_new_independent_entry")
        review_from = str(Path(review_from).absolute())
        _, saved = load(
            review_from,
            input_checksum=prepared.manifest.checksum,
            intent=intent.model_dump(mode="json"),
        )
        completed_propositions = len(saved["trees"])
    if continuation_of:
        from .continuation import load

        if previous or not independent_propositions or not resilient_vision:
            raise ValueError("continuation_requires_new_resilient_proposition_entry")
        continuation_of = str(Path(continuation_of).absolute())
        load(
            continuation_of,
            checksum=prepared.manifest.checksum,
            intent=intent.model_dump(mode="json"),
        )
    if independent_propositions and (
        view_profile != "detail-v1" or previous or task != "glaucoma_four_hypotheses"
    ):
        raise ValueError("independent_propositions_require_fresh_detail_four_task")
    if view_profile not in {"overview-v1", "detail-v1"} or (
        view_profile == "detail-v1" and previous
    ):
        raise ValueError("multimodal_view_profile_new_attempt_required")
    if confirmed_observations_only:
        if previous is None or review_replacement:
            raise ValueError("confirmed_observations_requires_visual_previous")
        known_only = True
    if known_only and previous is None:
        raise ValueError("multimodal_known_only_requires_previous")
    if review_replacement and not (known_only and previous is not None):
        raise ValueError("multimodal_review_replacement_requires_known_previous")
    if previous is not None:
        previous = str(Path(previous).absolute())
        old = json.loads((Path(previous) / "entry.json").read_bytes())
        result = json.loads((Path(previous) / "result.json").read_bytes())
        if (
            old["input_checksum"] != prepared.manifest.checksum
            or old["intent"] != intent.model_dump(mode="json")
            or result["complete"]
        ):
            raise ValueError("multimodal_previous_attempt_binding")
    out.mkdir(mode=0o700, parents=True, exist_ok=automatic_selection is not None)
    path = persist(prepared, out / "input")
    verify(path)
    result = subprocess.run(
        [
            "/opt/conda/bin/python",
            "-m",
            "packages.multimodal_research.views_worker",
            str(path),
            str(out / "views"),
            "--profile",
            view_profile,
        ],
        timeout=180,
        capture_output=True,
    )
    if result.returncode:
        raise ValueError("multimodal_display_worker_failed:" + str(result.returncode))
    specification = dict(
        schema_version="multimodal-live-entry/1.7.0"
        if resilient_vision
        else "multimodal-live-entry/1.5.0"
        if independent_propositions
        else "multimodal-live-entry/1.4.0"
        if view_profile == "detail-v1"
        else "multimodal-live-entry/1.3.0"
        if confirmed_observations_only
        else "multimodal-live-entry/1.2.0"
        if reuse_procedures
        else "multimodal-live-entry/1.1.0"
        if review_replacement
        else "multimodal-live-entry/1.0.0",
        dataset=dataset,
        root=str(Path(root).absolute()),
        sample=sample,
        selection=selection,
        cutoff_visit=cutoff_visit,
        task=task,
        intent=intent.model_dump(mode="json"),
        registry=registry,
        registry_path=registry_path,
        input_checksum=prepared.manifest.checksum,
        provider=configuration(
            review_replacement=review_replacement,
            resilient_vision=resilient_vision,
            directional=bool(active_profile and "definition_expansion" in active_profile),
            exact_primary_status=bool(
                active_profile
                and active_profile["schema_version"]
                in {
                    "source-bound-acquisition-profile/4.0.0",
                    "source-bound-acquisition-profile/5.0.0",
                }
                and not discovery_continuation_of
            ),
            qualified_observations=bool(input_semantics),
            execution_profile=execution_profile,
            network_route=route,
            model_selection=selected_model,
            http_client_selection=selected_client,
            vision_feedback=vision_feedback,
            review_contract=review_contract,
            review_batch_size=review_batch_size,
        ).model_dump(mode="json"),
        implementation=freeze(),
        max_requests=40,
        seconds=2700 + 900 * (3 - completed_propositions)
        if resilient_vision and independent_propositions
        else 2700,
        clinical_acceptance=False,
        previous=previous,
        known_only=known_only,
        review_replacement=review_replacement,
        reuse_procedures=reuse_procedures,
    )
    if confirmed_observations_only:
        specification["confirmed_observations_only"] = True
    if view_profile == "detail-v1":
        specification["view_profile"] = view_profile
    if independent_propositions:
        specification["independent_propositions"] = True
    if resilient_vision:
        specification["resilient_vision"] = True
        specification["continuation_of"] = continuation_of
        specification["review_from"] = review_from
    if active_profile:
        specification["schema_version"] = (
            "multimodal-live-entry/1.15.0"
            if active_profile["schema_version"] == "source-bound-acquisition-profile/5.0.0"
            else (
                "multimodal-live-entry/1.12.0"
                if discovery_continuation_of
                else "multimodal-live-entry/1.13.0"
            )
            if active_profile["schema_version"] == "source-bound-acquisition-profile/4.0.0"
            else "multimodal-live-entry/1.11.0"
            if active_profile.get("observation_failure_policy")
            else "multimodal-live-entry/1.10.0"
            if "definition_expansion" in active_profile
            else "multimodal-live-entry/1.9.0"
        )
        specification["seconds"] = 5400
        specification["active_continuation_of"] = active_continuation_of
        if discovery_continuation_of:
            specification["discovery_continuation_of"] = discovery_continuation_of
        specification["active_acquisition"] = dict(
            path=active_acquisition, checksum=digest(active_profile)
        )
    if consumer:
        specification["schema_version"] = (
            "multimodal-live-entry/1.16.0"
            if active_profile["schema_version"] == "source-bound-acquisition-profile/5.0.0"
            else "multimodal-live-entry/1.14.0"
        )
        specification["component_consumer"] = dict(
            path=component_consumer, checksum=digest(consumer)
        )
    if input_semantics:
        if active_profile["schema_version"] != "source-bound-acquisition-profile/5.0.0":
            raise ValueError("qualified_input_profile_version")
        specification["schema_version"] = "multimodal-live-entry/1.17.0"
        specification["input_semantics"] = input_semantics
    if execution_profile:
        if active_profile["schema_version"] != "source-bound-acquisition-profile/5.0.0":
            raise ValueError("contextual_execution_profile_version")
        specification["schema_version"] = (
            "multimodal-live-entry/1.20.0"
            if component_consumer
            else "multimodal-live-entry/1.19.0"
            if execution_profile == "relay-snapshot-context-v1"
            else "multimodal-live-entry/1.18.0"
        )
        specification["execution_profile"] = execution_profile
    if input_semantics == "grape-v1":
        specification["schema_version"] = "multimodal-live-entry/1.23.0"
    if definition_failure_of:
        from .discovery_continuation import (
            definition_parent,
            definition_provider_transition,
            definition_successor_policy,
        )

        parent_path = str(Path(definition_failure_of).absolute())
        # This only selects a versioned verification path. The original owner
        # below must prove the actual confirmed failure and complete ledger.
        root_confirmed = bool(
            root_backlog
            and not routed_backlog
            and json.loads((Path(parent_path) / "stage-failure.json").read_bytes()).get(
                "error_code"
            )
            == "schema"
        )
        old_spec, old_plan, failure = definition_parent(
            parent_path,
            checksum=prepared.manifest.checksum,
            intent=intent.model_dump(mode="json"),
            defer=defer_failed_definition_expansion,
            confirmed_backlog=confirmed_backlog,
            root_backlog=root_backlog,
            root_confirmed=root_confirmed,
            routed_backlog=routed_backlog,
        )
        definition_successor_policy(
            old_spec,
            active_profile,
            defer=defer_failed_definition_expansion,
            root_confirmed=root_confirmed,
            routed_backlog=routed_backlog,
        )
        transition = (
            definition_provider_transition(old_spec, specification["provider"], review_contract)
            if routed_backlog
            else None
        )
        if (
            (transition is None and old_spec["provider"] != specification["provider"])
            or old_spec.get("component_consumer") != specification.get("component_consumer")
            or old_spec.get("model_selection") != selected_model
            or old_spec.get("http_client_selection") != selected_client
            or old_spec.get("vision_feedback") != vision_feedback
            or (transition is None and old_spec.get("review_contract") != review_contract)
        ):
            raise ValueError("definition_successor_provider_or_component")
        specification["schema_version"] = (
            "multimodal-live-entry/1.24.0"
            if confirmed_backlog
            else "multimodal-live-entry/1.22.0"
            if defer_failed_definition_expansion
            else "multimodal-live-entry/1.21.0"
        )
        specification["definition_failure_of"] = dict(
            parent=parent_path,
            entry=digest(old_spec),
            protocol=digest(old_plan),
            failure=digest(failure),
        )
        if transition is not None:
            specification["definition_failure_of"]["provider_transition"] = transition
        if retain_closed_public_wire_failures:
            from .discovery_continuation import PUBLIC_WIRE_POLICY

            specification["definition_failure_of"]["public_wire_policy"] = PUBLIC_WIRE_POLICY
        if retain_resolved_definition_merges:
            from .resolved_source_decisions import POLICY

            specification["definition_failure_of"]["resolved_decisions_policy"] = POLICY
        if defer_failed_definition_expansion:
            specification["definition_failure_of"]["mode"] = (
                "routed_root_source_backlog_no_source_replay"
                if routed_backlog
                else "root_confirmed_schema_backlog_no_source_replay"
                if root_confirmed
                else "root_technical_backlog_no_source_replay"
                if root_backlog
                else "confirmed_schema_backlog_no_source_replay"
                if confirmed_backlog
                else "technical_backlog_no_source_replay"
            )
    if execution_profile == "relay-root-question-v1":
        from .single_proposition import root_direction_spec

        specification["schema_version"] = "multimodal-live-entry/1.25.0"
        specification["root_direction_review"] = root_direction_spec(intent)
        if procedure_contract is not None:
            specification["procedure_reuse_contract"] = procedure_contract
        if review_batch_size != 1:
            specification["schema_version"] = "multimodal-live-entry/1.26.0"
            specification["review_batch_size"] = review_batch_size
        if root_backlog:
            specification["schema_version"] = (
                "multimodal-live-entry/1.28.0" if root_confirmed else "multimodal-live-entry/1.27.0"
            )
    if route is not None:
        if specification["schema_version"] != (
            "multimodal-live-entry/1.27.0" if routed_backlog else "multimodal-live-entry/1.26.0"
        ):
            raise ValueError("network_route_parent_entry_version")
        specification.update(
            schema_version=(
                "multimodal-live-entry/1.30.0" if routed_backlog else "multimodal-live-entry/1.29.0"
            ),
            network_route=route.model_dump(mode="json"),
        )
        if source_fallback_seconds is not None:
            from .discovery_continuation import fallback_window

            specification["source_fallback_window"] = fallback_window(
                old_plan,
                seconds=source_fallback_seconds,
                authorization=source_fallback_authorization,
            )
    if selected_client is not None:
        from packages.llm_provider.client_selection import check_entry_binding as check_client

        specification["http_client_selection"] = selected_client
        check_client(specification)
    if vision_feedback is not None:
        from packages.llm_provider.vision_correction import check_entry_binding as check_vision

        specification["vision_feedback"] = vision_feedback
        check_vision(specification)
    if review_contract is not None:
        from packages.llm_provider.review_rating import check_entry_binding as check_review

        specification["review_contract"] = review_contract
        check_review(specification)
    if selected_model is not None:
        from packages.llm_provider.model_selection import check_entry_binding

        specification["model_selection"] = selected_model
        check_entry_binding(specification)
    if entry_execution_lock is not None:
        from .execution_lease import (
            ENTRY_PROFILE,
            SOURCE_PROFILE,
            entry_implementation,
            validate_entry,
        )

        if entry_execution_lock not in {ENTRY_PROFILE, SOURCE_PROFILE}:
            raise ValueError("multimodal_entry_execution_lock_profile")
        specification["entry_execution_lock"] = dict(
            schema_version=entry_execution_lock, implementation=entry_implementation()
        )
        validate_entry(specification)
    if automatic_selection is not None:
        from .automatic_profile import check

        specification["automatic_requirement_selection"] = automatic_selection
        check(specification)
    write_once(out / "entry.json", specification)
    return dict(
        output=str(out),
        input_checksum=prepared.manifest.checksum,
        task=intent.code,
        views=len(json.loads((out / "views/views.json").read_bytes())["views"]),
        api_requests=0,
        database_writes=0,
    )


def assemble_entry(output, *, trial_initialize=False, trial_read_only=False, preflight_only=False):
    from .campaign import InputEvidence, MultimodalCampaign, configuration
    from .implementation import checked_snapshot
    from .material import MedicalSources

    out = Path(output).absolute()
    spec = json.loads((out / "entry.json").read_bytes())
    from .automatic_profile import check as check_automatic

    check_automatic(spec)
    from .execution_lease import require_entry, validate_entry

    validate_entry(spec)
    if not preflight_only:
        require_entry(out, spec)
    trial = None
    if (out / "definition-trial.json").exists():
        from .definition_trial import read as read_trial

        trial = read_trial(out)
        if trial_initialize == trial_read_only:
            raise ValueError("definition_trial_use_explicit_stage_entry")
    elif trial_initialize or trial_read_only:
        raise ValueError("definition_trial_missing")
    active = spec.get("active_acquisition")
    active_continue = spec.get("active_continuation_of")
    component = spec.get("component_consumer")
    contextual_versions = {
        "multimodal-live-entry/1.18.0": "relay-context-v1",
        "multimodal-live-entry/1.19.0": "relay-snapshot-context-v1",
        "multimodal-live-entry/1.20.0": "relay-snapshot-context-v1",
        "multimodal-live-entry/1.21.0": "relay-snapshot-context-v1",
        "multimodal-live-entry/1.22.0": "relay-snapshot-context-v1",
        "multimodal-live-entry/1.23.0": "relay-snapshot-context-v1",
        "multimodal-live-entry/1.24.0": "relay-snapshot-context-v1",
        "multimodal-live-entry/1.25.0": "relay-root-question-v1",
        "multimodal-live-entry/1.26.0": "relay-root-question-v1",
        "multimodal-live-entry/1.27.0": "relay-root-question-v1",
        "multimodal-live-entry/1.28.0": "relay-root-question-v1",
        "multimodal-live-entry/1.29.0": "relay-root-question-v1",
        "multimodal-live-entry/1.30.0": "relay-root-question-v1",
    }
    contextual = spec.get("schema_version") in contextual_versions
    if (
        contextual and contextual_versions[spec["schema_version"]] != spec.get("execution_profile")
    ) or (spec.get("execution_profile") is not None and not contextual):
        raise ValueError("contextual_execution_entry_version")
    if contextual and (
        trial
        or (
            component
            and spec["schema_version"]
            not in {
                "multimodal-live-entry/1.20.0",
                "multimodal-live-entry/1.21.0",
                "multimodal-live-entry/1.22.0",
            }
        )
        or not active
        or active_continue
        or spec.get("discovery_continuation_of")
    ):
        raise ValueError("contextual_execution_requires_fresh_active_entry")
    routed_backlog = spec.get("schema_version") == "multimodal-live-entry/1.30.0"
    from packages.llm_provider.client_selection import check_entry_binding as check_client

    check_client(spec)
    from packages.llm_provider.model_selection import check_entry_binding

    check_entry_binding(spec)
    from packages.llm_provider.vision_correction import check_entry_binding as check_vision

    check_vision(spec)
    from packages.llm_provider.review_rating import check_entry_binding as check_review

    check_review(spec)
    root_network = routed_backlog or spec.get("schema_version") == "multimodal-live-entry/1.29.0"
    if spec.get("source_fallback_window") is not None and not routed_backlog:
        raise ValueError("definition_fallback_window_requires_routed_parent")
    if root_network != ("network_route" in spec):
        raise ValueError("network_route_entry_version")
    if root_network:
        from packages.llm_provider.network_route import route_opener

        route_opener(spec["network_route"], spec["provider"]["transport"]["base_url"])
        if not routed_backlog and (
            spec.get("definition_failure_of") or spec.get("defer_failed_definition_expansion")
        ):
            raise ValueError("network_route_requires_fresh_entry_not_replay")
    root_confirmed = spec.get("schema_version") == "multimodal-live-entry/1.28.0"
    root_backlog = (
        routed_backlog
        or root_confirmed
        or spec.get("schema_version") == "multimodal-live-entry/1.27.0"
    )
    root_batched = (
        root_network or root_backlog or spec.get("schema_version") == "multimodal-live-entry/1.26.0"
    )
    batch_size = spec.get("review_batch_size", 1)
    if (
        type(batch_size) is not int
        or (root_batched and batch_size not in (2, 3))
        or (not root_batched and "review_batch_size" in spec)
    ):
        raise ValueError("root_review_batch_size")
    root_question = root_batched or spec.get("schema_version") == "multimodal-live-entry/1.25.0"
    from .reuse import check_root_entry_contract

    root_method_reuse = check_root_entry_contract(spec)
    if root_question != ("root_direction_review" in spec) or (
        root_question
        and any(
            spec.get(k)
            and not (root_backlog and k == "definition_failure_of")
            and not (root_method_reuse and k == "reuse_procedures")
            for k in (
                "component_consumer",
                "definition_failure_of",
                "previous",
                "continuation_of",
                "review_from",
                "known_only",
                "review_replacement",
                "reuse_procedures",
                "confirmed_observations_only",
                "independent_propositions",
            )
        )
    ):
        raise ValueError("root_question_entry_binding")
    active_v124 = spec.get("schema_version") == "multimodal-live-entry/1.24.0"
    active_v123 = active_v124 or spec.get("schema_version") == "multimodal-live-entry/1.23.0"
    grape_qualified = active_v123 or (root_question and spec.get("input_semantics") == "grape-v1")
    if grape_qualified and (
        spec.get("input_semantics") != "grape-v1" or spec["dataset"] != "grape"
    ):
        raise ValueError("qualified_input_entry_version")
    harvard_qualified = spec.get("schema_version") == "multimodal-live-entry/1.17.0" or (
        contextual and spec.get("input_semantics") == "harvard-gdp-v1"
    )
    if harvard_qualified != (spec.get("input_semantics") == "harvard-gdp-v1") or (
        spec.get("input_semantics") is not None and not (harvard_qualified or grape_qualified)
    ):
        raise ValueError("qualified_input_entry_version")
    active_v117 = harvard_qualified or grape_qualified
    active_v116 = spec.get("schema_version") == "multimodal-live-entry/1.16.0"
    active_v115 = (
        contextual
        or active_v117
        or active_v116
        or spec.get("schema_version") == "multimodal-live-entry/1.15.0"
    )
    active_v114 = spec.get("schema_version") == "multimodal-live-entry/1.14.0"
    consumer = None
    failure_reference = spec.get("definition_failure_of")
    if (
        spec.get("schema_version")
        in {
            "multimodal-live-entry/1.21.0",
            "multimodal-live-entry/1.22.0",
            "multimodal-live-entry/1.24.0",
            "multimodal-live-entry/1.27.0",
            "multimodal-live-entry/1.28.0",
            "multimodal-live-entry/1.30.0",
        }
    ) != bool(failure_reference):
        raise ValueError("definition_successor_entry_version")
    defer_failure = bool(
        failure_reference and failure_reference.get("mode") == "technical_backlog_no_source_replay"
    )
    if (spec.get("schema_version") == "multimodal-live-entry/1.22.0") != defer_failure:
        raise ValueError("definition_successor_deferral_version")
    if root_backlog != bool(
        failure_reference
        and failure_reference.get("mode")
        == (
            "routed_root_source_backlog_no_source_replay"
            if routed_backlog
            else "root_confirmed_schema_backlog_no_source_replay"
            if root_confirmed
            else "root_technical_backlog_no_source_replay"
        )
    ):
        raise ValueError("definition_successor_deferral_version")
    if active_v124 != bool(
        failure_reference
        and failure_reference.get("mode") == "confirmed_schema_backlog_no_source_replay"
    ):
        raise ValueError("definition_successor_confirmed_backlog_version")
    if (
        active_v114
        or active_v116
        or spec.get("schema_version")
        in {
            "multimodal-live-entry/1.20.0",
            "multimodal-live-entry/1.21.0",
            "multimodal-live-entry/1.22.0",
        }
    ) != bool(component):
        raise ValueError("component_consumer_entry_version")
    if component:
        from .component_consumer import load

        consumer = load(component["path"], expected=component["checksum"])
        if (
            not active
            or (
                trial
                and trial["schema_version"]
                not in {
                    "multimodal-definition-trial/1.10.0",
                    "multimodal-definition-trial/1.11.0",
                    "multimodal-definition-trial/1.12.0",
                    "multimodal-definition-trial/1.13.0",
                    "multimodal-definition-trial/1.14.0",
                    "multimodal-definition-trial/1.15.0",
                    "multimodal-definition-trial/1.16.0",
                    "multimodal-definition-trial/1.17.0",
                    "multimodal-definition-trial/1.18.0",
                    "multimodal-definition-trial/1.7.0",
                    "multimodal-definition-trial/1.8.0",
                    "multimodal-definition-trial/1.9.0",
                }
            )
            or any(
                spec.get(k)
                for k in (
                    "previous",
                    "continuation_of",
                    "review_from",
                    "active_continuation_of",
                    "discovery_continuation_of",
                    "independent_propositions",
                    "known_only",
                    "confirmed_observations_only",
                    "review_replacement",
                    "reuse_procedures",
                )
            )
        ):
            raise ValueError("component_consumer_fresh_active_binding")
    active_v113 = (
        active_v115 or active_v114 or spec.get("schema_version") == "multimodal-live-entry/1.13.0"
    )
    active_v112 = active_v113 or spec.get("schema_version") == "multimodal-live-entry/1.12.0"
    active_v111 = active_v112 or spec.get("schema_version") == "multimodal-live-entry/1.11.0"
    active_v110 = active_v111 or spec.get("schema_version") == "multimodal-live-entry/1.10.0"
    active_v19 = spec.get("schema_version") in {
        "multimodal-live-entry/1.9.0",
        "multimodal-live-entry/1.10.0",
        "multimodal-live-entry/1.11.0",
        "multimodal-live-entry/1.12.0",
        "multimodal-live-entry/1.13.0",
        "multimodal-live-entry/1.14.0",
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
    if active_continue and not (active and active_v19):
        raise ValueError("active_continuation_entry_version")
    if active:
        from .active_requirements import load

        active_profile = load(active["path"])
        if active_v115 != (
            active_profile["schema_version"] == "source-bound-acquisition-profile/5.0.0"
        ):
            raise ValueError("adaptive_gap_profile_entry_version")
        if active_v112 != (
            active_profile["schema_version"]
            in {"source-bound-acquisition-profile/4.0.0", "source-bound-acquisition-profile/5.0.0"}
        ):
            raise ValueError("discovery_policy_entry_version")
        if active_v111 != (active_profile.get("observation_failure_policy") is not None):
            raise ValueError("observation_closure_entry_version")
        if active_v110 != ("definition_expansion" in active_profile):
            raise ValueError("definition_expansion_entry_version")
        if active_continue and "definition_expansion" in active_profile:
            raise ValueError("definition_expansion_requires_fresh_campaign")
        if (
            digest(active_profile) != active["checksum"]
            or spec["intent"]["target_code"] not in active_profile["goals"]
        ):
            raise ValueError("active_acquisition_profile_changed")
        if (
            spec.get("previous")
            or spec.get("continuation_of")
            or spec.get("review_from")
            or spec.get("independent_propositions")
            or not spec.get("resilient_vision")
            or spec.get("view_profile") != "detail-v1"
        ):
            raise ValueError("active_acquisition_entry_binding")
    replacement = spec.get("review_replacement", False)
    confirmed = spec.get("confirmed_observations_only", False)
    detailed = spec.get("view_profile") == "detail-v1"
    independent = spec.get("independent_propositions", False)
    resilient = spec.get("resilient_vision", False)
    continuation = spec.get("continuation_of")
    review_from = spec.get("review_from")
    if review_from and (continuation or spec.get("previous") or not independent or not resilient):
        raise ValueError("remaining_review_entry_binding")
    completed_propositions = 0
    if review_from:
        from .remaining_review import load

        _, saved = load(review_from, input_checksum=spec["input_checksum"], intent=spec["intent"])
        completed_propositions = len(saved["trees"])
    expected_seconds = (
        5400
        if active_v19
        else (2700 + 900 * (3 - completed_propositions) if resilient and independent else 2700)
    )
    if spec["seconds"] != expected_seconds:
        raise ValueError("multimodal_case_deadline_binding")
    if type(resilient) is not bool or (
        continuation and (not resilient or not independent or spec.get("previous"))
    ):
        raise ValueError("multimodal_continuation_entry_binding")
    if type(independent) is not bool or (
        independent
        and (not detailed or spec.get("previous") or spec["task"] != "glaucoma_four_hypotheses")
    ):
        raise ValueError("independent_propositions_entry_binding")
    if spec.get("view_profile") not in {None, "detail-v1"} or (detailed and spec.get("previous")):
        raise ValueError("multimodal_view_profile_binding")
    if type(confirmed) is not bool or (
        confirmed and (replacement or not spec.get("known_only") or not spec.get("previous"))
    ):
        raise ValueError("confirmed_observations_entry_binding")
    reuse = spec.get("reuse_procedures", False)
    if type(reuse) is not bool:
        raise ValueError("multimodal_reuse_selection")
    if type(replacement) is not bool or (
        replacement and not (spec.get("known_only") and spec.get("previous"))
    ):
        raise ValueError("multimodal_review_replacement_binding")
    if (
        spec["schema_version"]
        != (
            (
                "multimodal-live-entry/1.30.0"
                if routed_backlog
                else "multimodal-live-entry/1.29.0"
                if root_network
                else "multimodal-live-entry/1.28.0"
                if root_confirmed
                else "multimodal-live-entry/1.27.0"
                if root_backlog
                else "multimodal-live-entry/1.26.0"
                if root_batched
                else "multimodal-live-entry/1.25.0"
                if root_question
                else "multimodal-live-entry/1.24.0"
                if active_v124
                else "multimodal-live-entry/1.23.0"
                if active_v123
                else "multimodal-live-entry/1.22.0"
                if defer_failure
                else "multimodal-live-entry/1.21.0"
                if contextual and failure_reference and component
                else "multimodal-live-entry/1.20.0"
                if contextual and component
                else {
                    "relay-context-v1": "multimodal-live-entry/1.18.0",
                    "relay-snapshot-context-v1": "multimodal-live-entry/1.19.0",
                }[spec["execution_profile"]]
                if contextual
                else "multimodal-live-entry/1.17.0"
                if active_v117
                else "multimodal-live-entry/1.16.0"
                if active_v116
                else "multimodal-live-entry/1.15.0"
                if active_v115
                else "multimodal-live-entry/1.14.0"
                if active_v114
                else "multimodal-live-entry/1.13.0"
                if active_v113
                else "multimodal-live-entry/1.12.0"
                if active_v112
                else "multimodal-live-entry/1.11.0"
                if active_v111
                else "multimodal-live-entry/1.10.0"
                if active_v110
                else "multimodal-live-entry/1.9.0"
                if active_v19
                else "multimodal-live-entry/1.8.0"
            )
            if active
            else "multimodal-live-entry/1.7.0"
            if resilient
            else "multimodal-live-entry/1.5.0"
            if independent
            else "multimodal-live-entry/1.4.0"
            if detailed
            else "multimodal-live-entry/1.3.0"
            if confirmed
            else "multimodal-live-entry/1.2.0"
            if reuse
            else "multimodal-live-entry/1.1.0"
            if replacement
            else "multimodal-live-entry/1.0.0"
        )
        or spec["implementation"] != checked_snapshot(out, spec["implementation"])
        or spec["provider"]
        != configuration(
            review_replacement=replacement,
            resilient_vision=resilient,
            directional=bool(active and "definition_expansion" in active_profile),
            exact_primary_status=active_v113,
            qualified_observations=active_v117,
            execution_profile=spec.get("execution_profile"),
            network_route=spec.get("network_route"),
            model_selection=spec.get("model_selection"),
            http_client_selection=spec.get("http_client_selection"),
            vision_feedback=spec.get("vision_feedback"),
            review_contract=spec.get("review_contract"),
            review_batch_size=batch_size,
        ).model_dump(mode="json")
    ):
        raise ValueError("multimodal_entry_implementation_changed")
    prepared = prepare(
        spec["dataset"],
        spec["root"],
        spec["sample"],
        selection=spec["selection"],
        cutoff_visit=spec["cutoff_visit"],
        semantic_profile=spec.get("input_semantics"),
    )
    if (
        prepared.manifest.checksum != spec["input_checksum"]
        or verify(out / "input/preparation.json") != prepared.manifest
    ):
        raise ValueError("multimodal_original_inputs_changed")
    if routed_backlog and spec.get("source_fallback_window") is not None:
        from .discovery_continuation import check_fallback_window

        parent_plan = json.loads((Path(failure_reference["parent"]) / "protocol.json").read_bytes())
        check_fallback_window(spec["source_fallback_window"], parent_plan)
    task, registry = load_intent(
        spec["task"], spec.get("registry_path", "configs/research/multimodal_tasks_v1.json")
    )
    if task.model_dump(mode="json") != spec["intent"] or registry != spec["registry"]:
        raise ValueError("multimodal_task_changed")
    if root_question:
        from .single_proposition import root_direction_spec

        if spec["root_direction_review"] != root_direction_spec(task):
            raise ValueError("root_question_task_binding")
    from .entry_preflight import check_local_route

    route = check_local_route(prepared.manifest, task, active_profile if active else None, consumer)
    if preflight_only:
        return dict(
            status="local_preflight_passed",
            schema_version=spec["schema_version"],
            route=route,
            current_owner_qualification="pending_assembly_recheck",
            api_requests=0,
            database_writes=0,
            tool_dispatches=0,
            **(
                dict(
                    review_batch_size=batch_size,
                    model_review_capacity=spec["provider"]["review_batching"]["max_batches"]
                    * batch_size,
                )
                if root_batched
                else {}
            ),
        )
    from .execution_lease import check_current

    check_current(out)
    runtime, objects, inputs, target = owners()
    if consumer:
        from .component_consumer import registration as read_component_registration

        registration = read_component_registration(inputs, consumer, prepared.manifest, task)
    elif trial_read_only:
        registration = inputs.repository.read_registration(
            registration_key(out, spec), prepared.manifest.checksum
        )
        if registration.schema_version != "dataset-registration/4.0.0":
            raise ValueError("definition_trial_registration_version")
    else:
        registration = inputs.register_multimodal(prepared, key=registration_key(out, spec))
    base = InputEvidence(
        registered=registration,
        inputs=inputs,
        runtime=runtime,
        objects=objects,
        prepared=prepared,
        task=task,
        output=out,
        sources=MedicalSources(),
    )
    if active:
        from .active_acquisition import install

        base.active_seconds = trial["seconds"] if trial else spec["seconds"]
        base.active_schedule_v19 = active_v19
        if failure_reference:
            from .discovery_continuation import (
                definition_parent,
                definition_provider_transition,
                definition_successor_policy,
            )

            old_spec, old_plan, failure = definition_parent(
                failure_reference["parent"],
                checksum=prepared.manifest.checksum,
                intent=task.model_dump(mode="json"),
                defer=defer_failure or active_v124 or root_backlog,
                confirmed_backlog=active_v124,
                root_backlog=root_backlog,
                root_confirmed=root_confirmed,
                routed_backlog=routed_backlog,
            )
            definition_successor_policy(
                old_spec,
                active_profile,
                defer=defer_failure or active_v124 or root_backlog,
                root_confirmed=root_confirmed,
                routed_backlog=routed_backlog,
            )
            expected_reference = dict(
                parent=failure_reference["parent"],
                entry=digest(old_spec),
                protocol=digest(old_plan),
                failure=digest(failure),
            )
            if defer_failure or active_v124 or root_backlog:
                expected_reference["mode"] = (
                    "routed_root_source_backlog_no_source_replay"
                    if routed_backlog
                    else "root_confirmed_schema_backlog_no_source_replay"
                    if root_confirmed
                    else "root_technical_backlog_no_source_replay"
                    if root_backlog
                    else "confirmed_schema_backlog_no_source_replay"
                    if active_v124
                    else "technical_backlog_no_source_replay"
                )
            transition = (
                definition_provider_transition(
                    old_spec, spec["provider"], spec.get("review_contract")
                )
                if routed_backlog
                else None
            )
            if transition is not None:
                expected_reference["provider_transition"] = transition
            from .discovery_continuation import public_wire_reference
            from .resolved_source_decisions import policy_reference

            expected_reference.update(policy_reference(failure_reference, routed=routed_backlog))
            expected_reference.update(
                public_wire_reference(failure_reference, routed=routed_backlog)
            )
            if failure_reference != expected_reference:
                raise ValueError("definition_successor_reference_changed")
            start = json.loads(
                (Path(failure_reference["parent"]) / "active-start.json").read_bytes()
            )
            if (
                start["deadline"] != old_plan["deadline"]
                or start["input_checksum"] != prepared.manifest.checksum
            ):
                raise ValueError("definition_successor_original_clock")
            if routed_backlog and spec.get("source_fallback_window") is not None:
                from .discovery_continuation import check_fallback_window

                start = dict(
                    start,
                    **check_fallback_window(spec["source_fallback_window"], old_plan),
                )
                base.source_fallback_window = spec["source_fallback_window"]
            write_once(out / "active-start.json", start)
        if consumer:
            install(
                base,
                active_profile,
                defer_native=True,
                read_only=(out / "protocol.json").exists(),
            )
        elif trial:
            install(base, active_profile, defer_native=True, read_only=trial_read_only)
        elif active_continue:
            from .active_continuation import reuse_dependencies

            discoverer, native_runner = reuse_dependencies(base, active_continue, active_profile)
            install(base, active_profile, discoverer=discoverer, native_runner=native_runner)
        elif spec.get("discovery_continuation_of"):
            from .discovery_continuation import reuse_native

            install(
                base,
                active_profile,
                native_runner=reuse_native(base, spec["discovery_continuation_of"]),
            )
        elif root_backlog:
            from .discovery_continuation import reuse_failed_native

            # No new GPU process: current Tool/Runtime must confirm the exact
            # parent's original bundle. Routed Source UNKNOWN stays quarantined;
            # it does not make a separately confirmed native artifact unknown.
            def saved_native(*args, **kwargs):
                return reuse_failed_native(
                    base,
                    failure_reference,
                    following_provider=spec["provider"],
                    following_review_contract=spec.get("review_contract"),
                )(*args, **kwargs)

            install(base, active_profile, native_runner=saved_native)
        else:
            install(base, active_profile)
    if spec.get("entry_execution_lock") is not None:
        base.entry_execution_lock = spec["entry_execution_lock"]
    base.previous = spec.get("previous")
    base.resilient_vision = resilient
    base.exact_primary_status = active_v113
    if root_network:
        base.network_route = spec["network_route"]
    if spec.get("model_selection") is not None:
        base.model_selection = spec["model_selection"]
    if spec.get("vision_feedback") is not None:
        base.vision_feedback = spec["vision_feedback"]
    if spec.get("review_contract") is not None:
        base.review_contract = spec["review_contract"]
    if spec.get("http_client_selection") is not None:
        base.http_client_selection = spec["http_client_selection"]
    if contextual:
        base.execution_profile = spec["execution_profile"]
    if root_question:
        base.root_direction_review = spec["root_direction_review"]
        base.review_batch_size = batch_size
    if base.detailed != detailed:
        raise ValueError("multimodal_view_profile_manifest_binding")
    base.known_only = spec.get("known_only", False)
    base.review_replacement = replacement
    if type(base.known_only) is not bool or (base.known_only and base.previous is None):
        raise ValueError("multimodal_known_only_binding")
    if base.known_only and not confirmed:
        for row in base.ledger:
            if row["id"] not in base.observations:
                row["missing_reason"] = "visual_attempt_not_replayed_review_known_measurements_only"
    from .proposition_campaign import PropositionCampaign

    if trial:
        from .definition_trial import DefinitionTrialCampaign

        campaign = DefinitionTrialCampaign(base, target, campaign_id=out.name, trial=trial)
    else:
        campaign = (PropositionCampaign if independent else MultimodalCampaign)(
            base, target, campaign_id=out.name
        )
    if (active_v117 or contextual) and campaign.config.model_dump(mode="json") != spec["provider"]:
        raise ValueError("qualified_provider_not_installed")
    if consumer:
        from .component_consumer import install_campaign

        install_campaign(campaign, consumer, component)
    if failure_reference:
        from .discovery_continuation import ConfirmedDefinitionContinuation

        campaign.definition_continuation = ConfirmedDefinitionContinuation(
            campaign, failure_reference
        )
    if spec.get("discovery_continuation_of"):
        if not active_v112 or active_v113:
            raise ValueError("discovery_continuation_entry_version")
        from .discovery_continuation import DiscoveryContinuation

        campaign.discovery_continuation = DiscoveryContinuation(
            campaign, spec["discovery_continuation_of"]
        )
    if active_continue:
        from .active_continuation import ActiveContinuation, ordered_targets

        campaign.continuation = ActiveContinuation(campaign, active_continue)
        pending = tuple(t for t in campaign.targets if t not in campaign.continuation.attempted)
        campaign.targets = ordered_targets(
            base.ledger, pending, {**base.observations, **campaign.continuation.values}
        )
    if continuation:
        from .continuation import Continuation

        campaign.continuation = Continuation(campaign, continuation)
        campaign.targets = tuple(
            t for t in campaign.targets if t not in campaign.continuation.values
        )
    if review_from:
        from .remaining_review import RemainingReview

        campaign.remaining_review = RemainingReview(campaign, review_from)
        campaign.targets = ()
    if confirmed:
        from .confirmed_observations import ConfirmedObservations

        campaign.confirmed_observations = ConfirmedObservations(campaign)
    if reuse:
        from .reuse import ProcedureReuse, repository

        campaign.procedure_reuse = ProcedureReuse(campaign, repository(campaign))
        old_protocol = out / "protocol.json"
        if old_protocol.exists():
            campaign.procedure_reuse.restore(
                json.loads(old_protocol.read_bytes())["procedure_reuse"]
            )
        else:
            from psycopg import OperationalError

            try:
                campaign.procedure_reuse.prepare()
            except (OSError, OperationalError) as error:
                campaign.procedure_reuse = ProcedureReuse(campaign, None)
                campaign.procedure_reuse.prepare()
                campaign.procedure_reuse.events.append(
                    dict(
                        reason="memory_technical_unavailable",
                        error_type=type(error).__name__,
                        action="current_owner_baseline",
                    )
                )
    return campaign

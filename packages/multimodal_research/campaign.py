"""Task-v2 composition of the original Planner, Provider and acceptance ports.

Views are display derivatives of the original registered inputs, not new medical
measurements. Every external dispatch shares one case budget and deadline.
"""

import importlib.util
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_case_materials import ref
from apps.vectorengine_single_case import load_config, write_once
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository
from packages.evaluation_preparation.dynamic_campaign import Campaign
from packages.evaluation_preparation.dynamic_stages import TargetReader
from packages.evidence_details.acceptance_models import DomainState, read_receipt, read_records
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ImageBinding, ObservationInput
from packages.llm_provider.provider import ProviderError
from packages.object_store import checksum
from packages.research_runtime.task_context import TaskRuns, check_current_inputs

from . import acquisition_scope, material


def configuration(
    *,
    network_route=None,
    model_selection=None,
    vision_feedback=None,
    review_contract=None,
    http_client_selection=None,
    **kwargs,
):
    config = _configuration(**kwargs)
    if network_route is not None:
        from packages.llm_provider.config import with_network_route

        config = with_network_route(config, network_route)
    if model_selection is not None:
        from packages.llm_provider.model_selection import apply_selection

        if kwargs.get("execution_profile") != "relay-root-question-v1":
            raise ValueError("model_selection_root_profile_required")
        config = apply_selection(config, model_selection)
    from packages.llm_provider.vision_correction import apply

    if vision_feedback is not None and kwargs.get("execution_profile") != "relay-root-question-v1":
        raise ValueError("vision_feedback_root_profile_required")
    config = apply(config, vision_feedback)
    from packages.llm_provider.review_rating import apply as apply_review

    if review_contract is not None and kwargs.get("execution_profile") != "relay-root-question-v1":
        raise ValueError("review_rating_root_profile_required")
    config = apply_review(config, review_contract)
    from packages.llm_provider.client_selection import apply_selection as apply_client

    if (
        http_client_selection is not None
        and kwargs.get("execution_profile") != "relay-root-question-v1"
    ):
        raise ValueError("client_selection_root_profile_required")
    return apply_client(config, http_client_selection)


def _configuration(
    *,
    review_replacement=False,
    resilient_vision=False,
    directional=False,
    exact_primary_status=False,
    qualified_observations=False,
    execution_profile=None,
    review_batch_size=1,
):
    """Explicit derived config; old released requests/config files stay unchanged."""
    if (
        type(review_batch_size) is not int
        or review_batch_size not in (1, 2, 3)
        or (review_batch_size != 1 and execution_profile != "relay-root-question-v1")
    ):
        raise ValueError("root_review_batch_size")
    if execution_profile is not None:
        if execution_profile not in {
            "relay-context-v1",
            "relay-snapshot-context-v1",
            "relay-root-question-v1",
        } or not (directional and resilient_vision):
            raise ValueError("contextual_execution_configuration")
        from packages.llm_provider.config import review_format_upgrade

        from .context_review import configuration as context_configuration

        config = review_format_upgrade(
            context_configuration(
                configuration(
                    review_replacement=review_replacement,
                    resilient_vision=resilient_vision,
                    directional=directional,
                    exact_primary_status=exact_primary_status,
                    qualified_observations=qualified_observations,
                ),
                profile=(
                    "configs/agents/relayrouter_snapshot_evidence_v1.json"
                    if execution_profile in {"relay-snapshot-context-v1", "relay-root-question-v1"}
                    else "configs/agents/relayrouter_generic_evidence_v1.json"
                ),
            )
        )
        if execution_profile == "relay-root-question-v1":
            from .root_review_window import configuration as root_configuration

            return root_configuration(
                config, citation_projection=True, batch_size=review_batch_size
            )
        return config
    config = load_config("configs/agents/vectorengine_generic_evidence_v1.json")
    raw = config.model_dump(mode="json")
    if review_replacement:
        raw["review_batching"]["batch_size"] = 1
    role = next(r for r in raw["roles"] if r["role"] == "vision")
    role["prompt_version"] = (
        "vision-partial-multimodal-1.2.0" if resilient_vision else "vision-partial-multimodal-1.1.0"
    )
    role["prompt_checksum"] = digest(
        (Path("configs/prompts/agents") / (role["prompt_version"] + ".txt")).read_text()
    )
    if resilient_vision:
        role["max_output_tokens"] = 2048
    if directional:
        from packages.llm_provider.config import DirectionalProviderConfig

        raw.update(
            schema_version="agent-provider-config/8.0.0",
            direction_policy="direction-assessment/1.0.0",
        )
        review = next(r for r in raw["roles"] if r["role"] == "review")
        review["prompt_version"] = (
            "review-direction-2.2.0"
            if qualified_observations
            else "review-direction-2.1.0"
            if exact_primary_status
            else "review-direction-2.0.0"
        )
        if qualified_observations:
            raw["review_batching"]["batch_size"] = 1
            raw["transport"]["timeout_seconds"] = 300
            raw["transport"]["max_retries"] = 0
        review["prompt_checksum"] = digest(
            (Path("configs/prompts/agents") / (review["prompt_version"] + ".txt")).read_text()
        )
        return validate(DirectionalProviderConfig, raw)
    return validate(type(config), raw)


class InputEvidence:
    def __init__(self, *, registered, inputs, runtime, objects, prepared, task, output, sources):
        self.visible_value, self.inputs = registered.visible, inputs
        self.runtime, self.objects, self.prepared, self.task = runtime, objects, prepared, task
        self.output, self.sources = Path(output), sources
        self.root = self.visible_value.case.artifacts[0]
        self.binding = dict(case_code=self.visible_value.case.case_code)
        self.manifest = prepared.manifest
        self.views_path = self.output / "views/views.json"
        self.views = json.loads(self.views_path.read_bytes())
        if self.views["input_checksum"] != self.manifest.checksum:
            raise ValueError("multimodal_views_input")
        self.view_checksum = checksum(self.views_path.read_bytes())
        self.ledger = material.requirements(self.manifest, sources.initial, task)
        payloads = dict(prepared.payload_bytes)
        self.observations = {}
        for r in self.ledger:
            if r["component"] is None:
                continue
            c = next(c for c in self.manifest.components if c.id == r["component"])
            value = material.numeric(c, payloads)
            if value is not None:
                self.observations[r["id"]] = material.observation(r["id"], value)
        self.detailed = self.views.get("schema_version") in {
            "multimodal-inspection-views/1.1.0",
            "multimodal-inspection-views/1.2.0",
        }
        if self.detailed:
            from .numeric_detail import expand

            self.ledger, self.observations = expand(
                self.ledger, self.observations, self.manifest, payloads
            )
        if task.output_kind == "observed_change":
            paired, values = material.paired_numeric(self.manifest, payloads, sources.initial)
            self.ledger.extend(paired)
            self.observations.update(values)
        self.resources = (ref("resource", self.root.asset_code, self.root.checksum),)
        self.contexts = {}

    def check(self):
        self.check_definition_materials()
        if getattr(self, "component_observations", None) is not None:
            self.component_observations.check()

    def check_definition_materials(self):
        """Current public-definition dependencies, not tool-result consumption.

        Expansion reads the input manifest, source definitions and requirements;
        it does not read the installed component measurement. Review and ordinary
        campaign guards still use check(), including that component's owner.
        This is a fresh read on every call, never a cached qualification.
        """
        from .execution_lease import check_current

        check_current()
        self.read_dependency_input()
        if getattr(self, "active_owner", None) is not None:
            self.active_owner.check()
        if getattr(self, "definition_current", None) is not None:
            self.definition_current()
        if getattr(self, "component_observations", None) is not None:
            self.component_observations.check_projection()

    def read_dependency_input(self):
        """Original current input/source checks, without recursion into definition reads."""
        current = self.inputs.read_visible(self.binding["case_code"])
        if current != self.visible_value:
            raise ValueError("multimodal_current_input_changed")
        if self.sources.read() != self.sources.initial:
            raise ValueError("multimodal_current_medical_source_changed")
        if checksum(self.views_path.read_bytes()) != self.view_checksum:
            raise ValueError("multimodal_view_manifest_changed")
        if current.components != self.manifest or current.components != self.prepared.manifest:
            raise ValueError("multimodal_dependency_manifest_binding")
        return current.components

    def read_dependency_fact(self, contract):
        """Exact visible-input read, not a quality gate or model validation claim."""
        manifest = self.read_dependency_input()
        selector = dict(modality=contract.modality, concept=contract.concept)
        components = [
            c
            for c in manifest.components
            if c.modality == contract.modality
            and c.concept == contract.concept
            and c.status != "withheld"
        ]
        from packages.evidence_reuse.definitions import OwnerDependencyContract

        if isinstance(contract, OwnerDependencyContract) and contract.predicate != "unresolved":
            from packages.evidence_reuse.definition_dependencies import alias_component

            components = [
                alias_component(manifest, contract.input_alias, contract.modality, contract.concept)
            ]
        value = dict(
            input_checksum=manifest.checksum,
            selector=selector,
            status="patient_input_missing",
            components=[digest(c) for c in components],
        )
        if len(components) > 1:
            return dict(value, status="dependency_input_ambiguous")
        if not components or components[0].status not in {"obtained", "partial"}:
            return value
        c = components[0]
        payloads = dict(self.prepared.payload_bytes)
        declared = {p.id: p for p in manifest.payloads}
        proof = []
        for key in c.payload_ids:
            record, data = declared.get(key), payloads.get(key)
            if (
                record is None
                or data is None
                or (checksum(data), len(data)) != (record.checksum, record.byte_size)
            ):
                raise ValueError("multimodal_dependency_payload_binding")
            proof.append(dict(id=key, checksum=record.checksum, byte_size=record.byte_size))
        return dict(
            value,
            status="input_available",
            component_id=c.id,
            component_checksum=digest(c),
            payloads=proof,
            numeric=material.numeric(c, payloads),
            qualification="visible_input_only_not_method_or_quality_approval",
        )

    def view(self, target):
        from .input_scope import current_entry, is_target

        alternative = getattr(self, "alternative_view", None)
        if alternative is not None and target == alternative.target:
            return alternative.view()
        r = (
            {"component": current_entry(self, target)["task"]["component_id"]}
            if is_target(target)
            else next(r for r in self.ledger if r["id"] == target)
        )
        v = next(v for v in self.views["views"] if v["component"] == r["component"])
        if v["status"] != "obtained":
            raise ValueError("multimodal_view_unavailable")
        name = checksum(r["component"].encode())[7:31] + ".png"
        if v["file"] != name:
            raise ValueError("multimodal_view_path")
        path = self.output / "views" / name
        if path.is_symlink():
            raise ValueError("multimodal_view_symlink")
        data = path.read_bytes()
        if len(data) > 4194304 or checksum(data) != v["checksum"]:
            raise ValueError("multimodal_view_bytes_changed")
        image = ImageBinding(
            reference=ref("image", "ASSET:VIEW:" + digest(target)[7:23], v["checksum"]),
            media_type="image/png",
            width=v["width"],
            height=v["height"],
            coordinate_system="pixel_xy_top_left",
            location=(0, 0, v["width"], v["height"]),
            preprocessing=ref("preprocessing", "POLICY:MULTIMODAL_VIEW", digest(v)),
        )
        return image, data, v

    def image(self, bound):
        from packages.planner_control.models import Reference

        bound = validate(Reference, bound)
        rows = list(self.ledger) + [
            {"id": e["target"]} for e in getattr(self, "input_scope_plan", {}).get("entries", [])
        ]
        for r in rows:
            if bound.code == "ASSET:VIEW:" + digest(r["id"])[7:23]:
                expected, raw, _ = self.view(r["id"])
                if expected.reference != bound:
                    raise ValueError("multimodal_image_binding")
                acquisition_scope.check(self, r["id"])
                return raw
        raise ValueError("multimodal_unknown_image")


class PartialTargetReader(TargetReader):
    partial_observation_contract = "1.0.0"


def review_observations(base, observations):
    selected = getattr(base, "review_target_ids", None)
    if selected is None:
        return observations
    return {k: v for k, v in observations.items() if k in selected}


class Phase:
    def __init__(self, base, *, target=None, observation_reader=None, model_reader=None):
        self.base, self.target, self.observation_reader = base, target, observation_reader
        self.targeted_acquisition = False

    def __getattr__(self, key):
        return getattr(self.base, key)

    def wrap_owner(self, owner):
        from .input_scope import ScopeTargetReader, is_target

        if is_target(self.target):
            return ScopeTargetReader(owner, self)
        if not self.target and getattr(self.base, "confirmed_review_reader", None) is not None:
            from packages.evidence_details.review_continuation import ContinuationOwner

            return ContinuationOwner(owner, self.base.confirmed_review_reader)
        return PartialTargetReader(owner) if self.target else owner

    def current_projection(self):
        if getattr(self.base, "root_direction_review", None):
            from packages.dataset_adapters.current_read_scope import _projection_visible_reads
            from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

            # One read-only projection can visit several accepted ancestors.
            # End input reuse before fresh fact and durable return checks;
            # never extend this scope over dispatch or persistence.
            with historical_fact_read_scope(), _projection_visible_reads():
                return self._current_projection()
        return self._current_projection()

    def review_reuse_preview(self, *, run_id, config, preview_run_id):
        """Read-only compatibility estimate, never a dispatch or saved phase.

        The new input-scope survey has not run during plan preparation. Its
        metadata is omitted only here; normal projection and verify still
        require the current Scope owner. The original accepted reader checks
        the complete previously visible facts, sources, policy and receipts.
        Actual Review must requalify the returned subset after the survey.
        """
        if self.target is not None or preview_run_id == run_id:
            raise ValueError("review_reuse_preview_identity")
        reader = getattr(self.base, "confirmed_review_reader", None)
        if reader is None:
            raise ValueError("review_reuse_preview_original_owner_required")
        self.context, self.config = self.base.contexts[run_id], config
        from packages.dataset_adapters.current_read_scope import _projection_visible_reads
        from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

        with historical_fact_read_scope(), _projection_visible_reads():
            state = self._current_projection(reassessment_preview=True)
            raw = state.model_dump(mode="json")
            raw["material"]["scope"]["run_id"] = preview_run_id
            raw["material"]["scope"].pop("material_checksum")
            raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
            preview = validate(DomainState, raw)
            origin = reader.current(preview)
            return origin, dict(
                schema_version="review-compatibility-preview/1.0.0",
                state_checksum=digest(preview),
                policy_checksum=digest(preview.policy),
                source_checksum=digest(preview.material.sources),
                assessment_ids=list(origin.assessment_ids),
                execution_authorized=False,
                current_scope_required_before_dispatch=True,
            )

    def _current_projection(self, *, reassessment_preview=False):
        from .input_scope import is_target, projection

        if is_target(self.target):
            return projection(self)
        observations = dict(self.base.observations)
        if self.target:
            image, _, view = self.base.view(self.target)
            body = dict(
                status="unknown",
                method="live_image_observation",
                unit="qualitative",
                value=canonical(
                    {
                        k: view[k]
                        for k in (
                            "modality",
                            "indices",
                            "available_members",
                            "inspected_fraction",
                            "transform",
                            "limits",
                        )
                    }
                ).decode(),
                image=image,
            )
            observations = {
                self.target: ObservationInput(
                    reference=ref(
                        "observation", "OBS:MM:" + digest(self.target)[7:19], digest(body)
                    ),
                    **body,
                )
            }
            ledger = []
            for r in self.base.ledger:
                if r["id"] == self.target:
                    definition = r["definition"]
                    raw = definition.model_dump(mode="json")
                    raw["applicability_conditions"] += (
                        " 展示采样/变换见输入观察的元数据；它不是医学检查结果。"
                    )
                    raw["reference"]["checksum"] = digest(
                        {k: v for k, v in raw.items() if k != "reference"}
                    )
                    definition = validate(type(definition), raw)
                    r = dict(r, definition=definition)
                ledger.append(r)
        else:
            observations.update(self.observation_reader())
            current_observations = dict(observations)
            observations = review_observations(self.base, observations)
            ledger = self.base.ledger
            if self.context.request.task != self.base.task:
                from .single_proposition import bind_requirements

                ledger = bind_requirements(ledger, self.context.request.task)
        medical = {**self.sources.initial, **getattr(self.base, "definition_materials", {})}
        if getattr(self.base, "definition_source_scope", None) == "requirements-and-items/1.0.0":
            medical = material.requirement_sources(
                medical, [r for r in ledger if r["id"] == self.target] if self.target else ledger
            )
        supplementary = ()
        if self.target is None and getattr(self.base, "execution_profile", None):
            from .review_context import collect_campaign
            from .review_context import supplementary as context_observations

            supplementary = context_observations(collect_campaign(self.base, observations))
        if self.target is None and getattr(self.base, "review_display_context", None):
            from .review_context import collect
            from .review_context import supplementary as context_observations

            current_context = collect(
                self.base,
                observations,
                sorted(self.base.review_display_context)
                if getattr(self.base, "review_metadata_profile", None)
                else self.base.review_target_ids,
                metadata_profile=getattr(self.base, "review_metadata_profile", None),
            )
            if current_context != self.base.review_display_context:
                raise ValueError("review_display_context_changed")
            method_context = context_observations(current_context)
            policy = getattr(self.base, "aggregation_method_context_policy", None)
            if policy is None:
                supplementary = method_context
            else:
                if (
                    policy != "full-owner-and-selected-method-context/1.0.0"
                    or not getattr(self.base, "review_metadata_profile", None)
                    or getattr(self.base, "review_target_ids", None) is not None
                ):
                    raise ValueError("accepted_aggregation_method_context_scope")
                # Both inputs were rebuilt by their original current owners above.
                # Metadata cannot replace a medical observation or add a vote.
                combined = {o.reference.code: o for o in supplementary}
                for observation in method_context:
                    old = combined.get(observation.reference.code)
                    if old is not None and old != observation:
                        raise ValueError("accepted_aggregation_method_context_conflict")
                    combined[observation.reference.code] = observation
                supplementary = tuple(combined.values())
        if self.target is None:
            from .review_context import retain_observations

            supplementary = retain_observations(
                self.base, current_observations, observations, supplementary
            )
        from .scope_selection import context_observations as scope_context

        if reassessment_preview:
            if self.target is not None:
                raise ValueError("review_reuse_preview_acquisition_forbidden")
        else:
            supplementary = (*supplementary, *scope_context(self.base, self.target))
        projected = material.state(
            context=self.context,
            visible=self.visible_value,
            root=self.root,
            config=self.config,
            medical=medical,
            ledger=ledger,
            observations=observations,
            resources=acquisition_scope.resources(self.base, self.target),
            acquisition=self.target,
            supplementary_observations=supplementary,
        )
        if self.target is None:
            from packages.evaluation_preparation.generic_aliases import alias_review

            alias_policy = getattr(self.base, "review_alias_policy", None)
            if alias_policy not in {None, "retain-current-owner-source-aliases/1.0.0"}:
                raise ValueError("review_source_alias_policy")
            previous = ()
            if alias_policy is not None:
                reader = getattr(self.base, "confirmed_review_reader", None)
                if reader is None:
                    raise ValueError("review_source_alias_original_owner_required")
                previous = reader.state_reader().material.sources
            projected, self.source_alias_manifest = alias_review(projected, previous=previous)
            clarification = getattr(self.base, "review_clarification", None)
            if clarification:
                from .single_proposition import clarify_review_state

                projected = clarify_review_state(
                    projected, self.context.request.task, clarification
                )
            root_question = getattr(self.base, "root_direction_review", None)
            if root_question:
                from .single_proposition import root_direction_state

                projected = root_direction_state(
                    projected, self.context.request.task, root_question
                )
        return projected

    def project(self, *, run_id, config):
        self.context, self.config = self.base.contexts[run_id], config
        self.initial = self.current_projection()
        return self.initial, dict(input=self.manifest.checksum, views=self.view_checksum)

    def verify(self, state):
        if getattr(self.base, "root_direction_review", None):
            from packages.dataset_adapters.current_read_scope import _projection_visible_reads
            from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

            with historical_fact_read_scope(), _projection_visible_reads():
                return self._verify(state)
        return self._verify(state)

    def _verify(self, state):
        acquisition_scope.check(self.base, self.target)
        check_current_inputs(self.context, self.inputs)
        expected = self.current_projection()
        if (
            expected != self.initial
            or state.policy != self.initial.policy
            or state.material.sources != self.initial.material.sources
        ):
            raise ValueError("multimodal_phase_dependencies_changed")
        for key in ("case_code", "run_id", "input_checksum", "visible_checksum"):
            if getattr(state.material.scope, key) != getattr(self.initial.material.scope, key):
                raise ValueError("multimodal_phase_scope")


class MultimodalCampaign(Campaign):
    version = "multimodal-live/1.0.0"
    phase_class = Phase

    def __init__(self, base, target, *, campaign_id, synthetic=False):
        self.base, self.target = base, target
        self.output, self.sample = base.output, base.manifest.sample
        self.config = configuration(
            review_replacement=getattr(base, "review_replacement", False),
            resilient_vision=getattr(base, "resilient_vision", False),
            directional=hasattr(base, "definition_expansion_policy"),
            exact_primary_status=getattr(base, "exact_primary_status", False),
            qualified_observations=(
                getattr(base.manifest, "adapter_version", None)
                in {"multimodal-adapters/1.1.0", "multimodal-adapters/1.2.0"}
            ),
            execution_profile=getattr(base, "execution_profile", None),
            review_batch_size=getattr(base, "review_batch_size", 1),
            network_route=getattr(base, "network_route", None),
            model_selection=getattr(base, "model_selection", None),
            vision_feedback=getattr(base, "vision_feedback", None),
            review_contract=getattr(base, "review_contract", None),
            http_client_selection=getattr(base, "http_client_selection", None),
        )
        self.contexts, self.sessions = ResearchRunRepository(target), PlannerRepository(target)
        self.runs = TaskRuns(runtime=self.contexts, inputs=base.inputs)
        self.campaign_id, self.synthetic = campaign_id, synthetic
        self.targets = self.available_targets(base)
        self.initial_targets = self.targets
        from .input_scope import install

        install(self)

    @staticmethod
    def available_targets(base):
        """The same prepared-input method selection for initial and successor work."""
        # Legacy maps remain numeric-only. Qualified new inputs explicitly allow
        # a separately defined map observation, sharing its numeric dependency.
        visual = {
            c.id
            for c in base.manifest.components
            if c.kind in {"image", "volume", "image_series"}
            or (
                c.kind == "map"
                and getattr(base.manifest, "adapter_version", None) == "multimodal-adapters/1.1.0"
            )
        }
        available = {v["component"] for v in base.views["views"] if v["status"] == "obtained"}
        return MultimodalCampaign.select_targets(base, visual & available)

    def scoped_acquisition_inventory(self):
        """Retain phase identities for history, budget and Review reads."""
        return self.available_targets(self.base)

    def scoped_acquisition_targets(self):
        """Current method candidates minus original-owner confirmed observations.

        Scope recovery must not redispatch a fact merely because it is stored
        by the phase owner rather than in the input/tool observation projection.
        Owner errors propagate; an unreadable history is not an empty history.
        """
        confirmed = self.accepted_observations()
        return tuple(t for t in self.scoped_acquisition_inventory() if t not in confirmed)

    @staticmethod
    def select_targets(base, available):
        if getattr(base, "known_only", False):
            return ()  # Never replay a prior unknown visual request in this mode.
        targets = tuple(
            r["id"]
            for r in base.ledger
            if r["component"] in available
            and r.get("acquisition_allowed", True)
            and (not getattr(base, "active_owner", None) or r["id"] not in base.observations)
            and r.get("acquisition_method", "vision") in {"vision", "disc_cup_measurement"}
        )
        if getattr(base, "active_schedule_v19", False):
            from .active_continuation import ordered_targets

            return ordered_targets(base.ledger, targets, base.observations)
        return targets

    def read_model(self):
        return None

    def paths(self):
        return [self.phase_path(t) for t in getattr(self, "scope_targets", ()) + self.targets] + [
            self.phase_path(None)
        ]

    def usage(self):
        used = super().usage()
        expansion = getattr(self, "definition_expansion", None)
        if expansion:
            # Provider reservations count once; local lookups do not pretend to be HTTP.
            used += expansion.budget.used
            used += sum(o["kind"] in {"wire", "search", "fetch"} for o in expansion.operations())
        return used

    def expand_definitions(
        self,
        *,
        sources=None,
        provider_factory=None,
        restore_only=False,
        report_only=False,
        until=None,
    ):
        """Real campaign installation seam; injected transports are used by offline tests."""
        from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
        from packages.llm_provider.provider import RequestBudget, StructuredProvider
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        from .definition_expansion import (
            DefinitionExpansion,
            definition_config,
            read_expansion_policy,
        )

        if report_only and not restore_only:
            raise ValueError("definition_report_requires_restore_only")

        if hasattr(self, "definition_expansion"):
            if restore_only:
                expansion = self.definition_expansion
                reader = expansion.read_revision if report_only else expansion.current
                result = expansion.install(reader())
                self.base.definition_current = reader
                return result
            return self.definition_expansion.advance(until=until)
        phase = self.phase_class(
            self.base, target=None, observation_reader=self.accepted_observations
        )
        ctx, state = self.register_phase(phase, self.request_key("definition-expansion"))
        self.definition_run_id = ctx.run_id
        docs = self.documents(ctx.run_id)
        policy = self.base.definition_expansion_policy
        if sources is None:
            original = self.base.sources.reader.original
            sources = KnowledgeDefinitionSources(
                repository=original.knowledge,
                store=original.knowledge_objects,
                index_keys=policy["index_keys"],
                failure_policy=policy.get("search_failure_policy"),
                material_policy=policy.get("material_policy"),
                passage_plans=policy.get("passage_plans"),
            )
        config = definition_config(
            self.config,
            prompt_version=getattr(
                read_expansion_policy(policy), "prompt_version", "definition-draft-2.0.0"
            ),
            source_output_tokens=getattr(
                read_expansion_policy(policy), "source_output_tokens", None
            ),
        )
        campaign = self

        class SharedDefinitionBudget(RequestBudget):
            def check_control(self):
                campaign.definition_expansion.control()

            def before_reserve(self, config_checksum):
                campaign.definition_expansion.guard()

            def before_wire(self, *args):
                campaign.guard_definition_materials(dispatch=True)

        budget = SharedDefinitionBudget(
            run_id=ctx.run_id,
            deadline=datetime.fromisoformat(self.protocol["deadline"]),
            max_requests=self.config.max_requests,
            journal=RuntimeAgentJournal(docs),
        )
        provider = None
        if not restore_only:
            provider = (
                provider_factory(config)
                if provider_factory
                else StructuredProvider(config, live_opt_in=True)
            )
        self.definition_expansion = DefinitionExpansion(
            self,
            documents=docs,
            sources=sources,
            provider=provider,
            budget=budget,
            scope=state.material.scope,
            policy=policy,
        )
        topic_map = policy.get("public_topic_sets") or policy.get("public_topics")
        if (
            not restore_only
            and topic_map
            and sources.search_public is None
            and self.protocol.get("new_public_search_allowed") is not False
            and policy.get("retrieval_scope") != "confirmed-index-only/1.0.0"
        ):
            from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector

            connector = LiteratureDefinitionConnector(
                sources.repository, self.definition_expansion, topic_map
            )
            sources.search_public, sources.fetch_hit = connector.search, connector.fetch
        if restore_only:
            if report_only:
                result = self.definition_expansion.install(
                    self.definition_expansion.read_revision()
                )
                self.base.definition_current = self.definition_expansion.read_revision
                return result
            return self.definition_expansion.install(self.definition_expansion.current())
        return self.definition_expansion.advance(until=until)

    def phase_path(self, target):
        from .input_scope import is_target

        if is_target(target):
            return self.output / ("input-scope-" + digest(target)[7:19])
        return self.output / ("vision-" + digest(target)[7:19] if target else "review")

    def request_key(self, target):
        return "mm-live:" + digest((self.campaign_id, target or "review"))[7:55]

    def register_phase(self, evidence, key):
        from .input_scope import is_target
        from .scope_selection import enabled as scoped_methods

        acquisition_scope.check(self.base, evidence.target)
        native = getattr(getattr(self.base, "active_owner", None), "native", None)
        independent_scope = is_target(evidence.target) and scoped_methods(self.base)
        if independent_scope:
            native = None  # The survey never consumes later tool products.
        catalog = dict(
            kind="catalog", code="CATALOG:NO_NATIVE_TOOL", version="1.0.0", checksum=digest([])
        )
        if native:
            from .native_fundus import records

            if native.get("schema_version") == "native-fundus-collection/1.0.0":
                catalog.update(
                    code="CATALOG:ACTIVE_FUNDUS_COMPONENTS",
                    checksum=digest(
                        [
                            dict(
                                target=r["plan"]["target"],
                                component=r["plan"]["component"],
                                manifest_checksum=r["plan"]["manifest_checksum"],
                            )
                            for r in records(native)
                        ]
                    ),
                )
            else:
                catalog.update(
                    code="CATALOG:ACTIVE_FUNDUS", checksum=native["plan"]["manifest_checksum"]
                )
        ctx = self.runs.create(
            case_code=self.base.binding["case_code"],
            key=key,
            task=self.base.task,
            policy_ref=dict(
                kind="policy",
                code="POLICY:MULTIMODAL_PHASE",
                version="1.0.0",
                checksum=digest(
                    (
                        self.base.input_scope_plan if independent_scope else self.base.ledger,
                        self.config,
                        evidence.target,
                    )
                ),
            ),
            catalog_ref=catalog,
            synthetic=self.synthetic,
        )
        self.base.contexts[ctx.run_id] = ctx
        state, _ = evidence.project(run_id=ctx.run_id, config=self.config)
        return ctx, state

    def freeze(self):
        from .implementation import checked_snapshot, freeze

        seconds = getattr(self.base, "active_seconds", 2700)
        if getattr(self.base, "resilient_vision", False) and hasattr(self, "propositions"):
            pending = len(self.propositions)
            if hasattr(self, "remaining_review"):
                pending -= len(self.remaining_review.labels)
            seconds += max(0, pending - 1) * 900
        expected = dict(
            schema_version="multimodal-live-campaign/1.2.0"
            if getattr(self.base, "resilient_vision", False)
            else "multimodal-live-campaign/1.1.0"
            if getattr(self.base, "review_replacement", False)
            else "multimodal-live-campaign/1.0.0",
            campaign_id=self.campaign_id,
            input=self.base.manifest.checksum,
            visible=self.base.visible_value.checksum,
            task=self.base.task.checksum,
            views=self.base.view_checksum,
            medical=digest(self.base.sources.initial),
            provider=digest(self.config),
            targets=self.initial_targets
            if hasattr(self.base, "definition_expansion_policy")
            else self.targets,
            max_new_requests=40,
            seconds=seconds,
            implementation=freeze(),
            hidden_labels_used=False,
            clinical_acceptance=False,
            previous_attempts=self.previous_attempts(),
            known_measurements_only=getattr(self.base, "known_only", False),
        )
        expected = json.loads(canonical(expected))
        if getattr(self.base, "entry_execution_lock", None) is not None:
            expected["entry_execution_lock"] = self.base.entry_execution_lock
        if (self.output / "assembly-repair.json").exists():
            from .assembly_recovery import read_amendment

            amendment = read_amendment(self.output)
            expected["implementation"] = amendment["implementation_before"]
            expected["assembly_repair_checksum"] = digest(amendment)
        if acquisition_scope.enabled(self.base):
            expected["acquisition_scope_policy"] = self.base.acquisition_scope_policy
        if hasattr(self.base, "definition_expansion_policy"):
            expected["schema_version"] = "multimodal-live-campaign/1.3.0"
            expected["definition_expansion"] = self.base.definition_expansion_policy
        if getattr(self.base, "observation_failure_policy", None):
            expected["schema_version"] = "multimodal-live-campaign/1.4.0"
            expected["observation_failure_policy"] = self.base.observation_failure_policy
        if hasattr(self, "propositions"):
            expected["independent_propositions"] = [
                t.model_dump(mode="json") for t in self.propositions.values()
            ]
        if getattr(self.base, "detailed", False):
            expected["close_reserve_seconds"] = 300
            expected["local_gap_policy"] = "completed-read-only-outcome/1.0.0"
        if hasattr(self, "procedure_reuse"):
            expected["procedure_reuse"] = self.procedure_reuse.plan
        if hasattr(self, "confirmed_observations"):
            expected["confirmed_observations"] = self.confirmed_observations.report()
        if getattr(self.base, "resilient_vision", False):
            expected["vision_resilience"] = "compact-current-target/1.0.0"
        if hasattr(self, "continuation"):
            expected["continuation"] = self.continuation.report()
        if hasattr(self, "discovery_continuation"):
            expected["discovery_continuation"] = self.discovery_continuation.reference
        if hasattr(self, "definition_continuation"):
            expected["definition_continuation"] = self.definition_continuation.reference
        if hasattr(self, "remaining_review"):
            expected["remaining_review"] = self.remaining_review.report()
        if getattr(self.base, "review_replacement", False):
            expected["review_replacement"] = "cancelled-read-only-once/1.0.0"
        if getattr(self.base, "active_owner", None):
            expected["active_acquisition"] = digest(
                getattr(self.base, "initial_active_acquisition", None)
                or self.base.active_owner.report()
            )
        if getattr(self.base, "component_observations", None):
            self.base.component_observations.check()
            expected["component_observations"] = self.base.component_observations.report()
        if getattr(self.base, "component_consumer_selection", None):
            expected["component_consumer_selection"] = self.base.component_consumer_selection
        if getattr(self.base, "active_schedule_v19", False):
            expected["active_schedule"] = "modality_balanced_with_review_reserve/1.0.0"
        if getattr(self.base, "execution_profile", None):
            expected["schema_version"] = (
                "multimodal-live-campaign/1.12.0"
                if getattr(self.base, "component_consumer_selection", None)
                else "multimodal-live-campaign/1.11.0"
                if self.base.execution_profile == "relay-snapshot-context-v1"
                else "multimodal-live-campaign/1.10.0"
            )
            expected["execution_profile"] = self.base.execution_profile
        if getattr(self.base, "model_selection", None) is not None:
            expected["model_selection"] = self.base.model_selection
        if getattr(self.base, "vision_feedback", None) is not None:
            expected["vision_feedback"] = self.base.vision_feedback
        if getattr(self.base, "review_contract", None) is not None:
            expected["review_contract"] = self.base.review_contract
        if getattr(self.base, "http_client_selection", None) is not None:
            expected["http_client_selection"] = self.base.http_client_selection
        if getattr(self.base, "execution_profile", None) == "relay-root-question-v1":
            from .single_proposition import root_direction_spec

            question = root_direction_spec(self.base.task)
            if getattr(self.base, "root_direction_review", None) != question:
                raise ValueError("root_question_campaign_binding")
            expected["schema_version"] = "multimodal-live-campaign/1.15.0"
            expected["root_direction_review"] = question
            batch_size = getattr(self.base, "review_batch_size", 1)
            if batch_size != 1:
                if batch_size not in (2, 3) or self.config.review_batching.batch_size != batch_size:
                    raise ValueError("root_review_batch_size")
                expected["schema_version"] = "multimodal-live-campaign/1.16.0"
                expected["review_batch_size"] = batch_size
        if hasattr(self, "definition_continuation"):
            expected["schema_version"] = (
                "multimodal-live-campaign/1.19.0"
                if getattr(self.definition_continuation, "routed_backlog", False)
                else "multimodal-live-campaign/1.18.0"
                if getattr(self.definition_continuation, "root_confirmed", False)
                else "multimodal-live-campaign/1.17.0"
                if self.definition_continuation.root_backlog
                else "multimodal-live-campaign/1.14.0"
                if self.definition_continuation.defer
                else "multimodal-live-campaign/1.13.0"
            )
            if getattr(self.base, "source_fallback_window", None) is not None:
                expected["source_fallback_window"] = self.base.source_fallback_window
        if hasattr(self, "trial"):
            expected.update(
                schema_version=(
                    "multimodal-live-campaign/1.9.0"
                    if self.trial["schema_version"]
                    in {
                        "multimodal-definition-trial/1.11.0",
                        "multimodal-definition-trial/1.12.0",
                        "multimodal-definition-trial/1.13.0",
                        "multimodal-definition-trial/1.14.0",
                        "multimodal-definition-trial/1.15.0",
                        "multimodal-definition-trial/1.16.0",
                        "multimodal-definition-trial/1.17.0",
                        "multimodal-definition-trial/1.18.0",
                    }
                    else "multimodal-live-campaign/1.8.0"
                    if self.trial["schema_version"] == "multimodal-definition-trial/1.10.0"
                    else "multimodal-live-campaign/1.7.0"
                    if self.trial["schema_version"] == "multimodal-definition-trial/1.9.0"
                    else "multimodal-live-campaign/1.6.0"
                    if self.trial["schema_version"]
                    in {"multimodal-definition-trial/1.7.0", "multimodal-definition-trial/1.8.0"}
                    else "multimodal-live-campaign/1.5.0"
                ),
                definition_trial=self.trial,
                definition_increment_only=True,
                max_new_requests=self.trial["max_new_requests"],
            )
        if getattr(self.base, "input_scope_plan", None) is not None:
            from .input_scope import configuration as scope_configuration

            expected.update(
                schema_version="multimodal-live-campaign/1.20.0",
                input_scope=dict(
                    plan_checksum=digest(self.base.input_scope_plan),
                    provider_checksum=digest(
                        scope_configuration(self.config, self.base.input_scope_plan)
                    ),
                    targets=self.scope_targets,
                    budget_owner="original_campaign_and_RuntimeAgentJournal",
                    disease_votes=0,
                ),
            )
            expected = json.loads(canonical(expected))
        if getattr(self.base, "initial_active_acquisition", None) is not None:
            expected.update(
                schema_version="multimodal-live-campaign/1.21.0",
                scope_selection_policy=self.base.active_owner.profile["scope_selection_policy"],
            )
        path = self.output / "protocol.json"
        if path.exists():
            value = json.loads(path.read_bytes())
            expected["implementation"] = checked_snapshot(self.output, value["implementation"])
            if {k: v for k, v in value.items() if k not in {"started_at", "deadline"}} != expected:
                raise ValueError("multimodal_campaign_frozen_input_or_code_changed")
        else:
            now = datetime.now(UTC)
            value = dict(
                **expected,
                started_at=now.isoformat(),
                deadline=(now + timedelta(seconds=seconds)).isoformat(),
            )
            if getattr(self.base, "active_start", None):
                value.update({k: self.base.active_start[k] for k in ("started_at", "deadline")})
            write_once(path, value)
        self.protocol = value
        return value

    def work_deadline(self):
        """Legacy acquisition reserve, separate from the absolute protocol deadline."""
        deadline = datetime.fromisoformat(self.protocol["deadline"])
        if getattr(self.base, "detailed", False):
            deadline -= timedelta(seconds=300)
        return deadline

    def guard(self, *, dispatch=False):
        self.guard_window()
        from packages.dataset_adapters.current_read_scope import _projection_visible_reads
        from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

        # Synchronous qualification only: first and return checks finish before
        # any caller may reserve or dispatch. No source grant crosses requests.
        with historical_fact_read_scope(), _projection_visible_reads():
            acquisition_scope.check(self.base, getattr(self, "_acquisition_target", None))
            self.guard_control(dispatch=dispatch)
        self.guard_window()

    def guard_definition_materials(self, *, dispatch=False):
        """Only for the public definition owner; cannot authorize patient work."""
        self.guard_window()
        from packages.dataset_adapters.current_read_scope import _projection_visible_reads
        from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

        with historical_fact_read_scope(), _projection_visible_reads():
            getattr(self.base, "check_definition_materials", self.base.check)()
            self.guard_control(dispatch=dispatch)
        self.guard_window()

    def guard_window(self):
        """Fast stop only; never replaces source, identity or request-budget checks."""
        from .execution_lease import check_current

        check_current(self.output)
        if getattr(self.base, "active_owner", None) and (self.output / "cancel").exists():
            raise ProviderError("cancelled")
        if datetime.now(UTC) >= self.work_deadline():
            raise ProviderError("deadline")

    def cumulative_request_limit(self):
        return min(40, self.protocol["max_new_requests"]) if hasattr(self, "trial") else 40

    def guard_control(self, *, dispatch=False):
        """Budget/cancel part only; never an evidence or dispatch authorization.

        Read-only component owners use this to avoid recursively calling
        base.check while that same base is rechecking their observations.
        """
        self.guard_window()
        if hasattr(self, "procedure_reuse"):
            self.procedure_reuse.check()
        previous = self.previous_attempts()
        if previous != self.protocol["previous_attempts"]:
            raise ProviderError("scope", fields=("previous_attempts",))
        if datetime.now(UTC) >= self.work_deadline():
            raise ProviderError("deadline")
        used = self.usage() + sum(p["requests"] for p in previous)
        cap = self.cumulative_request_limit()
        if used > cap or (not dispatch and used >= cap):
            raise ProviderError("budget")

    def local_acquisition_gap(self, phase):
        """One completed read-only failure can be a gap; never replay its request."""
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        from .local_outcomes import reason

        if (
            not getattr(self.base, "detailed", False)
            or phase.get("target") is None
            or phase.get("error") not in {"schema", "candidate_failed_or_context_changed"}
        ):
            return False
        current = self.sessions.read(phase["run_id"])
        if (
            current is None
            or current.phase != "stopped"
            or current.pending is not None
            or current.cancelled
        ):
            return False
        docs = self.documents(phase["run_id"])
        if reason(RuntimeAgentJournal(docs).read()) is None:
            return False
        if docs.read("dynamic-phase-result") != (phase,):
            return False
        failures = 0
        for target in self.targets:
            path = self.phase_path(target) / "result.json"
            if path.exists():
                value = json.loads(path.read_bytes())
                failures += bool(value.get("error") and not value.get("accepted"))
        limit = 2 if getattr(self.base, "resilient_vision", False) else 1
        if not 1 <= failures <= limit:
            return False
        initial = validate(DomainState, docs.read("dynamic-phase-input")[0])
        graph = initial.material.scope.graph
        if any(
            getattr(current.graph, k) != getattr(graph, k) for k in ("code", "version", "checksum")
        ):
            return False
        evidence = self.phase_class(
            self.base,
            target=phase["target"],
            observation_reader=self.accepted_observations,
            model_reader=self.read_model,
        )
        evidence.project(run_id=phase["run_id"], config=self.config)
        evidence.verify(initial)
        self.guard()  # Current source, input, scope and remaining budget must still hold.
        return True

    def acquisition_exhausted(self):
        """Stop further images after two local failures; do not discard good facts."""
        if not getattr(self.base, "resilient_vision", False):
            return False
        failures = 0
        for target in self.targets:
            path = self.phase_path(target) / "result.json"
            if path.exists():
                value = json.loads(path.read_bytes())
                failures += bool(value.get("error") and not value.get("accepted"))
        return failures >= 2

    def active_review_reserved(self):
        if not getattr(self.base, "active_schedule_v19", False):
            return False
        path = self.output / "acquisition-budget-stop.json"
        if path.exists():
            old = json.loads(path.read_bytes())
            if (
                old.get("input_checksum") != self.base.manifest.checksum
                or old.get("protocol_checksum") != digest(self.protocol)
                or old.get("stop_acquisition") is not True
            ):
                raise ValueError("active_acquisition_budget_stop_binding")
            return True  # Monotonic: never restart acquisitions after choosing closure.
        from .active_continuation import reserve_review

        observations = {**self.base.observations, **self.accepted_observations()}
        acquired = len(observations)
        incremental = self.protocol.get("mode") == "acquisition"
        if incremental:
            acquired = max(0, acquired - self.reused_review_count)
        left = (
            datetime.fromisoformat(self.protocol["deadline"]) - datetime.now(UTC)
        ).total_seconds() - 300
        reserve_policy = getattr(getattr(self.base, "active_owner", None), "profile", {}).get(
            "review_reserve_policy"
        )
        decision = reserve_review(
            seconds_left=left,
            acquired=acquired,
            batch_size=self.config.review_batching.batch_size,
            requests_used=self.usage()
            + (0 if incremental else sum(p["requests"] for p in self.previous_attempts())),
            request_limit=self.protocol["max_new_requests"]
            if incremental
            else self.trial["max_new_requests"]
            if hasattr(self, "trial")
            else 40,
            minimum_requests=(
                self.trial["expansion_policy"]["reserve_review_requests"]
                if hasattr(self, "trial")
                else 0
            ),
            local_request_headroom=getattr(self, "request_finalization_headroom", 1),
        )
        if reserve_policy is not None:
            from .active_continuation import reserve_current_review, reviewable_ids

            selected = review_observations(self.base, observations)
            pending = reviewable_ids(
                selected, excluded=self.protocol.get("reuse_assessment_ids", ())
            )
            decision = reserve_current_review(
                seconds_left=left + 300,  # New policy reserves closing time exactly once.
                assessment_ids=pending,
                batch_size=self.config.review_batching.batch_size,
                timeout_seconds=self.config.transport.timeout_seconds,
                max_retries=self.config.transport.max_retries,
                requests_used=self.usage()
                + (0 if incremental else sum(p["requests"] for p in self.previous_attempts())),
                request_limit=self.protocol["max_new_requests"]
                if incremental
                else (self.trial["max_new_requests"] if hasattr(self, "trial") else 40),
                policy=reserve_policy,
                local_request_headroom=getattr(self, "request_finalization_headroom", 1),
            )
        if decision["stop_acquisition"]:
            write_once(
                path,
                dict(
                    **decision,
                    input_checksum=self.base.manifest.checksum,
                    protocol_checksum=digest(self.protocol),
                ),
            )
            return True
        return False

    def previous_attempts(self):
        if hasattr(self, "definition_continuation"):
            return self.definition_continuation.check_attempts()
        if hasattr(self, "discovery_continuation"):
            return self.discovery_continuation.check_attempts()
        if hasattr(self, "remaining_review"):
            return self.remaining_review.check_attempts()
        if hasattr(self, "continuation"):
            return self.continuation.check_attempts()
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        from .replacement import review_replaceable

        path, rows, seen = getattr(self.base, "previous", None), [], set()
        replacement = getattr(self.base, "review_replacement", False)
        replacements = 0
        while path is not None:
            path = Path(path).absolute()
            if path in seen or len(seen) >= 8:
                raise ValueError("multimodal_previous_chain")
            seen.add(path)
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("multimodal_previous_path")
            spec = json.loads((path / "entry.json").read_bytes())
            result = json.loads((path / "result.json").read_bytes())
            if (
                bool(spec.get("review_replacement")) != bool(result.get("review_replacement"))
                or (replacement and spec.get("review_replacement"))
                or (replacement and not rows and not spec.get("known_only"))
            ):
                raise ValueError("multimodal_review_replacement_once")
            if (
                spec["input_checksum"] != self.base.manifest.checksum
                or spec["intent"] != self.base.task.model_dump(mode="json")
                or result["case_code"] != self.base.binding["case_code"]
                or result["complete"]
                or self.documents(result["phases"][-1]["run_id"]).read("multimodal-case-result")
                != (result,)
            ):
                raise ValueError("multimodal_previous_owner_binding")
            count = 0
            for phase in result["phases"] + [r["phase"] for r in result.get("input_scopes", [])]:
                docs = self.documents(phase["run_id"])
                control = self.sessions.read(phase["run_id"])
                events = RuntimeAgentJournal(docs).read()
                replace_this_review = (
                    replacement
                    and not rows
                    and not self.targets
                    and getattr(self.base, "known_only", False)
                    and review_replaceable(phase, control, events)
                )
                if replace_this_review:
                    replacements += 1
                if (
                    docs.read("dynamic-phase-result") != (phase,)
                    or control.phase != "stopped"
                    or control.pending is not None
                    or phase["requests"]["pending_unknown"]
                    or (
                        phase["transport"]["remote_response_unknown"]
                        and not replace_this_review
                        and not (
                            getattr(self.base, "known_only", False)
                            and phase["target"] is not None
                            and not self.targets
                        )
                    )
                ):
                    raise ValueError("multimodal_previous_unknown")
                count += sum(e.get("phase") == "reserved" for e in events)
            if count != result["new_requests"]:
                raise ValueError("multimodal_previous_accounting")
            rows.append(dict(directory=str(path), result_checksum=digest(result), requests=count))
            path = spec.get("previous")
        if replacement and replacements != 1:
            raise ValueError("multimodal_review_replacement_requires_one_cancelled_timeout")
        return rows

    def execute_phase(self, target=None):
        from .input_scope import configuration as scope_configuration
        from .input_scope import current_entry, is_target
        from .replacement import close_failed_review

        previous = getattr(self, "_acquisition_target", None)
        original_config = self.config
        self._acquisition_target = target
        try:
            if is_target(target):
                current_entry(self.base, target)
                scope = self.protocol.get("input_scope")
                self.config = scope_configuration(original_config, self.base.input_scope_plan)
                if (
                    not scope
                    or scope["provider_checksum"] != digest(self.config)
                    or scope["plan_checksum"] != digest(self.base.input_scope_plan)
                    or target not in scope["targets"]
                ):
                    raise ValueError("input_scope_campaign_not_frozen")
            else:
                from .scope_selection import ensure

                ensure(self, allow_native=not (self.phase_path(target) / "result.json").exists())
            result = super().execute_phase(target)
            closed = close_failed_review(self.sessions, result)
            if closed is not None:
                docs = self.documents(result["run_id"])
                with docs.session():
                    docs.write("multimodal-failed-review-cancellation", (closed,))
            return result  # Historical result and remote unknown are not rewritten.
        finally:
            self.config = original_config
            self._acquisition_target = previous

    def accepted_observations(self):
        observations = (
            dict(self.confirmed_observations.check())
            if hasattr(self, "confirmed_observations")
            else {}
        )
        if hasattr(self, "continuation"):
            observations.update(self.continuation.check())
        if hasattr(self, "remaining_review"):
            observations.update(self.remaining_review.check())
        for target in self.targets:
            path = self.phase_path(target) / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_bytes())
            docs = self.documents(result["run_id"])
            from packages.evidence_details.accepted_read_scope import (
                historical_observation_documents,
            )

            phase, initial, values = historical_observation_documents(docs, result)
            if phase != (result,):
                raise ValueError("multimodal_phase_owner_binding")
            if not result["accepted"]:
                continue
            parent = validate(DomainState, initial[0])
            records, receipt = read_records(values[0]), read_receipt(values[3])
            if (
                receipt.status != "accepted"
                or receipt.records_checksum != digest(records)
                or records.scope != parent.material.scope
                or len(records.observations) != 1
            ):
                raise ValueError("multimodal_observation_receipt")
            observation = records.observations[0]
            expected_image, _, _ = self.base.view(target)
            if (
                observation.image != expected_image
                or observation.reference.code != parent.material.observations[0].reference.code
            ):
                raise ValueError("multimodal_observation_identity")
            observations[target] = observation
        return observations

    @contextmanager
    def execution_session(self):
        import fcntl

        if getattr(self.base, "entry_execution_lock", None) is not None:
            from .execution_lease import check_current

            active = check_current(self.output)
            if (
                active is None
                or active.spec["entry_execution_lock"] != self.base.entry_execution_lock
            ):
                raise ValueError("multimodal_entry_original_execution_lease_required")
            yield
            return
        with (self.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def execute(self):
        from .report import export, restore_export

        with self.execution_session():
            self.freeze()
            restore_committed_definitions = False
            if (self.output / "review-context-repair.json").exists():
                from .context_recovery import execute as recover_context

                return recover_context(self)
            if (self.output / "stage-failure.json").exists():
                from .stage_failure import export_saved

                amendment_path = self.output / "prewire-repair.json"
                amendment = (
                    json.loads(amendment_path.read_bytes()) if amendment_path.exists() else {}
                )
                # freeze() above validates the exact persisted repair, source and
                # implementation. It never deletes or rewrites the original failure.
                continuation = (
                    self.protocol.get("schema_version") == "multimodal-campaign-renewal/1.15.0"
                    and amendment.get("reason") == "honor_narrowed_pending_review_window"
                    and amendment.get("original_failure_checksum")
                    == digest(json.loads((self.output / "stage-failure.json").read_bytes()))
                )
                if (self.output / "phase-budget-repair.json").exists():
                    from .context_recovery import read_budget_amendment

                    amendment = read_budget_amendment(self.output)
                    key = "multimodal:phase-budget-repair"
                    if amendment["schema_version"] == "pre-wire-phase-budget-repair/1.1.0":
                        key += ":r02"
                        restore_committed_definitions = True
                    if self.documents(amendment["run_id"]).read(key) != (amendment,):
                        raise ValueError("phase_budget_repair_owner_binding")
                    continuation = True  # Only the original never-dispatched phase was authorized.
                if (self.output / "unscored-progress-repair.json").exists():
                    from .unscored_acquisition import read_progress_repair

                    read_progress_repair(self)
                    # The confirmed observation is read back, not dispatched again.
                    # Keep the old failure and the same absolute window/budget.
                    continuation = True
                if getattr(self, "local_material_repair", None):
                    if (self.output / "local-repair-failure.json").exists():
                        from .stage_failure import export_local_repair_failure

                        return export_local_repair_failure(self)
                    continuation = True  # Exact owner/failure/freeze checked above.
                if not continuation:
                    return export_saved(self)
            if (self.output / "stage-closure-error.json").exists():
                from .pending_review import partition_projection_repair_current

                if not partition_projection_repair_current(self):
                    raise ValueError("stage_previous_failure_read_only")
            if (self.output / "result.json").exists():
                result = json.loads((self.output / "result.json").read_bytes())
                docs = self.documents(result["phases"][-1]["run_id"])
                if docs.read("multimodal-case-result") != (result,):
                    raise ValueError("multimodal_result_readback")
                if docs.read(getattr(self, "report_key", "dynamic-case-report")) is not None:
                    exported = restore_export(self, result)
                else:
                    # A crash can occur between saving the final result and saving
                    # its report. Restore the immutable revision, never re-search.
                    if hasattr(self.base, "definition_expansion_policy"):
                        self.expand_definitions(restore_only=True)
                    exported = export(self, result)
                from .memory import archive

                return dict(exported, memory=archive(self, result))
            self.guard()
            spec = importlib.util.spec_from_file_location(
                "private_launcher", ".local/vectorengine/run.py"
            )
            launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launcher)
            name = self.config.transport.api_key_env
            previous = os.environ.get(name)
            try:
                os.environ[name] = launcher.load_secret()
                try:
                    if getattr(self, "scope_targets", ()):
                        from .input_scope import execute as execute_input_scope

                        execute_input_scope(self)
                    from .scope_selection import ensure

                    ensure(self)
                except Exception as error:
                    from .stage_failure import close

                    return close(self, error, stage="acquisition")
                if hasattr(self.base, "definition_expansion_policy"):
                    try:
                        self.expand_definitions(
                            restore_only=restore_committed_definitions,
                            report_only=restore_committed_definitions,
                        )
                    except Exception as error:
                        from .stage_failure import close

                        return close(self, error)
                phases = []
                from .scope_selection import enabled as scope_selection_enabled

                dispatch_targets = (
                    self.scoped_acquisition_targets()
                    if scope_selection_enabled(self.base)
                    else self.targets
                )
                for target in self.targets + (None,):
                    if (
                        target is not None
                        and target not in dispatch_targets
                        and not (self.phase_path(target) / "result.json").exists()
                    ):
                        # Parent-confirmed facts need no phase in this window.
                        # A phase already saved here still follows the original
                        # exact-readback path and remains in this window's result.
                        continue
                    try:
                        if target is not None and (
                            hasattr(self, "observation_closure")
                            or self.acquisition_exhausted()
                            or self.active_review_reserved()
                        ):
                            continue
                        phase = self.execute_phase(target)
                    except Exception as error:
                        # Preparation/owner checks can fail before a phase Control
                        # exists. Preserve actual durable facts through the same
                        # closure as staged execution; never replay or fabricate a
                        # phase result to make the full CLI appear successful.
                        from .stage_failure import close

                        return close(self, error, stage="acquisition" if target else "review")
                    phases.append(phase)
                    diagnostic = {}
                    if phase["error"] and getattr(self.base, "observation_failure_policy", None):
                        from .observation_closure import failure_summary

                        diagnostic["failure_detail"] = failure_summary(self, phase)
                    print(
                        json.dumps(
                            dict(
                                dataset=self.base.manifest.dataset,
                                phase=target or "review",
                                accepted=phase["accepted"],
                                error=phase["error"],
                                **diagnostic,
                            )
                        ),
                        flush=True,
                    )
                    if phase["error"]:
                        from .observation_closure import close_observation

                        if close_observation(self, phase):
                            continue
                    # A failed observation remains a gap. A transport/budget failure stops dispatch.
                    if (
                        phase["error"]
                        and not phase["accepted"]
                        and not self.local_acquisition_gap(phase)
                    ):
                        break
            finally:
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
            try:
                return self.finish(phases)
            except Exception as error:
                from .stage_failure import close

                return close(self, error, stage="export")

    def finish(self, phases):
        """Original result/score/report ownership, shared by full and staged execution."""
        from .report import export, restore_export
        from .report_io import authorize_recovery, publish_json

        if not phases:
            raise ValueError("multimodal_finish_requires_actual_phase")
        docs = self.documents(phases[-1]["run_id"])
        result_key = getattr(self, "result_key", "multimodal-case-result")
        result_path = self.output / getattr(self, "result_filename", "result.json")
        saved = docs.read(result_key)
        if saved:
            if saved[0]["phases"] != phases:
                raise ValueError("multimodal_result_readback")
            result = saved[0]
            publish_json(result_path, result, recover=lambda: authorize_recovery(self, phases[-1]))
            if docs.read(getattr(self, "report_key", "dynamic-case-report")):
                return restore_export(self, result)
            return export(self, result)
        result = dict(
            schema_version="multimodal-case-result/1.2.0"
            if hasattr(self, "confirmed_observations")
            else "multimodal-case-result/1.1.0"
            if getattr(self.base, "review_replacement", False)
            else "multimodal-case-result/1.0.0",
            sample=self.sample,
            dataset=self.base.manifest.dataset,
            case_code=self.base.binding["case_code"],
            task=self.base.task.model_dump(mode="json"),
            phases=phases,
            complete=phases[-1]["target"] is None
            and (phases[-1]["accepted"] or phases[-1]["evidence_limited"]),
            new_requests=self.usage(),
            hidden_labels_used=False,
            clinical_acceptance=False,
            known_measurements_only=getattr(self.base, "known_only", False),
            previous_attempts=self.protocol["previous_attempts"],
        )
        if self.protocol.get("schema_version") in {
            "multimodal-campaign-renewal/1.35.0",
            "multimodal-campaign-renewal/1.40.0",
        }:
            result.update(
                schema_version="multimodal-method-reviewed-case-result/1.0.0",
                text_review_method=self.protocol["text_review_method"],
                case_related_requests=(
                    result["new_requests"]
                    + sum(p["requests"] for p in result["previous_attempts"])
                    + self.protocol["text_review_method"]["diagnostic_requests"]
                ),
            )
        if self.protocol.get("schema_version") == "multimodal-campaign-renewal/1.36.0":
            result.update(
                schema_version="multimodal-method-continued-case-result/1.0.0",
                method_budget=self.protocol["method_budget"],
                case_related_requests=(
                    result["new_requests"]
                    + sum(p["requests"] for p in result["previous_attempts"])
                    + self.protocol["method_budget"]["diagnostic_requests"]
                ),
            )
        if getattr(self.base, "review_replacement", False):
            result["review_replacement"] = "cancelled-read-only-once/1.0.0"
        if getattr(self.base, "input_scope_plan", None) is not None:
            from .input_scope import read_results

            result.update(
                schema_version="multimodal-scope-case-result/1.0.0",
                input_scopes=read_results(self, require_all=True),
                input_scope_plan_checksum=digest(self.base.input_scope_plan),
            )
            if getattr(self.base, "scope_selection_owner", None) is not None:
                result.update(
                    schema_version="multimodal-scoped-method-case-result/1.0.0",
                    scope_method_plan=self.base.scope_selection_owner.check(),
                )
        if hasattr(self, "confirmed_observations"):
            result["confirmed_observations"] = self.confirmed_observations.report()
        if hasattr(self, "local_review_closure"):
            result.update(
                schema_version="multimodal-stage-recovered-case-result/1.0.0",
                local_review_closure=self.local_review_closure,
            )
        docs = self.documents(phases[-1]["run_id"])
        with docs.session():
            docs.write(result_key, (result,))
        publish_json(result_path, result, recover=lambda: authorize_recovery(self, phases[-1]))
        exported = export(self, result)
        if hasattr(self, "local_review_closure"):
            return exported  # Local report repair never archives a new Memory.
        from .memory import archive

        return dict(exported, memory=archive(self, result))

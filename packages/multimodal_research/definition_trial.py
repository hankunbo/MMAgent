"""Explicit fresh definition experiment, composing original campaign/stage owners.

The frozen envelope narrows (never upgrades) a live entry. Baseline requirements
remain visible; native work is deferred, not silently executed during assembly.
"""

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field

from apps.vectorengine_single_case import write_once
from packages.grounding.models import Value, digest, validate
from packages.llm_provider.provider import ProviderError

from .campaign import MultimodalCampaign
from .campaign_renewal import compact_configuration, gate_selection
from .definition_expansion import read_expansion_policy
from .implementation import freeze


class Trial(Value):
    schema_version: Literal["multimodal-definition-trial/1.0.0"] = (
        "multimodal-definition-trial/1.0.0"
    )
    entry_checksum: str
    authorization: str = Field(min_length=12, max_length=400)
    expansion_policy: dict
    max_new_requests: int = Field(default=12, ge=6, le=12)
    seconds: int = Field(default=3600, ge=900, le=3600)
    initial_native_policy: Literal["deferred_not_executed"] = "deferred_not_executed"
    review_scope: Literal["definition_increment_only"] = "definition_increment_only"


class TargetTrial(Trial):
    schema_version: Literal["multimodal-definition-trial/1.1.0"] = (
        "multimodal-definition-trial/1.1.0"
    )


class JatsTrial(TargetTrial):
    schema_version: Literal["multimodal-definition-trial/1.2.0"] = (
        "multimodal-definition-trial/1.2.0"
    )


class DependencyTrial(TargetTrial):
    schema_version: Literal["multimodal-definition-trial/1.3.0"] = (
        "multimodal-definition-trial/1.3.0"
    )


class CapacityTrial(DependencyTrial):
    schema_version: Literal["multimodal-definition-trial/1.4.0"] = (
        "multimodal-definition-trial/1.4.0"
    )


class EvidenceTrial(CapacityTrial):
    schema_version: Literal["multimodal-definition-trial/1.5.0"] = (
        "multimodal-definition-trial/1.5.0"
    )


class CitationTrial(EvidenceTrial):
    schema_version: Literal["multimodal-definition-trial/1.6.0"] = (
        "multimodal-definition-trial/1.6.0"
    )


class ComponentTrial(CitationTrial):
    """A partial review: one confirmed original component plus the new increment."""

    schema_version: Literal["multimodal-definition-trial/1.7.0"] = (
        "multimodal-definition-trial/1.7.0"
    )
    review_scope: Literal["definition_increment_and_confirmed_component"] = (
        "definition_increment_and_confirmed_component"
    )


class IntentComponentTrial(ComponentTrial):
    schema_version: Literal["multimodal-definition-trial/1.8.0"] = (
        "multimodal-definition-trial/1.8.0"
    )


class RequirementScopeTrial(IntentComponentTrial):
    schema_version: Literal["multimodal-definition-trial/1.9.0"] = (
        "multimodal-definition-trial/1.9.0"
    )


class GapDiscoveryTrial(RequirementScopeTrial):
    schema_version: Literal["multimodal-definition-trial/1.10.0"] = (
        "multimodal-definition-trial/1.10.0"
    )


class UnseenFirstTrial(GapDiscoveryTrial):
    schema_version: Literal["multimodal-definition-trial/1.11.0"] = (
        "multimodal-definition-trial/1.11.0"
    )


class WindowedTrial(UnseenFirstTrial):
    schema_version: Literal["multimodal-definition-trial/1.12.0"] = (
        "multimodal-definition-trial/1.12.0"
    )


class CompletePoolTrial(WindowedTrial):
    schema_version: Literal["multimodal-definition-trial/1.13.0"] = (
        "multimodal-definition-trial/1.13.0"
    )


class EntryWindowTrial(CompletePoolTrial):
    """Explicit fresh window bounded by the unchanged original entry lifetime."""

    schema_version: Literal["multimodal-definition-trial/1.14.0"] = (
        "multimodal-definition-trial/1.14.0"
    )
    seconds: int = Field(default=5400, ge=900, le=5400)


class RequirementContextTrial(EntryWindowTrial):
    schema_version: Literal["multimodal-definition-trial/1.15.0"] = (
        "multimodal-definition-trial/1.15.0"
    )


class BalancedSourceTrial(RequirementContextTrial):
    schema_version: Literal["multimodal-definition-trial/1.16.0"] = (
        "multimodal-definition-trial/1.16.0"
    )


class ConfirmedInterpretationTrial(BalancedSourceTrial):
    schema_version: Literal["multimodal-definition-trial/1.17.0"] = (
        "multimodal-definition-trial/1.17.0"
    )


class OwnerDependencyTrial(ConfirmedInterpretationTrial):
    schema_version: Literal["multimodal-definition-trial/1.18.0"] = (
        "multimodal-definition-trial/1.18.0"
    )


def read(output):
    out = Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("definition_trial_symlink")
    spec = json.loads((out / "entry.json").read_bytes())
    return checked(json.loads((out / "definition-trial.json").read_bytes()), spec)


def checked(value, spec):
    cls = (
        OwnerDependencyTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.18.0"
        else ConfirmedInterpretationTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.17.0"
        else BalancedSourceTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.16.0"
        else RequirementContextTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.15.0"
        else EntryWindowTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.14.0"
        else CompletePoolTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.13.0"
        else WindowedTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.12.0"
        else UnseenFirstTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.11.0"
        else GapDiscoveryTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.10.0"
        else RequirementScopeTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.9.0"
        else IntentComponentTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.8.0"
        else ComponentTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.7.0"
        else CitationTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.6.0"
        else EvidenceTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.5.0"
        else CapacityTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.4.0"
        else DependencyTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.3.0"
        else JatsTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.2.0"
        else TargetTrial
        if value.get("schema_version") == "multimodal-definition-trial/1.1.0"
        else Trial
    )
    trial = validate(cls, value)
    if issubclass(cls, EntryWindowTrial) and (
        type(spec.get("seconds")) is not int or trial.seconds > spec["seconds"]
    ):
        raise ValueError("definition_trial_entry_time_window")
    if trial.entry_checksum != digest(spec) or spec["implementation"] != freeze():
        raise ValueError("definition_trial_entry_binding")
    required_entry = (
        "multimodal-live-entry/1.14.0"
        if issubclass(cls, ComponentTrial)
        else "multimodal-live-entry/1.13.0"
    )
    if issubclass(cls, ComponentTrial) != bool(spec.get("component_consumer")):
        raise ValueError("definition_trial_component_binding")
    if issubclass(cls, ComponentTrial):
        from packages.research_runtime.task_context import TaskIntent

        from .component_consumer import load

        reference = spec["component_consumer"]
        selected = load(reference["path"], expected=reference["checksum"])
        if (
            selected.input_checksum != spec["input_checksum"]
            or selected.task_checksum != validate(TaskIntent, spec["intent"]).checksum
        ):
            raise ValueError("definition_trial_component_binding")
    if spec["schema_version"] != required_entry or any(
        spec.get(k)
        for k in (
            "previous",
            "continuation_of",
            "review_from",
            "active_continuation_of",
            "discovery_continuation_of",
            "independent_propositions",
            "reuse_procedures",
        )
    ):
        raise ValueError("definition_trial_requires_fresh_single_entry")
    policy = read_expansion_policy(trial.expansion_policy)
    if (
        policy.schema_version
        != (
            "campaign-definition-expansion/1.22.0"
            if cls is OwnerDependencyTrial
            else "campaign-definition-expansion/1.21.0"
            if cls is ConfirmedInterpretationTrial
            else "campaign-definition-expansion/1.20.0"
            if cls is BalancedSourceTrial
            else "campaign-definition-expansion/1.19.0"
            if cls is RequirementContextTrial
            else "campaign-definition-expansion/1.18.0"
            if issubclass(cls, CompletePoolTrial)
            else "campaign-definition-expansion/1.17.0"
            if cls is WindowedTrial
            else "campaign-definition-expansion/1.16.0"
            if cls is UnseenFirstTrial
            else "campaign-definition-expansion/1.15.0"
            if cls is GapDiscoveryTrial
            else "campaign-definition-expansion/1.14.0"
            if cls is RequirementScopeTrial
            else "campaign-definition-expansion/1.13.0"
            if cls is IntentComponentTrial
            else "campaign-definition-expansion/1.12.0"
            if issubclass(cls, CitationTrial)
            else "campaign-definition-expansion/1.11.0"
            if cls is EvidenceTrial
            else "campaign-definition-expansion/1.10.0"
            if cls is CapacityTrial
            else "campaign-definition-expansion/1.9.0"
            if cls is DependencyTrial
            else "campaign-definition-expansion/1.8.0"
            if cls is JatsTrial
            else "campaign-definition-expansion/1.7.0"
            if cls is TargetTrial
            else "campaign-definition-expansion/1.6.0"
        )
        or policy.reserve_review_requests >= trial.max_new_requests
    ):
        raise ValueError("definition_trial_policy")
    if issubclass(cls, TargetTrial) and policy.public_target != spec.get("intent", {}).get(
        "target_code"
    ):
        raise ValueError("definition_trial_public_target_binding")
    return trial.model_dump(mode="json")


def prepare(
    output,
    *,
    expansion_policy,
    authorization,
    max_new_requests=12,
    seconds=3600,
    entry_time_window=False,
):
    """Attach an explicit scope before any owner registration, native or wire work."""
    out = Path(output).absolute()
    if any(
        (out / p).exists()
        for p in (
            "protocol.json",
            "active-start.json",
            "active-plan.json",
            "cancel",
            "result.json",
        )
    ):
        raise ValueError("definition_trial_already_started")
    policy = json.loads(Path(expansion_policy).read_bytes())
    spec = json.loads((out / "entry.json").read_bytes())
    if type(entry_time_window) is not bool or (
        entry_time_window
        and (
            not spec.get("component_consumer")
            or policy.get("schema_version")
            not in {
                "campaign-definition-expansion/1.18.0",
                "campaign-definition-expansion/1.19.0",
                "campaign-definition-expansion/1.20.0",
                "campaign-definition-expansion/1.21.0",
                "campaign-definition-expansion/1.22.0",
            }
        )
    ):
        raise ValueError("definition_trial_entry_time_window")
    cls = (
        OwnerDependencyTrial
        if entry_time_window
        and policy.get("schema_version") == "campaign-definition-expansion/1.22.0"
        else ConfirmedInterpretationTrial
        if entry_time_window
        and policy.get("schema_version") == "campaign-definition-expansion/1.21.0"
        else BalancedSourceTrial
        if entry_time_window
        and policy.get("schema_version") == "campaign-definition-expansion/1.20.0"
        else RequirementContextTrial
        if entry_time_window
        and policy.get("schema_version") == "campaign-definition-expansion/1.19.0"
        else EntryWindowTrial
        if entry_time_window
        else CompletePoolTrial
        if spec.get("component_consumer")
        and policy.get("schema_version") == "campaign-definition-expansion/1.18.0"
        else WindowedTrial
        if spec.get("component_consumer")
        and policy.get("schema_version") == "campaign-definition-expansion/1.17.0"
        else UnseenFirstTrial
        if spec.get("component_consumer")
        and policy.get("schema_version") == "campaign-definition-expansion/1.16.0"
        else GapDiscoveryTrial
        if spec.get("component_consumer")
        and policy.get("schema_version") == "campaign-definition-expansion/1.15.0"
        else RequirementScopeTrial
        if spec.get("component_consumer")
        and policy.get("schema_version") == "campaign-definition-expansion/1.14.0"
        else IntentComponentTrial
        if spec.get("component_consumer")
        and policy.get("schema_version") == "campaign-definition-expansion/1.13.0"
        else ComponentTrial
        if spec.get("component_consumer")
        else CitationTrial
        if policy.get("schema_version") == "campaign-definition-expansion/1.12.0"
        else EvidenceTrial
        if policy.get("schema_version") == "campaign-definition-expansion/1.11.0"
        else CapacityTrial
        if policy.get("schema_version") == "campaign-definition-expansion/1.10.0"
        else DependencyTrial
        if policy.get("schema_version") == "campaign-definition-expansion/1.9.0"
        else JatsTrial
        if policy.get("schema_version") == "campaign-definition-expansion/1.8.0"
        else TargetTrial
        if policy.get("schema_version") == "campaign-definition-expansion/1.7.0"
        else Trial
    )
    value = cls(
        entry_checksum=digest(json.loads((out / "entry.json").read_bytes())),
        expansion_policy=policy,
        authorization=authorization,
        max_new_requests=max_new_requests,
        seconds=seconds,
    ).model_dump(mode="json")
    # Validate before publishing even this non-executable envelope.
    checked(value, json.loads((out / "entry.json").read_bytes()))
    write_once(out / "definition-trial.json", value)
    return read(out)


class DefinitionTrialCampaign(MultimodalCampaign):
    """Only scope/time overrides; execution, score, journal and reports stay original."""

    def __init__(self, *args, trial, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.config = compact_configuration()
        if trial["schema_version"] in {
            "multimodal-definition-trial/1.17.0",
            "multimodal-definition-trial/1.18.0",
            "multimodal-definition-trial/1.16.0",
            "multimodal-definition-trial/1.15.0",
            "multimodal-definition-trial/1.14.0",
            "multimodal-definition-trial/1.13.0",
            "multimodal-definition-trial/1.12.0",
            "multimodal-definition-trial/1.11.0",
            "multimodal-definition-trial/1.10.0",
            "multimodal-definition-trial/1.9.0",
            "multimodal-definition-trial/1.7.0",
            "multimodal-definition-trial/1.8.0",
        }:
            from packages.llm_provider.config import read_provider_config

            raw = self.config.model_dump(mode="json")
            for role in raw["roles"]:
                if role["role"] == "review":
                    role["max_output_tokens"] = 4096
            self.config = read_provider_config(raw)
        self.base.definition_expansion_policy = trial["expansion_policy"]
        self.base.review_target_ids = ()
        self.targets = self.initial_targets = ()

    def freeze(self):
        value = super().freeze()
        lease = self.base.active_start
        if (
            any(value[k] != lease[k] for k in ("started_at", "deadline"))
            or (
                datetime.fromisoformat(value["deadline"])
                - datetime.fromisoformat(value["started_at"])
            ).total_seconds()
            != self.trial["seconds"]
        ):
            raise ValueError("definition_trial_lifetime_binding")
        return value

    def verify_initialization(self):
        from db.runtime.repository.case_pipeline import identity

        run_id = identity("RUN", "task-research:" + self.request_key("definition-expansion"))
        if self.contexts.read_research_run(run_id, missing_ok=True) is None or self.documents(
            run_id
        ).read("definition-trial-initialization") != (self.protocol,):
            raise ValueError("definition_trial_initialization_binding")

    @contextmanager
    def phase_context(self, target):
        previous = getattr(self, "_final_review_active", False)
        self._final_review_active = target is None
        try:
            yield
        finally:
            self._final_review_active = previous

    def execute_phase(self, target=None):
        with self.phase_context(target):
            return super().execute_phase(target)

    def work_deadline(self):
        reserve = 60 if getattr(self, "_final_review_active", False) else 300
        return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(seconds=reserve)

    def guard(self, *, dispatch=False):
        self._trial_control(dispatch=dispatch)
        super().guard(dispatch=dispatch)

    def guard_definition_materials(self, *, dispatch=False):
        self._trial_control(dispatch=dispatch)
        super().guard_definition_materials(dispatch=dispatch)

    def _trial_control(self, *, dispatch=False):
        if (self.output / "cancel").exists():
            raise ProviderError("cancelled")
        if self.usage() > self.trial["max_new_requests"] or (
            not dispatch and self.usage() >= self.trial["max_new_requests"]
        ):
            raise ProviderError("budget")

    def expand_definitions(self, **kwargs):
        revision = super().expand_definitions(**kwargs)
        if revision.get("schema_version") != "definition-stage-checkpoint/1.0.0":
            selected = gate_selection(self, revision)
            if self.trial["schema_version"] in {
                "multimodal-definition-trial/1.17.0",
                "multimodal-definition-trial/1.18.0",
                "multimodal-definition-trial/1.16.0",
                "multimodal-definition-trial/1.15.0",
                "multimodal-definition-trial/1.14.0",
                "multimodal-definition-trial/1.13.0",
                "multimodal-definition-trial/1.12.0",
                "multimodal-definition-trial/1.11.0",
                "multimodal-definition-trial/1.10.0",
                "multimodal-definition-trial/1.9.0",
                "multimodal-definition-trial/1.7.0",
                "multimodal-definition-trial/1.8.0",
            }:
                selected = self.select_component_review(selected, revision)
            write_once(
                self.output / "increment-schedule.json",
                dict(
                    revision_checksum=digest(revision),
                    added=revision["added"],
                    scheduled=list(self.targets),
                    review_ids=list(selected),
                    whole_case_rescored=False,
                ),
            )
        return revision

    def select_component_review(self, selected, revision):
        adapter = self.base.component_observations
        adapter.check()
        rows = [r for r in self.base.ledger if r["id"] == adapter.target]
        if len(rows) != 1 or adapter.target not in self.base.observations:
            raise ValueError("definition_trial_component_missing")
        # Do not override a revised definition's interpretation refusal.
        if rows[0].get("review_allowed", True):
            selected = tuple(sorted(set(selected) | {adapter.target}))
        if len(selected) > 3:
            raise ValueError("definition_trial_component_review_bound")
        self.base.review_target_ids = selected
        self.targets = tuple(t for t in self.targets if t != adapter.target)
        return selected


def initialize(output):
    """Explicit original input registration; never Source/Vision/native dispatch."""
    from .entry import assemble_entry
    from .stages import prepare as stage_prepare

    out = Path(output).absolute()
    trial = read(out)
    if (out / "cancel").exists() or (out / "stage-failure.json").exists():
        raise ValueError("definition_trial_stopped")
    with (out / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exists = (out / "protocol.json").exists()
        campaign = assemble_entry(out, trial_initialize=not exists, trial_read_only=exists)
        campaign.freeze()
        campaign.guard()
        phase = campaign.phase_class(
            campaign.base, target=None, observation_reader=campaign.accepted_observations
        )
        ctx, _ = campaign.register_phase(phase, campaign.request_key("definition-expansion"))
        docs = campaign.documents(ctx.run_id)
        with docs.session():
            docs.write("definition-trial-initialization", (campaign.protocol,))
        campaign.verify_initialization()
        stage_prepare(out)
    return dict(
        protocol_checksum=digest(campaign.protocol),
        trial_checksum=digest(trial),
        max_new_requests=trial["max_new_requests"],
        deadline=campaign.protocol["deadline"],
        real_api_calls=0,
        native_tools_started=0,
    )


def cold(output):
    from .entry import assemble_entry

    out = Path(output).absolute()
    read(out)
    if not (out / "stage-plan.json").is_file():
        raise ValueError("definition_trial_initialize_required")
    campaign = assemble_entry(out, trial_read_only=True)
    campaign.freeze()
    campaign.verify_initialization()
    return campaign

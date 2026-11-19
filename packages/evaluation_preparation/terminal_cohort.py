"""Thin, append-only terminal supervision of existing case CLIs.

Command completion is not medical acceptance. This module never changes owner
budgets, replays an uncertain request, or invents a case's clinical result.
"""

import errno
import fcntl
import html
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from db.runtime.repository.case_pipeline import SessionLeaseLostError
from packages.grounding.models import Value, digest, validate
from packages.multimodal_research.report_io import publish_html, publish_json
from packages.object_store import checksum

ROOT = Path(__file__).resolve().parents[2]
TOKEN = r"^[a-z][a-z0-9-]{0,47}$"


class Case(Value):
    id: str = Field(pattern=TOKEN)
    dataset: Literal["fairvlmed", "harvard_gdp", "gamma", "grape"]
    root: str = Field(min_length=1)
    sample: str = Field(min_length=1, max_length=160)
    task: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    selection: Literal["training", "development_preparation"] = "training"
    active_acquisition: str | None = None
    registry: str = "configs/research/multimodal_propositions_v1.json"
    input_semantics: Literal["harvard-gdp-v1", "grape-v1"] | None = None
    cutoff_visit: int | None = Field(default=None, ge=0)
    review_batch_size: Literal[1, 2, 3] = 3

    @model_validator(mode="after")
    def binding(self):
        if self.dataset == "fairvlmed":
            if self.task != "glaucoma" or self.active_acquisition or self.input_semantics:
                raise ValueError("cohort_fairvlmed_original_task_only")
            if self.selection != "training" or self.cutoff_visit is not None:
                raise ValueError("cohort_fairvlmed_selection")
        elif self.active_acquisition is None:
            raise ValueError("cohort_active_profile_required")
        expected = {"harvard_gdp": "harvard-gdp-v1", "grape": "grape-v1"}.get(self.dataset)
        if self.input_semantics != expected:
            raise ValueError("cohort_input_semantics")
        if (self.cutoff_visit is not None) != (self.dataset == "grape"):
            raise ValueError("cohort_visit_scope")
        if not Path(self.root).is_absolute() or "\x00" in self.sample:
            raise ValueError("cohort_input_path")
        return self


class Manifest(Value):
    schema_version: Literal["terminal-case-cohort/1.0.0", "terminal-case-cohort/1.1.0"]
    id: str = Field(pattern=TOKEN)
    purpose: Literal["development_coverage_not_accuracy"]
    cases: tuple[Case, ...] = Field(min_length=1, max_length=20)
    command_timeout_seconds: int = Field(ge=60, le=14400)
    max_cases_per_invocation: int = Field(ge=1, le=4)

    @model_validator(mode="after")
    def unique(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("cohort_duplicate_case_id")
        identities = {(c.dataset, c.root, c.sample, c.task, c.cutoff_visit) for c in self.cases}
        if len(identities) != len(self.cases):
            raise ValueError("cohort_duplicate_case_input")
        return self


class RoutedManifest(Manifest):
    schema_version: Literal["terminal-case-cohort/1.2.0"]
    network_route: str


class SourceRecoveryPolicy(Value):
    mode: Literal["technical_backlog_no_source_replay"]
    window_seconds: int = Field(ge=600, le=5400)
    authorization_code: str = Field(pattern=TOKEN)


class RoutedRecoveryManifest(RoutedManifest):
    schema_version: Literal["terminal-case-cohort/1.3.0"]
    source_recovery: SourceRecoveryPolicy

    @model_validator(mode="after")
    def recovery_scope(self):
        if any(c.dataset != "fairvlmed" and c.review_batch_size not in (2, 3) for c in self.cases):
            raise ValueError("cohort_routed_recovery_original_root_profile_required")
        if self.command_timeout_seconds < self.source_recovery.window_seconds:
            raise ValueError("cohort_recovery_supervisor_window_too_short")
        return self


class ModelSelectedManifest(RoutedRecoveryManifest):
    schema_version: Literal["terminal-case-cohort/1.4.0"]
    model_selection: Literal["configs/agents/relay_existing_model_selection_v1.json"]


class VisionFeedbackManifest(RoutedRecoveryManifest):
    schema_version: Literal["terminal-case-cohort/1.5.0"]
    vision_feedback: Literal["state-consistency-once/1.0.0"]


class UniformModelManifest(VisionFeedbackManifest):
    schema_version: Literal["terminal-case-cohort/1.6.0"]
    expected_model: str = Field(min_length=1, max_length=120)
    fairvlmed_model_selection: Literal["configs/agents/relay_existing_model_selection_v1.json"]


class ReviewContractManifest(UniformModelManifest):
    schema_version: Literal["terminal-case-cohort/1.7.0"]
    review_contract: Literal["rating-presence/1.0.0", "rating-presence/1.1.0"]


class RuntimeLeaseManifest(UniformModelManifest):
    schema_version: Literal["terminal-case-cohort/1.8.0"]
    supervision_lease: Literal["original-runtime-case-pipeline-session/1.0.0"]


class ChildRuntimeLeaseManifest(RuntimeLeaseManifest):
    schema_version: Literal["terminal-case-cohort/1.9.0"]
    fairvlmed_execution_lock: Literal["runtime-case-host-resource/1.0.0"]


class AllChildRuntimeLeaseManifest(ChildRuntimeLeaseManifest):
    schema_version: Literal["terminal-case-cohort/1.10.0"]
    multimodal_execution_lock: Literal["multimodal-runtime-entry/1.0.0"]


class LocalProviderManifest(AllChildRuntimeLeaseManifest):
    schema_version: Literal["terminal-case-cohort/1.11.0"]
    network_route: Literal["configs/agents/server_local_https_route_v1.json"]
    fairvlmed_model_selection: None = None
    fairvlmed_provider_profile: Literal["relay-local-gpt6-shape-v1"]
    review_contract: Literal["rating-presence/1.1.0"]


class ClientSelectedManifest(LocalProviderManifest):
    schema_version: Literal["terminal-case-cohort/1.12.0"]
    multimodal_http_client: Literal["configs/agents/system_curl_client_v1.json"]


class UnstartedContinuation(Value):
    previous_cohort: str = Field(min_length=1)
    manifest_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    freeze_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    start_index: int = Field(ge=0, le=19)


class EnvironmentContinuationManifest(ClientSelectedManifest):
    """Only an untouched ordered slice; no rerouting of previously sent cases."""

    schema_version: Literal["terminal-case-cohort/1.13.0"]
    network_route: Literal["configs/agents/environment_https_route_v1.json"]
    continuation: UnstartedContinuation

    @model_validator(mode="after")
    def multimodal_only(self):
        if any(c.dataset == "fairvlmed" for c in self.cases):
            raise ValueError("cohort_environment_fairvlmed_not_installed")
        return self


class AllEnvironmentContinuationManifest(ClientSelectedManifest):
    """An untouched queue slice with the separately bound Fair environment profile."""

    schema_version: Literal["terminal-case-cohort/1.14.0"]
    network_route: Literal["configs/agents/environment_https_route_v1.json"]
    fairvlmed_provider_profile: Literal["relay-environment-gpt6-shape-v1"]
    continuation: UnstartedContinuation


ENVIRONMENT_CONTINUATIONS = (EnvironmentContinuationManifest, AllEnvironmentContinuationManifest)


def read_manifest(value):
    cls = (
        AllEnvironmentContinuationManifest
        if value.get("schema_version") == "terminal-case-cohort/1.14.0"
        else EnvironmentContinuationManifest
        if value.get("schema_version") == "terminal-case-cohort/1.13.0"
        else ClientSelectedManifest
        if value.get("schema_version") == "terminal-case-cohort/1.12.0"
        else LocalProviderManifest
        if value.get("schema_version") == "terminal-case-cohort/1.11.0"
        else AllChildRuntimeLeaseManifest
        if value.get("schema_version") == "terminal-case-cohort/1.10.0"
        else ChildRuntimeLeaseManifest
        if value.get("schema_version") == "terminal-case-cohort/1.9.0"
        else RuntimeLeaseManifest
        if value.get("schema_version") == "terminal-case-cohort/1.8.0"
        else ReviewContractManifest
        if value.get("schema_version") == "terminal-case-cohort/1.7.0"
        else UniformModelManifest
        if value.get("schema_version") == "terminal-case-cohort/1.6.0"
        else VisionFeedbackManifest
        if value.get("schema_version") == "terminal-case-cohort/1.5.0"
        else ModelSelectedManifest
        if value.get("schema_version") == "terminal-case-cohort/1.4.0"
        else RoutedRecoveryManifest
        if value.get("schema_version") == "terminal-case-cohort/1.3.0"
        else RoutedManifest
        if value.get("schema_version") == "terminal-case-cohort/1.2.0"
        else Manifest
    )
    return validate(cls, value)


def read(path):
    return json.loads(Path(path).read_bytes())


def public_config(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT / "configs") or value.suffix != ".json":
        raise ValueError("cohort_public_configuration_required")
    return value


def implementation():
    # Public implementation only, never .local or credentials. Include both
    # original case engines so their source collection changes are detected.
    files = set()
    for name in (
        "apps",
        "packages",
        "agents",
        "configs/prompts/agents",
        "configs/agents",
        "configs/research",
    ):
        files.update(
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".txt", ".js", ".html"}
        )
    return {str(p.relative_to(ROOT)): checksum(p.read_bytes()) for p in sorted(files)}


def runtime_owner_implementation():
    path = "db/runtime/repository/case_pipeline.py"
    return {path: checksum((ROOT / path).read_bytes())}


def provider_bindings(manifest):
    """Resolve the original owners' actual public configs, without credentials or I/O dispatch.

    Legacy FairVLMed retains its 4o profile and explicit model-only selector.
    Manifest1.11 directly selects its verified local 6-sol config. Multimodal
    retains its snapshot and owner-specific Prompt/contract, not a second model selection.
    """
    from packages.llm_provider.client_selection import load_selection as load_client
    from packages.llm_provider.config import read_provider_config
    from packages.llm_provider.model_selection import apply_selection, load_selection
    from packages.llm_provider.network_route import load_route
    from packages.multimodal_research.campaign import configuration

    from .generic_input import (
        ALTERNATE_PROFILE,
        ENVIRONMENT_PROFILE,
        ENVIRONMENT_PROVIDER,
        LOCAL_PROFILE,
        LOCAL_PROVIDER,
        ROUTED_PROVIDER,
        provider_config,
    )

    if not isinstance(manifest, UniformModelManifest):
        raise ValueError("cohort_uniform_model_version_required")
    route = load_route(manifest.network_route)
    bindings = {}
    for case in manifest.cases:
        if case.dataset == "fairvlmed":
            if isinstance(manifest, LocalProviderManifest):
                environment = isinstance(manifest, AllEnvironmentContinuationManifest)
                profile = ENVIRONMENT_PROFILE if environment else LOCAL_PROFILE
                provider = ENVIRONMENT_PROVIDER if environment else LOCAL_PROVIDER
                selected = read_provider_config(read(public_config(provider)))
                spec = dict(
                    schema_version="fairvlmed-run-input/1.11.0"
                    if environment
                    else "fairvlmed-run-input/1.10.0",
                    provider_profile=profile,
                    provider_configuration=provider,
                    provider_configuration_checksum=digest(selected),
                )
            else:
                selection = load_selection(manifest.fairvlmed_model_selection)
                original = read_provider_config(read(public_config(ROUTED_PROVIDER)))
                selected = apply_selection(original, selection)
                spec = dict(
                    schema_version="fairvlmed-run-input/1.8.0",
                    provider_profile=ALTERNATE_PROFILE,
                    provider_configuration=ROUTED_PROVIDER,
                    provider_configuration_checksum=digest(selected),
                    model_selection=selection,
                )
            config = provider_config(spec)
        else:
            config = configuration(
                resilient_vision=True,
                directional=True,
                exact_primary_status=True,
                qualified_observations=bool(case.input_semantics),
                execution_profile="relay-root-question-v1",
                review_batch_size=case.review_batch_size,
                network_route=route,
                vision_feedback=manifest.vision_feedback,
                review_contract=getattr(manifest, "review_contract", None),
                http_client_selection=(
                    load_client(manifest.multimodal_http_client)
                    if isinstance(manifest, ClientSelectedManifest)
                    else None
                ),
            )
        if config.transport.model != manifest.expected_model or any(
            getattr(role, "model", None) not in (None, manifest.expected_model)
            for role in config.roles
        ):
            raise ValueError("cohort_actual_provider_model_mismatch")
        if config.network_route != route:
            raise ValueError("cohort_actual_provider_route_mismatch")
        bindings[case.id] = dict(
            model=config.transport.model,
            provider_checksum=digest(config),
            network_route_checksum=digest(route),
        )
    return bindings


def commands(manifest, case, output, action):
    """No user-supplied executable, shell fragment, or arbitrary CLI options."""
    prefix = [sys.executable, "-u", "-m"]
    if action.startswith("source-"):
        if isinstance(manifest, AllChildRuntimeLeaseManifest):
            raise ValueError("cohort_entry_lease_source_successor_not_supported")
        if (
            manifest.schema_version
            not in {
                "terminal-case-cohort/1.1.0",
                "terminal-case-cohort/1.3.0",
                "terminal-case-cohort/1.4.0",
                "terminal-case-cohort/1.5.0",
                "terminal-case-cohort/1.6.0",
                "terminal-case-cohort/1.7.0",
                "terminal-case-cohort/1.8.0",
                "terminal-case-cohort/1.9.0",
            }
            or case.dataset == "fairvlmed"
        ):
            raise ValueError("cohort_source_recovery_not_supported")
        original = commands(manifest, case, output, "prepare")[0]
        position = original.index("--output") + 1
        parent = original[position]
        child = str(
            Path(output)
            / "inputs"
            / (
                "recovery-"
                + digest((manifest.id, case.id, str(Path(output).resolve()), "source"))[7:55]
            )
        )
        if action == "source-prepare":
            argv = list(original)
            argv[position] = child
            argv += ["--definition-failure-of", parent, "--defer-failed-definition-expansion"]
            if isinstance(manifest, RoutedRecoveryManifest):
                argv += [
                    "--source-fallback-seconds",
                    str(manifest.source_recovery.window_seconds),
                    "--source-fallback-authorization",
                    manifest.source_recovery.authorization_code + ":" + manifest.id + ":" + case.id,
                ]
            return [argv]
        verb = {
            "source-preflight": "preflight",
            "source-run": "run",
            "source-export": "restore-report",
        }.get(action)
        if verb is None:
            raise ValueError("cohort_unknown_source_action")
        return [prefix + ["apps.multimodal_case", verb, "--input", child]]
    key = (
        "cohort-" + digest((manifest.id, case.id, str(Path(output).resolve())))[7:55]
        if manifest.schema_version
        in {
            "terminal-case-cohort/1.1.0",
            "terminal-case-cohort/1.2.0",
            "terminal-case-cohort/1.3.0",
            "terminal-case-cohort/1.4.0",
            "terminal-case-cohort/1.5.0",
            "terminal-case-cohort/1.6.0",
            "terminal-case-cohort/1.7.0",
            "terminal-case-cohort/1.8.0",
            "terminal-case-cohort/1.9.0",
            "terminal-case-cohort/1.10.0",
            "terminal-case-cohort/1.11.0",
            "terminal-case-cohort/1.12.0",
            "terminal-case-cohort/1.13.0",
            "terminal-case-cohort/1.14.0",
        }
        else manifest.id + "-" + case.id
    )
    if len(key) > 64:
        raise ValueError("cohort_owner_key_length")
    if case.dataset == "fairvlmed":
        command = prefix + ["apps.fairvlmed_case"]
        identity = ["--root", case.root, "--sample", case.sample, "--split", case.selection]
        if action == "prepare":
            return [
                command
                + ["prepare"]
                + identity
                + [
                    "--key",
                    key,
                    "--evidence-profile",
                    "source-complete-view/1.0.0",
                    "--upstream-profile",
                    "evidence-reuse/1.0.0",
                    "--report-profile",
                    "case-report-production/1.0.0",
                    "--provider-profile",
                    manifest.fairvlmed_provider_profile
                    if isinstance(manifest, LocalProviderManifest)
                    else "relay-grounded-shape-proxy-alternate-v1"
                    if isinstance(manifest, (ModelSelectedManifest, UniformModelManifest))
                    else "relay-grounded-shape-proxy-v1"
                    if isinstance(manifest, RoutedManifest)
                    else "relay-grounded-shape-v1",
                ]
                + (
                    ["--execution-lock-profile", manifest.fairvlmed_execution_lock]
                    if isinstance(manifest, ChildRuntimeLeaseManifest)
                    else []
                )
            ]
        if action == "preflight":
            return [command + ["check"] + identity]
        return [command + [{"export": "export", "run": "run"}[action], "--key", key]]
    command = prefix + ["apps.multimodal_case"]
    entry = str(
        output
        / "inputs"
        / (
            key
            if manifest.schema_version
            in {
                "terminal-case-cohort/1.1.0",
                "terminal-case-cohort/1.2.0",
                "terminal-case-cohort/1.3.0",
                "terminal-case-cohort/1.4.0",
                "terminal-case-cohort/1.5.0",
                "terminal-case-cohort/1.6.0",
                "terminal-case-cohort/1.7.0",
                "terminal-case-cohort/1.8.0",
                "terminal-case-cohort/1.9.0",
                "terminal-case-cohort/1.10.0",
                "terminal-case-cohort/1.11.0",
                "terminal-case-cohort/1.12.0",
                "terminal-case-cohort/1.13.0",
                "terminal-case-cohort/1.14.0",
            }
            else case.id
        )
    )
    if action == "prepare":
        argv = command + [
            "live-prepare",
            "--dataset",
            case.dataset,
            "--root",
            case.root,
            "--sample",
            case.sample,
            "--task",
            case.task,
            "--selection",
            case.selection,
            "--registry",
            case.registry,
            "--active-acquisition",
            case.active_acquisition,
            "--resilient-vision",
            "--view-profile",
            "detail-v1",
            "--execution-profile",
            "relay-root-question-v1",
            "--review-batch-size",
            str(case.review_batch_size),
            "--output",
            entry,
        ]
        if case.input_semantics:
            argv += ["--input-semantics", case.input_semantics]
        if case.cutoff_visit is not None:
            argv += ["--cutoff-visit", str(case.cutoff_visit)]
        if isinstance(manifest, RoutedManifest):
            argv += ["--network-route", manifest.network_route]
        if isinstance(manifest, ModelSelectedManifest):
            argv += ["--model-selection", manifest.model_selection]
        if isinstance(manifest, VisionFeedbackManifest):
            argv += ["--vision-feedback", manifest.vision_feedback]
        if isinstance(manifest, (ReviewContractManifest, LocalProviderManifest)):
            argv += ["--review-contract", manifest.review_contract]
        if isinstance(manifest, AllChildRuntimeLeaseManifest):
            argv += ["--entry-execution-lock", manifest.multimodal_execution_lock]
        if isinstance(manifest, ClientSelectedManifest):
            argv += ["--http-client", manifest.multimodal_http_client]
        return [argv]
    return [
        command
        + [
            {"preflight": "preflight", "run": "run", "export": "restore-report"}[action],
            "--input",
            entry,
        ]
    ]


def check_unstarted_continuation(manifest, output):
    """Bind the prior frozen queue without rewriting or weakening its own freeze."""
    if not isinstance(manifest, ENVIRONMENT_CONTINUATIONS):
        return None
    proof = manifest.continuation
    old, original, freeze = load(proof.previous_cohort, check_current=False)
    if (
        old == Path(output).resolve()
        or original.schema_version != "terminal-case-cohort/1.12.0"
        or digest(original) != proof.manifest_checksum
        or digest(freeze) != proof.freeze_checksum
        or tuple(original.cases[proof.start_index : proof.start_index + len(manifest.cases)])
        != manifest.cases
    ):
        raise ValueError("cohort_continuation_original_slice_binding")
    current = manifest.model_dump(mode="json")
    if isinstance(manifest, AllEnvironmentContinuationManifest):
        # The new Fair profile is route-only. provider_bindings validates its
        # full config against the original profile, not an arbitrary override.
        from .generic_input import ENVIRONMENT_PROFILE, LOCAL_PROFILE

        if original.fairvlmed_provider_profile != LOCAL_PROFILE:
            raise ValueError("cohort_continuation_original_fair_profile")
        if current["fairvlmed_provider_profile"] != ENVIRONMENT_PROFILE:
            raise ValueError("cohort_continuation_new_fair_profile")
        current["fairvlmed_provider_profile"] = LOCAL_PROFILE
    for key, value in original.model_dump(mode="json").items():
        if (
            key not in {"schema_version", "id", "network_route", "cases"}
            and current.get(key) != value
        ):
            raise ValueError("cohort_continuation_unrelated_setting_changed")
    for case in manifest.cases:
        if (old / "commands" / case.id).exists():
            raise ValueError("cohort_continuation_case_already_started")
        argv = commands(original, case, old, "prepare")[0]
        if case.dataset == "fairvlmed":
            from .generic_input import directory

            old_input = directory(argv[argv.index("--key") + 1])
        else:
            old_input = Path(argv[argv.index("--output") + 1])
        if old_input.exists():
            raise ValueError("cohort_continuation_input_already_prepared")
    return dict(
        original_manifest_checksum=digest(original),
        original_freeze_checksum=digest(freeze),
        selected_case_ids=[c.id for c in manifest.cases],
        original_selected_requests=0,
        preceding_case_ids=[c.id for c in original.cases[: proof.start_index]],
        following_case_ids=[
            c.id for c in original.cases[proof.start_index + len(manifest.cases) :]
        ],
        historical_cases_rerun=False,
        source_freeze_unchanged=True,
    )


def initialize(manifest_path, output):
    manifest = read_manifest(read(manifest_path))
    if manifest.schema_version not in {
        "terminal-case-cohort/1.1.0",
        "terminal-case-cohort/1.2.0",
        "terminal-case-cohort/1.3.0",
        "terminal-case-cohort/1.4.0",
        "terminal-case-cohort/1.5.0",
        "terminal-case-cohort/1.6.0",
        "terminal-case-cohort/1.7.0",
        "terminal-case-cohort/1.8.0",
        "terminal-case-cohort/1.9.0",
        "terminal-case-cohort/1.10.0",
        "terminal-case-cohort/1.11.0",
        "terminal-case-cohort/1.12.0",
        "terminal-case-cohort/1.13.0",
        "terminal-case-cohort/1.14.0",
    }:
        raise ValueError("cohort_legacy_creation_disabled")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("cohort_output_must_be_new")
    continuation = check_unstarted_continuation(manifest, output)
    configs = {}
    if isinstance(manifest, ClientSelectedManifest):
        from packages.llm_provider.client_selection import load_selection as load_client

        load_client(manifest.multimodal_http_client)
        path = public_config(manifest.multimodal_http_client)
        configs[str(path.relative_to(ROOT))] = checksum(path.read_bytes())
    if isinstance(manifest, (ModelSelectedManifest, UniformModelManifest)) and not isinstance(
        manifest, LocalProviderManifest
    ):
        from packages.llm_provider.model_selection import load_selection

        selection = load_selection(
            manifest.fairvlmed_model_selection
            if isinstance(manifest, UniformModelManifest)
            else manifest.model_selection
        )
        for name in (selection["path"], selection["selection"]["profile"]):
            path = public_config(name)
            configs[str(path.relative_to(ROOT))] = checksum(path.read_bytes())
    if isinstance(manifest, RoutedManifest):
        from packages.llm_provider.config import read_provider_config
        from packages.llm_provider.network_route import load_route, route_opener

        route = load_route(manifest.network_route)
        route_opener(route, "https://api.relayrouter.ai/v1")
        path = public_config(manifest.network_route)
        configs[str(path.relative_to(ROOT))] = checksum(path.read_bytes())
        if any(c.dataset == "fairvlmed" for c in manifest.cases):
            from .generic_input import ENVIRONMENT_PROVIDER, LOCAL_PROVIDER, ROUTED_PROVIDER

            config_path = public_config(
                ENVIRONMENT_PROVIDER
                if isinstance(manifest, AllEnvironmentContinuationManifest)
                else LOCAL_PROVIDER
                if isinstance(manifest, LocalProviderManifest)
                else ROUTED_PROVIDER
            )
            selected = read_provider_config(read(config_path))
            if selected.network_route != route:
                raise ValueError("cohort_fairvlmed_network_route_binding")
            configs[str(config_path.relative_to(ROOT))] = checksum(config_path.read_bytes())
    for case in manifest.cases:
        for name in (case.registry, case.active_acquisition):
            if name is not None:
                path = public_config(name)
                configs[str(path.relative_to(ROOT))] = checksum(path.read_bytes())
        # Validate the command before creating a partially prepared cohort.
        commands(manifest, case, output, "prepare")
    bindings = provider_bindings(manifest) if isinstance(manifest, UniformModelManifest) else None
    output.mkdir(parents=True, mode=0o700)
    (output / "inputs").mkdir()
    publish_json(output / "manifest.json", manifest)
    freeze = dict(
        schema_version="terminal-cohort-freeze/1.0.0",
        manifest_checksum=digest(manifest),
        implementation=implementation(),
        configurations=configs,
        output=str(output),
        clinical_validation=False,
        automatic_unknown_replay=False,
    )
    if bindings is not None:
        freeze["provider_bindings"] = bindings
    if continuation is not None:
        freeze["continuation"] = continuation
    if isinstance(manifest, ChildRuntimeLeaseManifest):
        freeze.update(
            schema_version="terminal-cohort-freeze/1.1.0",
            runtime_owner=runtime_owner_implementation(),
        )
    publish_json(output / "freeze.json", freeze)
    return dict(output=str(output), cases=len(manifest.cases), api_requests=0)


def load(output, *, check_current):
    output = Path(output).resolve()
    manifest = read_manifest(read(output / "manifest.json"))
    freeze = read(output / "freeze.json")
    if freeze["manifest_checksum"] != digest(manifest) or freeze["output"] != str(output):
        raise ValueError("cohort_manifest_binding")
    if check_current:
        if isinstance(manifest, ChildRuntimeLeaseManifest) and (
            freeze.get("schema_version") != "terminal-cohort-freeze/1.1.0"
            or freeze.get("runtime_owner") != runtime_owner_implementation()
        ):
            raise ValueError("cohort_runtime_owner_changed")
        if freeze["implementation"] != implementation():
            raise ValueError("cohort_implementation_changed")
        if any(
            checksum(public_config(p).read_bytes()) != h
            for p, h in freeze["configurations"].items()
        ):
            raise ValueError("cohort_configuration_changed")
        if isinstance(manifest, UniformModelManifest) and freeze.get(
            "provider_bindings"
        ) != provider_bindings(manifest):
            raise ValueError("cohort_provider_binding_changed")
        if isinstance(manifest, ENVIRONMENT_CONTINUATIONS) and freeze.get(
            "continuation"
        ) != check_unstarted_continuation(manifest, output):
            raise ValueError("cohort_continuation_changed")
    return output, manifest, freeze


def runtime_supervision_session(output):
    """Use the original Runtime session owner; no case grant or persistent write.

    The key depends only on the canonical directory, not a mutable manifest or
    selected subset. A lost session never permits replay of a durable intent.
    The original child owners still enforce their own execution reservations.
    """
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_pipeline import CasePipelineRepository

    target = credential_boundary().resolve_target("runtime")
    key = "terminal-cohort-supervision:" + digest(str(Path(output).resolve()))[7:]
    return CasePipelineRepository(target).guarded_session("COHORT:SUPERVISION", key)


@contextmanager
def lock(output, manifest=None):
    if isinstance(manifest, RuntimeLeaseManifest):
        with runtime_supervision_session(output) as lease:
            yield lease
        return
    with (output / "cohort.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("cohort_operation_in_progress") from None
        yield


def child_lock_capability(output):
    """Probe the actual workspace used by legacy case/tool locks, without dispatch.

    This is intentionally not a silent /tmp fallback. Until the children have
    explicit versioned Runtime leases, a working supervisor alone is insufficient.
    """
    try:
        with tempfile.TemporaryFile(dir=output) as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP):
            raise
        return dict(available=False, errno=error.errno, implicit_fallback=False)
    return dict(available=True, errno=None, implicit_fallback=False)


def execution_lock_capability(output, manifest, cases=None):
    filesystem = child_lock_capability(output)
    if not isinstance(manifest, ChildRuntimeLeaseManifest):
        return dict(available=filesystem["available"], filesystem=filesystem)
    from .generic_execution_lock import capability

    selected = manifest.cases if cases is None else cases
    fair = capability() if any(c.dataset == "fairvlmed" for c in selected) else None
    multi = None
    if isinstance(manifest, AllChildRuntimeLeaseManifest) and any(
        c.dataset != "fairvlmed" for c in selected
    ):
        from packages.multimodal_research.execution_lease import entry_capability

        multi = entry_capability()
    per_case = {
        c.id: bool(fair and fair["available"])
        if c.dataset == "fairvlmed"
        else bool(multi["available"])
        if multi is not None
        else filesystem["available"]
        for c in selected
    }
    return dict(
        available=all(per_case.values()),
        filesystem=filesystem,
        fairvlmed=fair,
        multimodal=multi,
        cases=per_case,
        implicit_fallback=False,
    )


def lease_preflight(output):
    output, manifest, _ = load(output, check_current=True)
    with lock(output, manifest) as lease:
        load(output, check_current=True)
        if lease is not None:
            lease()
        children = execution_lock_capability(output, manifest)
        if lease is not None:
            lease()
    return dict(
        schema_version="terminal-cohort-lease-preflight/1.0.0",
        supervision_lease_acquired=True,
        supervision_profile=getattr(manifest, "supervision_lease", "legacy-filesystem-flock"),
        child_filesystem_lock=children["filesystem"],
        child_execution_locks=children,
        execution_lock_ready=children["available"],
        owner_readiness_not_inferred=True,
        api_requests=0,
        database_writes=0,
    )


def process_identity(pid):
    """Linux process identity, not an inference from a stale PID or lock."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw.rsplit(") ", 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0])
    except (OSError, IndexError):
        return None


def process_status(directory):
    path = directory / "process.json"
    saved = read(path) if path.exists() else None
    if not saved or not saved.get("start_ticks"):
        return "identity_not_recorded_no_liveness_inference"
    current = process_identity(saved["pid"])
    if current and current["start_ticks"] == saved["start_ticks"]:
        return "live_same_process" if current["state"] != "Z" else "exited_waiting_for_receipt"
    return "original_process_not_present_owner_reconciliation_required"


def check_pending_commands(output):
    """A lost supervisor cannot skip its unresolved child by selecting another case."""
    for intent in sorted((output / "commands").glob("*/*/intent.json")):
        receipt = intent.with_name("result.json")
        if not receipt.exists():
            raise ValueError("cohort_unconfirmed_command_requires_owner_reconciliation")
        saved = read(receipt)
        if saved.get("status") != "command_exited" or type(saved.get("exit_code")) is not int:
            raise ValueError("cohort_uncertain_command_requires_owner_reconciliation")


def stop_child(process):
    """Stop our process group and collect its actual exit, never a remote receipt."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def operation(output, case, action, argv, timeout, *, lease_guard=None):
    directory = output / "commands" / case.id / action
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = directory / "result.json"
    intent = directory / "intent.json"
    if receipt.exists():
        value = read(receipt)
        if value["command"] != argv:
            raise ValueError("cohort_command_conflict")
        if value["log_checksum"] != checksum((directory / "stdout.log").read_bytes()):
            raise ValueError("cohort_command_log_changed")
        return value
    if intent.exists():
        return dict(
            status="unconfirmed_process_requires_owner_inspection",
            exit_code=None,
            replay_authorized=False,
            command=argv,
        )
    if lease_guard is not None:
        lease_guard()
    publish_json(
        intent,
        dict(
            command=argv,
            started_at=datetime.now(UTC).isoformat(),
            supervisor_pid=os.getpid(),
            retry_allowed=False,
        ),
    )
    started = time.monotonic()
    code, status = None, "launch_error"
    child_exit = None
    with (directory / "stdout.log").open("xb") as log:
        os.chmod(log.name, 0o600)
        try:
            if lease_guard is not None:
                lease_guard()
            process = subprocess.Popen(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        except SessionLeaseLostError:
            status = "lease_lost_remote_state_not_inferred"
        except OSError:
            pass
        else:
            identity = process_identity(process.pid)
            publish_json(directory / "process.json", identity or dict(pid=process.pid))
            try:
                if lease_guard is None:
                    code = process.wait(timeout=timeout)
                else:
                    deadline = started + timeout
                    while True:
                        lease_guard()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(argv, timeout)
                        try:
                            code = process.wait(timeout=min(1.0, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() >= deadline:
                                raise
                status = "command_exited"
            except subprocess.TimeoutExpired:
                status = "timeout_remote_state_not_inferred"
                code = stop_child(process)
            except SessionLeaseLostError:
                status = "lease_lost_remote_state_not_inferred"
                child_exit = stop_child(process)
    value = dict(
        command=argv,
        status=status,
        exit_code=code,
        elapsed_seconds=time.monotonic() - started,
        ended_at=datetime.now(UTC).isoformat(),
        log_checksum=checksum((directory / "stdout.log").read_bytes()),
        medical_completion="not_inferred_from_process_exit",
        replay_authorized=False,
    )
    if status == "lease_lost_remote_state_not_inferred":
        value.update(
            schema_version="terminal-command-receipt/1.1.0",
            child_exit_code=child_exit,
            supervision_lease_lost=True,
        )
    publish_json(receipt, value)
    return value


def status(output):
    output, manifest, _ = load(output, check_current=False)
    rows = []
    for case in manifest.cases:
        actions = {}
        for action in (
            "prepare",
            "preflight",
            "run",
            "export",
            "source-prepare",
            "source-preflight",
            "source-run",
            "source-export",
        ):
            path = output / "commands" / case.id / action
            actions[action] = (
                read(path / "result.json")
                if (path / "result.json").exists()
                else (
                    dict(
                        status="started_or_interrupted_not_authorized_for_replay",
                        process_state=process_status(path),
                    )
                    if (path / "intent.json").exists()
                    else dict(status="not_started")
                )
            )
        try:
            if case.dataset == "fairvlmed":
                from packages.evaluation_preparation.cohort_progress import fair
                from packages.evaluation_preparation.generic_input import directory

                argv = commands(manifest, case, output, "prepare")[0]
                progress = fair(directory(argv[argv.index("--key") + 1]))
            else:
                from packages.evaluation_preparation.cohort_progress import multimodal

                argv = commands(manifest, case, output, "prepare")[0]
                progress = multimodal(argv[argv.index("--output") + 1])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            progress = dict(
                status="progress_not_readable", error_type=type(exc).__name__, authorization=False
            )
        rows.append(
            dict(
                case=case.id,
                dataset=case.dataset,
                sample=case.sample,
                task=case.task,
                commands=actions,
                progress=progress,
                recovery_progress=recovery_progress(manifest, case, output),
            )
        )
    return dict(
        schema_version="terminal-cohort-status/1.1.0",
        cases=rows,
        semantics="process_supervision_not_clinical_or_owner_acceptance",
        status_command_api_requests=0,
        case_api_requests="read_original_request_journal_not_inferred_from_logs",
    )


def recovery_progress(manifest, case, output):
    """Optional saved successor mirror, not a substitute for original-owner checks."""
    if case.dataset == "fairvlmed" or manifest.schema_version not in {
        "terminal-case-cohort/1.1.0",
        "terminal-case-cohort/1.3.0",
        "terminal-case-cohort/1.4.0",
        "terminal-case-cohort/1.5.0",
        "terminal-case-cohort/1.6.0",
        "terminal-case-cohort/1.7.0",
        "terminal-case-cohort/1.8.0",
        "terminal-case-cohort/1.9.0",
    }:
        return dict(status="not_supported", authorization=False)
    from .cohort_progress import multimodal

    argv = commands(manifest, case, output, "source-run")[0]
    try:
        return multimodal(argv[argv.index("--input") + 1])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(
            status="progress_not_readable", error_type=type(exc).__name__, authorization=False
        )


def source_recovery_ready(output, manifest, case, *, runtime_backlog=False):
    """Fast rejection only. Original prepare/Runtime still authorizes the successor."""
    from .cohort_progress import bounded

    if case.dataset == "fairvlmed" or (
        manifest.schema_version
        not in {
            "terminal-case-cohort/1.1.0",
            "terminal-case-cohort/1.3.0",
            "terminal-case-cohort/1.4.0",
            "terminal-case-cohort/1.5.0",
            "terminal-case-cohort/1.6.0",
            "terminal-case-cohort/1.7.0",
            "terminal-case-cohort/1.8.0",
            "terminal-case-cohort/1.9.0",
        }
        and not (runtime_backlog and isinstance(manifest, LocalProviderManifest))
    ):
        return False
    receipt = output / "commands" / case.id / "run" / "result.json"
    if not receipt.exists():
        return False  # Missing exit is not a failed run we may resume.
    try:
        saved, _ = bounded(receipt)
        log_checksum = checksum((receipt.parent / "stdout.log").read_bytes())
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(saved, dict):
        return False
    if (
        saved.get("status") != "command_exited"
        or type(saved.get("exit_code")) is not int
        or saved["exit_code"] == 0
        or saved.get("command") != commands(manifest, case, output, "run")[0]
        or saved.get("log_checksum") != log_checksum
    ):
        return False
    argv = commands(manifest, case, output, "prepare")[0]
    parent = Path(argv[argv.index("--output") + 1])
    if (parent / "cancel").exists() or (parent / "result.json").exists():
        return False
    try:
        failure, _ = bounded(parent / "stage-failure.json")
    except (OSError, ValueError, TypeError):
        return False
    if isinstance(manifest, RoutedRecoveryManifest):
        transport = failure.get("transport") or {}
        unknown = [transport.get(key) for key in ("remote_response_unknown", "dispatch_unknown")]
        if any(type(n) is not int or n not in (0, 1) for n in unknown):
            return False
        error = failure.get("error_code")
        return (
            failure.get("stage") == "definitions"
            and error in {"schema", "timeout", "transport"}
            and (
                error == "schema"
                and failure.get("failure_kind") == "confirmed_response_or_domain_refusal"
                and unknown == [0, 0]
                or error in {"timeout", "transport"}
                and failure.get("failure_kind") == "dispatched_unknown"
                and sum(unknown) == 1
            )
        )
    return (
        failure.get("stage") == "definitions"
        and failure.get("error_code") == "schema"
        and failure.get("failure_kind") == "confirmed_response_or_domain_refusal"
        and failure.get("transport", {}).get("remote_response_unknown") == 0
        and failure.get("transport", {}).get("dispatch_unknown") == 0
    )


def case_action(output, manifest, case, action, *, lease_guard=None):
    load(output, check_current=True)
    prerequisites = {
        "prepare": (),
        "preflight": ("prepare",),
        "run": ("prepare", "preflight"),
        "export": ("prepare",),
        "source-prepare": ("prepare", "preflight"),
        "source-preflight": ("source-prepare",),
        "source-run": ("source-prepare", "source-preflight"),
        "source-export": ("source-prepare",),
    }[action]
    missing = []
    for needed in prerequisites:
        path = output / "commands" / case.id / needed / "result.json"
        if not path.exists() or read(path)["exit_code"] != 0:
            missing.append(needed)
    if (
        action.startswith("source-")
        and action != "source-export"
        and not source_recovery_ready(output, manifest, case)
    ):
        missing.append(
            "original_eligible_source_backlog"
            if isinstance(manifest, RoutedRecoveryManifest)
            else "original_confirmed_source_failure"
        )
    if missing:
        result = dict(
            case=case.id,
            action=action,
            status="prerequisite_not_completed",
            exit_code=None,
            missing=missing,
            replay_authorized=False,
        )
        print(json.dumps(result), flush=True)
        return [result]
    print(
        json.dumps(dict(case=case.id, action=action, status="starting_or_reading_receipt")),
        flush=True,
    )
    results = []
    for argv in commands(manifest, case, output, action):
        options = {} if lease_guard is None else {"lease_guard": lease_guard}
        result = operation(output, case, action, argv, manifest.command_timeout_seconds, **options)
        results.append(dict(case=case.id, action=action, **result))
        print(
            json.dumps(
                dict(
                    case=case.id,
                    action=action,
                    exit_code=result["exit_code"],
                    status=result["status"],
                )
            ),
            flush=True,
        )
    return results


def run(output, action, case_ids):
    if action not in {
        "prepare",
        "preflight",
        "run",
        "export",
        "execute",
        "recover-source",
        "export-recovery",
    }:
        raise ValueError("cohort_unknown_action")
    output, manifest, _ = load(output, check_current=True)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("cohort_explicit_unique_case_selection_required")
    selected = [c for c in manifest.cases if c.id in case_ids]
    if len(selected) != len(case_ids) or len(selected) > manifest.max_cases_per_invocation:
        raise ValueError("cohort_case_selection_limit")
    results = []
    with lock(output, manifest) as lease:
        load(output, check_current=True)
        if isinstance(manifest, RuntimeLeaseManifest):
            check_pending_commands(output)
        if (
            isinstance(manifest, RuntimeLeaseManifest)
            and not execution_lock_capability(output, manifest, selected)["available"]
        ):
            raise ValueError("cohort_child_filesystem_lock_unavailable")
        for case in selected:
            # Prepare just in time. Preparing all cases hours before execution
            # would spend their original absolute deadlines while they queued.
            actions = (
                ("prepare", "preflight", "run")
                if action == "execute"
                else ("source-prepare", "source-preflight", "source-run")
                if action == "recover-source"
                else ("source-export",)
                if action == "export-recovery"
                else (action,)
            )
            for current in actions:
                options = {}
                if lease is not None:
                    lease()
                    check_pending_commands(output)
                    options["lease_guard"] = lease
                values = case_action(output, manifest, case, current, **options)
                results.extend(values)
                if any(v["exit_code"] != 0 for v in values):
                    break
    return results


def local_recovery_identity(cohort, manifest, case, plan):
    """Original owners bind a stopped fact lineage to the declared cohort case.

    This is not an acceptance shortcut: the child aggregation command separately
    verifies every Review origin against current facts before any local scoring.
    """
    from packages.multimodal_research.accepted_aggregation import checked_path, plan_campaign
    from packages.multimodal_research.campaign_renewal import completed_parent

    argv = commands(manifest, case, cohort, "prepare")[0]
    root = checked_path(argv[argv.index("--output") + 1])
    original, root_result = completed_parent(root)
    facts, result, _ = plan_campaign(plan)
    if (
        result["case_code"] != root_result["case_code"]
        or result["task"] != root_result["task"]
        or facts.base.manifest.checksum != original.base.manifest.checksum
    ):
        raise ValueError("cohort_local_recovery_case_binding")
    expected = dict(
        directory=str(root), checksum=digest(root_result), requests=root_result["new_requests"]
    )
    lineage = result["previous_attempts"] + [
        dict(
            directory=str(checked_path(plan.facts)),
            checksum=digest(result),
            requests=result["new_requests"],
        )
    ]
    if sum(row == expected for row in lineage) != 1:
        raise ValueError("cohort_local_recovery_root_lineage")
    return dict(root=str(root), root_result=digest(root_result), fact_result=digest(result))


def recover_accepted(cohort, case_id, plan_path, output):
    """Explicit local repair under a new freeze; never relax old cohort dispatch gates."""
    from packages.multimodal_research.accepted_aggregation import (
        AccountedAggregationPlan,
        checked_path,
        read_plan,
    )

    cohort, manifest, old_freeze = load(cohort, check_current=False)
    selected = [case for case in manifest.cases if case.id == case_id]
    if len(selected) != 1 or selected[0].dataset == "fairvlmed":
        raise ValueError("cohort_local_recovery_multimodal_case_required")
    case = selected[0]
    plan = read_plan(read(checked_path(plan_path)))
    if not isinstance(plan, AccountedAggregationPlan):
        raise ValueError("cohort_local_recovery_accounted_plan_required")
    out = checked_path(output)
    protected = [
        cohort,
        checked_path(plan.facts),
        *(checked_path(origin.campaign) for origin in plan.origins),
        *(
            checked_path(origin.local_finalization)
            for origin in plan.origins
            if origin.local_finalization
        ),
    ]
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError("cohort_local_recovery_output_overlaps_history")
    # Explicit local recovery may follow an implementation repair, but public
    # configuration drift is not silently authorized for old case consumption.
    if any(
        checksum(public_config(p).read_bytes()) != h
        for p, h in old_freeze["configurations"].items()
    ):
        raise ValueError("cohort_configuration_changed")
    current = implementation()
    identity = local_recovery_identity(cohort, manifest, case, plan)
    record = dict(
        schema_version="terminal-cohort-local-recovery/1.0.0",
        original_cohort=str(cohort),
        original_freeze_checksum=digest(old_freeze),
        case=case.id,
        plan=plan.model_dump(mode="json"),
        identity=identity,
        implementation=current,
        mode="local_accepted_aggregation_only",
        maximum_new_api_requests=0,
        command_timeout_seconds=900,
        automatic_unknown_replay=False,
    )
    if out.exists() and (
        not (out / "recovery.json").is_file() or read(out / "recovery.json") != record
    ):
        raise ValueError("cohort_local_recovery_freeze_or_plan_conflict")
    # The original exporter derives filenames from the result directory name.
    # Bind that basename to the full supervision path and plan, not a shared
    # literal "aggregation" which would collide across independent recoveries.
    aggregation = out / ("local-fold-" + digest((str(out), plan))[7:39])
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(out):
        if implementation() != current:
            raise ValueError("cohort_implementation_changed")
        publish_json(out / "recovery.json", record)
        publish_json(out / "aggregation-plan.json", plan)
        argv = [
            sys.executable,
            "-u",
            "-m",
            "apps.multimodal_case",
            "aggregate-accepted",
            "--plan",
            str(out / "aggregation-plan.json"),
            "--output",
            str(aggregation),
        ]
        receipt = operation(out, case, "aggregate-accepted", argv, 900)
    return dict(
        case=case.id,
        action="recover-accepted",
        **receipt,
        aggregation_output=str(aggregation),
        current_consumption_authorized=False,
        whole_case_coverage="requires_separate_owner_audit",
        scope="local_aggregation_not_new_review_or_unknown_replay",
    )


def progress_text(value):
    if value.get("status") == "local_original_fairvlmed_status":
        return f"工具步骤回执 {value['tool_step_records']}；结果已保存 {value['result_saved']}"
    if value.get("status") != "local_mirrors_only":
        return value.get("status", "unknown")
    completion = value.get("completion") or {}
    pending = completion.get("pending_review_ids")
    pending_count = len(pending) if isinstance(pending, list) else "未知"
    conclusion = (
        f"评阅收尾 {completion.get('status', 'unknown')}；"
        f"待评阅 {pending_count}；"
        f"远端响应未知至少 {completion.get('remote_response_unknown_lower_bound', '未知')}；"
        "退出0或局部接纳不代表整例完成；"
    )
    return (
        f"观察阶段已接纳 {value['saved_accepted_visual_phases']}；"
        f"Review阶段已接纳 {value['saved_accepted_review_phases']}；"
        f"已记录阶段API至少 {value['phase_api_requests_lower_bound']}（未含Source/未完成请求）；"
        f"镜像读取错误 {len(value['errors'])}；"
        + conclusion
        + "; ".join(
            f"{p.get('target', p['path'])}: {p['status']} / error={p.get('error')}"
            for p in value["phases"]
        )
    )


def export_status(output):
    value = status(output)
    path = Path(output) / ("status-" + digest(value)[7:23] + ".html")
    rows = "".join(
        "<tr><td>"
        + "</td><td>".join(
            html.escape(str(v))
            for v in (
                c["case"],
                c["dataset"],
                c["sample"],
                c["task"],
                "; ".join(
                    k + ": " + v["status"] + " / exit=" + str(v.get("exit_code"))
                    for k, v in c["commands"].items()
                ),
                progress_text(c["progress"]),
                progress_text(c["recovery_progress"]),
            )
        )
        + "</td></tr>"
        for c in value["cases"]
    )
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>病例批次进度</title>'
    page += (
        "<h1>终端批次进度</h1><p>这是进程状态，不是医学接纳或完整取证结论；"
        "未确认请求不会自动重发。</p>"
    )
    page += (
        '<table border="1"><tr><th>病例</th><th>数据集</th><th>输入</th>'
        "<th>命题</th><th>命令状态</th><th>原阶段镜像进度（不授权执行）</th>"
        "<th>Source后继镜像（不授权执行）</th></tr>" + rows + "</table></html>"
    )
    publish_html(path, page)
    return dict(html=str(path), api_requests=0)

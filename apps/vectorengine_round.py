"""Explicit next-round preparation and locks; uses the original 07/08 execution owners."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from apps.vectorengine_round_materials import RoundCaseEvidence, load_catalog
from apps.vectorengine_single_case import load_config, open_case, write_once
from packages.grounding.models import Checksum, Code, Token, Value, canonical, digest, validate
from packages.object_store import checksum


class LiveRound(Value):
    schema_version: Literal["bounded-live-round/1.0.0"] = "bounded-live-round/1.0.0"
    round_id: Token
    run_id: Code
    request_key: Token
    request_code: Code
    provider_config: Literal["configs/agents/vectorengine_single_case.json"]
    provider_checksum: Checksum
    material_catalog: Literal["configs/research/vectorengine_round2_materials.json"]
    material_checksum: Checksum
    output: str
    gate: str
    max_requests: Literal[40]
    max_concurrency: Literal[1]
    seconds: Literal[2700]
    max_steps: Literal[20]
    max_no_progress: Literal[2]
    max_retries: Literal[1]
    max_searches: Literal[8]
    deadline_origin: Literal["original_runtime_created_at_no_reset"]
    phase_order: Literal["public_probes_then_admitted_single_case"]
    preparation_authorizes_requests: Literal[False]

    @model_validator(mode="after")
    def identity(self):
        from db.runtime.repository.case_pipeline import identity

        if (
            self.run_id != identity("RUN", self.round_id)
            or self.request_key != self.round_id
            or self.round_id == "vectorengine-single-case-v1"
            or self.request_code != "RESEARCH:VECTOR_SINGLE_R02"
            or self.output != ".local/" + self.round_id + "/prepared-02"
            or self.gate != "docs/acceptance/12-live-round2-preparation-v1/execution-gate.json"
        ):
            raise ValueError("round_explicit_identity")
        return self


class LiveRoundV2(LiveRound):
    """Explicit round-03 only; v1 rules and its saved configuration stay frozen."""

    schema_version: Literal["bounded-live-round/2.0.0"] = "bounded-live-round/2.0.0"
    round_id: Literal["vectorengine-single-case-round-03"]
    request_code: Literal["RESEARCH:VECTOR_SINGLE_R03"]
    output: Literal[".local/vectorengine-single-case-round-03/prepared-01"]
    gate: Literal["docs/acceptance/12-live-round3-v1/execution-gate.json"]

    @model_validator(mode="after")
    def identity(self):
        from db.runtime.repository.case_pipeline import identity

        if self.run_id != identity("RUN", self.round_id) or self.request_key != self.round_id:
            raise ValueError("round_explicit_identity")
        return self


class LiveRoundV3(LiveRound):
    """One derived identity for a public diagnostic stage and its single case stage."""

    schema_version: Literal["bounded-live-round/3.0.0"] = "bounded-live-round/3.0.0"
    session_id: Token
    stage: Literal["public", "case"]
    provider_config: str
    public_config: str
    phase_order: Literal["qualified_public_then_single_case"]
    seconds: Literal[1200, 2700]

    @model_validator(mode="after")
    def identity(self):
        from db.runtime.repository.case_pipeline import identity

        if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+){0,4}", self.session_id):
            raise ValueError("session_identity")
        stem = "vectorengine-" + self.session_id
        if (
            self.round_id != stem + "-" + self.stage
            or self.run_id != identity("RUN", self.round_id)
            or self.request_key != self.round_id
            or self.request_code != "RESEARCH:" + self.round_id.upper().replace("-", "_")
            or self.output != ".local/" + self.round_id + "/prepared-01"
            or self.gate
            != "docs/acceptance/12-live-"
            + self.session_id
            + "-v1/"
            + self.stage
            + "-execution-gate.json"
            or self.provider_config
            not in {
                "configs/agents/" + stem.replace("-", "_") + suffix
                for suffix in (".json", "_json.json")
            }
            or self.public_config != "configs/research/" + stem.replace("-", "_") + "_public.json"
            or self.seconds != (1200 if self.stage == "public" else 2700)
        ):
            raise ValueError("round_explicit_identity")
        return self


class CaseTargetRepairRound(LiveRoundV3):
    """One new case identity consuming the unchanged successful public round.

    This is not a reset of the cancelled case or a new public request budget.
    Current zero-request cancellation and the explicit repair delta are checked
    at consumption, before any new run is created.
    """

    schema_version: Literal["bounded-live-round/4.0.0"] = "bounded-live-round/4.0.0"
    session_id: Literal["batch-citations"]
    stage: Literal["case"]
    prior_case_config: Literal["configs/research/vectorengine_batch_citations_case.json"]

    @model_validator(mode="after")
    def identity(self):
        from db.runtime.repository.case_pipeline import identity

        stem = "vectorengine-batch-citations-case-target-r01"
        if (
            self.round_id != stem
            or self.run_id != identity("RUN", stem)
            or self.request_key != stem
            or self.request_code != "RESEARCH:" + stem.upper().replace("-", "_")
            or self.provider_config != "configs/agents/vectorengine_batch_citations.json"
            or self.public_config != "configs/research/vectorengine_batch_citations_public.json"
            or self.output != ".local/" + stem + "/prepared-01"
            or self.gate
            != "docs/acceptance/12-completion-c-target-repair-v1/case-execution-gate.json"
            or self.seconds != 2700
        ):
            raise ValueError("case_target_repair_identity")
        return self


class CaseQuoteRepairRound(CaseTargetRepairRound):
    """New bounded case with opt-in layout proof and charged previous case requests."""

    schema_version: Literal["bounded-live-round/5.0.0"] = "bounded-live-round/5.0.0"
    prior_case_config: Literal["configs/research/vectorengine_batch_citations_case_target_r01.json"]
    source_quote_policy: Literal["research-quote-policy/layout-whitespace/1.0.0"]

    @model_validator(mode="after")
    def identity(self):
        from db.runtime.repository.case_pipeline import identity

        stem = "vectorengine-batch-citations-case-quote-r01"
        if (
            self.round_id != stem
            or self.run_id != identity("RUN", stem)
            or self.request_key != stem
            or self.request_code != "RESEARCH:" + stem.upper().replace("-", "_")
            or self.provider_config != "configs/agents/vectorengine_batch_citations.json"
            or self.public_config != "configs/research/vectorengine_batch_citations_public.json"
            or self.output != ".local/" + stem + "/prepared-01"
            or self.gate
            != "docs/acceptance/12-completion-c-quote-integration-v1/case-execution-gate.json"
            or self.seconds != 2700
        ):
            raise ValueError("case_quote_repair_identity")
        return self


class IntegratedBatchRound(LiveRoundV3):
    schema_version: Literal["bounded-live-round/6.0.0"] = "bounded-live-round/6.0.0"
    session_id: Literal["batch-integrated"]
    prior_round_configs: tuple[str, ...]
    source_quote_policy: Literal["research-quote-policy/layout-whitespace/1.0.0"]

    @model_validator(mode="after")
    def prior_identity(self):
        if self.prior_round_configs != (
            "configs/research/vectorengine_batch_citations_public.json",
            "configs/research/vectorengine_batch_citations_case_target_r01.json",
            "configs/research/vectorengine_batch_citations_case_quote_r01.json",
        ):
            raise ValueError("integrated_history_inventory")
        return self


class SourceSelectionRound(LiveRoundV3):
    schema_version: Literal["bounded-live-round/7.0.0"] = "bounded-live-round/7.0.0"
    session_id: Literal["batch-integrated-r01"]
    prior_round_configs: tuple[str, ...]
    source_quote_policy: Literal["research-quote-policy/layout-whitespace/1.0.0"]

    @model_validator(mode="after")
    def history_identity(self):
        if self.prior_round_configs != (
            "configs/research/vectorengine_batch_citations_public.json",
            "configs/research/vectorengine_batch_citations_case_target_r01.json",
            "configs/research/vectorengine_batch_citations_case_quote_r01.json",
            "configs/research/vectorengine_batch_integrated_public.json",
            "configs/research/vectorengine_batch_integrated_case.json",
        ):
            raise ValueError("source_selection_history_inventory")
        return self


class CurrentReadPerformanceRound(SourceSelectionRound):
    """Case-only continuation with unchanged Provider/public wire and charged history."""

    schema_version: Literal["bounded-live-round/8.0.0"] = "bounded-live-round/8.0.0"
    stage: Literal["case"]

    @model_validator(mode="after")
    def identity(self):
        from db.runtime.repository.case_pipeline import identity

        stem = "vectorengine-batch-integrated-r01-case-runtime-r01"
        if (
            self.round_id != stem
            or self.run_id != identity("RUN", stem)
            or self.request_key != stem
            or self.request_code != "RESEARCH:" + stem.upper().replace("-", "_")
            or self.provider_config != "configs/agents/vectorengine_batch_integrated_r01.json"
            or self.public_config
            != "configs/research/vectorengine_batch_integrated_r01_public.json"
            or self.output != ".local/" + stem + "/prepared-01"
            or self.gate != "docs/acceptance/12-completion-performance-v1/case-execution-gate.json"
            or self.seconds != 2700
        ):
            raise ValueError("performance_case_identity")
        return self

    @model_validator(mode="after")
    def history_identity(self):
        if self.prior_round_configs != (
            "configs/research/vectorengine_batch_citations_public.json",
            "configs/research/vectorengine_batch_citations_case_target_r01.json",
            "configs/research/vectorengine_batch_citations_case_quote_r01.json",
            "configs/research/vectorengine_batch_integrated_public.json",
            "configs/research/vectorengine_batch_integrated_case.json",
            "configs/research/vectorengine_batch_integrated_r01_case.json",
        ):
            raise ValueError("performance_history_inventory")
        return self


class ActiveAcquisitionRound(LiveRoundV3):
    """New user-authorized pilot session; never reopens an older stopped run."""

    schema_version: Literal["bounded-live-round/9.0.0"] = "bounded-live-round/9.0.0"
    session_id: Literal["active-acquisition-v1"]
    acquisition_manifest: Literal["docs/experiments/active-acquisition-v1/manifest.json"]
    acquisition_checksum: Checksum
    source_quote_policy: Literal["research-quote-policy/layout-whitespace/1.0.0"]


class ActiveAcquisitionStatusRound(ActiveAcquisitionRound):
    """One explicit status-contract correction; previous spending remains charged."""

    schema_version: Literal["bounded-live-round/10.0.0"] = "bounded-live-round/10.0.0"
    session_id: Literal["active-acquisition-r01"]
    prior_round_configs: tuple[str, ...]

    @model_validator(mode="after")
    def active_history(self):
        if self.prior_round_configs != (
            "configs/research/vectorengine_active_acquisition_v1_public.json",
            "configs/research/vectorengine_active_acquisition_v1_case.json",
        ):
            raise ValueError("active_history_inventory")
        return self


class ActiveAcquisitionAliasRound(ActiveAcquisitionRound):
    """Short citation projection, retaining all previous active pilot spending."""

    schema_version: Literal["bounded-live-round/11.0.0"] = "bounded-live-round/11.0.0"
    session_id: Literal["active-acquisition-r02"]
    prior_round_configs: tuple[str, ...]

    @model_validator(mode="after")
    def alias_history(self):
        expected = tuple(
            f"configs/research/vectorengine_active_acquisition_{profile}_{stage}.json"
            for profile in ("v1", "r01")
            for stage in ("public", "case")
        )
        if self.prior_round_configs != expected:
            raise ValueError("active_history_inventory")
        return self


def evidence_for_round(original, round_config):
    cls = RoundCaseEvidence
    extra = {}
    if isinstance(round_config, ActiveAcquisitionRound):
        from apps.active_evidence_materials import ActiveCaseEvidence

        cls = ActiveCaseEvidence
        extra["manifest_path"] = round_config.acquisition_manifest
        extra["short_sources"] = isinstance(round_config, ActiveAcquisitionAliasRound)
    return cls(
        original,
        load_catalog(round_config.material_catalog),
        catalog_path=round_config.material_catalog,
        **extra,
    )


def load_round(path):
    raw = json.loads(Path(path).read_text())
    model = {
        "bounded-live-round/1.0.0": LiveRound,
        "bounded-live-round/2.0.0": LiveRoundV2,
        "bounded-live-round/3.0.0": LiveRoundV3,
        "bounded-live-round/4.0.0": CaseTargetRepairRound,
        "bounded-live-round/5.0.0": CaseQuoteRepairRound,
        "bounded-live-round/6.0.0": IntegratedBatchRound,
        "bounded-live-round/7.0.0": SourceSelectionRound,
        "bounded-live-round/8.0.0": CurrentReadPerformanceRound,
        "bounded-live-round/9.0.0": ActiveAcquisitionRound,
        "bounded-live-round/10.0.0": ActiveAcquisitionStatusRound,
        "bounded-live-round/11.0.0": ActiveAcquisitionAliasRound,
    }.get(raw.get("schema_version"))
    if model is None:
        raise ValueError("round_version")
    round_config = validate(model, raw)
    if digest(load_config(round_config.provider_config)) != round_config.provider_checksum:
        raise ValueError("round_provider_lock")
    if digest(load_catalog(round_config.material_catalog)) != round_config.material_checksum:
        raise ValueError("round_material_lock")
    if isinstance(round_config, ActiveAcquisitionRound):
        from apps.active_evidence_materials import load_manifest

        if (
            digest(load_manifest(round_config.acquisition_manifest))
            != round_config.acquisition_checksum
        ):
            raise ValueError("round_acquisition_lock")
    return round_config


def implementation_files():
    paths = {
        str(p)
        for base in ("apps", "agents", "packages", "configs", "db", "tests")
        for p in Path(base).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".sql", ".txt"}
    }
    paths.add(".local/vectorengine/run.py")  # Source only; never the adjacent key file.
    return paths


def check_round_gate(round_config, initial=None):
    gate = json.loads(Path(round_config.gate).read_text())
    if (
        gate.get("schema_version") != "bounded-live-round-gate/1.0.0"
        or gate.get("synthetic_regression_passed") is not True
        or gate.get("round_checksum") != digest(round_config)
        or set(gate["implementation"]) != implementation_files()
    ):
        raise ValueError("round_regression_gate_required")
    for name, sha in {**gate["implementation"], **gate["materials"], **gate["evidence"]}.items():
        if checksum(Path(name).read_bytes()) != sha:
            raise ValueError("round_regression_gate_drift")
    if initial is not None and digest(initial) != gate["initial_state_checksum"]:
        raise ValueError("round_prepared_state_changed")
    if getattr(round_config, "stage", None) == "case":
        from apps.vectorengine_case_preflight import check_case_admissibility
        from packages.evidence_details.acceptance_models import DomainState

        saved = validate(
            DomainState, json.loads((Path(round_config.output) / "domain-state.json").read_text())
        )
        if digest(saved) != gate["initial_state_checksum"]:
            raise ValueError("round_prepared_state_changed")
        check_case_admissibility(saved, load_config(round_config.provider_config))
    return gate


def read_deployment(evidence, target, run_id):
    """Metadata/read methods only, through the existing role and connection boundary."""
    from datetime import UTC, datetime

    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository

    contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
    old = contexts.read_research_run(evidence.view["case_evidence"]["fairclip"]["run_id"])
    current = sessions.read(run_id)
    tables = (
        "runtime.run",
        "runtime.planner_control",
        "runtime.idempotency_request",
        "runtime.artifact_metadata",
        "runtime.artifact_derivation",
        "runtime.case_record",
    )
    functions = (
        "runtime.create_research_run(jsonb,text,jsonb)",
        "runtime.save_planner_control(jsonb,jsonb,jsonb)",
        "runtime.append_case_pipeline_event(jsonb)",
    )
    with evidence.runtime._connect() as c:
        deployed = {
            n: bool(c.execute("SELECT to_regclass(%s) AS value", (n,)).fetchone()["value"])
            for n in tables
        }
        callable_owners = {
            n: bool(
                c.execute(
                    "SELECT has_function_privilege(to_regprocedure(%s),'EXECUTE') AS value", (n,)
                ).fetchone()["value"]
            )
            for n in functions
        }
        run_exists = (
            c.execute("SELECT run_id FROM runtime.run WHERE run_id=%s", (run_id,)).fetchone()
            is not None
        )
    if not all(deployed.values()) or not all(callable_owners.values()):
        raise ValueError("round_owner_deployment_missing")
    return dict(
        checked_at=datetime.now(UTC).isoformat(),
        tables=deployed,
        functions=callable_owners,
        source_run_checksum=digest(old),
        target_run_exists=run_exists,
        target_control_phase=current.phase if current else None,
        actual_writes=0,
        current_source_qualification="original_Knowledge_get_and_exact_object_bytes",
        write_transaction_executed=False,
        recheck_required_before_send=True,
    )


def prepare_round(round_config, config, evidence):
    from apps.vectorengine_case_preflight import check_case_admissibility
    from packages.evidence_details.review_tasks import project_review_tasks, review_payload

    state, manifest = evidence.project(run_id=round_config.run_id, config=config)
    manifest["joint_admissibility"] = check_case_admissibility(state, config)
    policy = getattr(config, "review_batching", None)
    if policy:
        from packages.evidence_details.review_batches import batch_payload, make_plan

        plan = make_plan(state, policy, key="preflight-only")
        payloads = [batch_payload(state, plan, b) for b in plan.batches]
        payload = dict(
            schema_version="review-batch-preflight/1.0.0",
            not_executable=True,
            actual_parent_frozen_after_source_vision=True,
            plan=plan.model_dump(mode="json"),
            batch_input_checksums=[digest(p) for p in payloads],
        )
    else:
        payload = review_payload(state.material, project_review_tasks(state), "A")
        payloads = [payload]
    if max(len(canonical(p)) for p in payloads) > next(
        r.max_input_bytes for r in config.roles if r.role == "review"
    ):
        raise ValueError("live_review_input_bound")
    manifest.update(round_checksum=digest(round_config), run_id=round_config.run_id)
    output = Path(round_config.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, value in (
        ("manifest", manifest),
        ("visible-material", state.material),
        ("domain-state", state),
        ("review-input", payload),
        ("round-lock", round_config),
    ):
        write_once(output / (name + ".json"), value)
    return state, manifest


def main_round(path, *, execute=False, check_only=False):
    round_config = load_round(path)
    config = load_config(round_config.provider_config)
    if check_only:
        check_round_gate(round_config)
        print(
            json.dumps(
                dict(status="round_config_verified_no_key_no_network", run_id=round_config.run_id)
            )
        )
        return 0
    if execute:
        # No credentials or run creation before local locks are validated.
        check_round_gate(round_config)
        import os

        from apps.vectorengine_single_case import KEY_ENV

        if not os.environ.get(KEY_ENV):
            print(
                json.dumps(dict(status="missing_designated_key", variable=KEY_ENV, paid_requests=0))
            )
            return 2
    if isinstance(round_config, LiveRoundV3) and round_config.stage == "public":
        from apps.vectorengine_public_diagnostic import main_diagnostic

        return main_diagnostic(round_config, config, execute=execute)
    original, target = open_case()
    evidence = evidence_for_round(original, round_config)
    deployment = read_deployment(evidence, target, round_config.run_id)
    state, manifest = prepare_round(round_config, config, evidence)
    output = Path(round_config.output)
    write_once(output / ("deployment-" + digest(deployment)[7:23] + ".json"), deployment)
    if not execute:
        print(
            json.dumps(
                dict(
                    status="prepared_no_requests",
                    output=str(output),
                    run_id=round_config.run_id,
                    manifest_checksum=digest(manifest),
                    actual_runtime_run_created=False,
                    paid_requests=0,
                )
            )
        )
        return 0
    check_round_gate(round_config, state)
    from apps.vectorengine_live_assembly import execute as execute_original

    return execute_original(
        evidence=evidence,
        target=target,
        initial=state,
        config=config,
        output=output,
        round_config=round_config,
    )

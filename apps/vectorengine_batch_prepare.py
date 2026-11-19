"""Offline batch preflight by default; explicit gate freeze never loads Key or sends."""

import argparse
import json
from pathlib import Path

from apps.vectorengine_round import check_round_gate, implementation_files, load_round
from apps.vectorengine_single_case import load_config, write_once
from packages.grounding.models import digest, validate
from packages.object_store import checksum

ACCEPTANCE = Path("docs/acceptance/12-completion-a-batched-review-v1")


def round_path(stage, round_config=None):
    if stage not in {"public", "case"}:
        raise ValueError("batch_stage")
    old = "configs/research/vectorengine_batch_review_" + stage + ".json"
    new = "configs/research/vectorengine_batch_citations_" + stage + ".json"
    allowed = {old, new, "configs/research/vectorengine_batch_integrated_" + stage + ".json"}
    allowed.add("configs/research/vectorengine_batch_integrated_r01_" + stage + ".json")
    if stage == "case":
        allowed.add("configs/research/vectorengine_batch_integrated_r01_case_runtime_r01.json")
        allowed.add("configs/research/vectorengine_batch_citations_case_target_r01.json")
        allowed.add("configs/research/vectorengine_batch_citations_case_quote_r01.json")
    if round_config is not None and str(round_config) not in allowed:
        raise ValueError("batch_round_selection")
    return str(round_config) if round_config is not None else old


def preview(stage, round_config=None):
    from packages.llm_provider.public_batches import public_plan

    r = load_round(round_path(stage, round_config))
    if r.round_id != Path(round_path(stage, round_config)).stem.replace("_", "-"):
        raise ValueError("batch_round_file_identity")
    config = load_config(r.provider_config)
    public = load_round(r.public_config)
    plan = public_plan(public.run_id, config)
    return dict(
        schema_version="batch-offline-preflight/1.0.0",
        round=r.model_dump(mode="json"),
        public_plan=plan,
        preparation_authorizes_requests=False,
        key_loaded=False,
        database_connected=False,
        real_qualification=False,
        case_parent_frozen_only_after_source_vision=True,
        requests_limit=40,
        public_requests_limit=12,
        planner_steps_limit=20,
        maximum_review_batches=config.review_batching.max_batches,
        maximum_normal_planner_steps=14,
        max_http_seconds=config.transport.timeout_seconds,
        absolute_seconds=r.seconds,
        deadline_reset_on_resume=False,
    )


def check_implementation():
    frozen = json.loads((ACCEPTANCE / "implementation-freeze.json").read_text())
    if set(frozen) != implementation_files() or any(
        checksum(Path(p).read_bytes()) != sha for p, sha in frozen.items()
    ):
        raise ValueError("batch_implementation_drift")
    return frozen


def check_acceptance(round_config=None):
    if getattr(round_config, "schema_version", None) == "bounded-live-round/8.0.0":
        from apps.vectorengine_performance_gate import check_acceptance as performance_acceptance

        return performance_acceptance()
    if getattr(round_config, "schema_version", None) == "bounded-live-round/7.0.0":
        from apps.vectorengine_source_selection_gate import check_acceptance as source_acceptance

        return source_acceptance()
    if getattr(round_config, "schema_version", None) == "bounded-live-round/6.0.0":
        from apps.vectorengine_integrated_gate import check_acceptance as integrated_acceptance

        return integrated_acceptance()
    if getattr(round_config, "schema_version", None) == "bounded-live-round/5.0.0":
        from apps.vectorengine_case_quote_gate import check_acceptance as quote_acceptance

        return quote_acceptance()
    if getattr(round_config, "schema_version", None) == "bounded-live-round/4.0.0":
        from apps.vectorengine_case_target_gate import check_acceptance as target_acceptance

        return target_acceptance()
    if round_config is not None and round_config.session_id == "batch-citations":
        from apps.vectorengine_citation_gate import check_acceptance as citation_acceptance

        return citation_acceptance()
    implementation = check_implementation()
    return implementation, baseline_evidence(implementation)


def baseline_evidence(implementation):
    proof = json.loads((ACCEPTANCE / "verification.json").read_text())
    if (
        not proof.get("cpu_passed")
        or not proof.get("schemas_passed")
        or not proof.get("static_passed")
    ):
        raise ValueError("batch_offline_verification_required")
    gate = json.loads((ACCEPTANCE / "pg-gate.json").read_text())
    if gate["implementation_checksum"] != digest(implementation):
        raise ValueError("batch_pg_implementation_binding")
    evidence = {}
    for name in ("cpu-final", "schemas", "static-final", "pg"):
        directory = ACCEPTANCE / "synthetic" / name
        exit_value = json.loads((directory / "exit.json").read_text())
        if exit_value["exit_code"] != 0:
            raise ValueError("batch_gate_failed:" + name)
        if name == "pg" and (
            not exit_value.get("owned_cleanup_verified")
            or exit_value.get("implementation_checksum") != digest(implementation)
            or exit_value.get("passed") != 1
        ):
            raise ValueError("batch_pg_required")
        for file in ("exit.json", "command.json", "output.log"):
            p = directory / file
            evidence[str(p)] = checksum(p.read_bytes())
    return evidence


def freeze_gate(stage, round_config=None):
    r = load_round(round_path(stage, round_config))
    if r.round_id != Path(round_path(stage, round_config)).stem.replace("_", "-"):
        raise ValueError("batch_round_file_identity")
    implementation, evidence = check_acceptance(r)
    output = Path(r.output)
    required = (
        ("public-state", "public-batch-plan")
        if stage == "public"
        else ("domain-state", "manifest", "visible-material", "review-input", "round-lock")
    )
    materials = {
        str(output / (name + ".json")): checksum((output / (name + ".json")).read_bytes())
        for name in required
    }
    initial = None
    if stage == "public":
        if (
            json.loads((output / "public-batch-plan.json").read_text())
            != preview(stage, round_config)["public_plan"]
        ):
            raise ValueError("public_batch_preparation_changed")
    else:
        from apps.vectorengine_case_preflight import check_case_admissibility
        from packages.evidence_details.acceptance_models import DomainState

        initial_state = validate(
            DomainState, json.loads((output / "domain-state.json").read_text())
        )
        check_case_admissibility(initial_state, load_config(r.provider_config))
        public = load_round(r.public_config)
        if r.schema_version == "bounded-live-round/8.0.0":
            from apps.vectorengine_performance_gate import check_public_baseline

            check_public_baseline(public)
        elif r.schema_version == "bounded-live-round/5.0.0":
            from apps.vectorengine_case_quote_gate import check_public_baseline

            check_public_baseline(public)
        elif r.schema_version == "bounded-live-round/4.0.0":
            from apps.vectorengine_case_target_gate import check_public_baseline

            check_public_baseline(public)
        else:
            check_round_gate(public)
        # The actual public receipts are revalidated by consume_qualification before run creation.
        q = Path(public.output) / "qualification.json"
        value = json.loads(q.read_text())
        if (
            value.get("schema_version") != "bounded-public-qualification/3.0.0"
            or value.get("execution") != "live_http"
        ):
            raise ValueError("real_public_batch_qualification_required")
        evidence[str(q)] = checksum(q.read_bytes())
        initial = digest(initial_state)
    # Lock the exact historical public role evidence that is intentionally reused.
    for name in ("public-request-audit", "public-results"):
        p = Path("docs/acceptance/12-live-schema-compat-v1/actual") / (name + ".json")
        evidence[str(p)] = checksum(p.read_bytes())
    value = dict(
        schema_version="bounded-live-round-gate/1.0.0",
        round_checksum=digest(r),
        synthetic_regression_passed=True,
        implementation=implementation,
        materials=materials,
        evidence=evidence,
        initial_state_checksum=initial,
        clinical_acceptance=False,
        preparation_authorizes_requests=False,
    )
    path = Path(r.gate)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, value)
    check_round_gate(r)
    return dict(status="batch_gate_verified_no_requests", stage=stage, gate=str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("public", "case"), default="public")
    p.add_argument("--freeze-gate", action="store_true")
    p.add_argument("--round-config", help="explicit installed legacy or citation-repair round")
    args = p.parse_args(argv)
    if args.freeze_gate:
        result = freeze_gate(args.stage, args.round_config)
    else:
        result = preview(args.stage, args.round_config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

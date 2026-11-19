"""Bounded opt-in VectorEngine entry point; default prepares visible inputs only."""

import argparse
import json
import os
from pathlib import Path

from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
from apps.vectorengine_case_materials import FirstCaseEvidence
from packages.evidence_details.live_permission import validate_live_config
from packages.grounding.models import canonical, digest
from packages.llm_provider.config import read_provider_config

CONFIG = Path("configs/agents/vectorengine_single_case.json")
OUTPUT = Path(".local/vectorengine-single-case-v1/prepared-02")
KEY_ENV = "MMAGENT_VECTORENGINE_API_KEY"


def load_config(path=CONFIG):
    # This independent config never inherits MMAGENT_LLM_* overrides or another key name.
    return validate_live_config(read_provider_config(json.loads(Path(path).read_text())))


def open_case():
    from apps.knowledge_source_tools import repository
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from db.runtime.repository.dataset_input import DatasetInputRepository
    from packages.dataset_adapters.service import VisibleInputReader
    from packages.object_store import LocalObjectStore

    for purpose in ("visible", "knowledge"):
        check_object_root(OBJECT_ROOTS[purpose], purpose)
    boundary = credential_boundary()
    target = boundary.resolve_target("runtime")
    runtime = CaseEvidenceRepository(target)
    objects = LocalObjectStore(OBJECT_ROOTS["visible"])
    evidence = FirstCaseEvidence(
        runtime=runtime,
        visible=VisibleInputReader(DatasetInputRepository(runtime), objects),
        objects=objects,
        knowledge=repository("editor"),
        knowledge_objects=LocalObjectStore(OBJECT_ROOTS["knowledge"]),
    )
    return evidence, target


def write_once(path, value):
    data = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("live_output_conflict_no_overwrite")
    else:
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o600)


def prepare(evidence, config, output=OUTPUT):
    from apps.vectorengine_case_preflight import check_case_admissibility
    from db.runtime.repository.case_pipeline import identity

    # One predetermined run for this invocation. Resume cannot silently create another run.
    run_id = identity("RUN", "vectorengine-single-case-v1:" + digest(config))
    state, manifest = evidence.project(run_id=run_id, config=config)
    manifest["joint_admissibility"] = check_case_admissibility(state, config)
    from packages.evidence_details.review_tasks import project_review_tasks, review_payload

    tasks = project_review_tasks(state)
    payload = review_payload(state.material, tasks, "A")
    if len(canonical(payload)) > next(
        r.max_input_bytes for r in config.roles if r.role == "review"
    ):
        raise ValueError("live_review_input_bound")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(output / "manifest.json", manifest)
    write_once(output / "visible-material.json", state.material)
    # Domain policy contains the original fixed model row, never sent to the model.
    write_once(output / "domain-state.json", state)
    write_once(output / "review-input.json", payload)
    return state, manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--round-config", type=Path)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    if args.round_config:
        if args.check_config and args.execute:
            parser.error("--check-config cannot execute")
        from apps.vectorengine_round import main_round

        return main_round(args.round_config, execute=args.execute, check_only=args.check_config)
    if args.check_config:
        parser.error("--check-config requires --round-config")
    config = load_config()
    if args.execute and not os.environ.get(KEY_ENV):
        print(json.dumps(dict(status="missing_designated_key", variable=KEY_ENV, paid_requests=0)))
        return 2
    evidence, target = open_case()
    state, manifest = prepare(evidence, config)
    if not args.execute:
        print(
            json.dumps(
                dict(
                    status="prepared_no_requests",
                    output=str(OUTPUT),
                    manifest_checksum=digest(manifest),
                    key_present=bool(os.environ.get(KEY_ENV)),
                    actual_runtime_run_created=False,
                    paid_requests=0,
                )
            )
        )
        return 0
    if not os.environ.get(KEY_ENV):
        print(json.dumps(dict(status="missing_designated_key", variable=KEY_ENV, paid_requests=0)))
        return 2
    from apps.vectorengine_live_assembly import execute

    return execute(evidence=evidence, target=target, initial=state, config=config, output=OUTPUT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Exceptions may contain remote bodies, source text or connection strings.
        print(
            json.dumps(dict(status="preparation_or_run_rejected", error_type=type(error).__name__))
        )
        raise SystemExit(2) from None

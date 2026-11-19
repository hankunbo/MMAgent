"""Prepare/freeze the authorized pilot, without opening the private API configuration."""

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from apps.active_evidence_materials import MANIFEST, RULES, load_manifest
from apps.active_evidence_sources import DESTINATION
from apps.vectorengine_round import (
    ActiveAcquisitionAliasRound,
    ActiveAcquisitionRound,
    ActiveAcquisitionStatusRound,
    evidence_for_round,
    implementation_files,
    load_round,
    prepare_round,
)
from apps.vectorengine_single_case import load_config, open_case, write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest, validate
from packages.object_store import checksum

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("docs/acceptance/12-live-active-acquisition-v1-v1")
SESSION = "active-acquisition-v1"
STEM = "vectorengine-active-acquisition-v1"
CONFIG = Path("configs/agents/vectorengine_active_acquisition_v1.json")
PYTHON = "/opt/data/test/envs/MMAgent1/bin/python"
COMMANDS = {
    "targeted": [
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "tests/evidence_acquisition",
        "tests/expert_agent/test_batch_assembly.py",
        "tests/agent_provider/test_source_selection.py",
        "tests/agent_provider/test_review_batches.py",
        "tests/agent_provider/test_batch_public.py",
        "tests/agent_provider/test_current_source_batch.py",
        "tests/agent_provider/test_review_citation_contract.py",
        "tests/agent_provider/test_integrated_review.py",
        "tests/agent_provider/test_source_quote_integration.py",
    ],
    "schema": [PYTHON, "-m", "packages.evidence_acquisition.schema_tools"],
    "provider-schema": [PYTHON, "-m", "packages.llm_provider.schema_tools"],
    "static": ["make", "env-check", "lock-check", "format-check", "lint", "schema-check"],
}
LIVE_COMMANDS = {
    stage: [
        PYTHON,
        ".local/vectorengine/run.py",
        "--execute",
        "--round-config",
        "configs/research/vectorengine_active_acquisition_v1_" + stage + ".json",
    ]
    for stage in ("public", "case")
}
LIVE_COMMANDS["readback"] = [
    PYTHON,
    "-m",
    "apps.vectorengine_integrated_readback",
    "--round-config",
    "configs/research/vectorengine_active_acquisition_v1_case.json",
    "--output",
    str(EVIDENCE / "readback"),
]


def hashes():
    return {p: checksum(Path(p).read_bytes()) for p in sorted(implementation_files())}


def prepare():
    members = []
    for key, indices in (("phene", (0, 5)), ("rim", (0, 2))):
        path = DESTINATION / (key + ".json")
        entry = json.loads(path.read_text())
        members.append(
            dict(
                id=key,
                archive=str(path),
                archive_checksum=digest(entry),
                excerpts=[
                    dict(
                        reference=entry["chunks"][i]["reference"],
                        span=[0, len(entry["chunks"][i]["text"])],
                        text=entry["chunks"][i]["text"],
                    )
                    for i in indices
                ],
            )
        )
    raw = dict(
        schema_version="active-acquisition-manifest/1.0.0",
        members=members,
        rules=RULES,
        reviewed_by="assistant_source_binding_not_clinical_validation",
        policy="three_pilot_tasks_one_targeted_image_one_full_review_then_stop",
        provenance="installed_Knowledge_exact_citation_fetch; no_new_provider",
        limits=dict(
            citation_documents=2,
            pilot_iterations=1,
            model_requests=40,
            patient_images=1,
            tool_executions=0,
        ),
    )
    write_once(MANIFEST, dict(**raw, checksum=digest(raw)))
    provider = json.loads(Path("configs/agents/vectorengine_batch_integrated_r01.json").read_text())
    if SESSION in {"active-acquisition-r01", "active-acquisition-r02"}:
        prompt_version = "review-batch-1.3.0" if SESSION.endswith("r01") else "review-batch-1.4.0"
        for role in provider["roles"]:
            if role["role"] == "review":
                role["prompt_version"] = prompt_version
                role["prompt_checksum"] = digest(
                    Path("configs/prompts/agents/" + prompt_version + ".txt").read_text()
                )
    write_once(CONFIG, provider)
    base = json.loads(
        Path("configs/research/vectorengine_batch_integrated_r01_public.json").read_text()
    )
    base.pop("prior_round_configs")
    configs = []
    for stage in ("public", "case"):
        rid = STEM + "-" + stage
        raw = dict(
            base,
            schema_version="bounded-live-round/9.0.0",
            session_id=SESSION,
            stage=stage,
            round_id=rid,
            run_id=identity("RUN", rid),
            request_key=rid,
            request_code="RESEARCH:" + rid.upper().replace("-", "_"),
            provider_config=str(CONFIG),
            provider_checksum=digest(load_config(CONFIG)),
            public_config="configs/research/" + STEM.replace("-", "_") + "_public.json",
            output=".local/" + rid + "/prepared-01",
            seconds=1200 if stage == "public" else 2700,
            gate=str(EVIDENCE / (stage + "-execution-gate.json")),
            acquisition_manifest=str(MANIFEST),
            acquisition_checksum=digest(load_manifest()),
        )
        cls = ActiveAcquisitionRound
        if SESSION == "active-acquisition-r01":
            cls = ActiveAcquisitionStatusRound
            raw.update(
                schema_version="bounded-live-round/10.0.0",
                prior_round_configs=[
                    "configs/research/vectorengine_active_acquisition_v1_public.json",
                    "configs/research/vectorengine_active_acquisition_v1_case.json",
                ],
            )
        if SESSION == "active-acquisition-r02":
            cls = ActiveAcquisitionAliasRound
            raw.update(
                schema_version="bounded-live-round/11.0.0",
                prior_round_configs=[
                    f"configs/research/vectorengine_active_acquisition_{profile}_{part}.json"
                    for profile in ("v1", "r01")
                    for part in ("public", "case")
                ],
            )
        round_config = validate(cls, raw)
        path = Path("configs/research/" + STEM.replace("-", "_") + "_" + stage + ".json")
        write_once(path, round_config)
        configs.append(round_config)
    original, _ = open_case()
    for round_config in configs:
        evidence = evidence_for_round(original, round_config)
        state, _ = prepare_round(round_config, load_config(CONFIG), evidence)
        evidence.verify(state)
        print(
            json.dumps(
                dict(
                    stage=round_config.stage,
                    prepared=True,
                    requests=0,
                    tasks=len(state.policy.items),
                    sources=len(state.material.sources),
                )
            )
        )


def capture(step, output):
    output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    command = COMMANDS[step] if step in COMMANDS else LIVE_COMMANDS[step]
    started, start = datetime.now(UTC).isoformat(), time.monotonic()
    with (output / "output.log").open("wb") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    value = dict(
        command=command,
        started_at=started,
        ended_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.monotonic() - start,
        exit_code=result.returncode,
        implementation_before=digest(before),
        implementation_after=digest(hashes()),
        log_checksum=checksum((output / "output.log").read_bytes()),
    )
    write_once(output / "result.json", value)
    print(json.dumps(value))
    return result.returncode


def freeze(suffix="-final"):
    implementation = hashes()
    evidence = {}
    for step in COMMANDS:
        path = EVIDENCE / (step + suffix) / "result.json"
        result = json.loads(path.read_text())
        log = path.parent / "output.log"
        if (
            result["exit_code"] != 0
            or result["implementation_before"] != digest(implementation)
            or result["implementation_after"] != digest(implementation)
            or result["log_checksum"] != checksum(log.read_bytes())
        ):
            raise ValueError("active_final_verification_required:" + step)
        evidence[str(path)], evidence[str(log)] = (
            checksum(path.read_bytes()),
            checksum(log.read_bytes()),
        )
    prior = Path("docs/acceptance/12-acquisition-preparation-v1/verification.json")
    evidence[str(prior)] = checksum(prior.read_bytes())
    materials = {str(p): checksum(p.read_bytes()) for p in DESTINATION.glob("*.json")}
    for stage in ("public", "case"):
        round_config = load_round(
            "configs/research/" + STEM.replace("-", "_") + "_" + stage + ".json"
        )
        state = json.loads((Path(round_config.output) / "domain-state.json").read_text())
        write_once(
            Path(round_config.gate),
            dict(
                schema_version="bounded-live-round-gate/1.0.0",
                synthetic_regression_passed=True,
                round_checksum=digest(round_config),
                implementation=implementation,
                materials=materials,
                evidence=evidence,
                initial_state_checksum=digest(state),
                scope=(
                    "existing persistence reused; targeted active assembly tested; "
                    "not full-suite or clinical validation"
                ),
            ),
        )
    write_once(EVIDENCE / "implementation-freeze.json", implementation)
    print(json.dumps(dict(frozen_files=len(implementation), gates_ready=True, api_requests=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "freeze", *COMMANDS, *LIVE_COMMANDS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verification-suffix", default="-final")
    parser.add_argument(
        "--profile",
        choices=("active-acquisition-v1", "active-acquisition-r01", "active-acquisition-r02"),
        default="active-acquisition-v1",
    )
    args = parser.parse_args()
    select_profile(args.profile)
    if args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze(args.verification_suffix)
    else:
        suffix = "-final" if args.action in COMMANDS else "-execute"
        raise SystemExit(capture(args.action, args.output or EVIDENCE / (args.action + suffix)))


def select_profile(session):
    """Choose one explicit identity; never manufacture arbitrary fresh budgets."""
    global SESSION, STEM, EVIDENCE, CONFIG, LIVE_COMMANDS
    if session not in {"active-acquisition-v1", "active-acquisition-r01", "active-acquisition-r02"}:
        raise ValueError("active_profile")
    SESSION = session
    STEM = "vectorengine-" + session
    EVIDENCE = Path("docs/acceptance/12-live-" + session + "-v1")
    CONFIG = Path("configs/agents/" + STEM.replace("-", "_") + ".json")
    LIVE_COMMANDS = {
        stage: [
            PYTHON,
            ".local/vectorengine/run.py",
            "--execute",
            "--round-config",
            "configs/research/" + STEM.replace("-", "_") + "_" + stage + ".json",
        ]
        for stage in ("public", "case")
    }
    LIVE_COMMANDS["readback"] = [
        PYTHON,
        "-m",
        "apps.vectorengine_integrated_readback",
        "--round-config",
        "configs/research/" + STEM.replace("-", "_") + "_case.json",
        "--output",
        str(EVIDENCE / "readback"),
    ]


if __name__ == "__main__":
    main()

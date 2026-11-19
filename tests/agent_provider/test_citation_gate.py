"""Offline round selection and repair evidence binding; no actual gate is created."""

import json
from pathlib import Path

import pytest

from apps import vectorengine_citation_gate as gate
from apps.vectorengine_batch_prepare import baseline_evidence, freeze_gate, preview, round_path
from apps.vectorengine_round import check_round_gate, implementation_files, load_round
from packages.grounding.models import digest
from packages.object_store import checksum

PUBLIC = "configs/research/vectorengine_batch_citations_public.json"
CASE = "configs/research/vectorengine_batch_citations_case.json"


def test_new_pair_offline_selection_old_identity_and_budget_remain(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline_no_connection")

    import socket

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("apps.vectorengine_single_case.open_case", forbidden)
    monkeypatch.setattr("apps.vectorengine_round.open_case", forbidden)
    a, b = preview("public", PUBLIC), preview("case", CASE)
    old = preview("public")
    assert a == preview("public", PUBLIC)
    assert a["round"]["session_id"] == b["round"]["session_id"] == "batch-citations"
    assert b["round"]["public_config"] == PUBLIC
    assert len({x["round"]["run_id"] for x in (a, b, old)}) == 3
    assert a["maximum_review_batches"] == 15 and a["max_http_seconds"] == 240
    assert a["requests_limit"] == 40 and a["absolute_seconds"] == 1200
    assert b["absolute_seconds"] == 2700
    assert not a["key_loaded"] and not a["database_connected"]
    assert [s["task_count"] for s in a["public_plan"]["steps"]] == [1, 3]
    assert all(s["reused_not_sent"] for s in a["public_plan"]["source_vision_reuse"])
    assert not Path(a["round"]["output"]).exists()
    assert not Path(a["round"]["gate"]).exists()


@pytest.mark.parametrize("path", [CASE, "../../private.json", ".local/vectorengine/config.json"])
def test_round_selection_rejects_uninstalled_or_wrong_stage_without_read(path):
    with pytest.raises(ValueError, match="batch_round_selection"):
        round_path("public", path)


def test_old_gate_cannot_bind_new_pair():
    new = load_round(PUBLIC)
    old = load_round("configs/research/vectorengine_batch_review_public.json")
    with pytest.raises(ValueError, match="round_regression_gate_required"):
        check_round_gate(new.model_copy(update={"gate": old.gate}))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    baseline = json.loads((gate.BASELINE / "implementation-freeze.json").read_text())
    current = {p: checksum(Path(p).read_bytes()) for p in sorted(implementation_files())}
    delta = gate.difference(baseline, current)
    inherited = baseline_evidence(baseline)
    proof = dict(
        schema_version="review-citation-repair-verification/1.0.0",
        offline_passed=True,
        persistent_semantics_changed=False,
        baseline_checksum=digest(baseline),
        implementation_checksum=digest(current),
        delta_checksum=digest(delta),
        covered_changes=sorted(delta),
        inherited_evidence=inherited,
    )
    checks = {}
    python = "/opt/data/test/envs/MMAgent1/bin/python"
    commands = {
        "cpu-final": [
            python,
            "-m",
            "pytest",
            "tests/agent_provider/test_review_citation_contract.py",
            "tests/agent_provider/test_citation_gate.py",
            "-q",
        ],
        "schemas": [python, "-m", "packages.llm_provider.schema_tools"],
        "static-final": ["make", "env-check", "lock-check", "format-check", "lint", "schema-check"],
    }

    def save(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    for name, command in commands.items():
        for file, value in {
            "command.json": command,
            "exit.json": dict(
                exit_code=0,
                passed=1,
                owned_cleanup_verified=True,
                implementation_checksum=digest(current),
            ),
            "output.log": "Synthetic evidence-validator fixture only; not a real test result.",
        }.items():
            p = save("synthetic/" + name + "/" + file, value)
            checks[str(p)] = checksum(p.read_bytes())
    proof["validation_evidence"] = checks
    for name, value in (
        ("implementation-freeze.json", current),
        ("delta.json", delta),
        ("verification.json", proof),
    ):
        save(name, value)
    monkeypatch.setattr(gate, "PACKAGE", tmp_path)
    return baseline, current, delta, proof, save


def test_repair_proof_validates_exact_current_files_and_inherited_PG(evidence):
    _, current, _, _, _ = evidence
    actual, refs = gate.check_acceptance()
    assert actual == current
    assert any("synthetic/pg/exit.json" in p for p in refs)


@pytest.mark.parametrize("mutation", ["uncovered", "omitted", "stale", "failed", "log", "command"])
def test_uncovered_or_unverified_delta_cannot_freeze_gate(evidence, mutation):
    _, current, delta, proof, save = evidence
    if mutation == "uncovered":
        proof["covered_changes"] = []
    elif mutation == "omitted":
        current.pop("packages/llm_provider/transport.py")
        save("implementation-freeze.json", current)
    elif mutation == "stale":
        delta["packages/llm_provider/transport.py"]["after"] = digest("other")
        save("delta.json", delta)
    elif mutation == "failed":
        save("synthetic/cpu-final/exit.json", dict(exit_code=1))
    elif mutation == "log":
        save("synthetic/cpu-final/output.log", "changed")
    else:
        save("synthetic/cpu-final/command.json", ["true"])
    save("verification.json", proof)
    with pytest.raises(ValueError):
        freeze_gate("public", PUBLIC)
    assert not Path(load_round(PUBLIC).gate).exists()

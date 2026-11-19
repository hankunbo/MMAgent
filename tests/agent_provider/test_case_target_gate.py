"""Offline proof/identity/launcher boundaries; no key, HTTP or deployed PG."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.vectorengine_round import CaseTargetRepairRound, implementation_files, load_round
from packages.grounding.models import digest, validate
from packages.object_store import checksum

CONFIG = "configs/research/vectorengine_batch_citations_case_target_r01.json"


def test_new_case_identity_preserves_public_session_and_limits():
    from apps.vectorengine_batch_prepare import preview

    new = load_round(CONFIG)
    old = load_round(new.prior_case_config)
    public = load_round(new.public_config)
    assert new.run_id != old.run_id != public.run_id
    assert new.session_id == old.session_id == public.session_id
    assert new.provider_checksum == old.provider_checksum == public.provider_checksum
    assert new.seconds == old.seconds == 2700 and new.max_requests == 40
    assert preview("case", CONFIG)["round"]["run_id"] == new.run_id
    with pytest.raises(ValueError, match="batch_round_selection"):
        preview("public", CONFIG)
    for field, value in (
        ("stage", "public"),
        ("run_id", old.run_id),
        ("session_id", "fresh"),
        ("seconds", 2701),
        ("public_config", "OTHER"),
        ("prior_case_config", "OTHER"),
    ):
        raw = new.model_dump(mode="json")
        raw[field] = value
        with pytest.raises(ValueError):
            validate(CaseTargetRepairRound, raw)


def delta_fixture():
    from apps.vectorengine_case_target_gate import BASELINE

    before = json.loads((BASELINE / "implementation-before.json").read_text())
    current = {p: checksum(Path(p).read_bytes()) for p in sorted(implementation_files())}
    delta = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(set(before) | set(current))
        if before.get(p) != current.get(p)
    }
    proof = dict(
        schema_version="case-target-repair-verification/1.0.0",
        offline_passed=True,
        persistent_semantics_changed=False,
        baseline_checksum=digest(before),
        implementation_checksum=digest(current),
        delta_checksum=digest(delta),
    )
    return before, current, delta, proof


def test_exact_delta_rejects_uncovered_changes_omission_or_failed_proof():
    from apps.vectorengine_case_target_gate import verify_delta

    before, current, delta, proof = delta_fixture()
    verify_delta(before, current, delta, proof)
    with pytest.raises(ValueError, match="uncovered_delta"):
        verify_delta(before, current, {}, proof)
    with pytest.raises(ValueError, match="uncovered_delta"):
        verify_delta(before, current, delta, dict(proof, offline_passed=False))
    foreign = dict(current, **{"apps/foreign.py": digest("uncovered")})
    with pytest.raises(ValueError, match="uncovered_delta"):
        verify_delta(before, foreign, delta, proof)


@pytest.mark.parametrize("mode", ["valid", "running", "not_cancelled", "pending", "request"])
def test_prior_case_requires_current_zero_request_cancelled_owner(mode):
    from apps.vectorengine_case_target_gate import check_prior_case
    from tests.agent_provider.support import Documents

    d = Documents()
    d.run_id = "RUN:PRIOR"
    state = SimpleNamespace(phase="stopped", cancelled=True, pending=None)
    if mode == "running":
        state.phase = "initialize"
    if mode == "not_cancelled":
        state.cancelled = False
    if mode == "pending":
        state.pending = object()
    if mode == "request":
        d.write("journal:0", ({"phase": "reserved"},))
    d.sessions = SimpleNamespace(read=lambda _: state)
    if mode == "valid":
        check_prior_case(d)
    else:
        with pytest.raises(ValueError, match="case_target_prior"):
            check_prior_case(d)


def saved_cancelled_case():
    from packages.planner_control.models import Control
    from tests.agent_provider.support import Documents

    saved = Path("docs/acceptance/12-completion-cde-citations-case-v1")
    control = validate(Control, json.loads((saved / "original-case-control.json").read_text()))
    events = json.loads((saved / "original-case-journal.json").read_text())
    documents = Documents()
    documents.run_id = control.binding.run_id
    documents.sessions = SimpleNamespace(read=lambda _: control)
    for i, event in enumerate(events):
        documents.write("journal:" + str(i), (event,))
    return documents, events


def test_saved_cancelled_metadata_is_not_a_request_and_remains_unchanged():
    from apps.vectorengine_case_target_gate import check_prior_case
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    documents, events = saved_cancelled_case()
    check_prior_case(documents)
    assert list(RuntimeAgentJournal(documents).read()) == events


@pytest.mark.parametrize(
    "extra",
    [
        {"phase": "reserved"},
        {"phase": "completed"},
        {"phase": "response_confirmation"},
        {"phase": "http_observation"},
        {"schema_version": "future-journal/1.0.0"},
        {"phase": "live_cancelled", "request_id": "unconfirmed"},
    ],
)
def test_zero_request_bridge_rejects_attempts_unknown_and_disguised_events(extra):
    from apps.vectorengine_case_target_gate import check_prior_case

    documents, events = saved_cancelled_case()
    documents.write("journal:" + str(len(events)), (extra,))
    with pytest.raises(ValueError, match="case_target_prior"):
        check_prior_case(documents)


@pytest.mark.parametrize("part", ["run", "deadline", "allocation", "limits"])
def test_zero_request_bridge_requires_exact_metadata_binding(part):
    from apps.vectorengine_case_target_gate import check_prior_case
    from tests.agent_provider.support import Documents

    original, events = saved_cancelled_case()
    if part == "run":
        events[0]["binding"]["run_id"] = "RUN:OTHER"
    elif part == "deadline":
        events[0]["binding"]["deadline"] = "2026-10-01T23:59:59+00:00"
    elif part == "allocation":
        events[2]["public_used"] = 1
    else:
        events[1]["max_requests"] = 500
    documents = Documents()
    documents.run_id, documents.sessions = original.run_id, original.sessions
    for i, event in enumerate(events):
        documents.write("journal:" + str(i), (event,))
    with pytest.raises(ValueError, match="case_target_prior"):
        check_prior_case(documents)


def test_private_launcher_checks_round_before_loading_any_key(monkeypatch):
    spec = importlib.util.spec_from_file_location("target_launcher", ".local/vectorengine/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["run.py", "--execute", "--round-config", CONFIG])

    def reject(*args):
        raise ValueError("synthetic_gate_missing")

    monkeypatch.setattr("apps.vectorengine_round.check_round_gate", reject)
    monkeypatch.setattr(module, "load_secret", lambda: pytest.fail("key must not be read"))
    monkeypatch.setattr(module.os, "execve", lambda *a: pytest.fail("must not launch"))
    assert module.main() == 2


def test_public_saved_wire_unchanged_after_case_only_repair():
    from apps.vectorengine_single_case import load_config
    from packages.evidence_details.review_batches import ReviewBatchPlan
    from packages.llm_provider.public_batches import batch_wire, source_vision_reuse
    from packages.llm_provider.public_shape import PublicReviewMaterials

    package = Path("docs/acceptance/12-completion-b-citations-public-v1")
    config = load_config("configs/agents/vectorengine_batch_citations.json")
    docs = json.loads((package / "original-public-documents.json").read_text())
    for step in docs["public-batch-plan"][0]["steps"]:
        plan = validate(ReviewBatchPlan, step["plan"])
        state = PublicReviewMaterials(
            plan.scope.run_id, config, task_count=step["task_count"]
        ).current()
        assert batch_wire(config, state, plan, plan.batches[0]) == json.loads(
            (package / f"wire-review-{step['task_count']}.json").read_text()
        )
    assert source_vision_reuse(config)


def test_case_gate_checks_joint_admissibility_before_secret_or_run(monkeypatch, tmp_path):
    from apps.vectorengine_round import check_round_gate
    from apps.vectorengine_single_case import load_config
    from tests.agent_provider.test_case_graph_target import with_graph

    r = load_round(CONFIG).model_copy(
        update={"gate": str(tmp_path / "gate.json"), "output": str(tmp_path)}
    )
    state, _ = with_graph().project(run_id=r.run_id, config=load_config(r.provider_config))
    state = state.model_copy(update={"nodes": ("EVIDENCE:SYNTHETIC",)})
    (tmp_path / "domain-state.json").write_text(state.model_dump_json())
    impl = {p: checksum(Path(p).read_bytes()) for p in sorted(implementation_files())}
    gate = dict(
        schema_version="bounded-live-round-gate/1.0.0",
        synthetic_regression_passed=True,
        round_checksum=digest(r),
        implementation=impl,
        materials={},
        evidence={},
        initial_state_checksum=digest(state),
    )
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="research_relation_scope"):
        check_round_gate(r)

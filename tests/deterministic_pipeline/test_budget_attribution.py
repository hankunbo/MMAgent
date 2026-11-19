"""Promoted independent regressions and synthetic-only budget/recovery coverage."""

from datetime import UTC, datetime, timedelta

import pytest

from packages.contracts import ValidationResult
from packages.rule_engine import bind_validation
from tests.deterministic_pipeline.test_pipeline import invoke


def test_no_new_execution_after_budget_expires_during_compile(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    import packages.deterministic_pipeline.service as module

    class Clock(datetime):
        expired = False

        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + (timedelta(hours=2) if cls.expired else timedelta())

    monkeypatch.setattr(module, "datetime", Clock)
    compile_original = p.tools.compile
    router_original = p.tools.executor_for
    executions = []

    def compile_then_expire(*args):
        plan = compile_original(*args)
        Clock.expired = True
        return plan

    def track_executor(plan):
        executor = router_original(plan)
        execute = executor.execute

        def record(*args):
            executions.append(Clock.now(UTC).isoformat())
            return execute(*args)

        monkeypatch.setattr(executor, "execute", record)
        return executor

    monkeypatch.setattr(p.tools, "compile", compile_then_expire)
    monkeypatch.setattr(p.tools, "executor_for", track_executor)
    report = invoke(p, "expired-before-launch")
    assert report["stop_reason"] == "time_budget"
    assert executions == [], "NEW_EXECUTION_STARTED_AFTER_DEADLINE"
    assert report["unstarted_actions"][0]["reason"] == "case_deadline_reached_before_execute"
    assert not processes(p)


@pytest.mark.parametrize("assembled_pipeline", ["shared"], indirect=True)
def test_report_effect_is_bound_to_target_not_only_shared_asset(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    # Test report attribution using two typed, graph-bound synthetic validation outcomes.
    # This deliberately substitutes a deterministic test interpreter, not a clinical rule.
    evaluate = p.service.validator.evaluate
    second = p.data["pack"].definition.evidence_slots[1].node_code

    def opposite_second(graph, node_code, artifact, rule, step):
        bound = evaluate(graph, node_code, artifact, rule, step)
        if node_code != second:
            return bound
        raw = bound.item.model_dump(mode="json")
        raw["observation"] = "NEGATIVE"
        raw["clinical_effect"] = "REFUTES"
        raw["graph_delta_candidate"]["operations"][0]["to_status"] = "REFUTED"
        return bind_validation(graph, ValidationResult.model_validate(raw))

    monkeypatch.setattr(p.service.validator, "evaluate", opposite_second)
    report = invoke(p, "target-attribution")
    nodes = {n["node_code"]: n["status"] for n in report["graph"]["nodes"]}
    observations = {o["target"]: o["clinical_effect"] for o in report["observations"]}
    first = p.data["pack"].definition.evidence_slots[0].node_code
    assert nodes[first] == "SUPPORTED" and nodes[second] == "REFUTED"
    assert observations[first] == "SUPPORTS", (nodes, observations)
    assert observations[second] == "REFUTES"
    assert_effects_agree(p, report, {first: "SUPPORTS", second: "REFUTES"})


def freeze_case_clock(monkeypatch, deadline):
    import packages.deterministic_pipeline.service as module

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return deadline

    monkeypatch.setattr(module, "datetime", Clock)


def request_state(p, key):
    return p.runtime.read_request(p.runtime.key(p.case.case_code, key))


def processes(p):
    return list((p.objects.root.parent / "attempts").glob("*/*/process.json"))


def test_expired_saved_plan_never_starts_and_retry_keeps_identity(assembled_pipeline, monkeypatch):
    p, key = assembled_pipeline, "saved-unstarted"
    checkpoint = p.runtime.checkpoint

    def lose_plan_receipt(*args, **kwargs):
        saved = checkpoint(*args, **kwargs)
        if kwargs.get("actions"):
            raise OSError("saved plan receipt lost")
        return saved

    monkeypatch.setattr(p.runtime, "checkpoint", lose_plan_receipt)
    with pytest.raises(OSError, match="saved plan"):
        invoke(p, key)
    before = request_state(p, key)
    assert before.actions and not processes(p)
    monkeypatch.setattr(p.runtime, "checkpoint", checkpoint)
    freeze_case_clock(
        monkeypatch, before.started_at + timedelta(seconds=p.config.policy.max_seconds)
    )
    router = p.tools.executor_for

    def no_new_start(plan):
        executor = router(plan)
        monkeypatch.setattr(
            executor, "execute", lambda *a: pytest.fail("expired saved plan started")
        )
        return executor

    monkeypatch.setattr(p.tools, "executor_for", no_new_start)
    report = invoke(p, key)
    assert report["stop_reason"] == "time_budget"
    assert report["unstarted_actions"][0]["action"] == before.actions[0]
    assert invoke(p, key) == report
    after = request_state(p, key)
    assert (after.run_id, after.started_at, after.actions, after.cost) == (
        before.run_id,
        before.started_at,
        before.actions,
        before.cost,
    )
    with p.runtime._connect() as c:
        assert (
            c.execute(
                "SELECT count(*) AS n FROM runtime.tool_attempt WHERE run_id=%s", (before.run_id,)
            ).fetchone()["n"]
            == 0
        )
        assert (
            c.execute(
                "SELECT count(*) AS n FROM runtime.artifact_metadata WHERE run_id=%s",
                (before.run_id,),
            ).fetchone()["n"]
            == 0
        )
    assert not processes(p)


@pytest.mark.parametrize("phase", ["confirmed", "journal"])
def test_expired_request_recovers_execution_facts_without_restart(
    assembled_pipeline, monkeypatch, phase
):
    p, key = assembled_pipeline, "expired-" + phase
    router = p.tools.executor_for

    def fail_receipt(plan):
        executor = router(plan)
        if phase == "confirmed":
            confirm = executor._repo.confirm

            def lost(workflow):
                confirm(workflow)
                raise OSError("confirmed receipt lost")

            monkeypatch.setattr(executor._repo, "confirm", lost)
        else:
            from tests.tool_execution.test_execution import FailingStepCommit

            # Reuse the existing 06D actual PG/Audit transaction fault, after journal write.
            monkeypatch.setattr(executor, "_repo", FailingStepCommit(p.runtime._target))
        return executor

    monkeypatch.setattr(p.tools, "executor_for", fail_receipt)
    import psycopg

    with pytest.raises((OSError, psycopg.Error)):
        invoke(p, key)
    before = request_state(p, key)
    assert len(processes(p)) == 1
    freeze_case_clock(
        monkeypatch, before.started_at + timedelta(seconds=p.config.policy.max_seconds)
    )
    recovered = []

    def recover_only(plan):
        executor = router(plan)
        commit = executor.recover_commit
        monkeypatch.setattr(
            executor, "execute", lambda *a: pytest.fail("expired recovery restarted")
        )

        def track(*args):
            recovered.append(phase)
            return commit(*args)

        monkeypatch.setattr(executor, "recover_commit", track)
        return executor

    monkeypatch.setattr(p.tools, "executor_for", recover_only)
    report = invoke(p, key)
    assert report["stop_reason"] == "time_budget"
    assert report["observations"][0]["execution_status"] == "SUCCEEDED"
    assert report["observations"][0]["qc_status"] == "PASS"
    assert recovered == (["journal"] if phase == "journal" else [])
    assert invoke(p, key) == report and len(processes(p)) == 1
    assert request_state(p, key).started_at == before.started_at
    with p.runtime._connect() as c:
        assert (
            c.execute(
                "SELECT status FROM runtime.tool_attempt WHERE run_id=%s", (before.run_id,)
            ).fetchone()["status"]
            == "CONFIRMED"
        )
        assert (
            c.execute(
                "SELECT count(*) AS n FROM runtime.artifact_metadata WHERE run_id=%s",
                (before.run_id,),
            ).fetchone()["n"]
            == 1
        )


@pytest.mark.parametrize("assembled_pipeline", ["shared"], indirect=True)
def test_reverse_shared_effects_with_both_observations_ready(assembled_pipeline, monkeypatch):
    """Both reviewed slots consume one committed product before rule propagation.

    The test driver makes the two observations available together; the production
    AND stopping rule and the medical interpreter are not changed.
    """
    p = assembled_pipeline
    first, second = p.data["pack"].definition.evidence_slots
    evaluate, validate = p.service.validator.evaluate, p.service._validate_results

    def opposite_first(graph, target, artifact, rule, step):
        bound = evaluate(graph, target, artifact, rule, step)
        if target == first.node_code:
            raw = bound.item.model_dump(mode="json")
            raw["observation"], raw["clinical_effect"] = "NEGATIVE", "REFUTES"
            raw["graph_delta_candidate"]["operations"][0]["to_status"] = "REFUTED"
            return bind_validation(graph, ValidationResult.model_validate(raw))
        return bound

    shared = False

    def together(key, state, i, cfg, admission, graph):
        nonlocal shared
        events = [p.service._read(r) for r in state.trace]
        executions = [e for e in events if e["kind"] == "execution"]
        if executions and not shared:
            from packages.grounding.models import Ref

            shared = True
            requirement = Ref.model_validate_json(
                p.data["requirements"][1].reference().model_dump_json()
            )
            _, query = p.tools.query(
                i,
                graph,
                p.service.locks.read(state.run_id),
                p.runtime.artifacts(i.case.case_code, state.run_id),
                requirement,
                cfg,
                state.started_at,
            )
            assert query.existing_artifacts
            prior = executions[0]
            state = p.service._trace(
                key,
                state,
                "execution",
                action=prior["action"],
                target=second.node_code,
                requirement=requirement,
                workflow=prior["workflow"],
                reuse_only=True,
            )
            state = validate(key, state, i, cfg, admission, graph)
            return validate(key, state, i, cfg, admission, p.runtime.current_graph(state.run_id))
        return validate(key, state, i, cfg, admission, graph)

    monkeypatch.setattr(p.service.validator, "evaluate", opposite_first)
    monkeypatch.setattr(p.service, "_validate_results", together)
    report = invoke(p, "reverse-shared")
    assert shared and len(processes(p)) == 1
    assert_effects_agree(p, report, {first.node_code: "REFUTES", second.node_code: "SUPPORTS"})


def assert_effects_agree(p, report, expected):
    states = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED"}
    graph = {n["node_code"]: n["status"] for n in report["graph"]["nodes"]}
    audit = {
        e.payload["accepted_delta_event"]["validation_result"]["target_node_code"]: e.payload[
            "accepted_delta_event"
        ]["validation_result"]["clinical_effect"]
        for e in p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
        if e.run_id == report["run_id"] and e.payload.get("validation_checksum")
    }
    for observation in report["observations"]:
        target = observation["target"]
        assert observation["clinical_effect"] == audit[target] == expected[target]
        assert graph[target] == states[expected[target]]
        details = observation["evidence_validations"][0]["validations"]
        assert len(details) == 1 and details[0]["clinical_effect"] == expected[target]
        assert (
            details[0]["source_snapshot"]["graph_version"] + 1
            == details[0]["accepted_graph"]["graph_version"]
        )
        assert details[0]["audit_event_id"] and details[0]["rule_refs"]
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]
    # Optional export of new facts, never rewriting the predecessor evidence.
    import json
    import os
    from pathlib import Path

    destination = os.environ.get("MMAGENT_07B_REPAIR_EVIDENCE_DIR")
    if destination:
        root = Path(destination) / ("shared-" + "-".join(expected.values()))
        root.mkdir(parents=True, exist_ok=False)
        events = [
            e.model_dump(mode="json")
            for e in p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
            if e.run_id == report["run_id"]
        ]
        for name, value in dict(
            report=report,
            graph=report["graph"],
            audit=events,
            trace=[
                p.service._read(r)
                for r in request_state(
                    p,
                    "target-attribution"
                    if next(iter(expected.values())) == "SUPPORTS"
                    else "reverse-shared",
                ).trace
            ],
        ).items():
            (root / (name + ".json")).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            )

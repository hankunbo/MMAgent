"""09E: one accepted PG fact shared by path, simulation and recovery consumers.

Provider and domain/action owners are explicit synthetic fixtures; the original
acceptance, scoring, Planner, Runtime/Object and Audit implementations are real.
This test never executes a patient tool, model, extension worker or Memory writer.
"""

import html
import json
import os
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.path_graph_validation import PathGraphValidation
from packages.planner_control.actions import RankedActions
from packages.validation_recovery import (
    BranchRoute,
    Failure,
    NearestLegalActions,
    RecoveryPolicy,
    ValidationRecovery,
)
from tests.agent_provider.support import ref
from tests.counterfactual.pg_repair_support import alternative_pg
from tests.counterfactual.support import candidate, fixture
from tests.database.knowledge_support_04a import sql
from tests.expert_agent.pg_support import runtime_fixture
from tests.expert_agent.support import expert_reply
from tests.expert_agent.test_pg_boundaries import database as database


def decoded(value):
    return json.loads(html.unescape(value.split("<pre>")[1].split("</pre>")[0]))


def test_shared_pg_fact_path_simulation_then_stop_handoff(database, tmp_path):
    pg = runtime_fixture(tmp_path)
    alternative = fixture(pg=alternative_pg(pg))
    x = fixture(
        pg=pg,
        allow={"remove_evidence": ("item:one",), "select_check": ("item:one",)},
        policy_updates={"alternatives": {"alt": alternative.fact.checksum}},
    )
    x.service.alternative_readers = {"alt": alternative.reader}
    paths = PathGraphValidation(reader=x.reader.path_reader, documents=pg.documents)
    summary = paths.inspect(key="09e-shared")
    assert all(g["requirements_closed"] for g in summary["goals"])
    assert summary["coverage"] == x.fact.scores["root"]["coverage"]
    original = pg.repo.get_case_bundle(x.f.binding.case_code)
    raw = {a.asset_code: pg.documents.read_artifact(a) for a in original.artifacts}
    original_audit = sql(database.admin, "SELECT * FROM audit.event ORDER BY event_id")
    before_control = x.h.planner.read(x.f.binding.run_id)
    original_report = x.reader().summary.original_report
    requests = x.f.budget.used, alternative.f.budget.used
    remov = candidate(targets=("item:one",), id="09e-remove")
    paired = candidate("select_check", ("item:one",), id="09e-pair", alternative="alt")
    x.service.declare((remov, paired))
    reports = {c.id: x.service.run(c)["report"] for c in (remov, paired)}
    assert pg.documents.read("simulation:policy")[0]["semantics"] == (
        "counterfactual-semantics/1.1.0"
    )
    assert reports[remov.id]["simulated"]["root"]["summary"]["confidence"]["value"] is None
    pair = reports[paired.id]["alternatives"]["alt"]
    assert pair["before"]["run_id"] != pair["after"]["run_id"]
    assert pair["before"]["input_checksum"] == pair["after"]["input_checksum"]
    assert pair["created_runs"] == 0
    for c in (remov, paired):
        saved = pg.documents.read("simulation:result:" + c.id)
        assert saved[0] == reports[c.id] == decoded(saved[1]["html"])
        assert saved[0]["schema_version"] == "counterfactual-report/1.1.0"
        assert saved[0]["baseline"]["reference"]["report_checksum"] == digest(original_report)
        assert saved[0]["counters"]["actual"] == dict(
            model_requests=0, tool_starts=0, created_runs=0
        )
    assert x.h.planner.read(x.f.binding.run_id) == before_control
    assert (x.f.budget.used, alternative.f.budget.used) == requests == (1, 1)
    assert x.reader().summary.original_report == original_report

    # Move through original Planner events, then consume the SAME persisted summary
    # at the recovery choice point. Synthetic execution supplies no patient facts.
    for action in ("retrieve", "instantiate", "identify", "select", "prepare", "execute", "verify"):
        x.h.planner.step(x.f.binding.run_id, x.h.command(action))
    policy = RecoveryPolicy(
        routes=(BranchRoute(branch="A", forks=("root",)), BranchRoute(branch="B", forks=("root",))),
        source=next(r for r in x.f.binding.locks if r.role == "rule"),
    )
    nearest = NearestLegalActions(x.f.source, policy)
    x.h.planner.ranking = RankedActions(nearest)
    x.h.service.context.source = nearest
    x.f.provider.transport.replies.append(expert_reply(action="safe_stop"))
    recovery = ValidationRecovery(
        planner=x.h.planner,
        review=x.h.service,
        documents=pg.documents,
        policy=policy,
        failure_reader=lambda _: Failure(
            category="missing_input",
            target="GOAL:SYNTHETIC",
            reason="synthetic_additional_input_unavailable",
            sources=(
                ref("graph", summary["graph"]["code"], checksum=summary["graph"]["checksum"]),
            ),
        ),
        summary_reader=lambda: paths.read(key="09e-shared"),
    )
    result = recovery.resolve(x.f.binding.run_id, key="09e-stop")
    assert result["control"]["phase"] == "stopped"
    assert x.h.planner.read(x.f.binding.run_id).pending is None
    assert result["report"]["incident"]["summary"] == summary
    assert result["report"]["clinical_effect"] is None
    assert result["report"]["incident"]["failure"]["category"] == "missing_input"
    saved_stop = pg.documents.read("recovery:09e-stop:report")
    assert decoded(saved_stop[1]["html"]) == saved_stop[0] == result["report"]
    assert x.f.budget.used == 2 and alternative.f.budget.used == 1
    assert x.f.budget.summary().api_requests_observed == 0
    assert x.h.planner.replay(x.f.binding.run_id).graph == before_control.graph
    assert paths.read(key="09e-shared", historical=True) == summary
    assert x.reader().summary.original_report == original_report
    assert all(pg.documents.read_artifact(a) == raw[a.asset_code] for a in original.artifacts)
    after_audit = sql(database.admin, "SELECT * FROM audit.event ORDER BY event_id")
    assert all(row in after_audit for row in original_audit)
    assert sql(database.admin, "SELECT count(*) FROM runtime.run")[0][0] == 2
    assert sql(database.admin, "SELECT count(*) FROM runtime.tool_attempt")[0][0] == 0
    with pytest.raises(ValueError, match="simulation_control:"):
        x.service.read(candidate_id=paired.id)
    assert x.service.read(candidate_id=paired.id, historical=True) == reports[paired.id]
    assert x.service.run(paired)["status"] == "stopped"
    count = len(after_audit)
    assert recovery.resolve(x.f.binding.run_id, key="09e-stop") == result
    assert len(sql(database.admin, "SELECT * FROM audit.event")) == count
    assert x.f.budget.used == 2

    evidence = dict(
        fixture="synthetic_domain_and_fake_provider_real_owned_pg",
        original_artifacts=len(raw),
        original_audit_rows=len(original_audit),
        added_audit_rows=len(after_audit) - len(original_audit),
        original_bytes_unchanged=True,
        simulation_preserved_control_and_requests=True,
        final_pending=None,
        primary_fake_requests=2,
        alternative_fake_requests=1,
        api_requests=0,
        tool_attempts=0,
        assembly_created_runs=2,
        simulation_created_runs=0,
        memory_writes=0,
    )
    print(json.dumps(evidence, sort_keys=True))
    # Optional review artifact export; normal project tests only use their owned tmp_path.
    export = Path(os.environ.get("MMAGENT_09E_EXPORT_DIR", str(tmp_path / "examples")))
    export.mkdir(parents=True, exist_ok=True)
    values = {
        "fact": original_report,
        "path": summary,
        "policy": pg.documents.read("simulation:policy")[0],
        "alternative-fact": alternative.reader().summary.original_report,
        "recovery": saved_stop[0],
        "control": result["control"],
        "evidence": evidence,
        **reports,
    }
    for name, value in values.items():
        (export / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    for name, body in {
        "path": pg.documents.read("path-graph:09e-shared")[1]["html"],
        "recovery": saved_stop[1]["html"],
        **{
            c.id: pg.documents.read("simulation:result:" + c.id)[1]["html"] for c in (remov, paired)
        },
    }.items():
        (export / (name + ".html")).write_text(body)

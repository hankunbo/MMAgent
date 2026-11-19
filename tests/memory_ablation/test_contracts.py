"""New freeze/split contracts only. CPU ports do not count as ablation results."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.memory_review.task_models import TaskQuery
from packages.planner_control.actions import RankedActions
from tests.memory_ablation.freeze import independent, sha, write
from tests.memory_review.support import QueryOwner
from tests.memory_review.test_planner import Documents
from tests.planner.support import command
from tests.planner.test_actions import setup_ranked
from tests.research_task_completion.protocol import CONFIG, GROUPS, install, load_protocol


def frozen(tmp_path, rid="RUN:QUERY", case="CASE:QUERY"):
    value = json.loads(CONFIG.read_text())
    value.update(
        protocol_id="10e-abcd-actual-v1",
        execution_status="frozen_before_comparison",
        task_profile=digest("task"),
    )
    query = dict(run_id=rid, case_code=case, patient_hash=digest("query"))
    pool = {"queries": [query], "entries": {}}
    for group in ("success", "recovery", "counterfactual"):
        pool["entries"][group] = [
            dict(
                publication=group,
                group=group,
                supports=[
                    dict(
                        run_id="RUN:" + group.upper(),
                        case_code="CASE:" + group.upper(),
                        patient_hash=digest(group),
                    )
                ],
            )
        ]
    value["memory_pool_freeze"] = write(tmp_path / "pool.json", pool)
    value["task_profile_ref"] = write(
        tmp_path / "profile.json", {"task_profile": value["task_profile"]}
    )
    path = tmp_path / "actual.json"
    write(path, value)
    return path, pool


@pytest.mark.parametrize("group", GROUPS)
def test_explicit_actual_installation_binds_real_query_groups(tmp_path, group):
    planner, runtime, owner, source = setup_ranked()
    state = runtime.state
    path, pool = frozen(tmp_path, state.binding.run_id, state.binding.case_code)
    identities = {s["run_id"]: s for s in pool["queries"]}
    for entries in pool["entries"].values():
        identities.update({s["run_id"]: s for e in entries for s in e["supports"]})
    contexts = SimpleNamespace(
        read_research_run=lambda rid: SimpleNamespace(
            synthetic=True, request=SimpleNamespace(scope=SimpleNamespace(**identities[rid]))
        )
    )

    # Deliberately revoked CPU pool. Actual PG gate must independently prove nonempty hits.
    def revoked(key):
        raise ValueError("memory_qualification_revoked")

    repo = SimpleNamespace(contexts=contexts, read_publication=revoked)
    query = QueryOwner(state.binding.run_id, state.binding.case_code)
    query.value = TaskQuery(
        **query.value.model_dump(exclude={"schema_version"}), task_profile=digest("task")
    )
    docs = Documents(state.binding.run_id)
    planner.ranking = install(
        group,
        source,
        query_reader=query,
        repository=repo,
        documents=docs,
        protocol_path=path,
        protocol_sha256=sha(path),
    )
    event = planner.step(state.binding.run_id, command(planner, "select"))
    assert event.outcome.decision.selected.branch == "A"
    assert planner.ranking.experiment_binding["protocol_sha256"] == sha(path)
    if group == "A":
        assert type(planner.ranking) is RankedActions and query.calls == docs.writes == 0
    else:
        receipt = planner.ranking.read_influence(runtime.state, event.outcome.decision)
        assert tuple(receipt["retrieval"]["query"]["groups"]) == GROUPS[group]
        assert receipt["retrieval"]["rejected"]


@pytest.mark.parametrize("group", ("success", "recovery", "counterfactual"))
@pytest.mark.parametrize("field", ("case_code", "patient_hash"))
def test_all_groups_reject_support_or_related_patient_overlap(tmp_path, group, field):
    path, pool = frozen(tmp_path)
    pool["entries"][group][0]["supports"][0][field] = pool["queries"][0][field]
    value = json.loads(path.read_text())
    new_pool = tmp_path / "overlap.json"
    value["memory_pool_freeze"] = write(new_pool, pool)
    altered = tmp_path / "overlap-actual.json"
    write(altered, value)
    with pytest.raises(ValueError, match="10e_support_query_overlap"):
        load_protocol(altered, expected_sha256=sha(altered))


def test_changed_reference_pending_template_and_wrong_digest_refused(tmp_path):
    path, _ = frozen(tmp_path)
    with pytest.raises(ValueError, match="10e_protocol_digest"):
        load_protocol(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="10e_protocol_mismatch"):
        load_protocol(CONFIG, expected_sha256=sha(CONFIG))
    (tmp_path / "profile.json").write_text("{}\n")
    with pytest.raises(ValueError, match="10e_reference_digest"):
        load_protocol(path, expected_sha256=sha(path))
    with pytest.raises(ValueError, match="10e_split_empty"):
        independent([], [])


def test_working_recovery_body_matches_full_previously_passed_correction():
    root = Path(__file__).resolve().parents[2]
    name = "test_pg_prebound_definition_recovery_and_required_failures"

    def body(path):
        tree = ast.parse((root / path).read_text())
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    working = body("tests/research_task_completion/test_postgres.py")
    historical = body("tests/research_task_binding_increment/test_postgres.py")
    # Only historical extra export follows the exact complete corrected recovery body.
    assert isinstance(historical.body[-1], ast.Expr)
    assert historical.body[-1].value.func.id == "export"
    historical.body.pop()
    assert ast.dump(working, include_attributes=False) == ast.dump(
        historical, include_attributes=False
    )


@pytest.mark.parametrize("group", ("success", "recovery", "counterfactual"))
def test_current_identity_and_profile_rechecked_after_install(tmp_path, group):
    from tests.memory_ablation.freeze import FrozenPool
    from tests.research_task_completion.protocol import GroupQueryReader

    path, pool = frozen(tmp_path)
    identities = {s["run_id"]: s for s in pool["queries"]}
    for entries in pool["entries"].values():
        identities.update({s["run_id"]: s for e in entries for s in e["supports"]})
    contexts = SimpleNamespace(
        read_research_run=lambda rid: SimpleNamespace(
            request=SimpleNamespace(scope=SimpleNamespace(**identities[rid]))
        )
    )
    repository = SimpleNamespace(contexts=contexts)
    q = QueryOwner("RUN:QUERY", "CASE:QUERY")
    q.value = TaskQuery(
        **q.value.model_dump(exclude={"schema_version"}), task_profile=digest("task")
    )
    reader = GroupQueryReader(q, (group,), frozen=FrozenPool(path, sha(path), repository))
    assert reader.current("RUN:QUERY").groups == (group,)
    old = q.value
    q.value = old.model_copy(update={"task_profile": digest("wrong-profile")})
    with pytest.raises(ValueError, match="10e_query_profile"):
        reader.current("RUN:QUERY")
    q.value = old
    rid = pool["entries"][group][0]["supports"][0]["run_id"]
    identities[rid] = {**identities[rid], "patient_hash": digest("query")}
    with pytest.raises(ValueError, match="10e_support_identity_drift"):
        reader.check_versions(old)
    assert load_protocol()["execution_status"] == "task_pg_gate_pending"

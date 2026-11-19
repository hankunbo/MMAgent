import html
import json

import pytest

from packages.counterfactual.models import Candidate, Intervention
from packages.evidence_scoring.accepted import rows_from_candidate
from packages.evidence_scoring.api import validate as validate_score
from packages.evidence_scoring.continuous import summarize
from packages.grounding.models import digest
from tests.counterfactual.support import candidate, fixture, protection, run


@pytest.mark.parametrize(
    "directions,target",
    [
        ((0.8, 0.2), "item:0"),
        ((0.8, 0.2), "item:1"),
        ((0.8, -0.7), "item:1"),
        ((-0.8, 0.2), "item:0"),
    ],
)
def test_original_formula_direction_not_assumed_monotone(directions, target):
    x = fixture(statuses=("obtained", "obtained"), directions=directions)
    before = protection(x)
    r = run(x, candidate(targets=(target,)))["report"]
    expected = summarize(rows_from_candidate(x.parent, x.fact.candidate), removed_ids={target})
    assert r["simulated"]["root"]["summary"] == expected
    assert r["comparisons"]["score"]["delta"] * directions[int(target[-1])] < 0
    assert protection(x) == before
    assert r["identity"] == "hypothetical_candidate" and r["disease_probability"] is None
    assert r["counters"]["actual"] == dict(model_requests=0, tool_starts=0, created_runs=0)
    assert r["counters"]["expected_cost"] is None


def test_all_removed_null_fixed_denominator_and_html_identity():
    x = fixture()
    r = run(x)["report"]
    assert r["simulated"]["root"]["summary"]["confidence"]["value"] is None
    assert r["classification"] == "necessary"
    for scope, row in r["baseline"]["root"]["coverage"].items():
        new = r["simulated"]["root"]["coverage"][scope]
        assert new["denominator"] == row["denominator"] and new["acquisition"] == 0
    saved = x.f.documents.read("simulation:result:one")
    assert json.loads(html.unescape(saved[1]["html"].split("<pre>")[1].split("</pre>")[0])) == r
    assert x.service.read(candidate_id="one") == r
    with pytest.raises(ValueError, match="unsupported_schema"):
        validate_score(dict(state=x.parent, candidate=x.fact.candidate), r["simulated"])


def test_reference_missing_only_acquisition_changes():
    x = fixture(statuses=("obtained", "reference_missing"))
    r = run(x, candidate(targets=("item:1",)))["report"]
    assert r["comparisons"]["score"]["delta"] == 0
    assert r["comparisons"]["comprehensive_research:acquisition"]["delta"] < 0
    assert r["comparisons"]["comprehensive_research:interpretation"]["delta"] == 0


def test_correlated_redundancy_and_same_observation_aliases():
    x = fixture(statuses=("obtained", "obtained"), same_source=True, alternative_slot=True)
    r = run(x)["report"]
    assert r["comparisons"]["score"]["delta"] == 0
    assert r["comparisons"]["comprehensive_research:interpretation"]["delta"] == 0
    x = fixture(statuses=("obtained", "obtained"), shared_product=True)
    r = run(x)["report"]
    assert {a["item"] for a in r["affected"]} == {"item:0", "item:1"}
    assert r["remaining"] == []


def test_resource_AND_OR_shared_dependencies():
    from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
    from packages.grounding.models import validate
    from packages.llm_provider.models import Material
    from tests.agent_provider.support import ref

    def mutate(reader, binding):
        state = reader.state
        policy = state.policy.model_dump(mode="json")
        policy["items"][0]["dependency_paths"] = [["RES:A", "RES:B"], ["RES:C"]]
        policy["items"][1]["dependency_paths"] = [["RES:A", "RES:B"]]
        policy["items"][2]["dependency_paths"] = [["RES:C"]]
        policy = validate(TaskPolicyV2, policy)
        raw = state.material.model_dump(mode="json")
        raw["scope"]["policy_checksum"] = digest(policy)
        raw["policy_ref"]["checksum"] = digest(policy)
        raw["scope"].pop("material_checksum")
        raw["scope"]["material_checksum"] = digest(raw)
        reader.state = validate(
            DomainState,
            state.model_copy(
                update=dict(
                    policy=policy,
                    material=validate(Material, raw),
                    resources=tuple(
                        ref("resource", c, checksum=digest(c)) for c in ("RES:A", "RES:B", "RES:C")
                    ),
                )
            ),
        )
        return reader, binding

    x = fixture(statuses=("obtained",) * 3, mutate=mutate)
    cs = (
        candidate("remove_resource", ("RES:A",), id="and"),
        candidate("remove_resource", ("RES:A", "RES:C"), id="or"),
    )
    x.service.declare(cs)
    a, b = (x.service.run(c)["report"] for c in cs)
    assert a["remaining"] == ["item:0", "item:2"]
    assert b["remaining"] == []


def test_bounded_search_records_scope_and_never_claims_global_optimum():
    x = fixture(statuses=("obtained",) * 3, same_source=True, policy_updates=dict(max_search=2))
    r = x.service.search()
    assert r["explored"] == 2 and r["total_subsets"] == 7
    assert not r["complete"] and not r["global_optimum"]
    assert x.f.budget.used == 1


def test_policy_rejection_before_computation():
    x = fixture()
    with pytest.raises(ValueError, match="policy_denied"):
        x.service.declare((candidate("reverse"),))
    with pytest.raises(ValueError, match="not_predeclared"):
        x.service.run(candidate())
    bad = Candidate(
        id="bad",
        interventions=(
            Intervention(kind="remove_evidence", targets=("item:0",)),
            Intervention(kind="reverse", targets=("item:0",)),
        ),
    )
    with pytest.raises(ValueError):
        x.service.declare((bad,))


def test_dependency_closure_transitive_observation_resource_and_alias():
    from types import SimpleNamespace

    from packages.graph_engine.simulation import removal_closure
    from tests.agent_provider.support import ref

    r1 = ref("resource", "OBS:A")
    r2 = ref("resource", "OBS:B")
    alias = r1.model_copy(update=dict(code="RES:ALIAS"))
    state = SimpleNamespace(
        policy=SimpleNamespace(
            items=(
                SimpleNamespace(id="a", observation_code="OBS:A", dependency_paths=(("ROOT",),)),
                SimpleNamespace(id="b", observation_code="OBS:B", dependency_paths=(("OBS:A",),)),
                SimpleNamespace(id="c", observation_code="OBS:C", dependency_paths=(("OBS:B",),)),
                SimpleNamespace(
                    id="independent", observation_code="OBS:D", dependency_paths=(("ROOT",),)
                ),
                SimpleNamespace(
                    id="alias", observation_code="OBS:E", dependency_paths=(("RES:ALIAS",),)
                ),
            )
        ),
        resources=(r1, r2, alias, ref("resource", "ROOT")),
        material=SimpleNamespace(
            observations=tuple(
                SimpleNamespace(reference=r)
                for r in (
                    r1,
                    r2,
                    ref("observation", "OBS:C"),
                    ref("observation", "OBS:D"),
                    ref("observation", "OBS:E"),
                )
            )
        ),
    )
    affected, removed = removal_closure(state, evidence=("a",))
    assert set(affected) == {"a", "b", "c", "alias"}
    assert removed == {"OBS:A", "OBS:B", "RES:ALIAS"}

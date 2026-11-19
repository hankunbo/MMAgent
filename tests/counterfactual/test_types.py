import pytest

from packages.counterfactual.engine import paired
from packages.counterfactual.service import CounterfactualSimulation
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material
from tests.counterfactual.support import candidate, fixture, protection, run


def other_run(*, new_tool=True, same_output=False, unit=None, case=None, input_changed=False):
    def mutate(reader, binding):
        changes = dict(run_id="RUN:SIM_ALTERNATIVE")
        if case:
            changes["case_code"] = case
        if input_changed:
            changes["input_checksum"] = digest("different input")
        if new_tool:
            changes["locks"] = tuple(
                r.model_copy(update=dict(version="2.0.0", checksum=digest("new tool")))
                if r.role == "catalog"
                else r
                for r in binding.locks
            )
        binding = binding.model_copy(update=changes)
        state = reader.state
        raw = state.material.model_dump(mode="json")
        raw["scope"].update(
            run_id=binding.run_id,
            case_code=binding.case_code,
            input_checksum=binding.input_checksum,
        )
        if not same_output:
            raw["observations"][0]["reference"]["checksum"] = digest("new output actual bytes")
            raw["observations"][0]["value"] = "Synthetic replacement result"
        policy = state.policy
        if unit:
            policy_raw = policy.model_dump(mode="json")
            policy_raw["items"][0]["unit"] = unit
            policy = validate(TaskPolicyV2, policy_raw)
            raw["observations"][0]["unit"] = unit
            raw["policy_ref"]["checksum"] = digest(policy)
            raw["scope"]["policy_checksum"] = digest(policy)
        raw["scope"].pop("material_checksum")
        raw["scope"]["material_checksum"] = digest(raw)
        reader.state = validate(
            DomainState,
            state.model_copy(update=dict(policy=policy, material=validate(Material, raw))),
        )
        return reader, binding

    return mutate


def replacing(kind, *, directions=(0.3,), statuses=("obtained",), **kwargs):
    y = fixture(directions=directions, statuses=statuses, mutate=other_run(**kwargs))
    x = fixture(
        allow={kind: ("item:0",)}, policy_updates=dict(alternatives={"alt": y.fact.checksum})
    )
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=x.f.documents,
        policy=x.policy,
        alternative_readers={"alt": y.reader},
    )
    return x, y


@pytest.mark.parametrize("kind", ["select_check", "replace_tool", "replace_path"])
@pytest.mark.parametrize("effect", [0.3, -0.7])
def test_typed_replacement_legal_equivalent_and_changed(kind, effect):
    x, y = replacing(kind, directions=(effect,))
    before = protection(x), protection(y)
    r = run(x, candidate(kind, alternative="alt"))["report"]
    assert r["simulated"]["root"]["summary"] == y.fact.scores["root"]["summary"]
    assert (
        r["comparisons"]["score"]["delta"] == 0
        if effect == 0.3
        else r["comparisons"]["score"]["delta"] < 0
    )
    assert r["alternatives"]["alt"]["created_runs"] == 0
    assert r["alternatives"]["alt"]["attribution"] == "no_single_tool_or_causal_attribution"
    assert (protection(x), protection(y)) == before


@pytest.mark.parametrize("kind", ["select_check", "replace_tool", "replace_path"])
@pytest.mark.parametrize("failure", ["domain", "missing", "unprovided", "policy"])
def test_replacement_rejects_or_reports_missing(kind, failure):
    if failure == "unprovided":
        x = fixture(allow={kind: ("item:0",)})
        r = run(x, candidate(kind))["report"]
        assert r["simulated"] is None and r["classification"] == "indeterminate"
        return
    x, y = replacing(
        kind,
        unit="mm" if failure == "domain" else None,
        statuses=("reference_missing",) if failure == "missing" else ("obtained",),
    )
    c = candidate(kind, alternative="unknown" if failure == "policy" else "alt")
    if failure == "missing":
        r = run(x, c)["report"]
        assert r["simulated"] is None and "missing_observation_or_interpretation" in r["reasons"][0]
    else:
        with pytest.raises(ValueError, match="type_unit_domain|not_declared"):
            run(x, c)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (dict(same_output=True), "new_output"),
        (dict(new_tool=False), "catalog_change"),
        (dict(case="CASE:OTHER"), "common_input"),
        (dict(input_changed=True), "common_input"),
    ],
)
def test_tool_pair_rejects_false_replay(kwargs, reason):
    x, _ = replacing("replace_tool", **kwargs)
    with pytest.raises(ValueError, match=reason):
        run(x, candidate("replace_tool", alternative="alt"))


def test_pair_reports_each_changed_lock_and_same_run_rejected():
    x, y = replacing("replace_tool")
    r = paired(x.fact, y.fact)
    assert {d["field"] for d in r["differences"]} >= {"lock:catalog", "observations"}
    with pytest.raises(ValueError, match="common_input_or_run"):
        paired(x.fact, x.fact)


@pytest.mark.parametrize("status", ["obtained", "reference_missing"])
def test_reverse_hypothesis_never_changes_measurement(status):
    x = fixture(statuses=(status,), allow={"reverse": ("item:0",)})
    before = protection(x)
    if status == "reference_missing":
        with pytest.raises(ValueError, match="missing_interpretation"):
            run(x, candidate("reverse"))
    else:
        r = run(x, candidate("reverse"))["report"]
        assert r["simulated"]["root"]["summary"]["combined"]["refutation"] > 0
        assert r["classification"] == "conclusion_changed"
    assert protection(x) == before

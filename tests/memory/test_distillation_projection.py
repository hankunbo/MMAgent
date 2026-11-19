"""Small projection boundaries. Persistence is separately exercised on owned PG."""

from types import SimpleNamespace

import pytest

from apps.memory_execution_sources import ExecutionSources
from apps.memory_sources import ResearchSources, SourceRegistry
from packages.counterfactual.models import Candidate, Intervention
from packages.counterfactual.service import CounterfactualSimulation
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material
from packages.memory.distillation_models import PairData
from packages.memory_distillation.projection import build
from packages.memory_writer import MemoryWriter
from tests.counterfactual.support import candidate, fixture, protection, run
from tests.counterfactual.test_types import other_run


def opaque(reader, binding):
    binding = binding.model_copy(
        update=dict(
            case_code="CASE:OID_PROJECT",
            run_id=(
                "RUN:OID_ALTERNATIVE"
                if binding.run_id == "RUN:SIM_ALTERNATIVE"
                else "RUN:OID_PROJECT"
            ),
        )
    )
    raw = reader.state.material.model_dump(mode="json")
    raw["scope"].update(case_code=binding.case_code, run_id=binding.run_id)
    raw["scope"].pop("material_checksum")
    raw["scope"]["material_checksum"] = digest(raw)
    reader.state = reader.state.model_copy(update=dict(material=validate(Material, raw)))
    return reader, binding


def sources(x):
    b = x.f.binding
    runtime = SimpleNamespace(
        read_memory_execution=lambda rid: dict(
            case_code=b.case_code,
            run_id=b.run_id,
            context_checksum=b.run_context_checksum,
            synthetic=True,
            runtime_status="CREATED",
            lifecycle_phase=None,
            report=None,
            attempts=[],
            preparation_rejected=False,
            versions={},
        )
    )
    research = ResearchSources(
        fact_reader=x.reader,
        applicability=("USE:SYNTHETIC",),
        simulation=x.service,
        documents=x.f.documents,
    )
    return ExecutionSources(
        runtime=runtime,
        run_id=b.run_id,
        applicability=("USE:SYNTHETIC",),
        planner=x.h.planner,
        research=research,
    )


def extract(x, c, alternative=None):
    report = run(x, c)["report"]
    x.h.planner.step(x.f.binding.run_id, x.h.command("stop"))
    owner = sources(x)
    registry = SourceRegistry({owner.run_id: owner})
    if alternative:
        alt_owner = sources(alternative)
        registry.owners[alt_owner.run_id] = alt_owner
    episode = MemoryWriter(repository=None, sources=owner).prepare(
        run_id=owner.run_id,
        request_key="projection",
        supplemental=(("simulation", c.id, "counterfactual_report"),),
    )[-1]
    protected = protection(x), protection(alternative) if alternative else None
    records = build("projection", (episode,), registry)
    pair = next(r.detail for r in records if isinstance(r.detail, PairData))
    assert (protection(x), protection(alternative) if alternative else None) == protected
    assert pair.context.input_checksum == x.f.binding.input_checksum
    assert pair.context.episode_input_checksum == episode.detail.input_checksum
    assert pair.context.input_checksum != pair.context.episode_input_checksum
    assert pair.classification == report["classification"]
    assert pair.comparisons == report["comparisons"]
    for field in ("baseline", "simulated", "alternatives"):
        assert getattr(pair, field) == report[field]
    for field in ("reasons", "affected", "remaining"):
        assert getattr(pair, field) == tuple(report[field])
    return pair


@pytest.mark.parametrize(
    "metrics,expected",
    [
        (("score",), "context_redundant"),
        (("support", "refutation", "conflict"), "components_changed"),
    ],
)
def test_zero_score_delta_keeps_original_multimetric_classification(metrics, expected):
    x = fixture(
        mutate=opaque,
        statuses=("obtained",) * 4,
        directions=(0.8, -0.8, 0.2, -0.2),
        policy_updates=dict(metrics=metrics, coverage_retention=0.25),
    )
    pair = extract(x, candidate(targets=("item:0", "item:1")))
    assert pair.comparisons["score"]["delta"] == 0
    assert pair.classification == expected
    assert pair.comparisons["support"]["delta"] != 0
    assert pair.comparisons["refutation"]["delta"] != 0
    assert len(pair.assessment_contexts["original"]) == 4
    assert all(
        "correlation_group" in p and "dependency_paths" in p
        for p in pair.assessment_contexts["original"]
    )


@pytest.mark.parametrize("kind", ["select_check", "replace_tool", "replace_path"])
def test_typed_trusted_replacement_projection(kind):
    alternative = fixture(directions=(-0.7,), mutate=lambda r, b: opaque(*other_run()(r, b)))
    x = fixture(
        mutate=opaque,
        allow={kind: ("item:0",)},
        policy_updates=dict(alternatives={"alt": alternative.fact.checksum}),
    )
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=x.f.documents,
        policy=x.policy,
        alternative_readers={"alt": alternative.reader},
    )
    pair = extract(x, candidate(kind, alternative="alt"), alternative)
    assert pair.changes[0].operation == "replace" and not pair.changes[0].removed
    assert pair.changes[0].original_value == 0.3
    assert pair.changes[0].simulated_value == -0.7
    assert pair.classification == "conclusion_changed"
    assert pair.alternatives["alt"]["attribution"] == "no_single_tool_or_causal_attribution"


@pytest.mark.parametrize("kind", ["remove_evidence", "remove_resource"])
def test_removal_explicit_null_fixed_denominator_and_dependency_closure(kind):
    x = fixture(mutate=opaque, statuses=("obtained", "reference_missing"), shared_product=True)
    targets = ("item:1",) if kind == "remove_evidence" else (x.parent.resources[0].code,)
    pair = extract(x, candidate(kind, targets=targets))
    assert all(c.removed and c.simulated_value is None for c in pair.changes)
    assert pair.affected
    for scope, before in pair.baseline["root"]["coverage"].items():
        assert pair.simulated["root"]["coverage"][scope]["denominator"] == before["denominator"]


def test_reverse_retains_observation_and_hypothetical_identity():
    x = fixture(mutate=opaque, allow={"reverse": ("item:0",)})
    pair = extract(x, candidate("reverse"))
    change = pair.changes[0]
    assert change.original_observation == change.simulated_observation
    assert change.original_value == -change.simulated_value
    assert pair.identity == "hypothetical_comparison"


def test_valid_replacement_keeps_its_own_availability_when_other_replacement_missing():
    alternative = fixture(
        statuses=("obtained", "obtained"),
        directions=(-0.7, 0.2),
        mutate=lambda r, b: opaque(*other_run()(r, b)),
    )
    baseline = fixture(
        statuses=("obtained", "obtained"),
        directions=(0.3, 0.2),
        mutate=opaque,
        allow={"select_check": ("item:0",), "replace_tool": ("item:1",)},
        policy_updates=dict(alternatives={"alt": alternative.fact.checksum}),
    )
    baseline.service = CounterfactualSimulation(
        reader=baseline.reader,
        documents=baseline.f.documents,
        policy=baseline.policy,
        alternative_readers={"alt": alternative.reader},
    )
    candidate = Candidate(
        id="mixed-availability",
        interventions=(
            Intervention(kind="select_check", targets=("item:0",), alternative="alt"),
            Intervention(kind="replace_tool", targets=("item:1",)),
        ),
    )
    pair = extract(baseline, candidate, alternative)
    obtained, missing = pair.changes
    assert pair.simulated is None and pair.classification == "indeterminate"
    assert pair.reasons == ("replace_tool:missing_alternative_result",)
    assert missing.missing_reason == "missing_alternative_result"
    assert missing.simulated_value is None
    assert obtained.original_value == 0.3 and obtained.simulated_value == -0.7
    assert obtained.simulated_observation is not None
    assert obtained.missing_reason is None, (
        "Another intervention's failure must not erase this target's confirmed interpretation",
        obtained.model_dump(mode="json"),
    )


@pytest.mark.parametrize(
    "status,value",
    [("reference_missing", 0.0), ("unknown", -0.7), ("missing", 0.0), ("ungradable", -0.7)],
)
def test_one_alternative_partial_targets_keep_local_interpretations(status, value):
    alternative = fixture(
        statuses=("obtained", status),
        directions=(value, None),
        mutate=lambda r, b: opaque(*other_run()(r, b)),
    )
    x = fixture(
        statuses=("obtained", "obtained"),
        directions=(0.0, -0.2),
        mutate=opaque,
        allow={"select_check": ("item:0", "item:1")},
        policy_updates=dict(alternatives={"alt": alternative.fact.checksum}),
    )
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=x.f.documents,
        policy=x.policy,
        alternative_readers={"alt": alternative.reader},
    )
    pair = extract(
        x, candidate("select_check", targets=("item:0", "item:1"), alternative="alt"), alternative
    )
    complete, missing = pair.changes
    assert pair.simulated is None and pair.classification == "indeterminate"
    assert pair.reasons == ("select_check:missing_observation_or_interpretation:item:1",)
    assert complete.original_value == 0.0 and complete.simulated_value == value
    assert complete.missing_reason is None
    assert missing.original_value == -0.2 and missing.simulated_value is None
    assert missing.missing_reason == "missing_observation_or_interpretation"
    for change, observation in zip(
        pair.changes, alternative.fact.state.material.observations, strict=True
    ):
        assert change.simulated_observation == observation.reference.model_dump(mode="json")


@pytest.mark.parametrize("with_alternative", [False, True])
def test_single_genuinely_missing_replacement(with_alternative):
    alternative = (
        fixture(statuses=("reference_missing",), mutate=lambda r, b: opaque(*other_run()(r, b)))
        if with_alternative
        else None
    )
    x = fixture(
        mutate=opaque,
        allow={"select_check": ("item:0",)},
        policy_updates=dict(alternatives={"alt": alternative.fact.checksum} if alternative else {}),
    )
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=x.f.documents,
        policy=x.policy,
        alternative_readers={"alt": alternative.reader} if alternative else {},
    )
    pair = extract(
        x, candidate("select_check", alternative="alt" if alternative else None), alternative
    )
    change = pair.changes[0]
    assert pair.simulated is None and pair.classification == "indeterminate"
    assert change.simulated_value is None and not change.removed
    assert change.missing_reason == (
        "missing_observation_or_interpretation" if alternative else "missing_alternative_result"
    )
    assert (change.simulated_observation is not None) is with_alternative

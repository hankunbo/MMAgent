"""Only immutable per-phase scope projection is shared, never current authorization."""

from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.comprehensive_materials import ComprehensivePhase
from packages.evaluation_preparation.generic_projection import FreshAcquisitionProjection, bind_run
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def test_shared_seed_matches_original_per_phase_projection_and_reads_fresh_next_call():
    seed = dynamic_fixture(2, empty=True).owner.current()
    holder = [seed]
    calls = []

    def project(*, run_id, config):
        calls.append(run_id)
        return bind_run(holder[0], run_id), {}

    base = SimpleNamespace(
        project=project,
        view_policy=dict(transform="synthetic"),
        vision_artifact=SimpleNamespace(
            dimensions=(64, 64), asset_code="ASSET:SYNTHETIC", checksum=digest("pixels")
        ),
    )
    targets = [r.id for r in seed.policy.requirements][:2]
    expected = [
        ComprehensivePhase(base, target=t).project(run_id="RUN:" + str(i), config=None)[0]
        for i, t in enumerate(targets)
    ]
    assert len(calls) == 2
    shared = FreshAcquisitionProjection(base, None)
    actual = [
        shared(ComprehensivePhase(base, target=t), "RUN:" + str(i))[0]
        for i, t in enumerate(targets)
    ]
    assert actual == expected and len(calls) == 3
    raw = seed.model_dump(mode="json")
    raw["material"]["scope"]["input_checksum"] = digest("changed current input")
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    holder[0] = validate(DomainState, raw)
    fresh = FreshAcquisitionProjection(base, None)
    assert fresh(ComprehensivePhase(base, target=targets[0]), "RUN:0")[0] != expected[0]
    assert len(calls) == 4


def test_projection_does_not_accept_review_or_other_base():
    base = object()
    project = FreshAcquisitionProjection(base, None)
    for phase in (
        SimpleNamespace(base=base, target=None),
        SimpleNamespace(base=object(), target="item"),
    ):
        with pytest.raises(ValueError, match="projection_scope"):
            project(phase, "RUN:ANY")

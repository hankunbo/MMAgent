"""Optional current-owner procedure reuse cannot copy a medical result."""

from types import SimpleNamespace as S

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.reuse import OPERATIONS, ProcedureReuse


def fixture(*, same=False, modality="OCT", fault=None):
    record = S(
        kind="tool_experience",
        case_code="CASE:QUERY" if same else "CASE:OLD",
        detail=S(
            target="check:oct",
            modality=modality,
            steps=[
                S(operation=s, tool="RESEARCH:MULTIMODAL_TASK", version="1.0.0") for s in OPERATIONS
            ],
        ),
    )

    class Reader:
        def procedure_keys(self, **kwargs):
            assert kwargs["limit"] == 3
            return ("saved",)

        def read(self, key):
            if fault:
                raise fault
            return S(record=record, checksum=digest("record"))

    campaign = S(
        base=S(
            ledger=[dict(id="check:oct:inner", component="oct", input_modalities=("OCT",))],
            binding=dict(case_code="CASE:QUERY"),
            manifest=S(checksum=digest("input")),
            task=S(checksum=digest("task")),
        )
    )
    return ProcedureReuse(campaign, Reader())


def test_only_method_refs_reused_with_current_owner_not_scores_or_old_artifacts():
    service = fixture()
    value = service.prepare()
    assert value["plans"][0]["selected"]["key"] == "saved"
    assert not value["old_patient_values_copied"] and not value["clinical_evidence"]
    assert service.check() == ("saved",)


@pytest.mark.parametrize(
    "kwargs",
    [dict(same=True), dict(modality="SLO"), dict(fault=ValueError("memory_source_withdrawn"))],
)
def test_mismatch_is_explicit_baseline(kwargs):
    s = fixture(**kwargs)
    assert s.prepare()["plans"][0]["selected"] is None
    assert s.check() == ()


def test_unavailable_is_not_fake_empty_and_mutation_not_authorization():
    with pytest.raises(OSError, match="offline"):
        fixture(fault=OSError("offline")).prepare()
    s = fixture()
    s.reader = None
    assert s.prepare()["plans"][0]["origin"] == "current_owner_baseline"
    s.plan["input_checksum"] = digest("changed")
    with pytest.raises(ValueError, match="reuse_plan_changed"):
        s.check()


def test_withdrawal_after_selection_falls_back_without_new_action():
    s = fixture()
    s.prepare()

    def fail(key):
        raise ValueError("memory_source_withdrawn")

    s.reader.read = fail
    report = s.report()
    assert report["active_keys"] == []
    assert report["checks"][0]["action"] == "current_owner_baseline"


def test_restore_same_selection_never_requeries_pool():
    first = fixture()
    plan = first.prepare()
    second = fixture()

    def forbidden(**kwargs):
        raise AssertionError("selection must not be repeated")

    second.reader.procedure_keys = forbidden
    assert second.restore(plan) == plan
    assert second.check() == ("saved",)


def test_technical_failure_after_selection_is_marked_not_empty():
    s = fixture()
    s.prepare()

    def fail(key):
        raise OSError("unavailable")

    s.reader.read = fail
    assert s.report()["checks"][0]["reason"] == "memory_technical_unavailable"


def test_new_hypothesis_branch_can_be_retrieved_without_copying_its_scores():
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.hypotheses import REGISTRY
    from packages.multimodal_research.reuse import targets_for

    s = fixture()
    s.campaign.base.task = load_intent("glaucoma_four_hypotheses", REGISTRY)[0]
    target = "check:oct:inner:hyp:non_glaucoma"
    s.reader.read("saved").record.detail.target = target
    assert target in targets_for(s.campaign.base, s.campaign.base.ledger[0])
    plan = s.prepare()
    assert plan["plans"][0]["selected"]["key"] == "saved"
    assert s.restore(plan) == plan and s.check() == ("saved",)
    assert not plan["old_patient_values_copied"]

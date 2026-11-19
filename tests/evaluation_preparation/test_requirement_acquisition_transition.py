"""Acquisition method changes do not silently change medical-source authority."""

import copy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from apps.active_evidence_materials import ActiveCaseEvidence
from apps.vectorengine_case_materials import ref
from packages.evidence_reuse import integration
from packages.grounding.models import digest


def definitions():
    owner = SimpleNamespace(
        packet=lambda: SimpleNamespace(reference=ref("source", "SRC:A", digest("source")))
    )
    base = ActiveCaseEvidence.definition(owner, "criterion:VCDR_GE_07").model_dump(mode="json")
    visual = copy.deepcopy(base)
    visual["examination"] = (
        "垂直杯盘比图像估计：本例分割未得到视杯。直接查看本例图像，"
        "如边界可见，描述杯与盘的垂直边界/尺度、估计比值及范围；"
        "这不是掩膜测量。不可辨认时明确ungradable，不补0，不作诊断。"
    )
    visual["applicability_conditions"] += " 本次为VLM近似估计，降低观察可靠性并保留估计范围。"
    visual["reference"] = ref(
        "definition",
        "POLICY:QUALITY:VISUAL_VCDR",
        digest({k: v for k, v in visual.items() if k != "reference"}),
    ).model_dump(mode="json")
    return base, visual


def checker(monkeypatch, frozen, current):
    from packages.evaluation_preparation import generic_input

    monkeypatch.setattr(integration, "implementation", lambda: {})
    monkeypatch.setattr(generic_input, "validate_registered", lambda *a: None)

    class Components(BaseModel):
        pass

    monkeypatch.setattr(integration, "visible_components", lambda *a: Components())
    monkeypatch.setattr(integration, "recheck_plans", lambda *a, **kw: None)
    row = dict(
        id="criterion:VCDR_GE_07",
        definition=frozen,
        input_modalities=["SLO"],
        basis_kind="quantitative",
    )
    requirement = SimpleNamespace(
        **dict(row, definition=SimpleNamespace(model_dump=lambda **kw: current))
    )
    state = SimpleNamespace(
        policy=SimpleNamespace(requirements=[requirement]), material=SimpleNamespace(sources=[])
    )
    result = object.__new__(integration.DispatchCheck)
    result.spec = dict(upstream_implementation={}, sample="synthetic.npz")
    result.registered, result.memory, result.config = None, None, None
    result.original_plan = dict(requirements=[row])
    result.upstream = dict(components={}, skills={})
    result.evidence = SimpleNamespace(
        project=lambda **kw: (state, {}), verify=lambda s: None, visible_value={}, input_context={}
    )
    return result


@pytest.mark.parametrize("reverse", [False, True])
def test_original_two_acquisition_methods_keep_same_current_authority(monkeypatch, reverse):
    base, visual = definitions()
    a, b = (base, visual) if reverse else (visual, base)
    item = checker(monkeypatch, a, b)
    assert item.current()["requirements"]


@pytest.mark.parametrize(
    "field",
    ["interpretation", "source_refs", "applicability_conditions", "examination", "reference"],
)
def test_method_switch_cannot_hide_medical_or_source_changes(monkeypatch, field):
    base, visual = definitions()
    changed = copy.deepcopy(base)
    changed[field] = (
        []
        if field == "source_refs"
        else {"code": "WRONG"}
        if field == "reference"
        else "unauthorized change"
    )
    with pytest.raises(ValueError, match="upstream_definition_changed"):
        checker(monkeypatch, visual, changed).current()


@pytest.mark.parametrize("change", ["none", "modality", "extra", "other_target"])
def test_same_comparison_used_for_historical_plan_recovery(change):
    from packages.evidence_reuse.requirements import align_acquisition_requirements

    base, visual = definitions()
    old = [
        dict(
            id="criterion:VCDR_GE_07",
            definition=visual,
            input_modalities=["SLO"],
            basis_kind="qualitative",
        )
    ]
    current = [dict(old[0], definition=base, basis_kind="quantitative")]
    original = copy.deepcopy(old)
    if change == "modality":
        current[0]["input_modalities"] = ["OCT"]
    elif change == "extra":
        current.append(copy.deepcopy(current[0]))
    elif change == "other_target":
        old[0]["id"] = current[0]["id"] = "criterion:OTHER"
    if change == "none":
        assert align_acquisition_requirements(old, current) == original
        assert old == original and current[0]["definition"] == base
    else:
        with pytest.raises(ValueError, match="upstream_definition_changed"):
            align_acquisition_requirements(old, current)

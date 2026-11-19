"""Large actual-style OCT layouts stay lossless under the original 1000-char wire."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical, digest
from packages.multimodal_research.review_context import VERSION, context_values, supplementary
from packages.multimodal_research.stage_failure import unstarted_phase


@pytest.mark.parametrize(
    "value",
    [
        {"tiles": [{"slice": i, "box": [i, 0, i + 1, 32]} for i in range(128)]},
        {"text": '\\"中文\n' * 1200},
    ],
)
def test_fragments_losslessly_reconstruct_original_with_bounded_values(value):
    original = canonical(
        dict(assessment_id="check", part="display", context=VERSION, data=value)
    ).decode()
    fragments = context_values("check", "display", value)
    assert len(fragments) > 1
    rows = [json.loads(text) for _, _, text in fragments]
    assert all(len(text) <= 1000 for _, _, text in fragments)
    assert "".join(r["fragment"] for r in rows) == original
    assert all(r["checksum"] == digest(original) and r["count"] == len(rows) for r in rows)
    assert [r["index"] for r in rows] == list(range(len(rows)))
    context = dict(
        unit_basis="declared",
        source_limits=[],
        transform=value,
        numeric=None,
        interpretation_limits={},
    )
    obs = supplementary({"check": context})
    assert all(len(o.value) <= 1000 and o.unit == "context_only" for o in obs)
    assert len(set(o.reference.code for o in obs)) == len(obs)


def test_small_context_keeps_exact_old_encoding():
    old = canonical(dict(assessment_id="a", part="p", context=VERSION, data={"x": 1})).decode()
    assert context_values("a", "p", {"x": 1}) == [(VERSION, None, old)]


def test_registered_unstarted_requires_identity_and_no_dispatch(tmp_path):
    protocol = dict(input="input", visible="visible", task="task")
    req = SimpleNamespace(
        input_checksum="input",
        visible_checksum="visible",
        task=SimpleNamespace(checksum="task"),
        scope=SimpleNamespace(case_code="case"),
        policy_ref=SimpleNamespace(checksum=digest(([], {}, None))),
    )
    records = {}
    c = SimpleNamespace(
        protocol=protocol,
        config={},
        base=SimpleNamespace(binding={"case_code": "case"}, ledger=[]),
        contexts=SimpleNamespace(read_research_run=lambda _: SimpleNamespace(request=req)),
        sessions=SimpleNamespace(read=lambda _: None),
        phase_path=lambda _: tmp_path,
    )
    docs = SimpleNamespace(read=records.get)
    assert unstarted_phase(c, None, "run", docs)
    records["dynamic-phase-input"] = ({"parent": "existing"},)
    assert not unstarted_phase(c, None, "run", docs)
    records.clear()
    records["journal:0"] = ({"phase": "reserved"},)
    assert not unstarted_phase(c, None, "run", docs)
    records.clear()
    req.input_checksum = "changed"
    with pytest.raises(ValueError, match="stage_phase_binding"):
        unstarted_phase(c, None, "run", docs)


def test_repair_delta_does_not_allow_provider_or_medical_changes():
    from packages.multimodal_research.context_recovery import check_delta

    path = "packages/multimodal_research/review_context.py"
    assert check_delta({path: "old"}, {path: "new"}) == [path]
    with pytest.raises(ValueError, match="uncovered_delta"):
        check_delta({path: "old"}, {path: "new", "configs/provider.json": "uncovered"})
    with pytest.raises(ValueError, match="uncovered_delta"):
        check_delta({path: "old"}, {})


@pytest.mark.parametrize("confirmed", [False, True])
def test_repair_never_reissues_started_review(tmp_path, monkeypatch, confirmed):
    from packages.multimodal_research import context_recovery, stage_failure, stages

    acquisition = dict(target="old", accepted=True)
    review = dict(target=None, accepted=True) if confirmed else None
    monkeypatch.setattr(
        context_recovery,
        "read_amendment",
        lambda _: dict(phases=[dict(target="old")], review_run_id="run"),
    )
    monkeypatch.setattr(stages, "confirmed_phase", lambda c, t: acquisition if t else review)
    monkeypatch.setattr(stage_failure, "unstarted_phase", lambda *a: False)
    c = SimpleNamespace(
        output=tmp_path,
        expand_definitions=lambda **kw: None,
        documents=lambda _: object(),
        finish=lambda p: p,
    )
    if confirmed:
        assert context_recovery.execute(c) == [acquisition, review]
    else:
        with pytest.raises(ValueError, match="started_or_unknown"):
            context_recovery.execute(c)

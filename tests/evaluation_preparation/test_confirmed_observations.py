"""Confirmed observations are reusable facts, not permission to replay UNKNOWN."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.confirmed_observations import ConfirmedObservations


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "not_text_only",
        "vision_targets",
        "already_continued",
        "known_parent",
        "empty",
        "owner_reject",
    ],
)
def test_explicit_one_time_confirmed_fact_continuation(tmp_path, monkeypatch, fault):
    config = configuration()
    (tmp_path / "entry.json").write_text(
        json.dumps(
            dict(
                provider=config.model_dump(mode="json"),
                confirmed_observations_only=fault == "already_continued",
                known_only=fault == "known_parent",
            )
        )
    )
    (tmp_path / "result.json").write_text(json.dumps(dict(phases=[])))
    c = SimpleNamespace(
        base=SimpleNamespace(known_only=fault != "not_text_only", previous=tmp_path),
        targets=("vision",) if fault == "vision_targets" else (),
        config=config,
    )
    calls = []

    def owner():
        calls.append("owner")
        if fault == "owner_reject":
            raise ValueError("multimodal_previous_unknown")

    c.previous_attempts = owner
    values = {} if fault == "empty" else {"check:one": "confirmed synthetic value"}
    monkeypatch.setattr(
        ConfirmedObservations, "read", lambda self: (values, [{"receipt": "synthetic"}])
    )
    if fault:
        with pytest.raises(ValueError):
            ConfirmedObservations(c)
    else:
        reader = ConfirmedObservations(c)
        assert calls == ["owner"] and reader.check() == values
        assert not reader.report()["unknown_vision_replayed"]
        assert reader.report()["new_vision_requests"] == 0
        values["check:two"] = "not originally confirmed"
        with pytest.raises(ValueError, match="confirmed_observations_changed"):
            reader.check()


def test_confirmed_reader_rejects_review_as_original_vision():
    reader = object.__new__(ConfirmedObservations)
    reader.campaign = SimpleNamespace(base=object())
    reader.result = dict(phases=[dict(target=None, accepted=True)])
    with pytest.raises(ValueError, match="vision_parent_required"):
        reader.read()


def test_hypothesis_overview_uses_separate_scores_and_shared_counts():
    from pathlib import Path

    script = Path("apps/templates/evidence_tree_reader_v3.js").read_text()
    assert (
        "hypotheses.map(h=>[h.hypothesis.title+' · 研究分',h.summary.confidence.value])" in script
    )
    assert "个共享检查单元" in script
    assert script.index("分别评阅的研究命题") < script.index("检查清单")
    assert digest(script)

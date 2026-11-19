"""Method change preserves owner-bound per-item wording, unlike root upgrades."""

import json
from types import SimpleNamespace

import pytest

from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from tests.evaluation_preparation.test_review_method_window import old_config


@pytest.mark.parametrize("version", ["1.35.0", "1.40.0", "1.30.0"])
def test_real_constructor_preserves_wording_only_for_explicit_method(
    monkeypatch, tmp_path, version
):
    from packages.multimodal_research import root_review_window

    # This test isolates constructor projection, not source/current authority.
    # Original owner correctness is tested separately with persistent material.
    (tmp_path / "result.json").write_text(json.dumps({"complete": True}))
    clarification = {"assessment_ids": ["x"], "explanation": "retained source-bound wording"}
    previous_reader = object()
    base = SimpleNamespace(
        confirmed_review_reader=previous_reader,
        review_clarification=clarification,
        review_target_ids=("x",),
    )
    parent = SimpleNamespace(base=base, target=object(), config=old_config())
    plan = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="review",
        facts=str(tmp_path),
        root_direction_review={"question": {"root": "p"}},
        review_target_ids=["x", "y"],
        text_review_method={"profile": "relay-retained-direction-v1"},
    )
    if version not in {"1.35.0", "1.40.0"}:
        plan.pop("text_review_method")
    monkeypatch.setattr(root_review_window, "check", lambda *args: None)
    monkeypatch.setattr(root_review_window, "configuration_for", lambda *args: parent.config)
    monkeypatch.setattr(
        MultimodalCampaign,
        "__init__",
        lambda self, base, target, **kwargs: setattr(self, "base", base),
    )
    current = RenewedCampaign(parent, tmp_path / "new", plan)
    assert current.base is not base
    assert current.base.confirmed_review_reader is None
    assert current.base.review_target_ids == ("x", "y")
    assert current.base.review_clarification == (
        clarification if version in {"1.35.0", "1.40.0"} else None
    )
    assert base.review_clarification == clarification
    assert base.confirmed_review_reader is previous_reader and base.review_target_ids == ("x",)

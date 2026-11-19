"""Confirmed subset material can close a gap without overwriting request history."""

import json
from pathlib import Path

import pytest

from packages.llm_provider.transport import Reply
from packages.qualitative_analysis import runner
from packages.qualitative_analysis.comparison import verified_result
from packages.qualitative_analysis.storage import read
from packages.qualitative_analysis.supplement import supplement_confirmed
from tests.qualitative_analysis.test_analyzer import Transport
from tests.qualitative_analysis.test_analyzer import prepared as old_fixture
from tests.qualitative_analysis.test_integration import CONFIG, v2


def test_same_contract_subset_and_nonreplacement(tmp_path):
    packet, _ = old_fixture.__wrapped__(tmp_path)
    scope = dict(mode="archive_description", component_ids=["C1", "C2"], background_ids=[])
    current = tmp_path / "current"
    runner.plan(packet, current, CONFIG, scope)
    good = Reply(content=json.dumps(v2()).encode(), api_requests=1, response_confirmed=True)

    class Partial(Transport):
        def send(self, *args, **kwargs):
            super().send(*args, **kwargs)
            return (
                good
                if len(self.calls) == 1
                else Reply(error="transport", api_requests=1, response_confirmed=None)
            )

    first = runner.run(current, live=True, transport=Partial())
    old_bytes = Path(first["json"]).read_bytes()
    additional = tmp_path / "supplement"
    runner.plan(
        packet,
        additional,
        CONFIG,
        dict(mode="explicit_selection", component_ids=["C2"], background_ids=[]),
    )
    second = runner.run(additional, live=True, transport=Transport(good))
    result = supplement_confirmed(first["json"], second["json"], tmp_path / "assembled")
    v = verified_result(result["json"])
    assert result["case_visual_completed"] and result["supplemented_components"] == ["C2"]
    assert v["execution"]["cumulative_api_requests"] == 3
    assert v["execution"]["historical_unknown_request_count"] == 1
    assert not v["coverage"]["all_planned_jobs_completed"]  # original attempt is preserved
    assert Path(first["json"]).read_bytes() == old_bytes
    assert v["components"][0] == read(first["json"])["components"][0]
    assert (
        len(list((tmp_path / "assembled/views").glob("*.png"))) == 4
    )  # 1 fundus + 3 sampled OCT views
    assert "新版独立补评已确认" in Path(result["html"]).read_text()
    with pytest.raises(ValueError, match="would_replace_confirmed"):
        supplement_confirmed(second["json"], second["json"], tmp_path / "invalid")
    with pytest.raises(ValueError, match="not_confirmed"):
        supplement_confirmed(first["json"], first["json"], tmp_path / "invalid2")

"""Current checks finish before returning qualification; nothing survives dispatch."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.campaign import MultimodalCampaign


@pytest.mark.parametrize("method", ["guard", "guard_definition_materials"])
@pytest.mark.parametrize("failure", [None, "deadline", "source_changed"])
def test_return_checks_and_time_finish_before_qualification(monkeypatch, method, failure):
    from packages.dataset_adapters import current_read_scope
    from packages.evidence_details import accepted_read_scope

    calls = []
    state = {"expired": False}

    @contextmanager
    def facts():
        calls.append("fact_entry")
        yield
        calls.append("fact_return")
        if failure == "source_changed":
            raise ValueError("historical_source_changed")
        state["expired"] = failure == "deadline"

    @contextmanager
    def inputs():
        calls.append("input_entry")
        yield
        calls.append("input_return")

    monkeypatch.setattr(accepted_read_scope, "historical_fact_read_scope", facts)
    monkeypatch.setattr(current_read_scope, "_projection_visible_reads", inputs)
    campaign = object.__new__(MultimodalCampaign)
    campaign.base = SimpleNamespace(check=lambda: calls.append("original_sources"))

    def window():
        calls.append("window")
        if state["expired"]:
            raise ProviderError("deadline")

    campaign.guard_window = window
    campaign.guard_control = lambda **kw: calls.append(("original_control", kw))
    if failure is None:
        getattr(campaign, method)(dispatch=True)
    elif failure == "deadline":
        with pytest.raises(ProviderError, match="deadline"):
            getattr(campaign, method)(dispatch=True)
    else:
        with pytest.raises(ValueError, match="historical_source_changed"):
            getattr(campaign, method)(dispatch=True)
    assert calls[:6] == [
        "window",
        "fact_entry",
        "input_entry",
        "original_sources",
        ("original_control", {"dispatch": True}),
        "input_return",
    ]
    assert calls[6] == "fact_return"
    if failure != "source_changed":
        assert calls[7:] == ["window"]

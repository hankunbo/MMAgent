"""Public full-run dispatch preparation must use the original technical closure."""

from types import SimpleNamespace

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research import campaign as module
from packages.multimodal_research import stage_failure


def bare_campaign(tmp_path, monkeypatch):
    value = object.__new__(module.MultimodalCampaign)
    value.output = tmp_path
    value.targets = ("active:test",)
    value.base = SimpleNamespace(manifest=SimpleNamespace(dataset="fixture"))
    value.config = SimpleNamespace(transport=SimpleNamespace(api_key_env="MMAGENT_TEST_CLOSURE"))
    value.freeze = lambda: None
    value.guard = lambda: None
    value.acquisition_exhausted = lambda: False
    value.active_review_reserved = lambda: False
    monkeypatch.setattr(
        module.importlib.util,
        "spec_from_file_location",
        lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)),
    )
    monkeypatch.setattr(
        module.importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(load_secret=lambda: "offline-placeholder"),
    )
    return value


@pytest.mark.parametrize(
    ("where", "error", "expected_stage"),
    [
        ("review", ProviderError("deadline"), "review"),
        ("acquisition", ValueError("source_unavailable"), "acquisition"),
        ("export", OSError("report write failed"), "export"),
    ],
)
def test_full_run_closes_preparation_and_export_errors(
    tmp_path, monkeypatch, where, error, expected_stage
):
    value = bare_campaign(tmp_path, monkeypatch)
    monkeypatch.setenv("MMAGENT_TEST_CLOSURE", "previous-value")
    dispatched, closures = [], []

    def phase(target):
        dispatched.append(target)
        if where == ("acquisition" if target else "review"):
            raise error
        return dict(accepted=True, error=None, target=target)

    def finish(phases):
        if where == "export":
            raise error
        pytest.fail("must not finish after failed preparation")

    def close(current, actual, *, stage):
        assert current is value and actual is error
        closures.append(stage)
        return dict(complete=False, stage=stage, new_score=False)

    value.execute_phase, value.finish = phase, finish
    monkeypatch.setattr(stage_failure, "close", close)
    assert value.execute() == dict(complete=False, stage=expected_stage, new_score=False)
    assert closures == [expected_stage]
    assert dispatched == (["active:test"] if where == "acquisition" else ["active:test", None])
    assert module.os.environ["MMAGENT_TEST_CLOSURE"] == "previous-value"


def test_full_run_success_does_not_install_failure(tmp_path, monkeypatch):
    value = bare_campaign(tmp_path, monkeypatch)
    monkeypatch.delenv("MMAGENT_TEST_CLOSURE", raising=False)
    value.execute_phase = lambda target: dict(accepted=True, error=None, target=target)
    value.finish = lambda phases: dict(complete=True, phases=phases)
    monkeypatch.setattr(stage_failure, "close", lambda *a, **kw: pytest.fail("unexpected closure"))
    result = value.execute()
    assert result["complete"] and len(result["phases"]) == 2
    assert "MMAGENT_TEST_CLOSURE" not in module.os.environ

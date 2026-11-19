"""Public fixture regression only; no credentials, API, patients or databases."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from packages.grounding.models import canonical
from packages.llm_provider.transport import Reply
from tests.evaluation_preparation.test_partial_observation import detail

PATH = Path("docs/acceptance/client-binding-Gx4FJ4sA/public_vision_route_probe.py")
PROFILE = "configs/agents/multimodal_relay_environment_gpt6_review_curl_v1.json"


def module():
    spec = importlib.util.spec_from_file_location("public_route_probe_test", PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("size", [32, 512])
def test_only_route_changes_and_actual_vision_projection(size):
    probe = module()
    old = json.loads(
        Path("configs/agents/multimodal_relay_local_gpt6_review_curl_v1.json").read_bytes()
    )
    new = json.loads(Path(PROFILE).read_bytes())
    assert [k for k in old if old[k] != new[k]] == ["network_route"]
    config, _ = probe.selected_provider(PROFILE, "gpt-6-sol")
    wire, binding, code = probe.build(config, "RUN:PUBLIC_ROUTE_TEST", image_size=size)
    text = json.dumps(wire)
    assert code in text and "Describe dominant visible color" in text
    assert "data:image/png;base64," in text
    assert "vision-partial-multimodal-1.3.0" == next(
        r.prompt_version for r in config.roles if r.role == "vision"
    )
    assert binding["location"] == (0, 0, size, size)
    assert wire["model"] == "gpt-6-sol"


@pytest.mark.parametrize("unknown", [True, False])
def test_explicit_run_no_retries_and_stop_on_unknown(tmp_path, monkeypatch, unknown):
    probe = module()
    calls, candidates = [], []
    original_build = probe.build

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        candidates.append(
            dict(
                schema_version="visual-observation-content/1.0.0",
                binding_code=result[2],
                status="observed",
                finding="red square",
                partial=detail(),
            )
        )
        return result

    class Transport:
        def __init__(self, **kwargs):
            pass

        def send(self, config, body, **kwargs):
            index = len(calls)
            calls.append(body)
            assert config.max_retries == 0
            if unknown:
                return Reply(error="transport", api_requests=1, response_confirmed=None)
            return Reply(
                content=canonical(candidates[index]),
                api_requests=1,
                response_confirmed=True,
                finish_reason="stop",
            )

        def observation(self, reply):
            return dict(http_status=None if unknown else 200)

    monkeypatch.setattr(probe, "build", build)
    monkeypatch.setattr(probe, "secret", lambda: "synthetic-test-token")
    monkeypatch.setattr(probe, "HTTPTransport", Transport)
    monkeypatch.setattr(probe, "route_opener", lambda *a, **k: None)
    output = tmp_path / "evidence"
    monkeypatch.setattr(
        sys, "argv", ["probe", "--provider-config", PROFILE, "--output", str(output), "--execute"]
    )
    assert probe.main() == (2 if unknown else 0)
    summary = json.loads((output / "summary.json").read_bytes())
    assert len(calls) == summary["actual_requests"] == (1 if unknown else 3)
    assert summary["retries"] == 0 and summary["patient_data"] is False
    assert summary["unknown_count"] == int(unknown)
    with pytest.raises(FileExistsError):
        probe.main()
    assert len(calls) == (1 if unknown else 3)

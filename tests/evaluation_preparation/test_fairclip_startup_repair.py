import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cwd", ["repository", "other"])
def test_absolute_entrypoint_does_not_shadow_stdlib_platform(tmp_path, cwd):
    entry = str(ROOT / "apps/fairclip_tool_v2.py")
    code = (
        "import sys,runpy,json; "
        f"sys.path.insert(0, {str(ROOT / 'apps')!r}); "
        f"runpy.run_path({entry!r}, run_name='entry_import_probe'); "
        "import platform,apps.platform; "
        "print(json.dumps({'platform': platform.__file__, 'system': platform.system(), "
        "'app': apps.platform.__file__}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT if cwd == "repository" else tmp_path,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert not Path(data["platform"]).is_relative_to(ROOT)
    assert data["system"] == "Linux"
    assert Path(data["app"]) == ROOT / "apps/platform/__init__.py"


@pytest.mark.parametrize("revision", ["../escape", "", "a/b", "x" * 25])
def test_repair_revision_cannot_escape_or_register_invalid_identity(revision, monkeypatch):
    from packages.evaluation_preparation import fairclip_execution as module
    from packages.evaluation_preparation.dynamic_model import read_model

    monkeypatch.setattr(
        module, "register", lambda _: pytest.fail("must reject before registration")
    )
    with pytest.raises(ValueError, match="fairclip_execution_revision"):
        module.execute("data_00002.npz", revision=revision)
    with pytest.raises(ValueError, match="fairclip_execution_revision"):
        read_model(None, None, revision=revision)


def test_model_increment_does_not_have_a_vision_phase_and_keeps_prior_budget(tmp_path):
    from types import SimpleNamespace

    from packages.evaluation_preparation.model_repair_campaign import ModelRepairCampaign

    campaign = object.__new__(ModelRepairCampaign)
    campaign.output = tmp_path
    observations = {"target": "original-current-qualified-observation"}
    campaign.previous = SimpleNamespace(
        prior=lambda: {"reservations": 18},
        usage=lambda: 4,
        accepted_observations=lambda: observations,
    )
    assert campaign.paths() == [tmp_path / "review"]
    assert campaign.prior()["reservations"] == 22
    assert campaign.accepted_observations() is observations
    assert campaign.version != "13-dynamic-case-v1"


def test_old_launcher_is_preserved_for_historical_freezes():
    import hashlib

    assert hashlib.sha256((ROOT / "apps/fairclip_tool.py").read_bytes()).hexdigest() == (
        "046d4cc2793eab86474c63e4a52261f8beac88edde4e3a35a0ee9437c8580ece"
    )


def test_existing_logs_are_checksum_bound_and_diagnostics_do_not_leak(tmp_path):
    import hashlib

    from packages.tool_execution.diagnostics import explain_local_failure

    attempt, identity = "a" * 32, "step_" + "b" * 64
    path = tmp_path / attempt / identity / "stderr.log"
    path.parent.mkdir(parents=True)
    raw = b"secret-must-not-escape\nAttributeError: module 'platform' has no attribute 'system'"
    path.write_bytes(raw)
    step = dict(
        step_id=identity,
        result=dict(status="FAILED"),
        status_history=["QUEUED", "RUNNING", "FAILED"],
        stderr_checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
    )
    value = explain_local_failure(tmp_path, attempt_id=attempt, step=step)
    assert value["category"] == "python_stdlib_shadowing" and value["log_verified"]
    assert "secret-must-not-escape" not in json.dumps(value)
    assert value["retry_authorized"] is False
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="diagnostic_log_checksum"):
        explain_local_failure(tmp_path, attempt_id=attempt, step=step)


def test_prelaunch_cancellation_is_not_model_failure(tmp_path):
    from packages.tool_execution.diagnostics import explain_local_failure

    value = explain_local_failure(
        tmp_path,
        attempt_id="a" * 32,
        step=dict(
            step_id="step_" + "b" * 64,
            result=dict(status="CANCELLED"),
            status_history=["QUEUED", "CANCELLED"],
        ),
    )
    assert value["category"] == "not_started" and not value["process_started"]


def test_success_diagnostic_is_not_reported_as_failure(tmp_path):
    from packages.tool_execution.diagnostics import explain_local_failure

    value = explain_local_failure(
        tmp_path,
        attempt_id="a" * 32,
        step=dict(
            step_id="step_" + "b" * 64,
            result=dict(status="SUCCEEDED"),
            status_history=["QUEUED", "RUNNING", "SUCCEEDED"],
            exit_code=0,
        ),
    )
    assert value["category"] == "none" and value["stage"] == "completed"
    assert value["process_started"] and not value["retry_authorized"]

"""Actual harmless processes. Fake GPU is scheduler coverage, never CUDA evidence."""

import json
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest, validate
from packages.grounding.synthetic import ref
from packages.object_store import checksum
from packages.tool_execution.models import ExecutionError, ExecutionPolicy
from packages.tool_execution.trusted_models import (
    ReviewedFile,
    TrustedCapabilities,
    TrustedLocalPolicy,
    TrustedToolReview,
)
from packages.tool_execution.trusted_process import TrustedLocalProcessRunner, path_checked

UUID = "GPU-00000000-0000-0000-0000-000000000001"
SCRIPT = """import os, sys, time
mode, output = sys.argv[1:3]
if mode == "environment":
 assert not any(k in os.environ for k in ("DATABASE_URL", "API_KEY", "PGPASSWORD", "PYTHONPATH"))
 for k in ("HOME", "TMPDIR", "TORCH_HOME", "HF_HOME", "CUDA_CACHE_PATH"):
  assert os.environ[k].startswith("/opt/data/test/")
 assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
 assert len(os.sched_getaffinity(0)) == 1
 open(output, "x").write(str(sorted(os.environ)))
elif mode in ("timeout", "cancel"):
 print("ready", flush=True)
 time.sleep(30)
elif mode == "logs":
 while True: print("x"*8192, flush=True)
elif mode == "finite_logs": print("x" * 200000)
elif mode == "nonzero": sys.exit(7)
elif mode == "fork":
 pid = os.fork()
 if pid == 0:
  time.sleep(30)
  os._exit(0)
 print(pid, flush=True)
else: open(output, "x").write("ok")
"""


def pinned(path):
    return ReviewedFile(
        path=str(path), checksum=checksum(path.read_bytes()), byte_size=path.stat().st_size
    )


class FakeGPU:
    def __init__(self):
        self.busy, self.missing, self.unavailable = False, False, False

    def read(self, policy):
        if self.unavailable:
            raise ExecutionError("gpu_observation_unavailable")
        return ({} if self.missing else {UUID: (24000, 0)}), ([(UUID, 999999)] if self.busy else [])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # pytest's standard tmpdir is explicitly redirected under /opt/data/test.
    assert tmp_path.is_relative_to(Path("/opt/data/test"))
    from packages.tool_execution import trusted_process

    monkeypatch.setattr(trusted_process, "LOCK_ROOT", tmp_path / "project-locks")
    environment = tmp_path / "environment"
    binary = environment / "bin/python3.12"
    binary.parent.mkdir(parents=True)
    shutil.copyfile(Path(sys.executable).resolve(), binary)
    binary.chmod(0o500)
    stdlib = environment / "lib/python3.12"
    stdlib.mkdir(parents=True)
    original = Path(sys.executable).resolve().parent.parent / "lib/python3.12"
    shutil.copyfile(original / "os.py", stdlib / "os.py")
    shutil.copytree(
        original / "encodings", stdlib / "encodings", ignore=shutil.ignore_patterns("__pycache__")
    )
    code = tmp_path / "code"
    code.mkdir()
    script = code / "harmless.py"
    script.write_text(SCRIPT)
    record = tmp_path / "review.md"
    record.write_text("Harmless fixture source shown in test. No model, no CUDA validation.\n")
    files = tuple(
        pinned(p) for root in (environment, code) for p in sorted(root.rglob("*")) if p.is_file()
    )
    review = TrustedToolReview(
        implementation_ref=ref("implementation", "IMPLEMENTATION:TEST:TRUSTED"),
        manifest_checksum=digest("manifest"),
        allowlist_checksum=digest("allowlist"),
        review_record=pinned(record),
        environment_root=str(environment),
        executable=pinned(binary),
        entrypoint=pinned(script),
        files=files,
        inventory_roots=(str(environment), str(code)),
    )
    policy = TrustedLocalPolicy(
        reference=ref("policy", "POLICY:TEST:TRUSTED"),
        purpose="shadow",
        allowed_plan_checksums=(digest("plan"),),
        reviews=(review,),
        gpu_pool=(UUID,),
        selected_gpu=UUID,
        cpu_count=1,
        ram_budget_bytes=268435456,
        vram_budget_mib=1024,
        timeout_seconds=2,
        nvidia_smi=pinned(binary),
    )
    runner = TrustedLocalProcessRunner(tmp_path / "attempts", policy)
    runner.observation = FakeGPU()
    step = SimpleNamespace(
        implementation_ref=review.implementation_ref,
        step_id="step_" + "a" * 64,
        deployment=SimpleNamespace(
            manifest=SimpleNamespace(checksum=review.manifest_checksum),
            allowlist=SimpleNamespace(checksum=review.allowlist_checksum),
        ),
    )
    return SimpleNamespace(
        runner=runner,
        step=step,
        policy=policy,
        review=review,
        script=script,
        config=SimpleNamespace(cpu_count=1, max_output_bytes=1048576),
        root=tmp_path,
        lock_root=trusted_process.LOCK_ROOT,
    )


def launch(f, mode="environment", *, cancel=None, on_start=None):
    with f.runner.slot():
        base, root, _ = f.runner.prepare("a" * 32, f.step, f.config)
        result = f.runner.run(
            base,
            root,
            [f.review.executable.path, f.review.entrypoint.path, mode, str(root / "out/result")],
            f.config,
            0.15 if mode == "timeout" else 2,
            cancel or threading.Event(),
            on_start or (lambda p: None),
        )
    return result, base


def test_actual_harmless_minimal_environment_and_observed_identity(setup, monkeypatch):
    for key in ("DATABASE_URL", "API_KEY", "PGPASSWORD", "PYTHONPATH"):
        monkeypatch.setenv(key, "DO_NOT_FORWARD")
    identities = []
    result, base = launch(setup, on_start=identities.append)
    assert result["exit_code"] == 0, result
    assert result["error"] is None
    assert set(identities[0]) == {"pid", "start_ticks", "boot_id"}
    assert "DO_NOT_FORWARD" not in (base / "launch.json").read_text()
    assert json.loads((base / "trusted-cleanup.json").read_text())["group_cleanup_observed"]
    assert not (setup.lock_root / "active.json").exists()
    assert (base / "root/out/result").read_text()


@pytest.mark.parametrize(
    "mode,error",
    [
        ("timeout", "timeout"),
        ("nonzero", "process_nonzero"),
        ("logs", "log_size_limit"),
        ("finite_logs", "log_size_limit"),
        ("fork", None),
    ],
)
def test_actual_timeout_logs_nonzero_and_group_cleanup(setup, mode, error):
    result, base = launch(setup, mode)
    assert result["error"] == error, result
    assert len(result["stdout"]) <= setup.policy.log_bytes
    assert json.loads((base / "trusted-cleanup.json").read_text())["group_cleanup_observed"]


def test_actual_cancel(setup):
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        result, _ = launch(setup, "cancel", cancel=cancel)
    finally:
        timer.join()
    assert result["error"] == "cancelled" and result["exit_code"] != 0


def test_review_digest_change_rejected_even_same_size(setup):
    setup.script.write_text(SCRIPT.replace('"ready"', '"drift"'))
    with pytest.raises(ExecutionError, match="digest_drift"):
        launch(setup)
    assert not list(setup.runner.workspace.glob("*/*/launch.json"))


def test_new_code_in_environment_is_inventory_drift(setup):
    (Path(setup.review.environment_root) / "unreviewed.pth").write_text("import evil")
    with pytest.raises(ExecutionError, match="inventory_drift"):
        launch(setup)


def test_review_record_drift_does_not_reapprove(setup):
    Path(setup.review.review_record.path).write_text("approved")
    with pytest.raises(ExecutionError, match="resource_size_drift"):
        launch(setup)


@pytest.mark.parametrize(
    "path", ["/tmp/new", "/etc/passwd", "/opt/data/test/../etc", "/opt/data/test"]
)
def test_path_bounds(path):
    with pytest.raises(ExecutionError):
        path_checked(path)


def test_symlink_rejected(setup):
    link = setup.root / "link"
    link.symlink_to(setup.script)
    with pytest.raises(ExecutionError, match="path_rejected"):
        path_checked(link)


@pytest.mark.parametrize(
    "changes",
    [
        {"purpose": "decisive_research"},
        {"purpose": "clinical_deployment"},
        {"backend": "linux_chroot"},
        {"synthetic_execution": True},
        {"approved": True},
        {"max_concurrent": 2},
        {"environment": {"API_KEY": "secret"}},
        {"capabilities": {"network": "deny"}},
        {"decisive_evidence": True},
        {"selected_gpu": "cuda:0"},
    ],
)
def test_strict_policy_rejects_unsafe_extensions(setup, changes):
    with pytest.raises(ValueError):
        validate(TrustedLocalPolicy, {**setup.policy.model_dump(mode="json"), **changes})


def test_old_policy_cannot_be_mislabelled(setup):
    with pytest.raises(ValueError):
        validate(ExecutionPolicy, setup.policy)
    assert TrustedCapabilities().network == "not_guaranteed"


def test_project_lock_across_workspaces_and_unknown_start(setup):
    other = TrustedLocalProcessRunner(setup.root / "another", setup.policy)
    with setup.runner.slot():
        with pytest.raises(ExecutionError, match="concurrency_limit"), other.slot():
            pass
    (setup.lock_root / "active.json").write_text("{}")
    with pytest.raises(ExecutionError, match="quarantined_no_replay"), other.slot():
        pass


@pytest.mark.parametrize(
    "field,reason",
    [
        ("busy", "gpu_busy"),
        ("missing", "gpu_uuid_unavailable"),
        ("unavailable", "gpu_observation_unavailable"),
    ],
)
def test_fake_gpu_scheduling_only(setup, field, reason):
    setattr(setup.runner.observation, field, True)
    with pytest.raises(ExecutionError, match=reason):
        launch(setup)


def test_unconfirmed_cleanup_quarantines_and_prevents_retry(setup, monkeypatch):
    actual = setup.runner._cleanup

    def unknown(proc, observed):
        assert actual(proc, observed)  # Clean the actual test child, then simulate lost evidence.
        return False

    monkeypatch.setattr(setup.runner, "_cleanup", unknown)
    before = time.monotonic()
    with pytest.raises(ExecutionError, match="cleanup_unconfirmed"):
        launch(setup)
    assert time.monotonic() - before < 15
    with pytest.raises(ExecutionError, match="quarantined_no_replay"), setup.runner.slot():
        pass


def test_duplicate_attempt_never_reexecutes(setup):
    result, _ = launch(setup)
    assert result["exit_code"] == 0
    with pytest.raises(FileExistsError):
        launch(setup)


@pytest.mark.parametrize("used", [3638, 23999])
def test_shared_gpu_keeps_observation_without_empty_card_or_vram_gate(setup, used):
    old = setup.policy.model_dump_json()
    setup.runner.policy = validate(
        TrustedLocalPolicy,
        setup.policy.model_copy(update={"version": "trusted-local-policy/1.1.0"}),
    )
    setup.runner.observation.read = lambda _: ({UUID: (24000, used)}, [(UUID, 999999)])
    assert setup.runner.gpu_ready()["compute_processes"] == [(UUID, 999999)]
    result, _ = launch(setup)
    assert result["exit_code"] == 0  # Harmless CPU fixture; not a CUDA capacity claim.
    assert TrustedLocalPolicy.model_validate_json(old).model_dump_json() == old
    assert digest(setup.runner.policy) != digest(setup.policy)


def test_shared_gpu_still_serializes_project_and_retains_nonzero_failure(setup):
    setup.runner.policy = validate(
        TrustedLocalPolicy,
        setup.policy.model_copy(update={"version": "trusted-local-policy/1.1.0"}),
    )
    setup.runner.observation.busy = True
    other = TrustedLocalProcessRunner(setup.root / "other-shared", setup.runner.policy)
    with setup.runner.slot():
        with pytest.raises(ExecutionError, match="concurrency_limit"), other.slot():
            pass
    result, _ = launch(setup, "nonzero")
    assert result["exit_code"] == 7


@pytest.mark.parametrize(
    "field,reason",
    [("missing", "gpu_uuid_unavailable"), ("unavailable", "gpu_observation_unavailable")],
)
def test_shared_gpu_still_requires_observed_device(setup, field, reason):
    setup.runner.policy = validate(
        TrustedLocalPolicy,
        setup.policy.model_copy(update={"version": "trusted-local-policy/1.1.0"}),
    )
    setattr(setup.runner.observation, field, True)
    with pytest.raises(ExecutionError, match=reason):
        setup.runner.gpu_ready()


def test_shared_gpu_does_not_release_observed_own_gpu_process(setup):
    setup.runner.policy = validate(
        TrustedLocalPolicy,
        setup.policy.model_copy(update={"version": "trusted-local-policy/1.1.0"}),
    )
    own = []
    setup.runner.observation.read = lambda _: ({UUID: (24000, 3638)}, own)
    with pytest.raises(ExecutionError, match="process_cleanup_unconfirmed"):
        launch(setup, on_start=lambda observed: own.append((UUID, observed["pid"])))
    with pytest.raises(ExecutionError, match="quarantined_no_replay"), setup.runner.slot():
        pass

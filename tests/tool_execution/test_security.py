"""Path/format/authority counterexamples and deterministic domain schemas."""

import json
from pathlib import Path
from threading import Event

import pytest

from packages.grounding.models import digest, validate
from packages.tool_execution import ExecutionError, StepResult, WorkflowResult
from packages.tool_execution.process import read_file
from packages.tool_execution.qc import grayscale_png
from packages.tool_execution.schema_tools import schemas
from tests.tool_admission.support import png
from tests.tool_execution.support import prepared, runtime_files
from tests.tool_execution.test_dependencies import TestLedger


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    return runtime_files(tmp_path_factory.mktemp("06d-security-runtime"))


@pytest.mark.parametrize(
    "variant", ["leaf_symlink", "parent_symlink", "fifo", "directory", "oversized", "traversal"]
)
def test_controlled_file_reads_reject_unsafe_paths(tmp_path, variant):
    source = tmp_path / "source"
    source.write_bytes(b"test")
    target = tmp_path / "target"
    if variant == "leaf_symlink":
        target.symlink_to(source)
    elif variant == "parent_symlink":
        target.symlink_to(tmp_path, target_is_directory=True)
        target = target / "source"
    elif variant == "fifo":
        import os

        os.mkfifo(target)
    elif variant == "directory":
        target.mkdir()
    elif variant == "oversized":
        target.write_bytes(b"x" * 20)
    else:
        target = tmp_path / ".." / tmp_path.name / "source"
    with pytest.raises((ExecutionError, OSError)):
        read_file(target, 10)


@pytest.mark.parametrize(
    "data", [b"", b"not png", png()[:-1], png() + b"extra", png()[:40] + b"corrupt"]
)
def test_full_output_decoder_rejects_corrupt_or_truncated_bytes(data):
    with pytest.raises(ExecutionError):
        grayscale_png(data, 100)


def test_schema_artifacts_are_frozen_and_strict():
    root = Path(__file__).resolve().parents[2] / "packages/tool_execution/schemas"
    for name, content in schemas().items():
        assert (root / name).read_text() == content
        assert json.loads(content)["additionalProperties"] is False


def test_probe_availability_is_separate_from_actual_enforcement(tmp_path, closure):
    p = prepared(tmp_path, mode="isolation", repository=TestLedger(), files=closure)
    capabilities = p.executor._runner.check_capabilities()
    assert capabilities["seccomp"] == "tested_at_each_launch_fail_closed"
    w = p.executor.execute(p.plan, "enforcement")
    assert w.status == "SUCCEEDED"
    assert w.steps[0].exit_code == 0


def test_unavailable_chroot_refuses_real_child_setup(tmp_path, closure):
    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    step = p.plan.document.steps[0]
    runner = p.executor._runner
    base, root, _ = runner.prepare("b" * 32, step, p.plan.document.config)
    # Real child setup tries an absent owned root; no command runs with relaxed policy.
    result = runner.run(
        base, root / "absent", list(step.argv), p.plan.document.config, 1, Event(), lambda _: None
    )
    assert result["exit_code"] == 125 and result["error"] == "isolation_setup_failed"


def test_runtime_and_source_replacement_after_copy_cannot_change_executed_bytes(tmp_path, closure):
    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    original = p.executor._runner.run

    def replaced(*args, **kwargs):
        p.script.write_text("raise RuntimeError('must never execute replaced source')")
        return original(*args, **kwargs)

    p.executor._runner.run = replaced
    result = p.executor.execute(p.plan, "race-after-copy")
    assert result.status == "SUCCEEDED"
    assert result.steps[0].copied_checksums[p.script.as_uri()] != digest(p.script.read_text())


def test_result_cannot_forge_clinical_approval_or_implementation(tmp_path, closure):
    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    w = p.executor.execute(p.plan, "strict-results")
    step = w.steps[0].model_dump(mode="json")
    step["result"]["decisive_evidence"] = True
    with pytest.raises(ValueError):
        validate(StepResult, step)
    step = w.steps[0].model_dump(mode="json")
    step["result"]["implementation_checksum"] = digest("foreign")
    with pytest.raises(ValueError):
        validate(StepResult, step)
    workflow = w.model_dump(mode="json")
    workflow["unstarted_steps"] = ["step_" + "a" * 64]
    with pytest.raises(ValueError):
        validate(WorkflowResult, workflow)


def test_input_replacement_after_copy_consumes_bound_original(tmp_path, closure):
    from packages.object_store import checksum

    p = prepared(tmp_path, mode="isolation", repository=TestLedger(), files=closure)
    originals = [(r.artifact.uri, r.artifact.checksum) for r in p.f.owners.records.values()]
    original_run = p.executor._runner.run

    def replace_inputs(*args, **kwargs):
        for uri, _ in originals:
            p.f.objects._path(uri).write_bytes(b"changed after copying")
        return original_run(*args, **kwargs)

    p.executor._runner.run = replace_inputs
    w = p.executor.execute(p.plan, "input-race")
    assert w.status == "SUCCEEDED"
    for uri, expected in originals:
        if uri in w.steps[0].logical_to_physical:
            assert checksum(Path(w.steps[0].logical_to_physical[uri]).read_bytes()) == expected


def test_world_writable_or_symlink_workspace_refuses(tmp_path, closure):
    from packages.tool_execution.process import LinuxProcessRunner

    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    directory = tmp_path / "untrusted-workspace"
    directory.mkdir()
    directory.chmod(0o777)
    with pytest.raises(ExecutionError, match="service_owned"):
        LinuxProcessRunner(directory, p.policy)
    link = tmp_path / "workspace-link"
    link.symlink_to(p.root / "attempts", target_is_directory=True)
    with pytest.raises(ExecutionError, match="symlink"):
        LinuxProcessRunner(link, p.policy)


def test_controller_death_kills_real_tool_leader(tmp_path, closure):
    import ctypes
    import os
    import signal
    import subprocess
    import sys
    import time

    from packages.tool_execution.process import write_copy

    p = prepared(tmp_path, mode="cancel", repository=TestLedger(), files=closure)
    step = p.plan.document.steps[0]
    base, root, _ = p.executor._runner.prepare("c" * 32, step, p.plan.document.config)
    write_copy(root, str(p.script), p.script.read_bytes())
    write_copy(root, "/in/input.png", png())
    spec = dict(
        base=str(base),
        root=str(root),
        workspace=str(p.root / "attempts"),
        policy=p.policy.model_dump(mode="json"),
        config=p.plan.document.config.model_dump(mode="json"),
        argv=[step.argv[0], str(p.script), "/in/input.png", "/out/result.png", "cancel"],
    )
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(spec))
    marker = base / "owned-child.json"
    code = """import json,sys
from pathlib import Path
from threading import Event
from packages.tool_execution.process import LinuxProcessRunner
from packages.tool_execution.models import ExecutionPolicy
from packages.tool_workflow import CompilerConfig
s=json.loads(Path(sys.argv[1]).read_text())
p=ExecutionPolicy.model_validate_json(json.dumps(s['policy']))
c=CompilerConfig.model_validate_json(json.dumps(s['config']))
r=LinuxProcessRunner(s['workspace'],p)
r.run(Path(s['base']),Path(s['root']),s['argv'],c,10,Event(),
      lambda proc: Path(s['base'],'owned-child.json').write_text(json.dumps(proc)))
"""
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    controller = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        env={},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert controller.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        child = json.loads(marker.read_text())["pid"]
        # Let the child drop privileges and enter the trusted fixture before killing its controller.
        time.sleep(0.2)
        controller.kill()
        controller.wait(timeout=5)
        deadline = time.monotonic() + 5
        while True:
            reaped, status = os.waitpid(child, os.WNOHANG)
            if reaped:
                assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
                child = None
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
        if child is not None:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("purpose", ["debug", "shadow"])
def test_registered_research_entry_keeps_formal_authorization_separate(tmp_path, closure, purpose):
    from packages.grounding.models import seal
    from packages.tool_admission import InvocationInput
    from packages.tool_admission.models import ResearchRequest
    from packages.tool_execution import ToolExecutionService
    from packages.tool_workflow import ResearchContext
    from tests.grounding.query_support import NOW
    from tests.tool_workflow.support import choose

    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    f = p.f
    candidate = f.mapping.candidates[0]
    step = next(
        s for s in p.plan.document.steps if s.implementation_ref == candidate.implementation_ref
    )
    request = seal(
        ResearchRequest,
        code="RESEARCH:06D:REGISTERED",
        version="1.0.0",
        purpose=purpose,
        compatibility_policy=f.policy,
        lock=dict(run_id=f.q.lock.run_id, catalog_ref=f.q.lock.catalog_ref),
        scope=f.q.scope,
        window_start=f.q.window_start,
        window_end=f.q.window_end,
        snapshot_at=NOW,
        satisfied_terms=f.q.satisfied_terms,
        artifacts=f.q.artifacts,
        candidate=candidate,
        inputs=[
            dict(
                port=i.port,
                asset_codes=[
                    a.asset_code
                    for a in f.q.artifacts
                    if a.artifact_ref == i.constraint.artifact_ref
                ],
            )
            for i in candidate.inputs
        ],
        invocation=InvocationInput(
            implementation_ref=candidate.implementation_ref,
            manifest_checksum=step.deployment.manifest.checksum,
            profile_checksum=step.deployment.profile.checksum,
            parameters={},
        ),
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    admitted = p.compiler.admission.admit_research(request)
    assert admitted.status == "preparation_only"
    plan = p.compiler.compile(
        ResearchContext(request=request, admission=admitted), choose(admitted)
    )
    policy = p.policy.model_copy(
        update={"purpose": purpose, "allowed_plan_checksums": (plan.checksum,)}
    )
    service = ToolExecutionService(
        compiler=p.compiler,
        catalogs=f.catalogs,
        objects=f.objects,
        qc=p.qc,
        repository=TestLedger(),
        policy=policy,
        workspace=p.root / "research-attempts",
    )
    with pytest.raises(ExecutionError, match="not_admitted"):
        service.execute(p.plan, "formal-denied")
    result = service.execute_bundle(plan.to_bundle_json(), "research")
    assert result.status == "SUCCEEDED" and result.decisive_evidence is False
    assert result.decision.purpose == purpose


@pytest.mark.parametrize("change", ["revocation", "expired_probe"])
def test_current_safety_and_expiration_are_rechecked(tmp_path, closure, change):
    from datetime import timedelta

    from tests.grounding.query_support import NOW

    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    if change == "revocation":
        p.f.catalogs.disable(p.plan.document.steps[0].implementation_ref)
    else:
        p.compiler.admission.clock = lambda: NOW + timedelta(days=2)
    with pytest.raises(ValueError):
        p.executor.execute(p.plan, "stale")
    assert not list((p.root / "attempts").glob("*/*/process.json"))


@pytest.mark.parametrize(
    "method,encoded",
    [
        (0, bytes([1, 2, 3, 4, 5, 6])),
        (1, bytes([1, 1, 1, 4, 1, 1])),
        (2, bytes([1, 2, 3, 3, 3, 3])),
        (3, bytes([1, 2, 2, 4, 2, 2])),
        (4, bytes([1, 1, 1, 3, 1, 1])),
    ],
)
def test_png_numeric_filters_decode_known_pixels(method, encoded):
    import struct
    import zlib

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = bytes([method]) + encoded[:3] + bytes([method]) + encoded[3:]
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 0, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    shape, pixels = grayscale_png(data, 10)
    assert shape == (2, 3) and pixels == bytes([1, 2, 3, 4, 5, 6])

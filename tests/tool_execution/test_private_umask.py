"""Private host attempt, usable dropped-uid jail, independent of service umask."""

import os
import stat

import pytest

from packages.tool_execution import ExecutionError
from packages.tool_execution.process import write_copy
from tests.tool_execution.support import prepared, runtime_files
from tests.tool_execution.test_dependencies import TestLedger


def test_original_sandbox_runs_under_private_umask(tmp_path):
    previous = os.umask(0o077)
    try:
        closure = runtime_files(tmp_path)
        p = prepared(
            tmp_path / "deployment", mode="isolation", repository=TestLedger(), files=closure
        )
        result = p.executor.execute(p.plan, "private-mask")
        assert result.status == "SUCCEEDED", result
        step = result.steps[0]
        assert step.exit_code == 0
        base = p.executor._runner.workspace / result.attempt_id / step.step_id
        assert stat.S_IMODE(base.stat().st_mode) == 0o700
        assert stat.S_IMODE((base / "root").stat().st_mode) == 0o755
        assert stat.S_IMODE((base / "root/out").stat().st_mode) == 0o777
    finally:
        os.umask(previous)


def test_copy_parents_are_traversable_but_host_ancestors_unchanged(tmp_path):
    private = tmp_path / "host-private"
    private.mkdir(mode=0o700)
    root = private / "root"
    root.mkdir(mode=0o755)
    previous = os.umask(0o077)
    try:
        target = write_copy(root, "/nested/lib/value", b"exact bytes")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(target.parent.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(target.stat().st_mode) == 0o444


def test_copy_does_not_follow_existing_directory_symlinks_or_chmod_existing_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExecutionError, match="deployment_parent_rejected"):
        write_copy(root, "/escape/leak", b"must not escape")
    assert not (outside / "leak").exists()
    assert stat.S_IMODE(outside.stat().st_mode) == 0o700
    existing = root / "existing"
    existing.mkdir(mode=0o700)
    write_copy(root, "/existing/exact", b"existing owner permissions are not silently changed")
    assert stat.S_IMODE(existing.stat().st_mode) == 0o700

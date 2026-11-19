"""Actual owned child distinguishes running, terminated zombie and reaped states."""

import os
import time

import pytest

from tests.research_task_binding_increment.processes import classify, observe_owned, process_state
from tests.research_task_binding_r01.observer import export


@pytest.fixture(autouse=True)
def diagnostic_directory(monkeypatch, tmp_path):
    if "MMAGENT_BINDING_DIAGNOSTIC_DIR" not in os.environ:
        directory = tmp_path / "binding-diagnostics"
        directory.mkdir()
        monkeypatch.setenv("MMAGENT_BINDING_DIAGNOSTIC_DIR", str(directory))


def test_real_owned_zombie_is_terminated_and_only_exact_child_is_reaped():
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(write_fd)
        os.read(read_fd, 1)
        os._exit(0)
    os.close(read_fd)
    try:
        original = process_state(pid)
        assert original and classify(original, original) == "running"
        os.close(write_fd)
        write_fd = None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = process_state(pid)
            if current and current["state"] == "Z":
                break
            time.sleep(0.01)
        assert current and current["state"] == "Z"
        # The old predicate really reports this non-running process as alive.
        assert current["start_ticks"] == original["start_ticks"]
        assert classify(original, current) == "terminated"
        observation = observe_owned(original, reap=True)
        assert observation["reaped"] and observation["after_reap"] is None
        export("zombie-counterexample", observation)
    finally:
        if write_fd is not None:
            os.close(write_fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


@pytest.mark.parametrize("state", ["R", "S", "D", "T", "I"])
def test_matching_live_identity_is_never_called_clean(state):
    original = dict(pid=123, start_ticks="456")
    assert classify(original, dict(original, state=state)) == "running"


def test_absent_and_reused_pid_do_not_authorize_deleting_or_stopping_replacement():
    original = dict(pid=123, start_ticks="456")
    assert classify(original, None) == "absent"
    assert classify(original, dict(pid=123, start_ticks="789", state="S")) == "pid_reused"

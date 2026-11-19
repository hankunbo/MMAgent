"""Standard subprocess entry and ownership, including unsuccessful sessions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.session_temp import SessionTemp


@pytest.mark.parametrize("fail", [False, True])
def test_standard_entry_without_tmpdir_preserves_caller_directory(tmp_path, fail):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("not session owned")
    observed = tmp_path / "observed"
    env = {k: v for k, v in os.environ.items() if k not in {"TMPDIR", "TMP", "TEMP"}}
    env["MMAGENT_TEMP_PROBE"] = str(observed)
    env["MMAGENT_TEMP_PROBE_FAIL"] = str(int(fail))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            __file__ + "::test_standard_child",
            "--basetemp",
            str(existing),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == int(fail), result.stdout + result.stderr
    assert sentinel.read_text() == "not session owned"
    path = Path(observed.read_text())
    session = next(p for p in path.parents if p.name.startswith("pytest-session-"))
    assert not session.exists()


def test_standard_child(tmp_path):
    assert tmp_path.is_relative_to(Path("/opt/data/test/MMAgent/.local"))
    if output := os.environ.get("MMAGENT_TEMP_PROBE"):
        Path(output).write_text(str(tmp_path))
        assert os.environ["MMAGENT_TEMP_PROBE_FAIL"] == "0", "intentional failure"


def test_interrupted_root_is_not_reused_and_ownership_change_refuses_cleanup(tmp_path):
    old = SessionTemp(tmp_path)
    new = SessionTemp(tmp_path)
    assert old.root != new.root
    new.cleanup()
    assert old.root.exists()
    original = old.marker.read_text()
    old.marker.write_text("{}")
    with pytest.raises(RuntimeError, match="ownership"):
        old.cleanup()
    assert old.root.exists()
    old.marker.write_text(original)
    old.cleanup()

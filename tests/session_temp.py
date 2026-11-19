"""One owned temporary tree per standard pytest session; never reuse a basetemp."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1] / ".local"


class SessionTemp:
    def __init__(self, parent=ROOT):
        if parent.is_symlink():
            raise pytest.UsageError("pytest temporary parent must not be a symlink")
        parent.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="pytest-session-", dir=parent))
        self.identity = self.root.stat().st_ino
        self.owner = {"pid": os.getpid(), "nonce": uuid4().hex}
        self.marker = self.root / "owner.json"
        self.marker.write_text(json.dumps(self.owner))
        # pytest may clear basetemp: this child has never existed, unlike root.
        self.base = self.root / "tmp"

    def cleanup(self):
        if not self.root.exists() and not self.root.is_symlink():
            return
        if (
            self.root.is_symlink()
            or self.root.stat().st_ino != self.identity
            or self.marker.is_symlink()
            or json.loads(self.marker.read_text()) != self.owner
        ):
            raise RuntimeError("pytest temporary ownership changed; retaining directory")
        shutil.rmtree(self.root)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Both python -m pytest and make test-main load this through tests/conftest.
    # Caller basetemp (possibly pre-existing) is never cleared or used.
    lease = SessionTemp()
    previous = config.option.basetemp
    config.option.basetemp = str(lease.base)

    def cleanup():
        try:
            lease.cleanup()
        finally:
            config.option.basetemp = previous

    config.add_cleanup(cleanup)

"""Versioned case/resource exclusion, without models, credentials or patient input."""

import errno
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from apps import segman_resource_lock as resource
from packages.evaluation_preparation import generic_execution_lock as locks


def bound_case(directory):
    directory.mkdir(exist_ok=True)
    spec = dict(
        schema_version=resource.INPUT_VERSION,
        key=directory.name,
        execution_lock_profile=resource.PROFILE,
        execution_lock_implementation=resource.implementation(locks.ROOT),
    )
    (directory / "input.json").write_text(json.dumps(spec))
    return spec


class Repository:
    def __init__(self, target):
        self.calls = []

    @contextmanager
    def guarded_session(self, case, key):
        self.calls.append((case, key))
        yield lambda: None


def setup_owner(monkeypatch):
    monkeypatch.setattr(locks, "target", lambda: "isolated-test")
    monkeypatch.setattr(locks, "CasePipelineRepository", Repository)


def test_nested_case_and_resource_context_reset(tmp_path, monkeypatch):
    setup_owner(monkeypatch)
    spec = bound_case(tmp_path / "case-a")
    with locks.run_lock(spec, tmp_path / "case-a"):
        active = locks.current("case-a")
        with locks.scoped_lock(active.root / "lock"):
            assert locks.current() is active
        with locks.scoped_lock(active.root / "lock", resource="segman-component"):
            assert len(locks.current().guards) == 2
        assert locks.current() is active
        assert [c[0] for c in active.repository.calls] == [
            "CASE:GENERIC_EXECUTION",
            "RESOURCE:SEGMAN_COMPONENT",
        ]
        with pytest.raises(ValueError, match="nested_run"):
            with locks.run_lock(spec, active.root):
                pytest.fail("nested root permitted")
        with pytest.raises(ValueError, match="path_scope"):
            with locks.scoped_lock(tmp_path / "outside"):
                pytest.fail("cross-case scope permitted")
        with pytest.raises(ValueError, match="resource_scope"):
            with locks.scoped_lock(active.root / "lock", resource="unknown"):
                pytest.fail("arbitrary resource permitted")
    assert locks.current() is None


@pytest.mark.parametrize("mutation", ["input", "scope", "implementation"])
def test_current_binding_fails_closed(tmp_path, monkeypatch, mutation):
    setup_owner(monkeypatch)
    spec = bound_case(tmp_path)
    with pytest.raises(ValueError, match="generic_execution_lock"):
        with locks.run_lock(spec, tmp_path):
            if mutation == "input":
                (tmp_path / "input.json").write_text("{}")
            elif mutation == "implementation":
                monkeypatch.setattr(locks, "implementation", lambda root: {})
            locks.current("different" if mutation == "scope" else None)
    assert locks.current() is None


def test_legacy_and_new_profile_are_not_silently_interchanged(tmp_path):
    assert not locks.validate_profile({"schema_version": "fairvlmed-run-input/1.8.0"})
    with pytest.raises(ValueError, match="legacy_execution_lock_extension"):
        locks.validate_profile(
            dict(
                schema_version="fairvlmed-run-input/1.8.0", execution_lock_profile=resource.PROFILE
            )
        )
    spec = bound_case(tmp_path)
    spec["execution_lock_profile"] = "unbound"
    with pytest.raises(ValueError, match="configuration_changed"):
        locks.validate_profile(spec)


def test_current_campaign_requires_original_case_lease():
    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    campaign = object.__new__(GenericCampaign)
    campaign.spec = dict(schema_version=resource.INPUT_VERSION, key="test")
    with pytest.raises(ValueError, match="original_execution_lease_required"):
        campaign.execution_session()


def test_input_change_stops_owned_child_not_replays(tmp_path, monkeypatch):
    from packages.evaluation_preparation.terminal_cohort import stop_child

    setup_owner(monkeypatch)
    spec = bound_case(tmp_path)
    with locks.run_lock(spec, tmp_path):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        try:
            (tmp_path / "input.json").write_text("{}")
            with pytest.raises(ValueError, match="input_changed"):
                locks.wait_process(child, timeout=5, stop=stop_child)
            assert child.poll() is not None
        finally:
            if child.poll() is None:
                stop_child(child)


def test_stdlib_binding_checks_source_and_input(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    for filename in resource.LOCK_FILES:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source")
    case = tmp_path / "case-a"
    spec = bound_case(case)
    spec["execution_lock_implementation"] = resource.implementation(root)
    (case / "input.json").write_text(json.dumps(spec))
    binding = dict(
        schema_version="generic-case-binding/1.2.0",
        execution_namespace=case.name,
        input_spec_checksum=resource.digest(spec),
    )
    assert resource.verify_input(root, case, binding) == spec
    (root / resource.LOCK_FILES[0]).write_text("changed")
    with pytest.raises(ValueError, match="execution_lock_binding"):
        resource.verify_input(root, case, binding)


@pytest.fixture
def host_filesystem():
    # Production host exclusion lives in /tmp, not the NFS repository temp tree.
    with tempfile.TemporaryDirectory(prefix="mmagent-host-lock-test-", dir="/tmp") as directory:
        yield Path(directory)


def test_host_lock_and_legacy_bridge_exclude_other_runner(host_filesystem):
    tmp_path = host_filesystem
    (tmp_path / ".local").mkdir()
    private = tmp_path / "host"
    with resource.resource_lock(tmp_path, private_root=private) as held:
        assert held["host_lock"] and held["legacy_lock"]
        with pytest.raises(BlockingIOError):
            with resource.resource_lock(tmp_path, private_root=private):
                pytest.fail("concurrent host runner")
    with resource.resource_lock(tmp_path, private_root=private):
        pass


@pytest.mark.parametrize("error", [errno.ENOLCK, errno.EACCES])
def test_only_unsupported_legacy_lock_uses_explicit_host_profile(
    host_filesystem, monkeypatch, error
):
    tmp_path = host_filesystem
    (tmp_path / ".local").mkdir()
    original = resource.fcntl.flock
    calls = 0

    def flock(handle, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(error, "synthetic legacy filesystem")
        return original(handle, mode)

    monkeypatch.setattr(resource.fcntl, "flock", flock)
    if error == errno.ENOLCK:
        with resource.resource_lock(tmp_path, private_root=tmp_path / "host") as held:
            assert held == dict(host_lock=True, legacy_lock=False, profile=resource.PROFILE)
    else:
        with pytest.raises(OSError) as caught:
            with resource.resource_lock(tmp_path, private_root=tmp_path / "host"):
                pytest.fail("unrelated failure bypassed")
        assert caught.value.errno == error


def test_host_lock_directory_symlink_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(actual)
    with pytest.raises(ValueError, match="resource_lock_directory"):
        with resource.resource_lock(tmp_path, private_root=link):
            pytest.fail("symlink followed")


def test_terminal_profile_changes_only_explicit_fair_child(tmp_path, monkeypatch):
    from packages.evaluation_preparation import terminal_cohort as batch
    from tests.evaluation_preparation.test_terminal_runtime_lease import manifest

    old = manifest()
    raw = old.model_dump(mode="json")
    raw.update(
        schema_version="terminal-case-cohort/1.9.0", fairvlmed_execution_lock=resource.PROFILE
    )
    new = batch.read_manifest(raw)
    for case in new.cases:
        previous = batch.commands(old, case, tmp_path, "prepare")[0]
        current = batch.commands(new, case, tmp_path, "prepare")[0]
        if case.dataset == "fairvlmed":
            assert current == previous + ["--execution-lock-profile", resource.PROFILE]
        else:
            assert current == previous
    monkeypatch.setattr(
        batch, "child_lock_capability", lambda output: dict(available=False, errno=errno.ENOLCK)
    )
    monkeypatch.setattr(locks, "capability", lambda: dict(available=True, profile=resource.PROFILE))
    all_cases = batch.execution_lock_capability(tmp_path, new)
    assert not all_cases["available"]
    fair = [case for case in new.cases if case.dataset == "fairvlmed"]
    assert batch.execution_lock_capability(tmp_path, new, fair)["available"]
    assert not batch.execution_lock_capability(tmp_path, old, fair)["available"]


def test_new_terminal_freeze_checks_original_runtime_owner(tmp_path, monkeypatch):
    from packages.evaluation_preparation import terminal_cohort as batch
    from packages.grounding.models import digest
    from tests.evaluation_preparation.test_terminal_runtime_lease import manifest

    raw = manifest().model_dump(mode="json")
    raw.update(
        schema_version="terminal-case-cohort/1.9.0", fairvlmed_execution_lock=resource.PROFILE
    )
    selected = batch.read_manifest(raw)
    (tmp_path / "manifest.json").write_text(json.dumps(raw))
    (tmp_path / "freeze.json").write_text(
        json.dumps(
            dict(
                schema_version="terminal-cohort-freeze/1.1.0",
                manifest_checksum=digest(selected),
                output=str(tmp_path),
                runtime_owner={"owner": "old"},
                implementation={},
            )
        )
    )
    monkeypatch.setattr(batch, "runtime_owner_implementation", lambda: {"owner": "new"})
    with pytest.raises(ValueError, match="cohort_runtime_owner_changed"):
        batch.load(tmp_path, check_current=True)
    assert batch.load(tmp_path, check_current=False)[1] == selected

import json
import sys

import pytest

from packages.evaluation_preparation.case_execution import process, selected_input


def test_process_once_exact_recovery(tmp_path):
    command = [sys.executable, "-c", "print('synthetic no model')"]
    first = process(command, directory=tmp_path / "job", seconds=10)

    def forbidden(*args, **kwargs):
        raise AssertionError("duplicate launch")

    second = process(command, directory=tmp_path / "job", seconds=10, runner=forbidden)
    assert first == second and first["exit_code"] == 0


def test_failed_not_retried(tmp_path):
    command = [sys.executable, "-c", "raise SystemExit(2)"]
    with pytest.raises(ValueError, match="process_failed"):
        process(command, directory=tmp_path, seconds=10)
    with pytest.raises(ValueError, match="prior_failure"):
        process(command, directory=tmp_path, seconds=10)


def test_unknown_claim_no_launch(tmp_path):
    (tmp_path / "process-claim.json").write_text("{}")
    with pytest.raises(FileExistsError):
        process([sys.executable, "-c", "print('should not launch')"], directory=tmp_path)
    assert not (tmp_path / "process.log").exists()


def test_different_command_rejected(tmp_path):
    command = [sys.executable, "-c", "print('ok')"]
    process(command, directory=tmp_path, seconds=10)
    with pytest.raises(ValueError, match="command_conflict"):
        process([sys.executable, "-c", "print('changed')"], directory=tmp_path)


def test_allowlist_before_data_access():
    with pytest.raises(ValueError, match="outside_two_image"):
        selected_input("data_06231.npz")


def test_invalid_protocol_before_original_owner(tmp_path):
    (tmp_path / "protocol.json").write_text(
        json.dumps(dict(input_set_checksum="wrong", input_files={}))
    )
    with pytest.raises(ValueError, match="protocol_changed"):
        selected_input("data_00002.npz", input_root=tmp_path)


def test_directory_adapter_creates_original_audit_target(tmp_path):
    from apps.segman_case_runner import initialize

    job = tmp_path / "new-job"
    initialize(job)
    assert (job / "artifacts").is_dir()
    assert (job / "tmp").is_dir()
    import pytest

    with pytest.raises(FileExistsError):
        initialize(job)

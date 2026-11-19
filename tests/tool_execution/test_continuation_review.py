"""Versioned real-runtime closure checks; harmless fixture, no CUDA claims."""

from pathlib import Path

import pytest

from packages.grounding.models import validate
from packages.tool_execution.trusted_models import TrustedToolReview
from tests.tool_execution.test_trusted_local import pinned
from tests.tool_execution.test_trusted_local import setup as setup


def test_empty_init_requires_new_review_version_and_is_inventory_checked(setup):
    f = setup
    path = Path(f.review.environment_root) / "empty_init.py"
    path.write_bytes(b"")
    document = f.review.model_dump(mode="json")
    document["files"].append(pinned(path).model_dump(mode="json"))
    with pytest.raises(ValueError, match="empty_closure_file_requires_review_1_1"):
        validate(TrustedToolReview, document)
    document["version"] = "trusted-tool-review/1.1.0"
    review = validate(TrustedToolReview, document)
    f.runner.verify_review(review)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trusted_resource_size_drift"):
        f.runner.verify_review(review)


def test_cpu_bounded_logs_are_retained_and_match_result(tmp_path):
    from packages.object_store import checksum
    from tests.tool_execution.support import prepared
    from tests.tool_execution.test_dependencies import TestLedger

    f = prepared(tmp_path, mode="logs", repository=TestLedger())
    result = f.executor.execute(f.plan, "retained-cpu-log")
    step = result.steps[0]
    assert step.error_code == "log_size_limit"
    base = f.executor._runner.workspace / result.attempt_id / step.step_id
    assert 0 < (base / "stdout.log").stat().st_size <= f.policy.log_bytes
    assert checksum((base / "stdout.log").read_bytes()) == step.stdout_checksum
    assert checksum((base / "stderr.log").read_bytes()) == step.stderr_checksum
    assert f.executor.execute(f.plan, "retained-cpu-log") == result

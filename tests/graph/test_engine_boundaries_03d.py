"""Fresh-process dependency isolation and exhaustive contract state ownership."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from packages.contracts import GraphNodeStatus, RunStatus, can_transition


@pytest.mark.parametrize("module", ["packages.graph_engine", "packages.rule_engine"])
@pytest.mark.parametrize("hash_seed", ["1", "23", "random"])
def test_public_import_in_fresh_process_loads_only_deterministic_dependencies(module, hash_seed):
    code = """
import importlib
import importlib.abc
import json
import sys
forbidden = {'apps', 'agents', 'db', 'openai', 'anthropic', 'httpx', 'requests',
             'psycopg', 'torch', 'numpy', 'transformers', 'Algorithm_Tool', 'Seg_Model',
             'packages.retrieval', 'packages.workflow_compiler'}
def blocked(name):
    return any(name == root or name.startswith(root + '.') for root in forbidden)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if blocked(fullname):
            raise AssertionError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
public = importlib.import_module(sys.argv[1])
assert all(hasattr(public, name) for name in public.__all__)
assert not [name for name in sys.modules if blocked(name)]
print(json.dumps(sorted(public.__all__)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, module],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ | {"PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)


@pytest.mark.parametrize("current", list(GraphNodeStatus))
@pytest.mark.parametrize("target", list(GraphNodeStatus))
def test_frozen_graph_state_machine_truth_table(current, target):
    expected = {
        "UNKNOWN": {"PENDING", "INSUFFICIENT_EVIDENCE"},
        "PENDING": {
            "UNKNOWN",
            "SUPPORTED",
            "REFUTED",
            "CONFLICTED",
            "EXCLUDED",
            "INSUFFICIENT_EVIDENCE",
        },
        "SUPPORTED": {"REFUTED", "CONFLICTED", "PENDING"},
        "REFUTED": {"SUPPORTED", "CONFLICTED", "PENDING"},
        "CONFLICTED": {"SUPPORTED", "REFUTED", "PENDING"},
        "EXCLUDED": {"PENDING"},
        "INSUFFICIENT_EVIDENCE": {"PENDING"},
    }
    assert can_transition(current, target, allow_idempotent=False) == (
        target.value in expected[current.value]
    )
    assert can_transition(current, target) == (
        current is target or target.value in expected[current.value]
    )


def test_online_run_has_no_human_wait_state():
    assert all("HUMAN" not in state.value for state in RunStatus)

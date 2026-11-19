"""Explicit new configuration and local-only preparation; no runtime or credentials."""

import copy
from pathlib import Path

import pytest

from apps.vectorengine_batch_prepare import preview
from apps.vectorengine_round import LiveRoundV3, load_round, prepare_round
from packages.grounding.models import digest, validate
from packages.llm_provider.config import read_provider_config
from tests.agent_provider.batch_support import fixture


def test_prepare_local_has_no_key_network_or_database(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("offline_preparation_must_not_open_connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("apps.vectorengine_single_case.open_case", forbidden)
    monkeypatch.setattr("apps.vectorengine_round.open_case", forbidden)
    a, b = preview("public"), preview("case")
    assert a["key_loaded"] is b["key_loaded"] is False
    assert a["real_qualification"] is False and b["database_connected"] is False
    assert a["round"]["run_id"] != b["round"]["run_id"]
    assert preview("public") == a
    assert a["maximum_normal_planner_steps"] == 14


@pytest.mark.parametrize("field", ["run_id", "request_key", "output", "gate", "provider_config"])
def test_round_identity_directory_gate_conflict_rejected(field):
    raw = copy.deepcopy(preview("public")["round"])
    raw[field] += "other"
    with pytest.raises(ValueError):
        validate(LiveRoundV3, raw)


def test_case_preflight_reads_provided_owner_and_only_projects_batch_payloads(tmp_path):
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    f = fixture(7)
    f.owner.state = ready_entry_state(f.owner.state)
    r = load_round("configs/research/vectorengine_batch_review_case.json")
    # Controlled test output path only; production load_round continues enforcing fixed paths.
    r = r.model_copy(update={"output": str(tmp_path)})

    class Evidence:
        def project(self, **kwargs):
            return f.owner.current(), dict(approved_synthetic=True)

    state, _ = prepare_round(r, f.config, Evidence())
    import json

    p = json.loads((tmp_path / "review-input.json").read_text())
    assert p["not_executable"] and p["actual_parent_frozen_after_source_vision"]
    assert len(p["batch_input_checksums"]) == 3 and state == f.owner.current()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    prepare_round(r, f.config, Evidence())
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert f.budget.used == 0


def test_legacy_provider_and_round_exact_versions_remain_readable():
    import json

    old = Path("configs/agents/provider.json")
    from packages.llm_provider.config import load_provider_config

    config = load_provider_config(old, environ={})
    assert config.schema_version == "agent-provider-config/1.0.0"
    assert getattr(config, "review_batching", None) is None
    assert read_provider_config(config) == config
    raw = json.loads(Path("configs/research/vectorengine_round2.json").read_text())
    assert digest(load_round("configs/research/vectorengine_round2.json")) == digest(raw)

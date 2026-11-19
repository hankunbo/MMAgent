"""Batch reads preserve original parent lifecycle and exact byte checks."""

from types import SimpleNamespace

import pytest

from apps.vectorengine_round_materials import RoundCaseEvidence
from db.knowledge.models import seal
from db.knowledge.repository.postgres import KnowledgeRepository
from packages.grounding.models import digest
from packages.object_store import checksum
from tests.database.knowledge_support_04a import candidate_records


def fixture():
    records = candidate_records("SOURCE_BATCH")
    raw = b"This is a synthetic literal observation for the batch reader."
    old = records["chunk"]
    data = old.model_dump(mode="python", exclude={"checksum"})
    data.update(content_checksum=checksum(raw), size_bytes=len(raw))
    records["chunk"] = seal(type(old), **data)
    values = {
        digest(r.reference()): SimpleNamespace(
            record=r, withdrawn=False, status=SimpleNamespace(value="draft")
        )
        for r in records.values()
    }
    calls = []

    class Reader:
        def get(self, ref):
            return values[digest(ref)]

        def get_many(self, refs):
            calls.append(refs)
            return tuple(self.get(r) for r in refs)

    owner = object.__new__(RoundCaseEvidence)
    owner.original = SimpleNamespace(
        knowledge=Reader(), knowledge_objects=SimpleNamespace(read_bounded=lambda *a: raw)
    )
    owner.chunks = [
        dict(
            reference=records["chunk"].reference().model_dump(mode="json"),
            selector=dict(kind="text", start=0, end=len(raw)),
            locator="synthetic:1",
            text=raw.decode(),
            text_checksum=checksum(raw),
        )
    ]
    return owner, records, values, calls


def test_batch_and_sequential_projection_equal_and_no_snapshot_cache():
    owner, records, values, calls = fixture()
    assert owner.read_sources() == tuple(owner.read_source(c) for c in owner.chunks)
    assert len(calls) == 3
    values[digest(records["source"].reference())].withdrawn = True
    with pytest.raises(ValueError, match="source_withdrawn"):
        owner.read_sources()
    assert len(calls) == 6


def test_changed_object_bytes_still_rejected_after_batch_metadata():
    owner, _, _, _ = fixture()
    owner.original.knowledge_objects.read_bounded = lambda *a: b"changed"
    with pytest.raises(ValueError, match="source_object_changed"):
        owner.read_sources()


def test_batch_bounds_before_any_connection():
    repo = object.__new__(KnowledgeRepository)
    ref = candidate_records("SOURCE_BOUNDS")["source"].reference()
    for refs, error in (((), "bound"), ((ref,) * 65, "bound"), ((ref, ref), "duplicate")):
        with pytest.raises(ValueError, match=error):
            repo.get_many(refs)


def test_case_only_entry_keeps_provider_and_charges_six_prior_runs():
    from apps.vectorengine_batch_prepare import preview

    value = preview(
        "case", "configs/research/vectorengine_batch_integrated_r01_case_runtime_r01.json"
    )
    r = value["round"]
    assert r["schema_version"] == "bounded-live-round/8.0.0"
    assert len(r["prior_round_configs"]) == 6
    assert r["provider_config"] == "configs/agents/vectorengine_batch_integrated_r01.json"
    assert r["seconds"] == 2700 and r["max_requests"] == 40
    assert not value["key_loaded"] and not value["database_connected"]


def test_historical_gate_rejects_new_delta_but_actual_old_role_wire_is_unchanged(monkeypatch):
    import apps.vectorengine_performance_gate as gate
    from apps.vectorengine_round import load_round
    from apps.vectorengine_single_case import load_config
    from packages.llm_provider.public_batches import source_vision_reuse

    monkeypatch.setattr(gate, "check_acceptance", lambda: None)
    public = load_round("configs/research/vectorengine_batch_integrated_r01_public.json")
    # This old gate deliberately cannot authorize the later acquisition delta.
    # Its historical success remains unchanged; new execution needs its own gate.
    with pytest.raises(ValueError, match="performance_public_wire_changed"):
        gate.check_public_baseline(public)
    assert len(source_vision_reuse(load_config(public.provider_config))) == 2
    original = gate.checksum
    monkeypatch.setattr(
        gate,
        "checksum",
        lambda data: (
            "sha256:" + "0" * 64
            if data == __import__("pathlib").Path(public.provider_config).read_bytes()
            else original(data)
        ),
    )
    with pytest.raises(ValueError, match="performance_public"):
        gate.check_public_baseline(public)


def test_actual_case_freeze_uses_incremental_public_owner(monkeypatch):
    from pathlib import Path

    import apps.vectorengine_batch_prepare as prepare
    import apps.vectorengine_performance_gate as gate
    from apps.vectorengine_round import load_round

    config = "configs/research/vectorengine_batch_integrated_r01_case_runtime_r01.json"
    r = load_round(config)
    calls = []
    writes = []
    monkeypatch.setattr(prepare, "check_acceptance", lambda _: ({}, {}))
    monkeypatch.setattr(gate, "check_public_baseline", lambda public: calls.append(public.run_id))

    def check_current(value):
        assert value.run_id == r.run_id, "old public global gate must not be called"
        assert len(writes) == 1

    monkeypatch.setattr(prepare, "check_round_gate", check_current)
    monkeypatch.setattr(prepare, "write_once", lambda p, value: writes.append(value))
    result = prepare.freeze_gate("case", config)
    assert result["status"] == "batch_gate_verified_no_requests"
    assert calls == [load_round(r.public_config).run_id]
    assert (
        str(Path(load_round(r.public_config).output) / "qualification.json")
        in writes[0]["evidence"]
    )
    assert writes[0]["initial_state_checksum"]

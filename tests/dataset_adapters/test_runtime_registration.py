"""Selected-input persistence through the original Runtime/objects, not mock SQL."""

import json

import psycopg
import pytest

from packages.dataset_adapters.models import read_registration
from packages.dataset_adapters.reading import AdapterError
from packages.dataset_adapters.service import RestrictedEvaluatorReader
from tests.database.test_runtime_repository_02b import database as database
from tests.dataset_adapters.support import csv_write, fair
from tests.dataset_adapters.test_registration import service
from tests.dataset_adapters.test_runtime_input import replace_npz, runtime_adapter


def test_optional_input_registration_readback_and_conflict(database, tmp_path):
    old, _ = fair(tmp_path / "source")
    replace_npz(old.root / "Dataset/Training/data_00000.npz", missing=("age", "glaucoma"))
    csv_write(
        old.root / "ReadMe/data_summary.csv",
        ("filename", "use"),
        [dict(filename="data_00000.npz", use="training")],
    )
    adapter = runtime_adapter(old)
    sample = adapter.prepare("data_00000.npz")
    svc = service(database, tmp_path)
    value = svc.register(sample, key="runtime-optional")
    assert value.schema_version == "dataset-registration/3.0.0"
    assert value.visible.text_review == "not_applicable"
    assert value.visible.input_profile.missing_optional_fields == ("age", "glaucoma")
    assert read_registration(value.model_dump_json()) == value
    assert svc.read_visible(value.visible.case.case_code) == value.visible
    assert svc.register(sample, key="runtime-optional") == value
    hidden = RestrictedEvaluatorReader(svc.evaluator_store).manifest(value.restricted_manifest)
    assert hidden.label_scope == "source_separated" and hidden.original_note is None
    with database[3]._connect() as conn:
        before = conn.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"]
    replace_npz(old.root / "Dataset/Training/data_00000.npz", missing=("race",))
    with pytest.raises(AdapterError, match="dataset_input_conflict"):
        svc.register(adapter.prepare("data_00000.npz"), key="runtime-optional")
    with database[3]._connect() as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"] == before
        )


def test_runtime_ack_recovery_and_legacy_exact_read(database, tmp_path):
    old, _ = fair(tmp_path / "source")
    svc = service(database, tmp_path)
    legacy = svc.register(old.prepare("data_00000.npz"), key="runtime-legacy")
    wire = legacy.model_dump_json()
    assert (
        svc.register(old.prepare("data_00000.npz"), key="runtime-legacy").model_dump_json() == wire
    )
    sample = runtime_adapter(old).prepare("data_00000.npz")
    original = svc.repository.register

    def lose(*args):
        original(*args)
        raise psycopg.OperationalError("synthetic ack loss")

    svc.repository.register = lose
    with pytest.raises(AdapterError, match="commit_unknown"):
        svc.register(sample, key="runtime-ack")
    before = set(svc.visible_store.root.rglob("*"))
    svc.repository.register = original
    current = svc.register(sample, key="runtime-ack")
    assert set(svc.visible_store.root.rglob("*")) == before
    assert svc.read_visible(current.visible.case.case_code) == current.visible
    with database[3]._connect() as conn:
        text = json.dumps(
            conn.execute("SELECT response_payload FROM runtime.idempotency_request").fetchall()
        )
    assert "SECRET_" not in text


def test_runtime_registration_rollback(database, tmp_path, monkeypatch):
    old, _ = fair(tmp_path / "source")
    svc = service(database, tmp_path)
    original = database[3]._create_case_bundle
    with database[3]._connect() as conn:
        before = conn.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"]

    def reject(*args):
        original(*args)
        raise psycopg.errors.CheckViolation("synthetic rollback")

    monkeypatch.setattr(database[3], "_create_case_bundle", reject)
    with pytest.raises(AdapterError, match="registration_failed"):
        svc.register(runtime_adapter(old).prepare("data_00000.npz"), key="runtime-rollback")
    with database[3]._connect() as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"] == before
        )
    assert not list(svc.visible_store.root.glob("objects/*"))
